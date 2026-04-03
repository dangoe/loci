use std::collections::HashMap;

use chrono::{DateTime, Utc};
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, DeletePointsBuilder, Distance, GetCollectionInfoResponse,
    GetPointsBuilder, PointId, PointStruct, PointsIdsList, SearchPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder, point_id::PointIdOptions,
    vectors_config::Config as VectorsConfigVariant,
};
use uuid::Uuid;

use ai_memory_core::{
    Embedding, Memory, MemoryEntry, MemoryInput, MemoryQuery, MemoryStore, MemoryStoreError, Score,
    TextEmbedder,
};

use crate::config::QdrantConfig;

// Payload field names
const FIELD_CONTENT: &str = "content";
const FIELD_METADATA: &str = "metadata";
const FIELD_CREATED_AT: &str = "created_at";

/// [`MemoryStore`] implementation backed by Qdrant using cosine vector similarity.
///
/// Embeddings are computed internally via the provided [`TextEmbedder`]; callers
/// work with plain text and are not exposed to embedding vectors.
pub struct QdrantMemoryStore<E> {
    client: Qdrant,
    config: QdrantConfig,
    embedder: E,
}

impl<E: TextEmbedder> QdrantMemoryStore<E> {
    /// Creates a new store connecting to the given Qdrant URL.
    ///
    /// Call [`initialize`][Self::initialize] once before using the store.
    pub fn new(url: &str, config: QdrantConfig, embedder: E) -> Result<Self, MemoryStoreError> {
        let client = Qdrant::from_url(url)
            .build()
            .map_err(|e| MemoryStoreError::Connection(e.to_string()))?;
        Ok(Self {
            client,
            config,
            embedder,
        })
    }

    /// Creates the Qdrant collection if it does not already exist. Must be called
    /// once before using the store.
    ///
    /// If the collection already exists, validates that its vector dimension matches
    /// the dimension reported by the embedder. Returns [`MemoryStoreError::Connection`]
    /// on a mismatch so callers receive an actionable error before any data operations.
    pub async fn initialize(&self) -> Result<(), MemoryStoreError> {
        let exists = self
            .client
            .collection_exists(&self.config.collection_name)
            .await
            .map_err(|e| MemoryStoreError::Connection(e.to_string()))?;

        if exists {
            let info = self
                .client
                .collection_info(&self.config.collection_name)
                .await
                .map_err(|e| MemoryStoreError::Connection(e.to_string()))?;

            let expected_dim = self.embedder.embedding_dimension() as u64;
            if let Some(dim) = extract_vector_dimension(&info)
                && dim != expected_dim
            {
                return Err(MemoryStoreError::Connection(format!(
                    "collection '{}' has vector dimension {dim} \
                     but embedder produces {expected_dim}",
                    self.config.collection_name
                )));
            }
        } else {
            let dim = self.embedder.embedding_dimension() as u64;
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.config.collection_name)
                        .vectors_config(VectorParamsBuilder::new(dim, Distance::Cosine)),
                )
                .await
                .map_err(|e| MemoryStoreError::Connection(e.to_string()))?;
        }

        Ok(())
    }

    /// Inserts a point into the collection and returns the resulting entry.
    async fn do_save(
        &self,
        memory: Memory,
        embedding: Embedding,
    ) -> Result<MemoryEntry, MemoryStoreError> {
        let metadata_json = serde_json::to_string(&memory.metadata)
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let mut payload = Payload::new();
        payload.insert(FIELD_CONTENT, memory.content.clone());
        payload.insert(FIELD_METADATA, metadata_json);
        payload.insert(FIELD_CREATED_AT, memory.created_at.to_rfc3339());

        let point = PointStruct::new(
            PointId::from(memory.id.to_string()),
            embedding.values().to_vec(),
            payload,
        );

        self.client
            .upsert_points(
                UpsertPointsBuilder::new(&self.config.collection_name, vec![point]).wait(true),
            )
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        Ok(MemoryEntry {
            memory,
            score: Score::new(1.0).expect("1.0 is always a valid score"),
        })
    }

    /// Searches by embedding and post-filters by score, metadata, and payload.
    async fn search_by_embedding(
        &self,
        embedding: &Embedding,
        max_results: usize,
        min_score: f64,
        filters: &HashMap<String, String>,
    ) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
        let vector: Vec<f32> = embedding.values().to_vec();

        let response = self
            .client
            .search_points(
                SearchPointsBuilder::new(&self.config.collection_name, vector, max_results as u64)
                    .with_payload(true)
                    .score_threshold(min_score as f32),
            )
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let mut entries = Vec::new();
        for point in response.result {
            let entry = Self::parse_scored_point(&point)?;
            // Post-filter by metadata key/value pairs
            if !filters.iter().all(|(k, v)| {
                entry
                    .memory
                    .metadata
                    .get(k)
                    .map(|s| s == v)
                    .unwrap_or(false)
            }) {
                continue;
            }
            entries.push(entry);
        }
        Ok(entries)
    }

    fn parse_scored_point(
        point: &qdrant_client::qdrant::ScoredPoint,
    ) -> Result<MemoryEntry, MemoryStoreError> {
        let id = extract_uuid_from_point_id(point.id.as_ref())?;
        let score_val = (point.score as f64).clamp(0.0, 1.0);
        let score = Score::new(score_val).map_err(|e| MemoryStoreError::Query(e.to_string()))?;
        let memory = parse_payload_to_memory(id, &point.payload)?;
        Ok(MemoryEntry { memory, score })
    }
}

impl<E: TextEmbedder> MemoryStore for QdrantMemoryStore<E> {
    async fn save(&self, input: MemoryInput) -> Result<MemoryEntry, MemoryStoreError> {
        let memory = Memory::new(input.content, input.metadata);
        let embedding = self
            .embedder
            .embed(&memory.content)
            .await
            .map_err(MemoryStoreError::Embedding)?;

        if let Some(threshold) = self.config.similarity_threshold {
            let candidates = self
                .search_by_embedding(&embedding, 1, threshold, &memory.metadata)
                .await?;
            if let Some(entry) = candidates.into_iter().next() {
                log::debug!(
                    "deduplication: reusing memory {} (score {:.4})",
                    entry.memory.id,
                    entry.score.value()
                );
                return Ok(entry);
            }
        }

        self.do_save(memory, embedding).await
    }

    async fn query(&self, query: MemoryQuery) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
        let embedding = self
            .embedder
            .embed(&query.topic)
            .await
            .map_err(MemoryStoreError::Embedding)?;
        self.search_by_embedding(
            &embedding,
            query.max_results,
            query.min_score.value(),
            &query.filters,
        )
        .await
    }

    async fn update(&self, id: Uuid, input: MemoryInput) -> Result<MemoryEntry, MemoryStoreError> {
        // Retrieve the existing point to verify it exists and read created_at
        let response = self
            .client
            .get_points(
                GetPointsBuilder::new(
                    &self.config.collection_name,
                    vec![PointId::from(id.to_string())],
                )
                .with_payload(true),
            )
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let existing = response
            .result
            .into_iter()
            .next()
            .ok_or(MemoryStoreError::NotFound(id))?;

        // Preserve the original created_at timestamp
        let created_at = extract_created_at_from_payload(&existing.payload)?;

        let embedding = self
            .embedder
            .embed(&input.content)
            .await
            .map_err(MemoryStoreError::Embedding)?;

        let metadata_json = serde_json::to_string(&input.metadata)
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let mut payload = Payload::new();
        payload.insert(FIELD_CONTENT, input.content.clone());
        payload.insert(FIELD_METADATA, metadata_json);
        payload.insert(FIELD_CREATED_AT, created_at.to_rfc3339());

        let point = PointStruct::new(
            PointId::from(id.to_string()),
            embedding.values().to_vec(),
            payload,
        );

        self.client
            .upsert_points(
                UpsertPointsBuilder::new(&self.config.collection_name, vec![point]).wait(true),
            )
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        Ok(MemoryEntry {
            memory: Memory {
                id,
                content: input.content,
                metadata: input.metadata,
                created_at,
            },
            score: Score::new(1.0).expect("1.0 is always a valid score"),
        })
    }

    async fn delete(&self, id: Uuid) -> Result<(), MemoryStoreError> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.config.collection_name)
                    .points(PointsIdsList {
                        ids: vec![PointId::from(id.to_string())],
                    })
                    .wait(true),
            )
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;
        Ok(())
    }

    async fn clear(&self) -> Result<(), MemoryStoreError> {
        self.client
            .delete_collection(&self.config.collection_name)
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let dim = self.embedder.embedding_dimension() as u64;
        self.client
            .create_collection(
                CreateCollectionBuilder::new(&self.config.collection_name)
                    .vectors_config(VectorParamsBuilder::new(dim, Distance::Cosine)),
            )
            .await
            .map_err(|e| MemoryStoreError::Connection(e.to_string()))?;

        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extracts the vector dimension from a [`GetCollectionInfoResponse`].
///
/// Returns `None` when any part of the config chain is absent (e.g. a sparse-only
/// collection), so callers can skip the validation rather than failing hard.
fn extract_vector_dimension(info: &GetCollectionInfoResponse) -> Option<u64> {
    info.result
        .as_ref()
        .and_then(|ci| ci.config.as_ref())
        .and_then(|cfg| cfg.params.as_ref())
        .and_then(|params| params.vectors_config.as_ref())
        .and_then(|vc| match &vc.config {
            Some(VectorsConfigVariant::Params(vp)) => Some(vp.size),
            _ => None,
        })
}

fn extract_uuid_from_point_id(id: Option<&PointId>) -> Result<Uuid, MemoryStoreError> {
    let point_id = id.ok_or_else(|| MemoryStoreError::Query("point has no ID".to_string()))?;
    match &point_id.point_id_options {
        Some(PointIdOptions::Uuid(s)) => {
            Uuid::parse_str(s).map_err(|e| MemoryStoreError::Query(e.to_string()))
        }
        _ => Err(MemoryStoreError::Query(
            "expected UUID point ID".to_string(),
        )),
    }
}

fn extract_created_at_from_payload(
    payload: &HashMap<String, qdrant_client::qdrant::Value>,
) -> Result<DateTime<Utc>, MemoryStoreError> {
    let raw = payload
        .get(FIELD_CREATED_AT)
        .and_then(|v| v.as_str())
        .ok_or_else(|| MemoryStoreError::Query("missing created_at in payload".to_string()))?;
    DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.into())
        .map_err(|e| MemoryStoreError::Query(e.to_string()))
}

fn parse_payload_to_memory(
    id: Uuid,
    payload: &HashMap<String, qdrant_client::qdrant::Value>,
) -> Result<Memory, MemoryStoreError> {
    let content = payload
        .get(FIELD_CONTENT)
        .and_then(|v| v.as_str())
        .ok_or_else(|| MemoryStoreError::Query("missing content in payload".to_string()))?
        .to_owned();

    let metadata: HashMap<String, String> = match payload.get(FIELD_METADATA).and_then(|v| v.as_str()) {
        Some(json) => serde_json::from_str(json)
            .map_err(|e| MemoryStoreError::Query(format!("invalid metadata JSON: {e}")))?,
        None => HashMap::new(),
    };

    let created_at = extract_created_at_from_payload(payload)?;

    Ok(Memory {
        id,
        content,
        metadata,
        created_at,
    })
}

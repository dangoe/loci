// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-memory-store-qdrant.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use loci_core::embedding::{Embedding, TextEmbedder};
use loci_core::error::MemoryStoreError;
use loci_core::memory::{
    Memory, MemoryEntry, MemoryInput, MemoryQuery, MemoryQueryMode, MemoryTier, Score,
};
use loci_core::store::MemoryStore;
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, DeletePointsBuilder, Distance, GetCollectionInfoResponse,
    GetPointsBuilder, PointId, PointStruct, PointsIdsList, SearchPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder, point_id::PointIdOptions,
    vectors_config::Config as VectorsConfigVariant,
};
use uuid::Uuid;

use crate::config::QdrantConfig;

// Payload field names
const FIELD_CONTENT: &str = "content";
const FIELD_METADATA: &str = "metadata";
const FIELD_CREATED_AT: &str = "created_at";
const FIELD_TIER: &str = "tier";
const FIELD_SEEN_COUNT: &str = "seen_count";
const FIELD_FIRST_SEEN: &str = "first_seen";
const FIELD_LAST_SEEN: &str = "last_seen";
const FIELD_EXPIRES_AT: &str = "expires_at";

const SOURCE_METADATA_KEY: &str = "source";

struct SearchCandidate {
    memory: Memory,
    similarity: f64,
}

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

    async fn do_upsert(
        &self,
        memory: &Memory,
        embedding: &Embedding,
    ) -> Result<(), MemoryStoreError> {
        let metadata_json = serde_json::to_string(&memory.metadata)
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let mut payload = Payload::new();
        payload.insert(FIELD_CONTENT, memory.content.clone());
        payload.insert(FIELD_METADATA, metadata_json);
        payload.insert(FIELD_CREATED_AT, memory.created_at.to_rfc3339());
        payload.insert(FIELD_TIER, memory.tier.as_str());
        payload.insert(FIELD_SEEN_COUNT, i64::from(memory.seen_count));
        payload.insert(FIELD_FIRST_SEEN, memory.first_seen.to_rfc3339());
        payload.insert(FIELD_LAST_SEEN, memory.last_seen.to_rfc3339());
        if let Some(expires_at) = memory.expires_at {
            payload.insert(FIELD_EXPIRES_AT, expires_at.to_rfc3339());
        }

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

        Ok(())
    }

    async fn load_memory(&self, id: Uuid) -> Result<Memory, MemoryStoreError> {
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

        let point = response
            .result
            .into_iter()
            .next()
            .ok_or(MemoryStoreError::NotFound(id))?;

        parse_payload_to_memory(id, &point.payload)
    }

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
                    .with_payload(true),
            )
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let now = Utc::now();
        let mut entries = Vec::new();
        for point in response.result {
            let candidate = Self::parse_scored_point(&point)?;
            if is_expired(candidate.memory.expires_at, now) {
                continue;
            }
            if !filters.iter().all(|(k, v)| {
                candidate
                    .memory
                    .metadata
                    .get(k)
                    .map(|s| s == v)
                    .unwrap_or(false)
            }) {
                continue;
            }

            let weighted = blend_score(candidate.similarity, candidate.memory.tier)?;
            if weighted.value() < min_score {
                continue;
            }

            entries.push(MemoryEntry {
                memory: candidate.memory,
                score: weighted,
            });
        }

        entries.sort_by(|a, b| b.score.value().total_cmp(&a.score.value()));
        entries.truncate(max_results);

        Ok(entries)
    }

    async fn search_for_dedup(
        &self,
        embedding: &Embedding,
        threshold: f64,
        filters: &HashMap<String, String>,
    ) -> Result<Option<MemoryEntry>, MemoryStoreError> {
        let vector: Vec<f32> = embedding.values().to_vec();
        let response = self
            .client
            .search_points(
                SearchPointsBuilder::new(&self.config.collection_name, vector, 1)
                    .with_payload(true),
            )
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let now = Utc::now();
        for point in response.result {
            let candidate = Self::parse_scored_point(&point)?;
            if is_expired(candidate.memory.expires_at, now) {
                continue;
            }
            if candidate.similarity < threshold {
                continue;
            }
            if !filters.iter().all(|(k, v)| {
                candidate
                    .memory
                    .metadata
                    .get(k)
                    .map(|s| s == v)
                    .unwrap_or(false)
            }) {
                continue;
            }

            let weighted = blend_score(candidate.similarity, candidate.memory.tier)?;
            return Ok(Some(MemoryEntry {
                memory: candidate.memory,
                score: weighted,
            }));
        }

        Ok(None)
    }

    fn maybe_promote_from_source(
        &self,
        memory: &mut Memory,
        incoming_metadata: &HashMap<String, String>,
    ) {
        if memory.tier != MemoryTier::Candidate {
            return;
        }

        let existing_source = memory.metadata.get(SOURCE_METADATA_KEY);
        let incoming_source = incoming_metadata.get(SOURCE_METADATA_KEY);

        if let (Some(existing), Some(incoming)) = (existing_source, incoming_source)
            && existing != incoming
            && self.config.promotion_source_threshold <= 2
        {
            memory.tier = MemoryTier::Stable;
            memory.expires_at = memory.tier.default_ttl().map(|ttl| Utc::now() + ttl);
        }
    }

    fn parse_scored_point(
        point: &qdrant_client::qdrant::ScoredPoint,
    ) -> Result<SearchCandidate, MemoryStoreError> {
        let id = extract_uuid_from_point_id(point.id.as_ref())?;
        let similarity = (point.score as f64).clamp(0.0, 1.0);
        let memory = parse_payload_to_memory(id, &point.payload)?;
        Ok(SearchCandidate { memory, similarity })
    }
}

impl<E: TextEmbedder> MemoryStore for QdrantMemoryStore<E> {
    async fn save(&self, input: MemoryInput) -> Result<MemoryEntry, MemoryStoreError> {
        let tier = input.tier.unwrap_or(MemoryTier::Candidate);
        if tier == MemoryTier::Ephemeral {
            return Err(MemoryStoreError::Query(
                "ephemeral memories are request-scoped and cannot be persisted".to_string(),
            ));
        }

        let memory = Memory::new_with_tier(input.content, input.metadata, tier);
        let embedding = self
            .embedder
            .embed(&memory.content)
            .await
            .map_err(MemoryStoreError::Embedding)?;

        if let Some(threshold) = self.config.similarity_threshold
            && let Some(mut existing) = self
                .search_for_dedup(&embedding, threshold, &memory.metadata)
                .await?
        {
            self.maybe_promote_from_source(&mut existing.memory, &memory.metadata);
            existing.memory.last_seen = Utc::now();
            existing.memory.seen_count = existing.memory.seen_count.saturating_add(1);

            let existing_embedding = self
                .embedder
                .embed(&existing.memory.content)
                .await
                .map_err(MemoryStoreError::Embedding)?;
            self.do_upsert(&existing.memory, &existing_embedding)
                .await?;

            log::debug!(
                "deduplication: reusing memory {} (score {:.4})",
                existing.memory.id,
                existing.score.value()
            );
            return Ok(existing);
        }

        self.do_upsert(&memory, &embedding).await?;
        Ok(MemoryEntry {
            memory,
            score: Score::new(1.0).expect("1.0 is always a valid score"),
        })
    }

    async fn get(&self, id: Uuid) -> Result<MemoryEntry, MemoryStoreError> {
        let memory = self.load_memory(id).await?;
        Ok(MemoryEntry {
            memory,
            score: Score::new(1.0).expect("1.0 is always a valid score"),
        })
    }

    async fn query(&self, query: MemoryQuery) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
        let embedding = self
            .embedder
            .embed(&query.topic)
            .await
            .map_err(MemoryStoreError::Embedding)?;

        let entries = self
            .search_by_embedding(
                &embedding,
                query.max_results,
                query.min_score.value(),
                &query.filters,
            )
            .await?;

        match query.mode {
            MemoryQueryMode::Lookup | MemoryQueryMode::Use => Ok(entries),
        }
    }

    async fn update(&self, id: Uuid, input: MemoryInput) -> Result<MemoryEntry, MemoryStoreError> {
        let existing = self.load_memory(id).await?;
        let tier = input.tier.unwrap_or(existing.tier);
        if tier == MemoryTier::Ephemeral {
            return Err(MemoryStoreError::Query(
                "ephemeral tier cannot be set on persisted memories".to_string(),
            ));
        }

        let now = Utc::now();
        let expires_at = if tier == existing.tier {
            existing.expires_at
        } else {
            tier.default_ttl().map(|ttl| now + ttl)
        };

        let memory = Memory {
            id,
            content: input.content,
            metadata: input.metadata,
            tier,
            seen_count: existing.seen_count,
            first_seen: existing.first_seen,
            last_seen: existing.last_seen,
            expires_at,
            created_at: existing.created_at,
        };

        let embedding = self
            .embedder
            .embed(&memory.content)
            .await
            .map_err(MemoryStoreError::Embedding)?;
        self.do_upsert(&memory, &embedding).await?;

        Ok(MemoryEntry {
            memory,
            score: Score::new(1.0).expect("1.0 is always a valid score"),
        })
    }

    async fn set_tier(&self, id: Uuid, tier: MemoryTier) -> Result<MemoryEntry, MemoryStoreError> {
        if tier == MemoryTier::Ephemeral {
            return Err(MemoryStoreError::Query(
                "ephemeral tier cannot be set on persisted memories".to_string(),
            ));
        }

        let mut memory = self.load_memory(id).await?;
        memory.tier = tier;
        memory.expires_at = tier.default_ttl().map(|ttl| Utc::now() + ttl);

        let embedding = self
            .embedder
            .embed(&memory.content)
            .await
            .map_err(MemoryStoreError::Embedding)?;
        self.do_upsert(&memory, &embedding).await?;

        Ok(MemoryEntry {
            memory,
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

fn parse_optional_datetime(
    payload: &HashMap<String, qdrant_client::qdrant::Value>,
    field: &str,
) -> Result<Option<DateTime<Utc>>, MemoryStoreError> {
    match payload.get(field).and_then(|v| v.as_str()) {
        Some(raw) => DateTime::parse_from_rfc3339(raw)
            .map(|dt| Some(dt.into()))
            .map_err(|e| MemoryStoreError::Query(e.to_string())),
        None => Ok(None),
    }
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

    let metadata: HashMap<String, String> =
        match payload.get(FIELD_METADATA).and_then(|v| v.as_str()) {
            Some(json) => serde_json::from_str(json)
                .map_err(|e| MemoryStoreError::Query(format!("invalid metadata JSON: {e}")))?,
            None => HashMap::new(),
        };

    let created_at = extract_created_at_from_payload(payload)?;
    let tier = payload
        .get(FIELD_TIER)
        .and_then(|v| v.as_str())
        .and_then(|s| MemoryTier::parse(s.as_str()))
        .unwrap_or(MemoryTier::Candidate);

    let seen_count = payload
        .get(FIELD_SEEN_COUNT)
        .and_then(|v| v.as_integer())
        .map(|v| v.max(0) as u32)
        .unwrap_or(1);

    let first_seen = parse_optional_datetime(payload, FIELD_FIRST_SEEN)?.unwrap_or(created_at);
    let last_seen = parse_optional_datetime(payload, FIELD_LAST_SEEN)?.unwrap_or(created_at);
    let expires_at = parse_optional_datetime(payload, FIELD_EXPIRES_AT)?
        .or_else(|| tier.default_ttl().map(|ttl| created_at + ttl));

    Ok(Memory {
        id,
        content,
        metadata,
        tier,
        seen_count,
        first_seen,
        last_seen,
        expires_at,
        created_at,
    })
}

fn blend_score(similarity: f64, tier: MemoryTier) -> Result<Score, MemoryStoreError> {
    let weighted = (similarity * tier.retrieval_weight()).clamp(0.0, 1.0);
    Score::new(weighted).map_err(|e| MemoryStoreError::Query(e.to_string()))
}

fn is_expired(expires_at: Option<DateTime<Utc>>, now: DateTime<Utc>) -> bool {
    expires_at.is_some_and(|dt| dt <= now)
}

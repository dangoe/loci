// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-memory-store-qdrant.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use loci_core::embedding::{Embedding, TextEmbedder};
use loci_core::error::MemoryStoreError;
use loci_core::memory::{
    MemoryEntry, MemoryInput, MemoryQuery, MemoryQueryMode, MemoryQueryResult, MemoryTier, Score,
};
use loci_core::store::{AddEntriesResult, MemoryStore};
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{Condition, Filter, Range};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, DeletePointsBuilder, Distance, GetCollectionInfoResponse,
    GetPointsBuilder, PointId, PointStruct, PointsIdsList, ScrollPointsBuilder,
    SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder, point_id::PointIdOptions,
    vectors_config::Config as VectorsConfigVariant,
};
#[cfg(feature = "background-delete")]
use tokio::spawn;
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
const FIELD_SOURCES: &str = "sources";

const SOURCE_METADATA_KEY: &str = "source";

/// Internal struct representing a search candidate with its associated memory entry and similarity score.
struct SearchCandidate {
    memory_entry: MemoryEntry,
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
        memory: &MemoryEntry,
        embedding: &Embedding,
    ) -> Result<(), MemoryStoreError> {
        let metadata_value = serde_json::to_value(&memory.metadata)
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let mut payload = Payload::new();
        payload.insert(FIELD_CONTENT, memory.content.clone());
        payload.insert(FIELD_METADATA, metadata_value);
        payload.insert(FIELD_CREATED_AT, memory.created_at.timestamp());
        payload.insert(FIELD_TIER, memory.tier.as_str());
        payload.insert(FIELD_SEEN_COUNT, i64::from(memory.seen_count));
        payload.insert(FIELD_FIRST_SEEN, memory.first_seen.timestamp());
        payload.insert(FIELD_LAST_SEEN, memory.last_seen.timestamp());
        if let Some(expires_at) = memory.expires_at {
            payload.insert(FIELD_EXPIRES_AT, expires_at.timestamp());
        }
        let sources_value = serde_json::to_value(&memory.sources)
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;
        payload.insert(FIELD_SOURCES, sources_value);

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

    async fn load_memory(&self, id: Uuid) -> Result<MemoryEntry, MemoryStoreError> {
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
    ) -> Result<Vec<MemoryQueryResult>, MemoryStoreError> {
        let vector: Vec<f32> = embedding.values().to_vec();
        let now = Utc::now();

        // Workaround until rescoring is supported
        let fetch_limit = (max_results * 5) as u64;

        let mut conditions: Vec<Condition> = filters
            .iter()
            .map(|(k, v)| Condition::matches(format!("{}.{}", FIELD_METADATA, k), v.to_string()))
            .collect();
        conditions.push(
            Filter::should(vec![
                Condition::is_null(FIELD_EXPIRES_AT),
                Condition::is_empty(FIELD_EXPIRES_AT),
                Condition::range(
                    FIELD_EXPIRES_AT,
                    Range {
                        gt: Some(now.timestamp() as f64),
                        ..Default::default()
                    },
                ),
            ])
            .into(),
        );

        let request = SearchPointsBuilder::new(&self.config.collection_name, vector, fetch_limit)
            .filter(Filter::must(conditions))
            .with_payload(true);

        let response = self
            .client
            .search_points(request)
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let mut entries = Vec::new();
        for point in response.result {
            let candidate = Self::parse_scored_point(&point)?;
            let weighted = blend_score(candidate.similarity, candidate.memory_entry.tier)?;

            if weighted.value() < min_score {
                continue;
            }

            entries.push(MemoryQueryResult {
                memory_entry: candidate.memory_entry,
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
        incoming_metadata: &HashMap<String, String>,
    ) -> Result<Option<MemoryQueryResult>, MemoryStoreError> {
        let vector: Vec<f32> = embedding.values().to_vec();
        let response = self
            .client
            .search_points(
                SearchPointsBuilder::new(&self.config.collection_name, vector, 16)
                    .with_payload(true),
            )
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let now = Utc::now();
        for point in response.result {
            let candidate = Self::parse_scored_point(&point)?;
            if is_expired(candidate.memory_entry.expires_at, now) {
                self.delete_expired(candidate.memory_entry.id).await;
                continue;
            }
            if candidate.similarity < threshold {
                continue;
            }
            if !metadata_matches_for_dedup(&candidate.memory_entry.metadata, incoming_metadata) {
                continue;
            }

            let weighted = blend_score(candidate.similarity, candidate.memory_entry.tier)?;
            return Ok(Some(MemoryQueryResult {
                memory_entry: candidate.memory_entry,
                score: weighted,
            }));
        }

        Ok(None)
    }

    fn maybe_promote_from_source(
        &self,
        memory: &mut MemoryEntry,
        incoming_metadata: &HashMap<String, String>,
    ) {
        if memory.tier != MemoryTier::Candidate {
            return;
        }

        if let Some(incoming_source) = incoming_metadata.get(SOURCE_METADATA_KEY) {
            if !memory.sources.contains(incoming_source) {
                memory.sources.push(incoming_source.clone());
            }

            if memory.sources.len() >= self.config.promotion_source_threshold as usize {
                memory.tier = MemoryTier::Stable;
                memory.expires_at = memory.tier.default_ttl().map(|ttl| Utc::now() + ttl);
            }
        }
    }

    fn parse_scored_point(
        point: &qdrant_client::qdrant::ScoredPoint,
    ) -> Result<SearchCandidate, MemoryStoreError> {
        let id = extract_uuid_from_point_id(point.id.as_ref())?;
        let similarity = (point.score as f64).clamp(0.0, 1.0);
        let memory = parse_payload_to_memory(id, &point.payload)?;
        Ok(SearchCandidate {
            memory_entry: memory,
            similarity,
        })
    }

    #[cfg(feature = "background-delete")]
    async fn delete_expired(&self, id: Uuid) {
        // Fire-and-forget: clone what we need and spawn a background task.
        // Qdrant client is cheap-to-clone (it holds an Arc internally) so this is ok.
        let client = self.client.clone();
        let collection = self.config.collection_name.clone();

        spawn(async move {
            let _ = client
                .delete_points(
                    DeletePointsBuilder::new(&collection)
                        .points(PointsIdsList {
                            ids: vec![PointId::from(id.to_string())],
                        })
                        .wait(true),
                )
                .await;
            // Intentionally ignore result: best-effort cleanup.
        });
    }

    #[cfg(not(feature = "background-delete"))]
    async fn delete_expired(&self, id: Uuid) {
        // Blocking (in-task) cleanup: await the deletion directly.
        // Intentionally ignore result: best-effort cleanup.
        let _ = self
            .client
            .delete_points(
                DeletePointsBuilder::new(&self.config.collection_name)
                    .points(PointsIdsList {
                        ids: vec![PointId::from(id.to_string())],
                    })
                    .wait(true),
            )
            .await;
    }
}

impl<E: TextEmbedder> MemoryStore for QdrantMemoryStore<E> {
    async fn add_entries(&self, inputs: Vec<MemoryInput>) -> AddEntriesResult {
        let mut added: Vec<MemoryQueryResult> = Vec::new();
        let mut failed: Vec<(MemoryInput, MemoryStoreError)> = Vec::new();

        for input in inputs.into_iter() {
            let orig_input = input.clone();

            let tier = input.tier.unwrap_or(MemoryTier::Candidate);
            if tier == MemoryTier::Ephemeral {
                failed.push((
                    orig_input,
                    MemoryStoreError::Query(
                        "ephemeral memory entries are request-scoped and cannot be persisted"
                            .to_string(),
                    ),
                ));
                continue;
            }

            let memory = MemoryEntry::new_with_tier(input.content, input.metadata, tier);

            let embedding = match self.embedder.embed(&memory.content).await {
                Ok(e) => e,
                Err(e) => {
                    failed.push((orig_input, MemoryStoreError::Embedding(e)));
                    continue;
                }
            };

            if let Some(threshold) = self.config.similarity_threshold {
                match self
                    .search_for_dedup(&embedding, threshold, &memory.metadata)
                    .await
                {
                    Ok(Some(mut existing)) => {
                        self.maybe_promote_from_source(
                            &mut existing.memory_entry,
                            &memory.metadata,
                        );
                        existing.memory_entry.last_seen = Utc::now();
                        existing.memory_entry.seen_count =
                            existing.memory_entry.seen_count.saturating_add(1);

                        if let Err(e) = self.do_upsert(&existing.memory_entry, &embedding).await {
                            failed.push((orig_input, e));
                            continue;
                        }

                        log::debug!(
                            "deduplication: reusing memory {} (score {:.4})",
                            existing.memory_entry.id,
                            existing.score.value()
                        );
                        added.push(existing);
                        continue;
                    }
                    Ok(None) => {
                        // No dedupe candidate found — proceed to upsert new memory.
                    }
                    Err(e) => {
                        failed.push((orig_input, e));
                        continue;
                    }
                }
            }

            if let Err(e) = self.do_upsert(&memory, &embedding).await {
                failed.push((orig_input, e));
                continue;
            }

            added.push(MemoryQueryResult {
                memory_entry: memory,
                score: Score::MAX,
            });
        }

        AddEntriesResult { added, failed }
    }

    async fn get_entry(&self, id: Uuid) -> Result<MemoryQueryResult, MemoryStoreError> {
        let memory = self.load_memory(id).await?;
        Ok(MemoryQueryResult {
            memory_entry: memory,
            score: Score::MAX,
        })
    }

    async fn query(&self, query: MemoryQuery) -> Result<Vec<MemoryQueryResult>, MemoryStoreError> {
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

    async fn update_entry(
        &self,
        id: Uuid,
        input: MemoryInput,
    ) -> Result<MemoryQueryResult, MemoryStoreError> {
        let existing = self.load_memory(id).await?;
        let tier = input.tier.unwrap_or(existing.tier);
        if tier == MemoryTier::Ephemeral {
            return Err(MemoryStoreError::Query(
                "ephemeral tier cannot be set on persisted memory entries".to_string(),
            ));
        }

        let now = Utc::now();
        let expires_at = if tier == existing.tier {
            existing.expires_at
        } else {
            tier.default_ttl().map(|ttl| now + ttl)
        };

        let memory = MemoryEntry {
            id,
            content: input.content,
            metadata: input.metadata,
            tier,
            seen_count: existing.seen_count,
            sources: existing.sources,
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

        Ok(MemoryQueryResult {
            memory_entry: memory,
            score: Score::MAX,
        })
    }

    async fn set_entry_tier(
        &self,
        id: Uuid,
        tier: MemoryTier,
    ) -> Result<MemoryQueryResult, MemoryStoreError> {
        if tier == MemoryTier::Ephemeral {
            return Err(MemoryStoreError::Query(
                "ephemeral tier cannot be set on persisted memory entries".to_string(),
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

        Ok(MemoryQueryResult {
            memory_entry: memory,
            score: Score::MAX,
        })
    }

    async fn delete_entry(&self, id: Uuid) -> Result<(), MemoryStoreError> {
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

    async fn prune_expired(&self) -> Result<(), MemoryStoreError> {
        let now = Utc::now();
        let mut expired_ids = Vec::new();
        let mut offset = None;

        loop {
            let mut request = ScrollPointsBuilder::new(&self.config.collection_name)
                .limit(256)
                .with_payload(true)
                .with_vectors(false);
            if let Some(next_offset) = offset.clone() {
                request = request.offset(next_offset);
            }

            let response = self
                .client
                .scroll(request)
                .await
                .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

            for point in response.result {
                let id = extract_uuid_from_point_id(point.id.as_ref())?;
                let expires_at =
                    parse_optional_value(&point.payload, FIELD_EXPIRES_AT, parse_timestamp)?;
                if is_expired(expires_at, now) {
                    expired_ids.push(PointId::from(id.to_string()));
                }
            }

            offset = response.next_page_offset;
            if offset.is_none() {
                break;
            }
        }

        if !expired_ids.is_empty() {
            self.client
                .delete_points(
                    DeletePointsBuilder::new(&self.config.collection_name)
                        .points(PointsIdsList { ids: expired_ids })
                        .wait(true),
                )
                .await
                .map_err(|e| MemoryStoreError::Query(e.to_string()))?;
        }

        Ok(())
    }
}

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

fn parse_mandatory_value<T, F>(
    payload: &HashMap<String, qdrant_client::qdrant::Value>,
    field: &str,
    parser: F,
) -> Result<T, MemoryStoreError>
where
    F: FnOnce(&qdrant_client::qdrant::Value) -> Result<T, MemoryStoreError>,
{
    let v = payload
        .get(field)
        .ok_or_else(|| MemoryStoreError::Query(format!("missing field: {field}")))?;
    parser(v)
}

fn parse_optional_value<T, F>(
    payload: &HashMap<String, qdrant_client::qdrant::Value>,
    field: &str,
    parser: F,
) -> Result<Option<T>, MemoryStoreError>
where
    F: FnOnce(&qdrant_client::qdrant::Value) -> Result<T, MemoryStoreError>,
{
    match payload.get(field) {
        Some(v) => parser(v).map(Some),
        None => Ok(None),
    }
}

fn parse_timestamp(
    value: &qdrant_client::qdrant::Value,
) -> Result<DateTime<Utc>, MemoryStoreError> {
    let raw = value
        .as_integer()
        .or_else(|| value.as_double().map(|f| f as i64))
        .ok_or_else(|| MemoryStoreError::Query(format!("{} is not a valid timestamp", value)))?;

    match DateTime::from_timestamp_secs(raw) {
        Some(dt) => Ok(dt),
        None => Err(MemoryStoreError::Query("invalid timestamp".to_string())),
    }
}

fn parse_payload_to_memory(
    id: Uuid,
    payload: &HashMap<String, qdrant_client::qdrant::Value>,
) -> Result<MemoryEntry, MemoryStoreError> {
    let content = payload
        .get(FIELD_CONTENT)
        .and_then(|v| v.as_str())
        .ok_or_else(|| MemoryStoreError::Query("missing content in payload".to_string()))?
        .to_owned();

    let metadata: HashMap<String, String> = payload
        .get(FIELD_METADATA)
        .and_then(|v| serde_json::to_value(v).ok()) // get the nested JSON object
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default();

    let created_at = parse_mandatory_value(payload, FIELD_CREATED_AT, parse_timestamp)?;
    let tier = payload
        .get(FIELD_TIER)
        .and_then(|value| value.as_str())
        .and_then(|value_as_string| {
            MemoryTier::parse(value_as_string).or_else(|| {
                log::warn!(
                    "unknown tier value '{value_as_string}' for entry {id}, defaulting to Candidate"
                );
                None
            })
        })
        .unwrap_or(MemoryTier::Candidate);

    let seen_count = payload
        .get(FIELD_SEEN_COUNT)
        .and_then(|v| v.as_integer())
        .map(|v| v.max(0) as u32)
        .unwrap_or(1);

    let first_seen =
        parse_optional_value(payload, FIELD_FIRST_SEEN, parse_timestamp)?.unwrap_or(created_at);
    let last_seen =
        parse_optional_value(payload, FIELD_LAST_SEEN, parse_timestamp)?.unwrap_or(created_at);
    let expires_at = parse_optional_value(payload, FIELD_EXPIRES_AT, parse_timestamp)?
        .or_else(|| tier.default_ttl().map(|ttl| created_at + ttl));

    let sources: Vec<String> = payload
        .get(FIELD_SOURCES)
        .and_then(|value| serde_json::to_value(value).ok())
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default();

    Ok(MemoryEntry {
        id,
        content,
        metadata,
        tier,
        seen_count,
        sources,
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

fn metadata_matches_for_dedup(
    existing: &HashMap<String, String>,
    incoming: &HashMap<String, String>,
) -> bool {
    existing
        .iter()
        .filter(|(k, _)| k.as_str() != SOURCE_METADATA_KEY)
        .all(|(k, v)| incoming.get(k) == Some(v))
        && incoming
            .iter()
            .filter(|(k, _)| k.as_str() != SOURCE_METADATA_KEY)
            .all(|(k, v)| existing.get(k) == Some(v))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use chrono::Utc;

    use loci_core::memory::MemoryTier;

    use super::{blend_score, is_expired, metadata_matches_for_dedup};

    #[test]
    fn blend_score_candidate_at_full_similarity() {
        // Candidate weight is 0.6; 1.0 * 0.6 = 0.6
        let score = blend_score(1.0, MemoryTier::Candidate).unwrap();
        assert!((score.value() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn blend_score_stable_at_full_similarity() {
        // Stable weight is 0.9
        let score = blend_score(1.0, MemoryTier::Stable).unwrap();
        assert!((score.value() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn blend_score_core_at_full_similarity() {
        // Core weight is 1.0
        let score = blend_score(1.0, MemoryTier::Core).unwrap();
        assert!((score.value() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn blend_score_ephemeral_is_zero_regardless_of_similarity() {
        // Ephemeral weight is 0.0
        let score = blend_score(1.0, MemoryTier::Ephemeral).unwrap();
        assert_eq!(score.value(), 0.0);
    }

    #[test]
    fn blend_score_zero_similarity_yields_zero() {
        for tier in [
            MemoryTier::Candidate,
            MemoryTier::Stable,
            MemoryTier::Core,
            MemoryTier::Ephemeral,
        ] {
            let score = blend_score(0.0, tier).unwrap();
            assert_eq!(score.value(), 0.0, "expected 0.0 for tier {tier:?}");
        }
    }

    #[test]
    fn blend_score_partial_similarity_is_scaled() {
        // Stable weight 0.9 * similarity 0.5 = 0.45
        let score = blend_score(0.5, MemoryTier::Stable).unwrap();
        assert!((score.value() - 0.45).abs() < 1e-9);
    }

    #[test]
    fn is_expired_none_expires_at_is_never_expired() {
        assert!(!is_expired(None, Utc::now()));
    }

    #[test]
    fn is_expired_future_timestamp_is_not_expired() {
        let future = Utc::now() + chrono::Duration::days(1);
        assert!(!is_expired(Some(future), Utc::now()));
    }

    #[test]
    fn is_expired_past_timestamp_is_expired() {
        let past = Utc::now() - chrono::Duration::days(1);
        assert!(is_expired(Some(past), Utc::now()));
    }

    #[test]
    fn is_expired_exact_now_is_expired() {
        // The function uses `<=`, so a timestamp equal to `now` is expired.
        let now = Utc::now();
        assert!(is_expired(Some(now), now));
    }

    #[test]
    fn metadata_matches_for_dedup_identical_maps_match() {
        let meta: HashMap<String, String> = [("lang".to_string(), "rust".to_string())]
            .into_iter()
            .collect();
        assert!(metadata_matches_for_dedup(&meta, &meta));
    }

    #[test]
    fn metadata_matches_for_dedup_empty_maps_match() {
        assert!(metadata_matches_for_dedup(&HashMap::new(), &HashMap::new()));
    }

    #[test]
    fn metadata_matches_for_dedup_differing_values_do_not_match() {
        let existing: HashMap<String, String> = [("env".to_string(), "prod".to_string())]
            .into_iter()
            .collect();
        let incoming: HashMap<String, String> = [("env".to_string(), "dev".to_string())]
            .into_iter()
            .collect();
        assert!(!metadata_matches_for_dedup(&existing, &incoming));
    }

    #[test]
    fn metadata_matches_for_dedup_different_keys_do_not_match() {
        let existing: HashMap<String, String> =
            [("a".to_string(), "1".to_string())].into_iter().collect();
        let incoming: HashMap<String, String> =
            [("b".to_string(), "1".to_string())].into_iter().collect();
        assert!(!metadata_matches_for_dedup(&existing, &incoming));
    }

    #[test]
    fn metadata_matches_for_dedup_source_key_is_excluded_from_comparison() {
        // Two maps that differ only in the "source" key must be considered matching,
        // because source provenance is intentionally excluded from dedup comparison.
        let existing: HashMap<String, String> = [
            ("source".to_string(), "wiki".to_string()),
            ("lang".to_string(), "rust".to_string()),
        ]
        .into_iter()
        .collect();
        let incoming: HashMap<String, String> = [
            ("source".to_string(), "blog".to_string()),
            ("lang".to_string(), "rust".to_string()),
        ]
        .into_iter()
        .collect();
        assert!(metadata_matches_for_dedup(&existing, &incoming));
    }

    #[test]
    fn metadata_matches_for_dedup_source_only_maps_match() {
        // Both maps contain only the "source" key with different values.
        // Since "source" is excluded, both filtered maps are empty → they match.
        let existing: HashMap<String, String> = [("source".to_string(), "a".to_string())]
            .into_iter()
            .collect();
        let incoming: HashMap<String, String> = [("source".to_string(), "b".to_string())]
            .into_iter()
            .collect();
        assert!(metadata_matches_for_dedup(&existing, &incoming));
    }
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-memory-store-qdrant.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use futures::future::BoxFuture;
use loci_core::embedding::{Embedding, TextEmbedder};
use loci_core::error::MemoryStoreError;
use loci_core::memory::store::{
    AddEntriesResult, MemoryInput, MemoryQuery, MemoryStore, PerEntryFailure,
};
use loci_core::memory::{MemoryEntry, MemoryTrust, Score, TrustEvidence};
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{Condition, Filter, Range};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, DeletePayloadPointsBuilder, DeletePointsBuilder, Distance,
    GetCollectionInfoResponse, GetPointsBuilder, PointId, PointStruct, PointsIdsList,
    ScrollPointsBuilder, SearchPointsBuilder, SetPayloadPointsBuilder, UpsertPointsBuilder,
    VectorParamsBuilder, point_id::PointIdOptions, vectors_config::Config as VectorsConfigVariant,
};
#[cfg(feature = "background-delete")]
use tokio::spawn;
use uuid::Uuid;

use crate::config::QdrantConfig;

// Payload field names
const FIELD_CONTENT: &str = "content";
const FIELD_METADATA: &str = "metadata";
const FIELD_CREATED_AT: &str = "created_at";
const FIELD_KIND: &str = "kind";
const FIELD_SEEN_COUNT: &str = "seen_count";
const FIELD_FIRST_SEEN: &str = "first_seen";
const FIELD_LAST_SEEN: &str = "last_seen";
const FIELD_EXPIRES_AT: &str = "expires_at";
const FIELD_CONFIDENCE: &str = "confidence";
const FIELD_TRUST_EVIDENCE_ALPHA: &str = "credibility_belief_alpha";
const FIELD_TRUST_EVIDENCE_BETA: &str = "credibility_belief_beta";

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
            .collection_exists(self.config.collection_name())
            .await
            .map_err(|e| MemoryStoreError::Connection(e.to_string()))?;

        if exists {
            let info = self
                .client
                .collection_info(self.config.collection_name())
                .await
                .map_err(|e| MemoryStoreError::Connection(e.to_string()))?;

            let expected_dim = self.embedder.embedding_dimension() as u64;
            if let Some(dim) = extract_vector_dimension(&info)
                && dim != expected_dim
            {
                return Err(MemoryStoreError::Connection(format!(
                    "collection '{}' has vector dimension {dim} \
                     but embedder produces {expected_dim}",
                    self.config.collection_name()
                )));
            }
        } else {
            let dim = self.embedder.embedding_dimension() as u64;
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(self.config.collection_name())
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
        let metadata_value = serde_json::to_value(memory.metadata())
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let mut payload = Payload::new();
        payload.insert(FIELD_CONTENT, memory.content().to_string());
        payload.insert(FIELD_METADATA, metadata_value);
        payload.insert(FIELD_CREATED_AT, memory.created_at().timestamp());
        payload.insert(FIELD_KIND, trust_kind_str(memory.trust()));
        payload.insert(FIELD_SEEN_COUNT, i64::from(memory.seen_count()));
        if let Some(first_seen) = memory.first_seen() {
            payload.insert(FIELD_FIRST_SEEN, first_seen.timestamp());
        }
        if let Some(last_seen) = memory.last_seen() {
            payload.insert(FIELD_LAST_SEEN, last_seen.timestamp());
        }
        if let Some(expires_at) = memory.expires_at() {
            payload.insert(FIELD_EXPIRES_AT, expires_at.timestamp());
        }
        if let MemoryTrust::Extracted {
            confidence,
            evidence,
        } = memory.trust()
        {
            payload.insert(FIELD_CONFIDENCE, *confidence);
            if let Some(alpha) = evidence.alpha {
                payload.insert(FIELD_TRUST_EVIDENCE_ALPHA, alpha);
            }
            if let Some(beta) = evidence.beta {
                payload.insert(FIELD_TRUST_EVIDENCE_BETA, beta);
            }
        }

        let point = PointStruct::new(
            PointId::from(memory.id().to_string()),
            embedding.values().to_vec(),
            payload,
        );

        self.client
            .upsert_points(
                UpsertPointsBuilder::new(self.config.collection_name(), vec![point]).wait(true),
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
                    self.config.collection_name(),
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
        let now = Utc::now();

        // Workaround until rescoring is supported by Qdrant rust client
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

        let request = SearchPointsBuilder::new(self.config.collection_name(), vector, fetch_limit)
            .filter(Filter::must(conditions))
            .with_payload(true);

        let response = self
            .client
            .search_points(request)
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let mut scored_entries: Vec<(Score, MemoryEntry)> = Vec::new();
        for point in response.result {
            let candidate = Self::parse_scored_point(&point)?;
            let weighted = blend_score(candidate.similarity, &candidate.memory_entry)?;

            if weighted.value() < min_score {
                continue;
            }

            scored_entries.push((weighted, candidate.memory_entry));
        }

        scored_entries.sort_by(|a, b| b.0.value().total_cmp(&a.0.value()));
        scored_entries.truncate(max_results);
        Ok(scored_entries.into_iter().map(|(_, e)| e).collect())
    }

    async fn search_for_dedup(
        &self,
        embedding: &Embedding,
        threshold: f64,
        incoming_metadata: &HashMap<String, String>,
    ) -> Result<Option<MemoryEntry>, MemoryStoreError> {
        let vector: Vec<f32> = embedding.values().to_vec();
        let response = self
            .client
            .search_points(
                SearchPointsBuilder::new(self.config.collection_name(), vector, 16)
                    .with_payload(true),
            )
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let now = Utc::now();
        for point in response.result {
            let candidate = Self::parse_scored_point(&point)?;
            if is_expired(candidate.memory_entry.expires_at(), now) {
                self.delete_expired(*candidate.memory_entry.id()).await;
                continue;
            }
            if candidate.similarity < threshold {
                continue;
            }
            if !metadata_matches_for_dedup(
                &candidate.memory_entry.metadata().clone(),
                incoming_metadata,
            ) {
                continue;
            }

            return Ok(Some(candidate.memory_entry));
        }

        Ok(None)
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
        let collection = self.config.collection_name().to_owned();

        spawn(async move {
            let _ = client
                .delete_points(
                    DeletePointsBuilder::new(collection)
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
                DeletePointsBuilder::new(self.config.collection_name())
                    .points(PointsIdsList {
                        ids: vec![PointId::from(id.to_string())],
                    })
                    .wait(true),
            )
            .await;
    }
    async fn promote_in_store(&self, memory: &MemoryEntry) -> Result<(), MemoryStoreError> {
        let mut payload = Payload::new();
        payload.insert(FIELD_KIND, trust_kind_str(memory.trust()));

        self.client
            .set_payload(
                SetPayloadPointsBuilder::new(self.config.collection_name(), payload)
                    .points_selector(PointsIdsList {
                        ids: vec![PointId::from(memory.id().to_string())],
                    })
                    .wait(true),
            )
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        self.client
            .delete_payload(
                DeletePayloadPointsBuilder::new(
                    self.config.collection_name(),
                    vec![
                        FIELD_EXPIRES_AT.to_string(),
                        FIELD_CONFIDENCE.to_string(),
                        FIELD_TRUST_EVIDENCE_ALPHA.to_string(),
                        FIELD_TRUST_EVIDENCE_BETA.to_string(),
                    ],
                )
                .points_selector(PointsIdsList {
                    ids: vec![PointId::from(memory.id().to_string())],
                })
                .wait(true),
            )
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        Ok(())
    }
}

impl<E: TextEmbedder> MemoryStore for QdrantMemoryStore<E> {
    fn add_entries<'a>(
        &'a self,
        inputs: &'a [MemoryInput],
    ) -> BoxFuture<'a, Result<AddEntriesResult, MemoryStoreError>> {
        Box::pin(async move {
            let mut added: Vec<MemoryEntry> = Vec::new();
            let mut failures: Vec<PerEntryFailure> = Vec::new();

            for (idx, input) in inputs.iter().enumerate() {
                let trust = input.trust().clone();
                let mut memory = MemoryEntry::new_with_trust(
                    input.content().to_string(),
                    input.metadata().clone(),
                    trust,
                );

                let embedding = match self.embedder.embed(memory.content()).await {
                    Ok(e) => e,
                    Err(e) => {
                        failures.push(PerEntryFailure::new(idx, MemoryStoreError::Embedding(e)));
                        continue;
                    }
                };

                if let Some(threshold) = self.config.similarity_threshold() {
                    match self
                        .search_for_dedup(&embedding, threshold, memory.metadata())
                        .await
                    {
                        Ok(Some(mut existing)) => {
                            existing.record_use();

                            if let Err(e) = self.do_upsert(&existing, &embedding).await {
                                failures.push(PerEntryFailure::new(idx, e));
                                continue;
                            }

                            log::debug!("deduplication: reusing memory {}", existing.id(),);
                            added.push(existing);
                            continue;
                        }
                        Ok(None) => {
                            // No dedupe candidate found — proceed to upsert new memory.
                        }
                        Err(e) => {
                            failures.push(PerEntryFailure::new(idx, e));
                            continue;
                        }
                    }
                }

                memory.record_use();

                if let Err(e) = self.do_upsert(&memory, &embedding).await {
                    failures.push(PerEntryFailure::new(idx, e));
                    continue;
                }

                added.push(memory);
            }

            Ok(AddEntriesResult::new(added, failures))
        })
    }

    fn get_entry<'a>(
        &'a self,
        id: &'a Uuid,
    ) -> BoxFuture<'a, Result<Option<MemoryEntry>, MemoryStoreError>> {
        Box::pin(async move {
            match self.load_memory(*id).await {
                Ok(memory) => Ok(Some(memory)),
                Err(MemoryStoreError::NotFound(_)) => Ok(None),
                Err(e) => Err(e),
            }
        })
    }

    fn query(
        &self,
        query: MemoryQuery,
    ) -> BoxFuture<'_, Result<Vec<MemoryEntry>, MemoryStoreError>> {
        Box::pin(async move {
            let embedding = self
                .embedder
                .embed(query.topic())
                .await
                .map_err(MemoryStoreError::Embedding)?;

            let entries = self
                .search_by_embedding(
                    &embedding,
                    query.max_results().get(),
                    query.min_score().value(),
                    query.filters(),
                )
                .await?;

            Ok(entries)
        })
    }

    fn promote<'a>(
        &'a self,
        id: &'a Uuid,
    ) -> BoxFuture<'a, Result<Option<MemoryEntry>, MemoryStoreError>> {
        Box::pin(async move {
            let memory = match self.load_memory(*id).await {
                Ok(m) => m,
                Err(MemoryStoreError::NotFound(_)) => return Ok(None),
                Err(e) => return Err(e),
            };

            let promoted = MemoryEntry::reconstruct(
                *memory.id(),
                memory.content().to_string(),
                memory.metadata().clone(),
                MemoryTrust::Fact,
                memory.seen_count(),
                memory.first_seen(),
                memory.last_seen(),
                None,
                memory.created_at(),
            );

            self.promote_in_store(&promoted).await?;

            Ok(Some(promoted))
        })
    }

    fn delete_entry<'a>(&'a self, id: &'a Uuid) -> BoxFuture<'a, Result<(), MemoryStoreError>> {
        Box::pin(async move {
            self.client
                .delete_points(
                    DeletePointsBuilder::new(self.config.collection_name())
                        .points(PointsIdsList {
                            ids: vec![PointId::from(id.to_string())],
                        })
                        .wait(true),
                )
                .await
                .map_err(|e| MemoryStoreError::Query(e.to_string()))?;
            Ok(())
        })
    }

    fn prune_expired(&self) -> BoxFuture<'_, Result<(), MemoryStoreError>> {
        Box::pin(async move {
            let now = Utc::now();
            let mut expired_ids = Vec::new();
            let mut offset = None;

            loop {
                let mut request = ScrollPointsBuilder::new(self.config.collection_name())
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
                        DeletePointsBuilder::new(self.config.collection_name())
                            .points(PointsIdsList { ids: expired_ids })
                            .wait(true),
                    )
                    .await
                    .map_err(|e| MemoryStoreError::Query(e.to_string()))?;
            }

            Ok(())
        })
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
        .and_then(|v| serde_json::to_value(v).ok())
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default();

    let created_at = parse_mandatory_value(payload, FIELD_CREATED_AT, parse_timestamp)?;

    let kind_str_owned;
    let kind_str = match payload.get(FIELD_KIND).and_then(|v| v.as_str()) {
        Some(s) => {
            kind_str_owned = s.clone();
            kind_str_owned.as_str()
        }
        None => {
            log::warn!("missing kind field for entry {id}, defaulting to extracted_memory");
            "extracted_memory"
        }
    };

    let confidence = payload.get(FIELD_CONFIDENCE).and_then(|v| v.as_double());
    let alpha = payload
        .get(FIELD_TRUST_EVIDENCE_ALPHA)
        .and_then(|v| v.as_double());
    let beta = payload
        .get(FIELD_TRUST_EVIDENCE_BETA)
        .and_then(|v| v.as_double());

    let trust = build_memory_trust(kind_str, confidence, alpha, beta);

    let seen_count = payload
        .get(FIELD_SEEN_COUNT)
        .and_then(|v| v.as_integer())
        .map(|v| v.max(0) as u32)
        .unwrap_or(1);

    let first_seen = parse_optional_value(payload, FIELD_FIRST_SEEN, parse_timestamp)?;
    let last_seen = parse_optional_value(payload, FIELD_LAST_SEEN, parse_timestamp)?;
    let expires_at = parse_optional_value(payload, FIELD_EXPIRES_AT, parse_timestamp)?
        .or_else(|| trust.default_ttl().map(|ttl| created_at + ttl));

    Ok(MemoryEntry::reconstruct(
        id, content, metadata, trust, seen_count, first_seen, last_seen, expires_at, created_at,
    ))
}

fn build_memory_trust(
    kind_str: &str,
    confidence: Option<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
) -> MemoryTrust {
    match kind_str {
        "fact" => MemoryTrust::Fact,
        _ => MemoryTrust::Extracted {
            confidence: confidence.unwrap_or(0.5),
            evidence: TrustEvidence { alpha, beta },
        },
    }
}

fn blend_score(similarity: f64, entry: &MemoryEntry) -> Result<Score, MemoryStoreError> {
    let weighted = (similarity * entry.trust().effective_score().value()).clamp(0.0, 1.0);
    Score::try_new(weighted).map_err(|e| MemoryStoreError::Query(e.to_string()))
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

fn trust_kind_str(trust: &MemoryTrust) -> &'static str {
    match trust {
        MemoryTrust::Fact => "fact",
        MemoryTrust::Extracted { .. } => "extracted_memory",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use chrono::Utc;

    use loci_core::memory::{MemoryEntry, MemoryTrust, TrustEvidence};

    use crate::store::build_memory_trust;

    use super::{blend_score, is_expired, metadata_matches_for_dedup};

    fn extracted_with_evidence(alpha: f64, beta: f64) -> MemoryEntry {
        MemoryEntry::new_with_trust(
            "".to_string(),
            HashMap::new(),
            MemoryTrust::Extracted {
                confidence: 0.5,
                evidence: TrustEvidence {
                    alpha: Some(alpha),
                    beta: Some(beta),
                },
            },
        )
    }

    fn extracted_with_confidence(confidence: f64) -> MemoryEntry {
        MemoryEntry::new_with_trust(
            "".to_string(),
            HashMap::new(),
            MemoryTrust::Extracted {
                confidence,
                evidence: TrustEvidence::default(),
            },
        )
    }

    fn fact_entry() -> MemoryEntry {
        MemoryEntry::new_with_trust("".to_string(), HashMap::new(), MemoryTrust::Fact)
    }

    #[test]
    fn blend_score_fact_uses_full_confidence() {
        let score = blend_score(1.0, &fact_entry()).unwrap();
        assert!((score.value() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn blend_score_extracted_memory_uses_bayesian_confidence() {
        // alpha=9, beta=1 → confidence = 0.9
        let entry = extracted_with_evidence(9.0, 1.0);
        let score = blend_score(1.0, &entry).unwrap();
        assert!((score.value() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn blend_score_extracted_memory_falls_back_to_stored_confidence() {
        let entry = extracted_with_confidence(0.7);
        let score = blend_score(1.0, &entry).unwrap();
        assert!((score.value() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn blend_score_extracted_memory_defaults_to_half_when_no_confidence() {
        let entry = MemoryEntry::new("".to_string(), HashMap::new());
        let score = blend_score(1.0, &entry).unwrap();
        assert!((score.value() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn blend_score_zero_similarity_yields_zero() {
        for entry in [
            fact_entry(),
            extracted_with_evidence(9.0, 1.0),
            extracted_with_confidence(0.8),
        ] {
            let score = blend_score(0.0, &entry).unwrap();
            assert_eq!(
                score.value(),
                0.0,
                "expected 0.0 for entry {:?}",
                entry.trust()
            );
        }
    }

    #[test]
    fn blend_score_scales_similarity_by_confidence() {
        // alpha=8, beta=2 → confidence = 0.8; similarity = 0.5 → score = 0.4
        let entry = extracted_with_evidence(8.0, 2.0);
        let score = blend_score(0.5, &entry).unwrap();
        assert!((score.value() - 0.4).abs() < 1e-9);
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
        let existing: HashMap<String, String> = [("source".to_string(), "a".to_string())]
            .into_iter()
            .collect();
        let incoming: HashMap<String, String> = [("source".to_string(), "b".to_string())]
            .into_iter()
            .collect();
        assert!(metadata_matches_for_dedup(&existing, &incoming));
    }

    #[test]
    fn test_build_memory_trust_from_payload_fact() {
        let t = build_memory_trust("fact", None, None, None);
        assert_eq!(t, MemoryTrust::Fact);
    }

    #[test]
    fn test_build_memory_trust_from_payload_extracted_with_values() {
        let t = build_memory_trust("extracted_memory", Some(0.7), Some(7.0), Some(3.0));
        assert!(matches!(
            t,
            MemoryTrust::Extracted {
                confidence,
                evidence: TrustEvidence {
                    alpha: Some(a),
                    beta: Some(b),
                },
            } if (confidence - 0.7).abs() < f64::EPSILON
              && (a - 7.0).abs() < f64::EPSILON
              && (b - 3.0).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn test_build_memory_trust_from_payload_extracted_defaults_confidence() {
        let t = build_memory_trust("extracted_memory", None, None, None);
        assert!(matches!(
            t,
            MemoryTrust::Extracted { confidence, .. }
            if (confidence - 0.5).abs() < f64::EPSILON
        ));
    }
}

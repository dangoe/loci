// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-memory-store-qdrant.

use std::collections::HashMap;
use std::future::Future;

use chrono::{Duration, Utc};
use loci_core::embedding::{Embedding, TextEmbedder};
use loci_core::error::{EmbeddingError, MemoryStoreError};
use loci_core::memory::{
    MemoryInput, MemoryQuery, MemoryQueryMode, MemoryQueryResult, MemoryTier, Score,
};
use loci_core::store::MemoryStore;
use loci_memory_store_qdrant::config::QdrantConfig;
use loci_memory_store_qdrant::store::QdrantMemoryStore;
use pretty_assertions::assert_eq;
use qdrant_client::qdrant::{PointsIdsList, SetPayloadPointsBuilder};
use qdrant_client::{Payload, Qdrant};
use testcontainers::core::wait::HttpWaitStrategy;
use testcontainers::core::{ContainerPort, IntoContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage};
use uuid::Uuid;

const DIM: usize = 4;
/// Qdrant gRPC port used by the client
const QDRANT_PORT: u16 = 6334;

/// Maps specific strings to predetermined unit vectors for predictable cosine similarity.
/// Any unmapped text returns the zero vector.
struct FakeTextEmbedder {
    mappings: HashMap<String, Vec<f32>>,
}

impl FakeTextEmbedder {
    fn new() -> Self {
        Self {
            mappings: HashMap::new(),
        }
    }

    fn with(mut self, text: &str, values: Vec<f32>) -> Self {
        self.mappings.insert(text.to_string(), values);
        self
    }
}

impl TextEmbedder for FakeTextEmbedder {
    fn embedding_dimension(&self) -> usize {
        DIM
    }

    fn embed(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<Embedding, EmbeddingError>> + Send + '_ {
        let values = self
            .mappings
            .get(text)
            .cloned()
            .unwrap_or_else(|| vec![0.0; DIM]);
        async move { Ok(Embedding::new(values)) }
    }
}

/// Unit vector along the given axis.
fn unit_vec(index: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[index] = 1.0;
    v
}

async fn start_store(
    embedder: FakeTextEmbedder,
    similarity_threshold: Option<f64>,
) -> (
    QdrantMemoryStore<FakeTextEmbedder>,
    ContainerAsync<GenericImage>,
) {
    let image = GenericImage::new("qdrant/qdrant", "latest")
        .with_exposed_port(QDRANT_PORT.tcp())
        .with_wait_for(WaitFor::http(
            HttpWaitStrategy::new("/healthz")
                .with_port(ContainerPort::Tcp(6333))
                .with_expected_status_code(200u16),
        ));

    let container: ContainerAsync<GenericImage> = image
        .start()
        .await
        .expect("Docker must be available to run Qdrant integration tests");

    let host = container.get_host().await.unwrap();
    let port = container.get_host_port_ipv4(QDRANT_PORT).await.unwrap();

    let url = format!("http://{host}:{port}");

    let config = QdrantConfig {
        collection_name: "memory_entries".to_string(),
        similarity_threshold,
        promotion_source_threshold: 2,
    };

    let store =
        QdrantMemoryStore::new(&url, config, embedder).expect("failed to create Qdrant client");
    store
        .initialize()
        .await
        .expect("failed to initialize Qdrant collection");

    (store, container)
}

fn input(content: &str) -> MemoryInput {
    MemoryInput::new(content.to_string(), HashMap::new())
}

fn input_with_metadata(content: &str, metadata: HashMap<String, String>) -> MemoryInput {
    MemoryInput::new(content.to_string(), metadata)
}

fn query(topic: &str, max_results: usize) -> MemoryQuery {
    MemoryQuery {
        topic: topic.to_string(),
        max_results,
        min_score: Score::ZERO,
        filters: HashMap::new(),
        mode: MemoryQueryMode::Lookup,
    }
}

async fn prepare_expired_entry(
    container: &ContainerAsync<GenericImage>,
    store: &QdrantMemoryStore<FakeTextEmbedder>,
) {
    let short_lived = store
        .add_entry(MemoryInput {
            content: "short-lived".to_string(),
            metadata: HashMap::new(),
            tier: Some(MemoryTier::Candidate),
        })
        .await
        .unwrap();

    let host = container.get_host().await.unwrap();
    let port = container.get_host_port_ipv4(QDRANT_PORT).await.unwrap();
    let url = format!("http://{host}:{port}");
    let client = Qdrant::from_url(&url).build().unwrap();

    let mut expired_payload = Payload::new();
    expired_payload.insert("expires_at", (Utc::now() - Duration::days(1)).timestamp());
    client
        .set_payload(
            SetPayloadPointsBuilder::new("memory_entries", expired_payload)
                .points_selector(PointsIdsList {
                    ids: vec![short_lived.memory_entry.id.to_string().into()],
                })
                .wait(true),
        )
        .await
        .unwrap();
}

fn extract_ids(results: &[MemoryQueryResult]) -> Vec<Uuid> {
    results.iter().map(|e| e.memory_entry.id).collect()
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_candidate_promotes_when_same_fact_arrives_from_different_source() {
    let embedder = FakeTextEmbedder::new()
        .with("fact from source a", unit_vec(0))
        .with("fact from source b", unit_vec(0));
    let (store, _container) = start_store(embedder, Some(0.9)).await;

    let first = store
        .add_entry(input_with_metadata(
            "fact from source a",
            HashMap::from([("source".to_string(), "https://a.example".to_string())]),
        ))
        .await
        .unwrap();
    assert_eq!(first.memory_entry.tier, MemoryTier::Candidate);

    let second = store
        .add_entry(input_with_metadata(
            "fact from source b",
            HashMap::from([("source".to_string(), "https://b.example".to_string())]),
        ))
        .await
        .unwrap();

    assert_eq!(second.memory_entry.id, first.memory_entry.id);
    assert_eq!(second.memory_entry.tier, MemoryTier::Stable);

    // Verify the promotion is persisted in the store.
    let fetched = store.get_entry(first.memory_entry.id).await.unwrap();
    assert_eq!(fetched.memory_entry.tier, MemoryTier::Stable);
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_deduplication_does_not_collapse_different_metadata() {
    // Same embedding but different non-source metadata -> must produce two distinct entries
    let embedder = FakeTextEmbedder::new()
        .with("same content a", unit_vec(0))
        .with("same content b", unit_vec(0));
    let (store, _container) = start_store(embedder, Some(0.9)).await;

    let first = store
        .add_entry(input_with_metadata(
            "same content a",
            HashMap::from([("label".to_string(), "a".to_string())]),
        ))
        .await
        .unwrap();
    let second = store
        .add_entry(input_with_metadata(
            "same content b",
            HashMap::from([("label".to_string(), "b".to_string())]),
        ))
        .await
        .unwrap();

    assert_ne!(second.memory_entry.id, first.memory_entry.id);
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_deduplication_reuses_id_when_score_meets_threshold() {
    // Both "original" and "near-duplicate" map to the same vector -> cosine = 1.0 >= 0.9
    let embedder = FakeTextEmbedder::new()
        .with("original", unit_vec(0))
        .with("near-duplicate", unit_vec(0));
    let (store, _container) = start_store(embedder, Some(0.9)).await;

    let original = store.add_entry(input("original")).await.unwrap();
    let duplicate = store.add_entry(input("near-duplicate")).await.unwrap();

    assert_eq!(duplicate.memory_entry.id, original.memory_entry.id);
    assert_eq!(duplicate.memory_entry.seen_count, 2);
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_deduplication_stores_new_memory_when_score_below_threshold() {
    // Orthogonal vectors -> cosine ~= 0 < 0.9 -> stored separately
    let embedder = FakeTextEmbedder::new()
        .with("x-axis topic", unit_vec(0))
        .with("y-axis topic", unit_vec(1));
    let (store, _container) = start_store(embedder, Some(0.9)).await;

    let first = store.add_entry(input("x-axis topic")).await.unwrap();
    let second = store.add_entry(input("y-axis topic")).await.unwrap();

    assert_ne!(second.memory_entry.id, first.memory_entry.id);
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_deleted_memory_is_not_returned_by_query() {
    let embedder = FakeTextEmbedder::new()
        .with("to be deleted", unit_vec(0))
        .with("survivor", unit_vec(0))
        .with("query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let deleted = store.add_entry(input("to be deleted")).await.unwrap();
    let survivor = store.add_entry(input("survivor")).await.unwrap();
    store.delete_entry(deleted.memory_entry.id).await.unwrap();

    let results = store.query(query("query", 10)).await.unwrap();

    // The surviving entry must still be returned so the assertion below is not vacuously true.
    assert!(
        results
            .iter()
            .any(|e| e.memory_entry.id == survivor.memory_entry.id),
        "surviving entry must appear in results"
    );
    assert!(
        results
            .iter()
            .all(|e| e.memory_entry.id != deleted.memory_entry.id),
        "deleted entry must not appear in results"
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_get_returns_not_found_for_unknown_id() {
    let embedder = FakeTextEmbedder::new();
    let (store, _container) = start_store(embedder, None).await;

    let unknown_id = Uuid::new_v4();
    let result = store.get_entry(unknown_id).await;

    assert!(matches!(result, Err(MemoryStoreError::NotFound(id)) if id == unknown_id));
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_get_returns_saved_entry() {
    let embedder = FakeTextEmbedder::new().with("fetch me", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let saved = store.add_entry(input("fetch me")).await.unwrap();
    let fetched = store.get_entry(saved.memory_entry.id).await.unwrap();

    assert_eq!(fetched.memory_entry.id, saved.memory_entry.id);
    assert_eq!(fetched.memory_entry.content, "fetch me");
    assert_eq!(fetched.memory_entry.tier, MemoryTier::Candidate);
    assert_eq!(fetched.memory_entry.seen_count, 1);
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_metadata_is_persisted_and_restored() {
    let embedder = FakeTextEmbedder::new()
        .with("tagged content", unit_vec(0))
        .with("tagged content query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let metadata = HashMap::from([
        ("source".to_string(), "test".to_string()),
        ("language".to_string(), "en".to_string()),
    ]);
    store
        .add_entry(input_with_metadata("tagged content", metadata.clone()))
        .await
        .unwrap();

    let results = store.query(query("tagged content query", 1)).await.unwrap();

    assert_eq!(results[0].memory_entry.metadata, metadata);
    assert_eq!(results[0].memory_entry.tier, MemoryTier::Candidate);
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_neither_query_mode_increments_seen_count() {
    let embedder = FakeTextEmbedder::new()
        .with("remember this", unit_vec(0))
        .with("query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let saved = store.add_entry(input("remember this")).await.unwrap();
    assert_eq!(saved.memory_entry.seen_count, 1);

    let _ = store
        .query(MemoryQuery {
            topic: "query".to_string(),
            max_results: 1,
            min_score: Score::ZERO,
            filters: HashMap::new(),
            mode: MemoryQueryMode::Use,
        })
        .await
        .unwrap();

    let looked_up = store.query(query("query", 1)).await.unwrap();
    assert_eq!(looked_up[0].memory_entry.seen_count, 1);
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_prune_expired_memory_entries() {
    let embedder = FakeTextEmbedder::new()
        .with("short-lived", unit_vec(0))
        .with("long-lived", unit_vec(0))
        .with("query", unit_vec(0));
    let (store, container) = start_store(embedder, None).await;

    let long_lived = store
        .add_entry(MemoryInput {
            content: "long-lived".to_string(),
            metadata: HashMap::new(),
            tier: Some(MemoryTier::Core),
        })
        .await
        .unwrap();

    prepare_expired_entry(&container, &store).await;

    store.prune_expired().await.unwrap();

    let results = store.query(query("query", 10)).await.unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory_entry.id, long_lived.memory_entry.id);
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_query_filters_by_metadata() {
    let embedder = FakeTextEmbedder::new()
        .with("doc en", unit_vec(0))
        .with("doc de", unit_vec(0))
        .with("topic", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    store
        .add_entry(input_with_metadata(
            "doc en",
            HashMap::from([("lang".to_string(), "en".to_string())]),
        ))
        .await
        .unwrap();
    store
        .add_entry(input_with_metadata(
            "doc de",
            HashMap::from([("lang".to_string(), "de".to_string())]),
        ))
        .await
        .unwrap();

    let results = store
        .query(MemoryQuery {
            topic: "topic".to_string(),
            max_results: 10,
            min_score: Score::ZERO,
            filters: HashMap::from([("lang".to_string(), "en".to_string())]),
            mode: MemoryQueryMode::Lookup,
        })
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory_entry.content, "doc en");
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_query_ranks_results_by_similarity() {
    let embedder = FakeTextEmbedder::new()
        .with("x-axis", unit_vec(0))
        .with("y-axis", unit_vec(1))
        .with("near x", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    store.add_entry(input("x-axis")).await.unwrap();
    store.add_entry(input("y-axis")).await.unwrap();

    // "near x" maps to the same vector as "x-axis" so it should rank first
    let results = store.query(query("near x", 2)).await.unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].memory_entry.content, "x-axis");
    assert_eq!(results[1].memory_entry.content, "y-axis");
    assert!(results[0].score.value() > results[1].score.value());
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_query_respects_expiration() {
    let embedder = FakeTextEmbedder::new()
        .with("short-lived", unit_vec(0))
        .with("core", unit_vec(0))
        .with("stable", unit_vec(0))
        .with("candidate", unit_vec(0))
        .with("query", unit_vec(0));
    let (store, container) = start_store(embedder, None).await;

    let mut valid_entries: Vec<MemoryQueryResult> = Vec::new();

    for tier in vec![MemoryTier::Core, MemoryTier::Stable, MemoryTier::Candidate] {
        valid_entries.push(
            store
                .add_entry(MemoryInput {
                    content: tier.as_str().to_string(),
                    metadata: HashMap::new(),
                    tier: Some(tier),
                })
                .await
                .unwrap(),
        );
    }

    prepare_expired_entry(&container, &store).await;

    let results = store.query(query("query", 10)).await.unwrap();

    assert_eq!(results.len(), valid_entries.len());

    let mut result_ids = extract_ids(results.as_slice());
    let mut valid_ids = extract_ids(valid_entries.as_slice());
    result_ids.sort();
    valid_ids.sort();
    assert_eq!(result_ids, valid_ids);
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_query_respects_max_results() {
    let mut embedder = FakeTextEmbedder::new().with("query", unit_vec(0));
    for i in 0..5 {
        embedder = embedder.with(&format!("memory {i}"), unit_vec(i % DIM));
    }
    let (store, _container) = start_store(embedder, None).await;

    for i in 0..5 {
        store.add_entry(input(&format!("memory {i}"))).await.unwrap();
    }

    let results = store.query(query("query", 3)).await.unwrap();

    assert_eq!(results.len(), 3);
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_query_respects_min_score() {
    // x-axis and y-axis are orthogonal -> cosine = 0
    let embedder = FakeTextEmbedder::new()
        .with("x-axis", unit_vec(0))
        .with("y-axis", unit_vec(1))
        .with("query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    store.add_entry(input("x-axis")).await.unwrap();
    store.add_entry(input("y-axis")).await.unwrap();

    // weighted min_score of 0.5 should exclude the orthogonal entry
    let results = store
        .query(MemoryQuery {
            topic: "query".to_string(),
            max_results: 10,
            min_score: Score::new(0.5).unwrap(),
            filters: HashMap::new(),
            mode: MemoryQueryMode::Lookup,
        })
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory_entry.content, "x-axis");
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_saved_memory_is_returned_by_query() {
    let embedder = FakeTextEmbedder::new()
        .with("hello world", unit_vec(0))
        .with("hello world query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let entry = store.add_entry(input("hello world")).await.unwrap();

    let results = store.query(query("hello world query", 1)).await.unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory_entry.id, entry.memory_entry.id);
    assert_eq!(results[0].memory_entry.content, "hello world");
    assert!(results[0].score.value() > 0.5);
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_save_ephemeral_returns_error() {
    let embedder = FakeTextEmbedder::new();
    let (store, _container) = start_store(embedder, None).await;

    let result = store
        .add_entry(MemoryInput {
            content: "ephemeral".to_string(),
            metadata: HashMap::new(),
            tier: Some(MemoryTier::Ephemeral),
        })
        .await;

    assert!(
        matches!(result, Err(MemoryStoreError::Query(_))),
        "saving an ephemeral entry must return a Query error"
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_set_tier_promotes_to_core() {
    let embedder = FakeTextEmbedder::new().with("curate me", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let saved = store.add_entry(input("curate me")).await.unwrap();
    let updated = store
        .set_entry_tier(saved.memory_entry.id, MemoryTier::Core)
        .await
        .unwrap();

    assert_eq!(updated.memory_entry.tier, MemoryTier::Core);
    assert_eq!(updated.memory_entry.expires_at, None);

    // Verify the tier change is persisted in the store.
    let fetched = store.get_entry(saved.memory_entry.id).await.unwrap();
    assert_eq!(fetched.memory_entry.tier, MemoryTier::Core);
    assert_eq!(fetched.memory_entry.expires_at, None);
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_set_tier_to_ephemeral_returns_error() {
    let embedder = FakeTextEmbedder::new().with("content", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let saved = store.add_entry(input("content")).await.unwrap();
    let result = store
        .set_entry_tier(saved.memory_entry.id, MemoryTier::Ephemeral)
        .await;

    assert!(
        matches!(result, Err(MemoryStoreError::Query(_))),
        "setting tier to ephemeral must return a Query error"
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_update_changes_content_and_returns_updated_entry() {
    let embedder = FakeTextEmbedder::new()
        .with("original content", unit_vec(0))
        .with("updated content", unit_vec(1));
    let (store, _container) = start_store(embedder, None).await;

    let saved = store.add_entry(input("original content")).await.unwrap();
    let updated = store
        .update_entry(saved.memory_entry.id, input("updated content"))
        .await
        .unwrap();

    assert_eq!(updated.memory_entry.id, saved.memory_entry.id);
    assert_eq!(updated.memory_entry.content, "updated content");

    // Verify the change is persisted in the store.
    let fetched = store.get_entry(saved.memory_entry.id).await.unwrap();
    assert_eq!(fetched.memory_entry.content, "updated content");
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_update_returns_not_found_for_unknown_id() {
    let embedder = FakeTextEmbedder::new().with("anything", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let unknown_id = Uuid::new_v4();
    let result = store.update_entry(unknown_id, input("anything")).await;

    assert!(matches!(result, Err(MemoryStoreError::NotFound(id)) if id == unknown_id));
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_update_to_ephemeral_returns_error() {
    let embedder = FakeTextEmbedder::new().with("content", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let saved = store.add_entry(input("content")).await.unwrap();
    let result = store
        .update_entry(
            saved.memory_entry.id,
            MemoryInput {
                content: "content".to_string(),
                metadata: HashMap::new(),
                tier: Some(MemoryTier::Ephemeral),
            },
        )
        .await;

    assert!(
        matches!(result, Err(MemoryStoreError::Query(_))),
        "updating to ephemeral tier must return a Query error"
    );
}

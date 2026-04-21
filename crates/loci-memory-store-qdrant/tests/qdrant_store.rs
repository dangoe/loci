// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-memory-store-qdrant.

#![cfg(feature = "integration")]

use std::collections::HashMap;

use chrono::{Duration, Utc};
use loci_core::memory::store::{MemoryInput, MemoryQuery, MemoryQueryMode, MemoryStore};
use loci_core::memory::{MemoryEntry, MemoryTrust, TrustEvidence};
use loci_core::testing::MockTextEmbedder;
use loci_memory_store_qdrant::store::QdrantMemoryStore;
use loci_memory_store_qdrant::testing::{QDRANT_GRPC_PORT, start_store};
use pretty_assertions::assert_eq;
use qdrant_client::qdrant::{PointsIdsList, SetPayloadPointsBuilder};
use qdrant_client::{Payload, Qdrant};
use testcontainers::{ContainerAsync, GenericImage};
use uuid::Uuid;

const DIM: usize = 4;

/// Unit vector along the given axis.
fn unit_vec(index: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[index] = 1.0;
    v
}

fn input(content: &str) -> MemoryInput {
    MemoryInput::new(
        content.to_owned(),
        MemoryTrust::Extracted {
            confidence: 0.5,
            evidence: TrustEvidence::default(),
        },
        HashMap::new(),
    )
}

fn input_with_metadata(content: &str, metadata: HashMap<String, String>) -> MemoryInput {
    MemoryInput::new(
        content.to_owned(),
        MemoryTrust::Extracted {
            confidence: 0.5,
            evidence: TrustEvidence::default(),
        },
        metadata,
    )
}

fn query(topic: &str, max_results: usize) -> MemoryQuery {
    use std::num::NonZeroUsize;
    MemoryQuery::new(topic.to_owned(), MemoryQueryMode::Lookup)
        .with_max_results(NonZeroUsize::new(max_results).unwrap())
}

async fn prepare_expired_entry(
    container: &ContainerAsync<GenericImage>,
    store: &QdrantMemoryStore<MockTextEmbedder>,
) {
    let short_lived = store
        .add_entry(&MemoryInput::new(
            "short-lived".to_owned(),
            MemoryTrust::Extracted {
                confidence: 0.5,
                evidence: TrustEvidence::default(),
            },
            HashMap::new(),
        ))
        .await
        .unwrap();

    let host = container.get_host().await.unwrap();
    let port = container
        .get_host_port_ipv4(QDRANT_GRPC_PORT)
        .await
        .unwrap();
    let url = format!("http://{host}:{port}");
    let client = Qdrant::from_url(&url).build().unwrap();

    let mut expired_payload = Payload::new();
    expired_payload.insert("expires_at", (Utc::now() - Duration::days(1)).timestamp());
    client
        .set_payload(
            SetPayloadPointsBuilder::new("memory_entries", expired_payload)
                .points_selector(PointsIdsList {
                    ids: vec![short_lived.id().to_string().into()],
                })
                .wait(true),
        )
        .await
        .unwrap();
}

fn extract_ids(results: &[MemoryEntry]) -> Vec<Uuid> {
    results.iter().map(|e| *e.id()).collect()
}

#[tokio::test]
async fn test_deduplication_does_not_collapse_different_metadata() {
    // Same embedding but different non-source metadata -> must produce two distinct entries
    let embedder = MockTextEmbedder::new(DIM)
        .with("same content a", unit_vec(0))
        .with("same content b", unit_vec(0));
    let (store, _container) = start_store(embedder, Some(0.9)).await;

    let first = store
        .add_entry(&input_with_metadata(
            "same content a",
            HashMap::from([("label".to_string(), "a".to_string())]),
        ))
        .await
        .unwrap();
    let second = store
        .add_entry(&input_with_metadata(
            "same content b",
            HashMap::from([("label".to_string(), "b".to_string())]),
        ))
        .await
        .unwrap();

    assert_ne!(second.id(), first.id());
}

#[tokio::test]
async fn test_deduplication_reuses_id_when_score_meets_threshold() {
    // Both "original" and "near-duplicate" map to the same vector -> cosine = 1.0 >= 0.9
    let embedder = MockTextEmbedder::new(DIM)
        .with("original", unit_vec(0))
        .with("near-duplicate", unit_vec(0));
    let (store, _container) = start_store(embedder, Some(0.9)).await;

    let original = store.add_entry(&input("original")).await.unwrap();
    let duplicate = store.add_entry(&input("near-duplicate")).await.unwrap();

    assert_eq!(duplicate.id(), original.id());
    assert_eq!(duplicate.seen_count(), 2);
}

#[tokio::test]
async fn test_deduplication_stores_new_memory_when_score_below_threshold() {
    // Orthogonal vectors -> cosine ~= 0 < 0.9 -> stored separately
    let embedder = MockTextEmbedder::new(DIM)
        .with("x-axis topic", unit_vec(0))
        .with("y-axis topic", unit_vec(1));
    let (store, _container) = start_store(embedder, Some(0.9)).await;

    let first = store.add_entry(&input("x-axis topic")).await.unwrap();
    let second = store.add_entry(&input("y-axis topic")).await.unwrap();

    assert_ne!(second.id(), first.id());
}

#[tokio::test]
async fn test_deleted_memory_is_not_returned_by_query() {
    let embedder = MockTextEmbedder::new(DIM)
        .with("to be deleted", unit_vec(0))
        .with("survivor", unit_vec(0))
        .with("query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let deleted = store.add_entry(&input("to be deleted")).await.unwrap();
    let survivor = store.add_entry(&input("survivor")).await.unwrap();
    store.delete_entry(deleted.id()).await.unwrap();

    let results = store.query(query("query", 10)).await.unwrap();

    // The surviving entry must still be returned so the assertion below is not vacuously true.
    assert!(
        results.iter().any(|e| e.id() == survivor.id()),
        "surviving entry must appear in results"
    );
    assert!(
        results.iter().all(|e| e.id() != deleted.id()),
        "deleted entry must not appear in results"
    );
}

#[tokio::test]
async fn test_get_entry_returns_none_for_unknown_id() {
    let embedder = MockTextEmbedder::new(DIM);
    let (store, _container) = start_store(embedder, None).await;

    let unknown_id = Uuid::new_v4();
    let result = store.get_entry(&unknown_id).await;

    assert!(matches!(result, Ok(None)));
}

#[tokio::test]
async fn test_get_entry_returns_added_entry() {
    let embedder = MockTextEmbedder::new(DIM).with("fetch me", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let saved = store.add_entry(&input("fetch me")).await.unwrap();
    let fetched = store.get_entry(saved.id()).await.unwrap().unwrap();

    assert_eq!(fetched.id(), saved.id());
    assert_eq!(fetched.content(), "fetch me");
    assert!(matches!(fetched.trust(), MemoryTrust::Extracted { .. }));
    assert_eq!(fetched.seen_count(), 1);
}

#[tokio::test]
async fn test_metadata_is_persisted_and_restored() {
    let embedder = MockTextEmbedder::new(DIM)
        .with("tagged content", unit_vec(0))
        .with("tagged content query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let metadata = HashMap::from([
        ("source".to_string(), "test".to_string()),
        ("language".to_string(), "en".to_string()),
    ]);
    store
        .add_entry(&input_with_metadata("tagged content", metadata.clone()))
        .await
        .unwrap();

    let results = store.query(query("tagged content query", 1)).await.unwrap();

    assert_eq!(results[0].metadata(), &metadata);
    assert!(matches!(results[0].trust(), MemoryTrust::Extracted { .. }));
}

#[tokio::test]
async fn test_neither_query_mode_increments_seen_count() {
    use std::num::NonZeroUsize;

    let embedder = MockTextEmbedder::new(DIM)
        .with("remember this", unit_vec(0))
        .with("query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let saved = store.add_entry(&input("remember this")).await.unwrap();
    assert_eq!(saved.seen_count(), 1);

    let after_use = store
        .query(
            MemoryQuery::new("query".to_owned(), MemoryQueryMode::Use)
                .with_max_results(NonZeroUsize::new(1).unwrap()),
        )
        .await
        .unwrap();
    assert_eq!(
        after_use[0].seen_count(),
        1,
        "Use mode should not increment seen_count"
    );

    let after_lookup = store.query(query("query", 1)).await.unwrap();
    assert_eq!(
        after_lookup[0].seen_count(),
        1,
        "Lookup mode should not increment seen_count"
    );
}

#[tokio::test]
async fn test_prune_expired_memory_entries() {
    let embedder = MockTextEmbedder::new(DIM)
        .with("short-lived", unit_vec(0))
        .with("long-lived", unit_vec(0))
        .with("query", unit_vec(0));
    let (store, container) = start_store(embedder, None).await;

    let long_lived = store
        .add_entry(&MemoryInput::new(
            "long-lived".to_owned(),
            MemoryTrust::Fact,
            HashMap::new(),
        ))
        .await
        .unwrap();

    prepare_expired_entry(&container, &store).await;

    store.prune_expired().await.unwrap();

    let results = store.query(query("query", 10)).await.unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id(), long_lived.id());
}

#[tokio::test]
async fn test_query_filters_by_metadata() {
    use std::num::NonZeroUsize;

    let embedder = MockTextEmbedder::new(DIM)
        .with("doc en", unit_vec(0))
        .with("doc de", unit_vec(0))
        .with("topic", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    store
        .add_entry(&input_with_metadata(
            "doc en",
            HashMap::from([("lang".to_string(), "en".to_string())]),
        ))
        .await
        .unwrap();
    store
        .add_entry(&input_with_metadata(
            "doc de",
            HashMap::from([("lang".to_string(), "de".to_string())]),
        ))
        .await
        .unwrap();

    let results = store
        .query(
            MemoryQuery::new("topic".to_owned(), MemoryQueryMode::Lookup)
                .with_max_results(NonZeroUsize::new(10).unwrap())
                .with_filters(HashMap::from([("lang".to_string(), "en".to_string())])),
        )
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content(), "doc en");
}

#[tokio::test]
async fn test_query_ranks_results_by_similarity() {
    let embedder = MockTextEmbedder::new(DIM)
        .with("x-axis", unit_vec(0))
        .with("y-axis", unit_vec(1))
        .with("near x", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    store.add_entry(&input("x-axis")).await.unwrap();
    store.add_entry(&input("y-axis")).await.unwrap();

    // "near x" maps to the same vector as "x-axis" so it should rank first
    let results = store.query(query("near x", 2)).await.unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].content(), "x-axis");
    assert_eq!(results[1].content(), "y-axis");
}

#[tokio::test]
async fn test_query_respects_expiration() {
    let embedder = MockTextEmbedder::new(DIM)
        .with("short-lived", unit_vec(0))
        .with("fact", unit_vec(0))
        .with("extracted", unit_vec(0))
        .with("query", unit_vec(0));
    let (store, container) = start_store(embedder, None).await;

    let mut valid_entries: Vec<MemoryEntry> = Vec::new();

    for trust in [
        MemoryTrust::Fact,
        MemoryTrust::Extracted {
            confidence: 0.5,
            evidence: Default::default(),
        },
    ] {
        let content = trust_kind_str(&trust).to_string();
        valid_entries.push(
            store
                .add_entry(&MemoryInput::new(content, trust, HashMap::new()))
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

/// Returns the storage-layer string key for a `MemoryTrust` variant (test helper).
fn trust_kind_str(trust: &MemoryTrust) -> &'static str {
    match trust {
        MemoryTrust::Fact => "fact",
        MemoryTrust::Extracted { .. } => "extracted_memory",
    }
}

#[tokio::test]
async fn test_query_respects_max_results() {
    let mut embedder = MockTextEmbedder::new(DIM).with("query", unit_vec(0));
    for i in 0..5 {
        embedder = embedder.with(&format!("memory {i}"), unit_vec(i % DIM));
    }
    let (store, _container) = start_store(embedder, None).await;

    for i in 0..5 {
        store
            .add_entry(&input(&format!("memory {i}")))
            .await
            .unwrap();
    }

    let results = store.query(query("query", 3)).await.unwrap();

    assert_eq!(results.len(), 3);
}

#[tokio::test]
async fn test_query_respects_min_score() {
    use loci_core::memory::Score;
    use std::num::NonZeroUsize;

    // x-axis and y-axis are orthogonal -> cosine = 0
    let embedder = MockTextEmbedder::new(DIM)
        .with("x-axis", unit_vec(0))
        .with("y-axis", unit_vec(1))
        .with("query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    store.add_entry(&input("x-axis")).await.unwrap();
    store.add_entry(&input("y-axis")).await.unwrap();

    // weighted min_score of 0.5 should exclude the orthogonal entry
    let results = store
        .query(
            MemoryQuery::new("query".to_owned(), MemoryQueryMode::Lookup)
                .with_max_results(NonZeroUsize::new(10).unwrap())
                .with_min_score(Score::try_new(0.5).unwrap()),
        )
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content(), "x-axis");
}

#[tokio::test]
async fn test_added_memory_is_returned_by_query() {
    let embedder = MockTextEmbedder::new(DIM)
        .with("hello world", unit_vec(0))
        .with("hello world query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let entry = store.add_entry(&input("hello world")).await.unwrap();

    let results = store.query(query("hello world query", 1)).await.unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id(), entry.id());
    assert_eq!(results[0].content(), "hello world");
}

#[tokio::test]
async fn test_promote_promotes_to_fact() {
    let embedder = MockTextEmbedder::new(DIM).with("curate me", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let saved = store.add_entry(&input("curate me")).await.unwrap();
    let promoted = store.promote(saved.id()).await.unwrap().unwrap();

    assert!(matches!(promoted.trust(), MemoryTrust::Fact));
    assert_eq!(promoted.expires_at(), None);

    // Verify the trust change is persisted in the store.
    let fetched = store.get_entry(saved.id()).await.unwrap().unwrap();
    assert!(matches!(fetched.trust(), MemoryTrust::Fact));
    assert_eq!(fetched.expires_at(), None);
}

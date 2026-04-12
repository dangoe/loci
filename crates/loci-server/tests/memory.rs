// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

#![cfg(feature = "testing")]

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use buffa::EnumValue;
use connectrpc::ErrorCode;
use pretty_assertions::assert_eq;
use rstest::rstest;
use uuid::Uuid;

use loci_core::memory::{MemoryQueryMode, MemoryTier};
use loci_core::model_provider::text_generation::TextGenerationResponse;
use loci_server::loci::memory::v1::{
    MemoryServiceAddEntryRequest, MemoryServiceDeleteEntryRequest, MemoryServiceGetEntryRequest,
    MemoryServiceQueryRequest, MemoryTier as ProtoMemoryTier,
};

use common::{
    EntryBehavior, MockStore, MockStoreErrorKind, MockTextGenerationModelProvider,
    ProviderBehavior, TestServer, make_result, mock_config,
};

fn default_provider() -> Arc<MockTextGenerationModelProvider> {
    Arc::new(MockTextGenerationModelProvider::new(
        ProviderBehavior::Stream(vec![TextGenerationResponse::done(
            "unused".to_string(),
            "test-text-model".to_string(),
            None,
        )]),
    ))
}

#[tokio::test]
async fn test_memory_add_entry_uses_real_server_and_preserves_request_mapping() {
    let id = Uuid::new_v4();
    let stored = make_result(id, "stored content", MemoryTier::Stable, 0.73);
    let store = Arc::new(MockStore::new().with_add(stored.clone()).with_get(stored));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), default_provider())
            .await;

    let response = server
        .memory_client()
        .add_entry(MemoryServiceAddEntryRequest {
            content: "remember this".to_string(),
            metadata: HashMap::from([("kind".to_string(), "fact".to_string())]),
            tier: EnumValue::from(ProtoMemoryTier::MEMORY_TIER_STABLE),
            ..Default::default()
        })
        .await
        .expect("add_entry should succeed");

    let entry = response
        .view()
        .entry
        .as_option()
        .expect("response should contain entry");
    assert_eq!(entry.id, id.to_string());
    assert_eq!(entry.content, "stored content");
    assert_eq!(
        entry.tier.as_known(),
        Some(ProtoMemoryTier::MEMORY_TIER_STABLE)
    );
    assert_eq!(entry.score, 0.73);

    let captured = store.snapshot();
    let input = captured
        .add_input
        .expect("store should capture add_entry input");
    assert_eq!(input.content, "remember this");
    assert_eq!(input.metadata.get("kind"), Some(&"fact".to_string()));
    assert_eq!(input.tier, Some(MemoryTier::Stable));
}

#[tokio::test]
async fn test_memory_query_uses_lookup_mode_and_preserves_filters() {
    let result = make_result(Uuid::new_v4(), "User name is Bob", MemoryTier::Core, 0.91);
    let store = Arc::new(
        MockStore::new()
            .with_add(result.clone())
            .with_get(result.clone())
            .with_query(vec![result]),
    );
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), default_provider())
            .await;

    let response = server
        .memory_client()
        .query(MemoryServiceQueryRequest {
            topic: "who is the user".to_string(),
            max_results: 3,
            min_score: 0.55,
            filters: HashMap::from([("scope".to_string(), "profile".to_string())]),
            ..Default::default()
        })
        .await
        .expect("query should succeed");

    assert_eq!(response.view().entries.len(), 1);
    assert_eq!(response.view().entries[0].content, "User name is Bob");
    assert_eq!(
        response.view().entries[0].tier.as_known(),
        Some(ProtoMemoryTier::MEMORY_TIER_CORE)
    );

    let captured = store.snapshot();
    let query = captured.query.expect("store should capture query");
    assert_eq!(query.topic, "who is the user");
    assert_eq!(query.max_results, 3);
    assert_eq!(query.min_score.value(), 0.55);
    assert_eq!(query.filters.get("scope"), Some(&"profile".to_string()));
    assert_eq!(query.mode, MemoryQueryMode::Lookup);
}

#[tokio::test]
async fn test_memory_get_entry_translates_not_found_errors() {
    let missing_id = Uuid::new_v4();
    let store = Arc::new(
        MockStore::new()
            .with_add_behavior(EntryBehavior::Err(MockStoreErrorKind::NotFound(missing_id)))
            .with_get_behavior(EntryBehavior::Err(MockStoreErrorKind::NotFound(missing_id))),
    );
    let server = TestServer::start_with_components(mock_config(), store, default_provider()).await;

    let error = server
        .memory_client()
        .get_entry(MemoryServiceGetEntryRequest {
            id: missing_id.to_string(),
            ..Default::default()
        })
        .await
        .expect_err("get_entry should return not found");

    assert_eq!(error.code, ErrorCode::NotFound);
    assert_eq!(
        error
            .message
            .as_deref()
            .expect("error should include a message"),
        format!("memory entry not found: {missing_id}")
    );
}

#[rstest]
#[case(true)]
#[case(false)]
#[tokio::test]
async fn test_memory_rpcs_reject_invalid_ids(#[case] get_method: bool) {
    let result = make_result(Uuid::new_v4(), "unused", MemoryTier::Candidate, 0.12);
    let store = Arc::new(MockStore::new().with_add(result.clone()).with_get(result));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), default_provider())
            .await;

    let error = if get_method {
        server
            .memory_client()
            .get_entry(MemoryServiceGetEntryRequest {
                id: "not-a-uuid".to_string(),
                ..Default::default()
            })
            .await
            .expect_err("get_entry should reject invalid ids")
    } else {
        server
            .memory_client()
            .delete_entry(MemoryServiceDeleteEntryRequest {
                id: "not-a-uuid".to_string(),
                ..Default::default()
            })
            .await
            .expect_err("delete_entry should reject invalid ids")
    };

    assert_eq!(error.code, ErrorCode::InvalidArgument);
    assert_eq!(
        error
            .message
            .as_deref()
            .expect("error should include a message"),
        "invalid id: not-a-uuid"
    );

    let captured = store.snapshot();
    assert_eq!(captured.get_id, None);
    assert_eq!(captured.delete_id, None);
}

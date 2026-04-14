// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

#![cfg(feature = "testing")]

mod common;

use pretty_assertions::assert_eq;
use uuid::Uuid;

use loci_core::memory::MemoryTier;
use loci_core::model_provider::text_generation::TextGenerationResponse;

use loci_cli::commands::memory::MemoryCommand;

use common::{MockStore, MockTextGenerationModelProvider, ProviderBehavior, TestCli, make_result};

fn default_provider() -> MockTextGenerationModelProvider {
    MockTextGenerationModelProvider::new(ProviderBehavior::Stream(vec![
        TextGenerationResponse::done("unused".to_string(), "mock".to_string(), None),
    ]))
}

#[tokio::test]
async fn test_memory_add_outputs_json_with_entry_fields() {
    let id = Uuid::new_v4();
    let entry = make_result(id, "hello world", MemoryTier::Candidate, 0.0);
    let cli = TestCli::new(MockStore::new().with_add(entry), default_provider());

    let output = cli
        .memory(MemoryCommand::Add {
            content: "hello world".to_string(),
            metadata: vec![],
            tier: None,
        })
        .await
        .expect("add should succeed");

    let v: serde_json::Value = serde_json::from_str(&output).expect("output should be valid JSON");
    assert_eq!(v["id"].as_str().unwrap(), id.to_string());
    assert_eq!(v["content"].as_str().unwrap(), "hello world");
    assert_eq!(v["tier"].as_str().unwrap(), "candidate");
}

#[tokio::test]
async fn test_memory_query_outputs_json_array() {
    let result = make_result(Uuid::new_v4(), "remembered fact", MemoryTier::Stable, 0.88);
    let cli = TestCli::new(
        MockStore::new().with_query(vec![result]),
        default_provider(),
    );

    let output = cli
        .memory(MemoryCommand::Query {
            topic: "facts".to_string(),
            max_results: 5,
            min_score: 0.0,
            filters: vec![],
        })
        .await
        .expect("query should succeed");

    let v: serde_json::Value = serde_json::from_str(&output).expect("output should be valid JSON");
    let arr = v.as_array().expect("output should be a JSON array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["content"].as_str().unwrap(), "remembered fact");
    assert_eq!(arr[0]["tier"].as_str().unwrap(), "stable");
}

#[tokio::test]
async fn test_memory_get_outputs_json() {
    let id = Uuid::new_v4();
    let entry = make_result(id, "found entry", MemoryTier::Core, 0.95);
    let cli = TestCli::new(MockStore::new().with_get(entry), default_provider());

    let output = cli
        .memory(MemoryCommand::Get { id })
        .await
        .expect("get should succeed");

    let v: serde_json::Value = serde_json::from_str(&output).expect("output should be valid JSON");
    assert_eq!(v["id"].as_str().unwrap(), id.to_string());
    assert_eq!(v["content"].as_str().unwrap(), "found entry");
    assert_eq!(v["tier"].as_str().unwrap(), "core");
}

#[tokio::test]
async fn test_memory_update_outputs_json() {
    let id = Uuid::new_v4();
    let updated = make_result(id, "updated content", MemoryTier::Stable, 0.85);
    let cli = TestCli::new(
        MockStore::new()
            .with_get(updated.clone())
            .with_update(updated),
        default_provider(),
    );

    let output = cli
        .memory(MemoryCommand::Update {
            id,
            content: Some("updated content".to_string()),
            metadata: vec![],
            tier: None,
        })
        .await
        .expect("update should succeed");

    let v: serde_json::Value = serde_json::from_str(&output).expect("output should be valid JSON");
    assert_eq!(v["id"].as_str().unwrap(), id.to_string());
    assert_eq!(v["content"].as_str().unwrap(), "updated content");
}

#[tokio::test]
async fn test_memory_delete_outputs_deleted_id() {
    let id = Uuid::new_v4();
    let cli = TestCli::new(MockStore::new(), default_provider());

    let output = cli
        .memory(MemoryCommand::Delete { id })
        .await
        .expect("delete should succeed");

    assert!(
        output.contains(&id.to_string()),
        "output should contain the deleted id"
    );
}

#[tokio::test]
async fn test_memory_prune_expired_outputs_success() {
    let cli = TestCli::new(MockStore::new(), default_provider());

    let output = cli
        .memory(MemoryCommand::PruneExpired)
        .await
        .expect("prune should succeed");

    let v: serde_json::Value = serde_json::from_str(&output).expect("output should be valid JSON");
    assert_eq!(
        v["expired pruned"].as_bool().unwrap(),
        true,
        "prune output should confirm pruning succeeded"
    );
}

#[tokio::test]
async fn test_memory_add_propagates_store_errors() {
    // MockStore::new() defaults add_behavior to Err(Connection("mock: not configured"))
    let cli = TestCli::new(MockStore::new(), default_provider());

    let result = cli
        .memory(MemoryCommand::Add {
            content: "will fail".to_string(),
            metadata: vec![],
            tier: None,
        })
        .await;

    assert!(result.is_err(), "add should propagate store error");
}

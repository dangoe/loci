// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

#![cfg(feature = "testing")]

mod common;

use pretty_assertions::assert_eq;
use uuid::Uuid;

use loci_core::memory::{MemoryTrust, TrustEvidence};
use loci_core::model_provider::text_generation::TextGenerationResponse;
use loci_core::testing::AddEntriesBehavior;

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
    let entry = make_result(
        id,
        "hello world",
        MemoryTrust::Extracted {
            confidence: 0.5,
            evidence: TrustEvidence::default(),
        },
        0.0,
    );
    let cli = TestCli::new(MockStore::new().with_add(entry), default_provider());

    let output = cli
        .memory(MemoryCommand::Add {
            content: "hello world".to_string(),
            metadata: vec![],
            kind: None,
        })
        .await
        .expect("add should succeed");

    let v: serde_json::Value = serde_json::from_str(&output).expect("output should be valid JSON");
    assert_eq!(v["id"].as_str().unwrap(), id.to_string());
    assert_eq!(v["content"].as_str().unwrap(), "hello world");
    assert_eq!(v["kind"].as_str().unwrap(), "extracted_memory");
}

#[tokio::test]
async fn test_memory_query_outputs_json_array() {
    let result = make_result(
        Uuid::new_v4(),
        "remembered fact",
        MemoryTrust::Extracted {
            confidence: 0.5,
            evidence: TrustEvidence::default(),
        },
        0.88,
    );
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
    assert_eq!(arr[0]["kind"].as_str().unwrap(), "extracted_memory");
}

#[tokio::test]
async fn test_memory_get_outputs_json() {
    let id = Uuid::new_v4();
    let entry = make_result(id, "found entry", MemoryTrust::Fact, 0.95);
    let cli = TestCli::new(MockStore::new().with_get(entry), default_provider());

    let output = cli
        .memory(MemoryCommand::Get { id })
        .await
        .expect("get should succeed");

    let v: serde_json::Value = serde_json::from_str(&output).expect("output should be valid JSON");
    assert_eq!(v["id"].as_str().unwrap(), id.to_string());
    assert_eq!(v["content"].as_str().unwrap(), "found entry");
    assert_eq!(v["kind"].as_str().unwrap(), "fact");
}

#[tokio::test]
async fn test_memory_promote_outputs_json() {
    let id = Uuid::new_v4();
    let promoted = make_result(id, "important fact", MemoryTrust::Fact, 1.0);
    let cli = TestCli::new(MockStore::new().with_set_kind(promoted), default_provider());

    let output = cli
        .memory(MemoryCommand::Promote { id })
        .await
        .expect("promote should succeed");

    let v: serde_json::Value = serde_json::from_str(&output).expect("output should be valid JSON");
    assert_eq!(v["id"].as_str().unwrap(), id.to_string());
    assert_eq!(v["content"].as_str().unwrap(), "important fact");
    assert_eq!(v["kind"].as_str().unwrap(), "fact");
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
            kind: None,
        })
        .await;

    assert!(result.is_err(), "add should propagate store error");
}

fn extraction_provider(response_json: &str) -> MockTextGenerationModelProvider {
    MockTextGenerationModelProvider::new(ProviderBehavior::Stream(vec![
        TextGenerationResponse::done(response_json.to_string(), "mock".to_string(), None),
    ]))
}

#[tokio::test]
async fn test_memory_extract_dry_run_outputs_candidates_without_persisting() {
    let id = uuid::Uuid::new_v4();
    let stored = vec![make_result(
        id,
        "extracted fact",
        MemoryTrust::Extracted {
            confidence: 0.5,
            evidence: TrustEvidence::default(),
        },
        0.0,
    )];
    let store = MockStore::new().with_add_entries_behavior(AddEntriesBehavior::Ok(stored));
    let provider = extraction_provider(r#"[{"content": "extracted fact", "confidence": 0.9}]"#);
    let cli = TestCli::new(store, provider);

    let output = cli
        .memory(MemoryCommand::Extract {
            text: Some("some interesting text".to_string()),
            files: vec![],
            metadata: vec![],
            max_entries: None,
            min_confidence: None,
            guidelines: None,
            dry_run: true,
        })
        .await
        .expect("dry_run extract should succeed");

    let v: serde_json::Value = serde_json::from_str(&output).expect("output should be valid JSON");
    let arr = v.as_array().expect("output should be a JSON array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["content"].as_str().unwrap(), "extracted fact");
    assert_eq!(arr[0]["kind"].as_str().unwrap(), "extracted_memory");
}

#[tokio::test]
async fn test_memory_extract_persists_and_outputs_added_result() {
    let id = uuid::Uuid::new_v4();
    let stored = vec![make_result(
        id,
        "a fact",
        MemoryTrust::Extracted {
            confidence: 0.5,
            evidence: TrustEvidence::default(),
        },
        0.0,
    )];
    let store = MockStore::new().with_add_entries_behavior(AddEntriesBehavior::Ok(stored));
    let provider = extraction_provider(r#"[{"content": "a fact", "confidence": 0.9}]"#);
    let cli = TestCli::new(store, provider);

    let output = cli
        .memory(MemoryCommand::Extract {
            text: Some("text containing a fact".to_string()),
            files: vec![],
            metadata: vec![],
            max_entries: None,
            min_confidence: None,
            guidelines: None,
            dry_run: false,
        })
        .await
        .expect("extract should succeed");

    let v: serde_json::Value = serde_json::from_str(&output).expect("output should be valid JSON");
    assert!(
        v.get("inserted").is_some(),
        "output should have 'inserted' key"
    );
    assert!(v.get("merged").is_some(), "output should have 'merged' key");
    assert!(
        v.get("promoted").is_some(),
        "output should have 'promoted' key"
    );
    assert!(
        v.get("discarded").is_some(),
        "output should have 'discarded' key"
    );
}

#[tokio::test]
async fn test_memory_extract_empty_text_returns_error() {
    let cli = TestCli::new(MockStore::new(), default_provider());

    let result = cli
        .memory(MemoryCommand::Extract {
            text: Some("   ".to_string()),
            files: vec![],
            metadata: vec![],
            max_entries: None,
            min_confidence: None,
            guidelines: None,
            dry_run: false,
        })
        .await;

    assert!(result.is_err(), "empty text should return an error");
}

#[tokio::test]
async fn test_memory_extract_conflicting_input_returns_error() {
    let cli = TestCli::new(MockStore::new(), default_provider());

    let result = cli
        .memory(MemoryCommand::Extract {
            text: Some("positional text".to_string()),
            files: vec![std::path::PathBuf::from("some_file.txt")],
            metadata: vec![],
            max_entries: None,
            min_confidence: None,
            guidelines: None,
            dry_run: false,
        })
        .await;

    assert!(result.is_err(), "conflicting input should return an error");
}

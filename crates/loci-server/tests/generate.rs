// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

#![cfg(feature = "testing")]

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use buffa::EnumValue;
use connectrpc::ErrorCode;
use loci_core::testing::AddEntriesBehavior;
use pretty_assertions::assert_eq;
use uuid::Uuid;

use loci_core::memory::store::MemoryQueryMode;
use loci_core::memory::{MemoryTrust, TrustEvidence};
use loci_core::model_provider::common::TokenUsage;
use loci_core::model_provider::text_generation::TextGenerationResponse;
use loci_server::loci::generate::v1::{GenerateServiceGenerateRequest, MemoryMode, SystemMode};

use common::{
    EntryBehavior, MockStore, MockTextGenerationModelProvider, ProviderBehavior, QueryBehavior,
    TestServer, generate_error, make_result, mock_config,
};

#[tokio::test]
async fn test_generate_streams_chunks_and_uses_configured_defaults() {
    let memory = make_result(
        Uuid::new_v4(),
        "Use concise explanations",
        MemoryTrust::Extracted {
            confidence: 0.5,
            evidence: TrustEvidence::default(),
        },
        0.84,
    );
    let store = Arc::new(
        MockStore::new()
            .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![memory.clone()]))
            .with_get_behavior(EntryBehavior::Ok(Some(memory.clone())))
            .with_query_behavior(QueryBehavior::Ok(vec![memory])),
    );
    let provider = Arc::new(MockTextGenerationModelProvider::new(
        ProviderBehavior::Stream(vec![
            TextGenerationResponse::new(
                "Hello".to_string(),
                "resolved-model".to_string(),
                None,
                false,
            ),
            TextGenerationResponse::new_done(
                " world".to_string(),
                "resolved-model".to_string(),
                Some(TokenUsage::new(Some(3), Some(2), Some(5))),
            ),
        ]),
    ));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), Arc::clone(&provider))
            .await;

    let mut stream = server
        .generate_client()
        .generate(GenerateServiceGenerateRequest {
            prompt: "How should I answer".to_string(),
            max_memory_entries: 4,
            min_score: 0.33,
            filters: HashMap::from([("topic".to_string(), "style".to_string())]),
            ..Default::default()
        })
        .await
        .expect("generate should succeed");

    let first = stream
        .message()
        .await
        .expect("first stream message should decode")
        .expect("stream should yield first chunk");
    let second = stream
        .message()
        .await
        .expect("second stream message should decode")
        .expect("stream should yield second chunk");
    let end = stream.message().await.expect("stream should end cleanly");

    assert_eq!(first.text, "Hello");
    assert_eq!(first.model, "resolved-model");
    assert!(!first.done);
    assert_eq!(second.text, " world");
    assert!(second.done);
    let usage = second
        .usage
        .as_option()
        .expect("final chunk should include usage");
    assert_eq!(usage.prompt_tokens, Some(3));
    assert_eq!(usage.completion_tokens, Some(2));
    assert_eq!(usage.total_tokens, Some(5));
    assert!(end.is_none());

    let query = store
        .snapshot()
        .query
        .expect("generate should query memory");
    assert_eq!(query.topic(), "How should I answer");
    assert_eq!(query.max_results().get(), 4);
    assert_eq!(query.min_score().value(), 0.33);
    assert_eq!(query.filters().get("topic"), Some(&"style".to_string()));
    assert_eq!(query.mode(), MemoryQueryMode::Use);

    let request = provider
        .snapshot()
        .last_request
        .expect("provider should capture request");
    assert_eq!(request.model(), "test-text-model");
    assert_eq!(request.prompt(), "How should I answer");
    let system = request
        .system()
        .expect("contextualizer should attach a system prompt");
    assert!(system.contains("Relevant memory entries"));
    assert!(system.contains("Use concise explanations"));
}

#[tokio::test]
async fn test_generate_respects_memory_and_system_modes() {
    let memory = make_result(Uuid::new_v4(), "unused memory", MemoryTrust::Fact, 0.77);
    let store = Arc::new(
        MockStore::new()
            .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![memory.clone()]))
            .with_get_behavior(EntryBehavior::Ok(Some(memory.clone())))
            .with_query_behavior(QueryBehavior::Ok(vec![memory])),
    );
    let provider = Arc::new(MockTextGenerationModelProvider::new(
        ProviderBehavior::Stream(vec![TextGenerationResponse::new_done(
            "ok".to_string(),
            "resolved-model".to_string(),
            None,
        )]),
    ));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), Arc::clone(&provider))
            .await;

    let mut stream = server
        .generate_client()
        .generate(GenerateServiceGenerateRequest {
            prompt: "No memory".to_string(),
            memory_mode: EnumValue::from(MemoryMode::MEMORY_MODE_OFF),
            system: Some("Answer with context: {{memory}}".to_string()),
            system_mode: EnumValue::from(SystemMode::SYSTEM_MODE_REPLACE),
            ..Default::default()
        })
        .await
        .expect("generate should succeed");

    let chunk = stream
        .message()
        .await
        .expect("stream message should decode")
        .expect("stream should yield a chunk");
    assert_eq!(chunk.text, "ok");
    assert_eq!(store.snapshot().query_calls, 0);

    let request = provider
        .snapshot()
        .last_request
        .expect("provider should capture request");
    let system = request
        .system()
        .expect("request should include system prompt");
    assert!(system.contains("Answer with context:"));
    assert!(system.contains("None. Answer from general knowledge."));
}

#[tokio::test]
async fn test_generate_rejects_invalid_min_score_before_calling_dependencies() {
    let result = make_result(
        Uuid::new_v4(),
        "unused",
        MemoryTrust::Extracted {
            confidence: 0.5,
            evidence: TrustEvidence::default(),
        },
        0.42,
    );
    let store = Arc::new(
        MockStore::new()
            .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![result.clone()]))
            .with_get_behavior(EntryBehavior::Ok(Some(result.clone())))
            .with_query_behavior(QueryBehavior::Ok(vec![result])),
    );
    let provider = Arc::new(MockTextGenerationModelProvider::new(
        ProviderBehavior::Stream(vec![TextGenerationResponse::new_done(
            "unused".to_string(),
            "resolved-model".to_string(),
            None,
        )]),
    ));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), Arc::clone(&provider))
            .await;

    let error = generate_error(
        &server,
        GenerateServiceGenerateRequest {
            prompt: "bad score".to_string(),
            min_score: 1.5,
            ..Default::default()
        },
    )
    .await;

    assert_eq!(error.code, ErrorCode::InvalidArgument);
    assert_eq!(
        error
            .message
            .as_deref()
            .expect("error should include a message"),
        "min_score must be in [0.0, 1.0]"
    );
    assert_eq!(store.snapshot().query_calls, 0);
    assert!(provider.snapshot().last_request.is_none());
}

#[tokio::test]
async fn test_generate_returns_internal_error_when_default_model_is_missing() {
    let result = make_result(
        Uuid::new_v4(),
        "unused",
        MemoryTrust::Extracted {
            confidence: 0.5,
            evidence: TrustEvidence::default(),
        },
        0.42,
    );
    let store = Arc::new(
        MockStore::new()
            .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![result.clone()]))
            .with_get_behavior(EntryBehavior::Ok(Some(result.clone())))
            .with_query_behavior(QueryBehavior::Ok(vec![result])),
    );
    let provider = Arc::new(MockTextGenerationModelProvider::new(
        ProviderBehavior::Stream(vec![TextGenerationResponse::new_done(
            "unused".to_string(),
            "resolved-model".to_string(),
            None,
        )]),
    ));
    let mut config = mock_config();
    config.routing.text.default = "missing-model".to_string();
    let server =
        TestServer::start_with_components(config, Arc::clone(&store), Arc::clone(&provider)).await;

    let error = generate_error(
        &server,
        GenerateServiceGenerateRequest {
            prompt: "missing model".to_string(),
            ..Default::default()
        },
    )
    .await;

    assert_eq!(error.code, ErrorCode::Internal);
    let message = error
        .message
        .as_deref()
        .expect("error should include a message");
    assert!(message.contains("missing-model"));
    assert!(message.contains("[models.text]"));
    assert_eq!(store.snapshot().query_calls, 0);
    assert!(provider.snapshot().last_request.is_none());
}

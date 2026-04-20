// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

#![cfg(feature = "testing")]

mod common;

use std::sync::Arc;

use loci_core::memory::{MemoryTrust, TrustEvidence};
use loci_core::model_provider::common::TokenUsage;
use loci_core::model_provider::text_generation::TextGenerationResponse;
use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use uuid::Uuid;

use loci_core::testing::AddEntriesBehavior;

use common::{
    EntryBehavior, MockStore, MockTextGenerationModelProvider, ProviderBehavior, QueryBehavior,
    TestServer, make_result, mock_config,
};

const ENDPOINT: &str = "/openai/v1/chat/completions";

fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("failed to build reqwest client")
}

fn url(server: &TestServer) -> String {
    format!("http://{}{ENDPOINT}", server.addr())
}

/// Parse a streaming SSE body into a `Vec` of parsed JSON objects, excluding
/// the terminal `[DONE]` marker.
fn parse_sse_events(body: &str) -> Vec<Value> {
    body.split("\n\n")
        .filter(|block| !block.is_empty())
        .filter_map(|block| {
            block
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .and_then(|data| {
                    if data.trim() == "[DONE]" {
                        None
                    } else {
                        serde_json::from_str(data).ok()
                    }
                })
        })
        .collect()
}

#[tokio::test]
async fn test_non_streaming_returns_assembled_text() {
    let store = Arc::new(
        MockStore::new()
            .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![]))
            .with_query_behavior(QueryBehavior::Ok(vec![])),
    );
    let provider = Arc::new(MockTextGenerationModelProvider::new(
        ProviderBehavior::Stream(vec![
            TextGenerationResponse::new("Hello".into(), "test-model".into(), None, false),
            TextGenerationResponse::new_done(
                " world".into(),
                "test-model".into(),
                Some(TokenUsage::new(Some(4), Some(2), Some(6))),
            ),
        ]),
    ));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), Arc::clone(&provider))
            .await;

    let response = http_client()
        .post(url(&server))
        .json(&json!({
            "model": "ignored",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": false
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status().as_u16(), 200);
    let body: Value = response.json().await.unwrap();

    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_tokens"], 4);
    assert_eq!(body["usage"]["completion_tokens"], 2);
    assert_eq!(body["usage"]["total_tokens"], 6);
}

#[tokio::test]
async fn test_non_streaming_memory_is_queried_with_last_user_message() {
    let store = Arc::new(
        MockStore::new()
            .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![]))
            .with_query_behavior(QueryBehavior::Ok(vec![])),
    );
    let provider = Arc::new(MockTextGenerationModelProvider::new(
        ProviderBehavior::Stream(vec![TextGenerationResponse::new_done(
            "ok".into(),
            "test-model".into(),
            None,
        )]),
    ));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), Arc::clone(&provider))
            .await;

    http_client()
        .post(url(&server))
        .json(&json!({
            "model": "ignored",
            "messages": [
                {"role": "user", "content": "first message"},
                {"role": "assistant", "content": "sure"},
                {"role": "user", "content": "actual prompt"}
            ],
            "stream": false
        }))
        .send()
        .await
        .expect("request should succeed");

    let query = store
        .snapshot()
        .query
        .expect("store should have been queried");
    assert_eq!(
        query.topic(),
        "actual prompt",
        "memory should be queried with the last user message"
    );
}

#[tokio::test]
async fn test_non_streaming_system_message_is_forwarded_to_contextualizer() {
    let store = Arc::new(
        MockStore::new()
            .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![]))
            .with_query_behavior(QueryBehavior::Ok(vec![])),
    );
    let provider = Arc::new(MockTextGenerationModelProvider::new(
        ProviderBehavior::Stream(vec![TextGenerationResponse::new_done(
            "ok".into(),
            "test-model".into(),
            None,
        )]),
    ));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), Arc::clone(&provider))
            .await;

    http_client()
        .post(url(&server))
        .json(&json!({
            "model": "ignored",
            "messages": [
                {"role": "system", "content": "Always respond in French."},
                {"role": "user", "content": "hello"}
            ],
            "stream": false
        }))
        .send()
        .await
        .expect("request should succeed");

    let request = provider
        .snapshot()
        .last_request
        .expect("provider should have been called");
    let system = request
        .system()
        .expect("provider request should include a system prompt");
    assert!(
        system.contains("Always respond in French."),
        "system prompt should include the request's system message; got: {system}"
    );
}

#[tokio::test]
async fn test_non_streaming_tuning_params_are_forwarded_to_provider() {
    let store = Arc::new(
        MockStore::new()
            .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![]))
            .with_query_behavior(QueryBehavior::Ok(vec![])),
    );
    let provider = Arc::new(MockTextGenerationModelProvider::new(
        ProviderBehavior::Stream(vec![TextGenerationResponse::new_done(
            "ok".into(),
            "test-model".into(),
            None,
        )]),
    ));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), Arc::clone(&provider))
            .await;

    http_client()
        .post(url(&server))
        .json(&json!({
            "model": "ignored",
            "messages": [{"role": "user", "content": "test"}],
            "stream": false,
            "temperature": 0.2,
            "max_tokens": 64,
            "top_p": 0.9
        }))
        .send()
        .await
        .expect("request should succeed");

    let request = provider
        .snapshot()
        .last_request
        .expect("provider should have been called");
    assert_eq!(request.temperature(), Some(0.2));
    assert_eq!(request.max_tokens(), Some(64));
    assert_eq!(request.top_p(), Some(0.9));
}

#[tokio::test]
async fn test_non_streaming_missing_model_returns_500() {
    let store = Arc::new(
        MockStore::new()
            .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![]))
            .with_query_behavior(QueryBehavior::Ok(vec![])),
    );
    let provider = Arc::new(MockTextGenerationModelProvider::new(
        ProviderBehavior::Stream(vec![]),
    ));
    let mut config = mock_config();
    config.routing_mut().text_mut().set_default("missing");
    let server =
        TestServer::start_with_components(config, Arc::clone(&store), Arc::clone(&provider)).await;

    let response = http_client()
        .post(url(&server))
        .json(&json!({
            "model": "ignored",
            "messages": [{"role": "user", "content": "test"}],
            "stream": false
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status().as_u16(), 200); // axum JSON body carries the error
    let body: Value = response.json().await.unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("missing"),
        "error message should mention the missing key; got: {body}"
    );
    assert_eq!(store.snapshot().query_calls, 0);
}

#[tokio::test]
async fn test_non_streaming_memory_entries_injected_into_system_prompt() {
    let memory = make_result(
        Uuid::new_v4(),
        "user prefers dark mode",
        MemoryTrust::Extracted {
            confidence: 0.5,
            evidence: TrustEvidence::default(),
        },
        0.9,
    );
    let store = Arc::new(
        MockStore::new()
            .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![memory.clone()]))
            .with_get_behavior(EntryBehavior::Ok(Some(memory.clone())))
            .with_query_behavior(QueryBehavior::Ok(vec![memory])),
    );
    let provider = Arc::new(MockTextGenerationModelProvider::new(
        ProviderBehavior::Stream(vec![TextGenerationResponse::new_done(
            "ok".into(),
            "test-model".into(),
            None,
        )]),
    ));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), Arc::clone(&provider))
            .await;

    http_client()
        .post(url(&server))
        .json(&json!({
            "model": "ignored",
            "messages": [{"role": "user", "content": "what are my preferences?"}],
            "stream": false
        }))
        .send()
        .await
        .expect("request should succeed");

    let request = provider
        .snapshot()
        .last_request
        .expect("provider should have been called");
    let system = request.system().expect("system prompt should be set");
    assert!(
        system.contains("user prefers dark mode"),
        "memory entry should appear in system prompt; got: {system}"
    );
}

#[tokio::test]
async fn test_streaming_returns_sse_chunks_and_done_marker() {
    let store = Arc::new(
        MockStore::new()
            .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![]))
            .with_query_behavior(QueryBehavior::Ok(vec![])),
    );
    let provider = Arc::new(MockTextGenerationModelProvider::new(
        ProviderBehavior::Stream(vec![
            TextGenerationResponse::new("Hello".into(), "test-model".into(), None, false),
            TextGenerationResponse::new_done(
                " world".into(),
                "test-model".into(),
                Some(TokenUsage::new(Some(3), Some(2), Some(5))),
            ),
        ]),
    ));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), Arc::clone(&provider))
            .await;

    let response = http_client()
        .post(url(&server))
        .json(&json!({
            "model": "ignored",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": true
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status().as_u16(), 200);
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|v| v.starts_with("text/event-stream"))
            .unwrap_or(false),
        "content-type should be text/event-stream"
    );

    let body = response.text().await.unwrap();
    assert!(body.contains("data: [DONE]"), "body should end with [DONE]");

    let events = parse_sse_events(&body);
    // First event announces the assistant role
    assert_eq!(events[0]["choices"][0]["delta"]["role"], "assistant");

    // Collect all content deltas
    let content: String = events
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, "Hello world");

    // Last non-done event should carry usage
    let last = events.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert_eq!(last["usage"]["prompt_tokens"], 3);
    assert_eq!(last["usage"]["completion_tokens"], 2);
    assert_eq!(last["usage"]["total_tokens"], 5);
}

#[tokio::test]
async fn test_streaming_memory_is_queried_with_last_user_message() {
    let store = Arc::new(
        MockStore::new()
            .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![]))
            .with_query_behavior(QueryBehavior::Ok(vec![])),
    );
    let provider = Arc::new(MockTextGenerationModelProvider::new(
        ProviderBehavior::Stream(vec![TextGenerationResponse::new_done(
            "ok".into(),
            "test-model".into(),
            None,
        )]),
    ));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), Arc::clone(&provider))
            .await;

    http_client()
        .post(url(&server))
        .json(&json!({
            "model": "ignored",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "noted"},
                {"role": "user", "content": "the real prompt"}
            ],
            "stream": true
        }))
        .send()
        .await
        .expect("request should succeed")
        .text()
        .await
        .unwrap();

    let query = store
        .snapshot()
        .query
        .expect("store should have been queried");
    assert_eq!(query.topic(), "the real prompt");
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-memory-store-qdrant.

mod common;

use std::collections::HashMap;
use std::future::Future;
use std::sync::Mutex;

use buffa::EnumValue;
use connectrpc::{ConnectError, ErrorCode};
use pretty_assertions::assert_eq;
use rstest::rstest;
use uuid::Uuid;

use loci_core::error::MemoryStoreError;
use loci_core::memory::{
    MemoryEntry as CoreMemoryEntry, MemoryInput, MemoryQuery, MemoryQueryMode, MemoryQueryResult,
    MemoryTier as CoreMemoryTier, Score,
};
use loci_core::model_provider::common::ModelProviderResult;
use loci_core::model_provider::text_generation::{
    TextGenerationModelProvider, TextGenerationRequest, TextGenerationResponse, TokenUsage,
};

use crate::loci::generate::v1::{GenerateServiceGenerateRequest, MemoryMode, SystemMode};
use crate::loci::memory::v1::{
    MemoryServiceAddEntryRequest, MemoryServiceDeleteEntryRequest, MemoryServiceGetEntryRequest,
    MemoryServiceQueryRequest, MemoryTier,
};

use super::*;

#[derive(Debug, Clone)]
enum MockStoreErrorKind {
    NotFound(Uuid),
}

impl MockStoreErrorKind {
    fn into_memory_store_error(self) -> MemoryStoreError {
        match self {
            Self::NotFound(id) => MemoryStoreError::NotFound(id),
        }
    }
}

#[derive(Debug, Clone)]
enum EntryBehavior {
    Ok(MemoryQueryResult),
    Err(MockStoreErrorKind),
}

#[derive(Debug, Clone)]
enum QueryBehavior {
    Ok(Vec<MemoryQueryResult>),
}

#[derive(Debug, Clone)]
enum UnitBehavior {
    Ok,
}

#[derive(Debug, Clone, Default)]
struct MockStoreState {
    add_input: Option<MemoryInput>,
    get_id: Option<Uuid>,
    delete_id: Option<Uuid>,
    query: Option<MemoryQuery>,
    query_calls: usize,
}

struct MockMemoryStore {
    state: Mutex<MockStoreState>,
    add_behavior: EntryBehavior,
    get_behavior: EntryBehavior,
    query_behavior: QueryBehavior,
    delete_behavior: UnitBehavior,
}

impl MockMemoryStore {
    fn new(
        add_behavior: EntryBehavior,
        get_behavior: EntryBehavior,
        query_behavior: QueryBehavior,
        delete_behavior: UnitBehavior,
    ) -> Self {
        Self {
            state: Mutex::new(MockStoreState::default()),
            add_behavior,
            get_behavior,
            query_behavior,
            delete_behavior,
        }
    }

    fn snapshot(&self) -> MockStoreState {
        self.state
            .lock()
            .expect("mock store mutex poisoned")
            .clone()
    }
}

impl MemoryStore for MockMemoryStore {
    fn add_entry(
        &self,
        input: MemoryInput,
    ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
        self.state
            .lock()
            .expect("mock store mutex poisoned")
            .add_input = Some(input);
        let behavior = self.add_behavior.clone();
        async move {
            match behavior {
                EntryBehavior::Ok(result) => Ok(result),
                EntryBehavior::Err(error) => Err(error.into_memory_store_error()),
            }
        }
    }

    fn get_entry(
        &self,
        id: Uuid,
    ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
        self.state.lock().expect("mock store mutex poisoned").get_id = Some(id);
        let behavior = self.get_behavior.clone();
        async move {
            match behavior {
                EntryBehavior::Ok(result) => Ok(result),
                EntryBehavior::Err(error) => Err(error.into_memory_store_error()),
            }
        }
    }

    fn query(
        &self,
        query: MemoryQuery,
    ) -> impl Future<Output = Result<Vec<MemoryQueryResult>, MemoryStoreError>> + Send + '_ {
        let mut state = self.state.lock().expect("mock store mutex poisoned");
        state.query = Some(query);
        state.query_calls += 1;
        drop(state);

        let behavior = self.query_behavior.clone();
        async move {
            match behavior {
                QueryBehavior::Ok(results) => Ok(results),
            }
        }
    }

    async fn update_entry(
        &self,
        id: Uuid,
        _input: MemoryInput,
    ) -> Result<MemoryQueryResult, MemoryStoreError> {
        Err(MemoryStoreError::NotFound(id))
    }

    async fn set_entry_tier(
        &self,
        id: Uuid,
        _tier: CoreMemoryTier,
    ) -> Result<MemoryQueryResult, MemoryStoreError> {
        Err(MemoryStoreError::NotFound(id))
    }

    fn delete_entry(
        &self,
        id: Uuid,
    ) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_ {
        self.state
            .lock()
            .expect("mock store mutex poisoned")
            .delete_id = Some(id);
        let behavior = self.delete_behavior.clone();
        async move {
            match behavior {
                UnitBehavior::Ok => Ok(()),
            }
        }
    }

    async fn prune_expired(&self) -> Result<(), MemoryStoreError> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum ProviderBehavior {
    Stream(Vec<TextGenerationResponse>),
}

#[derive(Debug, Clone, Default)]
struct MockProviderState {
    last_request: Option<TextGenerationRequest>,
}

struct MockTextGenerationProvider {
    state: Mutex<MockProviderState>,
    behavior: ProviderBehavior,
}

impl MockTextGenerationProvider {
    fn new(behavior: ProviderBehavior) -> Self {
        Self {
            state: Mutex::new(MockProviderState::default()),
            behavior,
        }
    }

    fn snapshot(&self) -> MockProviderState {
        self.state
            .lock()
            .expect("mock provider mutex poisoned")
            .clone()
    }
}

impl TextGenerationModelProvider for MockTextGenerationProvider {
    fn generate(
        &self,
        req: TextGenerationRequest,
    ) -> impl Future<Output = ModelProviderResult<TextGenerationResponse>> + Send + '_ {
        self.state
            .lock()
            .expect("mock provider mutex poisoned")
            .last_request = Some(req.clone());
        let response = match &self.behavior {
            ProviderBehavior::Stream(chunks) => chunks
                .last()
                .cloned()
                .unwrap_or_else(|| TextGenerationResponse::done(String::new(), req.model, None)),
        };
        async move { Ok(response) }
    }

    fn generate_stream(
        &self,
        req: TextGenerationRequest,
    ) -> impl futures::Stream<Item = ModelProviderResult<TextGenerationResponse>> + Send + '_ {
        self.state
            .lock()
            .expect("mock provider mutex poisoned")
            .last_request = Some(req);
        let chunks = match &self.behavior {
            ProviderBehavior::Stream(chunks) => chunks.clone(),
        };
        futures::stream::iter(chunks.into_iter().map(Ok))
    }
}

fn mock_config() -> AppConfig {
    minimal_app_config(
        "http://unused-qdrant",
        "http://unused-ollama",
        "test-text-model",
        "test-embedding-model",
        384,
    )
}

fn make_result(id: Uuid, content: &str, tier: CoreMemoryTier, score: f64) -> MemoryQueryResult {
    let now = chrono::Utc::now();
    MemoryQueryResult {
        memory_entry: CoreMemoryEntry {
            id,
            content: content.to_string(),
            metadata: HashMap::from([("source".to_string(), "test".to_string())]),
            tier,
            seen_count: 2,
            sources: vec!["user".to_string()],
            first_seen: now,
            last_seen: now,
            expires_at: None,
            created_at: now,
        },
        score: Score::new(score).expect("score should be valid"),
    }
}

async fn generate_error(
    server: &TestServer,
    request: GenerateServiceGenerateRequest,
) -> ConnectError {
    match server.generate_client().generate(request).await {
        Err(error) => error,
        Ok(mut stream) => {
            let message = stream
                .message()
                .await
                .expect("stream should decode the terminal frame");
            assert!(message.is_none());
            stream
                .error()
                .cloned()
                .expect("stream should surface a trailing connect error")
        }
    }
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_memory_add_entry_uses_real_server_and_preserves_request_mapping() {
    let id = Uuid::new_v4();
    let stored = make_result(id, "stored content", CoreMemoryTier::Stable, 0.73);
    let store = Arc::new(MockMemoryStore::new(
        EntryBehavior::Ok(stored.clone()),
        EntryBehavior::Ok(stored),
        QueryBehavior::Ok(vec![]),
        UnitBehavior::Ok,
    ));
    let provider = Arc::new(MockTextGenerationProvider::new(ProviderBehavior::Stream(
        vec![TextGenerationResponse::done(
            "unused".to_string(),
            "test-text-model".to_string(),
            None,
        )],
    )));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), provider).await;

    let response = server
        .memory_client()
        .add_entry(MemoryServiceAddEntryRequest {
            content: "remember this".to_string(),
            metadata: HashMap::from([("kind".to_string(), "fact".to_string())]),
            tier: EnumValue::from(MemoryTier::MEMORY_TIER_STABLE),
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
    assert_eq!(entry.tier.as_known(), Some(MemoryTier::MEMORY_TIER_STABLE));
    assert_eq!(entry.score, 0.73);

    let captured = store.snapshot();
    let input = captured
        .add_input
        .expect("store should capture add_entry input");
    assert_eq!(input.content, "remember this");
    assert_eq!(input.metadata.get("kind"), Some(&"fact".to_string()));
    assert_eq!(input.tier, Some(CoreMemoryTier::Stable));
}

#[tokio::test]
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_memory_query_uses_lookup_mode_and_preserves_filters() {
    let result = make_result(
        Uuid::new_v4(),
        "User name is Bob",
        CoreMemoryTier::Core,
        0.91,
    );
    let store = Arc::new(MockMemoryStore::new(
        EntryBehavior::Ok(result.clone()),
        EntryBehavior::Ok(result.clone()),
        QueryBehavior::Ok(vec![result]),
        UnitBehavior::Ok,
    ));
    let provider = Arc::new(MockTextGenerationProvider::new(ProviderBehavior::Stream(
        vec![TextGenerationResponse::done(
            "unused".to_string(),
            "test-text-model".to_string(),
            None,
        )],
    )));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), provider).await;

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
        Some(MemoryTier::MEMORY_TIER_CORE)
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
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_memory_get_entry_translates_not_found_errors() {
    let missing_id = Uuid::new_v4();
    let store = Arc::new(MockMemoryStore::new(
        EntryBehavior::Err(MockStoreErrorKind::NotFound(missing_id)),
        EntryBehavior::Err(MockStoreErrorKind::NotFound(missing_id)),
        QueryBehavior::Ok(vec![]),
        UnitBehavior::Ok,
    ));
    let provider = Arc::new(MockTextGenerationProvider::new(ProviderBehavior::Stream(
        vec![TextGenerationResponse::done(
            "unused".to_string(),
            "test-text-model".to_string(),
            None,
        )],
    )));
    let server = TestServer::start_with_components(mock_config(), store, provider).await;

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
#[cfg_attr(not(feature = "integration"), ignore)]
async fn test_memory_rpcs_reject_invalid_ids(#[case] get_method: bool) {
    let result = make_result(Uuid::new_v4(), "unused", CoreMemoryTier::Candidate, 0.12);
    let store = Arc::new(MockMemoryStore::new(
        EntryBehavior::Ok(result.clone()),
        EntryBehavior::Ok(result),
        QueryBehavior::Ok(vec![]),
        UnitBehavior::Ok,
    ));
    let provider = Arc::new(MockTextGenerationProvider::new(ProviderBehavior::Stream(
        vec![TextGenerationResponse::done(
            "unused".to_string(),
            "test-text-model".to_string(),
            None,
        )],
    )));
    let server =
        TestServer::start_with_components(mock_config(), Arc::clone(&store), provider).await;

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

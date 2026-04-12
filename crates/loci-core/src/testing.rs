// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

//! Shared test infrastructure for the loci workspace.
//!
//! Available only when the `testing` feature is enabled.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Mutex;

use chrono::Utc;
use futures::stream;
use uuid::Uuid;

use crate::embedding::{Embedding, TextEmbedder};
use crate::error::{EmbeddingError, MemoryStoreError};
use crate::memory::{MemoryEntry, MemoryInput, MemoryQuery, MemoryQueryResult, MemoryTier, Score};
use crate::model_provider::{
    common::ModelProviderResult,
    text_generation::{TextGenerationModelProvider, TextGenerationRequest, TextGenerationResponse},
};
use crate::store::MemoryStore;

/// Describes an error a [`MockStore`] operation should return.
#[derive(Debug, Clone)]
pub enum MockStoreErrorKind {
    NotFound(Uuid),
    Connection(String),
}

impl MockStoreErrorKind {
    pub fn into_memory_store_error(self) -> MemoryStoreError {
        match self {
            Self::NotFound(id) => MemoryStoreError::NotFound(id),
            Self::Connection(msg) => MemoryStoreError::Connection(msg),
        }
    }
}

/// Configures the outcome of a [`MockStore`] operation that returns a single entry.
#[derive(Debug, Clone)]
pub enum EntryBehavior {
    Ok(MemoryQueryResult),
    Err(MockStoreErrorKind),
}

/// Configures the outcome of a [`MockStore::query`] call.
#[derive(Debug, Clone)]
pub enum QueryBehavior {
    Ok(Vec<MemoryQueryResult>),
}

/// Configures the outcome of a [`MockStore`] operation that returns `()`.
#[derive(Debug, Clone)]
pub enum UnitBehavior {
    Ok,
    Err(MockStoreErrorKind),
}

/// Captured state from a [`MockStore`] after one or more operations.
#[derive(Debug, Clone, Default)]
pub struct MockStoreState {
    pub add_input: Option<MemoryInput>,
    pub get_id: Option<Uuid>,
    pub delete_id: Option<Uuid>,
    pub update_id: Option<Uuid>,
    pub update_input: Option<MemoryInput>,
    pub query: Option<MemoryQuery>,
    pub query_calls: usize,
}

/// A configurable in-memory store for tests.
///
/// Each operation returns a preset result and captures its input so tests can
/// assert on it via [`MockStore::snapshot`].
///
/// # Construction
///
/// Use [`MockStore::new`] for a mock with sensible defaults, then override
/// individual operations with the builder methods:
///
/// ```ignore
/// // Convenience: set return values directly
/// let store = MockStore::new().with_add(entry).with_query(vec![result]);
///
/// // Full control: set behaviors including error cases
/// let store = MockStore::new()
///     .with_get_behavior(EntryBehavior::Err(MockStoreErrorKind::NotFound(id)));
/// ```
pub struct MockStore {
    state: Mutex<MockStoreState>,
    add_behavior: EntryBehavior,
    get_behavior: EntryBehavior,
    query_behavior: QueryBehavior,
    update_behavior: EntryBehavior,
    delete_behavior: UnitBehavior,
    prune_behavior: UnitBehavior,
}

impl MockStore {
    /// Creates a new mock store with default behaviors.
    ///
    /// Defaults:
    /// - `add_entry` → `Err(Connection("mock: not configured"))`
    /// - `get_entry` → `Err(NotFound(Uuid::nil()))`
    /// - `query` → `Ok(vec![])`
    /// - `update_entry` → `Err(NotFound(Uuid::nil()))`
    /// - `delete_entry` → `Ok(())`
    /// - `prune_expired` → `Ok(())`
    pub fn new() -> Self {
        Self {
            state: Mutex::new(MockStoreState::default()),
            add_behavior: EntryBehavior::Err(MockStoreErrorKind::Connection(
                "mock: not configured".into(),
            )),
            get_behavior: EntryBehavior::Err(MockStoreErrorKind::NotFound(Uuid::nil())),
            query_behavior: QueryBehavior::Ok(vec![]),
            update_behavior: EntryBehavior::Err(MockStoreErrorKind::NotFound(Uuid::nil())),
            delete_behavior: UnitBehavior::Ok,
            prune_behavior: UnitBehavior::Ok,
        }
    }

    /// Returns a clone of all captured state.
    pub fn snapshot(&self) -> MockStoreState {
        self.state
            .lock()
            .expect("mock store mutex poisoned")
            .clone()
    }

    // -- Convenience builders (set Ok result directly) ----------------------

    /// Configures `add_entry` to return the given result.
    pub fn with_add(mut self, result: MemoryQueryResult) -> Self {
        self.add_behavior = EntryBehavior::Ok(result);
        self
    }

    /// Configures `get_entry` to return the given result.
    pub fn with_get(mut self, result: MemoryQueryResult) -> Self {
        self.get_behavior = EntryBehavior::Ok(result);
        self
    }

    /// Configures `query` to return the given results.
    pub fn with_query(mut self, results: Vec<MemoryQueryResult>) -> Self {
        self.query_behavior = QueryBehavior::Ok(results);
        self
    }

    /// Configures `update_entry` to return the given result.
    pub fn with_update(mut self, result: MemoryQueryResult) -> Self {
        self.update_behavior = EntryBehavior::Ok(result);
        self
    }

    // -- Behavior builders (full control) -----------------------------------

    /// Configures the behavior of `add_entry`.
    pub fn with_add_behavior(mut self, behavior: EntryBehavior) -> Self {
        self.add_behavior = behavior;
        self
    }

    /// Configures the behavior of `get_entry`.
    pub fn with_get_behavior(mut self, behavior: EntryBehavior) -> Self {
        self.get_behavior = behavior;
        self
    }

    /// Configures the behavior of `query`.
    pub fn with_query_behavior(mut self, behavior: QueryBehavior) -> Self {
        self.query_behavior = behavior;
        self
    }

    /// Configures the behavior of `update_entry`.
    pub fn with_update_behavior(mut self, behavior: EntryBehavior) -> Self {
        self.update_behavior = behavior;
        self
    }

    /// Configures the behavior of `delete_entry`.
    pub fn with_delete_behavior(mut self, behavior: UnitBehavior) -> Self {
        self.delete_behavior = behavior;
        self
    }

    /// Configures the behavior of `prune_expired`.
    pub fn with_prune_behavior(mut self, behavior: UnitBehavior) -> Self {
        self.prune_behavior = behavior;
        self
    }
}

impl MemoryStore for MockStore {
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

    fn update_entry(
        &self,
        id: Uuid,
        input: MemoryInput,
    ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
        let mut state = self.state.lock().expect("mock store mutex poisoned");
        state.update_id = Some(id);
        state.update_input = Some(input);
        drop(state);

        let behavior = self.update_behavior.clone();
        async move {
            match behavior {
                EntryBehavior::Ok(result) => Ok(result),
                EntryBehavior::Err(error) => Err(error.into_memory_store_error()),
            }
        }
    }

    async fn set_entry_tier(
        &self,
        id: Uuid,
        _tier: MemoryTier,
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
                UnitBehavior::Err(error) => Err(error.into_memory_store_error()),
            }
        }
    }

    fn prune_expired(&self) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_ {
        let behavior = self.prune_behavior.clone();
        async move {
            match behavior {
                UnitBehavior::Ok => Ok(()),
                UnitBehavior::Err(error) => Err(error.into_memory_store_error()),
            }
        }
    }
}

/// Configures the response behaviour of [`MockTextGenerationModelProvider`].
#[derive(Debug, Clone)]
pub enum ProviderBehavior {
    Stream(Vec<TextGenerationResponse>),
}

/// Captured state from a [`MockTextGenerationModelProvider`].
#[derive(Debug, Clone, Default)]
pub struct MockProviderState {
    pub last_request: Option<TextGenerationRequest>,
}

/// A configurable mock text-generation provider for tests.
///
/// Returns preset responses and captures the last request for assertion via
/// [`MockTextGenerationModelProvider::snapshot`].
pub struct MockTextGenerationModelProvider {
    state: Mutex<MockProviderState>,
    behavior: ProviderBehavior,
}

impl MockTextGenerationModelProvider {
    /// Creates a mock that streams back the given chunks.
    pub fn new(behavior: ProviderBehavior) -> Self {
        Self {
            state: Mutex::new(MockProviderState::default()),
            behavior,
        }
    }

    /// Creates a mock that streams back a single `"ok"` done-chunk.
    pub fn ok() -> Self {
        Self::new(ProviderBehavior::Stream(vec![
            TextGenerationResponse::done("ok".to_string(), "mock".to_string(), None),
        ]))
    }

    /// Creates a mock that streams back the given text chunks.
    ///
    /// The last chunk is marked `done: true`; preceding chunks have `done: false`.
    pub fn with_chunks(chunks: Vec<impl Into<String>>) -> Self {
        let n = chunks.len();
        let responses: Vec<TextGenerationResponse> = chunks
            .into_iter()
            .enumerate()
            .map(|(i, text)| TextGenerationResponse {
                text: text.into(),
                model: "mock".to_string(),
                usage: None,
                done: i + 1 == n,
            })
            .collect();
        Self::new(ProviderBehavior::Stream(responses))
    }

    /// Returns a clone of all captured state.
    pub fn snapshot(&self) -> MockProviderState {
        self.state
            .lock()
            .expect("mock provider mutex poisoned")
            .clone()
    }
}

impl TextGenerationModelProvider for MockTextGenerationModelProvider {
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
        stream::iter(chunks.into_iter().map(Ok))
    }
}

/// Maps specific strings to predetermined vectors for deterministic cosine
/// similarity. Unmapped text returns the zero vector.
pub struct MockTextEmbedder {
    mappings: HashMap<String, Vec<f32>>,
    dim: usize,
}

impl MockTextEmbedder {
    pub fn new(dim: usize) -> Self {
        Self {
            mappings: HashMap::new(),
            dim,
        }
    }

    pub fn with(mut self, text: &str, values: Vec<f32>) -> Self {
        self.mappings.insert(text.to_string(), values);
        self
    }
}

impl TextEmbedder for MockTextEmbedder {
    fn embedding_dimension(&self) -> usize {
        self.dim
    }

    fn embed(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<Embedding, EmbeddingError>> + Send + '_ {
        let values = self
            .mappings
            .get(text)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.dim]);
        async move { Ok(Embedding::new(values)) }
    }
}

/// Builds a [`MemoryQueryResult`] with sensible defaults for common test
/// scenarios.
pub fn make_result(id: Uuid, content: &str, tier: MemoryTier, score: f64) -> MemoryQueryResult {
    let now = Utc::now();
    MemoryQueryResult {
        memory_entry: MemoryEntry {
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

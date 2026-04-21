// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

//! Shared test infrastructure for the loci workspace.
//!
//! Available only when the `testing` feature is enabled.

use std::collections::HashMap;
use std::sync::Mutex;

use futures::future::BoxFuture;
use futures::stream;
use uuid::Uuid;

use crate::classification::{ClassificationError, ClassificationModelProvider, HitClass};
use crate::embedding::{Embedding, TextEmbedder};
use crate::error::{EmbeddingError, MemoryStoreError};
use crate::memory::store::{AddEntriesResult, MemoryInput, MemoryQuery, MemoryStore};
use crate::memory::{MemoryEntry, MemoryTrust, TrustEvidence};
use crate::model_provider::{
    common::ModelProviderResult,
    text_generation::{TextGenerationModelProvider, TextGenerationRequest, TextGenerationResponse},
};

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

/// Configures the outcome of a [`MockStore`] operation that returns an optional entry.
#[derive(Debug, Clone)]
pub enum EntryBehavior {
    Ok(Option<MemoryEntry>),
    Err(MockStoreErrorKind),
}

/// Configures the outcome of a [`MockStore::add_entries`] call.
#[derive(Debug, Clone)]
pub enum AddEntriesBehavior {
    Ok(Vec<MemoryEntry>),
    /// Indicates a global operation-level failure (e.g., connection error).
    Err(MockStoreErrorKind),
}

/// Configures the outcome of a [`MockStore::query`] call.
#[derive(Debug, Clone)]
pub enum QueryBehavior {
    Ok(Vec<MemoryEntry>),
    Err(MockStoreErrorKind),
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
    pub add_inputs: Option<Vec<MemoryInput>>,
    pub get_id: Option<Uuid>,
    pub delete_id: Option<Uuid>,
    pub promote_id: Option<Uuid>,
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
    add_entries_behavior: AddEntriesBehavior,
    get_behavior: EntryBehavior,
    query_behavior: QueryBehavior,
    promote_behavior: EntryBehavior,
    delete_behavior: UnitBehavior,
    prune_behavior: UnitBehavior,
}

impl Default for MockStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MockStore {
    /// Creates a new mock store with default behaviors.
    ///
    /// Defaults:
    /// - `add_entries` → `Err(Connection("mock: not configured"))`
    /// - `get_entry` → `Ok(None)`
    /// - `query` → `Ok(vec![])`
    /// - `promote` → `Ok(None)`
    /// - `delete_entry` → `Ok(())`
    /// - `prune_expired` → `Ok(())`
    pub fn new() -> Self {
        Self {
            state: Mutex::new(MockStoreState::default()),
            add_entries_behavior: AddEntriesBehavior::Err(MockStoreErrorKind::Connection(
                "mock: not configured".into(),
            )),
            get_behavior: EntryBehavior::Ok(None),
            query_behavior: QueryBehavior::Ok(vec![]),
            promote_behavior: EntryBehavior::Ok(None),
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

    /// Configures `add_entries` to return the given entry.
    pub fn with_add(mut self, result: MemoryEntry) -> Self {
        self.add_entries_behavior = AddEntriesBehavior::Ok(vec![result]);
        self
    }

    /// Configures `get_entry` to return the given entry.
    pub fn with_get(mut self, result: Option<MemoryEntry>) -> Self {
        self.get_behavior = EntryBehavior::Ok(result);
        self
    }

    /// Configures `query` to return the given results.
    pub fn with_query(mut self, results: Vec<MemoryEntry>) -> Self {
        self.query_behavior = QueryBehavior::Ok(results);
        self
    }

    // -- Behavior builders (full control) -----------------------------------

    /// Configures the behavior of `add_entries`.
    pub fn with_add_entries_behavior(mut self, behavior: AddEntriesBehavior) -> Self {
        self.add_entries_behavior = behavior;
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

    /// Configures the behavior of `promote`.
    pub fn with_promote_behavior(mut self, behavior: EntryBehavior) -> Self {
        self.promote_behavior = behavior;
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
    fn add_entries<'a>(
        &'a self,
        inputs: &'a [MemoryInput],
    ) -> BoxFuture<'a, Result<AddEntriesResult, MemoryStoreError>> {
        self.state
            .lock()
            .expect("mock store mutex poisoned")
            .add_inputs = Some(inputs.to_vec());
        let behavior = self.add_entries_behavior.clone();
        Box::pin(async move {
            match behavior {
                AddEntriesBehavior::Ok(entries) => Ok(AddEntriesResult::new(entries, vec![])),
                AddEntriesBehavior::Err(error) => Err(error.into_memory_store_error()),
            }
        })
    }

    fn get_entry<'a>(
        &'a self,
        id: &'a Uuid,
    ) -> BoxFuture<'a, Result<Option<MemoryEntry>, MemoryStoreError>> {
        self.state.lock().expect("mock store mutex poisoned").get_id = Some(*id);
        let behavior = self.get_behavior.clone();
        Box::pin(async move {
            match behavior {
                EntryBehavior::Ok(result) => Ok(result),
                EntryBehavior::Err(error) => Err(error.into_memory_store_error()),
            }
        })
    }

    fn query(
        &self,
        query: MemoryQuery,
    ) -> BoxFuture<'_, Result<Vec<MemoryEntry>, MemoryStoreError>> {
        let mut state = self.state.lock().expect("mock store mutex poisoned");
        state.query = Some(query);
        state.query_calls += 1;
        drop(state);

        let behavior = self.query_behavior.clone();
        Box::pin(async move {
            match behavior {
                QueryBehavior::Ok(results) => Ok(results),
                QueryBehavior::Err(error) => Err(error.into_memory_store_error()),
            }
        })
    }

    fn promote<'a>(
        &'a self,
        id: &'a Uuid,
    ) -> BoxFuture<'a, Result<Option<MemoryEntry>, MemoryStoreError>> {
        self.state
            .lock()
            .expect("mock store mutex poisoned")
            .promote_id = Some(*id);
        let behavior = self.promote_behavior.clone();
        Box::pin(async move {
            match behavior {
                EntryBehavior::Ok(result) => Ok(result),
                EntryBehavior::Err(error) => Err(error.into_memory_store_error()),
            }
        })
    }

    fn delete_entry<'a>(&'a self, id: &'a Uuid) -> BoxFuture<'a, Result<(), MemoryStoreError>> {
        self.state
            .lock()
            .expect("mock store mutex poisoned")
            .delete_id = Some(*id);
        let behavior = self.delete_behavior.clone();
        Box::pin(async move {
            match behavior {
                UnitBehavior::Ok => Ok(()),
                UnitBehavior::Err(error) => Err(error.into_memory_store_error()),
            }
        })
    }

    fn prune_expired(&self) -> BoxFuture<'_, Result<(), MemoryStoreError>> {
        let behavior = self.prune_behavior.clone();
        Box::pin(async move {
            match behavior {
                UnitBehavior::Ok => Ok(()),
                UnitBehavior::Err(error) => Err(error.into_memory_store_error()),
            }
        })
    }
}

/// Configures the response behaviour of [`MockTextGenerationModelProvider`].
#[derive(Debug, Clone)]
pub enum ProviderBehavior {
    /// The same stream of responses is returned on every call.
    Stream(Vec<TextGenerationResponse>),
    /// A distinct stream is served per call. Calls past the end reuse the last.
    Sequence(Vec<Vec<TextGenerationResponse>>),
}

/// Captured state from a [`MockTextGenerationModelProvider`].
#[derive(Debug, Clone, Default)]
pub struct MockProviderState {
    pub last_request: Option<TextGenerationRequest>,
    pub request_count: usize,
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
            TextGenerationResponse::new_done("ok".to_string(), "mock".to_string(), None),
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
            .map(|(i, text)| {
                TextGenerationResponse::new(text.into(), "mock".to_string(), None, i + 1 == n)
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
        let mut state = self.state.lock().expect("mock provider mutex poisoned");
        state.last_request = Some(req.clone());
        let call_index = state.request_count;
        state.request_count += 1;
        drop(state);
        let chunks = match &self.behavior {
            ProviderBehavior::Stream(chunks) => chunks.clone(),
            ProviderBehavior::Sequence(rounds) => rounds
                .get(call_index)
                .or_else(|| rounds.last())
                .cloned()
                .unwrap_or_default(),
        };
        let response = chunks.last().cloned().unwrap_or_else(|| {
            TextGenerationResponse::new_done(String::new(), req.model().to_owned(), None)
        });
        async move { Ok(response) }
    }

    fn generate_stream(
        &self,
        req: TextGenerationRequest,
    ) -> impl futures::Stream<Item = ModelProviderResult<TextGenerationResponse>> + Send + '_ {
        let mut state = self.state.lock().expect("mock provider mutex poisoned");
        state.last_request = Some(req);
        let call_index = state.request_count;
        state.request_count += 1;
        drop(state);
        let chunks = match &self.behavior {
            ProviderBehavior::Stream(chunks) => chunks.clone(),
            ProviderBehavior::Sequence(rounds) => rounds
                .get(call_index)
                .or_else(|| rounds.last())
                .cloned()
                .unwrap_or_default(),
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

    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Embedding, EmbeddingError>> {
        let values = self
            .mappings
            .get(text)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.dim]);
        Box::pin(async move { Ok(Embedding::new(values)) })
    }
}

/// Builds a [`MemoryEntry`] with the given trust level for use in tests.
pub fn make_result(id: Uuid, content: &str, trust: MemoryTrust, _score: f64) -> MemoryEntry {
    MemoryEntry::new_for_testing(id, content.to_string(), HashMap::new(), trust)
}

/// Convenience wrapper: builds an `Extracted` [`MemoryEntry`] where
/// `confidence` is used as the trust confidence score.
pub fn make_extracted_result(id: Uuid, content: &str, score: f64) -> MemoryEntry {
    make_result(
        id,
        content,
        MemoryTrust::Extracted {
            confidence: score,
            evidence: TrustEvidence::default(),
        },
        score,
    )
}

/// Convenience wrapper: builds a `Fact` [`MemoryEntry`].
pub fn make_fact_result(id: Uuid, content: &str, score: f64) -> MemoryEntry {
    make_result(id, content, MemoryTrust::Fact, score)
}

/// Configures the outcome of [`MockClassificationModelProvider::classify_hit`].
#[derive(Debug, Clone)]
pub enum ClassifyBehavior {
    Ok(HitClass),
    Err(String),
}

/// Captured state from a [`MockClassificationModelProvider`].
#[derive(Debug, Clone, Default)]
pub struct MockClassificationState {
    pub calls: Vec<(String, String)>,
}

/// A configurable mock classification provider for tests.
///
/// Returns a preset [`HitClass`] or error and captures every `(candidate, hit)`
/// pair for assertion via [`MockClassificationModelProvider::snapshot`].
///
/// # Construction
///
/// ```ignore
/// let provider = MockClassificationModelProvider::new()
///     .with_behavior(ClassifyBehavior::Ok(HitClass::Duplicate));
/// ```
pub struct MockClassificationModelProvider {
    state: Mutex<MockClassificationState>,
    behavior: ClassifyBehavior,
}

impl Default for MockClassificationModelProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MockClassificationModelProvider {
    /// Creates a mock that returns [`HitClass::Unrelated`] by default.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(MockClassificationState::default()),
            behavior: ClassifyBehavior::Ok(HitClass::Unrelated),
        }
    }

    /// Overrides the classification outcome.
    pub fn with_behavior(mut self, behavior: ClassifyBehavior) -> Self {
        self.behavior = behavior;
        self
    }

    /// Returns a clone of all captured state.
    pub fn snapshot(&self) -> MockClassificationState {
        self.state
            .lock()
            .expect("mock classification mutex poisoned")
            .clone()
    }
}

impl ClassificationModelProvider for MockClassificationModelProvider {
    fn classify_hit<'a>(
        &'a self,
        candidate: &'a str,
        hit: &'a str,
    ) -> BoxFuture<'a, Result<HitClass, ClassificationError>> {
        self.state
            .lock()
            .expect("mock classification mutex poisoned")
            .calls
            .push((candidate.to_string(), hit.to_string()));
        let behavior = self.behavior.clone();
        Box::pin(async move {
            match behavior {
                ClassifyBehavior::Ok(class) => Ok(class),
                ClassifyBehavior::Err(msg) => Err(ClassificationError::Parse(msg)),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::*;

    #[tokio::test]
    async fn test_mock_classification_default_returns_unrelated() {
        let provider = MockClassificationModelProvider::new();
        let result = provider.classify_hit("candidate", "hit").await.unwrap();
        assert_eq!(result, HitClass::Unrelated);
    }

    #[rstest]
    #[case(ClassifyBehavior::Ok(HitClass::Duplicate), Ok(HitClass::Duplicate))]
    #[case(
        ClassifyBehavior::Ok(HitClass::Complementary),
        Ok(HitClass::Complementary)
    )]
    #[case(
        ClassifyBehavior::Ok(HitClass::Contradiction),
        Ok(HitClass::Contradiction)
    )]
    #[case(ClassifyBehavior::Ok(HitClass::Unrelated), Ok(HitClass::Unrelated))]
    #[tokio::test]
    async fn test_mock_classification_ok_behavior(
        #[case] behavior: ClassifyBehavior,
        #[case] expected: Result<HitClass, ()>,
    ) {
        let provider = MockClassificationModelProvider::new().with_behavior(behavior);
        let result = provider.classify_hit("c", "h").await.unwrap();
        assert_eq!(result, expected.unwrap());
    }

    #[tokio::test]
    async fn test_mock_classification_err_behavior_returns_parse_error() {
        let provider = MockClassificationModelProvider::new()
            .with_behavior(ClassifyBehavior::Err("bad".into()));
        let err = provider.classify_hit("c", "h").await.unwrap_err();
        assert!(matches!(err, ClassificationError::Parse(ref msg) if msg == "bad"));
    }

    #[tokio::test]
    async fn test_mock_classification_snapshot_captures_calls() {
        let provider = MockClassificationModelProvider::new();
        provider.classify_hit("alpha", "beta").await.unwrap();
        provider.classify_hit("gamma", "delta").await.unwrap();
        let state = provider.snapshot();
        assert_eq!(
            state.calls,
            vec![
                ("alpha".to_string(), "beta".to_string()),
                ("gamma".to_string(), "delta".to_string()),
            ]
        );
    }
}

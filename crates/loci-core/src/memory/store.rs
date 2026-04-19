// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::{collections::HashMap, num::NonZeroUsize};

use futures::future::BoxFuture;
use uuid::Uuid;

use crate::{
    error::MemoryStoreError,
    memory::{MemoryEntry, MemoryTrust, Score},
};

type MaxResultCount = NonZeroUsize;

/// Input for adding a new memory entry. Model providers decide how to interpret the content (e.g. vector embedding, keyword indexing, etc.).
#[derive(Clone, Debug)]
pub struct MemoryInput {
    /// The main content of the memory entry.
    content: String,
    /// Trust level for this entry.
    trust: MemoryTrust,
    /// Arbitrary key/value metadata pairs associated with this entry.
    metadata: HashMap<String, String>,
}

impl MemoryInput {
    /// Creates a new `MemoryInput` with the given content, trust level, and metadata.
    pub fn new(content: String, trust: MemoryTrust, metadata: HashMap<String, String>) -> Self {
        Self {
            content,
            trust,
            metadata,
        }
    }

    /// Returns a reference to the content of this memory input.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Returns the trust level of this memory input.
    pub fn trust(&self) -> &MemoryTrust {
        &self.trust
    }

    /// Returns a reference to the metadata of this memory input.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
}

/// Query behavior mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryQueryMode {
    /// Retrieval-only lookup. Does not affect lifecycle counters.
    Lookup,
    /// Retrieval used for prompt-context memory. Updates usage counters.
    Use,
}

/// Input to [`crate::MemoryStore::query`]. Model providers decide how to
/// interpret the topic (vector similarity, keyword search, etc.).
#[derive(Clone, Debug)]
pub struct MemoryQuery {
    /// The query topic. Interpretation is up to the model provider (e.g. vector similarity, keyword search, etc.).
    topic: String,
    /// Query behavior mode.
    mode: MemoryQueryMode,
    /// Maximum number of results to return.
    max_results: MaxResultCount,
    /// Minimum final score a result must reach to be included. In [0.0, 1.0].
    min_score: Score,
    /// Only return entries whose metadata contains all of these key/value pairs.
    filters: HashMap<String, String>,
}

impl MemoryQuery {
    /// Creates a new `MemoryQuery` with the specified topic and mode, and default values for other fields.
    pub fn new(topic: String, mode: MemoryQueryMode) -> Self {
        Self {
            topic,
            mode,
            max_results: NonZeroUsize::new(10).unwrap(),
            min_score: Score::ZERO,
            filters: HashMap::new(),
        }
    }

    /// Returns a reference to the topic of this memory query.
    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// Returns the query behavior mode of this memory query.
    pub fn mode(&self) -> MemoryQueryMode {
        self.mode
    }

    /// Returns the maximum number of results to return for this query.
    pub fn max_results(&self) -> MaxResultCount {
        self.max_results
    }

    /// Returns the minimum score threshold for results to be included in this query.
    pub fn min_score(&self) -> Score {
        self.min_score
    }

    /// Returns a reference to the metadata filters of this memory query.
    pub fn filters(&self) -> &HashMap<String, String> {
        &self.filters
    }

    pub fn with_max_results(mut self, max_results: MaxResultCount) -> Self {
        self.max_results = max_results;
        self
    }

    pub fn with_min_score(mut self, min_score: Score) -> Self {
        self.min_score = min_score;
        self
    }

    pub fn with_filters(mut self, filters: HashMap<String, String>) -> Self {
        self.filters = filters;
        self
    }
}

/// Per-item failure information for `add_entries`.
#[derive(Debug)]
pub struct PerEntryFailure {
    /// Index into the original inputs vector.
    index: usize,
    /// The error that occurred while adding this entry.
    error: MemoryStoreError,
}

impl PerEntryFailure {
    /// Creates a new `PerEntryFailure` with the specified index and error.
    pub fn new(index: usize, error: MemoryStoreError) -> Self {
        Self { index, error }
    }

    /// Returns the index of the failed entry in the original input vector.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns a reference to the error that occurred while adding this entry.
    pub fn error(&self) -> &MemoryStoreError {
        &self.error
    }
}

/// Result of adding multiple memory entries, including both successful and failed entries.
#[derive(Debug)]
pub struct AddEntriesResult {
    /// Successfully added entries.
    added: Vec<MemoryEntry>,
    /// Failures with indexes pointing into the original input slice.
    failures: Vec<PerEntryFailure>,
}

impl AddEntriesResult {
    /// Creates a new `AddEntriesResult` with the specified added entries and failures.
    pub fn new(added: Vec<MemoryEntry>, failures: Vec<PerEntryFailure>) -> Self {
        Self { added, failures }
    }

    /// Returns a slice of the successfully added memory entries.
    pub fn added(&self) -> &[MemoryEntry] {
        &self.added
    }

    /// Returns a slice of per-entry failures that occurred while adding entries.
    pub fn failures(&self) -> &[PerEntryFailure] {
        &self.failures
    }
}

/// Persistent storage and semantic retrieval of [`crate::Memory`] entries.
pub trait MemoryStore: Send + Sync {
    /// Saves a new memory entry and returns the saved entry with assigned ID and timestamps.
    fn add_entry<'a>(
        &'a self,
        input: &'a MemoryInput,
    ) -> BoxFuture<'a, Result<MemoryEntry, MemoryStoreError>> {
        Box::pin(async move {
            let result = self.add_entries(std::slice::from_ref(input)).await;
            match result {
                Err(e) => Err(e),
                Ok(add_result) => {
                    if let Some(added) = add_result.added().first().cloned() {
                        Ok(added)
                    } else {
                        let msg = add_result
                            .failures()
                            .first()
                            .map(|f| f.error().to_string())
                            .unwrap_or_default();
                        Err(MemoryStoreError::GenericSave(msg))
                    }
                }
            }
        })
    }

    /// Saves multiple new memory entries and returns the saved entries with assigned IDs and timestamps.
    fn add_entries<'a>(
        &'a self,
        inputs: &'a [MemoryInput],
    ) -> BoxFuture<'a, Result<AddEntriesResult, MemoryStoreError>>;

    /// Retrieves a memory entry by its unique ID.
    fn get_entry<'a>(
        &'a self,
        id: &'a Uuid,
    ) -> BoxFuture<'a, Result<Option<MemoryEntry>, MemoryStoreError>>;

    /// Executes a semantic query and returns matching memory entries sorted by relevance.
    fn query(
        &self,
        query: MemoryQuery,
    ) -> BoxFuture<'_, Result<Vec<MemoryEntry>, MemoryStoreError>>;

    /// Sets the trust level of an existing entry by its ID and returns the updated entry.
    fn promote<'a>(
        &'a self,
        id: &'a Uuid,
    ) -> BoxFuture<'a, Result<Option<MemoryEntry>, MemoryStoreError>>;

    /// Deletes a memory entry by its ID.
    fn delete_entry<'a>(&'a self, id: &'a Uuid) -> BoxFuture<'a, Result<(), MemoryStoreError>>;

    /// Deletes all expired memory entries based on their TTL and last access time.
    fn prune_expired(&self) -> BoxFuture<'_, Result<(), MemoryStoreError>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryTrust;

    #[test]
    fn test_memory_input_new_stores_fields() {
        let metadata = HashMap::from([("key".to_string(), "val".to_string())]);
        let input = MemoryInput::new("content".to_string(), MemoryTrust::Fact, metadata.clone());
        assert_eq!(input.content(), "content");
        assert!(matches!(input.trust(), MemoryTrust::Fact));
        assert_eq!(*input.metadata(), metadata);
    }
}

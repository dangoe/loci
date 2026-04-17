// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::future::Future;

use uuid::Uuid;

use crate::{
    error::MemoryStoreError,
    memory::{MemoryInput, MemoryKind, MemoryQuery, MemoryQueryResult},
};

/// Per-item failure information for `add_entries`.
pub struct PerEntryFailure {
    /// Index into the original inputs vector.
    pub index: usize,
    /// The error that occurred while adding this entry.
    pub error: MemoryStoreError,
}

/// Result of adding multiple memory entries, including both successful and failed entries.
pub struct AddEntriesResult {
    /// Successfully added entries.
    pub added: Vec<MemoryQueryResult>,
    /// Failures with indexes pointing into the original input slice.
    pub failures: Vec<PerEntryFailure>,
}

/// Persistent storage and semantic retrieval of [`crate::Memory`] entries.
pub trait MemoryStore: Send + Sync {
    /// Saves a new memory entry and returns the saved entry with assigned ID and timestamps.
    fn add_entry(
        &self,
        input: MemoryInput,
    ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
        async move {
            let result = self.add_entries(vec![input]).await;
            match result {
                Err(e) => Err(e),
                Ok(add_result) => {
                    if let Some(added) = add_result.added.into_iter().next() {
                        Ok(added)
                    } else {
                        let msg = add_result
                            .failures
                            .into_iter()
                            .next()
                            .map(|f| f.error.to_string())
                            .unwrap_or_default();
                        Err(MemoryStoreError::GenericSave(msg))
                    }
                }
            }
        }
    }

    /// Saves multiple new memory entries and returns the saved entries with assigned IDs and timestamps.
    fn add_entries(
        &self,
        inputs: Vec<MemoryInput>,
    ) -> impl Future<Output = Result<AddEntriesResult, MemoryStoreError>> + Send + '_;

    /// Retrieves a memory entry by its unique ID.
    fn get_entry(
        &self,
        id: Uuid,
    ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_;

    /// Executes a semantic query and returns matching memory entries sorted by relevance.
    fn query(
        &self,
        query: MemoryQuery,
    ) -> impl Future<Output = Result<Vec<MemoryQueryResult>, MemoryStoreError>> + Send + '_;

    /// Updates an existing memory entry by its ID and returns the updated entry.
    fn update_entry(
        &self,
        id: Uuid,
        input: MemoryInput,
    ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_;

    /// Sets the memory kind of an existing entry by its ID and returns the updated entry.
    fn set_entry_kind(
        &self,
        id: Uuid,
        kind: MemoryKind,
    ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_;

    /// Deletes a memory entry by its ID.
    fn delete_entry(
        &self,
        id: Uuid,
    ) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_;

    /// Deletes all expired memory entries based on their TTL and last access time.
    fn prune_expired(&self) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_;
}

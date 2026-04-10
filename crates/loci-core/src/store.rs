// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::future::Future;

use uuid::Uuid;

use crate::{
    error::MemoryStoreError,
    memory::{MemoryInput, MemoryQuery, MemoryQueryResult, MemoryTier},
};

/// Persistent storage and semantic retrieval of [`crate::Memory`] entries.
pub trait MemoryStore: Send + Sync {
    /// Saves a new memory entry and returns the saved entry with assigned ID and timestamps.
    fn save(
        &self,
        input: MemoryInput,
    ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_;

    /// Retrieves a memory entry by its unique ID.
    fn get(
        &self,
        id: Uuid,
    ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_;

    /// Executes a semantic query and returns matching memory entries sorted by relevance.
    fn query(
        &self,
        query: MemoryQuery,
    ) -> impl Future<Output = Result<Vec<MemoryQueryResult>, MemoryStoreError>> + Send + '_;

    /// Updates an existing memory entry by its ID and returns the updated entry.
    fn update(
        &self,
        id: Uuid,
        input: MemoryInput,
    ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_;

    /// Sets the memory tier of an existing entry by its ID and returns the updated entry.
    fn set_tier(
        &self,
        id: Uuid,
        tier: MemoryTier,
    ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_;

    /// Deletes a memory entry by its ID.
    fn delete(&self, id: Uuid) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_;

    /// Deletes all expired memory entries based on their TTL and last access time.
    fn prune_expired(&self) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_;
}

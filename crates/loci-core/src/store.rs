// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-core.

use std::future::Future;

use uuid::Uuid;

use crate::{
    error::MemoryStoreError,
    memory::{MemoryEntry, MemoryInput, MemoryQuery, MemoryTier},
};

/// Persistent storage and semantic retrieval of [`crate::Memory`] entries.
pub trait MemoryStore: Send + Sync {
    fn save(
        &self,
        input: MemoryInput,
    ) -> impl Future<Output = Result<MemoryEntry, MemoryStoreError>> + Send + '_;
    fn query(
        &self,
        query: MemoryQuery,
    ) -> impl Future<Output = Result<Vec<MemoryEntry>, MemoryStoreError>> + Send + '_;
    fn update(
        &self,
        id: Uuid,
        input: MemoryInput,
    ) -> impl Future<Output = Result<MemoryEntry, MemoryStoreError>> + Send + '_;
    fn set_tier(
        &self,
        id: Uuid,
        tier: MemoryTier,
    ) -> impl Future<Output = Result<MemoryEntry, MemoryStoreError>> + Send + '_;
    fn delete(&self, id: Uuid) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_;
    fn clear(&self) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_;
}

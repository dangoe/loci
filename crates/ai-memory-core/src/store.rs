use std::future::Future;

use uuid::Uuid;

use crate::{Memory, MemoryEntry, MemoryQuery, MemoryStoreError};

/// Persistent storage and semantic retrieval of [`Memory`] entries.
pub trait MemoryStore: Send + Sync {
    fn save(&self, memory: Memory) -> impl Future<Output = Result<Uuid, MemoryStoreError>> + Send + '_;
    fn query(&self, query: MemoryQuery) -> impl Future<Output = Result<Vec<MemoryEntry>, MemoryStoreError>> + Send + '_;
    fn delete(&self, id: Uuid) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_;
}

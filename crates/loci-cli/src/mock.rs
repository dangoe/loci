use std::sync::{Arc, Mutex};

use loci_core::{
    error::MemoryStoreError as CoreMemoryStoreError,
    memory::{
        MemoryInput as CoreMemoryInput, MemoryQuery as CoreMemoryQuery,
        MemoryQueryResult as CoreMemoryQueryResult, MemoryTier as CoreMemoryTier,
    },
};
use uuid::Uuid;

/// A configurable in-memory store for unit tests.
///
/// Each operation returns a preset response. Operations not configured
/// fall back to a sensible error (`NotFound` for reads, `Connection` for writes).
pub struct MockStore {
    pub save_entry: Option<CoreMemoryQueryResult>,
    pub get_entry: Option<CoreMemoryQueryResult>,
    pub query_entries: Vec<CoreMemoryQueryResult>,
    pub update_entry: Option<CoreMemoryQueryResult>,
    /// Captures the last `MemoryInput` passed to `update()` for assertion.
    pub captured_update_input: Arc<Mutex<Option<CoreMemoryInput>>>,
}

impl MockStore {
    pub fn new() -> Self {
        Self {
            save_entry: None,
            get_entry: None,
            query_entries: vec![],
            update_entry: None,
            captured_update_input: Arc::new(Mutex::new(None)),
        }
    }

    pub fn with_save(mut self, entry: CoreMemoryQueryResult) -> Self {
        self.save_entry = Some(entry);
        self
    }

    pub fn with_get(mut self, entry: CoreMemoryQueryResult) -> Self {
        self.get_entry = Some(entry);
        self
    }

    pub fn with_query(mut self, entries: Vec<CoreMemoryQueryResult>) -> Self {
        self.query_entries = entries;
        self
    }

    pub fn with_update(mut self, entry: CoreMemoryQueryResult) -> Self {
        self.update_entry = Some(entry);
        self
    }
}

impl loci_core::store::MemoryStore for MockStore {
    fn save(
        &self,
        _input: CoreMemoryInput,
    ) -> impl Future<Output = Result<CoreMemoryQueryResult, CoreMemoryStoreError>> + Send + '_ {
        let result = self
            .save_entry
            .clone()
            .ok_or_else(|| CoreMemoryStoreError::Connection("mock: save not configured".into()));
        async move { result }
    }

    fn get(
        &self,
        id: Uuid,
    ) -> impl Future<Output = Result<CoreMemoryQueryResult, CoreMemoryStoreError>> + Send + '_ {
        let result = self
            .get_entry
            .clone()
            .ok_or_else(move || CoreMemoryStoreError::NotFound(id));
        async move { result }
    }

    fn query(
        &self,
        _query: CoreMemoryQuery,
    ) -> impl Future<Output = Result<Vec<CoreMemoryQueryResult>, CoreMemoryStoreError>> + Send + '_
    {
        let entries = self.query_entries.clone();
        async move { Ok(entries) }
    }

    fn update(
        &self,
        id: Uuid,
        input: CoreMemoryInput,
    ) -> impl Future<Output = Result<CoreMemoryQueryResult, CoreMemoryStoreError>> + Send + '_ {
        *self.captured_update_input.lock().unwrap() = Some(input);
        let result = self
            .update_entry
            .clone()
            .ok_or_else(move || CoreMemoryStoreError::NotFound(id));
        async move { result }
    }

    fn set_tier(
        &self,
        id: Uuid,
        _tier: CoreMemoryTier,
    ) -> impl Future<Output = Result<CoreMemoryQueryResult, CoreMemoryStoreError>> + Send + '_ {
        async move { Err(CoreMemoryStoreError::NotFound(id)) }
    }

    fn delete(
        &self,
        _id: Uuid,
    ) -> impl Future<Output = Result<(), CoreMemoryStoreError>> + Send + '_ {
        async move { Ok(()) }
    }

    fn prune_expired(&self) -> impl Future<Output = Result<(), CoreMemoryStoreError>> + Send + '_ {
        async move { Ok(()) }
    }
}

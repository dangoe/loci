// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::sync::{Arc, Mutex};

use crate::error::MemoryStoreError as CoreMemoryStoreError;
use crate::memory::{
    MemoryInput as CoreMemoryInput, MemoryQuery as CoreMemoryQuery,
    MemoryQueryResult as CoreMemoryQueryResult, MemoryTier as CoreMemoryTier,
};
use crate::model_provider::{
    common::ModelProviderResult,
    text_generation::{TextGenerationModelProvider, TextGenerationRequest, TextGenerationResponse},
};
use crate::store::MemoryStore;
use futures::stream;
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
    /// Captures the last `MemoryQuery` passed to `query()` for assertion.
    pub captured_query: Arc<Mutex<Option<CoreMemoryQuery>>>,
    /// When `true`, `delete()` returns a `Connection` error instead of `Ok(())`.
    pub delete_error: bool,
    /// When `true`, `prune_expired()` returns a `Connection` error instead of `Ok(())`.
    pub prune_error: bool,
}

impl MockStore {
    pub fn new() -> Self {
        Self {
            save_entry: None,
            get_entry: None,
            query_entries: vec![],
            update_entry: None,
            captured_update_input: Arc::new(Mutex::new(None)),
            captured_query: Arc::new(Mutex::new(None)),
            delete_error: false,
            prune_error: false,
        }
    }

    pub fn with_add(mut self, entry: CoreMemoryQueryResult) -> Self {
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

    /// Configures `delete_entry()` to return a `Connection` error.
    pub fn with_delete_error(mut self) -> Self {
        self.delete_error = true;
        self
    }

    /// Configures `prune_expired()` to return a `Connection` error.
    pub fn with_prune_error(mut self) -> Self {
        self.prune_error = true;
        self
    }
}

impl MemoryStore for MockStore {
    fn add_entry(
        &self,
        _input: CoreMemoryInput,
    ) -> impl Future<Output = Result<CoreMemoryQueryResult, CoreMemoryStoreError>> + Send + '_ {
        let result = self
            .save_entry
            .clone()
            .ok_or_else(|| CoreMemoryStoreError::Connection("mock: save not configured".into()));
        async move { result }
    }

    fn get_entry(
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
        query: CoreMemoryQuery,
    ) -> impl Future<Output = Result<Vec<CoreMemoryQueryResult>, CoreMemoryStoreError>> + Send + '_
    {
        *self.captured_query.lock().unwrap() = Some(query);
        let entries = self.query_entries.clone();
        async move { Ok(entries) }
    }

    fn update_entry(
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

    fn set_entry_tier(
        &self,
        id: Uuid,
        _tier: CoreMemoryTier,
    ) -> impl Future<Output = Result<CoreMemoryQueryResult, CoreMemoryStoreError>> + Send + '_ {
        async move { Err(CoreMemoryStoreError::NotFound(id)) }
    }

    fn delete_entry(
        &self,
        _id: Uuid,
    ) -> impl Future<Output = Result<(), CoreMemoryStoreError>> + Send + '_ {
        let err = self.delete_error;
        async move {
            if err {
                Err(CoreMemoryStoreError::Connection(
                    "mock: delete error".into(),
                ))
            } else {
                Ok(())
            }
        }
    }

    fn prune_expired(&self) -> impl Future<Output = Result<(), CoreMemoryStoreError>> + Send + '_ {
        let err = self.prune_error;
        async move {
            if err {
                Err(CoreMemoryStoreError::Connection("mock: prune error".into()))
            } else {
                Ok(())
            }
        }
    }
}

/// A configurable mock text-generation provider for unit tests.
///
/// Streams back preset text chunks. The last chunk is marked `done: true`;
/// all preceding chunks have `done: false`. Defaults to a single `"ok"` chunk.
pub struct MockTextGenerationModelProvider {
    pub chunks: Vec<String>,
}

impl MockTextGenerationModelProvider {
    pub fn new() -> Self {
        Self {
            chunks: vec!["ok".to_string()],
        }
    }

    pub fn with_chunks(mut self, chunks: Vec<impl Into<String>>) -> Self {
        self.chunks = chunks.into_iter().map(Into::into).collect();
        self
    }
}

impl TextGenerationModelProvider for MockTextGenerationModelProvider {
    fn generate(
        &self,
        _req: TextGenerationRequest,
    ) -> impl Future<Output = ModelProviderResult<TextGenerationResponse>> + Send + '_ {
        let text = self.chunks.first().cloned().unwrap_or_default();
        async move {
            Ok(TextGenerationResponse {
                text,
                model: "mock".to_string(),
                usage: None,
                done: true,
            })
        }
    }

    fn generate_stream(
        &self,
        _req: TextGenerationRequest,
    ) -> impl futures::Stream<Item = ModelProviderResult<TextGenerationResponse>> + Send + '_ {
        let n = self.chunks.len();
        let responses: Vec<ModelProviderResult<TextGenerationResponse>> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, text)| {
                Ok(TextGenerationResponse {
                    text: text.clone(),
                    model: "mock".to_string(),
                    usage: None,
                    done: i + 1 == n,
                })
            })
            .collect();
        stream::iter(responses)
    }
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

use std::sync::Arc;

use loci_config::AppConfig;
use loci_core::memory::store::MemoryStore;
use loci_core::model_provider::text_generation::TextGenerationModelProvider;

/// Shared application state injected into every axum handler.
pub(crate) struct AppState<M, E>
where
    M: MemoryStore,
    E: TextGenerationModelProvider + 'static,
{
    store: Arc<M>,
    llm_provider: Arc<E>,
    config: Arc<AppConfig>,
}

impl<M, E> AppState<M, E>
where
    M: MemoryStore,
    E: TextGenerationModelProvider + 'static,
{
    /// Constructs a new `AppState`.
    pub(crate) fn new(store: Arc<M>, llm_provider: Arc<E>, config: Arc<AppConfig>) -> Self {
        Self {
            store,
            llm_provider,
            config,
        }
    }

    /// Returns a reference to the memory store.
    pub(crate) fn store(&self) -> &Arc<M> {
        &self.store
    }

    /// Returns a reference to the LLM provider.
    pub(crate) fn llm_provider(&self) -> &Arc<E> {
        &self.llm_provider
    }

    /// Returns a reference to the application configuration.
    pub(crate) fn config(&self) -> &AppConfig {
        &self.config
    }
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

use std::sync::Arc;

use loci_config::AppConfig;
use loci_core::embedding::DefaultTextEmbedder;
use loci_memory_store_qdrant::store::QdrantMemoryStore;
use loci_model_provider_ollama::provider::OllamaModelProvider;

/// Shared application state injected into every axum handler.
pub(crate) struct AppState {
    pub store: Arc<QdrantMemoryStore<DefaultTextEmbedder<OllamaModelProvider>>>,
    pub llm_provider: Arc<OllamaModelProvider>,
    pub config: Arc<AppConfig>,
}

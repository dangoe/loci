// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

use std::sync::Arc;

use loci_config::AppConfig;
use loci_core::model_provider::text_generation::TextGenerationModelProvider;
use loci_core::store::MemoryStore;

/// Shared application state injected into every axum handler.
pub(crate) struct AppState<M, E>
where
    M: MemoryStore,
    E: TextGenerationModelProvider + 'static,
{
    pub store: Arc<M>,
    pub llm_provider: Arc<E>,
    pub config: Arc<AppConfig>,
}

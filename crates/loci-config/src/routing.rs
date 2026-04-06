// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-config.

use serde::Deserialize;

/// Top-level routing configuration, deserialized from `[routing]`.
#[derive(Debug, Clone, Deserialize)]
pub struct RoutingConfig {
    /// Text-generation routing, under `[routing.text]`.
    pub text: TextRoutingConfig,

    /// Embedding routing, under `[routing.embedding]`.
    pub embedding: EmbeddingRoutingConfig,

    /// Memory backend routing, under `[routing.memory]`.
    pub memory: MemoryRoutingConfig,
}

/// Routing for text-generation models, deserialized from `[routing.text]`.
#[derive(Debug, Clone, Deserialize)]
pub struct TextRoutingConfig {
    /// Name of the default text-generation model in `[models.text]`.
    pub default: String,

    /// Ordered fallback chain of model names. Currently parsed but not yet
    /// used at runtime.
    #[serde(default)]
    pub fallback: Vec<String>,
}

/// Routing for embedding models, deserialized from `[routing.embedding]`.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRoutingConfig {
    /// Name of the default embedding model in `[models.embedding]`.
    pub default: String,
}

/// Routing for memory backends, deserialized from `[routing.memory]`.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryRoutingConfig {
    /// Name of the default memory backend in `[memory.backends]`.
    pub default: String,
}

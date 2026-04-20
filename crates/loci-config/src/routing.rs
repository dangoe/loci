// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use serde::Deserialize;

/// Top-level routing configuration, deserialized from `[routing]`.
#[derive(Debug, Clone, Deserialize)]
pub struct RoutingConfig {
    /// Text-generation routing, under `[routing.text]`.
    text: TextRoutingConfig,

    /// Embedding routing, under `[routing.embedding]`.
    embedding: EmbeddingRoutingConfig,

    /// Memory backend routing, under `[routing.memory]`.
    memory: MemoryRoutingConfig,
}

impl RoutingConfig {
    /// Constructs a new `RoutingConfig`.
    pub fn new(
        text: TextRoutingConfig,
        embedding: EmbeddingRoutingConfig,
        memory: MemoryRoutingConfig,
    ) -> Self {
        Self {
            text,
            embedding,
            memory,
        }
    }

    /// Returns the text-generation routing configuration.
    pub fn text(&self) -> &TextRoutingConfig {
        &self.text
    }

    /// Returns the embedding routing configuration.
    pub fn embedding(&self) -> &EmbeddingRoutingConfig {
        &self.embedding
    }

    /// Returns the memory backend routing configuration.
    pub fn memory(&self) -> &MemoryRoutingConfig {
        &self.memory
    }

    /// Returns a mutable reference to the text-generation routing configuration.
    pub fn text_mut(&mut self) -> &mut TextRoutingConfig {
        &mut self.text
    }

    /// Returns a mutable reference to the embedding routing configuration.
    pub fn embedding_mut(&mut self) -> &mut EmbeddingRoutingConfig {
        &mut self.embedding
    }
}

/// Routing for text-generation models, deserialized from `[routing.text]`.
#[derive(Debug, Clone, Deserialize)]
pub struct TextRoutingConfig {
    /// Name of the default text-generation model in `[models.text]`.
    default: String,

    /// Ordered fallback chain of model names. Currently parsed but not yet
    /// used at runtime.
    #[serde(default)]
    fallback: Vec<String>,
}

impl TextRoutingConfig {
    /// Constructs a new `TextRoutingConfig`.
    pub fn new(default: impl Into<String>, fallback: Vec<String>) -> Self {
        Self {
            default: default.into(),
            fallback,
        }
    }

    /// Returns the default text model name.
    pub fn default(&self) -> &str {
        &self.default
    }

    /// Returns the fallback model chain.
    pub fn fallback(&self) -> &[String] {
        &self.fallback
    }

    /// Sets the default text model name.
    pub fn set_default(&mut self, val: impl Into<String>) {
        self.default = val.into();
    }
}

/// Routing for embedding models, deserialized from `[routing.embedding]`.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRoutingConfig {
    /// Name of the default embedding model in `[models.embedding]`.
    default: String,
}

impl EmbeddingRoutingConfig {
    /// Constructs a new `EmbeddingRoutingConfig`.
    pub fn new(default: impl Into<String>) -> Self {
        Self {
            default: default.into(),
        }
    }

    /// Returns the default embedding model name.
    pub fn default(&self) -> &str {
        &self.default
    }

    /// Sets the default embedding model name.
    pub fn set_default(&mut self, val: impl Into<String>) {
        self.default = val.into();
    }
}

/// Routing for memory backends, deserialized from `[routing.memory]`.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryRoutingConfig {
    /// Name of the default memory backend in `[memory.backends]`.
    default: String,
}

impl MemoryRoutingConfig {
    /// Constructs a new `MemoryRoutingConfig`.
    pub fn new(default: impl Into<String>) -> Self {
        Self {
            default: default.into(),
        }
    }

    /// Returns the default memory backend name.
    pub fn default(&self) -> &str {
        &self.default
    }
}

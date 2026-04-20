// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use serde::Deserialize;

/// An embedding model config entry, nested under `[models.embedding.<name>]`.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingModelConfig {
    /// The provider name that serves the embedding model.
    provider: String,

    /// The embedding model identifier as understood by the provider.
    model: String,

    /// The output embedding dimension.
    dimension: usize,
}

impl EmbeddingModelConfig {
    /// Constructs a new `EmbeddingModelConfig`.
    pub fn new(provider: impl Into<String>, model: impl Into<String>, dimension: usize) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            dimension,
        }
    }

    /// Returns the provider name.
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// Returns the model identifier.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Sets the provider name.
    pub fn set_provider(&mut self, val: impl Into<String>) {
        self.provider = val.into();
    }
}

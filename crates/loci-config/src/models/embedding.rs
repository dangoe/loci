// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use serde::Deserialize;

use super::text::ModelTuningConfig;

/// An embedding model config entry, nested under
/// `[resources.models.embedding.<name>]`.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingModelConfig {
    /// The provider name that serves the embedding model.
    provider: String,

    /// The embedding model identifier as understood by the provider.
    model: String,

    /// The output embedding dimension.
    dimension: usize,

    /// Optional generation tuning parameters for this model.
    #[serde(default)]
    tuning: Option<ModelTuningConfig>,
}

impl EmbeddingModelConfig {
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

    /// Returns the optional tuning parameters.
    pub fn tuning(&self) -> Option<&ModelTuningConfig> {
        self.tuning.as_ref()
    }

    /// Sets the provider name.
    pub fn set_provider(&mut self, val: impl Into<String>) {
        self.provider = val.into();
    }
}

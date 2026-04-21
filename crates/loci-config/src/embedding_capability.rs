// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use serde::Deserialize;

/// Embedding capability configuration, deserialized from `[embedding]`.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingCapabilityConfig {
    /// Key in `[resources.models.embedding]` that selects the active model.
    model: String,
}

impl EmbeddingCapabilityConfig {
    /// Returns the active embedding model key.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Sets the active embedding model key.
    pub fn set_model(&mut self, val: impl Into<String>) {
        self.model = val.into();
    }
}

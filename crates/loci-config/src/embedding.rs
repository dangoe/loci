// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-config.

use serde::Deserialize;

/// An embedding model config entry, nested under `[models.embedding.<name>]`.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingModelConfig {
    /// The provider name that serves the embedding model.
    pub provider: String,

    /// The embedding model identifier as understood by the provider.
    pub model: String,

    /// The output embedding dimension.
    pub dimension: usize,
}

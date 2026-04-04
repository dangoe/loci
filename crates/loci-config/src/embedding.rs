// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-config.

use serde::Deserialize;

/// A named embedding profile referencing a provider and model.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingProfileConfig {
    /// The provider name that serves the embedding model.
    pub provider: String,

    /// The embedding model identifier as understood by the provider.
    pub model: String,

    /// The output embedding dimension.
    pub dimension: usize,
}

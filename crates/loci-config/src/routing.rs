// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-config.

use serde::Deserialize;

/// Routing and default selection configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct RoutingConfig {
    /// Name of the model entry in `[models]` to use by default for inference.
    pub default_model: String,

    /// Ordered fallback chain of model names to try when the default model
    /// fails. Currently parsed but not yet used at runtime.
    #[serde(default)]
    pub fallback_models: Vec<String>,

    /// Name of the embedding profile entry in `[embeddings]` to use.
    pub embedding: String,
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-config.

use serde::Deserialize;

/// A named model alias referencing a provider.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// The provider name this model is served by.
    pub provider: String,

    /// The model identifier as understood by the provider (e.g. `"gpt-4.1"`).
    pub name: String,
}

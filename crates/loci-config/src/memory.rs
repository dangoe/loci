// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-config.

use serde::Deserialize;

/// Memory persistence configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryConfig {
    /// Name of the store entry in `[stores]` to use.
    pub store: String,

    /// Collection / namespace within the store.
    pub collection: String,

    /// Optional cosine similarity threshold for deduplication (0.0–1.0).
    /// When set, a new memory is not saved if an existing one already reaches
    /// this similarity score.
    pub similarity_threshold: Option<f64>,
}

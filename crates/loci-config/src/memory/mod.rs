// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use serde::Deserialize;

use self::extraction::MemoryExtractionConfig;

pub mod extraction;
pub mod store;

/// Top-level memory section, deserialized from `[memory]`.
#[derive(Debug, Clone, Deserialize)]
pub struct MemorySection {
    /// Key in `[resources.memory_stores]` that selects the active store.
    store: String,

    /// Optional cosine similarity threshold for deduplication (0.0–1.0).
    #[serde(default)]
    similarity_threshold: Option<f64>,

    /// LLM-based memory extraction configuration, under `[memory.extraction]`.
    extraction: MemoryExtractionConfig,
}

impl MemorySection {
    /// Returns the active memory store key.
    pub fn store(&self) -> &str {
        &self.store
    }

    /// Returns the optional similarity threshold.
    pub fn similarity_threshold(&self) -> Option<f64> {
        self.similarity_threshold
    }

    /// Returns the extraction configuration.
    pub fn extraction(&self) -> &MemoryExtractionConfig {
        &self.extraction
    }
}

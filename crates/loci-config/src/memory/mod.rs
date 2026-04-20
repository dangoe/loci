// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use std::collections::HashMap;

use serde::Deserialize;

use self::extraction::MemoryExtractionConfig;
use self::store::StoreConfig;

pub mod extraction;
pub mod store;

/// Top-level memory section, deserialized from `[memory]`.
#[derive(Debug, Clone, Deserialize)]
pub struct MemorySection {
    /// Named memory backend definitions, each under `[memory.backends.<name>]`.
    #[serde(default)]
    pub backends: HashMap<String, StoreConfig>,

    /// Active backend selection and tuning, under `[memory.config]`.
    pub config: MemoryConfig,

    /// LLM-based memory extraction configuration, under `[memory.extraction]`.
    pub extraction: MemoryExtractionConfig,
}

/// Memory persistence configuration, deserialized from `[memory.config]`.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryConfig {
    /// Name of the backend entry in `[memory.backends]` to use.
    pub backend: String,

    /// Optional cosine similarity threshold for deduplication (0.0–1.0).
    /// When set, a new memory is not saved if an existing one already reaches
    /// this similarity score.
    pub similarity_threshold: Option<f64>,
}

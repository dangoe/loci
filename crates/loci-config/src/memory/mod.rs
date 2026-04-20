// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use std::collections::HashMap;

use serde::Deserialize;

use self::extraction::MemoryExtractionConfig;
use self::store::StoreConfig;
use crate::ConfigError;

pub mod extraction;
pub mod store;

/// Top-level memory section, deserialized from `[memory]`.
#[derive(Debug, Clone, Deserialize)]
pub struct MemorySection {
    /// Named memory backend definitions, each under `[memory.backends.<name>]`.
    #[serde(default)]
    backends: HashMap<String, StoreConfig>,

    /// Active backend selection and tuning, under `[memory.config]`.
    config: MemoryConfig,

    /// LLM-based memory extraction configuration, under `[memory.extraction]`.
    extraction: MemoryExtractionConfig,
}

impl MemorySection {
    /// Constructs a new `MemorySection`.
    pub fn new(
        backends: HashMap<String, StoreConfig>,
        config: MemoryConfig,
        extraction: MemoryExtractionConfig,
    ) -> Self {
        Self {
            backends,
            config,
            extraction,
        }
    }

    /// Returns the named backend definitions.
    pub fn backends(&self) -> &HashMap<String, StoreConfig> {
        &self.backends
    }

    /// Returns the active backend configuration.
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Returns the extraction configuration.
    pub fn extraction(&self) -> &MemoryExtractionConfig {
        &self.extraction
    }

    /// Resolves `env:VAR` references in all backend API keys.
    pub(crate) fn resolve_secrets(&mut self) -> Result<(), ConfigError> {
        for backend in self.backends.values_mut() {
            backend.resolve_api_key()?;
        }
        Ok(())
    }
}

/// Memory persistence configuration, deserialized from `[memory.config]`.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryConfig {
    /// Name of the backend entry in `[memory.backends]` to use.
    backend: String,

    /// Optional cosine similarity threshold for deduplication (0.0–1.0).
    /// When set, a new memory is not saved if an existing one already reaches
    /// this similarity score.
    similarity_threshold: Option<f64>,
}

impl MemoryConfig {
    /// Constructs a new `MemoryConfig`.
    pub fn new(backend: impl Into<String>, similarity_threshold: Option<f64>) -> Self {
        Self {
            backend: backend.into(),
            similarity_threshold,
        }
    }

    /// Returns the backend name.
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// Returns the optional similarity threshold.
    pub fn similarity_threshold(&self) -> Option<f64> {
        self.similarity_threshold
    }
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use std::collections::HashMap;

use serde::Deserialize;

pub mod embedding;
pub mod text;

use self::embedding::EmbeddingModelConfig;
use self::text::TextModelConfig;

/// Container for all model registries, deserialized from `[models]`.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ModelsConfig {
    /// Named text-generation model configs, each under `[models.text.<name>]`.
    #[serde(default)]
    text: HashMap<String, TextModelConfig>,

    /// Named embedding model configs, each under `[models.embedding.<name>]`.
    #[serde(default)]
    embedding: HashMap<String, EmbeddingModelConfig>,
}

impl ModelsConfig {
    /// Constructs a new `ModelsConfig`.
    pub fn new(
        text: HashMap<String, TextModelConfig>,
        embedding: HashMap<String, EmbeddingModelConfig>,
    ) -> Self {
        Self { text, embedding }
    }

    /// Returns the text model registry.
    pub fn text(&self) -> &HashMap<String, TextModelConfig> {
        &self.text
    }

    /// Returns the embedding model registry.
    pub fn embedding(&self) -> &HashMap<String, EmbeddingModelConfig> {
        &self.embedding
    }

    /// Returns a mutable reference to the text model registry.
    pub fn text_entries_mut(&mut self) -> &mut HashMap<String, TextModelConfig> {
        &mut self.text
    }

    /// Returns a mutable reference to the embedding model registry.
    pub fn embedding_entries_mut(&mut self) -> &mut HashMap<String, EmbeddingModelConfig> {
        &mut self.embedding
    }
}

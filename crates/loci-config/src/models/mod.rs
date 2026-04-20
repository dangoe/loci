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
    pub text: HashMap<String, TextModelConfig>,

    /// Named embedding model configs, each under `[models.embedding.<name>]`.
    #[serde(default)]
    pub embedding: HashMap<String, EmbeddingModelConfig>,
}

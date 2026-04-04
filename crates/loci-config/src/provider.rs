// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-config.

use serde::Deserialize;

/// Which inference service hosts this model provider.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelProviderKind {
    Ollama,
    OpenAI,
    Anthropic,
}

impl std::fmt::Display for ModelProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelProviderKind::Ollama => write!(f, "ollama"),
            ModelProviderKind::OpenAI => write!(f, "openai"),
            ModelProviderKind::Anthropic => write!(f, "anthropic"),
        }
    }
}

/// A named model provider definition.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelProviderConfig {
    /// The model provider kind (e.g. `"ollama"`, `"openai"`, `"anthropic"`).
    pub kind: ModelProviderKind,

    /// The base URL for the model provider's API.
    pub endpoint: String,

    /// Optional API key. May be a literal value or `env:VAR_NAME`.
    pub api_key: Option<String>,
}

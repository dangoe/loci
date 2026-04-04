// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-config.

use serde::Deserialize;

/// Which inference service hosts this provider.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Ollama,
    OpenAI,
    Anthropic,
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderKind::Ollama => write!(f, "ollama"),
            ProviderKind::OpenAI => write!(f, "openai"),
            ProviderKind::Anthropic => write!(f, "anthropic"),
        }
    }
}

/// A named provider definition.
#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    /// The provider kind (e.g. `"ollama"`, `"openai"`, `"anthropic"`).
    pub kind: ProviderKind,

    /// The base URL for the provider's API.
    pub endpoint: String,

    /// Optional API key. May be a literal value or `env:VAR_NAME`.
    pub api_key: Option<String>,
}

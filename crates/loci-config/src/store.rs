// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-config.

use serde::Deserialize;

/// A named memory store definition.
///
/// Each variant maps to a different backend. Serde uses the `kind` field as the
/// discriminant so the TOML representation mirrors the provider pattern:
///
/// ```toml
/// [stores.qdrant]
/// kind = "qdrant"
/// url = "http://localhost:6333"
///
/// [stores.local]
/// kind = "markdown"
/// path = "./memory"
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum StoreConfig {
    /// Qdrant vector database.
    Qdrant {
        /// Qdrant HTTP URL (e.g. `http://localhost:6333`).
        url: String,
        /// Optional API key. May be a literal value or `env:VAR_NAME`.
        api_key: Option<String>,
    },
    /// Flat markdown files on disk (planned, not yet implemented).
    Markdown {
        /// Directory path where memory files are stored.
        path: String,
    },
}

impl StoreConfig {
    /// Returns the `kind` string for use in error messages.
    pub fn kind_str(&self) -> &'static str {
        match self {
            StoreConfig::Qdrant { .. } => "qdrant",
            StoreConfig::Markdown { .. } => "markdown",
        }
    }
}

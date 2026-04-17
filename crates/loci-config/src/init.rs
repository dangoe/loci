// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use std::path::Path;

use thiserror::Error;

/// Template written by config initialization
pub const DEFAULT_CONFIG_TEMPLATE: &str = r#"########################################
# Providers — registry of available backends
########################################
[providers.ollama]
kind     = "ollama"
endpoint = "http://localhost:11434"

# [providers.openai]
# kind     = "openai"
# endpoint = "https://api.openai.com/v1"
# api_key  = "env:OPENAI_API_KEY"

# [providers.anthropic]
# kind     = "anthropic"
# endpoint = "https://api.anthropic.com"
# api_key  = "env:ANTHROPIC_API_KEY"

########################################
# Models — registry of named model configs
########################################
[models.text.default]
provider = "ollama"
model    = "qwen3.5:0.8b"

# [models.text.default.tuning]
# temperature     = 0.2
# max_tokens      = 512
# top_p           = 0.95
# repeat_penalty  = 1.1
# repeat_last_n   = 64
# keep_alive_secs = 300
# stop            = ["<END>"]

[models.text.default.tuning.thinking]
mode         = "disabled"   # "enabled" | "effort" | "budgeted"
# level      = "low"        # for mode = "effort"
# max_tokens = 256          # for mode = "budgeted"

# [models.text.default.tuning.extra]
# seed = 42

[models.embedding.default]
provider  = "ollama"
model     = "qwen3-embedding:0.6b"
dimension = 768

########################################
# Memory backends — registry of available stores
########################################
[memory.backends.qdrant]
kind       = "qdrant"
url        = "http://localhost:6334"
collection = "memory_entries"
# api_key = "env:QDRANT_API_KEY"

########################################
# Memory config — active backend + tuning
########################################
[memory.config]
backend = "qdrant"
# similarity_threshold = 0.95  # deduplication threshold (0.0–1.0)

########################################
# Memory extraction — LLM-based extraction
########################################
[memory.extraction]
model = "default"               # key in [models.text] used for extraction
# max_entries    = 20           # hard cap per extraction run (prompt hint + post-processing)
# min_confidence = 0.7          # discard entries below this LLM confidence score (0.0–1.0)
# guidelines     = "Focus on technical facts only."

[memory.extraction.thinking]
mode = "disabled"               # extraction produces structured JSON; thinking adds no benefit
# mode  = "effort"             # override when using a thinking-capable model
# level = "low"                # for mode = "effort"

# [memory.extraction.chunking]
# chunk_size   = 2500            # max characters per chunk (splits at word boundary)
# overlap_size = 200             # characters of overlap between consecutive chunks

########################################
# Routing — default selections
########################################
[routing.text]
default  = "default"
fallback = []

[routing.embedding]
default = "default"

[routing.memory]
default = "qdrant"
"#;

/// Errors that can occur while initialising a new config file.
#[derive(Debug, Error)]
pub enum ConfigInitError {
    /// The config file already exists at the given path.
    #[error("config file already exists at {path}; remove it first if you want to regenerate it")]
    AlreadyExists { path: String },

    /// A parent directory could not be created.
    #[error("could not create config directory '{path}': {source}")]
    CreateDir {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// The config file could not be written.
    #[error("could not write config file '{path}': {source}")]
    Write {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// Writes the default config template to `path`.
///
/// Creates any missing parent directories. Fails if the file already exists.
pub fn init_config(path: &Path) -> Result<(), ConfigInitError> {
    if path.exists() {
        return Err(ConfigInitError::AlreadyExists {
            path: path.display().to_string(),
        });
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| ConfigInitError::CreateDir {
            path: parent.display().to_string(),
            source: e,
        })?;
    }

    std::fs::write(path, DEFAULT_CONFIG_TEMPLATE).map_err(|e| ConfigInitError::Write {
        path: path.display().to_string(),
        source: e,
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_config_writes_file_to_new_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        init_config(&path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(
            content.contains("[providers.ollama]"),
            "expected provider section"
        );
        assert!(
            content.contains("[memory.backends.qdrant]"),
            "expected memory backend section"
        );
        assert!(
            content.contains("[routing.text]"),
            "expected routing section"
        );
    }

    #[test]
    fn test_init_config_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("sub").join("config.toml");

        init_config(&path).unwrap();

        assert!(path.exists(), "config file should have been created");
    }

    #[test]
    fn test_init_config_fails_when_file_already_exists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "existing").unwrap();

        let err = init_config(&path).unwrap_err();

        assert!(
            matches!(err, ConfigInitError::AlreadyExists { .. }),
            "expected AlreadyExists, got: {err}"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "existing");
    }

    #[test]
    fn test_init_config_error_message_contains_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "existing").unwrap();

        let err = init_config(&path).unwrap_err();

        assert!(
            err.to_string().contains(path.to_str().unwrap()),
            "error should mention path, got: {err}"
        );
    }

    #[test]
    fn test_default_template_is_valid_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        init_config(&path).unwrap();

        crate::load_config(&path)
            .expect("DEFAULT_CONFIG_TEMPLATE should parse as a valid AppConfig");
    }
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use std::path::Path;

use thiserror::Error;

/// Template written by config initialization
pub const DEFAULT_CONFIG_TEMPLATE: &str = r#"########################################
# Resources — registries of available backends and models
########################################

[resources.model_providers.ollama]
kind     = "ollama"
endpoint = "http://localhost:11434"

# [resources.model_providers.openai]
# kind     = "openai"
# endpoint = "https://api.openai.com/v1"
# api_key  = "env:OPENAI_API_KEY"

# [resources.model_providers.anthropic]
# kind     = "anthropic"
# endpoint = "https://api.anthropic.com"
# api_key  = "env:ANTHROPIC_API_KEY"

[resources.models.text.default]
provider = "ollama"
model    = "qwen3.5:0.8b"

# [resources.models.text.default.tuning]
# temperature     = 0.2
# max_tokens      = 512
# top_p           = 0.95
# repeat_penalty  = 1.1
# repeat_last_n   = 64
# keep_alive_secs = 300
# stop            = ["<END>"]

[resources.models.text.default.tuning.thinking]
mode         = "disabled"   # "enabled" | "effort" | "budgeted"
# level      = "low"        # for mode = "effort"
# max_tokens = 256          # for mode = "budgeted"

# [resources.models.text.default.tuning.extra]
# seed = 42

[resources.models.embedding.default]
provider  = "ollama"
model     = "qwen3-embedding:0.6b"
dimension = 768

[resources.memory_stores.qdrant]
kind       = "qdrant"
url        = "http://localhost:6334"
collection = "memory_entries"
# api_key = "env:QDRANT_API_KEY"

########################################
# Capabilities — what to use the resources for
########################################

[generation.text]
model = "default"           # key in [resources.models.text]

[embedding]
model = "default"           # key in [resources.models.embedding]

[memory]
store = "qdrant"            # key in [resources.memory_stores]
# similarity_threshold = 0.95  # deduplication threshold (0.0–1.0)

[memory.extraction]
model = "default"           # key in [resources.models.text] used for extraction
# max_entries    = 20       # hard cap per extraction run (prompt hint + post-processing)
# min_confidence = 0.7      # discard entries below this LLM confidence score (0.0–1.0)
# guidelines     = "Focus on technical facts only."

[memory.extraction.thinking]
mode = "disabled"           # extraction produces structured JSON; thinking adds no benefit
# mode  = "effort"         # override when using a thinking-capable model
# level = "low"            # for mode = "effort"

# [memory.extraction.chunking]
# chunk_size   = 2500      # max characters per chunk (splits at word boundary)
# overlap_size = 200       # characters of overlap between consecutive chunks

[memory.extraction.extractor]
classification_model = "default"  # key in [resources.models.text] for hit-classification (use a small, fast model)
# direct_search.max_results  = 5
# direct_search.min_score    = 0.70
# inverted_search.max_results = 3
# inverted_search.min_score   = 0.60
# bayesian_seed_weight        = 10.0
# max_counter_increment       = 5.0
# max_counter                 = 100.0
# auto_discard_threshold      = 0.1
#
# [memory.extraction.extractor.merge_strategy]
# kind  = "best_score"     # or "llm"
# model = "default"        # (llm only) key in [resources.models.text] for the merge call
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
            content.contains("[resources.model_providers.ollama]"),
            "expected model_providers section"
        );
        assert!(
            content.contains("[resources.memory_stores.qdrant]"),
            "expected memory_stores section"
        );
        assert!(
            content.contains("[generation.text]"),
            "expected generation.text section"
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

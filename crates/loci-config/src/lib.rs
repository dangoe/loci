// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

//! Configuration types and loading for the loci CLI.
//!
//! Configuration is read from a TOML file. The top-level [`AppConfig`] struct
//! holds all sections. Secrets (API keys) may be written as `env:VAR_NAME` in
//! the TOML file; they are resolved from the process environment by
//! [`load_config`].

use std::collections::HashMap;
use std::path::Path;

pub use embedding::EmbeddingModelConfig;
pub use error::ConfigError;
pub use extraction::{ChunkingConfig, MemoryExtractionConfig};
pub use init::{ConfigInitError, DEFAULT_CONFIG_TEMPLATE, init_config};
pub use memory::{MemoryConfig, MemorySection};
pub use model::{
    ModelThinkingConfig, ModelThinkingEffortLevel, ModelTuningConfig, ModelsConfig, TextModelConfig,
};
pub use pipeline::PipelineExtractionConfig;
pub use provider::{ModelProviderConfig, ModelProviderKind};
pub use routing::{EmbeddingRoutingConfig, MemoryRoutingConfig, RoutingConfig, TextRoutingConfig};
pub use store::StoreConfig;

mod embedding;
mod error;
mod extraction;
mod init;
mod memory;
mod model;
mod pipeline;
mod provider;
mod resolve;
mod routing;
mod store;

/// Top-level application configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AppConfig {
    /// Named model provider definitions.
    #[serde(default)]
    pub providers: HashMap<String, ModelProviderConfig>,

    /// Text-generation and embedding model registries.
    #[serde(default)]
    pub models: ModelsConfig,

    /// Memory backend definitions and active backend config.
    pub memory: MemorySection,

    /// Routing and default selection settings.
    pub routing: RoutingConfig,
}

/// Loads and parses an [`AppConfig`] from the given TOML file path, resolving
/// any `env:VAR_NAME` secrets.
pub fn load_config(path: &Path) -> Result<AppConfig, ConfigError> {
    let raw = std::fs::read_to_string(path).map_err(|e| ConfigError::Io {
        path: path.display().to_string(),
        source: e,
    })?;

    let mut config: AppConfig = toml::from_str(&raw).map_err(|e| ConfigError::Parse {
        path: path.display().to_string(),
        source: e,
    })?;

    resolve_secrets(&mut config)?;

    Ok(config)
}

/// Walks the config and resolves all `env:` prefixed secret values in-place.
fn resolve_secrets(config: &mut AppConfig) -> Result<(), ConfigError> {
    for provider in config.providers.values_mut() {
        if let Some(key) = provider.api_key.as_mut() {
            *key = resolve::resolve_secret(key)?;
        }
    }
    for backend in config.memory.backends.values_mut() {
        if let StoreConfig::Qdrant { api_key, .. } = backend
            && let Some(key) = api_key.as_mut()
        {
            *key = resolve::resolve_secret(key)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write as _;

    fn write_temp_config(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    const MINIMAL_CONFIG: &str = r#"
[providers.ollama]
kind = "ollama"
endpoint = "http://localhost:11434"

[models.text.default]
provider = "ollama"
model = "qwen3:0.6b"

[models.embedding.default]
provider = "ollama"
model = "qwen3-embedding:0.6b"
dimension = 768

[memory.backends.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "memory_entries"

[memory.config]
backend = "qdrant"


[memory.extraction]
model = "default"


[routing.text]
default = "default"

[routing.embedding]
default = "default"

[routing.memory]
default = "qdrant"
"#;

    #[test]
    fn parses_minimal_config() {
        let f = write_temp_config(MINIMAL_CONFIG);
        let config = load_config(f.path()).unwrap();

        assert_eq!(config.providers.len(), 1);
        assert!(config.providers.contains_key("ollama"));
        assert_eq!(config.models.text["default"].provider, "ollama");
        assert_eq!(config.models.text["default"].model, "qwen3:0.6b");
        assert!(config.models.text["default"].tuning.is_none());
        assert_eq!(config.models.embedding["default"].dimension, 768);
        assert_eq!(config.memory.config.backend, "qdrant");
        assert_eq!(config.routing.text.default, "default");
        assert_eq!(config.routing.embedding.default, "default");
        assert_eq!(config.routing.memory.default, "qdrant");
    }

    #[test]
    fn resolves_literal_api_key() {
        let cfg = r#"
[providers.openai]
kind = "openai"
endpoint = "https://api.openai.com/v1"
api_key = "sk-literal-key"

[memory.backends.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[memory.config]
backend = "qdrant"

[memory.extraction]
model = "default"


[routing.text]
default = "x"

[routing.embedding]
default = "x"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        assert_eq!(
            config.providers["openai"].api_key.as_deref(),
            Some("sk-literal-key")
        );
    }

    #[test]
    fn resolves_env_api_key() {
        // SAFETY: single-threaded test process; no other threads read this var.
        unsafe { std::env::set_var("LOCI_TEST_SECRET", "resolved-value") };
        let cfg = r#"
[providers.openai]
kind = "openai"
endpoint = "https://api.openai.com/v1"
api_key = "env:LOCI_TEST_SECRET"

[memory.backends.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[memory.config]
backend = "qdrant"

[memory.extraction]
model = "default"


[routing.text]
default = "x"

[routing.embedding]
default = "x"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        assert_eq!(
            config.providers["openai"].api_key.as_deref(),
            Some("resolved-value")
        );
        // SAFETY: single-threaded test process; no other threads read this var.
        unsafe { std::env::remove_var("LOCI_TEST_SECRET") };
    }

    #[test]
    fn missing_env_var_returns_error() {
        // SAFETY: single-threaded test process; no other threads read this var.
        unsafe { std::env::remove_var("LOCI_TEST_MISSING_VAR") };
        let cfg = r#"
[providers.openai]
kind = "openai"
endpoint = "https://api.openai.com/v1"
api_key = "env:LOCI_TEST_MISSING_VAR"

[memory.backends.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[memory.config]
backend = "qdrant"

[memory.extraction]
model = "default"


[routing.text]
default = "x"

[routing.embedding]
default = "x"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let err = load_config(f.path()).unwrap_err();
        assert!(matches!(err, ConfigError::EnvVar { .. }));
    }

    #[test]
    fn missing_config_file_returns_io_error() {
        let err = load_config(Path::new("/nonexistent/path/config.toml")).unwrap_err();
        assert!(matches!(err, ConfigError::Io { .. }));
    }

    #[test]
    fn invalid_toml_returns_parse_error() {
        let f = write_temp_config("this is not toml ][");
        let err = load_config(f.path()).unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn similarity_threshold_is_optional() {
        let f = write_temp_config(MINIMAL_CONFIG);
        let config = load_config(f.path()).unwrap();
        assert!(config.memory.config.similarity_threshold.is_none());
    }

    #[test]
    fn similarity_threshold_parsed_when_set() {
        let cfg = r#"
[providers.ollama]
kind = "ollama"
endpoint = "http://localhost:11434"

[memory.backends.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "memory_entries"

[memory.config]
backend = "qdrant"

similarity_threshold = 0.92

[memory.extraction]
model = "default"


[routing.text]
default = "default"

[routing.embedding]
default = "default"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        assert_eq!(config.memory.config.similarity_threshold, Some(0.92));
    }

    #[test]
    fn qdrant_backend_with_env_api_key() {
        // SAFETY: single-threaded test process; no other threads read this var.
        unsafe { std::env::set_var("LOCI_QDRANT_KEY", "qdrant-secret") };
        let cfg = r#"
[memory.backends.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"
api_key = "env:LOCI_QDRANT_KEY"

[memory.config]
backend = "qdrant"

[memory.extraction]
model = "default"


[routing.text]
default = "x"

[routing.embedding]
default = "x"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        if let StoreConfig::Qdrant { api_key, .. } = &config.memory.backends["qdrant"] {
            assert_eq!(api_key.as_deref(), Some("qdrant-secret"));
        } else {
            panic!("expected Qdrant backend");
        }
        // SAFETY: single-threaded test process; no other threads read this var.
        unsafe { std::env::remove_var("LOCI_QDRANT_KEY") };
    }

    #[test]
    fn fallback_defaults_to_empty_vec() {
        let f = write_temp_config(MINIMAL_CONFIG);
        let config = load_config(f.path()).unwrap();
        assert!(config.routing.text.fallback.is_empty());
    }

    #[test]
    fn fallback_is_parsed_when_set() {
        let cfg = r#"
[providers.ollama]
kind = "ollama"
endpoint = "http://localhost:11434"

[memory.backends.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[memory.config]
backend = "qdrant"

[memory.extraction]
model = "default"


[routing.text]
default = "primary"
fallback = ["secondary", "tertiary"]

[routing.embedding]
default = "default"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        assert_eq!(config.routing.text.fallback, vec!["secondary", "tertiary"]);
    }

    /// `[providers]` is `#[serde(default)]`, so an absent section is valid and
    /// results in an empty providers map rather than a parse error.
    #[test]
    fn missing_providers_section_is_accepted_with_empty_map() {
        let cfg = r#"
[memory.backends.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[memory.config]
backend = "qdrant"

[memory.extraction]
model = "default"


[routing.text]
default = "x"

[routing.embedding]
default = "x"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        assert!(
            config.providers.is_empty(),
            "providers should be empty when the section is absent"
        );
    }

    #[test]
    fn invalid_provider_kind_returns_parse_error() {
        let cfg = r#"
[providers.bad]
kind = "invalid_kind"
endpoint = "http://localhost:11434"

[memory.backends.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[memory.config]
backend = "qdrant"

[memory.extraction]
model = "default"


[routing.text]
default = "x"

[routing.embedding]
default = "x"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let err = load_config(f.path()).unwrap_err();
        assert!(
            matches!(err, ConfigError::Parse { .. }),
            "expected Parse error for unknown provider kind, got: {err:?}"
        );
    }

    #[test]
    fn text_model_without_provider_returns_parse_error() {
        // `provider` is a required field on `TextModelConfig` (no `#[serde(default)]`).
        let cfg = r#"
[models.text.default]
model = "qwen3:0.6b"

[memory.backends.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[memory.config]
backend = "qdrant"

[memory.extraction]
model = "default"


[routing.text]
default = "default"

[routing.embedding]
default = "x"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let err = load_config(f.path()).unwrap_err();
        assert!(
            matches!(err, ConfigError::Parse { .. }),
            "expected Parse error when `provider` is missing from a text model, got: {err:?}"
        );
    }

    #[test]
    fn text_model_without_model_field_returns_parse_error() {
        // `model` is a required field on `TextModelConfig` (no `#[serde(default)]`).
        let cfg = r#"
[models.text.default]
provider = "ollama"

[memory.backends.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[memory.config]
backend = "qdrant"

[memory.extraction]
model = "default"


[routing.text]
default = "default"

[routing.embedding]
default = "x"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let err = load_config(f.path()).unwrap_err();
        assert!(
            matches!(err, ConfigError::Parse { .. }),
            "expected Parse error when `model` is missing from a text model, got: {err:?}"
        );
    }

    #[test]
    fn temperature_as_string_returns_parse_error() {
        // `temperature` is typed as `Option<f32>`; a string value is a type mismatch.
        let cfg = r#"
[providers.ollama]
kind = "ollama"
endpoint = "http://localhost:11434"

[models.text.default]
provider = "ollama"
model = "qwen3:0.6b"

[models.text.default.tuning]
temperature = "not_a_number"

[memory.backends.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[memory.config]
backend = "qdrant"

[memory.extraction]
model = "default"


[routing.text]
default = "default"

[routing.embedding]
default = "x"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let err = load_config(f.path()).unwrap_err();
        assert!(
            matches!(err, ConfigError::Parse { .. }),
            "expected Parse error for temperature = \"not_a_number\", got: {err:?}"
        );
    }

    #[test]
    fn markdown_backend_is_parsed_correctly() {
        let cfg = r#"
[memory.backends.local]
kind = "markdown"
path = "./memory"

[memory.config]
backend = "local"

[memory.extraction]
model = "default"


[routing.text]
default = "x"

[routing.embedding]
default = "x"

[routing.memory]
default = "local"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        if let StoreConfig::Markdown { path } = &config.memory.backends["local"] {
            assert_eq!(path, "./memory");
        } else {
            panic!("expected Markdown backend");
        }
    }

    #[test]
    fn model_tuning_is_parsed_when_set() {
        let cfg = r#"
[providers.ollama]
kind = "ollama"
endpoint = "http://localhost:11434"

[models.text.default]
provider = "ollama"
model = "qwen3:0.6b"

[models.text.default.tuning]
temperature = 0.2
max_tokens = 512
top_p = 0.95
repeat_penalty = 1.2
repeat_last_n = 64
keep_alive_secs = 300
stop = ["<END>"]

[models.text.default.tuning.thinking]
mode = "effort"
level = "low"

[models.text.default.tuning.extra]
seed = 42

[memory.backends.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "memory_entries"

[memory.config]
backend = "qdrant"


[memory.extraction]
model = "default"


[routing.text]
default = "default"

[routing.embedding]
default = "default"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        let tuning = config.models.text["default"].tuning.as_ref().unwrap();
        assert_eq!(tuning.temperature, Some(0.2));
        assert_eq!(tuning.max_tokens, Some(512));
        assert_eq!(tuning.top_p, Some(0.95));
        assert_eq!(tuning.repeat_penalty, Some(1.2));
        assert_eq!(tuning.repeat_last_n, Some(64));
        assert_eq!(tuning.keep_alive_secs, Some(300));
        assert_eq!(tuning.stop.as_ref().unwrap(), &vec!["<END>".to_string()]);
        assert!(matches!(
            tuning.thinking,
            Some(ModelThinkingConfig::Effort {
                level: ModelThinkingEffortLevel::Low
            })
        ));
        assert_eq!(tuning.extra.get("seed"), Some(&json!(42)));
    }
}

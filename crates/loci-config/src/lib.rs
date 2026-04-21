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

pub use error::ConfigError;
pub use init::{ConfigInitError, DEFAULT_CONFIG_TEMPLATE, init_config};
pub use memory::extraction::{ChunkingConfig, MemoryExtractionConfig};
pub use memory::extraction::{
    MemoryExtractorConfig, MemoryExtractorSearchResultsConfig, MergeStrategyConfig,
};
pub use memory::store::StoreConfig;
pub use memory::{MemoryConfig, MemorySection};
pub use models::ModelsConfig;
pub use models::embedding::EmbeddingModelConfig;
pub use models::text::{
    ModelThinkingConfig, ModelThinkingEffortLevel, ModelTuningConfig, TextModelConfig,
};
pub use providers::{ModelProviderConfig, ModelProviderKind};
pub use routing::{EmbeddingRoutingConfig, MemoryRoutingConfig, RoutingConfig, TextRoutingConfig};

mod error;
mod init;
mod memory;
mod models;
mod providers;
mod resolve;
mod routing;

/// Top-level application configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AppConfig {
    /// Named model provider definitions.
    #[serde(default)]
    providers: HashMap<String, ModelProviderConfig>,

    /// Text-generation and embedding model registries.
    #[serde(default)]
    models: ModelsConfig,

    /// Memory backend definitions and active backend config.
    memory: MemorySection,

    /// Routing and default selection settings.
    routing: RoutingConfig,
}

impl AppConfig {
    /// Constructs a new `AppConfig`.
    pub fn new(
        providers: HashMap<String, ModelProviderConfig>,
        models: ModelsConfig,
        memory: MemorySection,
        routing: RoutingConfig,
    ) -> Self {
        Self {
            providers,
            models,
            memory,
            routing,
        }
    }

    /// Returns the named provider definitions.
    pub fn providers(&self) -> &HashMap<String, ModelProviderConfig> {
        &self.providers
    }

    /// Returns the model registries.
    pub fn models(&self) -> &ModelsConfig {
        &self.models
    }

    /// Returns the memory section.
    pub fn memory(&self) -> &MemorySection {
        &self.memory
    }

    /// Returns the routing configuration.
    pub fn routing(&self) -> &RoutingConfig {
        &self.routing
    }

    /// Returns a mutable reference to the routing configuration.
    pub fn routing_mut(&mut self) -> &mut RoutingConfig {
        &mut self.routing
    }

    /// Returns a mutable reference to the model registries.
    pub fn models_mut(&mut self) -> &mut ModelsConfig {
        &mut self.models
    }
}

/// Loads and parses an [`AppConfig`] from the given TOML file path, resolving
/// any `env:VAR_NAME` secrets.
pub fn load_config(path: &Path) -> Result<AppConfig, ConfigError> {
    let raw = std::fs::read_to_string(path).map_err(|e| ConfigError::Io {
        path: path.display().to_string(),
        source: e,
    })?;

    load_config_from_str(&raw).map_err(|e| match e {
        ConfigError::Parse { source, .. } => ConfigError::Parse {
            path: path.display().to_string(),
            source,
        },
        other => other,
    })
}

/// Parses and resolves an [`AppConfig`] from a raw TOML string.
///
/// Useful for constructing configs in tests without writing to the filesystem.
pub fn load_config_from_str(raw: &str) -> Result<AppConfig, ConfigError> {
    let mut config: AppConfig = toml::from_str(raw).map_err(|e| ConfigError::Parse {
        path: "<string>".to_string(),
        source: e,
    })?;
    resolve_secrets(&mut config)?;
    Ok(config)
}

/// Walks the config and resolves all `env:` prefixed secret values in-place.
fn resolve_secrets(config: &mut AppConfig) -> Result<(), ConfigError> {
    for provider in config.providers.values_mut() {
        provider.resolve_api_key()?;
    }
    config.memory.resolve_secrets()?;
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

[memory.extraction.extractor]
classification_model = "x"


[routing.text]
default = "default"

[routing.embedding]
default = "default"

[routing.memory]
default = "qdrant"
"#;

    #[test]
    fn test_parses_minimal_config() {
        let f = write_temp_config(MINIMAL_CONFIG);
        let config = load_config(f.path()).unwrap();

        assert_eq!(config.providers().len(), 1);
        assert!(config.providers().contains_key("ollama"));
        assert_eq!(config.models().text()["default"].provider(), "ollama");
        assert_eq!(config.models().text()["default"].model(), "qwen3:0.6b");
        assert!(config.models().text()["default"].tuning().is_none());
        assert_eq!(config.models().embedding()["default"].dimension(), 768);
        assert_eq!(config.memory().config().backend(), "qdrant");
        assert_eq!(config.routing().text().default(), "default");
        assert_eq!(config.routing().embedding().default(), "default");
        assert_eq!(config.routing().memory().default(), "qdrant");
    }

    #[test]
    fn test_resolves_literal_api_key() {
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

[memory.extraction.extractor]
classification_model = "x"


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
            config.providers()["openai"].api_key(),
            Some("sk-literal-key")
        );
    }

    #[test]
    fn test_resolves_env_api_key() {
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

[memory.extraction.extractor]
classification_model = "x"


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
            config.providers()["openai"].api_key(),
            Some("resolved-value")
        );
        // SAFETY: single-threaded test process; no other threads read this var.
        unsafe { std::env::remove_var("LOCI_TEST_SECRET") };
    }

    #[test]
    fn test_missing_env_var_returns_error() {
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

[memory.extraction.extractor]
classification_model = "x"


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
    fn test_missing_config_file_returns_io_error() {
        let err = load_config(Path::new("/nonexistent/path/config.toml")).unwrap_err();
        assert!(matches!(err, ConfigError::Io { .. }));
    }

    #[test]
    fn test_invalid_toml_returns_parse_error() {
        let f = write_temp_config("this is not toml ][");
        let err = load_config(f.path()).unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn test_similarity_threshold_is_optional() {
        let f = write_temp_config(MINIMAL_CONFIG);
        let config = load_config(f.path()).unwrap();
        assert!(config.memory().config().similarity_threshold().is_none());
    }

    #[test]
    fn test_similarity_threshold_parsed_when_set() {
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

[memory.extraction.extractor]
classification_model = "x"


[routing.text]
default = "default"

[routing.embedding]
default = "default"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        assert_eq!(config.memory().config().similarity_threshold(), Some(0.92));
    }

    #[test]
    fn test_qdrant_backend_with_env_api_key() {
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

[memory.extraction.extractor]
classification_model = "x"


[routing.text]
default = "x"

[routing.embedding]
default = "x"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        if let StoreConfig::Qdrant { api_key, .. } = &config.memory().backends()["qdrant"] {
            assert_eq!(api_key.as_deref(), Some("qdrant-secret"));
        } else {
            panic!("expected Qdrant backend");
        }
        // SAFETY: single-threaded test process; no other threads read this var.
        unsafe { std::env::remove_var("LOCI_QDRANT_KEY") };
    }

    #[test]
    fn test_fallback_defaults_to_empty_vec() {
        let f = write_temp_config(MINIMAL_CONFIG);
        let config = load_config(f.path()).unwrap();
        assert!(config.routing().text().fallback().is_empty());
    }

    #[test]
    fn test_fallback_is_parsed_when_set() {
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

[memory.extraction.extractor]
classification_model = "x"


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
        assert_eq!(
            config.routing().text().fallback(),
            &["secondary", "tertiary"]
        );
    }

    /// `[providers]` is `#[serde(default)]`, so an absent section is valid and
    /// results in an empty providers map rather than a parse error.
    #[test]
    fn test_missing_providers_section_is_accepted_with_empty_map() {
        let cfg = r#"
[memory.backends.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[memory.config]
backend = "qdrant"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"


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
            config.providers().is_empty(),
            "providers should be empty when the section is absent"
        );
    }

    #[test]
    fn test_invalid_provider_kind_returns_parse_error() {
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

[memory.extraction.extractor]
classification_model = "x"


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
    fn test_text_model_without_provider_returns_parse_error() {
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

[memory.extraction.extractor]
classification_model = "x"


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
    fn test_text_model_without_model_field_returns_parse_error() {
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

[memory.extraction.extractor]
classification_model = "x"


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
    fn test_temperature_as_string_returns_parse_error() {
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

[memory.extraction.extractor]
classification_model = "x"


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
    fn test_markdown_backend_is_parsed_correctly() {
        let cfg = r#"
[memory.backends.local]
kind = "markdown"
path = "./memory"

[memory.config]
backend = "local"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"


[routing.text]
default = "x"

[routing.embedding]
default = "x"

[routing.memory]
default = "local"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        if let StoreConfig::Markdown { path } = &config.memory().backends()["local"] {
            assert_eq!(path, "./memory");
        } else {
            panic!("expected Markdown backend");
        }
    }

    #[test]
    fn test_model_tuning_is_parsed_when_set() {
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

[memory.extraction.extractor]
classification_model = "x"


[routing.text]
default = "default"

[routing.embedding]
default = "default"

[routing.memory]
default = "qdrant"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        let tuning = config.models().text()["default"].tuning().unwrap();
        assert_eq!(tuning.temperature(), Some(0.2));
        assert_eq!(tuning.max_tokens(), Some(512));
        assert_eq!(tuning.top_p(), Some(0.95));
        assert_eq!(tuning.repeat_penalty(), Some(1.2));
        assert_eq!(tuning.repeat_last_n(), Some(64));
        assert_eq!(tuning.keep_alive_secs(), Some(300));
        assert_eq!(tuning.stop().unwrap(), &["<END>".to_string()]);
        assert!(matches!(
            tuning.thinking(),
            Some(ModelThinkingConfig::Effort {
                level: ModelThinkingEffortLevel::Low
            })
        ));
        assert_eq!(tuning.extra().get("seed"), Some(&json!(42)));
    }
}

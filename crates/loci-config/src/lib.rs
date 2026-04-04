// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-config.

//! Configuration types and loading for the loci CLI.
//!
//! Configuration is read from a TOML file. The top-level [`AppConfig`] struct
//! holds all sections. Secrets (API keys) may be written as `env:VAR_NAME` in
//! the TOML file; they are resolved from the process environment by
//! [`load_config`].

use std::collections::HashMap;
use std::path::Path;

pub use embedding::EmbeddingProfileConfig;
pub use error::ConfigError;
pub use memory::MemoryConfig;
pub use model::ModelConfig;
pub use provider::{ModelProviderConfig, ModelProviderKind};
pub use routing::RoutingConfig;
pub use store::StoreConfig;

mod embedding;
mod error;
mod memory;
mod model;
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

    /// Named model aliases, each referencing a provider.
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,

    /// Named embedding profiles, each referencing a provider.
    #[serde(default)]
    pub embeddings: HashMap<String, EmbeddingProfileConfig>,

    /// Named memory store definitions.
    #[serde(default)]
    pub stores: HashMap<String, StoreConfig>,

    /// Memory persistence settings.
    pub memory: MemoryConfig,

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
    for store in config.stores.values_mut() {
        if let StoreConfig::Qdrant { api_key, .. } = store
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

[models.default]
provider = "ollama"
name = "qwen3:0.6b"

[embeddings.default]
provider = "ollama"
model = "qwen3-embedding:0.6b"
dimension = 768

[stores.qdrant]
kind = "qdrant"
url = "http://localhost:6333"

[memory]
store = "qdrant"
collection = "memories"

[routing]
default_model = "default"
embedding = "default"
"#;

    #[test]
    fn parses_minimal_config() {
        let f = write_temp_config(MINIMAL_CONFIG);
        let config = load_config(f.path()).unwrap();

        assert_eq!(config.providers.len(), 1);
        assert!(config.providers.contains_key("ollama"));
        assert_eq!(config.models["default"].provider, "ollama");
        assert_eq!(config.models["default"].name, "qwen3:0.6b");
        assert_eq!(config.embeddings["default"].dimension, 768);
        assert_eq!(config.memory.store, "qdrant");
        assert_eq!(config.memory.collection, "memories");
        assert_eq!(config.routing.default_model, "default");
        assert_eq!(config.routing.embedding, "default");
    }

    #[test]
    fn resolves_literal_api_key() {
        let cfg = r#"
[providers.openai]
kind = "openai"
endpoint = "https://api.openai.com/v1"
api_key = "sk-literal-key"

[stores.qdrant]
kind = "qdrant"
url = "http://localhost:6334"

[memory]
store = "qdrant"
collection = "mem"

[routing]
default_model = "x"
embedding = "x"
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

[stores.qdrant]
kind = "qdrant"
url = "http://localhost:6333"

[memory]
store = "qdrant"
collection = "mem"

[routing]
default_model = "x"
embedding = "x"
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

[stores.qdrant]
kind = "qdrant"
url = "http://localhost:6333"

[memory]
store = "qdrant"
collection = "mem"

[routing]
default_model = "x"
embedding = "x"
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
        assert!(config.memory.similarity_threshold.is_none());
    }

    #[test]
    fn similarity_threshold_parsed_when_set() {
        let cfg = r#"
[providers.ollama]
kind = "ollama"
endpoint = "http://localhost:11434"

[stores.qdrant]
kind = "qdrant"
url = "http://localhost:6333"

[memory]
store = "qdrant"
collection = "memories"
similarity_threshold = 0.92

[routing]
default_model = "default"
embedding = "default"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        assert_eq!(config.memory.similarity_threshold, Some(0.92));
    }

    #[test]
    fn qdrant_store_with_env_api_key() {
        // SAFETY: single-threaded test process; no other threads read this var.
        unsafe { std::env::set_var("LOCI_QDRANT_KEY", "qdrant-secret") };
        let cfg = r#"
[stores.qdrant]
kind = "qdrant"
url = "http://localhost:6333"
api_key = "env:LOCI_QDRANT_KEY"

[memory]
store = "qdrant"
collection = "mem"

[routing]
default_model = "x"
embedding = "x"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        if let StoreConfig::Qdrant { api_key, .. } = &config.stores["qdrant"] {
            assert_eq!(api_key.as_deref(), Some("qdrant-secret"));
        } else {
            panic!("expected Qdrant store");
        }
        // SAFETY: single-threaded test process; no other threads read this var.
        unsafe { std::env::remove_var("LOCI_QDRANT_KEY") };
    }

    #[test]
    fn fallback_models_defaults_to_empty_vec() {
        let f = write_temp_config(MINIMAL_CONFIG);
        let config = load_config(f.path()).unwrap();
        assert!(config.routing.fallback_models.is_empty());
    }

    #[test]
    fn fallback_models_are_parsed_when_set() {
        let cfg = r#"
[providers.ollama]
kind = "ollama"
endpoint = "http://localhost:11434"

[stores.qdrant]
kind = "qdrant"
url = "http://localhost:6333"

[memory]
store = "qdrant"
collection = "mem"

[routing]
default_model = "primary"
embedding = "default"
fallback_models = ["secondary", "tertiary"]
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        assert_eq!(
            config.routing.fallback_models,
            vec!["secondary", "tertiary"]
        );
    }

    #[test]
    fn markdown_store_is_parsed_correctly() {
        let cfg = r#"
[stores.local]
kind = "markdown"
path = "./memory"

[memory]
store = "local"
collection = "notes"

[routing]
default_model = "x"
embedding = "x"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        if let StoreConfig::Markdown { path } = &config.stores["local"] {
            assert_eq!(path, "./memory");
        } else {
            panic!("expected Markdown store");
        }
    }
}

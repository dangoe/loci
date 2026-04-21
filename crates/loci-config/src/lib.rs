// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

//! Configuration types and loading for the loci CLI.
//!
//! Configuration is read from a TOML file. The top-level [`AppConfig`] struct
//! holds all sections. Secrets (API keys) may be written as `env:VAR_NAME` in
//! the TOML file; they are resolved from the process environment by
//! [`load_config`].

use std::path::Path;

pub use embedding_capability::EmbeddingCapabilityConfig;
pub use error::ConfigError;
pub use generation::{GenerationConfig, TextGenerationConfig};
pub use init::{ConfigInitError, DEFAULT_CONFIG_TEMPLATE, init_config};
pub use memory::MemorySection;
pub use memory::extraction::{ChunkingConfig, MemoryExtractionConfig};
pub use memory::extraction::{
    MemoryExtractorConfig, MemoryExtractorSearchResultsConfig, MergeStrategyConfig,
};
pub use memory::store::StoreConfig;
pub use models::ModelsConfig;
pub use models::embedding::EmbeddingModelConfig;
pub use models::text::{
    ModelThinkingConfig, ModelThinkingEffortLevel, ModelTuningConfig, TextModelConfig,
};
pub use resources::{ModelProviderConfig, ModelProviderKind, ResourcesConfig};

mod embedding_capability;
mod error;
mod generation;
mod init;
mod memory;
mod models;
mod providers;
mod resolve;
mod resources;

/// Top-level application configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AppConfig {
    /// All infrastructure registries (model providers, models, memory stores).
    #[serde(default)]
    resources: ResourcesConfig,

    /// Text-generation capability configuration, under `[generation]`.
    generation: GenerationConfig,

    /// Embedding capability configuration, under `[embedding]`.
    embedding: EmbeddingCapabilityConfig,

    /// Memory backend selection and extraction config, under `[memory]`.
    memory: MemorySection,
}

impl AppConfig {
    /// Returns the resource registries.
    pub fn resources(&self) -> &ResourcesConfig {
        &self.resources
    }

    /// Returns a mutable reference to the resource registries.
    pub fn resources_mut(&mut self) -> &mut ResourcesConfig {
        &mut self.resources
    }

    /// Returns the generation capability configuration.
    pub fn generation(&self) -> &GenerationConfig {
        &self.generation
    }

    /// Returns a mutable reference to the generation capability configuration.
    pub fn generation_mut(&mut self) -> &mut GenerationConfig {
        &mut self.generation
    }

    /// Returns the embedding capability configuration.
    pub fn embedding(&self) -> &EmbeddingCapabilityConfig {
        &self.embedding
    }

    /// Returns a mutable reference to the embedding capability configuration.
    pub fn embedding_mut(&mut self) -> &mut EmbeddingCapabilityConfig {
        &mut self.embedding
    }

    /// Returns the memory section.
    pub fn memory(&self) -> &MemorySection {
        &self.memory
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
    for provider in config.resources.model_providers_mut().values_mut() {
        provider.resolve_api_key()?;
    }
    for store in config.resources.memory_stores_mut().values_mut() {
        store.resolve_api_key()?;
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
[resources.model_providers.ollama]
kind = "ollama"
endpoint = "http://localhost:11434"

[resources.models.text.default]
provider = "ollama"
model = "qwen3:0.6b"

[resources.models.embedding.default]
provider = "ollama"
model = "qwen3-embedding:0.6b"
dimension = 768

[resources.memory_stores.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "memory_entries"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "qdrant"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
"#;

    #[test]
    fn test_parses_minimal_config() {
        let f = write_temp_config(MINIMAL_CONFIG);
        let config = load_config(f.path()).unwrap();

        assert_eq!(config.resources().model_providers().len(), 1);
        assert!(config.resources().model_providers().contains_key("ollama"));
        assert_eq!(
            config.resources().models().text()["default"].provider(),
            "ollama"
        );
        assert_eq!(
            config.resources().models().text()["default"].model(),
            "qwen3:0.6b"
        );
        assert!(
            config.resources().models().text()["default"]
                .tuning()
                .is_none()
        );
        assert_eq!(
            config.resources().models().embedding()["default"].dimension(),
            768
        );
        assert_eq!(config.memory().store(), "qdrant");
        assert_eq!(config.generation().text().model(), "default");
        assert_eq!(config.embedding().model(), "default");
    }

    #[test]
    fn test_resolves_literal_api_key() {
        let cfg = r#"
[resources.model_providers.openai]
kind = "openai"
endpoint = "https://api.openai.com/v1"
api_key = "sk-literal-key"

[resources.memory_stores.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "qdrant"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        assert_eq!(
            config.resources().model_providers()["openai"].api_key(),
            Some("sk-literal-key")
        );
    }

    #[test]
    fn test_resolves_env_api_key() {
        // SAFETY: single-threaded test process; no other threads read this var.
        unsafe { std::env::set_var("LOCI_TEST_SECRET", "resolved-value") };
        let cfg = r#"
[resources.model_providers.openai]
kind = "openai"
endpoint = "https://api.openai.com/v1"
api_key = "env:LOCI_TEST_SECRET"

[resources.memory_stores.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "qdrant"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        assert_eq!(
            config.resources().model_providers()["openai"].api_key(),
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
[resources.model_providers.openai]
kind = "openai"
endpoint = "https://api.openai.com/v1"
api_key = "env:LOCI_TEST_MISSING_VAR"

[resources.memory_stores.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "qdrant"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
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
        assert!(config.memory().similarity_threshold().is_none());
    }

    #[test]
    fn test_similarity_threshold_parsed_when_set() {
        let cfg = r#"
[resources.model_providers.ollama]
kind = "ollama"
endpoint = "http://localhost:11434"

[resources.memory_stores.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "memory_entries"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "qdrant"
similarity_threshold = 0.92

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        assert_eq!(config.memory().similarity_threshold(), Some(0.92));
    }

    #[test]
    fn test_qdrant_backend_with_env_api_key() {
        // SAFETY: single-threaded test process; no other threads read this var.
        unsafe { std::env::set_var("LOCI_QDRANT_KEY", "qdrant-secret") };
        let cfg = r#"
[resources.memory_stores.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"
api_key = "env:LOCI_QDRANT_KEY"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "qdrant"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        if let StoreConfig::Qdrant { api_key, .. } = &config.resources().memory_stores()["qdrant"] {
            assert_eq!(api_key.as_deref(), Some("qdrant-secret"));
        } else {
            panic!("expected Qdrant backend");
        }
        // SAFETY: single-threaded test process; no other threads read this var.
        unsafe { std::env::remove_var("LOCI_QDRANT_KEY") };
    }

    #[test]
    fn test_missing_model_providers_section_is_accepted_with_empty_map() {
        let cfg = r#"
[resources.memory_stores.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "qdrant"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        assert!(
            config.resources().model_providers().is_empty(),
            "model_providers should be empty when the section is absent"
        );
    }

    #[test]
    fn test_invalid_provider_kind_returns_parse_error() {
        let cfg = r#"
[resources.model_providers.bad]
kind = "invalid_kind"
endpoint = "http://localhost:11434"

[resources.memory_stores.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "qdrant"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
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
        let cfg = r#"
[resources.models.text.default]
model = "qwen3:0.6b"

[resources.memory_stores.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "qdrant"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
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
        let cfg = r#"
[resources.models.text.default]
provider = "ollama"

[resources.memory_stores.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "qdrant"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
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
        let cfg = r#"
[resources.model_providers.ollama]
kind = "ollama"
endpoint = "http://localhost:11434"

[resources.models.text.default]
provider = "ollama"
model = "qwen3:0.6b"

[resources.models.text.default.tuning]
temperature = "not_a_number"

[resources.memory_stores.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "mem"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "qdrant"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
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
[resources.memory_stores.local]
kind = "markdown"
path = "./memory"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "local"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        if let StoreConfig::Markdown { path } = &config.resources().memory_stores()["local"] {
            assert_eq!(path, "./memory");
        } else {
            panic!("expected Markdown backend");
        }
    }

    #[test]
    fn test_model_tuning_is_parsed_when_set() {
        let cfg = r#"
[resources.model_providers.ollama]
kind = "ollama"
endpoint = "http://localhost:11434"

[resources.models.text.default]
provider = "ollama"
model = "qwen3:0.6b"

[resources.models.text.default.tuning]
temperature = 0.2
max_tokens = 512
top_p = 0.95
repeat_penalty = 1.2
repeat_last_n = 64
keep_alive_secs = 300
stop = ["<END>"]

[resources.models.text.default.tuning.thinking]
mode = "effort"
level = "low"

[resources.models.text.default.tuning.extra]
seed = 42

[resources.memory_stores.qdrant]
kind = "qdrant"
url = "http://localhost:6334"
collection = "memory_entries"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "qdrant"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
"#;
        let f = write_temp_config(cfg);
        let config = load_config(f.path()).unwrap();
        let tuning = config.resources().models().text()["default"]
            .tuning()
            .unwrap();
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

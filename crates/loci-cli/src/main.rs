// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-cli.

//! Binary entry-point for the `loci` CLI.
//!
//! All infrastructure configuration (providers, stores, models, embeddings) is
//! read from a TOML config file (default: `~/.config/loci/config.toml`).
//! The CLI itself only exposes operational flags and sub-commands.

mod cli;
mod commands;
#[cfg(test)]
mod fixture;
mod handlers;
#[cfg(test)]
mod mock;

use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "systemd-journal-logger")]
use systemd_journal_logger::JournalLog;

use clap::Parser;
use log::{LevelFilter, error, info};

use loci_config::{
    AppConfig, ConfigError, ModelProviderConfig, ModelProviderKind, StoreConfig, load_config,
};

use loci_core::embedding::DefaultTextEmbedder;
use loci_memory_store_qdrant::config::QdrantConfig;
use loci_memory_store_qdrant::store::QdrantMemoryStore;
use loci_model_provider_ollama::provider::{OllamaConfig, OllamaModelProvider};

use crate::cli::Cli;
use crate::commands::{Command, ConfigCommand};
use crate::handlers::CommandHandler;
use crate::handlers::generate::GenerateCommandHandler;
use crate::handlers::memory::MemoryCommandHandler;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    setup_logging(cli.verbose);

    info!("Starting loci CLI");

    if let Err(e) = run(cli).await {
        error!("error: {e}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = resolve_config_path(cli.config);

    // The `config init` command must be handled before loading the config file,
    // as the file may not exist yet.
    if let Command::Config {
        command: ConfigCommand::Init,
    } = &cli.command
    {
        return cmd_config_init(&config_path, &mut std::io::stdout());
    }

    info!("Loading config from {}", config_path.display());
    let config = load_config(&config_path)?;

    match cli.command {
        Command::Memory { command } => {
            let store = build_store(&config).await?;
            let handler = MemoryCommandHandler::new(&store);
            handler.handle(command, &mut std::io::stdout()).await
        }
        Command::Generate { args: command } => {
            let store = build_store(&config).await?;
            let llm_provider = build_llm_provider(&config)?;
            let handler = GenerateCommandHandler::new(&store, &llm_provider, &config);
            handler.handle(command, &mut std::io::stdout()).await
        }
        Command::Config { .. } => unreachable!("handled above"),
    }
}

/// Builds a `QdrantMemoryStore<DefaultTextEmbedder<OllamaModelProvider>>` from the active config.
///
/// Fails fast with [`ConfigError::UnsupportedKind`] if the configured store
/// or embedding provider is not yet implemented.
async fn build_store(
    config: &AppConfig,
) -> Result<QdrantMemoryStore<DefaultTextEmbedder<OllamaModelProvider>>, Box<dyn std::error::Error>> {
    let store_name = &config.memory.store;
    let store_cfg = config
        .stores
        .get(store_name)
        .ok_or_else(|| ConfigError::MissingKey {
            section: "stores".into(),
            key: store_name.clone(),
        })?;

    match store_cfg {
        StoreConfig::Qdrant { url, .. } => {
            let embed_provider = resolve_embedding_provider(config)?;
            let embed_provider_instance = build_ollama_provider(embed_provider)?;
            let embed_profile_name = &config.routing.embedding;
            let embed_profile = config.embeddings.get(embed_profile_name).ok_or_else(|| {
                ConfigError::MissingKey {
                    section: "embeddings".into(),
                    key: embed_profile_name.clone(),
                }
            })?;

            let embedder = DefaultTextEmbedder::new(
                Arc::new(embed_provider_instance),
                &embed_profile.model,
                embed_profile.dimension,
            );

            let qdrant_config = QdrantConfig {
                collection_name: config.memory.collection.clone(),
                similarity_threshold: config.memory.similarity_threshold,
                promotion_source_threshold: config.memory.promotion_source_threshold,
            };

            info!("Connecting to Qdrant at {url}");
            let store = QdrantMemoryStore::new(url, qdrant_config, embedder)?;
            store.initialize().await?;
            info!("Memory store initialized.");
            Ok(store)
        }
        StoreConfig::Markdown { .. } => Err(Box::new(ConfigError::UnsupportedKind {
            kind: "markdown".into(),
            context: "memory store".into(),
        })),
    }
}

/// Builds an [`OllamaModelProvider`] for text generation using the default model's provider.
fn build_llm_provider(
    config: &AppConfig,
) -> Result<OllamaModelProvider, Box<dyn std::error::Error>> {
    let provider = resolve_llm_provider(config)?;
    build_ollama_provider(provider)
}

/// Resolves the [`ModelProviderConfig`] for the active embedding profile.
fn resolve_embedding_provider(
    config: &AppConfig,
) -> Result<&ModelProviderConfig, Box<dyn std::error::Error>> {
    let profile_name = &config.routing.embedding;
    let profile = config
        .embeddings
        .get(profile_name)
        .ok_or_else(|| ConfigError::MissingKey {
            section: "embeddings".into(),
            key: profile_name.clone(),
        })?;
    config.providers.get(&profile.provider).ok_or_else(|| {
        Box::new(ConfigError::MissingKey {
            section: "providers".into(),
            key: profile.provider.clone(),
        }) as Box<dyn std::error::Error>
    })
}

/// Resolves the [`ModelProviderConfig`] for the default LLM model.
fn resolve_llm_provider(
    config: &AppConfig,
) -> Result<&ModelProviderConfig, Box<dyn std::error::Error>> {
    let model_name = &config.routing.default_model;
    let model = config
        .models
        .get(model_name)
        .ok_or_else(|| ConfigError::MissingKey {
            section: "models".into(),
            key: model_name.clone(),
        })?;
    config.providers.get(&model.provider).ok_or_else(|| {
        Box::new(ConfigError::MissingKey {
            section: "providers".into(),
            key: model.provider.clone(),
        }) as Box<dyn std::error::Error>
    })
}

/// Constructs an [`OllamaModelProvider`] from a provider config, failing if the
/// provider kind is not `ollama`.
fn build_ollama_provider(
    provider: &ModelProviderConfig,
) -> Result<OllamaModelProvider, Box<dyn std::error::Error>> {
    match provider.kind {
        ModelProviderKind::Ollama => {
            let cfg = OllamaConfig {
                base_url: provider.endpoint.clone(),
                timeout: None,
            };
            info!("Using Ollama model provider at {}", provider.endpoint);
            Ok(OllamaModelProvider::new(cfg)?)
        }
        ModelProviderKind::OpenAI => Err(Box::new(ConfigError::UnsupportedKind {
            kind: "openai".into(),
            context: "provider".into(),
        })),
        ModelProviderKind::Anthropic => Err(Box::new(ConfigError::UnsupportedKind {
            kind: "anthropic".into(),
            context: "provider".into(),
        })),
    }
}

fn cmd_config_init<W: std::io::Write>(
    path: &PathBuf,
    out: &mut W,
) -> Result<(), Box<dyn std::error::Error>> {
    if path.exists() {
        return Err(format!(
            "config file already exists at {}; remove it first if you want to regenerate it",
            path.display()
        )
        .into());
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            format!(
                "could not create config directory '{}': {e}",
                parent.display()
            )
        })?;
    }

    std::fs::write(path, DEFAULT_CONFIG_TEMPLATE)
        .map_err(|e| format!("could not write config file '{}': {e}", path.display()))?;

    writeln!(out, "Config file written to {}", path.display())?;
    Ok(())
}

/// Template written by `loci config init`.
const DEFAULT_CONFIG_TEMPLATE: &str = r#"########################################
# Providers (Compute Layer)
########################################

[providers.ollama]
kind = "ollama"
endpoint = "http://localhost:11434"

# Uncomment and fill in to use OpenAI:
# [providers.openai]
# kind = "openai"
# endpoint = "https://api.openai.com/v1"
# api_key = "env:OPENAI_API_KEY"

# Uncomment and fill in to use Anthropic:
# [providers.anthropic]
# kind = "anthropic"
# endpoint = "https://api.anthropic.com"
# api_key = "env:ANTHROPIC_API_KEY"

########################################
# Models (Inference Abstraction)
########################################

[models.default]
provider = "ollama"
name = "qwen3:0.6b"

# Optional model-specific tuning:
# [models.default.tuning]
# temperature = 0.2
# max_tokens = 512
# top_p = 0.95
# repeat_penalty = 1.1
# repeat_last_n = 64
# keep_alive_secs = 300
# stop = ["<END>"]
# [models.default.tuning.thinking]
# mode = "disabled" # or "enabled", "effort", "budgeted"
# level = "low"     # for mode = "effort"
# max_tokens = 256  # for mode = "budgeted"
# [models.default.tuning.extra_params]
# seed = 42

########################################
# Embeddings
########################################

[embeddings.default]
provider = "ollama"
model = "qwen3-embedding:0.6b"
dimension = 768

########################################
# Memory Store (Persistence Layer)
########################################

[stores.qdrant]
kind = "qdrant"
url = "http://localhost:6333"
# api_key = "env:QDRANT_API_KEY"

########################################
# Memory Configuration
########################################

[memory]
store = "qdrant"
collection = "memory_entries"
# similarity_threshold = 0.95        # optional deduplication threshold (0.0–1.0)
# promotion_source_threshold = 2        # Candidate -> Stable after corroboration from N independent sources

########################################
# Routing / Defaults
########################################

[routing]
default_model = "default"
fallback_models = []
embedding = "default"
"#;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolves the config file path: uses the provided value when set,
/// otherwise falls back to `~/.config/loci/config.toml`.
fn resolve_config_path(cli_value: Option<PathBuf>) -> PathBuf {
    if let Some(path) = cli_value {
        return path;
    }
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("loci")
        .join("config.toml")
}

/// Serialises a [`loci_core::memory::MemoryEntry`] to a [`serde_json::Value`].
fn entry_to_json(e: &loci_core::memory::MemoryQueryResult) -> serde_json::Value {
    serde_json::json!({
        "id": e.memory_entry.id.to_string(),
        "content": e.memory_entry.content,
        "metadata": e.memory_entry.metadata,
        "tier": e.memory_entry.tier.as_str(),
        "seen_count": e.memory_entry.seen_count,
        "first_seen": e.memory_entry.first_seen.to_rfc3339(),
        "last_seen": e.memory_entry.last_seen.to_rfc3339(),
        "expires_at": e.memory_entry.expires_at.map(|dt| dt.to_rfc3339()),
        "created_at": e.memory_entry.created_at.to_rfc3339(),
        "score": e.score.value(),
    })
}

fn setup_logging(verbose: bool) {
    #[cfg(feature = "systemd-journal-logger")]
    {
        JournalLog::new()
            .expect("Failed to connect to journald")
            .install()
            .expect("Failed to install journald logger");
    }

    #[cfg(not(feature = "systemd-journal-logger"))]
    {
        env_logger::init();
    }

    log::set_max_level(if verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    });
}

#[cfg(test)]
mod tests {
    use crate::fixture::minimal_ollama_config;

    use super::*;
    use std::collections::HashMap as StdHashMap;

    use pretty_assertions::assert_eq;
    use serde_json::Value as JsonValue;

    use loci_config::{ModelProviderConfig, ModelProviderKind};
    use loci_core::memory::{MemoryEntry, MemoryQueryResult, MemoryTier, Score};

    // ── Group A: cmd_config_init ──────────────────────────────────────────────

    #[test]
    fn test_config_init_writes_file_to_new_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut out = Vec::new();

        cmd_config_init(&path, &mut out).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(
            content.contains("[providers.ollama]"),
            "expected provider section"
        );
        assert!(
            content.contains("[stores.qdrant]"),
            "expected store section"
        );
        assert!(content.contains("[routing]"), "expected routing section");
    }

    #[test]
    fn test_config_init_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("sub").join("config.toml");
        let mut out = Vec::new();

        cmd_config_init(&path, &mut out).unwrap();

        assert!(path.exists(), "config file should have been created");
    }

    #[test]
    fn test_config_init_fails_when_file_already_exists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "existing").unwrap();
        let mut out = Vec::new();

        let err = cmd_config_init(&path, &mut out).unwrap_err();

        assert!(
            err.to_string().contains("already exists"),
            "expected 'already exists' in error, got: {err}"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "existing");
    }

    #[test]
    fn test_config_init_output_reports_config_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut out = Vec::new();

        cmd_config_init(&path, &mut out).unwrap();

        let output = String::from_utf8(out).unwrap();
        assert!(
            output.contains(path.to_str().unwrap()),
            "output should contain the config path, got: {output}"
        );
    }

    // ── Group B: Provider resolution helpers ──────────────────────────────────

    #[test]
    fn test_resolve_embedding_provider_returns_provider_config() {
        let config = minimal_ollama_config();
        let provider = resolve_embedding_provider(&config).unwrap();
        assert_eq!(provider.endpoint, "http://localhost:11434");
        assert_eq!(provider.kind, ModelProviderKind::Ollama);
    }

    #[test]
    fn test_resolve_embedding_provider_missing_embedding_key_returns_err() {
        let mut config = minimal_ollama_config();
        config.routing.embedding = "nonexistent".to_string();

        let err = resolve_embedding_provider(&config).unwrap_err();
        assert!(
            err.to_string().contains("nonexistent"),
            "expected missing key name in error, got: {err}"
        );
    }

    #[test]
    fn test_resolve_embedding_provider_missing_provider_key_returns_err() {
        let mut config = minimal_ollama_config();
        // Point the embedding profile at a provider that doesn't exist.
        config.embeddings.get_mut("default").unwrap().provider = "ghost".to_string();

        let err = resolve_embedding_provider(&config).unwrap_err();
        assert!(
            err.to_string().contains("ghost"),
            "expected missing provider name in error, got: {err}"
        );
    }

    #[test]
    fn test_resolve_llm_provider_returns_provider_config() {
        let config = minimal_ollama_config();
        let provider = resolve_llm_provider(&config).unwrap();
        assert_eq!(provider.endpoint, "http://localhost:11434");
    }

    #[test]
    fn test_resolve_llm_provider_missing_model_returns_err() {
        let mut config = minimal_ollama_config();
        config.routing.default_model = "nonexistent".to_string();

        let err = resolve_llm_provider(&config).unwrap_err();
        assert!(
            err.to_string().contains("nonexistent"),
            "expected missing key name in error, got: {err}"
        );
    }

    #[test]
    fn test_resolve_llm_provider_missing_provider_returns_err() {
        let mut config = minimal_ollama_config();
        config.models.get_mut("default").unwrap().provider = "ghost".to_string();

        let err = resolve_llm_provider(&config).unwrap_err();
        assert!(
            err.to_string().contains("ghost"),
            "expected missing provider name in error, got: {err}"
        );
    }

    #[test]
    fn test_build_ollama_provider_succeeds_for_ollama_kind() {
        let cfg = ModelProviderConfig {
            kind: ModelProviderKind::Ollama,
            endpoint: "http://localhost:11434".to_string(),
            api_key: None,
        };
        assert!(build_ollama_provider(&cfg).is_ok());
    }

    #[test]
    fn test_build_ollama_provider_fails_for_openai_kind() {
        let cfg = ModelProviderConfig {
            kind: ModelProviderKind::OpenAI,
            endpoint: "https://api.openai.com/v1".to_string(),
            api_key: None,
        };
        let err = build_ollama_provider(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("openai"),
            "expected 'openai' in error, got: {err}"
        );
    }

    #[test]
    fn test_build_ollama_provider_fails_for_anthropic_kind() {
        let cfg = ModelProviderConfig {
            kind: ModelProviderKind::Anthropic,
            endpoint: "https://api.anthropic.com".to_string(),
            api_key: None,
        };
        let err = build_ollama_provider(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("anthropic"),
            "expected 'anthropic' in error, got: {err}"
        );
    }

    #[test]
    fn resolve_config_path_uses_cli_value_when_set() {
        let p = resolve_config_path(Some(PathBuf::from("/tmp/my-config.toml")));
        assert_eq!(p, PathBuf::from("/tmp/my-config.toml"));
    }

    #[test]
    fn resolve_config_path_falls_back_to_xdg_when_empty() {
        let p = resolve_config_path(None);
        assert!(p.ends_with("loci/config.toml"));
    }

    #[test]
    fn entry_to_json_serializes_fields() {
        let mut metadata = StdHashMap::new();
        metadata.insert("source".to_string(), "unit-test".to_string());

        let entry = MemoryEntry::new_with_tier(
            "my content".to_string(),
            metadata.clone(),
            MemoryTier::Core,
        );
        let mq = MemoryQueryResult {
            memory_entry: entry.clone(),
            score: Score::new(0.75).unwrap(),
        };

        let v: JsonValue = entry_to_json(&mq);

        assert_eq!(v["id"].as_str().unwrap(), entry.id.to_string().as_str());
        assert_eq!(v["content"].as_str().unwrap(), "my content");
        assert_eq!(v["metadata"]["source"].as_str().unwrap(), "unit-test");
        assert_eq!(v["tier"].as_str().unwrap(), "core");
        assert_eq!(v["seen_count"].as_u64().unwrap(), entry.seen_count as u64);
        assert!(v.get("expires_at").unwrap().is_null());
        assert_eq!(v["score"].as_f64().unwrap(), 0.75);
        assert!(v["created_at"].as_str().is_some());
        assert!(v["first_seen"].as_str().is_some());
        assert!(v["last_seen"].as_str().is_some());
    }
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-cli.

//! Binary entry-point for the `loci` CLI.
//!
//! All infrastructure configuration (providers, stores, models, embeddings) is
//! read from a TOML config file (default: `~/.config/loci/config.toml`).
//! The CLI itself only exposes operational flags and sub-commands.

mod commands;
#[cfg(test)]
mod fixture;
mod handlers;
#[cfg(test)]
mod mock;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "systemd-journal-logger")]
use systemd_journal_logger::JournalLog;

use clap::{Parser, Subcommand};
use log::{LevelFilter, error, info};

use loci_config::{
    AppConfig, ConfigError, ModelProviderConfig, ModelProviderKind, ModelThinkingConfig,
    ModelThinkingEffortLevel, ModelTuningConfig, StoreConfig, load_config,
};
use loci_core::contextualization::{
    Contextualizer, ContextualizerConfig, ContextualizerSystemConfig, ContextualizerSystemMode,
    ContextualizerTuningConfig,
};

use loci_core::embedding::DefaultTextEmbedder;
use loci_core::memory::Score;
use loci_core::model_provider::text_generation::{ThinkingEffortLevel, ThinkingMode};
use loci_memory_store_qdrant::config::QdrantConfig;
use loci_memory_store_qdrant::store::QdrantMemoryStore;
use loci_model_provider_ollama::provider::{OllamaConfig, OllamaModelProvider};

use crate::commands::generate::{GenerateArgs, GenerateDebugFlags, GenerateSystemMode};
use crate::commands::memory::MemoryCommand;
use crate::handlers::CommandHandler;
use crate::handlers::memory::MemoryCommandHandler;

/// Top-level CLI arguments and global options.
#[derive(Parser)]
#[command(name = "loci", about = "loci CLI")]
struct Cli {
    /// Path to the TOML configuration file.
    #[arg(long, short, env = "LOCI_CONFIG", hide_env_values = true)]
    config: Option<PathBuf>,

    /// Verbose output.
    #[arg(long, short)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

/// Available sub-commands.
#[derive(Subcommand)]
enum Command {
    /// Memory store operations.
    Memory {
        #[command(subcommand)]
        command: MemoryCommand,
    },
    /// Generate a response for a prompt, with optional memory retrieval and contextualization.
    Gen {
        #[command(flatten)]
        command: GenerateArgs,
    },
    /// Configuration management.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
}

/// Config sub-commands.
#[derive(Subcommand)]
enum ConfigCommand {
    /// Scaffold a default configuration file at the config path.
    Init,
}

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
        Command::Gen {
            command:
                GenerateArgs {
                    prompt,
                    system,
                    system_mode,
                    max_memory_entries,
                    min_score,
                    memory_mode,
                    debug_flags,
                },
        } => {
            let store = Arc::new(build_store(&config).await?);
            let llm_provider = Arc::new(build_llm_provider(&config)?);
            let model = {
                let model_key = &config.routing.default_model;
                config
                    .models
                    .get(model_key)
                    .ok_or_else(|| ConfigError::MissingKey {
                        section: "models".into(),
                        key: model_key.clone(),
                    })?
                    .clone()
            };
            let min_score = Score::new(min_score).map_err(|e| format!("invalid min_score: {e}"))?;

            let ctx_config = ContextualizerConfig {
                system: system.map(|system| ContextualizerSystemConfig {
                    mode: match system_mode {
                        GenerateSystemMode::Append => ContextualizerSystemMode::Append,
                        GenerateSystemMode::Replace => ContextualizerSystemMode::Replace,
                    },
                    system,
                }),
                max_memory_entries,
                min_score,
                memory_mode: memory_mode.into(),
                filters: HashMap::new(),
                text_generation_model: model.name,
                tuning: model.tuning.as_ref().map(model_tuning_to_contextualizer),
            };

            let contextualizer = Contextualizer::new(store, llm_provider, ctx_config);

            if debug_flags.contains(&GenerateDebugFlags::Memory) {
                let (debug_info, stream) = contextualizer.contextualize_with_debug(&prompt).await?;

                eprintln!("Debug info:\n");
                eprintln!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                          "retrieved_memory": debug_info.memory_entries.iter().map(entry_to_json).collect::<Vec<_>>(),
                    }))?
                );

                println!("\nResponse:\n");

                stream_text_generation(stream, &mut std::io::stdout()).await?;
            } else {
                let stream = contextualizer.contextualize(&prompt).await?;
                stream_text_generation(stream, &mut std::io::stdout()).await?;
            }
            Ok(())
        }
        Command::Config { .. } => unreachable!("handled above"),
    }
}

/// Builds a `QdrantMemoryStore<DefaultTextEmbedder>` from the active config.
///
/// Fails fast with [`ConfigError::UnsupportedKind`] if the configured store
/// or embedding provider is not yet implemented.
async fn build_store(
    config: &AppConfig,
) -> Result<QdrantMemoryStore<DefaultTextEmbedder>, Box<dyn std::error::Error>> {
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

fn model_tuning_to_contextualizer(tuning: &ModelTuningConfig) -> ContextualizerTuningConfig {
    ContextualizerTuningConfig {
        temperature: tuning.temperature,
        max_tokens: tuning.max_tokens,
        top_p: tuning.top_p,
        repeat_penalty: tuning.repeat_penalty,
        repeat_last_n: tuning.repeat_last_n,
        thinking: tuning.thinking.as_ref().map(model_thinking_to_core),
        stop: tuning.stop.clone(),
        keep_alive: tuning.keep_alive_secs.map(std::time::Duration::from_secs),
        extra_params: tuning.extra_params.clone(),
    }
}

fn model_thinking_to_core(thinking: &ModelThinkingConfig) -> ThinkingMode {
    match thinking {
        ModelThinkingConfig::Enabled => ThinkingMode::Enabled,
        ModelThinkingConfig::Disabled => ThinkingMode::Disabled,
        ModelThinkingConfig::Effort { level } => ThinkingMode::Effort {
            level: match level {
                ModelThinkingEffortLevel::Low => ThinkingEffortLevel::Low,
                ModelThinkingEffortLevel::Medium => ThinkingEffortLevel::Medium,
                ModelThinkingEffortLevel::High => ThinkingEffortLevel::High,
            },
        },
        ModelThinkingConfig::Budgeted { max_tokens } => ThinkingMode::Budgeted {
            max_tokens: *max_tokens,
        },
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

/// Consumes a text-generation stream, printing each chunk to stdout.
///
/// A newline is printed after the final chunk (when `chunk.done` is `true`).
async fn stream_text_generation<W: std::io::Write>(
    mut stream: impl futures::Stream<
        Item = Result<
            loci_core::model_provider::text_generation::TextGenerationResponse,
            loci_core::error::ContextualizerError,
        >,
    > + Unpin,
    out: &mut W,
) -> Result<(), Box<dyn std::error::Error>> {
    use futures::StreamExt as _;
    while let Some(result) = stream.next().await {
        let chunk = result.map_err(|e| e.to_string())?;
        write!(out, "{}", chunk.text)?;
        out.flush()?;
        if chunk.done {
            writeln!(out)?;
        }
    }
    Ok(())
}

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
    use crate::{commands::generate::GenerateMemoryMode, fixture::minimal_ollama_config};

    use super::*;
    use std::collections::HashMap as StdHashMap;

    use pretty_assertions::assert_eq;
    use rstest::rstest;
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

    // ── Group B: Conversion helpers ───────────────────────────────────────────

    #[test]
    fn test_gen_memory_mode_auto_converts_to_contextualizer_auto() {
        use loci_core::contextualization::ContextualizationMemoryMode;
        let mode: ContextualizationMemoryMode = GenerateMemoryMode::Auto.into();
        assert_eq!(mode, ContextualizationMemoryMode::Auto);
    }

    #[test]
    fn test_gen_memory_mode_off_converts_to_contextualizer_off() {
        use loci_core::contextualization::ContextualizationMemoryMode;
        let mode: ContextualizationMemoryMode = GenerateMemoryMode::Off.into();
        assert_eq!(mode, ContextualizationMemoryMode::Off);
    }

    #[test]
    fn test_model_tuning_to_contextualizer_maps_all_fields() {
        use loci_config::{ModelThinkingConfig, ModelTuningConfig};
        use loci_core::model_provider::text_generation::ThinkingMode;
        use std::time::Duration;

        let tuning = ModelTuningConfig {
            temperature: Some(0.7),
            max_tokens: Some(512),
            top_p: Some(0.9),
            repeat_penalty: Some(1.1),
            repeat_last_n: Some(64),
            stop: Some(vec!["<END>".to_string()]),
            keep_alive_secs: Some(300),
            thinking: Some(ModelThinkingConfig::Enabled),
            extra_params: StdHashMap::new(),
        };

        let ctx = model_tuning_to_contextualizer(&tuning);

        assert_eq!(ctx.temperature, Some(0.7));
        assert_eq!(ctx.max_tokens, Some(512));
        assert_eq!(ctx.top_p, Some(0.9));
        assert_eq!(ctx.repeat_penalty, Some(1.1));
        assert_eq!(ctx.repeat_last_n, Some(64));
        assert_eq!(ctx.stop.as_deref(), Some(["<END>".to_string()].as_slice()));
        assert_eq!(ctx.keep_alive, Some(Duration::from_secs(300)));
        assert!(matches!(ctx.thinking, Some(ThinkingMode::Enabled)));
    }

    #[test]
    fn test_model_tuning_to_contextualizer_maps_none_fields() {
        use loci_config::ModelTuningConfig;

        let tuning = ModelTuningConfig::default();
        let ctx = model_tuning_to_contextualizer(&tuning);

        assert_eq!(ctx.temperature, None);
        assert_eq!(ctx.max_tokens, None);
        assert_eq!(ctx.top_p, None);
        assert_eq!(ctx.repeat_penalty, None);
        assert_eq!(ctx.repeat_last_n, None);
        assert_eq!(ctx.stop, None);
        assert_eq!(ctx.keep_alive, None);
        assert!(ctx.thinking.is_none());
    }

    #[rstest]
    #[case(loci_config::ModelThinkingConfig::Enabled, "enabled")]
    #[case(loci_config::ModelThinkingConfig::Disabled, "disabled")]
    #[case(
        loci_config::ModelThinkingConfig::Effort { level: loci_config::ModelThinkingEffortLevel::Low },
        "effort_low"
    )]
    #[case(
        loci_config::ModelThinkingConfig::Effort { level: loci_config::ModelThinkingEffortLevel::Medium },
        "effort_medium"
    )]
    #[case(
        loci_config::ModelThinkingConfig::Effort { level: loci_config::ModelThinkingEffortLevel::High },
        "effort_high"
    )]
    #[case(
        loci_config::ModelThinkingConfig::Budgeted { max_tokens: 256 },
        "budgeted"
    )]
    fn test_model_thinking_to_core_all_variants(
        #[case] input: loci_config::ModelThinkingConfig,
        #[case] label: &str,
    ) {
        use loci_core::model_provider::text_generation::{ThinkingEffortLevel, ThinkingMode};

        let result = model_thinking_to_core(&input);
        match (label, &result) {
            ("enabled", ThinkingMode::Enabled) => {}
            ("disabled", ThinkingMode::Disabled) => {}
            (
                "effort_low",
                ThinkingMode::Effort {
                    level: ThinkingEffortLevel::Low,
                },
            ) => {}
            (
                "effort_medium",
                ThinkingMode::Effort {
                    level: ThinkingEffortLevel::Medium,
                },
            ) => {}
            (
                "effort_high",
                ThinkingMode::Effort {
                    level: ThinkingEffortLevel::High,
                },
            ) => {}
            ("budgeted", ThinkingMode::Budgeted { max_tokens: 256 }) => {}
            _ => panic!("unexpected mapping for label '{label}': {result:?}"),
        }
    }

    // ── Group D: Provider resolution helpers ──────────────────────────────────

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

    // ── Group E: stream_text_generation ──────────────────────────────────────

    #[tokio::test]
    async fn test_stream_text_generation_writes_all_chunks() {
        use futures::stream;
        use loci_core::error::ContextualizerError;
        use loci_core::model_provider::text_generation::TextGenerationResponse;

        let chunks: Vec<Result<TextGenerationResponse, ContextualizerError>> = vec![
            Ok(TextGenerationResponse {
                text: "hello ".to_string(),
                model: "m".to_string(),
                usage: None,
                done: false,
            }),
            Ok(TextGenerationResponse {
                text: "world".to_string(),
                model: "m".to_string(),
                usage: None,
                done: true,
            }),
        ];

        let mut out = Vec::new();
        stream_text_generation(stream::iter(chunks), &mut out)
            .await
            .unwrap();

        let output = String::from_utf8(out).unwrap();
        // Text chunks are concatenated; a newline is appended after the done chunk.
        assert!(output.starts_with("hello world"), "got: {output:?}");
        assert!(
            output.ends_with('\n'),
            "final newline missing, got: {output:?}"
        );
    }

    #[tokio::test]
    async fn test_stream_text_generation_propagates_stream_error() {
        use futures::stream;
        use loci_core::error::{ContextualizerError, MemoryStoreError};
        use loci_core::model_provider::text_generation::TextGenerationResponse;

        let chunks: Vec<Result<TextGenerationResponse, ContextualizerError>> = vec![
            Ok(TextGenerationResponse {
                text: "partial".to_string(),
                model: "m".to_string(),
                usage: None,
                done: false,
            }),
            Err(ContextualizerError::MemoryStore(
                MemoryStoreError::Connection("boom".to_string()),
            )),
        ];

        let mut out = Vec::new();
        let result = stream_text_generation(stream::iter(chunks), &mut out).await;

        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("boom"),
            "error message should include the underlying cause"
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

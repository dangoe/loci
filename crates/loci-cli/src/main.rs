// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-cli.

//! Binary entry-point for the `loci` CLI.
//!
//! All infrastructure configuration (providers, stores, models, embeddings) is
//! read from a TOML config file (default: `~/.config/loci/config.toml`).
//! The CLI itself only exposes operational flags and sub-commands.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use futures::StreamExt as _;
use log::{LevelFilter, debug, error, info};
use uuid::Uuid;

use loci_config::{
    AppConfig, ConfigError, ModelProviderConfig, ModelProviderKind, ModelThinkingConfig,
    ModelThinkingEffortLevel, ModelTuningConfig, StoreConfig, load_config,
};
use loci_core::contextualization::{
    Contextualizer, ContextualizerConfig, ContextualizerTuningConfig,
};
use loci_core::embedding::DefaultTextEmbedder;
use loci_core::memory::{MemoryInput, MemoryQuery, MemoryQueryMode, MemoryTier, Score};
use loci_core::model_provider::text_generation::{ThinkingEffortLevel, ThinkingMode};
use loci_core::store::MemoryStore;
use loci_memory_store_qdrant::config::QdrantConfig;
use loci_memory_store_qdrant::store::QdrantMemoryStore;
use loci_model_provider_ollama::provider::{OllamaConfig, OllamaModelProvider};

/// Top-level CLI arguments and global options.
#[derive(Parser)]
#[command(name = "loci", about = "loci CLI")]
struct Cli {
    /// Path to the TOML configuration file.
    #[arg(
        long,
        short,
        default_value = "",
        env = "LOCI_CONFIG",
        hide_default_value = true
    )]
    config: String,

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
    /// Enhance a prompt with memory context and call an LLM.
    Prompt {
        /// The prompt to process.
        prompt: String,

        /// Maximum number of memories to inject into the prompt.
        #[arg(long, default_value_t = 5)]
        max_memories: usize,

        /// Minimum similarity score for memory retrieval (0.0–1.0).
        #[arg(long, default_value_t = 0.5)]
        min_score: f64,
    },
    /// Configuration management.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
}

/// Memory sub-commands.
#[derive(Subcommand)]
enum MemoryCommand {
    /// Save a new memory entry.
    Save {
        /// Memory content.
        #[arg(long)]
        content: String,
        /// Metadata as key=value pairs (repeatable).
        #[arg(long = "meta", value_parser = parse_key_value)]
        metadata: Vec<(String, String)>,
        /// Optional tier (candidate|stable|core).
        #[arg(long, value_parser = parse_memory_tier)]
        tier: Option<MemoryTier>,
    },
    /// Query memory entries by semantic similarity.
    Query {
        /// Topic to search for.
        #[arg(long)]
        topic: String,
        /// Maximum number of results.
        #[arg(long, default_value_t = 10)]
        max_results: usize,
        /// Minimum similarity score (0.0–1.0).
        #[arg(long, default_value_t = 0.0)]
        min_score: f64,
        /// Filter by metadata key=value pairs (repeatable).
        #[arg(long = "filter", value_parser = parse_key_value)]
        filters: Vec<(String, String)>,
        /// Query mode: lookup or use (reserved for downstream behavior control).
        #[arg(long, default_value = "lookup", value_parser = parse_query_mode)]
        mode: MemoryQueryMode,
    },
    /// Update an existing memory entry by ID.
    Update {
        /// Memory entry ID.
        #[arg(long)]
        id: Uuid,
        /// New content.
        #[arg(long)]
        content: String,
        /// New metadata as key=value pairs (repeatable).
        #[arg(long = "meta", value_parser = parse_key_value)]
        metadata: Vec<(String, String)>,
    },
    /// Delete a memory entry by ID.
    Delete {
        /// Memory entry ID.
        #[arg(long)]
        id: Uuid,
    },
    /// Set the tier of an existing memory entry.
    SetTier {
        /// Memory entry ID.
        #[arg(long)]
        id: Uuid,
        /// Tier (candidate|stable|core).
        #[arg(long, value_parser = parse_memory_tier)]
        tier: MemoryTier,
    },
    /// Clear all memories from the collection.
    Clear,
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
    let config_path = resolve_config_path(&cli.config);

    // The `config init` command must be handled before loading the config file,
    // as the file may not exist yet.
    if let Command::Config {
        command: ConfigCommand::Init,
    } = &cli.command
    {
        return cmd_config_init(&config_path);
    }

    info!("Loading config from {}", config_path.display());
    let config = load_config(&config_path)?;

    match cli.command {
        Command::Memory { command } => {
            let store = build_store(&config).await?;
            run_memory_command(store, command).await
        }
        Command::Prompt {
            prompt,
            max_memories,
            min_score,
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
                max_memories,
                min_score,
                filters: HashMap::new(),
                text_generation_model: model.name,
                tuning: model.tuning.as_ref().map(model_tuning_to_contextualizer),
            };

            let contextualizer = Contextualizer::new(store, llm_provider, ctx_config);
            let mut stream = contextualizer.contextualize(&prompt);
            while let Some(result) = stream.next().await {
                let chunk = result.map_err(|e| e.to_string())?;
                print!("{}", chunk.text);
                std::io::stdout().flush()?;
                if chunk.done {
                    println!();
                }
            }
            Ok(())
        }
        Command::Config { .. } => unreachable!("handled above"),
    }
}

// ---------------------------------------------------------------------------
// Store / model provider construction
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Memory command dispatch
// ---------------------------------------------------------------------------

async fn run_memory_command(
    store: QdrantMemoryStore<DefaultTextEmbedder>,
    command: MemoryCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        MemoryCommand::Save {
            content,
            metadata,
            tier,
        } => {
            debug!(
                "save memory entry: content={content}, metadata={:?}, tier={:?}",
                metadata, tier
            );
            let input = match tier {
                Some(tier) => MemoryInput::new_with_tier(content, pairs_to_map(metadata), tier),
                None => MemoryInput::new(content, pairs_to_map(metadata)),
            };
            let entry = store.save(input).await?;
            println!("{}", serde_json::to_string_pretty(&entry_to_json(&entry))?);
        }
        MemoryCommand::Query {
            topic,
            max_results,
            min_score,
            filters,
            mode,
        } => {
            debug!(
                "query memory: topic={topic}, max_results={max_results}, min_score={min_score}, filters={:?}, mode={:?}",
                filters, mode
            );
            let query = MemoryQuery {
                topic,
                max_results,
                min_score: Score::new(min_score).map_err(|e| format!("invalid min_score: {e}"))?,
                filters: pairs_to_map(filters),
                mode,
            };
            let entries = store.query(query).await?;
            let json: Vec<_> = entries.iter().map(entry_to_json).collect();
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        MemoryCommand::Update {
            id,
            content,
            metadata,
        } => {
            debug!(
                "update memory entry: id={id}, content={content}, metadata={:?}",
                metadata
            );
            let input = MemoryInput::new(content, pairs_to_map(metadata));
            let entry = store.update(id, input).await?;
            println!("{}", serde_json::to_string_pretty(&entry_to_json(&entry))?);
        }
        MemoryCommand::Delete { id } => {
            debug!("delete memory entry: id={id}");
            store.delete(id).await?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({ "deleted": id.to_string() }))?
            );
        }
        MemoryCommand::SetTier { id, tier } => {
            debug!("set memory tier: id={id}, tier={:?}", tier);
            let entry = store.set_tier(id, tier).await?;
            println!("{}", serde_json::to_string_pretty(&entry_to_json(&entry))?);
        }
        MemoryCommand::Clear => {
            debug!("clear memory");
            store.clear().await?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({ "cleared": true }))?
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Config init
// ---------------------------------------------------------------------------

fn cmd_config_init(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
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

    println!("Config file written to {}", path.display());
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
collection = "memories"
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

/// Resolves the config file path: uses the provided value when non-empty,
/// otherwise falls back to `~/.config/loci/config.toml`.
fn resolve_config_path(cli_value: &str) -> PathBuf {
    if !cli_value.is_empty() {
        return PathBuf::from(cli_value);
    }
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("loci")
        .join("config.toml")
}

/// Parses a `key=value` string into a `(String, String)` pair.
fn parse_key_value(s: &str) -> Result<(String, String), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("expected KEY=VALUE, got: {s:?}"))?;
    Ok((k.to_string(), v.to_string()))
}

fn parse_memory_tier(s: &str) -> Result<MemoryTier, String> {
    let tier = MemoryTier::parse(s)
        .ok_or_else(|| format!("invalid tier {s:?}; expected one of: candidate, stable, core"))?;

    if tier == MemoryTier::Ephemeral {
        return Err("ephemeral tier is request-scoped and cannot be persisted".to_string());
    }

    Ok(tier)
}

fn parse_query_mode(s: &str) -> Result<MemoryQueryMode, String> {
    match s {
        "lookup" => Ok(MemoryQueryMode::Lookup),
        "use" => Ok(MemoryQueryMode::Use),
        _ => Err(format!(
            "invalid query mode {s:?}; expected one of: lookup, use"
        )),
    }
}

/// Converts a list of `(key, value)` pairs into a [`HashMap`].
fn pairs_to_map(pairs: Vec<(String, String)>) -> HashMap<String, String> {
    pairs.into_iter().collect()
}

/// Serialises a [`loci_core::memory::MemoryEntry`] to a [`serde_json::Value`].
fn entry_to_json(e: &loci_core::memory::MemoryEntry) -> serde_json::Value {
    serde_json::json!({
        "id": e.memory.id.to_string(),
        "content": e.memory.content,
        "metadata": e.memory.metadata,
        "tier": e.memory.tier.as_str(),
        "seen_count": e.memory.seen_count,
        "first_seen": e.memory.first_seen.to_rfc3339(),
        "last_seen": e.memory.last_seen.to_rfc3339(),
        "expires_at": e.memory.expires_at.map(|dt| dt.to_rfc3339()),
        "created_at": e.memory.created_at.to_rfc3339(),
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
    use super::*;

    #[test]
    fn parse_key_value_valid() {
        let (k, v) = parse_key_value("project=ai-memory").unwrap();
        assert_eq!(k, "project");
        assert_eq!(v, "ai-memory");
    }

    #[test]
    fn parse_key_value_when_value_contains_equals() {
        let (k, v) = parse_key_value("url=http://host:1234").unwrap();
        assert_eq!(k, "url");
        assert_eq!(v, "http://host:1234");
    }

    #[test]
    fn parse_key_value_missing_equals_returns_err() {
        assert!(parse_key_value("noequalssign").is_err());
    }

    #[test]
    fn resolve_config_path_uses_cli_value_when_set() {
        let p = resolve_config_path("/tmp/my-config.toml");
        assert_eq!(p, PathBuf::from("/tmp/my-config.toml"));
    }

    #[test]
    fn resolve_config_path_falls_back_to_xdg_when_empty() {
        let p = resolve_config_path("");
        assert!(p.ends_with("loci/config.toml"));
    }

    #[test]
    fn parse_memory_tier_rejects_ephemeral() {
        assert!(parse_memory_tier("ephemeral").is_err());
    }

    #[test]
    fn parse_query_mode_accepts_known_values() {
        assert_eq!(parse_query_mode("lookup").unwrap(), MemoryQueryMode::Lookup);
        assert_eq!(parse_query_mode("use").unwrap(), MemoryQueryMode::Use);
    }
}

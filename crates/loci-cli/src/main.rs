// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-cli.

//! Binary entry-point for the `loci` CLI.
//!
//! All infrastructure configuration (providers, stores, models, embeddings) is
//! read from a TOML config file (default: `~/.config/loci/config.toml`).
//! The CLI itself only exposes operational flags and sub-commands.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "systemd-journal-logger")]
use systemd_journal_logger::JournalLog;

use clap::{Parser, Subcommand};
use log::{LevelFilter, debug, error, info};
use uuid::Uuid;

use loci_config::{
    AppConfig, ConfigError, ModelProviderConfig, ModelProviderKind, ModelThinkingConfig,
    ModelThinkingEffortLevel, ModelTuningConfig, StoreConfig, load_config,
};
use loci_core::contextualization::{
    ContextualizationMemoryMode, Contextualizer, ContextualizerConfig, ContextualizerSystemConfig,
    ContextualizerSystemMode, ContextualizerTuningConfig,
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
    #[arg(long, short, env = "LOCI_CONFIG", hide_env_values = true)]
    config: Option<PathBuf>,

    /// Verbose output.
    #[arg(long, short)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

/// Memory mode for the `gen` command, controlling memory retrieval and injection behavior.
#[derive(clap::ValueEnum, PartialEq, Eq, Clone, Debug)]
enum GenMemoryMode {
    /// Retrieves and injects memory entries into the prompt based on the configured contextualization settings.
    Auto,
    /// Skips memory retrieval and injection, generating a response based solely on the prompt.
    Off,
}

impl Into<ContextualizationMemoryMode> for GenMemoryMode {
    fn into(self) -> ContextualizationMemoryMode {
        match self {
            GenMemoryMode::Auto => ContextualizationMemoryMode::Auto,
            GenMemoryMode::Off => ContextualizationMemoryMode::Off,
        }
    }
}

/// Debug flags for the `gen` command, which prints additional info about the contextualization process when set.
#[derive(clap::ValueEnum, PartialEq, Eq, Clone, Debug)]
enum GenDebugFlags {
    /// Print the memory entries that were injected into the model provider prompt.
    Memory,
}

/// Memory mode for generation, which controls whether and how memory is retrieved and injected into the prompt.
#[derive(clap::ValueEnum, PartialEq, Eq, Clone, Debug)]
enum GenSystemMode {
    /// Append given system prompt to the default system prompt.
    Append,
    /// Replace default system prompt with the given system prompt.
    Replace,
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
        /// The prompt to process.
        prompt: String,

        /// Optional override for the system prompt used for generation. If not set, the default system prompt will be used.
        #[arg(long)]
        system: Option<String>,

        /// System prompt mode, which controls how the provided system prompt interacts with the default system prompt.
        #[arg(long, value_enum, default_value_t = GenSystemMode::Append)]
        system_mode: GenSystemMode,

        /// Maximum number of memory entries to inject into the prompt.
        #[arg(long, default_value_t = 5)]
        max_memory_entries: usize,

        /// Minimum similarity score for memory retrieval (0.0–1.0).
        #[arg(long, default_value_t = 0.5)]
        min_score: f64,

        /// Memory mode for generation, which controls whether and how memory is retrieved and injected into the prompt.
        #[arg(long, value_enum, default_value_t = GenMemoryMode::Auto)]
        memory_mode: GenMemoryMode,

        /// Print debug info about the contextualization process, such as retrieved memory entries.
        #[arg(long)]
        debug_flags: Vec<GenDebugFlags>,
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
    },
    /// Get a memory entry by ID.
    Get {
        /// Memory entry ID.
        id: Uuid,
    },
    /// Update an existing memory entry by ID.
    Update {
        /// Memory entry ID.
        id: Uuid,
        /// New content (optional).
        content: Option<String>,
        /// New metadata as key=value pairs (repeatable).
        #[arg(long = "meta", value_parser = parse_key_value)]
        metadata: Vec<(String, String)>,
        /// Optional tier override (candidate|stable|core).
        #[arg(long, value_parser = parse_memory_tier)]
        tier: Option<MemoryTier>,
    },
    /// Delete a memory entry by ID.
    Delete {
        /// Memory entry ID.
        id: Uuid,
    },
    /// Prunes all expired memory entries from the collection.
    PruneExpired,
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
            run_memory_command(store, command, &mut std::io::stdout()).await
        }
        Command::Gen {
            prompt,
            system,
            system_mode,
            max_memory_entries,
            min_score,
            memory_mode,
            debug_flags,
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
                        GenSystemMode::Append => ContextualizerSystemMode::Append,
                        GenSystemMode::Replace => ContextualizerSystemMode::Replace,
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

            if debug_flags.contains(&GenDebugFlags::Memory) {
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

async fn run_memory_command<S: MemoryStore, W: std::io::Write>(
    store: S,
    command: MemoryCommand,
    out: &mut W,
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
            writeln!(out, "{}", serde_json::to_string_pretty(&entry_to_json(&entry))?)?;
        }
        MemoryCommand::Query {
            topic,
            max_results,
            min_score,
            filters,
        } => {
            debug!(
                "query memory: topic={topic}, max_results={max_results}, min_score={min_score}, filters={:?}",
                filters
            );
            let query = MemoryQuery {
                topic,
                max_results,
                min_score: Score::new(min_score).map_err(|e| format!("invalid min_score: {e}"))?,
                filters: pairs_to_map(filters),
                mode: MemoryQueryMode::Lookup,
            };
            let entries = store.query(query).await?;
            let json: Vec<_> = entries.iter().map(entry_to_json).collect();
            writeln!(out, "{}", serde_json::to_string_pretty(&json)?)?;
        }
        MemoryCommand::Get { id } => {
            debug!("get memory entry: id={id}");
            let entry = store.get(id).await?;
            writeln!(out, "{}", serde_json::to_string_pretty(&entry_to_json(&entry))?)?;
        }
        MemoryCommand::Update {
            id,
            content,
            metadata,
            tier,
        } => {
            debug!(
                "update memory entry: id={id}, content={:?}, metadata={:?}, tier={:?}",
                content, metadata, tier
            );
            if content.is_none() && metadata.is_empty() && tier.is_none() {
                return Err(
                    "nothing to update; provide content, --meta, and/or --tier to change a memory"
                        .into(),
                );
            }

            let existing = store.get(id).await?;
            let content = content.unwrap_or(existing.memory_entry.content);
            let metadata = if metadata.is_empty() {
                existing.memory_entry.metadata
            } else {
                pairs_to_map(metadata)
            };
            let tier = tier.unwrap_or(existing.memory_entry.tier);

            let input = MemoryInput::new_with_tier(content, metadata, tier);
            let entry = store.update(id, input).await?;
            writeln!(out, "{}", serde_json::to_string_pretty(&entry_to_json(&entry))?)?;
        }
        MemoryCommand::Delete { id } => {
            debug!("delete memory entry: id={id}");
            store.delete(id).await?;
            writeln!(
                out,
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({ "deleted": id.to_string() }))?
            )?;
        }
        MemoryCommand::PruneExpired => {
            debug!("clear memory");
            store.prune_expired().await?;
            writeln!(
                out,
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({ "expired pruned": true }))?
            )?;
        }
    }
    Ok(())
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

/// Converts a list of `(key, value)` pairs into a [`HashMap`].
fn pairs_to_map(pairs: Vec<(String, String)>) -> HashMap<String, String> {
    pairs.into_iter().collect()
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
    use super::*;
    use std::collections::HashMap as StdHashMap;
    use std::future::Future;
    use std::sync::{Arc, Mutex};

    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use serde_json::Value as JsonValue;
    use uuid::Uuid;

    use loci_config::{
        AppConfig, EmbeddingProfileConfig, MemoryConfig, ModelConfig, ModelProviderConfig,
        ModelProviderKind, RoutingConfig, StoreConfig,
    };
    use loci_core::error::MemoryStoreError;
    use loci_core::memory::{
        MemoryEntry, MemoryInput, MemoryQuery, MemoryQueryResult, MemoryTier, Score,
    };

    // ── Mock store ────────────────────────────────────────────────────────────

    /// A configurable in-memory store for unit tests.
    ///
    /// Each operation returns a preset response. Operations not configured
    /// fall back to a sensible error (`NotFound` for reads, `Connection` for writes).
    struct MockStore {
        save_entry: Option<MemoryQueryResult>,
        get_entry: Option<MemoryQueryResult>,
        query_entries: Vec<MemoryQueryResult>,
        update_entry: Option<MemoryQueryResult>,
        /// Captures the last `MemoryInput` passed to `update()` for assertion.
        captured_update_input: Arc<Mutex<Option<MemoryInput>>>,
    }

    impl MockStore {
        fn new() -> Self {
            Self {
                save_entry: None,
                get_entry: None,
                query_entries: vec![],
                update_entry: None,
                captured_update_input: Arc::new(Mutex::new(None)),
            }
        }

        fn with_save(mut self, entry: MemoryQueryResult) -> Self {
            self.save_entry = Some(entry);
            self
        }

        fn with_get(mut self, entry: MemoryQueryResult) -> Self {
            self.get_entry = Some(entry);
            self
        }

        fn with_query(mut self, entries: Vec<MemoryQueryResult>) -> Self {
            self.query_entries = entries;
            self
        }

        fn with_update(mut self, entry: MemoryQueryResult) -> Self {
            self.update_entry = Some(entry);
            self
        }
    }

    impl loci_core::store::MemoryStore for MockStore {
        fn save(
            &self,
            _input: MemoryInput,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            let result = self
                .save_entry
                .clone()
                .ok_or_else(|| MemoryStoreError::Connection("mock: save not configured".into()));
            async move { result }
        }

        fn get(
            &self,
            id: Uuid,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            let result = self
                .get_entry
                .clone()
                .ok_or_else(move || MemoryStoreError::NotFound(id));
            async move { result }
        }

        fn query(
            &self,
            _query: MemoryQuery,
        ) -> impl Future<Output = Result<Vec<MemoryQueryResult>, MemoryStoreError>> + Send + '_
        {
            let entries = self.query_entries.clone();
            async move { Ok(entries) }
        }

        fn update(
            &self,
            id: Uuid,
            input: MemoryInput,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            *self.captured_update_input.lock().unwrap() = Some(input);
            let result = self
                .update_entry
                .clone()
                .ok_or_else(move || MemoryStoreError::NotFound(id));
            async move { result }
        }

        fn set_tier(
            &self,
            id: Uuid,
            _tier: MemoryTier,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::NotFound(id)) }
        }

        fn delete(
            &self,
            _id: Uuid,
        ) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_ {
            async move { Ok(()) }
        }

        fn prune_expired(
            &self,
        ) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_ {
            async move { Ok(()) }
        }
    }

    // ── Test fixtures ─────────────────────────────────────────────────────────

    fn make_result(content: &str, tier: MemoryTier) -> MemoryQueryResult {
        MemoryQueryResult {
            memory_entry: MemoryEntry::new_with_tier(
                content.to_string(),
                StdHashMap::new(),
                tier,
            ),
            score: Score::ZERO,
        }
    }

    fn make_result_with_metadata(
        content: &str,
        metadata: StdHashMap<String, String>,
    ) -> MemoryQueryResult {
        MemoryQueryResult {
            memory_entry: MemoryEntry::new(content.to_string(), metadata),
            score: Score::new(0.9).unwrap(),
        }
    }

    /// Builds a minimal `AppConfig` wired to a single Ollama provider.
    fn minimal_ollama_config() -> AppConfig {
        AppConfig {
            providers: StdHashMap::from([(
                "ollama".to_string(),
                ModelProviderConfig {
                    kind: ModelProviderKind::Ollama,
                    endpoint: "http://localhost:11434".to_string(),
                    api_key: None,
                },
            )]),
            models: StdHashMap::from([(
                "default".to_string(),
                ModelConfig {
                    provider: "ollama".to_string(),
                    name: "qwen3:0.6b".to_string(),
                    tuning: None,
                },
            )]),
            embeddings: StdHashMap::from([(
                "default".to_string(),
                EmbeddingProfileConfig {
                    provider: "ollama".to_string(),
                    model: "qwen3-embedding:0.6b".to_string(),
                    dimension: 768,
                },
            )]),
            stores: StdHashMap::from([(
                "qdrant".to_string(),
                StoreConfig::Qdrant {
                    url: "http://localhost:6333".to_string(),
                    api_key: None,
                },
            )]),
            memory: MemoryConfig {
                store: "qdrant".to_string(),
                collection: "memory_entries".to_string(),
                similarity_threshold: None,
                promotion_source_threshold: 2,
            },
            routing: RoutingConfig {
                default_model: "default".to_string(),
                fallback_models: vec![],
                embedding: "default".to_string(),
            },
        }
    }

    fn parse_json_output(buf: &[u8]) -> JsonValue {
        serde_json::from_str(std::str::from_utf8(buf).unwrap().trim()).unwrap()
    }

    // ── Group A: cmd_config_init ──────────────────────────────────────────────

    #[test]
    fn test_config_init_writes_file_to_new_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut out = Vec::new();

        cmd_config_init(&path, &mut out).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("[providers.ollama]"), "expected provider section");
        assert!(content.contains("[stores.qdrant]"), "expected store section");
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

    // ── Group B: run_memory_command ───────────────────────────────────────────

    #[tokio::test]
    async fn test_memory_save_outputs_json() {
        let entry = make_result("hello world", MemoryTier::Candidate);
        let id = entry.memory_entry.id;
        let store = MockStore::new().with_save(entry);
        let mut out = Vec::new();

        run_memory_command(
            store,
            MemoryCommand::Save {
                content: "hello world".to_string(),
                metadata: vec![],
                tier: None,
            },
            &mut out,
        )
        .await
        .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["id"].as_str().unwrap(), id.to_string().as_str());
        assert_eq!(v["content"].as_str().unwrap(), "hello world");
        assert_eq!(v["tier"].as_str().unwrap(), "candidate");
    }

    #[tokio::test]
    async fn test_memory_save_with_tier_outputs_tier_field() {
        let entry = make_result("core fact", MemoryTier::Core);
        let store = MockStore::new().with_save(entry);
        let mut out = Vec::new();

        run_memory_command(
            store,
            MemoryCommand::Save {
                content: "core fact".to_string(),
                metadata: vec![],
                tier: Some(MemoryTier::Core),
            },
            &mut out,
        )
        .await
        .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["tier"].as_str().unwrap(), "core");
    }

    #[tokio::test]
    async fn test_memory_save_propagates_store_error() {
        let store = MockStore::new(); // save_entry = None → Connection error
        let mut out = Vec::new();

        let result = run_memory_command(
            store,
            MemoryCommand::Save {
                content: "x".to_string(),
                metadata: vec![],
                tier: None,
            },
            &mut out,
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_memory_query_outputs_json_array() {
        let entries = vec![
            make_result("first", MemoryTier::Stable),
            make_result("second", MemoryTier::Core),
        ];
        let store = MockStore::new().with_query(entries);
        let mut out = Vec::new();

        run_memory_command(
            store,
            MemoryCommand::Query {
                topic: "something".to_string(),
                max_results: 10,
                min_score: 0.0,
                filters: vec![],
            },
            &mut out,
        )
        .await
        .unwrap();

        let v = parse_json_output(&out);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["content"].as_str().unwrap(), "first");
        assert_eq!(arr[1]["content"].as_str().unwrap(), "second");
    }

    #[tokio::test]
    async fn test_memory_query_invalid_min_score_returns_err() {
        let store = MockStore::new();
        let mut out = Vec::new();

        let result = run_memory_command(
            store,
            MemoryCommand::Query {
                topic: "t".to_string(),
                max_results: 5,
                min_score: 1.5, // invalid — outside [0.0, 1.0]
                filters: vec![],
            },
            &mut out,
        )
        .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid min_score"));
    }

    #[tokio::test]
    async fn test_memory_get_outputs_json() {
        let entry = make_result("specific entry", MemoryTier::Stable);
        let id = entry.memory_entry.id;
        let store = MockStore::new().with_get(entry);
        let mut out = Vec::new();

        run_memory_command(store, MemoryCommand::Get { id }, &mut out)
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["id"].as_str().unwrap(), id.to_string().as_str());
        assert_eq!(v["content"].as_str().unwrap(), "specific entry");
        assert_eq!(v["tier"].as_str().unwrap(), "stable");
    }

    #[tokio::test]
    async fn test_memory_get_not_found_returns_err() {
        let store = MockStore::new(); // get_entry = None → NotFound
        let mut out = Vec::new();

        let result = run_memory_command(
            store,
            MemoryCommand::Get { id: Uuid::new_v4() },
            &mut out,
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_memory_update_happy_path() {
        let id = Uuid::new_v4();
        let existing = MemoryQueryResult {
            memory_entry: MemoryEntry::new_with_tier(
                "old".to_string(),
                StdHashMap::new(),
                MemoryTier::Candidate,
            ),
            score: Score::ZERO,
        };
        let updated = make_result("new content", MemoryTier::Stable);
        let updated_id = updated.memory_entry.id;
        let store = MockStore::new().with_get(existing).with_update(updated);
        let mut out = Vec::new();

        run_memory_command(
            store,
            MemoryCommand::Update {
                id,
                content: Some("new content".to_string()),
                metadata: vec![],
                tier: Some(MemoryTier::Stable),
            },
            &mut out,
        )
        .await
        .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["id"].as_str().unwrap(), updated_id.to_string().as_str());
        assert_eq!(v["content"].as_str().unwrap(), "new content");
    }

    #[tokio::test]
    async fn test_memory_update_nothing_to_update_returns_err() {
        let store = MockStore::new();
        let mut out = Vec::new();

        let result = run_memory_command(
            store,
            MemoryCommand::Update {
                id: Uuid::new_v4(),
                content: None,
                metadata: vec![],
                tier: None,
            },
            &mut out,
        )
        .await;

        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("nothing to update"),
            "expected 'nothing to update' error message"
        );
    }

    #[tokio::test]
    async fn test_memory_update_preserves_existing_metadata_when_not_provided() {
        let id = Uuid::new_v4();
        let original_meta =
            StdHashMap::from([("source".to_string(), "original-source".to_string())]);
        let existing = make_result_with_metadata("original", original_meta.clone());
        let updated = make_result_with_metadata("updated", original_meta.clone());
        let captured = Arc::new(Mutex::new(None::<MemoryInput>));
        let store = MockStore {
            save_entry: None,
            get_entry: Some(existing),
            query_entries: vec![],
            update_entry: Some(updated),
            captured_update_input: captured.clone(),
        };
        let mut out = Vec::new();

        run_memory_command(
            store,
            MemoryCommand::Update {
                id,
                content: Some("updated".to_string()),
                metadata: vec![], // no --meta flag → should preserve existing
                tier: None,
            },
            &mut out,
        )
        .await
        .unwrap();

        let input = captured.lock().unwrap().take().unwrap();
        assert_eq!(
            input.metadata.get("source").unwrap(),
            "original-source",
            "existing metadata should be preserved when no --meta flags are given"
        );
    }

    #[tokio::test]
    async fn test_memory_delete_outputs_deleted_id() {
        let id = Uuid::new_v4();
        let store = MockStore::new();
        let mut out = Vec::new();

        run_memory_command(store, MemoryCommand::Delete { id }, &mut out)
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["deleted"].as_str().unwrap(), id.to_string().as_str());
    }

    #[tokio::test]
    async fn test_memory_prune_expired_outputs_success() {
        let store = MockStore::new();
        let mut out = Vec::new();

        run_memory_command(store, MemoryCommand::PruneExpired, &mut out)
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["expired pruned"].as_bool().unwrap(), true);
    }

    // ── Group C: Conversion helpers ───────────────────────────────────────────

    #[test]
    fn test_gen_memory_mode_auto_converts_to_contextualizer_auto() {
        use loci_core::contextualization::ContextualizationMemoryMode;
        let mode: ContextualizationMemoryMode = GenMemoryMode::Auto.into();
        assert_eq!(mode, ContextualizationMemoryMode::Auto);
    }

    #[test]
    fn test_gen_memory_mode_off_converts_to_contextualizer_off() {
        use loci_core::contextualization::ContextualizationMemoryMode;
        let mode: ContextualizationMemoryMode = GenMemoryMode::Off.into();
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
    #[case(
        loci_config::ModelThinkingConfig::Enabled,
        "enabled"
    )]
    #[case(
        loci_config::ModelThinkingConfig::Disabled,
        "disabled"
    )]
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
            ("effort_low", ThinkingMode::Effort { level: ThinkingEffortLevel::Low }) => {}
            ("effort_medium", ThinkingMode::Effort { level: ThinkingEffortLevel::Medium }) => {}
            ("effort_high", ThinkingMode::Effort { level: ThinkingEffortLevel::High }) => {}
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
        config
            .embeddings
            .get_mut("default")
            .unwrap()
            .provider = "ghost".to_string();

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
        assert!(output.ends_with('\n'), "final newline missing, got: {output:?}");
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

    // ── (existing helper tests) ───────────────────────────────────────────────

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
    fn parse_key_value_empty_value_is_allowed() {
        let (k, v) = parse_key_value("key=").unwrap();
        assert_eq!(k, "key");
        assert_eq!(v, "");
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
    fn parse_memory_tier_rejects_ephemeral() {
        assert!(parse_memory_tier("ephemeral").is_err());
    }

    #[test]
    fn parse_memory_tier_accepts_known_tiers() {
        assert_eq!(
            parse_memory_tier("candidate").unwrap(),
            MemoryTier::Candidate
        );
        assert_eq!(parse_memory_tier("stable").unwrap(), MemoryTier::Stable);
        assert_eq!(parse_memory_tier("core").unwrap(), MemoryTier::Core);
        assert!(parse_memory_tier("unknown").is_err());
    }

    #[test]
    fn pairs_to_map_converts_pairs() {
        let pairs = vec![
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "two".to_string()),
        ];
        let map = pairs_to_map(pairs);
        assert_eq!(map.get("a").unwrap(), "1");
        assert_eq!(map.get("b").unwrap(), "two");
        assert_eq!(map.len(), 2);
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

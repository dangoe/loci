// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-cli.

//! Binary entry-point for the `ai-memory` CLI.
//!
//! Exposes all [`MemoryStore`] operations through a `clap`-derived command
//! hierarchy. Global options accept environment-variable fallbacks so the CLI
//! works naturally in Docker / CI environments.

use std::collections::HashMap;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use uuid::Uuid;

use loci_backend_ollama::backend::{OllamaBackend, OllamaConfig};
use loci_core::contextualization::{Contextualizer, ContextualizerConfig};
use loci_core::embedding::DefaultTextEmbedder;
use loci_core::memory::{MemoryEntry, MemoryInput, MemoryQuery, Score};
use loci_core::store::MemoryStore;
use loci_memory_store_qdrant::config::QdrantConfig;
use loci_memory_store_qdrant::store::QdrantMemoryStore;

/// Embedding model pulled by `ollama-init` in docker-compose.
const EMBED_MODEL: &str = "nomic-embed-text";
/// Output dimension of `nomic-embed-text`.
const EMBED_DIM: usize = 768;
/// Default LLM model pulled by `ollama-init` in docker-compose.
const DEFAULT_LLM_MODEL: &str = "qwen3:0.6b";

/// Top-level CLI arguments and global options.
#[derive(Parser)]
#[command(name = "ai-memory", about = "AI memory store CLI")]
struct Cli {
    /// Qdrant gRPC URL
    #[arg(long, default_value = "http://localhost:6334", env = "QDRANT_URL")]
    qdrant_url: String,

    /// Ollama base URL
    #[arg(long, default_value = "http://localhost:11434", env = "OLLAMA_URL")]
    ollama_url: String,

    /// Qdrant collection name
    #[arg(long, default_value = "memories", env = "COLLECTION_NAME")]
    collection: String,

    /// Cosine similarity threshold for deduplication (0.0–1.0)
    #[arg(long, env = "SIMILARITY_THRESHOLD")]
    similarity_threshold: Option<f64>,

    #[command(subcommand)]
    command: Command,
}

/// Available sub-commands.
#[derive(Subcommand)]
enum Command {
    /// Save a new memory
    Save {
        /// Memory content
        #[arg(long)]
        content: String,
        /// Metadata as key=value pairs (repeatable)
        #[arg(long = "meta", value_parser = parse_key_value)]
        metadata: Vec<(String, String)>,
    },
    /// Query memories by semantic similarity
    Query {
        /// Topic to search for
        #[arg(long)]
        topic: String,
        /// Maximum number of results
        #[arg(long, default_value_t = 10)]
        max_results: usize,
        /// Minimum similarity score (0.0–1.0)
        #[arg(long, default_value_t = 0.0)]
        min_score: f64,
        /// Filter by metadata key=value pairs (repeatable)
        #[arg(long = "filter", value_parser = parse_key_value)]
        filters: Vec<(String, String)>,
    },
    /// Update an existing memory by ID
    Update {
        /// Memory ID (UUID)
        #[arg(long)]
        id: Uuid,
        /// New content
        #[arg(long)]
        content: String,
        /// New metadata as key=value pairs (repeatable)
        #[arg(long = "meta", value_parser = parse_key_value)]
        metadata: Vec<(String, String)>,
    },
    /// Delete a memory by ID
    Delete {
        /// Memory ID (UUID)
        #[arg(long)]
        id: Uuid,
    },
    /// Clear all memories from the collection
    Clear,
    /// Enhance a prompt with memory context and call an LLM
    Prompt {
        /// The prompt to process
        prompt: String,

        /// LLM model name
        #[arg(long, default_value = DEFAULT_LLM_MODEL, env = "LLM_MODEL")]
        llm_model: String,

        /// Maximum number of memories to inject into the prompt
        #[arg(long, default_value_t = 5)]
        max_memories: usize,

        /// Minimum similarity score for memory retrieval (0.0–1.0)
        #[arg(long, default_value_t = 0.0)]
        min_score: f64,
    },
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let ollama_config = OllamaConfig {
        base_url: cli.ollama_url.clone(),
        timeout: None,
    };

    let ollama = Arc::new(OllamaBackend::new(ollama_config)?);
    let embedder = DefaultTextEmbedder::new(
        Arc::clone(&ollama) as Arc<dyn loci_core::backend::embedding::EmbeddingBackend>,
        EMBED_MODEL,
        EMBED_DIM,
    );

    let qdrant_config = QdrantConfig {
        collection_name: cli.collection,
        similarity_threshold: cli.similarity_threshold,
    };

    let store = QdrantMemoryStore::new(&cli.qdrant_url, qdrant_config, embedder)?;
    store.initialize().await?;

    match cli.command {
        Command::Save { content, metadata } => {
            let input = MemoryInput::new(content, pairs_to_map(metadata));
            let entry = store.save(input).await?;
            println!("{}", serde_json::to_string_pretty(&entry_to_json(&entry))?);
        }
        Command::Query {
            topic,
            max_results,
            min_score,
            filters,
        } => {
            let query = MemoryQuery {
                topic,
                max_results,
                min_score: Score::new(min_score).map_err(|e| format!("invalid min_score: {e}"))?,
                filters: pairs_to_map(filters),
            };
            let entries = store.query(query).await?;
            let json: Vec<_> = entries.iter().map(entry_to_json).collect();
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        Command::Update {
            id,
            content,
            metadata,
        } => {
            let input = MemoryInput::new(content, pairs_to_map(metadata));
            let entry = store.update(id, input).await?;
            println!("{}", serde_json::to_string_pretty(&entry_to_json(&entry))?);
        }
        Command::Delete { id } => {
            store.delete(id).await?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({ "deleted": id.to_string() }))?
            );
        }
        Command::Clear => {
            store.clear().await?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({ "cleared": true }))?
            );
        }
        Command::Prompt {
            prompt,
            llm_model,
            max_memories,
            min_score,
        } => {
            let store = Arc::new(store);
            let min_score = Score::new(min_score).map_err(|e| format!("invalid min_score: {e}"))?;

            let config = ContextualizerConfig {
                max_memories,
                min_score,
                filters: HashMap::new(),
                text_generation_model: llm_model,
            };

            let contextualizer =
                Contextualizer::new(Arc::clone(&store), Arc::clone(&ollama)).with_config(config);

            let response = contextualizer.enhance(&prompt).await?;
            println!("{response}");
        }
    }

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Parses a `key=value` string into a `(String, String)` pair.
fn parse_key_value(s: &str) -> Result<(String, String), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("expected KEY=VALUE, got: {s:?}"))?;
    Ok((k.to_string(), v.to_string()))
}

/// Converts a list of `(key, value)` pairs into a [`HashMap`].
fn pairs_to_map(pairs: Vec<(String, String)>) -> HashMap<String, String> {
    pairs.into_iter().collect()
}

/// Serialises a [`MemoryEntry`] into a [`serde_json::Value`] suitable for
/// pretty-printing to stdout.
fn entry_to_json(e: &MemoryEntry) -> serde_json::Value {
    serde_json::json!({
        "id": e.memory.id.to_string(),
        "content": e.memory.content,
        "metadata": e.memory.metadata,
        "created_at": e.memory.created_at.to_rfc3339(),
        "score": e.score.value(),
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

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
}

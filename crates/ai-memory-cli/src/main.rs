//! Binary entry-point for the `ai-memory` CLI.
//!
//! Exposes all [`MemoryStore`] operations through a `clap`-derived command
//! hierarchy. Global options accept environment-variable fallbacks so the CLI
//! works naturally in Docker / CI environments.

use std::collections::HashMap;

use clap::{Parser, Subcommand};
use uuid::Uuid;

use ai_memory_core::{MemoryInput, MemoryQuery, MemoryStore, Score};
use ai_memory_embedding_ollama::OllamaTextEmbedder;
use ai_memory_qdrant::{QdrantConfig, QdrantMemoryStore};

// ── CLI definition ────────────────────────────────────────────────────────────

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

    /// Embedding model name
    #[arg(long, default_value = "nomic-embed-text", env = "EMBEDDING_MODEL")]
    model: String,

    /// Embedding dimension (must match the model)
    #[arg(long, default_value_t = 768, env = "EMBEDDING_DIMENSION")]
    dimension: usize,

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
    let embedder = OllamaTextEmbedder::new(&cli.ollama_url).with_model(&cli.model, cli.dimension);

    let config = QdrantConfig {
        collection_name: cli.collection,
        similarity_threshold: cli.similarity_threshold,
    };

    let store = QdrantMemoryStore::new(&cli.qdrant_url, config, embedder)?;
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
fn entry_to_json(e: &ai_memory_core::MemoryEntry) -> serde_json::Value {
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

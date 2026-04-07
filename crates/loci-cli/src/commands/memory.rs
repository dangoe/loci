use clap::Subcommand;
use uuid::Uuid;

use crate::commands::parse::parse_key_value;

/// Memory sub-commands.
#[derive(Subcommand)]
pub enum MemoryCommand {
    /// Save a new memory entry.
    #[command(name = "save")]
    Save {
        /// Memory content.
        content: String,
        /// Metadata as key=value pairs (repeatable).
        #[arg(long = "meta", value_parser = parse_key_value)]
        metadata: Vec<(String, String)>,
        /// Optional tier (candidate|stable|core).
        #[arg(long)]
        tier: Option<MemoryTier>,
    },
    /// Query memory entries by semantic similarity.
    #[command(name = "query")]
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
    #[command(name = "get")]
    Get {
        /// Memory entry ID.
        id: Uuid,
    },
    /// Update an existing memory entry by ID.
    #[command(name = "update")]
    Update {
        /// Memory entry ID.
        id: Uuid,
        /// New content (optional).
        content: Option<String>,
        /// New metadata as key=value pairs (repeatable).
        #[arg(long = "meta", value_parser = parse_key_value)]
        metadata: Vec<(String, String)>,
        /// Optional tier override (candidate|stable|core).
        #[arg(long)]
        tier: Option<MemoryTier>,
    },
    /// Delete a memory entry by ID.
    #[command(name = "delete")]
    Delete {
        /// Memory entry ID.
        id: Uuid,
    },
    /// Prunes all expired memory entries from the collection.
    #[command(name = "prune-expired")]
    PruneExpired,
}

/// A semantic memory tier.
#[derive(clap::ValueEnum, PartialEq, Eq, Clone, Debug)]
pub enum MemoryTier {
    /// Request-scoped only, not persisted.
    Ephemeral,
    /// New persisted memory with shorter TTL and lower retrieval priority.
    Candidate,
    /// Promoted memory with longer TTL and higher retrieval priority.
    Stable,
    /// Manually curated long-term memory that does not expire.
    Core,
}

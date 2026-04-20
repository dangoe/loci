// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

use std::path::PathBuf;

use clap::Subcommand;
use uuid::Uuid;

use crate::commands::parse::parse_key_value;

/// Memory sub-commands.
#[derive(Subcommand)]
pub enum MemoryCommand {
    /// Add a new memory entry.
    #[command(name = "add")]
    Add {
        /// Memory content.
        content: String,
        /// Metadata as key=value pairs (repeatable).
        #[arg(long = "meta", value_parser = parse_key_value)]
        metadata: Vec<(String, String)>,
        /// Optional kind (fact|extracted-memory).
        #[arg(long)]
        kind: Option<MemoryKind>,
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
    /// Promote a memory entry to Fact (confidence 1.0, no expiry).
    #[command(name = "promote")]
    Promote {
        /// Memory entry ID.
        id: Uuid,
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
    /// Extract discrete memory entries from text using the configured LLM and
    /// persist them.
    #[command(name = "extract")]
    Extract {
        /// Text to extract memories from.
        /// Mutually exclusive with --file and auto-stdin.
        text: Option<String>,
        /// File(s) to read input from. Use `-` for stdin. Repeatable.
        /// Mutually exclusive with a positional text argument.
        #[arg(long = "file", short = 'f')]
        files: Vec<PathBuf>,
        /// Metadata key=value pairs applied to every extracted entry (repeatable).
        #[arg(long = "meta", value_parser = parse_key_value)]
        metadata: Vec<(String, String)>,
        /// Hard cap on the number of entries extracted.
        #[arg(long)]
        max_entries: Option<usize>,
        /// Minimum LLM confidence score to keep an entry (0.0–1.0).
        /// Entries below this threshold are discarded before storing.
        #[arg(long)]
        min_confidence: Option<f64>,
        /// Free-form guidelines appended to the extraction prompt.
        #[arg(long)]
        guidelines: Option<String>,
    },
}

/// The kind of a memory entry.
#[derive(clap::ValueEnum, PartialEq, Eq, Clone, Debug)]
pub enum MemoryKind {
    /// Confidence 1.0, no decay, manual removal only.
    Fact,
    /// Bayesian confidence in (0.0, 1.0), subject to decay/discard/promotion.
    ExtractedMemory,
}

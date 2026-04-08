// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-core.

use std::collections::HashMap;
use std::fmt;

use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

/// Input passed to [`crate::MemoryStore::save`] and [`crate::MemoryStore::update`].
#[derive(Debug, Clone)]
pub struct MemoryInput {
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub tier: Option<MemoryTier>,
}

impl MemoryInput {
    pub fn new(content: String, metadata: HashMap<String, String>) -> Self {
        Self {
            content,
            metadata,
            tier: None,
        }
    }

    pub fn new_with_tier(
        content: String,
        metadata: HashMap<String, String>,
        tier: MemoryTier,
    ) -> Self {
        Self {
            content,
            metadata,
            tier: Some(tier),
        }
    }
}

/// A semantic memory tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl MemoryTier {
    /// Default retrieval weight used for score blending.
    pub fn retrieval_weight(self) -> f64 {
        match self {
            Self::Ephemeral => 0.0,
            Self::Candidate => 0.6,
            Self::Stable => 0.9,
            Self::Core => 1.0,
        }
    }

    /// Default expiry horizon by tier.
    pub fn default_ttl(self) -> Option<Duration> {
        match self {
            Self::Ephemeral => Some(Duration::zero()),
            Self::Candidate => Some(Duration::days(30)),
            Self::Stable => Some(Duration::days(365)),
            Self::Core => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ephemeral => "ephemeral",
            Self::Candidate => "candidate",
            Self::Stable => "stable",
            Self::Core => "core",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "ephemeral" => Some(Self::Ephemeral),
            "candidate" => Some(Self::Candidate),
            "stable" => Some(Self::Stable),
            "core" => Some(Self::Core),
            _ => None,
        }
    }
}

/// Query behavior mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryQueryMode {
    /// Retrieval-only lookup. Does not affect lifecycle counters.
    Lookup,
    /// Retrieval used for prompt-context memory. Updates usage counters.
    Use,
}

/// A similarity score in the range [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Score(f64);

impl Score {
    /// The minimum possible score (0.0).
    pub const ZERO: Score = Score(0.0);

    /// Creates a new `Score`. Returns [`InvalidScore`] if `value` is outside [0.0, 1.0].
    pub fn new(value: f64) -> Result<Self, InvalidScore> {
        if !(0.0..=1.0).contains(&value) {
            return Err(InvalidScore(value));
        }
        Ok(Self(value))
    }

    /// Returns the raw score value.
    pub fn value(self) -> f64 {
        self.0
    }
}

/// Returned when constructing a [`Score`] with a value outside [0.0, 1.0].
#[derive(Debug)]
pub struct InvalidScore(f64);

impl fmt::Display for InvalidScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "score must be between 0.0 and 1.0, but was: {}", self.0)
    }
}

impl std::error::Error for InvalidScore {}

/// A stored memory. Intentionally embedding-free — model providers that require
/// vector similarity compute embeddings internally.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub tier: MemoryTier,
    pub seen_count: u32,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

impl MemoryEntry {
    /// Creates a new `MemoryEntry` defaulting to `Candidate` tier.
    pub fn new(content: String, metadata: HashMap<String, String>) -> Self {
        Self::new_with_tier(content, metadata, MemoryTier::Candidate)
    }

    /// Creates a new `MemoryEntry` for a specific tier with default lifecycle fields.
    pub fn new_with_tier(
        content: String,
        metadata: HashMap<String, String>,
        tier: MemoryTier,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            content,
            metadata,
            tier,
            seen_count: 1,
            first_seen: now,
            last_seen: now,
            expires_at: tier.default_ttl().map(|ttl| now + ttl),
            created_at: now,
        }
    }
}

/// A query result pairing a [`MemoryEntry`] with its similarity [`Score`].
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryQueryResult {
    pub memory_entry: MemoryEntry,
    pub score: Score,
}

/// Input to [`crate::MemoryStore::query`]. Model providers decide how to interpret the topic
/// (vector similarity, keyword search, etc.).
#[derive(Debug, Clone)]
pub struct MemoryQuery {
    pub topic: String,
    pub max_results: usize,
    /// Minimum final score a result must reach to be included. In [0.0, 1.0].
    pub min_score: Score,
    /// Only return entries whose metadata contains all of these key/value pairs.
    pub filters: HashMap<String, String>,
    /// Query behavior mode.
    pub mode: MemoryQueryMode,
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_score_valid_boundaries() {
        assert!(Score::new(0.0).is_ok());
        assert!(Score::new(0.5).is_ok());
        assert!(Score::new(1.0).is_ok());
    }

    #[test]
    fn test_score_out_of_range() {
        assert!(Score::new(-0.1).is_err());
        assert!(Score::new(1.1).is_err());
    }

    #[test]
    fn test_score_value_returns_stored_value() {
        let s = Score::new(0.75).unwrap();
        assert_eq!(s.value(), 0.75);
    }

    #[test]
    fn test_score_zero_is_zero() {
        assert_eq!(Score::ZERO.value(), 0.0);
    }

    #[test]
    fn test_invalid_score_display_includes_value() {
        let err = Score::new(1.5).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("1.5"),
            "expected message to contain the bad value, got: {msg}"
        );
    }

    #[test]
    fn test_memory_input_new_stores_fields() {
        let metadata = HashMap::from([("key".to_string(), "val".to_string())]);
        let input = MemoryInput::new("content".to_string(), metadata.clone());
        assert_eq!(input.content, "content");
        assert_eq!(input.metadata, metadata);
        assert_eq!(input.tier, None);
    }

    #[test]
    fn test_memory_input_new_with_tier_stores_tier() {
        let input =
            MemoryInput::new_with_tier("content".to_string(), HashMap::new(), MemoryTier::Core);
        assert_eq!(input.tier, Some(MemoryTier::Core));
    }

    #[test]
    fn test_memory_new_generates_unique_ids() {
        let m1 = MemoryEntry::new("hello".to_string(), HashMap::new());
        let m2 = MemoryEntry::new("hello".to_string(), HashMap::new());
        assert_ne!(m1.id, m2.id);
    }

    #[test]
    fn test_memory_new_stores_content_and_metadata() {
        let metadata = HashMap::from([("source".to_string(), "test".to_string())]);
        let m = MemoryEntry::new("my content".to_string(), metadata.clone());
        assert_eq!(m.content, "my content");
        assert_eq!(m.metadata, metadata);
        assert_eq!(m.tier, MemoryTier::Candidate);
        assert_eq!(m.seen_count, 1);
        assert_eq!(m.first_seen, m.last_seen);
        assert!(m.expires_at.is_some());
    }

    #[test]
    fn test_core_default_ttl_is_none() {
        assert_eq!(MemoryTier::Core.default_ttl(), None);
    }

    #[test]
    fn test_memory_tier_roundtrip_str() {
        assert_eq!(MemoryTier::parse("candidate"), Some(MemoryTier::Candidate));
        assert_eq!(MemoryTier::Candidate.as_str(), "candidate");
        assert_eq!(MemoryTier::parse("unknown"), None);
    }
}

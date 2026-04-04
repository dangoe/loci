// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-core.

use std::collections::HashMap;
use std::fmt;

use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Input passed to [`crate::MemoryStore::save`].
///
/// The store assigns an ID and timestamp; callers only supply content and metadata.
#[derive(Debug, Clone)]
pub struct MemoryInput {
    pub content: String,
    pub metadata: HashMap<String, String>,
}

impl MemoryInput {
    pub fn new(content: String, metadata: HashMap<String, String>) -> Self {
        Self { content, metadata }
    }
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
#[derive(Debug, Clone)]
pub struct Memory {
    pub id: Uuid,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
}

impl Memory {
    /// Creates a new `Memory` with a generated ID and the current UTC timestamp.
    pub fn new(content: String, metadata: HashMap<String, String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            content,
            metadata,
            created_at: Utc::now(),
        }
    }
}

/// A query result pairing a [`Memory`] with its similarity [`Score`].
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub memory: Memory,
    pub score: Score,
}

/// Input to [`crate::MemoryStore::query`]. Model providers decide how to interpret the topic
/// (vector similarity, keyword search, etc.).
#[derive(Debug, Clone)]
pub struct MemoryQuery {
    pub topic: String,
    pub max_results: usize,
    /// Minimum similarity score a result must reach to be included. In [0.0, 1.0].
    pub min_score: Score,
    /// Only return entries whose metadata contains all of these key/value pairs.
    pub filters: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
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
    fn test_memory_new_generates_unique_ids() {
        let m1 = Memory::new("hello".to_string(), HashMap::new());
        let m2 = Memory::new("hello".to_string(), HashMap::new());
        assert_ne!(m1.id, m2.id);
    }
}

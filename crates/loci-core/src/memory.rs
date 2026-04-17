// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::collections::HashMap;
use std::fmt;

use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

/// Bayesian review state for a memory entry.
///
/// Tracks a Beta-distribution confidence estimate (`α / (α + β)`). Entries
/// that have never been through the extraction pipeline carry the `Default`
/// value (all fields `None`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReviewState {
    /// Bayesian positive counter.  Together with `beta` derives the effective
    /// confidence: `α / (α + β)`.
    pub alpha: Option<f64>,
    /// Bayesian negative counter.  Incremented on contradiction hits.
    pub beta: Option<f64>,
}

impl ReviewState {
    /// Initialises counters from an LLM-assigned `confidence` value using the
    /// rule `α = confidence × W`, `β = (1 − confidence) × W`, where `W` is
    /// the configurable `seed_weight`.
    pub fn from_confidence(confidence: f64, seed_weight: f64) -> Self {
        Self {
            alpha: Some(confidence * seed_weight),
            beta: Some((1.0 - confidence) * seed_weight),
        }
    }

    /// Returns the Bayesian mean `α / (α + β)`, or `None` when either counter
    /// is absent.
    pub fn bayesian_confidence(&self) -> Option<f64> {
        match (self.alpha, self.beta) {
            (Some(a), Some(b)) if a + b > 0.0 => Some(a / (a + b)),
            _ => None,
        }
    }
}

/// Upper bound for extracted-memory confidence.
///
/// 1.0 is reserved for `MemoryKind::Fact` (promoted by the pipeline); extracted
/// memories are clamped strictly below it. 0.99 is used instead of
/// `1.0 - f64::EPSILON` so that two-decimal display (`%.2f`) never rounds up to
/// `1.00`, keeping the Fact/extracted distinction visible in reports.
pub const MAX_EXTRACTED_CONFIDENCE: f64 = 0.99;

/// Clamps a confidence value into the range used for extracted memories:
/// `[f64::MIN_POSITIVE, MAX_EXTRACTED_CONFIDENCE]` — strictly greater than `0`
/// and strictly less than `1`, with an upper bound that is visibly distinct
/// from `Fact`'s `1.0` at any reasonable display precision.
pub fn clamp_confidence(c: f64) -> f64 {
    c.clamp(f64::MIN_POSITIVE, MAX_EXTRACTED_CONFIDENCE)
}

/// Input passed to [`crate::MemoryStore::add_entry`] and related methods.
#[derive(Debug, Clone)]
pub struct MemoryInput {
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub kind: Option<MemoryKind>,
    /// LLM-assigned confidence score in (0.0, 1.0). Used for retrieval ranking.
    /// `None` when the entry was not produced by LLM extraction.
    pub confidence: Option<f64>,
    /// Pipeline review state (Bayesian counters).
    /// Defaults to all-`None` for entries not processed by the extraction pipeline.
    pub review: ReviewState,
}

impl MemoryInput {
    pub fn new(content: String, metadata: HashMap<String, String>) -> Self {
        Self {
            content,
            metadata,
            kind: None,
            confidence: None,
            review: ReviewState::default(),
        }
    }

    pub fn new_with_kind(
        content: String,
        metadata: HashMap<String, String>,
        kind: MemoryKind,
    ) -> Self {
        Self {
            content,
            metadata,
            kind: Some(kind),
            confidence: None,
            review: ReviewState::default(),
        }
    }
}

/// The kind of a memory entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    /// Confidence 1.0, not subject to decay, can only be removed manually.
    Fact,
    /// Extracted memory with Bayesian confidence in (0.0, 1.0), subject to
    /// decay, auto-discard, and auto-promotion.
    ExtractedMemory,
}

impl MemoryKind {
    /// Default retrieval weight used for score blending.
    pub fn retrieval_weight(self) -> f64 {
        match self {
            Self::Fact => 1.0,
            Self::ExtractedMemory => 0.8,
        }
    }

    /// Default expiry horizon by kind.
    pub fn default_ttl(self) -> Option<Duration> {
        match self {
            Self::Fact => None,
            Self::ExtractedMemory => Some(Duration::days(365)),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fact => "fact",
            Self::ExtractedMemory => "extracted_memory",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "fact" => Some(Self::Fact),
            "extracted_memory" => Some(Self::ExtractedMemory),
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

    /// The maximum possible score (1.0).
    pub const MAX: Score = Score(1.0);

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
    pub kind: MemoryKind,
    pub seen_count: u32,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    /// LLM-assigned confidence score in (0.0, 1.0). Used for retrieval ranking.
    /// `None` when the entry was not produced by LLM extraction.
    pub confidence: Option<f64>,
    /// Pipeline review state (Bayesian counters).
    pub review: ReviewState,
}

impl MemoryEntry {
    /// Creates a new `MemoryEntry` defaulting to `ExtractedMemory` kind.
    pub fn new(content: String, metadata: HashMap<String, String>) -> Self {
        Self::new_with_kind(content, metadata, MemoryKind::ExtractedMemory)
    }

    /// Creates a new `MemoryEntry` for a specific kind with default lifecycle fields.
    pub fn new_with_kind(
        content: String,
        metadata: HashMap<String, String>,
        kind: MemoryKind,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            content,
            metadata,
            kind,
            seen_count: 1,
            first_seen: now,
            last_seen: now,
            expires_at: kind.default_ttl().map(|ttl| now + ttl),
            created_at: now,
            confidence: None,
            review: ReviewState::default(),
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
        assert_eq!(input.kind, None);
        assert_eq!(input.review, ReviewState::default());
    }

    #[test]
    fn test_memory_input_new_with_kind_stores_kind() {
        let input =
            MemoryInput::new_with_kind("content".to_string(), HashMap::new(), MemoryKind::Fact);
        assert_eq!(input.kind, Some(MemoryKind::Fact));
        assert_eq!(input.review, ReviewState::default());
    }

    #[test]
    fn test_review_state_default_is_all_none() {
        let r = ReviewState::default();
        assert!(r.alpha.is_none());
        assert!(r.beta.is_none());
    }

    #[test]
    fn test_review_state_from_confidence_sets_counters() {
        let r = ReviewState::from_confidence(0.8, 10.0);
        assert!((r.alpha.unwrap() - 8.0).abs() < 1e-10);
        assert!((r.beta.unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_review_state_from_confidence_respects_seed_weight() {
        let r = ReviewState::from_confidence(0.8, 5.0);
        assert!((r.alpha.unwrap() - 4.0).abs() < 1e-10);
        assert!((r.beta.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bayesian_confidence_returns_mean() {
        let r = ReviewState {
            alpha: Some(8.0),
            beta: Some(2.0),
        };
        let c = r.bayesian_confidence().unwrap();
        assert!((c - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bayesian_confidence_none_when_counters_absent() {
        assert!(ReviewState::default().bayesian_confidence().is_none());
    }

    #[test]
    fn test_bayesian_confidence_none_when_sum_is_zero() {
        let r = ReviewState {
            alpha: Some(0.0),
            beta: Some(0.0),
        };
        assert!(r.bayesian_confidence().is_none());
    }

    #[test]
    fn test_clamp_confidence_below_zero() {
        assert!(clamp_confidence(-1.0) > 0.0);
        assert_eq!(clamp_confidence(-1.0), f64::MIN_POSITIVE);
    }

    #[test]
    fn test_clamp_confidence_above_one() {
        assert!(clamp_confidence(2.0) < 1.0);
        assert_eq!(clamp_confidence(2.0), MAX_EXTRACTED_CONFIDENCE);
    }

    #[test]
    fn test_clamp_confidence_at_zero() {
        assert_eq!(clamp_confidence(0.0), f64::MIN_POSITIVE);
    }

    #[test]
    fn test_clamp_confidence_at_one() {
        assert_eq!(clamp_confidence(1.0), MAX_EXTRACTED_CONFIDENCE);
    }

    #[test]
    fn test_clamp_confidence_display_never_rounds_to_one_at_two_decimals() {
        // Regression: `%.2f` of `1.0 - f64::EPSILON` rounds to `1.00`, making
        // extracted memories visually indistinguishable from `Fact` in reports.
        let clamped = clamp_confidence(1.0);
        let display = format!("{:.2}", clamped);
        assert_ne!(display, "1.00", "clamped max must not render as 1.00");
    }

    #[test]
    fn test_clamp_confidence_passthrough_interior() {
        assert_eq!(clamp_confidence(0.5), 0.5);
    }

    #[test]
    fn test_memory_entry_new_has_default_review() {
        let m = MemoryEntry::new("content".to_string(), HashMap::new());
        assert_eq!(m.review, ReviewState::default());
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
        assert_eq!(m.kind, MemoryKind::ExtractedMemory);
        assert_eq!(m.seen_count, 1);
        assert_eq!(m.first_seen, m.last_seen);
        assert!(m.expires_at.is_some());
    }

    #[test]
    fn test_fact_default_ttl_is_none() {
        assert_eq!(MemoryKind::Fact.default_ttl(), None);
    }

    #[test]
    fn test_extracted_memory_default_ttl_is_one_year() {
        assert_eq!(
            MemoryKind::ExtractedMemory.default_ttl(),
            Some(Duration::days(365))
        );
    }

    #[test]
    fn test_memory_kind_roundtrip_str() {
        assert_eq!(MemoryKind::parse("fact"), Some(MemoryKind::Fact));
        assert_eq!(MemoryKind::Fact.as_str(), "fact");
        assert_eq!(
            MemoryKind::parse("extracted_memory"),
            Some(MemoryKind::ExtractedMemory)
        );
        assert_eq!(MemoryKind::ExtractedMemory.as_str(), "extracted_memory");
        assert_eq!(MemoryKind::parse("unknown"), None);
    }
}

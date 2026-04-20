// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

pub mod extraction;
pub mod store;

use std::fmt;
use std::{collections::HashMap, error::Error as StdError};

use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

/// Unified trust level for a memory entry.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryTrust {
    /// Ground truth. Confidence 1.0, not subject to decay, removed only manually.
    Fact,
    /// Extracted memory with an initial confidence and evolving Bayesian
    /// evidence. Subject to decay, auto-discard, and auto-promotion.
    Extracted {
        /// Assigned confidence score by the extractor.
        confidence: f64,
        /// Evolving Bayesian evidence (α/β counters) updated by the pipeline.
        evidence: TrustEvidence,
    },
}

impl MemoryTrust {
    /// Effective score used for retrieval and ranking, combining extractor assigned confidence
    ///
    /// `Fact` → 1.0.
    /// `Extracted` → Bayesian mean when counters are populated; falls back to
    /// the initial `confidence` value.
    pub fn effective_score(&self) -> Score {
        match self {
            Self::Fact => Score::MAX,
            Self::Extracted {
                confidence,
                evidence,
            } => Score::try_new(evidence.bayesian_confidence().unwrap_or(*confidence))
                .unwrap_or(Score::ZERO),
        }
    }

    /// Default retrieval weight used for score blending.
    pub fn retrieval_weight(&self) -> f64 {
        match self {
            Self::Fact => 1.0,
            Self::Extracted { .. } => 0.8,
        }
    }

    /// Default expiry horizon by trust level.
    pub fn default_ttl(&self) -> Option<Duration> {
        match self {
            Self::Fact => None,
            Self::Extracted { .. } => Some(Duration::days(365)),
        }
    }

    /// Clamps a confidence value into the range used for extracted memories.
    pub fn clamp_confidence(confidence: f64) -> f64 {
        confidence.clamp(f64::MIN_POSITIVE, 0.95)
    }
}

/// Accumulated bayesian evidence for an extracted memory entry.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrustEvidence {
    /// Bayesian positive counter.
    pub alpha: Option<f64>,
    /// Bayesian negative counter.
    pub beta: Option<f64>,
}

impl TrustEvidence {
    /// Initialises counters from an extractor assigned `confidence` value using the
    /// rule `α = confidence × W`, `β = (1 − confidence) × W`, where `W` is
    /// the configurable `seed_weight`.
    pub fn from_confidence(confidence: f64, seed_weight: f64) -> Self {
        Self {
            alpha: Some(confidence * seed_weight),
            beta: Some((1.0 - confidence) * seed_weight),
        }
    }

    /// Returns the bayesian mean `α / (α + β)`, or `None` when either counter
    /// is absent or their sum is zero.
    pub fn bayesian_confidence(&self) -> Option<f64> {
        match (self.alpha, self.beta) {
            (Some(a), Some(b)) if a + b > 0.0 => Some(a / (a + b)),
            _ => None,
        }
    }
}

/// A similarity score in the range [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Score(f64);

impl Score {
    /// The minimum possible score (0.0).
    pub const ZERO: Score = Score(0.0);

    /// The maximum possible score (1.0).
    pub const MAX: Score = Score(1.0);

    pub fn try_new(value: f64) -> Result<Self, InvalidScore> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(InvalidScore(value))
        }
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

impl StdError for InvalidScore {}

/// A stored memory. Intentionally embedding-free — model providers that require
/// vector similarity compute embeddings internally.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryEntry {
    /// Unique identifier for the memory entry.
    id: Uuid,
    /// The factual content of the memory entry.
    content: String,
    /// Trust level and evidence for this memory entry, used for retrieval ranking and lifecycle management.
    trust: MemoryTrust,
    /// Arbitrary key/value pairs describing the memory (e.g. source, type, etc.).
    metadata: HashMap<String, String>,
    /// Number of times this memory has been retrieved with `MemoryQueryMode::Use`.
    seen_count: u32,
    /// Timestamp of the first time this memory was retrieved with `MemoryQueryMode::Use`.
    first_seen: Option<DateTime<Utc>>,
    /// Timestamp of the most recent time this memory was retrieved with `MemoryQueryMode::Use`.
    last_seen: Option<DateTime<Utc>>,
    /// Optional timestamp when this memory expires and becomes ineligible for retrieval. Managed by the pipeline based on trust level and lifecycle policies.
    expires_at: Option<DateTime<Utc>>,
    /// Timestamp when this memory entry was created. Useful for debugging and lifecycle policies.
    created_at: DateTime<Utc>,
}

impl MemoryEntry {
    /// Creates a new `MemoryEntry` defaulting to an uninformed
    /// `Extracted` trust (confidence 0.5, no evidence counters).
    pub fn new(content: String, metadata: HashMap<String, String>) -> Self {
        Self::new_with_trust(
            content,
            metadata,
            MemoryTrust::Extracted {
                confidence: 0.5,
                evidence: TrustEvidence::default(),
            },
        )
    }

    /// Creates a new `MemoryEntry` with the given trust level and default
    /// lifecycle fields.
    pub fn new_with_trust(
        content: String,
        metadata: HashMap<String, String>,
        trust: MemoryTrust,
    ) -> Self {
        let now = Utc::now();
        let expires_at = trust.default_ttl().map(|ttl| now + ttl);
        Self {
            id: Uuid::new_v4(),
            content,
            metadata,
            trust,
            seen_count: 0,
            first_seen: None,
            last_seen: None,
            expires_at,
            created_at: now,
        }
    }

    /// Returns the unique identifier of this memory entry.
    pub fn id(&self) -> &Uuid {
        &self.id
    }

    /// Returns the content of this memory entry.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Returns the trust level of this memory entry.
    pub fn trust(&self) -> &MemoryTrust {
        &self.trust
    }

    /// Returns a reference to the metadata of this memory entry.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Returns the number of times this memory has been retrieved with `MemoryQueryMode::Use`.
    pub fn seen_count(&self) -> u32 {
        self.seen_count
    }

    /// Returns the timestamp of the first time this memory was retrieved with `MemoryQueryMode::Use`.
    pub fn first_seen(&self) -> Option<DateTime<Utc>> {
        self.first_seen
    }

    /// Returns the timestamp of the most recent time this memory was retrieved with `MemoryQueryMode::Use`.
    pub fn last_seen(&self) -> Option<DateTime<Utc>> {
        self.last_seen
    }

    /// Returns the optional timestamp when this memory expires and becomes ineligible for retrieval.
    pub fn expires_at(&self) -> Option<DateTime<Utc>> {
        self.expires_at
    }

    /// Returns the timestamp when this memory entry was created.
    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// Reconstructs a `MemoryEntry` from its complete set of stored field values.
    ///
    /// Intended for storage backends that need to deserialise persisted entries (e.g. the
    /// Qdrant store). All fields are supplied explicitly; no defaults are applied.
    #[allow(clippy::too_many_arguments)]
    pub fn reconstruct(
        id: Uuid,
        content: String,
        metadata: HashMap<String, String>,
        trust: MemoryTrust,
        seen_count: u32,
        first_seen: Option<DateTime<Utc>>,
        last_seen: Option<DateTime<Utc>>,
        expires_at: Option<DateTime<Utc>>,
        created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id,
            content,
            metadata,
            trust,
            seen_count,
            first_seen,
            last_seen,
            expires_at,
            created_at,
        }
    }

    /// Records a retrieval use of this entry: increments `seen_count`, updates
    /// `last_seen` to now, and sets `first_seen` if not yet recorded.
    pub fn record_use(&mut self) {
        let now = Utc::now();
        self.seen_count = self.seen_count.saturating_add(1);
        if self.first_seen.is_none() {
            self.first_seen = Some(now);
        }
        self.last_seen = Some(now);
    }

    /// Creates a `MemoryEntry` with explicit field values for use in tests.
    ///
    /// Only available under `#[cfg(any(feature = "testing", test))]`.
    #[cfg(any(feature = "testing", test))]
    pub fn new_for_testing(
        id: uuid::Uuid,
        content: String,
        metadata: std::collections::HashMap<String, String>,
        trust: MemoryTrust,
    ) -> Self {
        let now = chrono::Utc::now();
        let expires_at = trust.default_ttl().map(|ttl| now + ttl);
        Self {
            id,
            content,
            metadata,
            trust,
            seen_count: 0,
            first_seen: None,
            last_seen: None,
            expires_at,
            created_at: now,
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_score_new_with_valid_boundary_values_returns_ok() {
        assert!(Score::try_new(0.0).is_ok());
        assert!(Score::try_new(0.5).is_ok());
        assert!(Score::try_new(1.0).is_ok());
    }

    #[test]
    fn test_score_new_with_out_of_range_value_returns_err() {
        assert!(Score::try_new(-0.1).is_err());
        assert!(Score::try_new(1.1).is_err());
    }

    #[test]
    fn test_score_value_returns_stored_value() {
        let s = Score::try_new(0.75).unwrap();
        assert_eq!(s.value(), 0.75);
    }

    #[test]
    fn test_score_zero_constant_has_value_zero() {
        assert_eq!(Score::ZERO.value(), 0.0);
    }

    #[test]
    fn test_invalid_score_display_includes_value() {
        let err = Score::try_new(1.5).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("1.5"),
            "expected message to contain the bad value, got: {msg}"
        );
    }

    #[test]
    fn test_trust_evidence_default_is_all_none() {
        let b = TrustEvidence::default();
        assert!(b.alpha.is_none());
        assert!(b.beta.is_none());
    }

    #[test]
    fn test_trust_evidence_from_confidence_sets_counters() {
        let b = TrustEvidence::from_confidence(0.8, 10.0);
        assert!((b.alpha.unwrap() - 8.0).abs() < 1e-10);
        assert!((b.beta.unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_trust_evidence_from_confidence_respects_seed_weight() {
        let b = TrustEvidence::from_confidence(0.8, 5.0);
        assert!((b.alpha.unwrap() - 4.0).abs() < 1e-10);
        assert!((b.beta.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bayesian_confidence_returns_mean() {
        let b = TrustEvidence {
            alpha: Some(8.0),
            beta: Some(2.0),
        };
        let c = b.bayesian_confidence().unwrap();
        assert!((c - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bayesian_confidence_none_when_counters_absent() {
        assert!(TrustEvidence::default().bayesian_confidence().is_none());
    }

    #[test]
    fn test_bayesian_confidence_none_when_sum_is_zero() {
        let b = TrustEvidence {
            alpha: Some(0.0),
            beta: Some(0.0),
        };
        assert!(b.bayesian_confidence().is_none());
    }

    #[test]
    fn test_memory_trust_clamp_confidence_below_zero() {
        assert!(MemoryTrust::clamp_confidence(-1.0) > 0.0);
        assert_eq!(MemoryTrust::clamp_confidence(-1.0), f64::MIN_POSITIVE);
    }

    #[test]
    fn test_memory_trust_clamp_confidence_above_one() {
        assert!(MemoryTrust::clamp_confidence(2.0) < 1.0);
        assert_eq!(MemoryTrust::clamp_confidence(2.0), 0.95);
    }

    #[test]
    fn test_memory_trust_clamp_confidence_at_zero() {
        assert_eq!(MemoryTrust::clamp_confidence(0.0), f64::MIN_POSITIVE);
    }

    #[test]
    fn test_memory_trust_clamp_confidence_at_one() {
        assert_eq!(MemoryTrust::clamp_confidence(1.0), 0.95);
    }

    #[test]
    fn test_memory_trust_clamp_confidence_display_never_rounds_to_one_at_two_decimals() {
        let clamped = MemoryTrust::clamp_confidence(1.0);
        let display = format!("{:.2}", clamped);
        assert_ne!(display, "1.00", "clamped max must not render as 1.00");
    }

    #[test]
    fn test_memory_trust_clamp_confidence_passthrough_interior() {
        assert_eq!(MemoryTrust::clamp_confidence(0.5), 0.5);
    }

    #[test]
    fn test_memory_entry_new_has_default_extracted_trust() {
        let m = MemoryEntry::new("content".to_string(), HashMap::new());
        assert!(matches!(m.trust, MemoryTrust::Extracted { .. }));
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
        assert!(matches!(m.trust, MemoryTrust::Extracted { .. }));
        assert_eq!(m.seen_count, 0);
        assert!(m.first_seen.is_none());
        assert!(m.last_seen.is_none());
        assert!(m.expires_at.is_some());
    }

    #[test]
    fn test_fact_default_ttl_is_none() {
        assert_eq!(MemoryTrust::Fact.default_ttl(), None);
    }

    #[test]
    fn test_extracted_default_ttl_is_one_year() {
        let trust = MemoryTrust::Extracted {
            confidence: 0.8,
            evidence: TrustEvidence::default(),
        };
        assert_eq!(trust.default_ttl(), Some(Duration::days(365)));
    }

    #[test]
    fn test_effective_confidence_fact_is_one() {
        assert!((MemoryTrust::Fact.effective_score().value() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_effective_confidence_extracted_uses_bayesian_when_populated() {
        let trust = MemoryTrust::Extracted {
            confidence: 0.3,
            evidence: TrustEvidence {
                alpha: Some(8.0),
                beta: Some(2.0),
            },
        };
        assert!((trust.effective_score().value() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_effective_confidence_extracted_falls_back_to_confidence() {
        let trust = MemoryTrust::Extracted {
            confidence: 0.7,
            evidence: TrustEvidence::default(),
        };
        assert!((trust.effective_score().value() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fact_entry_has_no_ttl() {
        let m = MemoryEntry::new_with_trust("x".to_string(), HashMap::new(), MemoryTrust::Fact);
        assert!(m.expires_at.is_none());
    }

    #[test]
    fn test_extracted_entry_has_ttl() {
        let trust = MemoryTrust::Extracted {
            confidence: 0.8,
            evidence: TrustEvidence::default(),
        };
        let m = MemoryEntry::new_with_trust("x".to_string(), HashMap::new(), trust);
        assert!(m.expires_at.is_some());
    }
}

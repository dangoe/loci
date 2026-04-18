// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::collections::HashMap;
use std::fmt;

use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

/// Unified trust level for a memory entry.
///
/// Unified trust level for a memory entry.
///
/// Merges the previous `MemoryKind` + `confidence` + `TrustEvidence` triad
/// into one concept. All three aspects drive TTL, retrieval weight, storage
/// serialisation, and score blending.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryTrust {
    /// Ground truth. Confidence 1.0, not subject to decay, removed only
    /// manually.
    Fact,
    /// LLM-extracted memory with an initial confidence and evolving Bayesian
    /// evidence. Subject to decay, auto-discard, and auto-promotion.
    Extracted {
        /// Initial LLM-assigned confidence score, also the seed for `evidence`.
        confidence: f64,
        /// Evolving Bayesian evidence (α/β counters) updated by the pipeline.
        evidence: TrustEvidence,
    },
}

impl MemoryTrust {
    /// Effective confidence used for retrieval score blending.
    ///
    /// `Fact` → 1.0.
    /// `Extracted` → Bayesian mean when counters are populated; falls back to
    /// the initial `confidence` value.
    pub fn effective_confidence(&self) -> f64 {
        match self {
            Self::Fact => 1.0,
            Self::Extracted {
                confidence,
                evidence,
            } => evidence.bayesian_confidence().unwrap_or(*confidence),
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

    /// Storage string discriminant (`"fact"` or `"extracted_memory"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fact => "fact",
            Self::Extracted { .. } => "extracted_memory",
        }
    }
}

/// Accumulated Bayesian evidence for an extracted memory entry.
///
/// Tracks a Beta-distribution posterior (`α / (α + β)`). Entries that have
/// never been through the extraction pipeline carry the `Default` value (all
/// fields `None`), which causes `bayesian_confidence` to return `None` and
/// the owning [`MemoryTrust`] to fall back to the initial `confidence` value.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrustEvidence {
    /// Bayesian positive counter. Together with `beta` derives the effective
    /// confidence: `α / (α + β)`.
    pub alpha: Option<f64>,
    /// Bayesian negative counter. Incremented on contradiction hits.
    pub beta: Option<f64>,
}

impl TrustEvidence {
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
    /// is absent or their sum is zero.
    pub fn bayesian_confidence(&self) -> Option<f64> {
        match (self.alpha, self.beta) {
            (Some(a), Some(b)) if a + b > 0.0 => Some(a / (a + b)),
            _ => None,
        }
    }
}

/// Upper bound for extracted-memory confidence.
///
/// 1.0 is reserved for [`MemoryTrust::Fact`]; extracted memories are clamped
/// strictly below it. 0.99 is used instead of `1.0 - f64::EPSILON` so that
/// two-decimal display (`%.2f`) never rounds up to `1.00`, keeping the
/// Fact/extracted distinction visible in reports.
pub const MAX_EXTRACTED_CONFIDENCE: f64 = 0.99;

/// Clamps a confidence value into the range used for extracted memories:
/// `[f64::MIN_POSITIVE, MAX_EXTRACTED_CONFIDENCE]`.
pub fn clamp_confidence(c: f64) -> f64 {
    c.clamp(f64::MIN_POSITIVE, MAX_EXTRACTED_CONFIDENCE)
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

/// Input passed to [`crate::MemoryStore::add_entry`] and related methods.
#[derive(Debug, Clone)]
pub struct MemoryInput {
    pub content: String,
    pub metadata: HashMap<String, String>,
    /// Trust level for this entry. `None` at store time defaults to
    /// `Extracted { confidence: 0.5, evidence: Default::default() }`.
    pub trust: Option<MemoryTrust>,
}

impl MemoryInput {
    pub fn new(content: String, metadata: HashMap<String, String>) -> Self {
        Self {
            content,
            metadata,
            trust: None,
        }
    }

    pub fn new_with_trust(
        content: String,
        metadata: HashMap<String, String>,
        trust: MemoryTrust,
    ) -> Self {
        Self {
            content,
            metadata,
            trust: Some(trust),
        }
    }
}

/// A stored memory. Intentionally embedding-free — model providers that require
/// vector similarity compute embeddings internally.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub trust: MemoryTrust,
    pub seen_count: u32,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
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
            seen_count: 1,
            first_seen: now,
            last_seen: now,
            expires_at,
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

/// Input to [`crate::MemoryStore::query`]. Model providers decide how to
/// interpret the topic (vector similarity, keyword search, etc.).
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
    fn test_score_new_with_valid_boundary_values_returns_ok() {
        assert!(Score::new(0.0).is_ok());
        assert!(Score::new(0.5).is_ok());
        assert!(Score::new(1.0).is_ok());
    }

    #[test]
    fn test_score_new_with_out_of_range_value_returns_err() {
        assert!(Score::new(-0.1).is_err());
        assert!(Score::new(1.1).is_err());
    }

    #[test]
    fn test_score_value_returns_stored_value() {
        let s = Score::new(0.75).unwrap();
        assert_eq!(s.value(), 0.75);
    }

    #[test]
    fn test_score_zero_constant_has_value_zero() {
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
        assert!(input.trust.is_none());
    }

    #[test]
    fn test_memory_input_new_with_trust_stores_trust() {
        let input =
            MemoryInput::new_with_trust("content".to_string(), HashMap::new(), MemoryTrust::Fact);
        assert_eq!(input.trust, Some(MemoryTrust::Fact));
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
        let clamped = clamp_confidence(1.0);
        let display = format!("{:.2}", clamped);
        assert_ne!(display, "1.00", "clamped max must not render as 1.00");
    }

    #[test]
    fn test_clamp_confidence_passthrough_interior() {
        assert_eq!(clamp_confidence(0.5), 0.5);
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
        assert_eq!(m.seen_count, 1);
        assert_eq!(m.first_seen, m.last_seen);
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
    fn test_memory_trust_as_str() {
        assert_eq!(MemoryTrust::Fact.as_str(), "fact");
        let extracted = MemoryTrust::Extracted {
            confidence: 0.8,
            evidence: TrustEvidence::default(),
        };
        assert_eq!(extracted.as_str(), "extracted_memory");
    }

    #[test]
    fn test_effective_confidence_fact_is_one() {
        assert!((MemoryTrust::Fact.effective_confidence() - 1.0).abs() < f64::EPSILON);
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
        assert!((trust.effective_confidence() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_effective_confidence_extracted_falls_back_to_confidence() {
        let trust = MemoryTrust::Extracted {
            confidence: 0.7,
            evidence: TrustEvidence::default(),
        };
        assert!((trust.effective_confidence() - 0.7).abs() < f64::EPSILON);
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

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

mod chunker;
pub mod llm;

use futures::future::BoxFuture;
pub use llm::{LlmMemoryExtractionStrategy, LlmMemoryExtractionStrategyParams};
use log::{debug, info};

use std::{collections::HashSet, marker::PhantomData, num::NonZeroUsize, sync::Arc};

use uuid::Uuid;

use crate::{
    classification::{ClassificationModelProvider, HitClass},
    error::{MemoryExtractionError, MemoryStoreError},
    memory::{
        MemoryEntry, MemoryTrust, Score, TrustEvidence,
        store::{MemoryInput, MemoryQuery, MemoryQueryMode, MemoryStore},
    },
};

/// Configuration for the initial search stage.
#[derive(Debug, Clone)]
pub struct MemoryQueryOptions {
    /// Maximum number of results to retrieve in semantic search.
    max_results: NonZeroUsize,
    /// Minimum score threshold for search results.
    min_score: Score,
}

impl MemoryQueryOptions {
    /// Creates a new `MemoryQueryOptions` with the given parameters.
    pub fn new(max_results: NonZeroUsize, min_score: Score) -> Self {
        Self {
            max_results,
            min_score,
        }
    }

    /// Validates the parameters and creates a new `MemoryQueryOptions`.
    pub fn try_new(
        max_results: usize,
        min_score: f64,
    ) -> Result<Self, InvalidMemoryQueryOptionsError> {
        let max_results = NonZeroUsize::new(max_results).ok_or_else(|| {
            InvalidMemoryQueryOptionsError::MaxResults(format!(
                "max_results must be greater than 0, got {max_results}"
            ))
        })?;

        let min_score = Score::try_new(min_score).map_err(|e| {
            InvalidMemoryQueryOptionsError::MinScore(format!(
                "min_score must be between 0.0 and 1.0, got {min_score}: {e}"
            ))
        })?;

        Ok(Self::new(max_results, min_score))
    }
}

/// Errors related to invalid search results configuration.
#[derive(Debug)]
pub enum InvalidMemoryQueryOptionsError {
    MaxResults(String),
    MinScore(String),
}

impl std::fmt::Display for InvalidMemoryQueryOptionsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxResults(msg) => write!(f, "invalid max_results: {msg}"),
            Self::MinScore(msg) => write!(f, "invalid min_score: {msg}"),
        }
    }
}

impl std::error::Error for InvalidMemoryQueryOptionsError {}

/// Configuration for the memory extractor.
#[derive(Debug, Clone)]
pub struct MemoryExtractorConfig {
    /// Configuration for direct semantic search stage.
    direct_search: MemoryQueryOptions,
    /// Configuration for inverted semantic search stage.
    inverted_search: MemoryQueryOptions,
    /// Seed weight `W` used to initialise Bayesian counters from LLM confidence.
    /// `α = confidence × W`, `β = (1 − confidence) × W`. Default: 10.0.
    bayesian_seed_weight: f64,
    /// Maximum increment applied to a counter per evidence event. Default: 5.0.
    max_counter_increment: f64,
    /// Upper bound for each Bayesian counter (α and β). Default: 100.0.
    max_counter: f64,
    /// Score at or below which an entry is automatically discarded. Default: 0.1.
    auto_discard_threshold: f64,
    /// Score at or above which an entry is automatically promoted to Fact. Default: 0.9.
    auto_promotion_threshold: f64,
    /// Minimum accumulated `α` (evidence weight) required for auto-promotion
    /// to Fact — even when the Bayesian score clears `auto_promotion_threshold`.
    /// Prevents a single high-confidence observation from being promoted in
    /// isolation: a Fact should be corroborated, not just asserted once.
    /// Default: 12.0 (seed weight 10 + one strong corroborating observation).
    min_alpha_for_promotion: f64,
}

impl MemoryExtractorConfig {
    /// Creates a new `MemoryExtractorConfig` with all fields specified explicitly.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        direct_search: MemoryQueryOptions,
        inverted_search: MemoryQueryOptions,
        bayesian_seed_weight: f64,
        max_counter_increment: f64,
        max_counter: f64,
        auto_discard_threshold: f64,
        auto_promotion_threshold: f64,
        min_alpha_for_promotion: f64,
    ) -> Self {
        Self {
            direct_search,
            inverted_search,
            bayesian_seed_weight,
            max_counter_increment,
            max_counter,
            auto_discard_threshold,
            auto_promotion_threshold,
            min_alpha_for_promotion,
        }
    }
}

impl Default for MemoryExtractorConfig {
    fn default() -> Self {
        Self {
            direct_search: MemoryQueryOptions::new(
                NonZeroUsize::new(5).unwrap(),
                Score::try_new(0.70).unwrap_or(Score::ZERO),
            ),
            inverted_search: MemoryQueryOptions::new(
                NonZeroUsize::new(3).unwrap(),
                Score::try_new(0.60).unwrap_or(Score::ZERO),
            ),
            bayesian_seed_weight: 10.0,
            max_counter_increment: 5.0,
            max_counter: 100.0,
            auto_discard_threshold: 0.1,
            auto_promotion_threshold: 0.9,
            min_alpha_for_promotion: 12.0,
        }
    }
}

/// An entry that was discarded during extraction, with its content and reason.
#[derive(Debug, Clone)]
pub struct DiscardedEntry {
    /// The content of the discarded entry.
    content: String,
    /// The reason the entry was discarded.
    reason: DiscardReason,
}

impl DiscardedEntry {
    fn new(content: String, reason: DiscardReason) -> Self {
        Self { content, reason }
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn reason(&self) -> &DiscardReason {
        &self.reason
    }
}

/// Why a candidate was discarded by the extractor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscardReason {
    /// The candidate's Bayesian score fell at or below `auto_discard_threshold`.
    LowScore,
    /// The candidate contradicted an existing Fact.
    ContradictsAFact,
}

/// Outcome of a [`MemoryExtractor::extract_memory_entries`] call.
#[derive(Debug)]
pub struct MemoryExtractionResult {
    inserted: Vec<MemoryEntry>,
    merged: Vec<MemoryEntry>,
    promoted: Vec<MemoryEntry>,
    discarded: Vec<DiscardedEntry>,
}

impl MemoryExtractionResult {
    pub fn inserted(&self) -> &[MemoryEntry] {
        &self.inserted
    }

    pub fn merged(&self) -> &[MemoryEntry] {
        &self.merged
    }

    pub fn promoted(&self) -> &[MemoryEntry] {
        &self.promoted
    }

    pub fn discarded(&self) -> &[DiscardedEntry] {
        &self.discarded
    }
}

/// Extracts memory entries from a source.
pub trait MemoryExtractionStrategy<P: Send + Sync>: Send + Sync {
    fn extract<'a>(
        &'a self,
        input: &'a str,
        params: &'a P,
    ) -> BoxFuture<'a, Result<Vec<MemoryInput>, MemoryExtractionError>>;
}

/// Orchestrates memory extraction:
///
/// 1. Extract candidates via `E`
/// 2. Clamp candidate confidence to the open interval (0.0, 1.0)
/// 3. Dual semantic search (direct + inverted) per candidate
/// 4. Classify each hit via `C`
/// 5. Fact-contradiction check — discard immediately if any Fact is contradicted
/// 6. Update Bayesian counters: α += min(c, max_increment) for same/complementary;
///    β += min(c, max_increment) for contradictions; each counter capped at max_counter
/// 7. Auto-discard (score ≤ threshold) or auto-promote (score ≥ threshold) or insert/merge
/// 8. Persist
/// 9. Decay (handled by a separate CLI command)
pub struct MemoryExtractor<S, E, C, P> {
    store: Arc<S>,
    strategy: Arc<E>,
    classifier: Arc<C>,
    config: MemoryExtractorConfig,
    _phantom: PhantomData<P>,
}

impl<S, E, C, P> MemoryExtractor<S, E, C, P>
where
    S: MemoryStore,
    E: MemoryExtractionStrategy<P>,
    C: ClassificationModelProvider,
    P: Clone + Send + Sync,
{
    pub fn new(
        store: Arc<S>,
        strategy: Arc<E>,
        classifier: Arc<C>,
        config: MemoryExtractorConfig,
    ) -> Self {
        Self {
            store,
            strategy,
            classifier,
            config,
            _phantom: PhantomData,
        }
    }

    pub async fn extract_memory_entries(
        &self,
        input: &str,
        params: &P,
    ) -> Result<MemoryExtractionResult, MemoryExtractionError> {
        debug!("Extracting memory entries for input \"{}\".", input);

        let candidates = self.strategy.extract(input, params).await?;

        debug!(
            "Extracted candidates {:?} from strategy.",
            candidates.iter().collect::<Vec<_>>()
        );

        let mut result = MemoryExtractionResult {
            inserted: Vec::new(),
            merged: Vec::new(),
            promoted: Vec::new(),
            discarded: Vec::new(),
        };

        for candidate in candidates {
            let raw_confidence = match candidate.trust() {
                MemoryTrust::Extracted { confidence, .. } => *confidence,
                _ => 0.5,
            };
            let confidence = MemoryTrust::clamp_confidence(raw_confidence);

            let classified_hits = self.collect_classified_hits(&candidate).await?;

            let contradicts_fact = classified_hits.iter().any(|(hit, class)| {
                matches!(hit.trust(), MemoryTrust::Fact) && *class == HitClass::Contradiction
            });

            if contradicts_fact {
                result.discarded.push(DiscardedEntry::new(
                    candidate.content().to_string(),
                    DiscardReason::ContradictsAFact,
                ));
                continue;
            }

            let trust_evidence = self.compute_trust_evidence(&classified_hits, confidence);

            let score = trust_evidence.bayesian_confidence().unwrap_or(0.0);

            if score <= self.config.auto_discard_threshold {
                result.discarded.push(DiscardedEntry::new(
                    candidate.content().to_string(),
                    DiscardReason::LowScore,
                ));
                continue;
            }

            let alpha = trust_evidence.alpha().unwrap_or(0.0);
            let eligible_for_promotion = score >= self.config.auto_promotion_threshold
                && alpha >= self.config.min_alpha_for_promotion;

            let final_trust = if eligible_for_promotion {
                MemoryTrust::Fact
            } else {
                MemoryTrust::Extracted {
                    confidence: score,
                    evidence: trust_evidence,
                }
            };

            let is_promoted = matches!(final_trust, MemoryTrust::Fact);
            let match_ids = extract_match_ids(&classified_hits);

            if !match_ids.is_empty() {
                let merged = self
                    .merge_and_add(&candidate, final_trust, match_ids)
                    .await?;
                if is_promoted {
                    result.promoted.extend(merged);
                } else {
                    result.merged.extend(merged);
                }
            } else {
                let input = MemoryInput::new(
                    candidate.content().to_string(),
                    final_trust,
                    candidate.metadata().clone(),
                );
                let added = self.add_entries_or_fail(vec![input]).await?;
                if is_promoted {
                    result.promoted.extend(added);
                } else {
                    result.inserted.extend(added);
                }
            }
        }

        Ok(result)
    }

    async fn merge_and_add(
        &self,
        candidate: &MemoryInput,
        trust: MemoryTrust,
        match_ids: Vec<Uuid>,
    ) -> Result<Vec<MemoryEntry>, MemoryExtractionError> {
        let merged_input = MemoryInput::new(
            candidate.content().to_string(),
            trust,
            candidate.metadata().clone(),
        );

        let add_result = self.add_entries_or_fail(vec![merged_input]).await?;

        for id in &match_ids {
            self.store
                .delete_entry(id)
                .await
                .map_err(MemoryExtractionError::MemoryStore)?;
        }

        Ok(add_result)
    }

    async fn add_entries_or_fail(
        &self,
        inputs: Vec<MemoryInput>,
    ) -> Result<Vec<MemoryEntry>, MemoryExtractionError> {
        let add_result = self
            .store
            .add_entries(&inputs)
            .await
            .map_err(MemoryExtractionError::MemoryStore)?;

        if !add_result.failures().is_empty() {
            let msg = add_result
                .failures()
                .iter()
                .map(|f| f.error().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            Err(MemoryExtractionError::MemoryStore(
                MemoryStoreError::GenericSave(msg),
            ))
        } else {
            Ok(add_result.added().to_vec())
        }
    }

    /// Computes the updated Bayesian trust evidence for a candidate given its
    /// classified hits.
    ///
    /// Uses the incoming extraction confidence `c` (already clamped) to
    /// increment α (for Duplicate/Complementary hits) or β (for Contradiction
    /// hits), each bounded by `max_counter_increment` and `max_counter`.
    fn compute_trust_evidence(
        &self,
        classified_hits: &[(MemoryEntry, HitClass)],
        confidence: f64,
    ) -> TrustEvidence {
        let mut trust_evidence =
            TrustEvidence::from_confidence(confidence, self.config.bayesian_seed_weight);

        let increment = confidence.min(self.config.max_counter_increment);

        for (_, class) in classified_hits {
            match class {
                HitClass::Duplicate | HitClass::Complementary => {
                    trust_evidence.increment_alpha(increment, self.config.max_counter);
                }
                HitClass::Contradiction => {
                    trust_evidence.increment_beta(increment, self.config.max_counter);
                }
                HitClass::Unrelated => {}
            }
        }

        trust_evidence
    }

    async fn collect_classified_hits(
        &self,
        candidate: &MemoryInput,
    ) -> Result<Vec<(MemoryEntry, HitClass)>, MemoryExtractionError> {
        info!(
            "Collecting classified hits for candidate \"{}\".",
            candidate.content()
        );

        let mut classified_hits: Vec<(MemoryEntry, HitClass)> = Vec::new();

        for direct_hit in self.query_direct_hits(candidate).await? {
            info!(
                "Collecting direct hit with content \"{}\" and ID {}.",
                direct_hit.content(),
                direct_hit.id()
            );
            let class = self
                .classifier
                .classify_hit(candidate.content(), direct_hit.content())
                .await
                .map_err(|e| MemoryExtractionError::Other(e.to_string()))?;
            classified_hits.push((direct_hit, class));
        }

        for inverted_hit in self.query_inverted_hits(candidate).await? {
            info!(
                "Collecting inverted hit with content \"{}\" and ID {}.",
                inverted_hit.content(),
                inverted_hit.id()
            );

            let class = self
                .classifier
                .classify_hit(candidate.content(), inverted_hit.content())
                .await
                .map_err(|e| MemoryExtractionError::Other(e.to_string()))?;
            classified_hits.push((inverted_hit, class));
        }

        debug!(
            "Collected classified hits: {:?}",
            classified_hits
                .iter()
                .map(|(hit, class)| (hit.content(), class))
                .collect::<Vec<_>>()
        );

        Ok(classified_hits)
    }

    async fn query_direct_hits(
        &self,
        candidate: &MemoryInput,
    ) -> Result<Vec<MemoryEntry>, MemoryExtractionError> {
        self.query_memory_store(
            MemoryQuery::new(candidate.content().to_string(), MemoryQueryMode::Lookup)
                .with_max_results(self.config.direct_search.max_results)
                .with_min_score(self.config.direct_search.min_score),
        )
        .await
    }

    async fn query_inverted_hits(
        &self,
        candidate: &MemoryInput,
    ) -> Result<Vec<MemoryEntry>, MemoryExtractionError> {
        self.query_memory_store(
            MemoryQuery::new(
                format!("the opposite of {}", candidate.content()),
                MemoryQueryMode::Lookup,
            )
            .with_max_results(self.config.inverted_search.max_results)
            .with_min_score(self.config.inverted_search.min_score),
        )
        .await
    }

    async fn query_memory_store(
        &self,
        query: MemoryQuery,
    ) -> Result<Vec<MemoryEntry>, MemoryExtractionError> {
        self.store
            .query(query)
            .await
            .map_err(MemoryExtractionError::MemoryStore)
    }
}

fn extract_match_ids(classified_hits: &[(MemoryEntry, HitClass)]) -> Vec<Uuid> {
    let mut seen_ids: HashSet<Uuid> = HashSet::new();
    classified_hits
        .iter()
        .filter(|(_, class)| matches!(class, HitClass::Duplicate | HitClass::Complementary))
        .map(|(hit, _)| *hit.id())
        .filter(|id| seen_ids.insert(*id))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use futures::future::BoxFuture;
    use pretty_assertions::assert_eq;

    use crate::classification::HitClass;
    use crate::error::MemoryExtractionError;
    use crate::memory::store::MemoryInput;
    use crate::memory::{MemoryTrust, TrustEvidence};
    use crate::testing::{
        AddEntriesBehavior, ClassifyBehavior, MockClassificationModelProvider, MockStore,
        QueryBehavior, make_extracted_result, make_fact_result,
    };

    use super::{DiscardReason, MemoryExtractionStrategy, MemoryExtractor, MemoryExtractorConfig};

    struct FixedStrategy(Vec<MemoryInput>);

    impl MemoryExtractionStrategy<()> for FixedStrategy {
        fn extract(
            &self,
            _input: &str,
            _params: &(),
        ) -> BoxFuture<'_, Result<Vec<MemoryInput>, MemoryExtractionError>> {
            let entries = self.0.clone();
            Box::pin(async move { Ok(entries) })
        }
    }

    fn make_candidate(content: &str, confidence: f64) -> MemoryInput {
        MemoryInput::new(
            content.to_string(),
            MemoryTrust::Extracted {
                confidence,
                evidence: TrustEvidence::default(),
            },
            HashMap::new(),
        )
    }

    fn extractor(
        store: Arc<MockStore>,
        strategy: Arc<FixedStrategy>,
        classifier: Arc<MockClassificationModelProvider>,
    ) -> MemoryExtractor<MockStore, FixedStrategy, MockClassificationModelProvider, ()> {
        MemoryExtractor::new(
            store,
            strategy,
            classifier,
            MemoryExtractorConfig::default(),
        )
    }

    #[tokio::test]
    async fn test_extract_memory_entries_with_no_similar_entries_inserts_candidate() {
        // No existing hits → candidate is inserted as ExtractedMemory.
        let candidate = make_candidate("the sky is blue", 0.8);
        let inserted_result = make_extracted_result(uuid::Uuid::new_v4(), "the sky is blue", 0.8);

        let store = Arc::new(
            MockStore::new()
                .with_query_behavior(QueryBehavior::Ok(vec![]))
                .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![inserted_result])),
        );
        let strategy = Arc::new(FixedStrategy(vec![candidate]));
        let classifier = Arc::new(MockClassificationModelProvider::new());

        let result = extractor(store, strategy, classifier)
            .extract_memory_entries("ignored", &())
            .await
            .unwrap();

        assert_eq!(result.inserted.len(), 1);
        assert!(result.merged.is_empty());
        assert!(result.promoted.is_empty());
        assert!(result.discarded.is_empty());
    }

    #[tokio::test]
    async fn test_extract_memory_entries_when_contradiction_lowers_score_discards_candidate() {
        // Two Contradiction hits lower the score enough to trigger auto-discard.
        // confidence = 0.7 → alpha=7.0, beta=3.0
        // increment = min(0.7, 5.0) = 0.7
        // two Contradiction hits → beta += 0.7 * 2 = 4.4 → alpha=7.0, beta=7.4
        // score ≈ 0.486 — well above auto_discard_threshold of 0.1
        // To guarantee discard we need a very low confidence with many contradictions.
        // Use a very tight config: auto_discard_threshold = 0.5.
        let candidate = make_candidate("the sky is green", 0.1);
        let hit = make_extracted_result(uuid::Uuid::new_v4(), "the sky is blue", 0.9);

        let store = Arc::new(MockStore::new().with_query_behavior(QueryBehavior::Ok(vec![hit])));
        let strategy = Arc::new(FixedStrategy(vec![candidate.clone()]));
        let classifier = Arc::new(
            MockClassificationModelProvider::new()
                .with_behavior(ClassifyBehavior::Ok(HitClass::Contradiction)),
        );

        // confidence = 0.1 → alpha=1.0, beta=9.0; increment = min(0.1, 5.0) = 0.1
        // 1 contradiction → beta = 9.0 + 0.1 = 9.1 → score ≈ 0.099 ≤ 0.1 → discard
        let config = MemoryExtractorConfig {
            auto_discard_threshold: 0.1,
            ..MemoryExtractorConfig::default()
        };

        let extractor = MemoryExtractor::new(Arc::clone(&store), strategy, classifier, config);
        let result = extractor
            .extract_memory_entries("ignored", &())
            .await
            .unwrap();

        assert_eq!(result.discarded.len(), 1);
        assert_eq!(result.discarded[0].reason, DiscardReason::LowScore);
        assert!(result.inserted.is_empty());
        assert!(result.merged.is_empty());
        assert!(result.promoted.is_empty());
    }

    #[tokio::test]
    async fn test_fact_contradiction_discards_candidate() {
        // Candidate contradicts an existing Fact → immediately discarded.
        let candidate = make_candidate("the sky is green", 0.9);
        let fact_hit = make_fact_result(uuid::Uuid::new_v4(), "the sky is blue", 0.9);

        let store =
            Arc::new(MockStore::new().with_query_behavior(QueryBehavior::Ok(vec![fact_hit])));
        let strategy = Arc::new(FixedStrategy(vec![candidate]));
        let classifier = Arc::new(
            MockClassificationModelProvider::new()
                .with_behavior(ClassifyBehavior::Ok(HitClass::Contradiction)),
        );

        let result = extractor(Arc::clone(&store), strategy, classifier)
            .extract_memory_entries("ignored", &())
            .await
            .unwrap();

        assert_eq!(result.discarded.len(), 1);
        assert_eq!(result.discarded[0].reason, DiscardReason::ContradictsAFact);
        assert!(result.inserted.is_empty());
    }

    #[tokio::test]
    async fn test_extract_memory_entries_with_many_duplicate_hits_promotes_to_fact() {
        // Many Duplicate hits boost alpha enough to trigger auto-promotion to Fact.
        // confidence = 0.9 → alpha=9.0, beta=1.0; increment = min(0.9, 5.0) = 0.9
        // 5 duplicate hits → alpha = min(9.0 + 0.9*5, 100) = 13.5 → score ≈ 0.931 ≥ 0.9 → promote
        let candidate = make_candidate("cats have nine lives", 0.9);
        let hit_id = uuid::Uuid::new_v4();
        let hit = make_extracted_result(hit_id, "cats have nine lives", 0.9);
        let hits = vec![
            hit.clone(),
            hit.clone(),
            hit.clone(),
            hit.clone(),
            hit.clone(),
        ];
        let promoted_result = make_fact_result(uuid::Uuid::new_v4(), "cats have nine lives", 1.0);

        let store = Arc::new(
            MockStore::new()
                .with_query_behavior(QueryBehavior::Ok(hits))
                .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![promoted_result])),
        );
        let strategy = Arc::new(FixedStrategy(vec![candidate]));
        let classifier = Arc::new(
            MockClassificationModelProvider::new()
                .with_behavior(ClassifyBehavior::Ok(HitClass::Duplicate)),
        );

        let result = extractor(Arc::clone(&store), strategy, classifier)
            .extract_memory_entries("ignored", &())
            .await
            .unwrap();

        assert_eq!(result.promoted.len(), 1);
        assert!(result.inserted.is_empty());
        assert!(result.merged.is_empty());
        assert!(result.discarded.is_empty());
    }

    #[tokio::test]
    async fn test_extract_memory_entries_with_single_duplicate_hit_merges_entries() {
        // Duplicate hit → same ID deduplicated → 1 delete, 1 merged insert.
        // confidence = 0.7 → alpha=7.0, beta=3.0; increment = min(0.7, 5.0) = 0.7
        // 1 duplicate → alpha = 7.7 → score ≈ 0.72 (between thresholds → merged)
        let hit_id = uuid::Uuid::new_v4();
        let candidate = make_candidate("cats have nine lives", 0.7);
        let hit = make_extracted_result(hit_id, "cats have nine lives", 0.9);
        let merged_result =
            make_extracted_result(uuid::Uuid::new_v4(), "cats have nine lives", 0.72);

        let store = Arc::new(
            MockStore::new()
                .with_query_behavior(QueryBehavior::Ok(vec![hit]))
                .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![merged_result])),
        );
        let strategy = Arc::new(FixedStrategy(vec![candidate]));
        let classifier = Arc::new(
            MockClassificationModelProvider::new()
                .with_behavior(ClassifyBehavior::Ok(HitClass::Duplicate)),
        );

        let extractor = extractor(Arc::clone(&store), strategy, classifier);
        let result = extractor
            .extract_memory_entries("ignored", &())
            .await
            .unwrap();

        assert_eq!(result.merged.len(), 1);
        assert!(result.inserted.is_empty());
        assert!(result.promoted.is_empty());
        assert!(result.discarded.is_empty());

        let state = store.snapshot();
        assert_eq!(
            state.delete_id,
            Some(hit_id),
            "old entry must have been deleted"
        );
    }

    #[test]
    fn test_extractor_config_defaults() {
        let config = MemoryExtractorConfig::default();
        assert_eq!(config.direct_search.max_results.get(), 5);
        assert!((config.direct_search.min_score.value() - 0.70).abs() < f64::EPSILON);
        assert_eq!(config.inverted_search.max_results.get(), 3);
        assert!((config.inverted_search.min_score.value() - 0.60).abs() < f64::EPSILON);
        assert!((config.bayesian_seed_weight - 10.0).abs() < f64::EPSILON);
        assert!((config.max_counter_increment - 5.0).abs() < f64::EPSILON);
        assert!((config.max_counter - 100.0).abs() < f64::EPSILON);
        assert!((config.auto_discard_threshold - 0.1).abs() < f64::EPSILON);
        assert!((config.auto_promotion_threshold - 0.9).abs() < f64::EPSILON);
        assert!((config.min_alpha_for_promotion - 12.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_counter_caps_are_respected() {
        // Verify that max_counter_increment and max_counter are honoured.
        // confidence = 0.9, seed = 10 → alpha=9.0, beta=1.0
        // max_counter_increment = 1.0, max_counter = 9.5
        // 1 duplicate → increment = min(0.9, 1.0) = 0.9 → alpha = min(9.9, 9.5) = 9.5
        let store =
            Arc::new(
                MockStore::new()
                    .with_query_behavior(QueryBehavior::Ok(vec![make_extracted_result(
                        uuid::Uuid::new_v4(),
                        "hit",
                        0.9,
                    )]))
                    .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![
                        make_extracted_result(uuid::Uuid::new_v4(), "hit", 0.9),
                    ])),
            );
        let strategy = Arc::new(FixedStrategy(vec![make_candidate("hit", 0.9)]));
        let classifier = Arc::new(
            MockClassificationModelProvider::new()
                .with_behavior(ClassifyBehavior::Ok(HitClass::Duplicate)),
        );

        let config = MemoryExtractorConfig {
            bayesian_seed_weight: 10.0,
            max_counter_increment: 1.0,
            max_counter: 9.5,
            // Below promotion threshold so we can inspect the add input
            auto_promotion_threshold: 0.99,
            auto_discard_threshold: 0.0,
            ..MemoryExtractorConfig::default()
        };

        let extractor = MemoryExtractor::new(Arc::clone(&store), strategy, classifier, config);

        extractor
            .extract_memory_entries("ignored", &())
            .await
            .unwrap();

        let state = store.snapshot();
        let inputs = state
            .add_inputs
            .expect("add_entries should have been called");
        let alpha = match inputs[0].trust() {
            MemoryTrust::Extracted { evidence, .. } => {
                evidence.alpha().expect("alpha should be set")
            }
            _ => panic!("expected Extracted trust"),
        };
        // alpha must not exceed max_counter = 9.5
        assert!(alpha <= 9.5, "alpha {alpha} exceeded max_counter 9.5");
    }
}

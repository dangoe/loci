// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;

use uuid::Uuid;

use crate::classification::{ClassificationModelProvider, HitClass};
use crate::error::{MemoryExtractionError, MemoryStoreError};
use crate::memory::{
    TrustEvidence, MemoryInput, MemoryTrust, MemoryQuery, MemoryQueryMode, MemoryQueryResult,
    Score, clamp_confidence,
};
use crate::memory_extraction::MemoryExtractionStrategy;
use crate::store::MemoryStore;

/// Configuration for the initial search stage.
#[derive(Debug, Clone)]
pub struct PipelineSearchResultsConfig {
    /// Maximum number of results to retrieve in semantic search.
    pub max_results: usize,
    /// Minimum score threshold for search results.
    pub min_score: f64,
}

/// Configuration for the memory extraction pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Configuration for direct semantic search stage.
    pub direct_search: PipelineSearchResultsConfig,
    /// Configuration for inverted semantic search stage.
    pub inverted_search: PipelineSearchResultsConfig,
    /// Seed weight `W` used to initialise Bayesian counters from LLM confidence.
    /// `α = confidence × W`, `β = (1 − confidence) × W`. Default: 10.0.
    pub bayesian_seed_weight: f64,
    /// Maximum increment applied to a counter per evidence event. Default: 5.0.
    pub max_counter_increment: f64,
    /// Upper bound for each Bayesian counter (α and β). Default: 100.0.
    pub max_counter: f64,
    /// Score at or below which an entry is automatically discarded. Default: 0.1.
    pub auto_discard_threshold: f64,
    /// Score at or above which an entry is automatically promoted to Fact. Default: 0.9.
    pub auto_promotion_threshold: f64,
    /// Minimum accumulated `α` (evidence weight) required for auto-promotion
    /// to Fact — even when the Bayesian score clears `auto_promotion_threshold`.
    /// Prevents a single high-confidence observation from being promoted in
    /// isolation: a Fact should be corroborated, not just asserted once.
    /// Default: 12.0 (seed weight 10 + one strong corroborating observation).
    pub min_alpha_for_promotion: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            direct_search: PipelineSearchResultsConfig {
                max_results: 5,
                min_score: 0.70,
            },
            inverted_search: PipelineSearchResultsConfig {
                max_results: 3,
                min_score: 0.60,
            },
            bayesian_seed_weight: 10.0,
            max_counter_increment: 5.0,
            max_counter: 100.0,
            auto_discard_threshold: 0.1,
            auto_promotion_threshold: 0.9,
            min_alpha_for_promotion: 12.0,
        }
    }
}

/// An entry that was discarded during the pipeline, with its content and reason.
#[derive(Debug, Clone)]
pub struct DiscardedEntry {
    /// The content of the discarded entry.
    pub content: String,
    /// The reason the entry was discarded.
    pub reason: DiscardReason,
}

/// Why a candidate was discarded by the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscardReason {
    /// The candidate's Bayesian score fell at or below `auto_discard_threshold`.
    LowScore,
    /// The candidate contradicted an existing Fact.
    ContradictsAFact,
}

/// Outcome of a [`MemoryExtractionPipeline::extract_and_store`] call.
pub struct PipelineResult {
    /// Newly inserted entries (no prior match).
    pub inserted: Vec<MemoryQueryResult>,
    /// Entries that replaced existing duplicates/complements.
    pub merged: Vec<MemoryQueryResult>,
    /// Entries whose score reached `auto_promotion_threshold` and were stored as Facts.
    pub promoted: Vec<MemoryQueryResult>,
    /// Entries that were discarded (low score or contradicts a Fact).
    pub discarded: Vec<DiscardedEntry>,
}

/// Orchestrates the memory extraction pipeline.
///
/// Stages:
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
pub struct MemoryExtractionPipeline<S, E, C, P> {
    store: Arc<S>,
    strategy: Arc<E>,
    classifier: Arc<C>,
    config: PipelineConfig,
    _phantom: PhantomData<P>,
}

impl<S, E, C, P> MemoryExtractionPipeline<S, E, C, P>
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
        config: PipelineConfig,
    ) -> Self {
        Self {
            store,
            strategy,
            classifier,
            config,
            _phantom: PhantomData,
        }
    }

    pub async fn extract_and_store(
        &self,
        input: &str,
        params: P,
    ) -> Result<PipelineResult, MemoryExtractionError> {
        let candidates = self.strategy.extract(input, params).await?;

        let mut result = PipelineResult {
            inserted: Vec::new(),
            merged: Vec::new(),
            promoted: Vec::new(),
            discarded: Vec::new(),
        };

        for candidate in candidates {
            let confidence = clamp_confidence(match &candidate.trust {
                Some(MemoryTrust::Extracted { confidence, .. }) => *confidence,
                _ => 0.5,
            });

            let classified_hits = self.collect_classified_hits(&candidate).await?;

            let contradicts_fact = classified_hits.iter().any(|(hit, class)| {
                matches!(hit.memory_entry.trust, MemoryTrust::Fact)
                    && *class == HitClass::Contradiction
            });

            if contradicts_fact {
                result.discarded.push(DiscardedEntry {
                    content: candidate.content.clone(),
                    reason: DiscardReason::ContradictsAFact,
                });
                continue;
            }

            let trust_evidence = self.compute_trust_evidence(&classified_hits, confidence);

            let score = trust_evidence.bayesian_confidence().unwrap_or(0.0);

            if score <= self.config.auto_discard_threshold {
                result.discarded.push(DiscardedEntry {
                    content: candidate.content.clone(),
                    reason: DiscardReason::LowScore,
                });
                continue;
            }

            let alpha = trust_evidence.alpha.unwrap_or(0.0);
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
                let input = MemoryInput {
                    content: candidate.content.clone(),
                    metadata: candidate.metadata.clone(),
                    trust: Some(final_trust),
                };
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
    ) -> Result<Vec<MemoryQueryResult>, MemoryExtractionError> {
        let merged_input = MemoryInput {
            content: candidate.content.clone(),
            metadata: candidate.metadata.clone(),
            trust: Some(trust),
        };

        let add_result = self.add_entries_or_fail(vec![merged_input]).await?;

        for id in &match_ids {
            self.store
                .delete_entry(*id)
                .await
                .map_err(MemoryExtractionError::MemoryStore)?;
        }

        Ok(add_result)
    }

    async fn add_entries_or_fail(
        &self,
        inputs: Vec<MemoryInput>,
    ) -> Result<Vec<MemoryQueryResult>, MemoryExtractionError> {
        let add_result = self
            .store
            .add_entries(inputs)
            .await
            .map_err(MemoryExtractionError::MemoryStore)?;

        if !add_result.failures.is_empty() {
            let msg = add_result
                .failures
                .into_iter()
                .map(|f| f.error.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            Err(MemoryExtractionError::MemoryStore(
                MemoryStoreError::GenericSave(msg),
            ))
        } else {
            Ok(add_result.added)
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
        classified_hits: &[(MemoryQueryResult, HitClass)],
        confidence: f64,
    ) -> TrustEvidence {
        let mut trust_evidence =
            TrustEvidence::from_confidence(confidence, self.config.bayesian_seed_weight);

        let increment = confidence.min(self.config.max_counter_increment);

        for (_, class) in classified_hits {
            match class {
                HitClass::Duplicate | HitClass::Complementary => {
                    let alpha = trust_evidence.alpha.unwrap_or(0.0);
                    trust_evidence.alpha =
                        Some((alpha + increment).min(self.config.max_counter));
                }
                HitClass::Contradiction => {
                    let beta = trust_evidence.beta.unwrap_or(0.0);
                    trust_evidence.beta = Some((beta + increment).min(self.config.max_counter));
                }
                HitClass::Unrelated => {}
            }
        }

        trust_evidence
    }

    async fn collect_classified_hits(
        &self,
        candidate: &MemoryInput,
    ) -> Result<Vec<(MemoryQueryResult, HitClass)>, MemoryExtractionError> {
        let mut classified_hits: Vec<(MemoryQueryResult, HitClass)> = Vec::new();

        for direct_hit in self.query_direct_hits(candidate).await? {
            let class = self
                .classifier
                .classify_hit(&candidate.content, &direct_hit.memory_entry.content)
                .await
                .map_err(|e| MemoryExtractionError::Other(e.to_string()))?;
            classified_hits.push((direct_hit, class));
        }

        for inverted_hit in self.query_inverted_hits(candidate).await? {
            let class = self
                .classifier
                .classify_hit(&candidate.content, &inverted_hit.memory_entry.content)
                .await
                .map_err(|e| MemoryExtractionError::Other(e.to_string()))?;
            classified_hits.push((inverted_hit, class));
        }

        Ok(classified_hits)
    }

    async fn query_direct_hits(
        &self,
        candidate: &MemoryInput,
    ) -> Result<Vec<MemoryQueryResult>, MemoryExtractionError> {
        self.query_memory_store(MemoryQuery {
            topic: candidate.content.clone(),
            max_results: self.config.direct_search.max_results,
            min_score: Score::new(self.config.direct_search.min_score).unwrap_or(Score::ZERO),
            filters: HashMap::new(),
            mode: MemoryQueryMode::Lookup,
        })
        .await
    }

    async fn query_inverted_hits(
        &self,
        candidate: &MemoryInput,
    ) -> Result<Vec<MemoryQueryResult>, MemoryExtractionError> {
        self.query_memory_store(MemoryQuery {
            topic: format!("the opposite of {}", candidate.content),
            max_results: self.config.inverted_search.max_results,
            min_score: Score::new(self.config.inverted_search.min_score).unwrap_or(Score::ZERO),
            filters: HashMap::new(),
            mode: MemoryQueryMode::Lookup,
        })
        .await
    }

    async fn query_memory_store(
        &self,
        query: MemoryQuery,
    ) -> Result<Vec<MemoryQueryResult>, MemoryExtractionError> {
        self.store
            .query(query)
            .await
            .map_err(MemoryExtractionError::MemoryStore)
    }
}

fn extract_match_ids(classified_hits: &[(MemoryQueryResult, HitClass)]) -> Vec<Uuid> {
    let mut seen_ids: HashSet<Uuid> = HashSet::new();
    classified_hits
        .iter()
        .filter(|(_, class)| matches!(class, HitClass::Duplicate | HitClass::Complementary))
        .map(|(hit, _)| hit.memory_entry.id)
        .filter(|id| seen_ids.insert(*id))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::future::Future;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use crate::classification::HitClass;
    use crate::error::MemoryExtractionError;
    use crate::memory::{TrustEvidence, MemoryInput, MemoryTrust};
    use crate::memory_extraction::MemoryExtractionStrategy;
    use crate::testing::{
        AddEntriesBehavior, ClassifyBehavior, MockClassificationModelProvider, MockStore,
        QueryBehavior, make_extracted_result, make_fact_result,
    };

    use super::{DiscardReason, MemoryExtractionPipeline, PipelineConfig};

    struct FixedStrategy(Vec<MemoryInput>);

    impl MemoryExtractionStrategy<()> for FixedStrategy {
        fn extract(
            &self,
            _input: &str,
            _params: (),
        ) -> impl Future<Output = Result<Vec<MemoryInput>, MemoryExtractionError>> + Send {
            let entries = self.0.clone();
            async move { Ok(entries) }
        }
    }

    fn make_candidate(content: &str, confidence: f64) -> MemoryInput {
        MemoryInput {
            content: content.to_string(),
            metadata: HashMap::new(),
            trust: Some(MemoryTrust::Extracted {
                confidence,
                evidence: TrustEvidence::default(),
            }),
        }
    }

    fn pipeline(
        store: Arc<MockStore>,
        strategy: Arc<FixedStrategy>,
        classifier: Arc<MockClassificationModelProvider>,
    ) -> MemoryExtractionPipeline<MockStore, FixedStrategy, MockClassificationModelProvider, ()>
    {
        MemoryExtractionPipeline::new(store, strategy, classifier, PipelineConfig::default())
    }

    #[tokio::test]
    async fn test_fresh_insert_path() {
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

        let result = pipeline(store, strategy, classifier)
            .extract_and_store("ignored", ())
            .await
            .unwrap();

        assert_eq!(result.inserted.len(), 1);
        assert!(result.merged.is_empty());
        assert!(result.promoted.is_empty());
        assert!(result.discarded.is_empty());
    }

    #[tokio::test]
    async fn test_auto_discard_path() {
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
        let config = PipelineConfig {
            auto_discard_threshold: 0.1,
            ..PipelineConfig::default()
        };

        let pipeline =
            MemoryExtractionPipeline::new(Arc::clone(&store), strategy, classifier, config);
        let result = pipeline.extract_and_store("ignored", ()).await.unwrap();

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

        let result = pipeline(Arc::clone(&store), strategy, classifier)
            .extract_and_store("ignored", ())
            .await
            .unwrap();

        assert_eq!(result.discarded.len(), 1);
        assert_eq!(result.discarded[0].reason, DiscardReason::ContradictsAFact);
        assert!(result.inserted.is_empty());
    }

    #[tokio::test]
    async fn test_auto_promotion_path() {
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

        let result = pipeline(Arc::clone(&store), strategy, classifier)
            .extract_and_store("ignored", ())
            .await
            .unwrap();

        assert_eq!(result.promoted.len(), 1);
        assert!(result.inserted.is_empty());
        assert!(result.merged.is_empty());
        assert!(result.discarded.is_empty());
    }

    #[tokio::test]
    async fn test_merge_path() {
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

        let pipeline = pipeline(Arc::clone(&store), strategy, classifier);
        let result = pipeline.extract_and_store("ignored", ()).await.unwrap();

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
    fn test_pipeline_config_defaults() {
        let config = PipelineConfig::default();
        assert_eq!(config.direct_search.max_results, 5);
        assert!((config.direct_search.min_score - 0.70).abs() < f64::EPSILON);
        assert_eq!(config.inverted_search.max_results, 3);
        assert!((config.inverted_search.min_score - 0.60).abs() < f64::EPSILON);
        assert!((config.bayesian_seed_weight - 10.0).abs() < f64::EPSILON);
        assert!((config.max_counter_increment - 5.0).abs() < f64::EPSILON);
        assert!((config.max_counter - 100.0).abs() < f64::EPSILON);
        assert!((config.auto_discard_threshold - 0.1).abs() < f64::EPSILON);
        assert!((config.auto_promotion_threshold - 0.9).abs() < f64::EPSILON);
        assert!((config.min_alpha_for_promotion - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_counter_caps_are_respected() {
        // Verify that max_counter_increment and max_counter are honoured.
        // confidence = 0.9, seed = 10 → alpha=9.0, beta=1.0
        // max_counter_increment = 1.0, max_counter = 9.5
        // 1 duplicate → increment = min(0.9, 1.0) = 0.9 → alpha = min(9.9, 9.5) = 9.5
        let store = Arc::new(
            MockStore::new()
                .with_query_behavior(QueryBehavior::Ok(vec![make_extracted_result(
                    uuid::Uuid::new_v4(),
                    "hit",
                    0.9,
                )]))
                .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![make_extracted_result(
                    uuid::Uuid::new_v4(),
                    "hit",
                    0.9,
                )])),
        );
        let strategy = Arc::new(FixedStrategy(vec![make_candidate("hit", 0.9)]));
        let classifier = Arc::new(
            MockClassificationModelProvider::new()
                .with_behavior(ClassifyBehavior::Ok(HitClass::Duplicate)),
        );

        let config = PipelineConfig {
            bayesian_seed_weight: 10.0,
            max_counter_increment: 1.0,
            max_counter: 9.5,
            // Below promotion threshold so we can inspect the add input
            auto_promotion_threshold: 0.99,
            auto_discard_threshold: 0.0,
            ..PipelineConfig::default()
        };

        let pipeline =
            MemoryExtractionPipeline::new(Arc::clone(&store), strategy, classifier, config);

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            pipeline.extract_and_store("ignored", ()).await.unwrap();
        });

        let state = store.snapshot();
        let inputs = state
            .add_inputs
            .expect("add_entries should have been called");
        let alpha = match &inputs[0].trust {
            Some(MemoryTrust::Extracted { evidence, .. }) => {
                evidence.alpha.expect("alpha should be set")
            }
            _ => panic!("expected Extracted trust"),
        };
        // alpha must not exceed max_counter = 9.5
        assert!(alpha <= 9.5, "alpha {alpha} exceeded max_counter 9.5");
    }
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;

use uuid::Uuid;

use crate::classification::{ClassificationModelProvider, HitClass};
use crate::error::MemoryExtractionError;
use crate::memory::{
    MemoryInput, MemoryQuery, MemoryQueryMode, MemoryQueryResult, ReviewState, Score,
};
use crate::memory_extraction::MemoryExtractionStrategy;
use crate::store::MemoryStore;

/// Configuration for the memory extraction pipeline (Stages 2–7).
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum number of results to retrieve in direct semantic search. Default: 5
    pub direct_search_max_results: usize,
    /// Minimum score threshold for direct search results. Default: 0.70
    pub direct_search_min_score: f64,
    /// Maximum number of results to retrieve in inverted semantic search. Default: 3
    pub inverted_search_max_results: usize,
    /// Minimum score threshold for inverted search results. Default: 0.60
    pub inverted_search_min_score: f64,
    /// Alpha increment for Duplicate hits. Default: 3.0
    pub duplicate_alpha_weight: f64,
    /// Alpha increment for Complementary hits. Default: 1.0
    pub complementary_alpha_weight: f64,
    /// Beta increment for Contradiction hits. Default: 3.0
    pub contradiction_beta_weight: f64,
    /// Per-day exponential decay rate applied to alpha. Default: 0.99
    pub decay_rate: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            direct_search_max_results: 5,
            direct_search_min_score: 0.70,
            inverted_search_max_results: 3,
            inverted_search_min_score: 0.60,
            duplicate_alpha_weight: 3.0,
            complementary_alpha_weight: 1.0,
            contradiction_beta_weight: 3.0,
            decay_rate: 0.99,
        }
    }
}

/// Outcome of a [`MemoryExtractionPipeline::extract_and_store`] call.
pub struct PipelineResult {
    pub inserted: Vec<MemoryQueryResult>,
    pub merged: Vec<MemoryQueryResult>,
    pub pending_review: Vec<MemoryQueryResult>,
}

/// Orchestrates the 7-stage memory extraction pipeline.
///
/// - Stage 1: extract candidates via `E`
/// - Stage 2: dual semantic search (direct + inverted) per candidate
/// - Stage 3: classify each hit via `C`
/// - Stage 4: update Bayesian counters
/// - Stage 5: confidence gate — insert, merge, or defer to pending review
/// - Stage 6: persist (via `add_entries` calls in Stage 5)
/// - Stage 7: decay (handled by a separate CLI command)
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
        // Stage 1 — extract candidates
        let candidates = self.strategy.extract(input, params).await?;

        let mut result = PipelineResult {
            inserted: Vec::new(),
            merged: Vec::new(),
            pending_review: Vec::new(),
        };

        for candidate in candidates {
            // Stage 2 — dual search
            let direct_query = MemoryQuery {
                topic: candidate.content.clone(),
                max_results: self.config.direct_search_max_results,
                min_score: Score::new(self.config.direct_search_min_score).unwrap_or(Score::ZERO),
                filters: HashMap::new(),
                mode: MemoryQueryMode::Lookup,
            };
            let inverted_query = MemoryQuery {
                topic: format!("the opposite of {}", candidate.content),
                max_results: self.config.inverted_search_max_results,
                min_score: Score::new(self.config.inverted_search_min_score).unwrap_or(Score::ZERO),
                filters: HashMap::new(),
                mode: MemoryQueryMode::Lookup,
            };

            let direct_hits = self
                .store
                .query(direct_query)
                .await
                .map_err(MemoryExtractionError::MemoryStore)?;
            let inverted_hits = self
                .store
                .query(inverted_query)
                .await
                .map_err(MemoryExtractionError::MemoryStore)?;

            // Stage 3 — classify each hit
            let mut classified_hits: Vec<(MemoryQueryResult, HitClass)> = Vec::new();
            for hit in direct_hits {
                let class = self
                    .classifier
                    .classify_hit(&candidate.content, &hit.memory_entry.content)
                    .await
                    .map_err(|e| MemoryExtractionError::Other(e.to_string()))?;
                classified_hits.push((hit, class));
            }
            for hit in inverted_hits {
                let class = self
                    .classifier
                    .classify_hit(&candidate.content, &hit.memory_entry.content)
                    .await
                    .map_err(|e| MemoryExtractionError::Other(e.to_string()))?;
                classified_hits.push((hit, class));
            }

            // Stage 4 — update Bayesian counters
            let confidence = candidate.confidence.unwrap_or(0.0);
            let mut review = ReviewState::from_confidence(confidence);
            for (_, class) in &classified_hits {
                match class {
                    HitClass::Duplicate => {
                        review.alpha =
                            Some(review.alpha.unwrap_or(0.0) + self.config.duplicate_alpha_weight);
                    }
                    HitClass::Complementary => {
                        review.alpha = Some(
                            review.alpha.unwrap_or(0.0) + self.config.complementary_alpha_weight,
                        );
                    }
                    HitClass::Contradiction => {
                        review.beta = Some(
                            review.beta.unwrap_or(0.0) + self.config.contradiction_beta_weight,
                        );
                    }
                    HitClass::Unrelated => {}
                }
            }
            review.score = review.bayesian_confidence();

            // Stage 5 — confidence gate
            let c_new = review.bayesian_confidence().unwrap_or(0.0);
            let c_initial = confidence;

            if c_new >= c_initial {
                let mut seen_ids: HashSet<Uuid> = HashSet::new();
                let matching_ids: Vec<Uuid> = classified_hits
                    .iter()
                    .filter(|(_, class)| {
                        matches!(class, HitClass::Duplicate | HitClass::Complementary)
                    })
                    .map(|(hit, _)| hit.memory_entry.id)
                    .filter(|id| seen_ids.insert(*id))
                    .collect();

                if !matching_ids.is_empty() {
                    // Stage 5a — merge-reinsert
                    for id in &matching_ids {
                        self.store
                            .delete_entry(*id)
                            .await
                            .map_err(MemoryExtractionError::MemoryStore)?;
                    }
                    let merged_input = MemoryInput {
                        content: candidate.content.clone(),
                        metadata: candidate.metadata.clone(),
                        tier: candidate.tier,
                        confidence: Some(c_new),
                        review: review.clone(),
                    };
                    let add_result = self
                        .store
                        .add_entries(vec![merged_input])
                        .await
                        .map_err(MemoryExtractionError::MemoryStore)?;
                    result.merged.extend(add_result.added);
                } else {
                    // Stage 5b — fresh insert
                    let fresh_input = MemoryInput {
                        content: candidate.content.clone(),
                        metadata: candidate.metadata.clone(),
                        tier: candidate.tier,
                        confidence: Some(c_new),
                        review: review.clone(),
                    };
                    let add_result = self
                        .store
                        .add_entries(vec![fresh_input])
                        .await
                        .map_err(MemoryExtractionError::MemoryStore)?;
                    result.inserted.extend(add_result.added);
                }
            } else {
                // Stage 5c — pending review
                review.pending = true;
                let pending_input = MemoryInput {
                    content: candidate.content.clone(),
                    metadata: candidate.metadata.clone(),
                    tier: candidate.tier,
                    confidence: candidate.confidence,
                    review: review.clone(),
                };
                let add_result = self
                    .store
                    .add_entries(vec![pending_input])
                    .await
                    .map_err(MemoryExtractionError::MemoryStore)?;
                result.pending_review.extend(add_result.added);
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::future::Future;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use crate::classification::HitClass;
    use crate::error::MemoryExtractionError;
    use crate::memory::{MemoryInput, MemoryTier, Score};
    use crate::memory_extraction::MemoryExtractionStrategy;
    use crate::testing::{
        AddEntriesBehavior, ClassifyBehavior, MockClassificationModelProvider, MockStore,
        QueryBehavior, make_result,
    };

    use super::{MemoryExtractionPipeline, PipelineConfig};

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
            tier: Some(MemoryTier::Candidate),
            confidence: Some(confidence),
            review: Default::default(),
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
        let candidate = make_candidate("the sky is blue", 0.8);
        let inserted_result = make_result(
            uuid::Uuid::new_v4(),
            "the sky is blue",
            MemoryTier::Candidate,
            0.8,
        );

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
        assert!(result.pending_review.is_empty());
    }

    #[tokio::test]
    async fn test_gate_fail_path() {
        // confidence = 0.7 → alpha=7.0, beta=3.0
        // two Contradiction hits (direct + inverted query return same hit)
        // → beta += 3.0 * 2 = 9.0  → alpha=7.0, beta=9.0 → c_new ≈ 0.4375 < 0.7
        let candidate = make_candidate("the sky is green", 0.7);
        let hit = make_result(
            uuid::Uuid::new_v4(),
            "the sky is blue",
            MemoryTier::Candidate,
            0.9,
        );
        let pending_result = make_result(
            uuid::Uuid::new_v4(),
            "the sky is green",
            MemoryTier::Candidate,
            0.7,
        );

        let store = Arc::new(
            MockStore::new()
                .with_query_behavior(QueryBehavior::Ok(vec![hit]))
                .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![pending_result])),
        );
        let strategy = Arc::new(FixedStrategy(vec![candidate]));
        let classifier = Arc::new(
            MockClassificationModelProvider::new()
                .with_behavior(ClassifyBehavior::Ok(HitClass::Contradiction)),
        );

        let pipeline = pipeline(Arc::clone(&store), strategy, classifier);
        let result = pipeline.extract_and_store("ignored", ()).await.unwrap();

        assert_eq!(result.pending_review.len(), 1);
        assert!(result.inserted.is_empty());
        assert!(result.merged.is_empty());

        let state = store.snapshot();
        let inputs = state
            .add_inputs
            .expect("add_entries should have been called");
        assert!(inputs[0].review.pending, "entry must be marked pending");
    }

    #[tokio::test]
    async fn test_merge_path() {
        // confidence = 0.7 → alpha=7.0, beta=3.0
        // two Duplicate hits (direct + inverted return the same entry)
        // → alpha += 3.0 * 2 = 13.0 → c_new ≈ 0.8125 > 0.7 → gate passes
        // same ID deduplicated → 1 delete, 1 merged insert
        let hit_id = uuid::Uuid::new_v4();
        let candidate = make_candidate("cats have nine lives", 0.7);
        let hit = make_result(hit_id, "cats have nine lives", MemoryTier::Candidate, 0.9);
        let merged_result = make_result(
            uuid::Uuid::new_v4(),
            "cats have nine lives",
            MemoryTier::Candidate,
            Score::new(0.8125).unwrap().value(),
        );

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
        assert!(result.pending_review.is_empty());

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
        assert_eq!(config.direct_search_max_results, 5);
        assert!((config.direct_search_min_score - 0.70).abs() < f64::EPSILON);
        assert_eq!(config.inverted_search_max_results, 3);
        assert!((config.inverted_search_min_score - 0.60).abs() < f64::EPSILON);
        assert!((config.duplicate_alpha_weight - 3.0).abs() < f64::EPSILON);
        assert!((config.complementary_alpha_weight - 1.0).abs() < f64::EPSILON);
        assert!((config.contradiction_beta_weight - 3.0).abs() < f64::EPSILON);
        assert!((config.decay_rate - 0.99).abs() < f64::EPSILON);
    }
}

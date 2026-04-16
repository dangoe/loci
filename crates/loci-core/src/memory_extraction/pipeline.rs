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
    MemoryInput, MemoryQuery, MemoryQueryMode, MemoryQueryResult, ReviewState, Score,
};
use crate::memory_extraction::MemoryExtractionStrategy;
use crate::store::MemoryStore;

/// Configuration for the initial search stage.
#[derive(Debug, Clone)]
pub struct PipelineSearchResultsConfig {
    /// Maximum number of results to retrieve in semanticsearch.
    pub max_results: usize,
    /// Minimum score threshold for direct search results.
    pub min_score: f64,
}

/// Configuration for the memory extraction pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Configuration for direct semantic search stage.
    pub direct_search: PipelineSearchResultsConfig,
    /// Configuration for inverted semantic search stage.
    pub inverted_search: PipelineSearchResultsConfig,
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
            direct_search: PipelineSearchResultsConfig {
                max_results: 5,
                min_score: 0.70,
            },
            inverted_search: PipelineSearchResultsConfig {
                max_results: 3,
                min_score: 0.60,
            },
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
        let candidates = self.strategy.extract(input, params).await?;

        let mut result = PipelineResult {
            inserted: Vec::new(),
            merged: Vec::new(),
            pending_review: Vec::new(),
        };

        for candidate in candidates {
            let classified_hits = self.collect_classified_hits(&candidate).await?;

            let confidence = candidate.confidence.unwrap_or(0.0);
            let mut review_state = self.initialize_review_state(&classified_hits, confidence);

            let c_new = review_state.bayesian_confidence().unwrap_or(0.0);

            if c_new >= confidence {
                let match_ids = extract_match_ids(classified_hits);

                if !match_ids.is_empty() {
                    result.merged.extend(
                        self.merge_and_add(&candidate, &review_state, c_new, match_ids)
                            .await?,
                    );
                } else {
                    let input = MemoryInput {
                        content: candidate.content.clone(),
                        metadata: candidate.metadata.clone(),
                        tier: candidate.tier,
                        confidence: Some(c_new),
                        review: review_state.clone(),
                    };
                    result
                        .inserted
                        .extend(self.add_entries_or_fail(vec![input]).await?);
                }
            } else {
                review_state.pending = true;
                let pending_input = MemoryInput {
                    content: candidate.content.clone(),
                    metadata: candidate.metadata.clone(),
                    tier: candidate.tier,
                    confidence: candidate.confidence,
                    review: review_state.clone(),
                };

                result.pending_review.extend(
                    self.add_entries_or_fail(vec![pending_input.clone()])
                        .await?,
                );
            }
        }

        Ok(result)
    }

    async fn merge_and_add(
        &self,
        candidate: &MemoryInput,
        review_state: &ReviewState,
        c_new: f64,
        match_ids: Vec<Uuid>,
    ) -> Result<Vec<MemoryQueryResult>, MemoryExtractionError> {
        let merged_input = MemoryInput {
            content: candidate.content.clone(),
            metadata: candidate.metadata.clone(),
            tier: candidate.tier,
            confidence: Some(c_new),
            review: review_state.clone(),
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

    fn initialize_review_state(
        &self,
        classified_hits: &Vec<(MemoryQueryResult, HitClass)>,
        confidence: f64,
    ) -> ReviewState {
        let mut review = ReviewState::from_confidence(confidence);

        for (_, class) in classified_hits {
            match class {
                HitClass::Duplicate => {
                    review.alpha =
                        Some(review.alpha.unwrap_or(0.0) + self.config.duplicate_alpha_weight);
                }
                HitClass::Complementary => {
                    review.alpha =
                        Some(review.alpha.unwrap_or(0.0) + self.config.complementary_alpha_weight);
                }
                HitClass::Contradiction => {
                    review.beta =
                        Some(review.beta.unwrap_or(0.0) + self.config.contradiction_beta_weight);
                }
                HitClass::Unrelated => {}
            }
        }
        review.score = review.bayesian_confidence();

        review
    }

    async fn collect_classified_hits(
        &self,
        candidate: &MemoryInput,
    ) -> Result<Vec<(MemoryQueryResult, HitClass)>, MemoryExtractionError> {
        let mut classified_hits: Vec<(MemoryQueryResult, HitClass)> = Vec::new();

        for direct_hit in self.query_direct_hits(&candidate).await? {
            let class = self
                .classifier
                .classify_hit(&candidate.content, &direct_hit.memory_entry.content)
                .await
                .map_err(|e| MemoryExtractionError::Other(e.to_string()))?;
            classified_hits.push((direct_hit, class));
        }

        for inverted_hit in self.query_inverted_hits(&candidate).await? {
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

fn extract_match_ids(classified_hits: Vec<(MemoryQueryResult, HitClass)>) -> Vec<Uuid> {
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
        assert_eq!(config.direct_search.max_results, 5);
        assert!((config.direct_search.min_score - 0.70).abs() < f64::EPSILON);
        assert_eq!(config.inverted_search.max_results, 3);
        assert!((config.inverted_search.min_score - 0.60).abs() < f64::EPSILON);
        assert!((config.duplicate_alpha_weight - 3.0).abs() < f64::EPSILON);
        assert!((config.complementary_alpha_weight - 1.0).abs() < f64::EPSILON);
        assert!((config.contradiction_beta_weight - 3.0).abs() < f64::EPSILON);
        assert!((config.decay_rate - 0.99).abs() < f64::EPSILON);
    }
}

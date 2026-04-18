// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

pub mod chunker;
pub mod llm;
pub mod pipeline;

pub use chunker::{Chunker, SentenceAwareChunker};
pub use llm::{LlmMemoryExtractionStrategy, LlmMemoryExtractionStrategyParams};
use log::{debug, info};
pub use pipeline::{
    MemoryExtractionPipeline, MemoryExtractionResult, MemoryExtrationPipelineConfig,
    PipelineSearchResultsConfig,
};

use std::{future::Future, marker::PhantomData, sync::Arc};

use crate::{
    error::MemoryExtractionError,
    memory::MemoryInput,
    store::{AddEntriesResult, MemoryStore},
};

/// Extracts memory entries from a source.
pub trait MemoryExtractionStrategy<P>: Send + Sync {
    fn extract(
        &self,
        input: &str,
        params: P,
    ) -> impl Future<Output = Result<Vec<MemoryInput>, MemoryExtractionError>> + Send;
}

/// Orchestrates a [`MemoryExtractionStrategy`] and a [`MemoryStore`]:
/// extracts entries from text then persists them in a single call.
pub struct MemoryExtractor<S: MemoryStore, E: MemoryExtractionStrategy<P>, P> {
    memory_store: Arc<S>,
    memory_extraction_strategy: Arc<E>,
    phantom: PhantomData<P>,
}

impl<S: MemoryStore, E: MemoryExtractionStrategy<P>, P: Send + Sync> MemoryExtractor<S, E, P> {
    /// Creates an extractor that takes ownership of `memory_store` and
    /// `memory_extraction_strategy`, wrapping both in `Arc`.
    pub fn new(memory_store: S, memory_extraction_strategy: E) -> Self {
        Self {
            memory_store: Arc::new(memory_store),
            memory_extraction_strategy: Arc::new(memory_extraction_strategy),
            phantom: PhantomData,
        }
    }

    /// Creates an extractor from pre-existing `Arc` handles — useful when the
    /// store or strategy is already shared with another component.
    pub fn from_arcs(memory_store: Arc<S>, memory_extraction_strategy: Arc<E>) -> Self {
        Self {
            memory_store,
            memory_extraction_strategy,
            phantom: PhantomData,
        }
    }

    /// Extracts memory entries from `input` using `params`, optionally splits
    /// the input into chunks first, then persists all results.
    pub fn extract_and_store(
        &self,
        input: &str,
        params: P,
    ) -> impl Future<Output = Result<AddEntriesResult, MemoryExtractionError>> + Send
    where
        P: Clone,
    {
        info!("Extracting memory from {}", input);

        let strategy = Arc::clone(&self.memory_extraction_strategy);
        let store = Arc::clone(&self.memory_store);

        async move {
            let entries = strategy.extract(input, params.clone()).await?;

            debug!("Successfully extracted memory entries: {:#?}", entries);
            info!("Extracted {} entries, storing...", entries.len());

            store
                .add_entries(entries)
                .await
                .map_err(MemoryExtractionError::MemoryStore)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, future::Future};

    use pretty_assertions::assert_eq;

    use crate::{
        memory::{MemoryEntry, MemoryInput, MemoryQueryResult, Score},
        testing::{AddEntriesBehavior, MockStore},
    };

    use super::{MemoryExtractionStrategy, MemoryExtractor};

    struct FixedStrategy(Vec<MemoryInput>);

    impl MemoryExtractionStrategy<()> for FixedStrategy {
        fn extract(
            &self,
            _input: &str,
            _params: (),
        ) -> impl Future<Output = Result<Vec<MemoryInput>, crate::error::MemoryExtractionError>> + Send
        {
            let entries = self.0.clone();
            async move { Ok(entries) }
        }
    }

    fn make_input(content: &str) -> MemoryInput {
        MemoryInput {
            content: content.to_string(),
            metadata: HashMap::new(),
            trust: None,
        }
    }

    fn make_query_result(content: &str) -> MemoryQueryResult {
        MemoryQueryResult {
            memory_entry: MemoryEntry::new(content.to_string(), HashMap::new()),
            score: Score::MAX,
        }
    }

    #[tokio::test]
    async fn test_all_candidates_reach_store() {
        let inputs = vec![make_input("fact one"), make_input("fact two")];
        let strategy = FixedStrategy(inputs.clone());
        let store = MockStore::new().with_add_entries_behavior(AddEntriesBehavior::Ok(
            inputs
                .iter()
                .map(|i| make_query_result(&i.content))
                .collect(),
        ));

        let result = MemoryExtractor::new(store, strategy)
            .extract_and_store("ignored", ())
            .await
            .unwrap();

        assert_eq!(result.added.len(), 2);
    }
}

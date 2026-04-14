// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

pub mod llm;

pub use llm::{LlmMemoryExtractionStrategy, LlmMemoryExtractionStrategyParams};

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

    /// Extracts memory entries from `input` using `params`, then persists them.
    ///
    /// Both `Arc` handles are cloned before entering the async block so the
    /// returned future is independent of `&self`'s lifetime.
    pub fn extract_and_store(
        &self,
        input: &str,
        params: P,
    ) -> impl Future<Output = Result<AddEntriesResult, MemoryExtractionError>> + Send {
        let strategy = Arc::clone(&self.memory_extraction_strategy);
        let store = Arc::clone(&self.memory_store);
        let input = input.to_owned();

        async move {
            let memory_inputs = strategy.extract(&input, params).await?;
            store
                .add_entries(memory_inputs)
                .await
                .map_err(MemoryExtractionError::MemoryStore)
        }
    }
}

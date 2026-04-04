// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-core.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;

use crate::model_provider::text_generation::{self, TextGenerationRequest, TextGenerationResponse};
use crate::error::ContextualizerError;
use crate::memory::{MemoryEntry, MemoryQuery, Score};
use crate::store::MemoryStore;

/// Configuration for a [`Contextualizer`].
///
/// This configuration controls how many memories are retrieved and how the
/// memory store is queried, as well as which text generation model to use
/// for downstream requests.
#[derive(Debug, Clone)]
pub struct ContextualizerConfig {
    /// The model to use for text generation.
    ///
    /// Note: the default value is an empty string. Callers should provide a
    /// fully initialized `ContextualizerConfig` when constructing a `Contextualizer`.
    pub text_generation_model: String,
    /// Maximum number of memories to retrieve and inject into the prompt.
    pub max_memories: usize,
    /// Minimum similarity score a memory must have to be included.
    pub min_score: Score,
    /// Metadata filters applied when querying the memory store.
    pub filters: HashMap<String, String>,
}

/// Enhances a user prompt with relevant memories before calling an LLM.
///
/// On each call to [`Contextualizer::contextualize`] the contextualizer:
/// 1. Queries the memory store for entries relevant to the prompt.
/// 2. Constructs a system prompt that summarizes retrieved memories (the
///    preferred mechanism to condition many LLMs).
/// 3. Calls the configured text-generation model provider with the enriched request.
/// 4. Streams model output back to the caller.
///
/// The `Contextualizer` is generic over a `MemoryStore` implementation and a
/// text generation model provider. It keeps `Arc` references to each dependency so
/// instances can be cheaply cloned or shared across threads.
pub struct Contextualizer<M: MemoryStore, E: text_generation::TextGenerationModelProvider> {
    memory_store: Arc<M>,
    text_generation_provider: Arc<E>,
    config: ContextualizerConfig,
}

impl<M, E> Contextualizer<M, E>
where
    M: MemoryStore + 'static,
    E: text_generation::TextGenerationModelProvider + 'static,
{
    /// Creates a new `Contextualizer` with the provided configuration.
    ///
    /// The `memory_store` and `text_generation_provider` are stored by `Arc`
    /// for cheap cloning and concurrent use. Supply a properly initialized
    /// `ContextualizerConfig`.
    pub fn new(
        memory_store: Arc<M>,
        text_generation_provider: Arc<E>,
        config: ContextualizerConfig,
    ) -> Self {
        Self {
            memory_store,
            text_generation_provider,
            config,
        }
    }

    /// Contextualize a user prompt and stream model responses.
    ///
    /// The returned stream yields `TextGenerationResponse` items produced by the
    /// configured text-generation model provider. If memory retrieval or the model
    /// provider call fails the stream will yield `Err(ContextualizerError)`.
    ///
    /// The `prompt` is the user-provided input; relevant memories are retrieved
    /// from the `MemoryStore` and attached to the request as a system prompt.
    pub fn contextualize<'a>(
        &'a self,
        prompt: &'a str,
    ) -> Pin<Box<dyn Stream<Item = Result<TextGenerationResponse, ContextualizerError>> + Send + 'a>>
    {
        Box::pin(async_stream::try_stream! {
            let memory_entries = self.query_memory(prompt).await?;

            log::debug!("retrieved {} relevant memories", memory_entries.len());

            let system_prompt = self.build_system_prompt(memory_entries);

            let req = TextGenerationRequest::new(
                self.config.text_generation_model.to_string(),
                prompt,
            ).with_system(system_prompt);

            use futures::StreamExt as _;
            let mut stream = self.text_generation_provider.generate_stream(req);
            while let Some(result) = stream.next().await {
                let chunk = result.map_err(ContextualizerError::ModelProvider)?;
                yield chunk;
            }
        })
    }

    fn build_system_prompt(&self, memory_entries: Vec<MemoryEntry>) -> String {
        let mut buf = String::new();
        buf.push_str("You are a helpful, personable assistant with a private long-term memory.\n");
        buf.push_str("When you use the items below, present them as things you remember (e.g. \"I remember that...\").\n");
        buf.push_str("Do NOT mention retrieval, the memory store, the word \"context\", or any system internals. Do not say \"this was provided\" or \"retrieved memories\".\n");
        buf.push_str("Be honest: do not fabricate. If none of the memories apply, state that you have no relevant memory and proceed from general knowledge.\n");
        buf.push_str("Memories:\n");

        if !memory_entries.is_empty() {
            for entry in memory_entries {
                buf.push_str(&format!("- {}\n", entry.memory.content));
            }
        } else {
            buf.push_str("<no memory entries>\n");
        }
        buf
    }

    async fn query_memory(
        &self,
        prompt: impl Into<String>,
    ) -> Result<Vec<MemoryEntry>, ContextualizerError> {
        let query = MemoryQuery {
            topic: prompt.into(),
            max_results: self.config.max_memories,
            min_score: self.config.min_score,
            filters: self.config.filters.clone(),
        };

        self.memory_store
            .query(query)
            .await
            .map_err(ContextualizerError::MemoryStore)
    }
}

#[cfg(test)]
mod tests {
    use std::{future::Future, pin::Pin};

    use uuid::Uuid;

    use crate::{
        model_provider::{
            common::ModelProviderResult,
            text_generation::{TextGenerationModelProvider, TextGenerationResponse},
        },
        error::MemoryStoreError,
        memory::{Memory, MemoryInput, MemoryQuery, Score},
    };

    use super::*;

    // ── Mock store ────────────────────────────────────────────────────────────

    struct MockStore {
        entries: Vec<MemoryEntry>,
    }

    impl MemoryStore for MockStore {
        fn save(
            &self,
            input: MemoryInput,
        ) -> impl Future<Output = Result<MemoryEntry, MemoryStoreError>> + Send + '_ {
            let entry = MemoryEntry {
                memory: Memory::new(input.content, input.metadata),
                score: Score::ZERO,
            };
            async move { Ok(entry) }
        }

        fn query(
            &self,
            _query: MemoryQuery,
        ) -> impl Future<Output = Result<Vec<MemoryEntry>, MemoryStoreError>> + Send + '_ {
            let entries = self.entries.clone();
            async move { Ok(entries) }
        }

        fn update(
            &self,
            _id: Uuid,
            _input: MemoryInput,
        ) -> impl Future<Output = Result<MemoryEntry, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::NotFound(_id)) }
        }

        fn delete(
            &self,
            _id: Uuid,
        ) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_ {
            async move { Ok(()) }
        }

        fn clear(&self) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_ {
            async move { Ok(()) }
        }
    }

    struct MockTextGenerationProvider {
        reply: String,
    }

    impl TextGenerationModelProvider for MockTextGenerationProvider {
        fn generate(
            &self,
            req: TextGenerationRequest,
        ) -> Pin<Box<dyn Future<Output = ModelProviderResult<TextGenerationResponse>> + Send + '_>>
        {
            let reply = self.reply.clone();
            Box::pin(async move { Ok(TextGenerationResponse::done(reply, req.model, None)) })
        }
    }
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-core.

use std::collections::HashMap;
use std::sync::Arc;

use crate::backend::text_generation::{self, TextGenerationRequest};
use crate::error::ContextualizerError;
use crate::memory::{MemoryEntry, MemoryQuery, Score};
use crate::store::MemoryStore;

/// Configuration for a [`Contextualizer`].
#[derive(Debug, Clone)]
pub struct ContextualizerConfig {
    /// Maximum number of memories to retrieve and inject into the prompt.
    pub max_memories: usize,
    /// Minimum similarity score a memory must have to be included.
    pub min_score: Score,
    /// Metadata filters applied when querying the memory store.
    pub filters: HashMap<String, String>,
    /// The model to use for text generation.
    pub text_generation_model: String,
}

impl Default for ContextualizerConfig {
    fn default() -> Self {
        Self {
            max_memories: 5,
            min_score: Score::ZERO,
            filters: HashMap::new(),
            text_generation_model: String::new(),
        }
    }
}

/// Enhances a user prompt with relevant memories before calling an LLM.
///
/// On each call to [`ContextEnhancer::enhance`] the enhancer:
/// 1. Queries the memory store for entries relevant to the prompt.
/// 2. Prepends a `[MEMORY CONTEXT]` block to the prompt when memories exist.
/// 3. Calls the LLM with the enriched prompt.
/// 4. Optionally extracts new memories from the exchange in a background task.
pub struct Contextualizer<M: MemoryStore, E: text_generation::TextGenerationBackend> {
    memory_store: Arc<M>,
    text_generation_backend: Arc<E>,
    config: ContextualizerConfig,
}

impl<M, E> Contextualizer<M, E>
where
    M: MemoryStore + 'static,
    E: text_generation::TextGenerationBackend + 'static,
{
    /// Creates a new `ContextEnhancer` with default configuration.
    pub fn new(memory_store: Arc<M>, text_generation_backend: Arc<E>) -> Self {
        Self {
            memory_store,
            text_generation_backend,
            config: ContextualizerConfig::default(),
        }
    }

    /// Applies a custom [`EnhancerConfig`].
    pub fn with_config(mut self, config: ContextualizerConfig) -> Self {
        self.config = config;
        self
    }

    /// Enhances `prompt` with memory context and returns the LLM response.
    ///
    /// # Errors
    ///
    /// Returns [`EnhancerError::MemoryStore`] if the store query fails, or
    /// [`EnhancerError::Llm`] if the LLM call fails.
    pub async fn enhance(&self, prompt: &str) -> Result<String, ContextualizerError> {
        let memory_entries = self.query_memory(prompt).await?;

        log::debug!("retrieved {} relevant memories", memory_entries.len());

        let augmented_prompt = self.augment_prompt(prompt, memory_entries);

        self.generate_text(augmented_prompt).await
    }

    async fn query_memory(&self, prompt: &str) -> Result<Vec<MemoryEntry>, ContextualizerError> {
        let query = MemoryQuery {
            topic: prompt.to_string(),
            max_results: self.config.max_memories,
            min_score: self.config.min_score,
            filters: self.config.filters.clone(),
        };

        self.memory_store
            .query(query)
            .await
            .map_err(ContextualizerError::MemoryStore)
    }

    fn augment_prompt(&self, prompt: &str, memory_entries: Vec<MemoryEntry>) -> String {
        if memory_entries.is_empty() {
            prompt.to_string()
        } else {
            let mut buf = String::from("[MEMORY CONTEXT]\n");
            buf.push_str("The following memories are relevant to your request:\n\n");

            for (i, entry) in memory_entries.iter().enumerate() {
                buf.push_str(&format!(
                    "{}. (relevance: {:.2}) {}\n",
                    i + 1,
                    entry.score.value(),
                    entry.memory.content,
                ));
            }

            buf.push_str("\n[USER PROMPT]\n");
            buf.push_str(prompt);
            buf
        }
    }

    async fn generate_text(&self, augmented_prompt: String) -> Result<String, ContextualizerError> {
        self.text_generation_backend
            .generate(TextGenerationRequest::new(
                self.config.text_generation_model.to_string(),
                augmented_prompt,
            ))
            .await
            .map(|r| r.text)
            .map_err(ContextualizerError::RemoteModel)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::{future::Future, pin::Pin};

    use uuid::Uuid;

    use crate::{
        backend::{
            common::BackendResult,
            text_generation::{TextGenerationBackend, TextGenerationResponse},
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

    // ── Mock LLM ─────────────────────────────────────────────────────────────

    struct MockTextGenerationBackend {
        reply: String,
    }

    impl TextGenerationBackend for MockTextGenerationBackend {
        fn generate(
            &self,
            req: TextGenerationRequest,
        ) -> Pin<Box<dyn Future<Output = BackendResult<TextGenerationResponse>> + Send + '_>>
        {
            let reply = self.reply.clone();
            Box::pin(async move { Ok(TextGenerationResponse::done(reply, req.model, None)) })
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn enhance_no_memories_returns_plain_response() {
        let store = Arc::new(MockStore { entries: vec![] });
        let llm = Arc::new(MockTextGenerationBackend {
            reply: "Hello!".to_string(),
        });

        let enhancer = Contextualizer::new(store, llm);
        let result = enhancer.enhance("hi").await.unwrap();
        assert_eq!(result, "Hello!");
    }

    #[tokio::test]
    async fn enhance_with_memories_injects_context() {
        use std::collections::HashMap;

        let entry = MemoryEntry {
            memory: Memory::new("user likes Rust".to_string(), HashMap::new()),
            score: Score::new(0.9).unwrap(),
        };

        // Capture the message sent to the LLM via a channel.
        let store = Arc::new(MockStore {
            entries: vec![entry],
        });

        struct CaptureTextGenerationBackend {
            tx: std::sync::Mutex<std::sync::mpsc::SyncSender<String>>,
        }

        impl TextGenerationBackend for CaptureTextGenerationBackend {
            fn generate(
                &self,
                req: TextGenerationRequest,
            ) -> Pin<Box<dyn Future<Output = BackendResult<TextGenerationResponse>> + Send + '_>>
            {
                let content = req.prompt.clone();
                self.tx.lock().unwrap().send(content).unwrap();
                Box::pin(async move {
                    Ok(TextGenerationResponse::done(
                        "noted".to_string(),
                        req.model,
                        None,
                    ))
                })
            }
        }

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let llm = Arc::new(CaptureTextGenerationBackend {
            tx: std::sync::Mutex::new(tx),
        });

        let enhancer = Contextualizer::new(store, llm);
        enhancer.enhance("what do I like?").await.unwrap();

        let sent = rx.try_recv().unwrap();
        assert!(sent.contains("[MEMORY CONTEXT]"), "missing context block");
        assert!(sent.contains("user likes Rust"), "missing memory content");
        assert!(sent.contains("[USER PROMPT]"), "missing user prompt block");
        assert!(sent.contains("what do I like?"), "missing original prompt");
    }
}

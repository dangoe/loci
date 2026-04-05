// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-core.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::Stream;

use crate::error::ContextualizerError;
use crate::memory::{MemoryQuery, MemoryQueryMode, MemoryQueryResult, Score};
use crate::model_provider::common::ModelProviderParams;
use crate::model_provider::text_generation::{self, TextGenerationRequest, TextGenerationResponse};
use crate::store::MemoryStore;

const SYSTEM_PROMPT_BASE_TEMPLATE: &str = "You are a helpful assistant with a long-term memory of past conversations.
Rules:
- Speak naturally. Never mention retrieval, context, memory stores, or system internals.
- When drawing on a memory, phrase it as personal recall: \"I remember...\", \"You mentioned once...\", etc.
- Only use memory_entries that are clearly relevant to the current question. Ignore the rest silently.
- If no memory applies, answer from general knowledge without commenting on the absence of memory.
- Never fabricate or embellish. If you are uncertain, say so. Only speculate if the user explicitly asks you to.
- Be concise. Avoid unnecessary preamble.";

/// Configuration for a [`Contextualizer`].
///
/// This configuration controls how many memory entries are retrieved and how the
/// memory store is queried, as well as which text generation model to use
/// for downstream requests.
#[derive(Debug, Clone)]
pub struct ContextualizerConfig {
    /// The model to use for text generation.
    ///
    /// Note: the default value is an empty string. Callers should provide a
    /// fully initialized `ContextualizerConfig` when constructing a `Contextualizer`.
    pub text_generation_model: String,
    /// Maximum number of memory entries to retrieve and inject into the prompt.
    pub max_memory_entries: usize,
    /// Minimum similarity score a memory must have to be included.
    pub min_score: Score,
    /// Metadata filters applied when querying the memory store.
    pub filters: HashMap<String, String>,
    /// Optional text-generation tuning parameters.
    pub tuning: Option<ContextualizerTuningConfig>,
}

/// Provider-agnostic generation tuning options for contextualized prompts.
#[derive(Debug, Clone, Default)]
pub struct ContextualizerTuningConfig {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<u32>,
    pub thinking: Option<text_generation::ThinkingMode>,
    pub stop: Option<Vec<String>>,
    pub keep_alive: Option<Duration>,
    pub extra_params: ModelProviderParams,
}

/// Debug information produced by a contextualized prompt call.
///
/// Contains the memory entries that were retrieved from the store and injected
/// into the system prompt for this request.
#[derive(Debug, Clone)]
pub struct ContextualizationDebugInfo {
    /// The memory entries that were retrieved and injected into the prompt.
    pub memory_entries: Vec<MemoryQueryResult>,
}

/// Enhances a user prompt with relevant memory entries before calling an LLM.
///
/// On each call to [`Contextualizer::contextualize`] the contextualizer:
/// 1. Queries the memory store for entries relevant to the prompt.
/// 2. Constructs a system prompt that summarizes retrieved memory entries (the
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
    /// The `prompt` is the user-provided input; relevant memory entries are retrieved
    /// from the `MemoryStore` and attached to the request as a system prompt.
    pub fn contextualize<'a>(
        &'a self,
        prompt: &'a str,
    ) -> Pin<Box<dyn Stream<Item = Result<TextGenerationResponse, ContextualizerError>> + Send + 'a>>
    {
        Box::pin(async_stream::try_stream! {
            let memory_entries = self.query_memory(prompt).await?;

            log::debug!("retrieved {} relevant memory_entries", memory_entries.len());

            let req = self.build_request(prompt, &memory_entries);

            use futures::StreamExt as _;
            let mut stream = self.text_generation_provider.generate_stream(req);
            while let Some(result) = stream.next().await {
                let chunk = result.map_err(ContextualizerError::ModelProvider)?;
                yield chunk;
            }
        })
    }

    /// Contextualize a user prompt, returning debug information about the injected
    /// memory entries alongside the text-generation stream.
    ///
    /// This is identical to [`Contextualizer::contextualize`] but also returns a
    /// [`ContextualizationDebugInfo`] that describes which memory entries were
    /// retrieved and injected into the system prompt.
    ///
    /// # Errors
    ///
    /// Returns [`ContextualizerError`] if the memory store query fails before the
    /// stream is produced. Errors that occur during streaming are yielded by the
    /// returned stream.
    pub async fn contextualize_with_debug<'a>(
        &'a self,
        prompt: &'a str,
    ) -> Result<
        (
            ContextualizationDebugInfo,
            Pin<
                Box<
                    dyn Stream<Item = Result<TextGenerationResponse, ContextualizerError>>
                        + Send
                        + 'a,
                >,
            >,
        ),
        ContextualizerError,
    > {
        let memory_entries = self.query_memory(prompt).await?;

        log::debug!(
            "retrieved {} relevant memory_entries (debug mode)",
            memory_entries.len()
        );

        let debug_info = ContextualizationDebugInfo {
            memory_entries: memory_entries.clone(),
        };

        let req = self.build_request(prompt, &memory_entries);

        let provider = self.text_generation_provider.clone();
        let stream = Box::pin(async_stream::try_stream! {
            use futures::StreamExt as _;
            let mut s = provider.generate_stream(req);
            while let Some(result) = s.next().await {
                let chunk = result.map_err(ContextualizerError::ModelProvider)?;
                yield chunk;
            }
        });

        Ok((debug_info, stream))
    }

    fn build_request(
        &self,
        prompt: &str,
        memory_entries: &Vec<MemoryQueryResult>,
    ) -> TextGenerationRequest {
        let mut req =
            TextGenerationRequest::new(self.config.text_generation_model.to_string(), prompt)
                .with_system(self.build_system_prompt(memory_entries));

        if let Some(tuning) = &self.config.tuning {
            if let Some(v) = tuning.temperature {
                req = req.with_temperature(v);
            }
            if let Some(v) = tuning.max_tokens {
                req = req.with_max_tokens(v);
            }
            if let Some(v) = tuning.top_p {
                req = req.with_top_p(v);
            }
            if let Some(v) = tuning.repeat_penalty {
                req = req.with_repeat_penalty(v);
            }
            if let Some(v) = tuning.repeat_last_n {
                req = req.with_repeat_last_n(v);
            }
            if let Some(v) = &tuning.thinking {
                req = req.with_thinking(v.clone());
            }
            if let Some(v) = &tuning.stop {
                req = req.with_stop(v.clone());
            }
            if let Some(v) = tuning.keep_alive {
                req = req.with_keep_alive(v);
            }
            for (k, v) in &tuning.extra_params {
                req = req.with_extra(k.clone(), v.clone());
            }
        }

        req
    }

    fn build_system_prompt(&self, memory_entries: &Vec<MemoryQueryResult>) -> String {
        let mut buf = String::new();
        buf.push_str(SYSTEM_PROMPT_BASE_TEMPLATE.replace('\n', "\n- ").as_str());
        buf.push('\n');
        buf.push_str("## Relevant memory entries\n");

        if memory_entries.is_empty() {
            buf.push_str("None. Answer from general knowledge.\n");
        } else {
            for entry in memory_entries {
                buf.push_str(&format!("- {}\n", entry.memory_entry.content));
            }
        }

        buf
    }

    async fn query_memory(
        &self,
        prompt: impl Into<String>,
    ) -> Result<Vec<MemoryQueryResult>, ContextualizerError> {
        let query = MemoryQuery {
            topic: prompt.into(),
            max_results: self.config.max_memory_entries,
            min_score: self.config.min_score,
            filters: self.config.filters.clone(),
            mode: MemoryQueryMode::Use,
        };

        self.memory_store
            .query(query)
            .await
            .map_err(ContextualizerError::MemoryStore)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;
    use std::{collections::HashMap, future::Future, pin::Pin, sync::Arc};

    use futures::StreamExt as _;
    use pretty_assertions::assert_eq;
    use serde_json::json;
    use uuid::Uuid;

    use crate::{
        error::MemoryStoreError,
        memory::{
            MemoryEntry, MemoryInput, MemoryQuery, MemoryQueryMode, MemoryQueryResult, MemoryTier,
            Score,
        },
        model_provider::{
            common::ModelProviderResult,
            error::ModelProviderError,
            text_generation::{
                TextGenerationModelProvider, TextGenerationRequest, TextGenerationResponse,
                ThinkingMode,
            },
        },
    };

    use super::*;

    struct MockStore {
        entries: Vec<MemoryQueryResult>,
    }

    impl MemoryStore for MockStore {
        fn save(
            &self,
            input: MemoryInput,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            let entry = MemoryQueryResult {
                memory_entry: MemoryEntry::new(input.content, input.metadata),
                score: Score::ZERO,
            };
            async move { Ok(entry) }
        }

        fn get(
            &self,
            id: Uuid,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::NotFound(id)) }
        }

        fn query(
            &self,
            _query: MemoryQuery,
        ) -> impl Future<Output = Result<Vec<MemoryQueryResult>, MemoryStoreError>> + Send + '_
        {
            let entries = self.entries.clone();
            async move { Ok(entries) }
        }

        fn update(
            &self,
            id: Uuid,
            _input: MemoryInput,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::NotFound(id)) }
        }

        fn set_tier(
            &self,
            id: Uuid,
            _tier: MemoryTier,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::NotFound(id)) }
        }

        fn delete(
            &self,
            _id: Uuid,
        ) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_ {
            async move { Ok(()) }
        }

        fn prune_expired(&self) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_ {
            async move { Ok(()) }
        }
    }

    struct FailingStore;

    impl MemoryStore for FailingStore {
        fn save(
            &self,
            _input: MemoryInput,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::Connection("always fails".to_string())) }
        }

        fn get(
            &self,
            _id: Uuid,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::Connection("always fails".to_string())) }
        }

        fn query(
            &self,
            _query: MemoryQuery,
        ) -> impl Future<Output = Result<Vec<MemoryQueryResult>, MemoryStoreError>> + Send + '_
        {
            async move { Err(MemoryStoreError::Connection("always fails".to_string())) }
        }

        fn update(
            &self,
            id: Uuid,
            _input: MemoryInput,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::NotFound(id)) }
        }

        fn set_tier(
            &self,
            id: Uuid,
            _tier: MemoryTier,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::NotFound(id)) }
        }

        fn delete(
            &self,
            _id: Uuid,
        ) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_ {
            async move { Ok(()) }
        }

        fn prune_expired(&self) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_ {
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

    struct FailingTextGenerationProvider;

    impl TextGenerationModelProvider for FailingTextGenerationProvider {
        fn generate(
            &self,
            _req: TextGenerationRequest,
        ) -> Pin<Box<dyn Future<Output = ModelProviderResult<TextGenerationResponse>> + Send + '_>>
        {
            Box::pin(async move { Err(ModelProviderError::Timeout) })
        }
    }

    fn default_config() -> ContextualizerConfig {
        ContextualizerConfig {
            text_generation_model: "test-model".to_string(),
            max_memory_entries: 5,
            min_score: Score::ZERO,
            filters: HashMap::new(),
            tuning: None,
        }
    }

    fn make_contextualizer(
        entries: Vec<MemoryQueryResult>,
        reply: &str,
    ) -> Contextualizer<MockStore, MockTextGenerationProvider> {
        Contextualizer::new(
            Arc::new(MockStore { entries }),
            Arc::new(MockTextGenerationProvider {
                reply: reply.to_string(),
            }),
            default_config(),
        )
    }

    fn make_entry(content: &str) -> MemoryQueryResult {
        MemoryQueryResult {
            memory_entry: MemoryEntry::new(content.to_string(), HashMap::new()),
            score: Score::ZERO,
        }
    }

    #[test]
    fn test_build_system_prompt_includes_each_memory_content() {
        let ctx = make_contextualizer(vec![], "reply");
        let entries = vec![make_entry("I like cats"), make_entry("I live in Berlin")];
        let prompt = ctx.build_system_prompt(&entries);
        assert!(prompt.contains("I like cats"), "prompt: {prompt}");
        assert!(prompt.contains("I live in Berlin"), "prompt: {prompt}");
    }

    #[test]
    fn test_build_system_prompt_with_no_memory_entries_contains_placeholder() {
        let ctx = make_contextualizer(vec![], "reply");
        let prompt = ctx.build_system_prompt(&vec![]);
        assert!(
            prompt.contains("None. Answer from general knowledge."),
            "expected placeholder, got: {prompt}",
        );
    }

    #[derive(Default)]
    struct CapturingTextGenerationProvider {
        last_req: Mutex<Option<TextGenerationRequest>>,
    }

    impl TextGenerationModelProvider for CapturingTextGenerationProvider {
        fn generate(
            &self,
            req: TextGenerationRequest,
        ) -> Pin<Box<dyn Future<Output = ModelProviderResult<TextGenerationResponse>> + Send + '_>>
        {
            *self.last_req.lock().unwrap() = Some(req.clone());
            Box::pin(async move {
                Ok(TextGenerationResponse::done(
                    "ok".to_string(),
                    req.model,
                    None,
                ))
            })
        }
    }

    #[tokio::test]
    async fn test_contextualize_returns_model_response_text() {
        let ctx = make_contextualizer(vec![], "the answer is 42");
        let items: Vec<_> = ctx.contextualize("what is the answer?").collect().await;

        assert_eq!(items.len(), 1);
        let resp = items.into_iter().next().unwrap().unwrap();
        assert_eq!(resp.text, "the answer is 42");
    }

    #[tokio::test]
    async fn test_contextualize_injects_retrieved_memory_entries_into_system_prompt() {
        // The mock store returns this entry regardless of query topic.
        let entries = vec![make_entry("user prefers dark mode")];
        let ctx = make_contextualizer(entries, "noted");
        let items: Vec<_> = ctx.contextualize("set my preference").collect().await;

        // The stream should complete without errors.
        assert_eq!(items.len(), 1);
        assert!(items[0].is_ok());
    }

    #[tokio::test]
    async fn test_contextualize_propagates_store_error() {
        let ctx = Contextualizer::new(
            Arc::new(FailingStore),
            Arc::new(MockTextGenerationProvider {
                reply: "".to_string(),
            }),
            default_config(),
        );
        let items: Vec<_> = ctx.contextualize("any prompt").collect().await;

        assert_eq!(items.len(), 1);
        assert!(
            matches!(&items[0], Err(ContextualizerError::MemoryStore(_))),
            "expected MemoryStore error, got: {:?}",
            items[0],
        );
    }

    #[tokio::test]
    async fn test_contextualize_propagates_text_generation_error() {
        let ctx = Contextualizer::new(
            Arc::new(MockStore { entries: vec![] }),
            Arc::new(FailingTextGenerationProvider),
            default_config(),
        );
        let items: Vec<_> = ctx.contextualize("prompt").collect().await;

        assert_eq!(items.len(), 1);
        assert!(
            matches!(&items[0], Err(ContextualizerError::ModelProvider(_))),
            "expected ModelProvider error, got: {:?}",
            items[0],
        );
    }

    #[tokio::test]
    async fn test_contextualize_applies_tuning_from_config() {
        let provider = Arc::new(CapturingTextGenerationProvider::default());
        let mut extra = HashMap::new();
        extra.insert("seed".to_string(), json!(42));
        let ctx = Contextualizer::new(
            Arc::new(MockStore { entries: vec![] }),
            provider.clone(),
            ContextualizerConfig {
                tuning: Some(ContextualizerTuningConfig {
                    temperature: Some(0.3),
                    max_tokens: Some(256),
                    top_p: Some(0.85),
                    repeat_penalty: Some(1.1),
                    repeat_last_n: Some(32),
                    thinking: Some(ThinkingMode::Disabled),
                    stop: Some(vec!["<END>".to_string()]),
                    keep_alive: Some(Duration::from_secs(120)),
                    extra_params: extra,
                }),
                ..default_config()
            },
        );
        let items: Vec<_> = ctx.contextualize("prompt").collect().await;
        assert_eq!(items.len(), 1);
        assert!(items[0].is_ok());

        let req = provider.last_req.lock().unwrap().clone().unwrap();
        assert_eq!(req.temperature, Some(0.3));
        assert_eq!(req.max_tokens, Some(256));
        assert_eq!(req.top_p, Some(0.85));
        assert_eq!(req.repeat_penalty, Some(1.1));
        assert_eq!(req.repeat_last_n, Some(32));
        assert!(matches!(req.thinking, Some(ThinkingMode::Disabled)));
        assert_eq!(req.stop, Some(vec!["<END>".to_string()]));
        assert_eq!(req.keep_alive, Some(Duration::from_secs(120)));
        assert_eq!(req.extra_params.get("seed"), Some(&json!(42)));
    }

    #[test]
    fn test_query_memory_uses_use_mode() {
        let query = MemoryQuery {
            topic: "x".to_string(),
            max_results: 1,
            min_score: Score::ZERO,
            filters: HashMap::new(),
            mode: MemoryQueryMode::Use,
        };
        assert_eq!(query.mode, MemoryQueryMode::Use);
    }

    #[tokio::test]
    async fn test_contextualize_with_debug_returns_debug_info_and_stream() {
        let entries = vec![make_entry("user prefers dark mode")];
        let ctx = make_contextualizer(entries.clone(), "noted");

        let (debug_info, stream) = ctx
            .contextualize_with_debug("set my preference")
            .await
            .unwrap();

        assert_eq!(debug_info.memory_entries.len(), 1);
        assert_eq!(
            debug_info.memory_entries[0].memory_entry.content,
            "user prefers dark mode"
        );

        let items: Vec<_> = stream.collect().await;
        assert_eq!(items.len(), 1);
        assert!(items[0].is_ok());
    }

    #[tokio::test]
    async fn test_contextualize_with_debug_propagates_store_error() {
        let ctx = Contextualizer::new(
            Arc::new(FailingStore),
            Arc::new(MockTextGenerationProvider {
                reply: "".to_string(),
            }),
            default_config(),
        );

        let result = ctx.contextualize_with_debug("any prompt").await;
        assert!(
            matches!(result, Err(ContextualizerError::MemoryStore(_))),
            "expected MemoryStore error",
        );
    }

    #[tokio::test]
    async fn test_contextualize_with_debug_returns_empty_debug_info_when_no_entries() {
        let ctx = make_contextualizer(vec![], "the answer is 42");

        let (debug_info, stream) = ctx
            .contextualize_with_debug("what is the answer?")
            .await
            .unwrap();

        assert!(debug_info.memory_entries.is_empty());

        let items: Vec<_> = stream.collect().await;
        assert_eq!(items.len(), 1);
        let resp = items.into_iter().next().unwrap().unwrap();
        assert_eq!(resp.text, "the answer is 42");
    }
}

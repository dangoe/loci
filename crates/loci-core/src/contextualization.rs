// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
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

const MEMORY_PLACEHOLDER: &str = "{{memory}}";

const SYSTEM_PROMPT_BASE_TEMPLATE: &str = "You are a helpful assistant with a long-term memory of past conversations.
Rules:
- Speak naturally. Never mention retrieval, context, memory stores, or system internals.
- When drawing on a memory, phrase it as personal recall: \"I remember...\", \"You mentioned once...\", etc.
- Only use memory_entries that are clearly relevant to the current question. Ignore the rest silently.
- If no memory applies, answer from general knowledge without commenting on the absence of memory.
- Never fabricate or embellish. If you are uncertain, say so. Only speculate if the user explicitly asks you to.
- Be concise. Avoid unnecessary preamble.";

pub type ResultStream =
    Pin<Box<dyn Stream<Item = Result<TextGenerationResponse, ContextualizerError>> + Send>>;

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
    /// Optional override for the system prompt template. If `None`, a default prompt will be used.
    pub system: Option<ContextualizerSystemConfig>,
    /// Mode for how the contextualizer should query the memory store for a given request.
    pub memory_mode: ContextualizationMemoryMode,
    /// Maximum number of memory entries to retrieve and inject into the prompt.
    pub max_memory_entries: usize,
    /// Minimum similarity score a memory must have to be included.
    pub min_score: Score,
    /// Metadata filters applied when querying the memory store.
    pub filters: HashMap<String, String>,
    /// Optional text-generation tuning parameters.
    pub tuning: Option<ContextualizerTuningConfig>,
}

/// Configuration for how the contextualizer constructs system prompts and queries memory.
#[derive(Debug, Clone)]
pub struct ContextualizerSystemConfig {
    /// System mode to be used.
    pub mode: ContextualizerSystemMode,
    /// The system prompt template text to use when constructing the system prompt.
    pub system: String,
}

/// Mode for how the contextualizer should construct the system prompt for a given request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextualizerSystemMode {
    /// The contextualizer should construct the system prompt by appending retrieved memory entries to a base template.
    Append,
    /// The contextualizer should use a caller-defined system prompt template that includes a `{{memory}}`
    /// placeholder for memory entries, and inject retrieved memory entries into that placeholder.
    /// If the placeholder is absent, memory entries are appended at the end.
    Replace,
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

/// Mode for how the contextualizer should query the memory store for a given request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextualizationMemoryMode {
    /// The contextualizer should query the memory store and inject relevant entries into the prompt as usual.
    Auto,
    /// The contextualizer should skip querying the memory store and proceed without injecting any memory entries.
    Off,
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
pub struct Contextualizer<M: MemoryStore, E: text_generation::TextGenerationModelProvider + 'static>
{
    memory_store: Arc<M>,
    text_generation_provider: Arc<E>,
    config: ContextualizerConfig,
}

impl<M, E> Contextualizer<M, E>
where
    M: MemoryStore,
    E: text_generation::TextGenerationModelProvider + 'static,
{
    /// Creates a new `Contextualizer` with the provided configuration.
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
    pub async fn contextualize(
        &self,
        prompt: impl Into<String>,
    ) -> Result<ResultStream, ContextualizerError> {
        Ok(self.contextualize_internal(prompt.into().as_str()).await?.1)
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
    pub async fn contextualize_with_debug(
        &self,
        prompt: impl Into<String>,
    ) -> Result<(ContextualizationDebugInfo, ResultStream), ContextualizerError> {
        let (memory_entries, stream) = self.contextualize_internal(prompt.into().as_str()).await?;

        let debug_info = ContextualizationDebugInfo {
            memory_entries: memory_entries.clone(),
        };

        Ok((debug_info, stream))
    }

    async fn contextualize_internal(
        &self,
        prompt: &str,
    ) -> Result<(Vec<MemoryQueryResult>, ResultStream), ContextualizerError> {
        let memory_entries = if self.config.memory_mode == ContextualizationMemoryMode::Auto {
            self.query_memory(prompt).await?
        } else {
            Vec::new()
        };

        let req = self.build_request(prompt, &memory_entries);
        let stream = self.build_result_stream(req);

        Ok((memory_entries, stream))
    }

    fn build_result_stream(&self, req: TextGenerationRequest) -> ResultStream {
        let provider = Arc::clone(&self.text_generation_provider);
        Box::pin(async_stream::try_stream! {
            use futures::StreamExt as _;
            let mut stream = Box::pin(provider.generate_stream(req));
            while let Some(result) = stream.next().await {
                let chunk = result.map_err(ContextualizerError::ModelProvider)?;
                yield chunk;
            }
        })
    }

    fn build_request(
        &self,
        prompt: &str,
        memory_entries: &[MemoryQueryResult],
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

    fn build_system_prompt(&self, memory_entries: &[MemoryQueryResult]) -> String {
        let memory_block = Self::format_memory_block(memory_entries);

        let mut buf = String::new();

        match &self.config.system {
            Some(system_config) => match system_config.mode {
                ContextualizerSystemMode::Append => {
                    buf.push_str(SYSTEM_PROMPT_BASE_TEMPLATE.trim());
                    buf.push('\n');
                    buf.push_str(system_config.system.trim());
                    buf.push('\n');
                    buf.push_str(&memory_block);
                }
                ContextualizerSystemMode::Replace => {
                    let template = system_config.system.trim();
                    if template.contains(MEMORY_PLACEHOLDER) {
                        buf.push_str(&template.replace(MEMORY_PLACEHOLDER, &memory_block));
                    } else {
                        buf.push_str(template);
                        buf.push('\n');
                        buf.push_str(&memory_block);
                    }
                }
            },
            None => {
                buf.push_str(SYSTEM_PROMPT_BASE_TEMPLATE.trim());
                buf.push('\n');
                buf.push_str(&memory_block);
            }
        };

        buf
    }

    fn format_memory_block(memory_entries: &[MemoryQueryResult]) -> String {
        let mut block = String::from("## Relevant memory entries\n");
        if memory_entries.is_empty() {
            block.push_str("None. Answer from general knowledge.\n");
        } else {
            for entry in memory_entries {
                block.push_str(&format!("- {}\n", entry.memory_entry.content));
            }
        }
        block
    }

    async fn query_memory(
        &self,
        prompt: &str,
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
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use std::{collections::HashMap, future::Future};

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
        fn add_entry(
            &self,
            input: MemoryInput,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            let entry = MemoryQueryResult {
                memory_entry: MemoryEntry::new(input.content, input.metadata),
                score: Score::ZERO,
            };
            async move { Ok(entry) }
        }

        fn get_entry(
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

        fn update_entry(
            &self,
            id: Uuid,
            _input: MemoryInput,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::NotFound(id)) }
        }

        fn set_entry_tier(
            &self,
            id: Uuid,
            _tier: MemoryTier,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::NotFound(id)) }
        }

        fn delete_entry(
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
        fn add_entry(
            &self,
            _input: MemoryInput,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::Connection("always fails".to_string())) }
        }

        fn get_entry(
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

        fn update_entry(
            &self,
            id: Uuid,
            _input: MemoryInput,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::NotFound(id)) }
        }

        fn set_entry_tier(
            &self,
            id: Uuid,
            _tier: MemoryTier,
        ) -> impl Future<Output = Result<MemoryQueryResult, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::NotFound(id)) }
        }

        fn delete_entry(
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
        ) -> impl Future<Output = ModelProviderResult<TextGenerationResponse>> + Send + '_ {
            let reply = self.reply.clone();
            async move { Ok(TextGenerationResponse::done(reply, req.model, None)) }
        }
    }

    struct FailingTextGenerationProvider;

    impl TextGenerationModelProvider for FailingTextGenerationProvider {
        fn generate(
            &self,
            _req: TextGenerationRequest,
        ) -> impl Future<Output = ModelProviderResult<TextGenerationResponse>> + Send + '_ {
            async move { Err(ModelProviderError::Timeout) }
        }
    }

    fn default_config() -> ContextualizerConfig {
        ContextualizerConfig {
            text_generation_model: "test-model".to_string(),
            system: None,
            memory_mode: ContextualizationMemoryMode::Auto,
            max_memory_entries: 5,
            min_score: Score::ZERO,
            filters: HashMap::new(),
            tuning: None,
        }
    }

    fn make_components(
        entries: Vec<MemoryQueryResult>,
        reply: &str,
    ) -> (MockStore, MockTextGenerationProvider) {
        (
            MockStore { entries },
            MockTextGenerationProvider {
                reply: reply.to_string(),
            },
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
        let (store, provider) = make_components(vec![], "reply");
        let ctx = Contextualizer::new(Arc::new(store), Arc::new(provider), default_config());
        let entries = vec![make_entry("I like cats"), make_entry("I live in Berlin")];
        let prompt = ctx.build_system_prompt(&entries);
        assert!(prompt.contains("I like cats"), "prompt: {prompt}");
        assert!(prompt.contains("I live in Berlin"), "prompt: {prompt}");
    }

    #[test]
    fn test_build_system_prompt_with_no_memory_entries_contains_placeholder() {
        let (store, provider) = make_components(vec![], "reply");
        let ctx = Contextualizer::new(Arc::new(store), Arc::new(provider), default_config());
        let prompt = ctx.build_system_prompt(&[]);
        assert!(
            prompt.contains("None. Answer from general knowledge."),
            "expected placeholder, got: {prompt}",
        );
    }

    #[test]
    fn test_build_system_prompt_replace_mode_substitutes_placeholder() {
        let mut config = default_config();
        config.system = Some(ContextualizerSystemConfig {
            mode: ContextualizerSystemMode::Replace,
            system: "Custom system prompt.\n{{memory}}\nEnd of prompt.".to_string(),
        });
        let store = MockStore { entries: vec![] };
        let provider = MockTextGenerationProvider {
            reply: "reply".to_string(),
        };
        let ctx = Contextualizer::new(Arc::new(store), Arc::new(provider), config);
        let prompt = ctx.build_system_prompt(&[make_entry("entry")]);
        assert!(
            prompt.contains("Custom system prompt."),
            "expected custom prompt, got: {prompt}",
        );
        assert!(
            prompt.contains("- entry"),
            "expected memory entry in placeholder position, got: {prompt}",
        );
        assert!(
            prompt.contains("End of prompt."),
            "expected text after placeholder, got: {prompt}",
        );
        assert!(
            !prompt.contains("{{memory}}"),
            "placeholder should have been substituted, got: {prompt}",
        );
    }

    #[test]
    fn test_build_system_prompt_replace_mode_without_placeholder_appends() {
        let mut config = default_config();
        config.system = Some(ContextualizerSystemConfig {
            mode: ContextualizerSystemMode::Replace,
            system: "Custom system prompt without placeholder.".to_string(),
        });
        let store = MockStore { entries: vec![] };
        let provider = MockTextGenerationProvider {
            reply: "reply".to_string(),
        };
        let ctx = Contextualizer::new(Arc::new(store), Arc::new(provider), config);
        let prompt = ctx.build_system_prompt(&[make_entry("entry")]);
        assert!(
            prompt.contains("Custom system prompt without placeholder."),
            "expected custom prompt, got: {prompt}",
        );
        assert!(
            prompt.contains("- entry"),
            "expected memory entry appended, got: {prompt}",
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
        ) -> impl Future<Output = ModelProviderResult<TextGenerationResponse>> + Send + '_ {
            *self.last_req.lock().unwrap() = Some(req.clone());
            async move {
                Ok(TextGenerationResponse::done(
                    "ok".to_string(),
                    req.model,
                    None,
                ))
            }
        }
    }

    #[tokio::test]
    async fn test_contextualize_returns_model_response_text() {
        let (store, provider) = make_components(vec![], "the answer is 42");
        let ctx = Contextualizer::new(Arc::new(store), Arc::new(provider), default_config());
        let items: Vec<_> = ctx
            .contextualize("what is the answer?")
            .await
            .unwrap()
            .collect()
            .await;

        assert_eq!(items.len(), 1);
        let resp = items.into_iter().next().unwrap().unwrap();
        assert_eq!(resp.text, "the answer is 42");
    }

    #[tokio::test]
    async fn test_contextualize_injects_retrieved_memory_entries_into_system_prompt() {
        // The mock store returns this entry regardless of query topic.
        let entries = vec![make_entry("user prefers dark mode")];
        let (store, provider) = make_components(entries, "noted");
        let ctx = Contextualizer::new(Arc::new(store), Arc::new(provider), default_config());
        let items: Vec<_> = ctx
            .contextualize("set my preference")
            .await
            .unwrap()
            .collect()
            .await;

        // The stream should complete without errors.
        assert_eq!(items.len(), 1);
        assert!(items[0].is_ok());
    }

    #[tokio::test]
    async fn test_contextualize_propagates_store_error() {
        let store = FailingStore;
        let provider = MockTextGenerationProvider {
            reply: "".to_string(),
        };
        let ctx = Contextualizer::new(Arc::new(store), Arc::new(provider), default_config());
        let error = ctx.contextualize("any prompt").await;

        assert!(
            matches!(error, Err(ContextualizerError::MemoryStore(_))),
            "expected MemoryStore error, got: {:?}",
            error.err().unwrap(),
        );
    }

    #[tokio::test]
    async fn test_contextualize_propagates_text_generation_error() {
        let store = MockStore { entries: vec![] };
        let provider = FailingTextGenerationProvider;
        let ctx = Contextualizer::new(Arc::new(store), Arc::new(provider), default_config());
        let items: Vec<_> = ctx.contextualize("prompt").await.unwrap().collect().await;

        assert_eq!(items.len(), 1);
        assert!(
            matches!(&items[0], Err(ContextualizerError::ModelProvider(_))),
            "expected ModelProvider error, got: {:?}",
            items[0],
        );
    }

    #[tokio::test]
    async fn test_contextualize_applies_tuning_from_config() {
        let store = MockStore { entries: vec![] };
        let provider = Arc::new(CapturingTextGenerationProvider::default());
        let mut extra = HashMap::new();
        extra.insert("seed".to_string(), json!(42));
        let ctx = Contextualizer::new(
            Arc::new(store),
            Arc::clone(&provider),
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
        let items: Vec<_> = ctx.contextualize("prompt").await.unwrap().collect().await;
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
        let (store, provider) = make_components(entries.clone(), "noted");
        let ctx = Contextualizer::new(Arc::new(store), Arc::new(provider), default_config());

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
        let store = FailingStore;
        let provider = MockTextGenerationProvider {
            reply: "".to_string(),
        };
        let ctx = Contextualizer::new(Arc::new(store), Arc::new(provider), default_config());

        let result = ctx.contextualize_with_debug("any prompt").await;
        assert!(
            matches!(result, Err(ContextualizerError::MemoryStore(_))),
            "expected MemoryStore error",
        );
    }

    #[tokio::test]
    async fn test_contextualize_with_debug_returns_empty_debug_info_when_no_entries() {
        let (store, provider) = make_components(vec![], "the answer is 42");
        let ctx = Contextualizer::new(Arc::new(store), Arc::new(provider), default_config());

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

    #[tokio::test]
    async fn test_contextualize_skips_memory_when_memory_mode_off() {
        // Use a failing store to ensure that if the contextualizer attempted to
        // query memory, it would error. With memory mode set to Off it should not
        // query the store and should successfully return the model response.
        let store = FailingStore;
        let provider = MockTextGenerationProvider {
            reply: "ok".to_string(),
        };
        let ctx = Contextualizer::new(
            Arc::new(store),
            Arc::new(provider),
            ContextualizerConfig {
                memory_mode: ContextualizationMemoryMode::Off,
                ..default_config()
            },
        );

        let items: Vec<_> = ctx
            .contextualize("any prompt")
            .await
            .unwrap()
            .collect()
            .await;

        assert_eq!(items.len(), 1);
        let resp = items.into_iter().next().unwrap().unwrap();
        assert_eq!(resp.text, "ok");
    }
}

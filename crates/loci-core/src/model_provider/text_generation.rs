// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-core.

use std::collections::HashMap;
use std::future::Future;
use std::time::Duration;

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::model_provider::common::{ModelProviderParams, ModelProviderResult};

/// A text-generation request that is model-provider-agnostic.
///
/// Every field except `prompt` and `model` is optional; model providers will use
/// their own defaults when a field is `None`.
#[derive(Debug, Clone)]
pub struct TextGenerationRequest {
    /// The model identifier as understood by the target model provider
    /// (e.g. `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`, `"llama3"`).
    pub model: String,

    /// The user / system prompt (text only).
    pub prompt: String,

    /// Optional system prompt.  Model providers that do not support a dedicated
    /// system role will prepend it to `prompt`.
    pub system: Option<String>,

    /// Sampling temperature in `[0.0, 2.0]`.  `None` uses model provider default.
    pub temperature: Option<f32>,

    /// Maximum number of tokens to generate.  `None` uses model provider default.
    pub max_tokens: Option<u32>,

    /// Nucleus sampling probability cutoff.  `None` uses model provider default.
    pub top_p: Option<f32>,

    /// Penalises tokens that have already appeared in the generated text, reducing
    /// repetition.  Maps to `repeat_penalty` (Ollama/llama.cpp),
    /// `frequency_penalty` (OpenAI/Cohere), etc.
    /// Typical range: `[1.0, 1.5]` for Ollama; `[0.0, 2.0]` for OpenAI.
    /// `None` uses model provider default.
    pub repeat_penalty: Option<f32>,

    /// Number of most-recent tokens to consider when applying `repeat_penalty`.
    /// Primarily meaningful for Ollama/llama.cpp; other providers ignore it
    /// gracefully.
    /// `None` uses model provider default.
    pub repeat_last_n: Option<u32>,

    /// Controls chain-of-thought / reasoning mode for models that support it
    /// (e.g. Qwen3, DeepSeek-R1, Claude claude-sonnet-4-20250514+, OpenAI o-series).
    /// `None` uses the model provider default (typically off).
    pub thinking: Option<ThinkingMode>,

    /// Stop sequences.  Generation halts when any of these strings appears.
    pub stop: Option<Vec<String>>,

    /// How long the model provider should keep the model loaded in memory after
    /// the request completes.  Primarily meaningful for Ollama; other model
    /// providers ignore it gracefully.
    pub keep_alive: Option<Duration>,

    /// Pass-through map for model-provider-specific options not covered above
    /// (e.g. `presence_penalty` for OpenAI, `top_k` for Ollama).
    pub extra_params: ModelProviderParams,
}

/// Controls extended chain-of-thought / reasoning mode.
#[derive(Debug, Clone)]
pub enum ThinkingMode {
    /// Enable thinking with provider default budget.
    Enabled,
    /// Coarse effort hint. Maps to `reasoning_effort` (OpenAI) or equivalent.
    /// Well-known values: `"low"`, `"medium"`, `"high"`.
    Effort { level: ThinkingEffortLevel },
    /// Enable thinking with an explicit token budget (honoured by providers that
    /// support it, e.g. Anthropic).  Other providers treat this as `Enabled`.
    Budgeted { max_tokens: u32 },
    /// Disable thinking explicitly (useful for models where it is on by default,
    /// e.g. o-series via `reasoning_effort: "none"`).
    Disabled,
}

#[derive(Debug, Clone)]
pub enum ThinkingEffortLevel {
    Low,
    Medium,
    High,
}

impl TextGenerationRequest {
    /// Minimal constructor — everything else defaults.
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            system: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            thinking: None,
            stop: None,
            keep_alive: None,
            extra_params: HashMap::new(),
        }
    }

    /// Sets the system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Sets the maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Sets the nucleus-sampling probability cutoff.
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Sets the repetition penalty.
    pub fn with_repeat_penalty(mut self, penalty: f32) -> Self {
        self.repeat_penalty = Some(penalty);
        self
    }

    /// Sets the number of most-recent tokens to consider when applying the repetition penalty.
    pub fn with_repeat_last_n(mut self, n: u32) -> Self {
        self.repeat_last_n = Some(n);
        self
    }

    /// Sets the thinking mode.
    pub fn with_thinking(mut self, thinking: ThinkingMode) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Sets the stop sequences; generation halts when any of these strings appears.
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Sets how long the model provider should keep the model loaded after the request.
    pub fn with_keep_alive(mut self, d: Duration) -> Self {
        self.keep_alive = Some(d);
        self
    }

    /// Adds a model-provider-specific extra parameter.
    pub fn with_extra(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}

/// A single streamed or non-streamed generation response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextGenerationResponse {
    /// The generated text.
    pub text: String,

    /// Model name echoed back by the model provider (may differ from the requested
    /// alias, e.g. after Ollama resolves a tag).
    pub model: String,

    /// Token usage, if the model provider reports it.
    pub usage: Option<TokenUsage>,

    /// `true` if this chunk/response is the final one in a stream.
    pub done: bool,
}

impl TextGenerationResponse {
    pub fn done(text: String, model: String, usage: Option<TokenUsage>) -> Self {
        Self {
            text,
            model,
            usage,
            done: true,
        }
    }
}

/// Token consumption reported by the model provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

/// A model provider that generates text responses.
pub trait TextGenerationModelProvider: Send + Sync {
    /// Generates a single response for the given request.
    fn generate(
        &self,
        req: TextGenerationRequest,
    ) -> impl Future<Output = ModelProviderResult<TextGenerationResponse>> + Send + '_;

    /// Streams generation responses chunk by chunk.
    ///
    /// Each item in the returned stream is a partial or final
    /// [`TextGenerationResponse`].  The last item has `done: true`.
    ///
    /// The default implementation wraps [`Self::generate`] into a single-item stream.
    fn generate_stream(
        &self,
        req: TextGenerationRequest,
    ) -> impl Stream<Item = ModelProviderResult<TextGenerationResponse>> + Send + '_ {
        futures::stream::once(self.generate(req))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures::StreamExt as _;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use crate::model_provider::common::ModelProviderResult;

    use super::*;

    #[test]
    fn test_new_sets_model_prompt_and_defaults_options_to_none() {
        let req = TextGenerationRequest::new("gpt-4o", "hello");
        assert_eq!(req.model, "gpt-4o");
        assert_eq!(req.prompt, "hello");
        assert!(req.system.is_none());
        assert!(req.temperature.is_none());
        assert!(req.max_tokens.is_none());
        assert!(req.top_p.is_none());
        assert!(req.stop.is_none());
        assert!(req.keep_alive.is_none());
        assert!(req.extra_params.is_empty());
    }

    #[test]
    fn test_with_system_sets_system_prompt() {
        let req = TextGenerationRequest::new("m", "p").with_system("be helpful");
        assert_eq!(req.system.as_deref(), Some("be helpful"));
    }

    #[test]
    fn test_with_temperature_sets_temperature() {
        let req = TextGenerationRequest::new("m", "p").with_temperature(0.7);
        assert_eq!(req.temperature, Some(0.7));
    }

    #[test]
    fn test_with_max_tokens_sets_max_tokens() {
        let req = TextGenerationRequest::new("m", "p").with_max_tokens(256);
        assert_eq!(req.max_tokens, Some(256));
    }

    #[test]
    fn test_with_top_p_sets_top_p() {
        let req = TextGenerationRequest::new("m", "p").with_top_p(0.9);
        assert_eq!(req.top_p, Some(0.9));
    }

    #[test]
    fn test_with_stop_sets_stop_sequences() {
        let stops = vec!["END".to_string(), "STOP".to_string()];
        let req = TextGenerationRequest::new("m", "p").with_stop(stops.clone());
        assert_eq!(req.stop, Some(stops));
    }

    #[test]
    fn test_with_keep_alive_sets_duration() {
        let d = Duration::from_secs(300);
        let req = TextGenerationRequest::new("m", "p").with_keep_alive(d);
        assert_eq!(req.keep_alive, Some(d));
    }

    #[test]
    fn test_with_extra_inserts_param() {
        let req = TextGenerationRequest::new("m", "p").with_extra("seed", json!(42));
        assert_eq!(req.extra_params["seed"], json!(42));
    }

    #[test]
    fn test_done_constructor_sets_fields_and_marks_done() {
        let usage = TokenUsage {
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
            total_tokens: Some(30),
        };
        let resp =
            TextGenerationResponse::done("answer".to_string(), "gpt-4o".to_string(), Some(usage));
        assert_eq!(resp.text, "answer");
        assert_eq!(resp.model, "gpt-4o");
        assert!(resp.done);
        assert!(resp.usage.is_some());
    }

    struct EchoProvider;

    impl TextGenerationModelProvider for EchoProvider {
        fn generate(
            &self,
            req: TextGenerationRequest,
        ) -> impl Future<Output = ModelProviderResult<TextGenerationResponse>> + Send + '_ {
            let model = req.model.clone();
            let text = req.prompt.clone();
            async move { Ok(TextGenerationResponse::done(text, model, None)) }
        }
    }

    #[tokio::test]
    async fn test_generate_stream_default_impl_yields_single_item() {
        let provider = EchoProvider;
        let req = TextGenerationRequest::new("model", "ping");
        let items: Vec<ModelProviderResult<TextGenerationResponse>> =
            provider.generate_stream(req).collect().await;

        assert_eq!(items.len(), 1);
        let resp = items.into_iter().next().unwrap().unwrap();
        assert_eq!(resp.text, "ping");
        assert!(resp.done);
    }
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::time::Duration;

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::model_provider::common::{ModelProviderParams, ModelProviderResult, TokenUsage};

/// A text-generation request that is model-provider-agnostic.
///
/// Every field except `prompt` and `model` is optional; model providers will use
/// their own defaults when a field is `None`.
#[derive(Debug, Clone)]
pub struct TextGenerationRequest {
    /// The model identifier as understood by the target model provider
    /// (e.g. `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`, `"llama3"`).
    model: String,

    /// The user / system prompt (text only).
    prompt: String,

    /// Optional system prompt.  Model providers that do not support a dedicated
    /// system role will prepend it to `prompt`.
    system: Option<String>,

    /// Sampling temperature in `[0.0, 2.0]`.  `None` uses model provider default.
    temperature: Option<f32>,

    /// Maximum number of tokens to generate.  `None` uses model provider default.
    max_tokens: Option<u32>,

    /// Nucleus sampling probability cutoff.  `None` uses model provider default.
    top_p: Option<f32>,

    /// Penalises tokens that have already appeared in the generated text, reducing
    /// repetition.  Maps to `repeat_penalty` (Ollama/llama.cpp),
    /// `frequency_penalty` (OpenAI/Cohere), etc.
    /// Typical range: `[1.0, 1.5]` for Ollama; `[0.0, 2.0]` for OpenAI.
    /// `None` uses model provider default.
    repeat_penalty: Option<f32>,

    /// Number of most-recent tokens to consider when applying `repeat_penalty`.
    /// Primarily meaningful for Ollama/llama.cpp; other providers ignore it
    /// gracefully.
    /// `None` uses model provider default.
    repeat_last_n: Option<u32>,

    /// Controls chain-of-thought / reasoning mode for models that support it
    /// (e.g. Qwen3, DeepSeek-R1, Claude claude-sonnet-4-20250514+, OpenAI o-series).
    /// `None` uses the model provider default (typically off).
    thinking: Option<ThinkingMode>,

    /// Stop sequences.  Generation halts when any of these strings appears.
    stop: Option<Vec<String>>,

    /// Requested output format. When `Some(ResponseFormat::Json)` the provider
    /// instructs the model to emit valid JSON only — mapped to Ollama's
    /// top-level `format: "json"` and OpenAI's `response_format`. Providers that
    /// cannot enforce a format ignore it.
    response_format: Option<ResponseFormat>,

    /// How long the model provider should keep the model loaded in memory after
    /// the request completes.  Primarily meaningful for Ollama; other model
    /// providers ignore it gracefully.
    keep_alive: Option<Duration>,

    /// Pass-through map for model-provider-specific options not covered above
    /// (e.g. `presence_penalty` for OpenAI, `top_k` for Ollama).
    extra_params: ModelProviderParams,
}

/// Controls extended chain-of-thought / reasoning mode.
#[non_exhaustive]
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

/// Coarse effort levels for thinking / reasoning mode.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum ThinkingEffortLevel {
    Low,
    Medium,
    High,
}

impl fmt::Display for ThinkingEffortLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        };
        write!(f, "{s}")
    }
}

/// Requested response format. Providers that can enforce structured output
/// (Ollama's `format`, OpenAI's `response_format`) use it to reject non-matching
/// outputs server-side; others ignore it.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum ResponseFormat {
    /// Constrain output to valid JSON.
    Json,
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
            response_format: None,
            keep_alive: None,
            extra_params: HashMap::new(),
        }
    }

    /// Returns the model identifier.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the prompt text.
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Returns the system prompt, if set.
    pub fn system(&self) -> Option<&str> {
        self.system.as_deref()
    }

    /// Returns the sampling temperature, if set.
    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// Returns the maximum number of tokens to generate, if set.
    pub fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    /// Returns the nucleus-sampling probability cutoff, if set.
    pub fn top_p(&self) -> Option<f32> {
        self.top_p
    }

    /// Returns the repetition penalty, if set.
    pub fn repeat_penalty(&self) -> Option<f32> {
        self.repeat_penalty
    }

    /// Returns the number of most-recent tokens for repetition penalty, if set.
    pub fn repeat_last_n(&self) -> Option<u32> {
        self.repeat_last_n
    }

    /// Returns the thinking mode, if set.
    pub fn thinking(&self) -> Option<&ThinkingMode> {
        self.thinking.as_ref()
    }

    /// Returns the stop sequences, if set.
    pub fn stop(&self) -> Option<&[String]> {
        self.stop.as_deref()
    }

    /// Returns the requested response format, if set.
    pub fn response_format(&self) -> Option<&ResponseFormat> {
        self.response_format.as_ref()
    }

    /// Returns the keep-alive duration, if set.
    pub fn keep_alive(&self) -> Option<Duration> {
        self.keep_alive
    }

    /// Returns the extra model-provider-specific parameters.
    pub fn extra_params(&self) -> &ModelProviderParams {
        &self.extra_params
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

    /// Requests a specific response format from the provider.
    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
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
    text: String,
    model: String,
    usage: Option<TokenUsage>,
    done: bool,
}

impl TextGenerationResponse {
    /// Creates a new response with all fields specified.
    pub fn new(text: String, model: String, usage: Option<TokenUsage>, done: bool) -> Self {
        Self {
            text,
            model,
            usage,
            done,
        }
    }

    /// Creates a final (done) response.
    pub fn new_done(text: String, model: String, usage: Option<TokenUsage>) -> Self {
        Self::new(text, model, usage, true)
    }

    /// Returns the generated text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the model name echoed back by the provider.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns token usage, if reported.
    pub fn usage(&self) -> Option<&TokenUsage> {
        self.usage.as_ref()
    }

    /// Returns `true` if this is the final chunk/response in a stream.
    pub fn is_done(&self) -> bool {
        self.done
    }
}

/// A model provider that generates text responses.
///
/// # Object Safety
/// This trait uses return-position `impl Trait` (RPITIT) for ergonomics and
/// zero-cost dispatch in generic contexts. It is therefore **not object-safe**
/// and cannot be used as `dyn TextGenerationModelProvider`. If dynamic dispatch is
/// needed in future, convert to `BoxFuture` like `MemoryStore`.
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
        assert_eq!(req.model(), "gpt-4o");
        assert_eq!(req.prompt(), "hello");
        assert!(req.system().is_none());
        assert!(req.temperature().is_none());
        assert!(req.max_tokens().is_none());
        assert!(req.top_p().is_none());
        assert!(req.stop().is_none());
        assert!(req.keep_alive().is_none());
        assert!(req.extra_params().is_empty());
    }

    #[test]
    fn test_with_system_sets_system_prompt() {
        let req = TextGenerationRequest::new("m", "p").with_system("be helpful");
        assert_eq!(req.system(), Some("be helpful"));
    }

    #[test]
    fn test_with_temperature_sets_temperature() {
        let req = TextGenerationRequest::new("m", "p").with_temperature(0.7);
        assert_eq!(req.temperature(), Some(0.7));
    }

    #[test]
    fn test_with_max_tokens_sets_max_tokens() {
        let req = TextGenerationRequest::new("m", "p").with_max_tokens(256);
        assert_eq!(req.max_tokens(), Some(256));
    }

    #[test]
    fn test_with_top_p_sets_top_p() {
        let req = TextGenerationRequest::new("m", "p").with_top_p(0.9);
        assert_eq!(req.top_p(), Some(0.9));
    }

    #[test]
    fn test_with_stop_sets_stop_sequences() {
        let stops = vec!["END".to_string(), "STOP".to_string()];
        let req = TextGenerationRequest::new("m", "p").with_stop(stops.clone());
        assert_eq!(req.stop(), Some(stops.as_slice()));
    }

    #[test]
    fn test_with_keep_alive_sets_duration() {
        let d = Duration::from_secs(300);
        let req = TextGenerationRequest::new("m", "p").with_keep_alive(d);
        assert_eq!(req.keep_alive(), Some(d));
    }

    #[test]
    fn test_with_extra_inserts_param() {
        let req = TextGenerationRequest::new("m", "p").with_extra("seed", json!(42));
        assert_eq!(req.extra_params()["seed"], json!(42));
    }

    #[test]
    fn test_new_done_constructor_sets_fields_and_marks_done() {
        let usage = TokenUsage::new(Some(10), Some(20), Some(30));
        let resp = TextGenerationResponse::new_done(
            "answer".to_string(),
            "gpt-4o".to_string(),
            Some(usage),
        );
        assert_eq!(resp.text(), "answer");
        assert_eq!(resp.model(), "gpt-4o");
        assert!(resp.is_done());
        assert!(resp.usage().is_some());
    }

    #[test]
    fn test_new_constructor_respects_done_flag() {
        let resp = TextGenerationResponse::new("chunk".to_string(), "m".to_string(), None, false);
        assert_eq!(resp.text(), "chunk");
        assert!(!resp.is_done());

        let resp = TextGenerationResponse::new("final".to_string(), "m".to_string(), None, true);
        assert!(resp.is_done());
    }

    struct EchoProvider;

    impl TextGenerationModelProvider for EchoProvider {
        fn generate(
            &self,
            req: TextGenerationRequest,
        ) -> impl Future<Output = ModelProviderResult<TextGenerationResponse>> + Send + '_ {
            let model = req.model().to_owned();
            let text = req.prompt().to_owned();
            async move { Ok(TextGenerationResponse::new_done(text, model, None)) }
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
        assert_eq!(resp.text(), "ping");
        assert!(resp.is_done());
    }
}

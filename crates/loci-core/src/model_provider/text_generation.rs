// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-core.

use std::time::Duration;
use std::{collections::HashMap, pin::Pin};

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

    /// Stop sequences.  Generation halts when any of these strings appears.
    pub stop: Option<Vec<String>>,

    /// How long the model provider should keep the model loaded in memory after the
    /// request completes. Primarily meaningful for Ollama; other model providers
    /// ignore it gracefully.
    pub keep_alive: Option<Duration>,

    /// Pass-through map for model-provider-specific options not covered above.
    pub extra_params: ModelProviderParams,
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
    ) -> Pin<Box<dyn Future<Output = ModelProviderResult<TextGenerationResponse>> + Send + '_>>;

    /// Streams generation responses chunk by chunk.
    ///
    /// Each item in the returned stream is a partial or final
    /// [`TextGenerationResponse`].  The last item has `done: true`.
    ///
    /// The default implementation wraps [`Self::generate`] into a single-item stream.
    fn generate_stream(
        &self,
        req: TextGenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = ModelProviderResult<TextGenerationResponse>> + Send + '_>> {
        Box::pin(futures::stream::once(self.generate(req)))
    }
}

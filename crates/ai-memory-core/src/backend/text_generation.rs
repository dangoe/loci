use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::backend::common::{BackendParams, BackendResult};

/// A text-generation request that is backend-agnostic.
///
/// Every field except `prompt` and `model` is optional; backends will use
/// their own defaults when a field is `None`.
#[derive(Debug, Clone)]
pub struct TextGenerationRequest {
    /// The model identifier as understood by the target backend
    /// (e.g. `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`, `"llama3"`).
    pub model: String,

    /// The user / system prompt (text only).
    pub prompt: String,

    /// Optional system prompt.  Backends that do not support a dedicated
    /// system role will prepend it to `prompt`.
    pub system: Option<String>,

    /// Sampling temperature in `[0.0, 2.0]`.  `None` uses backend default.
    pub temperature: Option<f32>,

    /// Maximum number of tokens to generate.  `None` uses backend default.
    pub max_tokens: Option<u32>,

    /// Nucleus sampling probability cutoff.  `None` uses backend default.
    pub top_p: Option<f32>,

    /// Stop sequences.  Generation halts when any of these strings appears.
    pub stop: Option<Vec<String>>,

    /// How long the backend should keep the model loaded in memory after the
    /// request completes. Primarily meaningful for Ollama; other backends
    /// ignore it gracefully.
    pub keep_alive: Option<Duration>,

    /// Pass-through map for backend-specific options not covered above.
    pub extra_params: BackendParams,
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

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    pub fn with_keep_alive(mut self, d: Duration) -> Self {
        self.keep_alive = Some(d);
        self
    }

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

    /// Model name echoed back by the backend (may differ from the requested
    /// alias, e.g. after Ollama resolves a tag).
    pub model: String,

    /// Token usage, if the backend reports it.
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

/// Token consumption reported by the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

pub trait TextGenerationBackend: Send + Sync {
    fn generate(
        &self,
        req: TextGenerationRequest,
    ) -> impl Future<Output = BackendResult<TextGenerationResponse>> + Send + '_;
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-model-provider-ollama.

use futures::StreamExt as _;
use loci_core::model_provider::{
    common::ModelProviderResult,
    embedding::{EmbeddingModelProvider, EmbeddingRequest, EmbeddingResponse},
    error::ModelProviderError,
    text_generation::{
        ResponseFormat, TextGenerationModelProvider, TextGenerationRequest, TextGenerationResponse,
        ThinkingEffortLevel, ThinkingMode, TokenUsage,
    },
};
use log::debug;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use std::time::Duration;

/// Configuration for the Ollama model provider.
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    /// Base URL of the Ollama instance.  Defaults to `http://localhost:11434`.
    pub base_url: String,

    /// Optional request timeout.
    pub timeout: Option<Duration>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            timeout: None,
        }
    }
}

/// Request body for text generation requests.
#[derive(Debug, Serialize)]
struct OllamaTextGenerationRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
    /// Structured-output constraint. When set to `"json"`, Ollama rejects
    /// non-JSON output server-side rather than letting the model drift into prose.
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'static str>,
    /// keep_alive accepts a duration string like "5m" or an integer (seconds).
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
}

/// Response body for text generation requests.
#[derive(Debug, Deserialize)]
struct OllamaTextGenerationResponse {
    model: String,
    response: String,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

/// A single chunk from a streaming text generation response.
#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    model: String,
    response: String,
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

/// Request body for embedding requests.
#[derive(Debug, Serialize)]
struct OllamaEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
}

/// Response body for embedding requests.
#[derive(Debug, Deserialize)]
struct OllamaEmbeddingResponse {
    model: String,
    embeddings: Vec<Vec<f32>>,
}

/// An Ollama model provider implementing both text generation and embedding.
#[derive(Debug, Clone)]
pub struct OllamaModelProvider {
    config: OllamaConfig,
    client: Client,
}

impl OllamaModelProvider {
    /// Creates a new `OllamaModelProvider` instance.
    pub fn new(config: OllamaConfig) -> Result<Self, ModelProviderError> {
        let mut builder = Client::builder();
        if let Some(timeout) = config.timeout {
            builder = builder.timeout(timeout);
        }
        let client = builder.build().map_err(|e| ModelProviderError::Other {
            message: format!("Failed to build HTTP client: {e}"),
        })?;
        Ok(Self { config, client })
    }

    /// Build an Ollama text generation request from a generic request.
    ///
    /// - Merges `extra_params` into `options`, with typed fields overriding extras.
    /// - Maps `thinking` into the top-level `think` field:
    ///   - `Enabled` -> `true`
    ///   - `Disabled` -> `false`
    ///   - `Preset { preset_name }` -> string preset name
    ///   - `Budgeted { .. }` -> fallback to `true` (enabled) per instructions
    /// - Formats `keep_alive` as a duration string like "5m" or "30s".
    fn build_text_request<'a>(
        &self,
        req: &'a TextGenerationRequest,
        stream: bool,
    ) -> OllamaTextGenerationRequest<'a> {
        // Start with extras
        let mut opts_map = serde_json::Map::new();
        for (k, v) in req.extra_params.iter() {
            opts_map.insert(k.clone(), v.clone());
        }

        // Typed fields override extra_params
        if let Some(t) = req.temperature {
            opts_map.insert("temperature".to_string(), Value::from(t));
        }
        if let Some(p) = req.top_p {
            opts_map.insert("top_p".to_string(), Value::from(p));
        }
        if let Some(max) = req.max_tokens {
            opts_map.insert("max_tokens".to_string(), Value::from(max));
        }
        if let Some(stops) = &req.stop {
            let arr: Vec<Value> = stops.iter().map(|s| Value::from(s.clone())).collect();
            opts_map.insert("stop".to_string(), Value::from(arr));
        }
        if let Some(rp) = req.repeat_penalty {
            opts_map.insert("repeat_penalty".to_string(), Value::from(rp));
        }
        if let Some(rl) = req.repeat_last_n {
            opts_map.insert("repeat_last_n".to_string(), Value::from(rl));
        }

        let options = if opts_map.is_empty() {
            None
        } else {
            Some(Value::Object(opts_map))
        };

        // Map thinking mode to top-level `think` parameter. Budgeted falls back to enabled.
        let think = match &req.thinking {
            Some(ThinkingMode::Enabled) => Some(Value::Bool(true)),
            Some(ThinkingMode::Disabled) => Some(Value::Bool(false)),
            Some(ThinkingMode::Effort { level }) => Some(match level {
                ThinkingEffortLevel::Low => Value::String("low".to_string()),
                ThinkingEffortLevel::Medium => Value::String("medium".to_string()),
                ThinkingEffortLevel::High => Value::String("high".to_string()),
            }),
            Some(ThinkingMode::Budgeted { .. }) => Some(Value::Bool(true)), // fall back to Enabled
            None => None,
        };

        // Format keep_alive as "Xm" if whole minutes, otherwise "Ns"
        let keep_alive = req.keep_alive.map(|d| {
            let secs = d.as_secs();
            if secs != 0 && secs % 60 == 0 {
                format!("{}m", secs / 60)
            } else {
                format!("{}s", secs)
            }
        });

        let format = match req.response_format {
            Some(ResponseFormat::Json) => Some("json"),
            None => None,
        };

        OllamaTextGenerationRequest {
            model: &req.model,
            prompt: &req.prompt,
            stream,
            think,
            system: req.system.as_deref(),
            options,
            format,
            keep_alive,
        }
    }
}

impl TextGenerationModelProvider for OllamaModelProvider {
    async fn generate(
        &self,
        req: TextGenerationRequest,
    ) -> ModelProviderResult<TextGenerationResponse> {
        let body = self.build_text_request(&req, false);

        debug!("Sending request to Ollama: {:?}", body);

        let http_response = self
            .client
            .post(format!("{}/api/generate", self.config.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| ModelProviderError::Transport {
                message: e.to_string(),
            })?;

        debug!("Received response from Ollama: {:?}", http_response);

        if !http_response.status().is_success() {
            let status = http_response.status();
            let body_text = http_response.text().await.unwrap_or_default();
            let detail = serde_json::from_str::<Value>(&body_text)
                .ok()
                .and_then(|v| v["error"].as_str().map(str::to_owned))
                .unwrap_or(body_text);
            return Err(ModelProviderError::Other {
                message: format!("Ollama returned HTTP {status}: {detail}"),
            });
        }

        let parse_response: OllamaTextGenerationResponse =
            http_response
                .json()
                .await
                .map_err(|e| ModelProviderError::Parse {
                    message: format!("error decoding response body: {e}"),
                })?;

        debug!("Parsed response from Ollama: {:?}", parse_response);

        let total_tokens = match (parse_response.prompt_eval_count, parse_response.eval_count) {
            (Some(a), Some(b)) => Some(a + b),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        Ok(TextGenerationResponse {
            text: parse_response.response,
            model: parse_response.model,
            usage: Some(TokenUsage {
                prompt_tokens: parse_response.prompt_eval_count,
                completion_tokens: parse_response.eval_count,
                total_tokens,
            }),
            done: true,
        })
    }

    fn generate_stream(
        &self,
        req: TextGenerationRequest,
    ) -> impl futures::Stream<Item = ModelProviderResult<TextGenerationResponse>> + Send + '_ {
        async_stream::try_stream! {
            let body = self.build_text_request(&req, true);

            debug!("Sending streaming request to Ollama: {:?}", body);

            let http_response = self
                .client
                .post(format!("{}/api/generate", self.config.base_url))
                .json(&body)
                .send()
                .await
                .map_err(|e| ModelProviderError::Transport { message: e.to_string() })?;

            let status = http_response.status();
            if !status.is_success() {
                let body_text = http_response.text().await.unwrap_or_default();
                let detail = serde_json::from_str::<Value>(&body_text)
                    .ok()
                    .and_then(|v| v["error"].as_str().map(str::to_owned))
                    .unwrap_or(body_text);
                Err(ModelProviderError::Other {
                    message: format!("Ollama returned HTTP {status}: {detail}"),
                })?;
                return; // unreachable — satisfies borrow checker that http_response is not reused
            }

            let mut byte_stream = http_response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk) = byte_stream.next().await {
                let bytes = chunk.map_err(|e| ModelProviderError::Transport { message: e.to_string() })?;
                let text = String::from_utf8(bytes.to_vec())
                    .map_err(|e| ModelProviderError::Parse { message: e.to_string() })?;
                buffer.push_str(&text);

                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim().to_string();
                    buffer.drain(..=newline_pos);

                    if line.is_empty() {
                        continue;
                    }

                    let chunk: OllamaStreamChunk = serde_json::from_str(&line)
                        .map_err(|e| ModelProviderError::Parse { message: e.to_string() })?;

                    let is_done = chunk.done;
                    let usage = if is_done {
                        let total = match (chunk.prompt_eval_count, chunk.eval_count) {
                            (Some(a), Some(b)) => Some(a + b),
                            (Some(a), None) => Some(a),
                            (None, Some(b)) => Some(b),
                            (None, None) => None,
                        };
                        Some(TokenUsage {
                            prompt_tokens: chunk.prompt_eval_count,
                            completion_tokens: chunk.eval_count,
                            total_tokens: total,
                        })
                    } else {
                        None
                    };

                    yield TextGenerationResponse {
                        text: chunk.response,
                        model: chunk.model,
                        usage,
                        done: is_done,
                    };

                    if is_done {
                        return;
                    }
                }
            }
        }
    }
}

impl EmbeddingModelProvider for OllamaModelProvider {
    async fn embed(&self, req: EmbeddingRequest) -> ModelProviderResult<EmbeddingResponse> {
        let body = OllamaEmbeddingRequest {
            model: &req.model,
            input: &req.input,
            dimensions: req.embedding_dimension,
            options: None,
            keep_alive: None,
        };

        let http_response = self
            .client
            .post(format!("{}/api/embed", self.config.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| ModelProviderError::Transport {
                message: format!("Failed to send request: {e}"),
            })?;

        if !http_response.status().is_success() {
            let status = http_response.status();
            let body_text = http_response.text().await.unwrap_or_default();
            let detail = serde_json::from_str::<Value>(&body_text)
                .ok()
                .and_then(|v| v["error"].as_str().map(str::to_owned))
                .unwrap_or(body_text);
            return Err(ModelProviderError::Other {
                message: format!("Ollama returned HTTP {status}: {detail}"),
            });
        }

        let parsed_response: OllamaEmbeddingResponse =
            http_response
                .json()
                .await
                .map_err(|e| ModelProviderError::Parse {
                    message: format!("Failed to parse response: {e}"),
                })?;

        Ok(EmbeddingResponse {
            embeddings: parsed_response.embeddings,
            model: parsed_response.model,
            usage: None,
        })
    }
}

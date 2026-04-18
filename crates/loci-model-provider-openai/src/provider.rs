// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-model-provider-openai.

use std::time::Duration;

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

/// Configuration for the OpenAI-compatible model provider.
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// Base URL of the OpenAI-compatible API (without trailing slash).
    /// Defaults to `https://api.openai.com/v1`.
    pub base_url: String,

    /// Optional API key sent as `Authorization: Bearer <key>`.
    pub api_key: Option<String>,

    /// Optional request timeout.
    pub timeout: Option<Duration>,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: None,
            timeout: None,
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    /// Maps from `repeat_penalty` — closest semantic equivalent in OpenAI API.
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<&'a Vec<String>>,
    /// Maps `ThinkingMode::Effort` and `ThinkingMode::Disabled` for o-series models.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
    /// Request usage statistics in the final streaming chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    /// Structured-output constraint (`{"type": "json_object"}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<OpenAiResponseFormat>,
    /// Pass-through extra params (e.g. `presence_penalty`, `seed`).
    #[serde(flatten)]
    extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Serialize)]
struct OpenAiResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    model: String,
    choices: Vec<NonStreamChoice>,
    #[serde(default)]
    usage: Option<CompletionUsage>,
}

#[derive(Debug, Deserialize)]
struct NonStreamChoice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    model: String,
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<CompletionUsage>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: ChunkDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct ChunkDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CompletionUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct OpenAIEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    model: String,
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

/// A model provider that speaks the OpenAI REST API (`/v1/chat/completions`,
/// `/v1/embeddings`).  Compatible with OpenAI, Groq, LM Studio, llama.cpp,
/// Ollama (in OpenAI-compat mode), and any other drop-in.
#[derive(Debug, Clone)]
pub struct OpenAIModelProvider {
    config: OpenAIConfig,
    client: Client,
}

impl OpenAIModelProvider {
    /// Creates a new `OpenAIModelProvider` instance.
    pub fn new(config: OpenAIConfig) -> Result<Self, ModelProviderError> {
        let mut builder = Client::builder();
        if let Some(timeout) = config.timeout {
            builder = builder.timeout(timeout);
        }
        let client = builder.build().map_err(|e| ModelProviderError::Other {
            message: format!("Failed to build HTTP client: {e}"),
        })?;
        Ok(Self { config, client })
    }

    fn build_chat_request<'a>(
        req: &'a TextGenerationRequest,
        stream: bool,
    ) -> ChatCompletionRequest<'a> {
        let mut messages = Vec::new();
        if let Some(system) = req.system.as_deref() {
            messages.push(ChatMessage {
                role: "system",
                content: system,
            });
        }
        messages.push(ChatMessage {
            role: "user",
            content: &req.prompt,
        });

        // Extra params pass-through (typed fields override them below)
        let mut extra: serde_json::Map<String, Value> = req
            .extra_params
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // ThinkingMode mapping
        let reasoning_effort: Option<&'static str> = match &req.thinking {
            Some(ThinkingMode::Effort { level }) => Some(match level {
                ThinkingEffortLevel::Low => "low",
                ThinkingEffortLevel::Medium => "medium",
                ThinkingEffortLevel::High => "high",
            }),
            Some(ThinkingMode::Disabled) => Some("none"),
            // Budgeted: honour budget via max_completion_tokens extra param
            Some(ThinkingMode::Budgeted { max_tokens }) => {
                extra.insert(
                    "max_completion_tokens".to_string(),
                    Value::from(*max_tokens),
                );
                None
            }
            _ => None,
        };

        let response_format =
            req.response_format
                .as_ref()
                .map(|ResponseFormat::Json| OpenAiResponseFormat {
                    kind: "json_object",
                });

        ChatCompletionRequest {
            model: &req.model,
            messages,
            stream,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            top_p: req.top_p,
            frequency_penalty: req.repeat_penalty,
            stop: req.stop.as_ref(),
            reasoning_effort,
            stream_options: if stream {
                Some(StreamOptions {
                    include_usage: true,
                })
            } else {
                None
            },
            response_format,
            extra,
        }
    }

    fn add_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.config.api_key {
            Some(key) => builder.bearer_auth(key),
            None => builder,
        }
    }
}

impl TextGenerationModelProvider for OpenAIModelProvider {
    async fn generate(
        &self,
        req: TextGenerationRequest,
    ) -> ModelProviderResult<TextGenerationResponse> {
        let body = Self::build_chat_request(&req, false);

        debug!(
            "Sending non-streaming request to OpenAI: model={}",
            req.model
        );

        let http_response = self
            .add_auth(
                self.client
                    .post(format!("{}/chat/completions", self.config.base_url))
                    .json(&body),
            )
            .send()
            .await
            .map_err(|e| ModelProviderError::Transport {
                message: e.to_string(),
            })?;

        if !http_response.status().is_success() {
            let status = http_response.status().as_u16();
            let msg = http_response.text().await.unwrap_or_default();
            return Err(ModelProviderError::Http {
                message: msg,
                status: Some(status),
            });
        }

        let parsed: ChatCompletionResponse =
            http_response
                .json()
                .await
                .map_err(|e| ModelProviderError::Parse {
                    message: e.to_string(),
                })?;

        let text = parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default();

        let usage = parsed.usage.map(|u| TokenUsage {
            prompt_tokens: Some(u.prompt_tokens),
            completion_tokens: Some(u.completion_tokens),
            total_tokens: Some(u.total_tokens),
        });

        Ok(TextGenerationResponse {
            text,
            model: parsed.model,
            usage,
            done: true,
        })
    }

    fn generate_stream(
        &self,
        req: TextGenerationRequest,
    ) -> impl futures::Stream<Item = ModelProviderResult<TextGenerationResponse>> + Send + '_ {
        async_stream::try_stream! {
            let body = Self::build_chat_request(&req, true);

            debug!("Sending streaming request to OpenAI: model={}", req.model);

            let http_response = self
                .add_auth(
                    self.client
                        .post(format!("{}/chat/completions", self.config.base_url))
                        .json(&body),
                )
                .send()
                .await
                .map_err(|e| ModelProviderError::Transport { message: e.to_string() })?;

            if !http_response.status().is_success() {
                let status = http_response.status().as_u16();
                let msg = http_response.text().await.unwrap_or_default();
                Err(ModelProviderError::Http { message: msg, status: Some(status) })?;
                return;
            }

            let mut byte_stream = http_response.bytes_stream();
            let mut buffer = String::new();
            // Use the request model as a fallback until the first chunk arrives.
            let mut model_name = req.model.clone();

            while let Some(chunk) = byte_stream.next().await {
                let bytes = chunk.map_err(|e| ModelProviderError::Transport { message: e.to_string() })?;
                let text = String::from_utf8(bytes.to_vec())
                    .map_err(|e| ModelProviderError::Parse { message: e.to_string() })?;
                buffer.push_str(&text);

                // SSE events are separated by blank lines (\n\n).
                while let Some(end) = buffer.find("\n\n") {
                    let event = buffer[..end].to_string();
                    buffer.drain(..end + 2);

                    for line in event.lines() {
                        let data = match line.strip_prefix("data: ") {
                            Some(d) => d.trim(),
                            None => continue,
                        };

                        if data == "[DONE]" {
                            return;
                        }

                        let chunk: ChatCompletionChunk = serde_json::from_str(data)
                            .map_err(|e| ModelProviderError::Parse { message: e.to_string() })?;

                        if !chunk.model.is_empty() {
                            model_name = chunk.model.clone();
                        }

                        let choice = match chunk.choices.into_iter().next() {
                            Some(c) => c,
                            None => continue,
                        };

                        let is_done = choice.finish_reason.is_some();
                        let content = choice.delta.content.unwrap_or_default();

                        let usage = if is_done {
                            chunk.usage.map(|u| TokenUsage {
                                prompt_tokens: Some(u.prompt_tokens),
                                completion_tokens: Some(u.completion_tokens),
                                total_tokens: Some(u.total_tokens),
                            })
                        } else {
                            None
                        };

                        yield TextGenerationResponse {
                            text: content,
                            model: model_name.clone(),
                            usage,
                            done: is_done,
                        };
                    }
                }
            }
        }
    }
}

impl EmbeddingModelProvider for OpenAIModelProvider {
    async fn embed(&self, req: EmbeddingRequest) -> ModelProviderResult<EmbeddingResponse> {
        let body = OpenAIEmbeddingRequest {
            model: &req.model,
            input: &req.input,
            dimensions: req.embedding_dimension,
        };

        let http_response = self
            .add_auth(
                self.client
                    .post(format!("{}/embeddings", self.config.base_url))
                    .json(&body),
            )
            .send()
            .await
            .map_err(|e| ModelProviderError::Transport {
                message: format!("Failed to send request: {e}"),
            })?;

        if !http_response.status().is_success() {
            let status = http_response.status().as_u16();
            let msg = http_response.text().await.unwrap_or_default();
            return Err(ModelProviderError::Http {
                message: msg,
                status: Some(status),
            });
        }

        let mut parsed: OpenAIEmbeddingResponse =
            http_response
                .json()
                .await
                .map_err(|e| ModelProviderError::Parse {
                    message: format!("Failed to parse response: {e}"),
                })?;

        // Ensure vectors are in the same order as the inputs.
        parsed.data.sort_by_key(|d| d.index);
        let embeddings = parsed.data.into_iter().map(|d| d.embedding).collect();

        Ok(EmbeddingResponse {
            embeddings,
            model: parsed.model,
            usage: None,
        })
    }
}

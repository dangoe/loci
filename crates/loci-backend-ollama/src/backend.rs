// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-backend-ollama.

use futures::StreamExt as _;
use loci_core::backend::{
    common::BackendResult,
    embedding::{EmbeddingBackend, EmbeddingRequest, EmbeddingResponse},
    error::BackendError,
    text_generation::{
        TextGenerationBackend, TextGenerationRequest, TextGenerationResponse, TokenUsage,
    },
};
use log::{debug, error};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use std::{future::Future, pin::Pin, time::Duration};

/// Configuration for the Ollama backend.
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
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
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

/// A backend implementation for the Ollama API.
#[derive(Debug, Clone)]
pub struct OllamaBackend {
    config: OllamaConfig,
    client: Client,
}

/// Constructs a new `OllamaBackend` instance.
impl OllamaBackend {
    pub fn new(config: OllamaConfig) -> Result<Self, BackendError> {
        let mut builder = Client::builder();
        if let Some(timeout) = config.timeout {
            builder = builder.timeout(timeout);
        }
        let client = builder.build().map_err(|e| BackendError::Other {
            message: format!("Failed to build HTTP client: {e}"),
        })?;
        Ok(Self { config, client })
    }
}

/// Implements the `TextGenerationBackend` trait for `OllamaBackend`.
impl TextGenerationBackend for OllamaBackend {
    fn generate(
        &self,
        req: TextGenerationRequest,
    ) -> Pin<Box<dyn Future<Output = BackendResult<TextGenerationResponse>> + Send + '_>> {
        Box::pin(async move {
            let body = OllamaTextGenerationRequest {
                model: &req.model,
                prompt: &req.prompt,
                stream: false,
                system: req.system.as_deref(),
                options: None,
                keep_alive: None,
            };

            debug!("Sending request to Ollama: {:?}", body);

            let http_response = self
                .client
                .post(format!("{}/api/generate", self.config.base_url))
                .json(&body)
                .send()
                .await
                .map_err(|e| BackendError::Transport {
                    message: e.to_string(),
                })?;

            debug!("Received response from Ollama: {:?}", http_response);

            let parsed: Result<OllamaTextGenerationResponse, reqwest::Error> =
                http_response.json().await;

            if parsed.is_err() {
                error!("Failed to parse response: {:?}", parsed);
            }

            let parse_response: OllamaTextGenerationResponse =
                parsed.map_err(|e| BackendError::Parse {
                    message: e.to_string(),
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
        })
    }

    fn generate_stream(
        &self,
        req: TextGenerationRequest,
    ) -> Pin<
        Box<
            dyn futures::Stream<Item = BackendResult<TextGenerationResponse>> + Send + '_,
        >,
    > {
        Box::pin(async_stream::try_stream! {
            let body = OllamaTextGenerationRequest {
                model: &req.model,
                prompt: &req.prompt,
                stream: true,
                system: req.system.as_deref(),
                options: None,
                keep_alive: None,
            };

            debug!("Sending streaming request to Ollama: {:?}", body);

            let http_response = self
                .client
                .post(format!("{}/api/generate", self.config.base_url))
                .json(&body)
                .send()
                .await
                .map_err(|e| BackendError::Transport { message: e.to_string() })?;

            let mut byte_stream = http_response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk) = byte_stream.next().await {
                let bytes = chunk.map_err(|e| BackendError::Transport { message: e.to_string() })?;
                buffer.push_str(&String::from_utf8_lossy(&bytes));

                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim().to_string();
                    buffer.drain(..=newline_pos);

                    if line.is_empty() {
                        continue;
                    }

                    let chunk: OllamaStreamChunk = serde_json::from_str(&line)
                        .map_err(|e| BackendError::Parse { message: e.to_string() })?;

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
        })
    }
}

/// Implements the `EmbeddingBackend` trait for `OllamaBackend`.
impl EmbeddingBackend for OllamaBackend {
    fn embed(
        &self,
        req: EmbeddingRequest,
    ) -> Pin<Box<dyn Future<Output = BackendResult<EmbeddingResponse>> + Send + '_>> {
        Box::pin(async move {
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
                .map_err(|e| BackendError::Transport {
                    message: format!("Failed to send request: {e}"),
                })?;

            let parsed_response: OllamaEmbeddingResponse =
                http_response
                    .json()
                    .await
                    .map_err(|e| BackendError::Parse {
                        message: format!("Failed to parse response: {e}"),
                    })?;

            Ok(EmbeddingResponse {
                embeddings: parsed_response.embeddings,
                model: parsed_response.model,
                usage: None,
            })
        })
    }
}

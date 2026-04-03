use ai_memory_core::backend::{
    common::BackendResult,
    embedding::{EmbeddingBackend, EmbeddingRequest, EmbeddingResponse},
    error::BackendError,
    text_generation::{
        TextGenerationBackend, TextGenerationRequest, TextGenerationResponse, TokenUsage,
    },
};
use reqwest::{Client, Response};
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
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    stream: bool,
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

    async fn check_status(resp: Response) -> BackendResult<Response> {
        let status = resp.status();
        if !status.is_success() {
            let code = status.as_u16();
            let msg = resp.text().await.unwrap_or_default();
            return Err(BackendError::Http {
                message: msg,
                status: Some(code),
            });
        }
        Ok(resp)
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
                system: req.system.as_deref(),
                stream: false,
                options: None,
                keep_alive: None,
            };

            let http_reponse = self
                .client
                .post(format!("{}/api/generate", self.config.base_url))
                .json(&body)
                .send()
                .await
                .map_err(|e| BackendError::Transport {
                    message: e.to_string(),
                })?;

            let http_response = Self::check_status(http_reponse).await?;

            let parse_response: OllamaTextGenerationResponse =
                http_response
                    .json()
                    .await
                    .map_err(|e| BackendError::Parse {
                        message: e.to_string(),
                    })?;

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

            let http_response =
                Self::check_status(http_response)
                    .await
                    .map_err(|e| BackendError::Other {
                        message: e.to_string(),
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

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

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

#[derive(Debug, Serialize)]
struct OllamaGenerateRequest<'a> {
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

#[derive(Debug, Deserialize)]
struct OllamaGenerateResponse {
    model: String,
    response: String,
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[derive(Debug, Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    input: &'a Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    model: String,
    embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct OllamaBackend {
    config: OllamaConfig,
    client: Client,
}

impl OllamaBackend {
    pub fn new(config: OllamaConfig) -> LlmResult<Self> {
        let mut builder = Client::builder();
        if let Some(timeout) = config.timeout {
            builder = builder.timeout(timeout);
        }
        let client = builder
            .build()
            .map_err(|e| LlmError::Configuration(format!("Failed to build HTTP client: {e}")))?;
        Ok(Self { config, client })
    }

    fn build_options(req: &GenerateRequest) -> Option<Value> {
        let mut opts = serde_json::Map::new();
        if let Some(t) = req.temperature {
            opts.insert("temperature".into(), json!(t));
        }
        if let Some(p) = req.top_p {
            opts.insert("top_p".into(), json!(p));
        }
        if let Some(n) = req.max_tokens {
            opts.insert("num_predict".into(), json!(n));
        }
        if let Some(stop) = &req.stop {
            opts.insert("stop".into(), json!(stop));
        }
        for (k, v) in &req.extra_params {
            opts.insert(k.clone(), v.clone());
        }
        if opts.is_empty() {
            None
        } else {
            Some(Value::Object(opts))
        }
    }

    fn duration_to_ollama(d: Duration) -> String {
        // Express as "<N>s" — Ollama accepts Go duration strings.
        format!("{}s", d.as_secs().max(1))
    }

    async fn check_status(resp: reqwest::Response) -> LlmResult<reqwest::Response> {
        let status = resp.status();
        if !status.is_success() {
            let code = status.as_u16();
            let msg = resp.text().await.unwrap_or_default();
            return Err(LlmError::Backend {
                status: code,
                message: msg,
            });
        }
        Ok(resp)
    }

    pub async fn generate(&self, req: &GenerateRequest) -> LlmResult<GenerateResponse> {
        let options = Self::build_options(req);
        let keep_alive = req.keep_alive.map(Self::duration_to_ollama);

        let body = OllamaGenerateRequest {
            model: &req.model,
            prompt: &req.prompt,
            system: req.system.as_deref(),
            stream: false,
            options,
            keep_alive,
        };

        let resp = self
            .client
            .post(format!("{}/api/generate", self.config.base_url))
            .json(&body)
            .send()
            .await?;

        let resp = Self::check_status(resp).await?;
        let raw: OllamaGenerateResponse = resp.json().await?;

        Ok(GenerateResponse {
            text: raw.response,
            model: raw.model,
            done: raw.done,
            usage: Some(TokenUsage {
                prompt_tokens: raw.prompt_eval_count,
                completion_tokens: raw.eval_count,
                total_tokens: raw
                    .prompt_eval_count
                    .zip(raw.eval_count)
                    .map(|(a, b)| a + b),
            }),
        })
    }

    pub async fn generate_stream(&self, req: &GenerateRequest) -> LlmResult<GenerateStream> {
        let options = Self::build_options(req);
        let keep_alive = req.keep_alive.map(Self::duration_to_ollama);

        let body = OllamaGenerateRequest {
            model: &req.model,
            prompt: &req.prompt,
            system: req.system.as_deref(),
            stream: true,
            options,
            keep_alive,
        };

        let resp = self
            .client
            .post(format!("{}/api/generate", self.config.base_url))
            .json(&body)
            .send()
            .await?;

        let resp = Self::check_status(resp).await?;
        let model = req.model.clone();

        let stream = resp.bytes_stream().map(move |chunk| {
            let chunk = chunk.map_err(LlmError::Http)?;
            let line = std::str::from_utf8(&chunk)
                .map_err(|e| LlmError::Stream(e.to_string()))?
                .trim()
                .to_string();

            if line.is_empty() {
                return Ok(GenerateResponse {
                    text: String::new(),
                    model: model.clone(),
                    done: false,
                    usage: None,
                });
            }

            let raw: OllamaGenerateResponse = serde_json::from_str(&line)
                .map_err(|e| LlmError::Stream(format!("Failed to parse chunk: {e}")))?;

            Ok(GenerateResponse {
                text: raw.response,
                model: raw.model,
                done: raw.done,
                usage: if raw.done {
                    Some(TokenUsage {
                        prompt_tokens: raw.prompt_eval_count,
                        completion_tokens: raw.eval_count,
                        total_tokens: raw
                            .prompt_eval_count
                            .zip(raw.eval_count)
                            .map(|(a, b)| a + b),
                    })
                } else {
                    None
                },
            })
        });

        Ok(Box::pin(stream))
    }

    pub async fn embed(&self, req: &EmbedRequest) -> LlmResult<EmbedResponse> {
        let mut options: serde_json::Map<String, Value> = serde_json::Map::new();
        for (k, v) in &req.extra_params {
            options.insert(k.clone(), v.clone());
        }
        let options = if options.is_empty() {
            None
        } else {
            Some(Value::Object(options))
        };

        let body = OllamaEmbedRequest {
            model: &req.model,
            input: &req.input,
            options,
            keep_alive: None, // keep_alive not used for embeddings by default
        };

        let resp = self
            .client
            .post(format!("{}/api/embed", self.config.base_url))
            .json(&body)
            .send()
            .await?;

        let resp = Self::check_status(resp).await?;
        let raw: OllamaEmbedResponse = resp.json().await?;

        Ok(EmbedResponse {
            embeddings: raw.embeddings,
            model: raw.model,
            usage: Some(TokenUsage {
                prompt_tokens: raw.prompt_eval_count,
                completion_tokens: None,
                total_tokens: raw.prompt_eval_count,
            }),
        })
    }
}

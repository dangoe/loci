use std::future::Future;

use serde::{Deserialize, Serialize};

use crate::error::LlmError;
use crate::llm::{LlmClient, Message};

/// An LLM client that speaks the OpenAI chat-completions API wire format.
///
/// Compatible with any service that implements the `/chat/completions` endpoint,
/// including OpenAI, Azure OpenAI, Ollama (with the `/v1` prefix), and LM Studio.
pub struct OpenAiCompatibleClient {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
}

impl OpenAiCompatibleClient {
    /// Creates a new client.
    ///
    /// * `base_url` — API root, e.g. `"https://api.openai.com/v1"`.
    /// * `model`    — Model identifier, e.g. `"gpt-4o"`.
    /// * `api_key`  — Bearer token; omit for local models that require no auth.
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            model: model.into(),
            api_key,
        }
    }
}

impl LlmClient for OpenAiCompatibleClient {
    fn complete(
        &self,
        messages: &[Message],
    ) -> impl Future<Output = Result<String, LlmError>> + Send + '_ {
        let messages: Vec<ChatMessageRequest> = messages
            .iter()
            .map(|m| ChatMessageRequest {
                role: m.role.as_str().to_string(),
                content: m.content.clone(),
            })
            .collect();

        async move {
            log::debug!(
                "calling LLM {} with {} message(s)",
                self.model,
                messages.len()
            );

            let body = ChatRequest {
                model: &self.model,
                messages,
            };

            let mut req = self
                .client
                .post(format!("{}/chat/completions", self.base_url))
                .json(&body);

            if let Some(key) = &self.api_key {
                req = req.header(reqwest::header::AUTHORIZATION, format!("Bearer {key}"));
            }

            let response = req
                .send()
                .await
                .map_err(|e| LlmError::Http(e.to_string()))?;

            if !response.status().is_success() {
                return Err(LlmError::Http(format!(
                    "LLM API returned HTTP {}",
                    response.status()
                )));
            }

            let body: ChatResponse = response
                .json()
                .await
                .map_err(|e| LlmError::Parse(e.to_string()))?;

            body.choices
                .into_iter()
                .next()
                .map(|c| c.message.content)
                .ok_or_else(|| LlmError::Parse("choices array is empty".to_string()))
        }
    }
}

// ── Request / response shapes ─────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessageRequest>,
}

#[derive(Serialize)]
struct ChatMessageRequest {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Deserialize)]
struct ChatMessageResponse {
    content: String,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;

    #[test]
    fn role_as_str_round_trips() {
        assert_eq!(Role::System.as_str(), "system");
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::Assistant.as_str(), "assistant");
    }

    #[test]
    fn client_new_stores_fields() {
        let c = OpenAiCompatibleClient::new("http://localhost/v1", "gpt-4o", Some("sk".into()));
        assert_eq!(c.base_url, "http://localhost/v1");
        assert_eq!(c.model, "gpt-4o");
        assert_eq!(c.api_key.as_deref(), Some("sk"));
    }

    #[test]
    fn client_new_no_api_key() {
        let c = OpenAiCompatibleClient::new("http://localhost/v1", "llama3", None);
        assert!(c.api_key.is_none());
    }
}

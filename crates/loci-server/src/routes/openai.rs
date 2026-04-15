// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

//! OpenAI-compatible proxy endpoint (`POST /v1/chat/completions`).
//!
//! Any client that speaks the OpenAI chat-completions wire format can point at
//! loci-server and get transparent memory enrichment without modifications.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt as _;
use loci_config::ConfigError;
use loci_core::contextualization::{
    ContextualizationMemoryMode, Contextualizer, ContextualizerConfig, ContextualizerSystemConfig,
    ContextualizerSystemMode, ContextualizerTuningConfig,
};
use loci_core::memory::Score;
use loci_core::model_provider::text_generation::TextGenerationModelProvider;
use loci_core::store::MemoryStore;
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionRequest {
    /// Ignored — loci-server uses its configured backend model, not the
    /// client-requested one.
    #[allow(dead_code)]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<NonStreamChoice>,
    usage: Option<UsageInfo>,
}

#[derive(Debug, Serialize)]
struct NonStreamChoice {
    index: u32,
    message: AssistantMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<UsageInfo>,
}

#[derive(Debug, Serialize)]
struct StreamChoice {
    index: u32,
    delta: DeltaContent,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct DeltaContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct UsageInfo {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

pub(crate) async fn chat_completions_handler<M, E>(
    State(state): State<Arc<AppState<M, E>>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response
where
    M: MemoryStore + 'static,
    E: TextGenerationModelProvider + 'static,
{
    let contextualizer = match build_contextualizer(&state, &request) {
        Ok(c) => c,
        Err(e) => {
            return Json(json!({
                "error": { "message": e.to_string(), "type": "invalid_request_error" }
            }))
            .into_response();
        }
    };

    let last_user_msg = request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    if request.stream {
        stream_response(contextualizer, last_user_msg).await
    } else {
        collect_response(contextualizer, last_user_msg).await
    }
}

async fn collect_response<M, E>(contextualizer: Contextualizer<M, E>, prompt: String) -> Response
where
    M: MemoryStore + 'static,
    E: TextGenerationModelProvider + 'static,
{
    let stream = match contextualizer.contextualize(prompt).await {
        Ok(s) => s,
        Err(e) => {
            return Json(json!({
                "error": { "message": e.to_string(), "type": "server_error" }
            }))
            .into_response();
        }
    };

    let chunks: Vec<_> = stream.collect().await;

    let mut full_text = String::new();
    let mut model_name = String::new();
    let mut usage: Option<UsageInfo> = None;

    for chunk in chunks {
        match chunk {
            Ok(resp) => {
                full_text.push_str(&resp.text);
                if !resp.model.is_empty() {
                    model_name = resp.model;
                }
                if let Some(u) = resp.usage {
                    usage = Some(UsageInfo {
                        prompt_tokens: u.prompt_tokens.unwrap_or(0),
                        completion_tokens: u.completion_tokens.unwrap_or(0),
                        total_tokens: u.total_tokens.unwrap_or(0),
                    });
                }
            }
            Err(e) => {
                return Json(json!({
                    "error": { "message": e.to_string(), "type": "server_error" }
                }))
                .into_response();
            }
        }
    }

    Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion",
        created: unix_now(),
        model: model_name,
        choices: vec![NonStreamChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: full_text,
            },
            finish_reason: "stop",
        }],
        usage,
    })
    .into_response()
}

async fn stream_response<M, E>(contextualizer: Contextualizer<M, E>, prompt: String) -> Response
where
    M: MemoryStore + 'static,
    E: TextGenerationModelProvider + 'static,
{
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = unix_now();
    let id_clone = completion_id.clone();

    // Emit the role-announcing delta first, then content chunks.
    let role_event = {
        let id = id_clone.clone();
        let chunk = ChatCompletionChunk {
            id,
            object: "chat.completion.chunk",
            created,
            model: String::new(),
            choices: vec![StreamChoice {
                index: 0,
                delta: DeltaContent {
                    role: Some("assistant"),
                    content: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        match serde_json::to_string(&chunk) {
            Ok(json) => Ok(Event::default().data(json)),
            Err(e) => Err(e),
        }
    };

    let inner_stream = match contextualizer.contextualize(prompt).await {
        Ok(s) => s,
        Err(e) => {
            return Json(json!({
                "error": { "message": e.to_string(), "type": "server_error" }
            }))
            .into_response();
        }
    };

    let id_for_stream = id_clone.clone();
    let content_stream = inner_stream.map(move |item| {
        let id = id_for_stream.clone();
        match item {
            Ok(resp) => {
                let is_done = resp.done;
                let usage = resp.usage.map(|u| UsageInfo {
                    prompt_tokens: u.prompt_tokens.unwrap_or(0),
                    completion_tokens: u.completion_tokens.unwrap_or(0),
                    total_tokens: u.total_tokens.unwrap_or(0),
                });

                let chunk = ChatCompletionChunk {
                    id,
                    object: "chat.completion.chunk",
                    created,
                    model: resp.model,
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: DeltaContent {
                            role: None,
                            content: if resp.text.is_empty() && is_done {
                                None
                            } else {
                                Some(resp.text)
                            },
                        },
                        finish_reason: if is_done { Some("stop") } else { None },
                    }],
                    usage,
                };

                serde_json::to_string(&chunk).map(|json| Event::default().data(json))
            }
            Err(e) => {
                let err_json = json!({
                    "error": { "message": e.to_string(), "type": "server_error" }
                });
                serde_json::to_string(&err_json).map(|json| Event::default().data(json))
            }
        }
    });

    // Prepend the role announcement, append [DONE]
    let done_event = futures::stream::once(async {
        Ok::<Event, serde_json::Error>(Event::default().data("[DONE]"))
    });

    let full_stream = futures::stream::once(async move { role_event })
        .chain(content_stream)
        .chain(done_event);

    Sse::new(full_stream).into_response()
}

fn build_contextualizer<M, E>(
    state: &AppState<M, E>,
    request: &ChatCompletionRequest,
) -> Result<Contextualizer<M, E>, Box<dyn std::error::Error>>
where
    M: MemoryStore,
    E: TextGenerationModelProvider + 'static,
{
    let model_key = &state.config.routing.text.default;
    let model = state
        .config
        .models
        .text
        .get(model_key)
        .ok_or_else(|| ConfigError::MissingKey {
            section: "models.text".into(),
            key: model_key.clone(),
        })?;

    // Use the system message from the incoming request, if present.
    let system_message = request
        .messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.clone());

    let tuning =
        if request.temperature.is_some() || request.max_tokens.is_some() || request.top_p.is_some()
        {
            Some(ContextualizerTuningConfig {
                temperature: request.temperature,
                max_tokens: request.max_tokens,
                top_p: request.top_p,
                ..Default::default()
            })
        } else {
            None
        };

    let config = ContextualizerConfig {
        text_generation_model: model.model.clone(),
        system: system_message.map(|s| ContextualizerSystemConfig {
            mode: ContextualizerSystemMode::Append,
            system: s,
        }),
        memory_mode: ContextualizationMemoryMode::Auto,
        max_memory_entries: 5,
        min_score: Score::ZERO,
        filters: HashMap::new(),
        tuning,
    };

    Ok(Contextualizer::new(
        Arc::clone(&state.store),
        Arc::clone(&state.llm_provider),
        config,
    ))
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

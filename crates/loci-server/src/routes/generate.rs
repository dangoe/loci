// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use futures::StreamExt as _;
use serde::{Deserialize, Serialize};

use loci_config::{AppConfig, ConfigError};
use loci_core::contextualization::{
    ContextualizationMemoryMode, Contextualizer, ContextualizerConfig, ContextualizerSystemConfig,
    ContextualizerSystemMode,
};
use loci_core::memory::Score;
use loci_core::model_provider::text_generation::TokenUsage;

use crate::state::AppState;

#[derive(Deserialize)]
pub(crate) struct GenerateRequest {
    pub prompt: String,
    pub system: Option<String>,
    #[serde(default)]
    pub system_mode: ApiSystemMode,
    #[serde(default = "default_max_memory_entries")]
    pub max_memory_entries: usize,
    #[serde(default)]
    pub min_score: f64,
    #[serde(default)]
    pub memory_mode: ApiMemoryMode,
    #[serde(default)]
    pub filters: HashMap<String, String>,
}

fn default_max_memory_entries() -> usize {
    5
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ApiSystemMode {
    #[default]
    Append,
    Replace,
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ApiMemoryMode {
    #[default]
    Auto,
    Off,
}

#[derive(Serialize)]
struct GenerateChunk {
    text: String,
    model: String,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<ApiTokenUsage>,
}

#[derive(Serialize)]
struct ApiTokenUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

impl From<TokenUsage> for ApiTokenUsage {
    fn from(u: TokenUsage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }
    }
}

#[derive(Serialize)]
struct GenerateErrorChunk {
    error: String,
    code: i32,
}

pub(crate) async fn generate_stream_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<GenerateRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Convert config error to a string so the stream closure can be Send.
    let ctx_config_result =
        build_contextualizer_config(&state.config, &body).map_err(|e| e.to_string());

    let sse_stream = async_stream::stream! {
        let ctx_config = match ctx_config_result {
            Ok(cfg) => cfg,
            Err(e) => {
                let chunk = GenerateErrorChunk {
                    error: e,
                    code: -32602,
                };
                let data = serde_json::to_string(&chunk).unwrap_or_default();
                yield Ok::<Event, Infallible>(Event::default().event("error").data(data));
                return;
            }
        };

        let contextualizer = Contextualizer::new(
            Arc::clone(&state.store),
            Arc::clone(&state.llm_provider),
            ctx_config,
        );

        match contextualizer.contextualize(body.prompt).await {
            Err(e) => {
                let chunk = GenerateErrorChunk {
                    error: e.to_string(),
                    code: -32002,
                };
                let data = serde_json::to_string(&chunk).unwrap_or_default();
                yield Ok(Event::default().event("error").data(data));
            }
            Ok(stream) => {
                let mut pinned = std::pin::pin!(stream);
                while let Some(result) = pinned.next().await {
                    match result {
                        Ok(chunk) => {
                            let response = GenerateChunk {
                                text: chunk.text,
                                model: chunk.model,
                                done: chunk.done,
                                usage: chunk.usage.map(ApiTokenUsage::from),
                            };
                            let data = serde_json::to_string(&response).unwrap_or_default();
                            yield Ok(Event::default().data(data));
                        }
                        Err(e) => {
                            let error_chunk = GenerateErrorChunk {
                                error: e.to_string(),
                                code: -32002,
                            };
                            let data = serde_json::to_string(&error_chunk).unwrap_or_default();
                            yield Ok(Event::default().event("error").data(data));
                        }
                    }
                }
            }
        }
    };

    Sse::new(sse_stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

fn build_contextualizer_config(
    config: &AppConfig,
    body: &GenerateRequest,
) -> Result<ContextualizerConfig, Box<dyn std::error::Error>> {
    let model_key = &config.routing.text.default;
    let model = config
        .models
        .text
        .get(model_key)
        .ok_or_else(|| ConfigError::MissingKey {
            section: "models.text".into(),
            key: model_key.clone(),
        })?;

    let min_score = Score::new(body.min_score).map_err(|e| format!("invalid min_score: {e}"))?;

    Ok(ContextualizerConfig {
        text_generation_model: model.model.clone(),
        system: body.system.as_ref().map(|sys| ContextualizerSystemConfig {
            mode: match body.system_mode {
                ApiSystemMode::Append => ContextualizerSystemMode::Append,
                ApiSystemMode::Replace => ContextualizerSystemMode::Replace,
            },
            system: sys.clone(),
        }),
        memory_mode: match body.memory_mode {
            ApiMemoryMode::Auto => ContextualizationMemoryMode::Auto,
            ApiMemoryMode::Off => ContextualizationMemoryMode::Off,
        },
        max_memory_entries: body.max_memory_entries,
        min_score,
        filters: body.filters.clone(),
        tuning: None,
    })
}

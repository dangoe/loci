// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

use std::pin::Pin;
use std::sync::Arc;

use buffa::MessageField;
use buffa::view::OwnedView;
use connectrpc::{ConnectError, Context};
use futures::{Stream, StreamExt as _};

use loci_config::ConfigError;
use loci_core::contextualization::{
    ContextualizationMemoryMode, Contextualizer, ContextualizerConfig, ContextualizerSystemConfig,
    ContextualizerSystemMode,
};
use loci_core::memory::Score;
use loci_core::memory_store::MemoryStore;
use loci_core::model_provider::text_generation::{
    TextGenerationModelProvider, TokenUsage as CoreTokenUsage,
};

use crate::loci::generate::v1::{
    GenerateServiceGenerateRequestView, GenerateServiceGenerateResponse, MemoryMode, SystemMode,
    TokenUsage,
};
use crate::state::AppState;

pub(crate) struct GenerateServiceImpl<M, E>
where
    M: MemoryStore,
    E: TextGenerationModelProvider + 'static,
{
    state: Arc<AppState<M, E>>,
}

impl<M, E> GenerateServiceImpl<M, E>
where
    M: MemoryStore,
    E: TextGenerationModelProvider + 'static,
{
    pub(crate) fn new(state: Arc<AppState<M, E>>) -> Self {
        Self { state }
    }
}

type GenerateStream =
    Pin<Box<dyn Stream<Item = Result<GenerateServiceGenerateResponse, ConnectError>> + Send>>;

impl<M, E> crate::loci::generate::v1::GenerateService for GenerateServiceImpl<M, E>
where
    M: MemoryStore + 'static,
    E: TextGenerationModelProvider + 'static,
{
    #[allow(clippy::result_large_err)]
    async fn generate(
        &self,
        ctx: Context,
        request: OwnedView<GenerateServiceGenerateRequestView<'static>>,
    ) -> Result<(GenerateStream, Context), ConnectError> {
        let ctx_config = build_contextualizer_config(&self.state, &request)?;

        let contextualizer = Contextualizer::new(
            Arc::clone(&self.state.store),
            Arc::clone(&self.state.llm_provider),
            ctx_config,
        );

        let prompt = request.prompt.to_owned();
        let stream = contextualizer
            .contextualize(prompt)
            .await
            .map_err(|e| ConnectError::internal(e.to_string()))?;

        let mapped = stream.map(|item| {
            item.map(|chunk| GenerateServiceGenerateResponse {
                text: chunk.text,
                model: chunk.model,
                done: chunk.done,
                usage: chunk
                    .usage
                    .map(token_usage_to_proto)
                    .map(MessageField::some)
                    .unwrap_or_default(),
                ..Default::default()
            })
            .map_err(|e| ConnectError::internal(e.to_string()))
        });

        Ok((Box::pin(mapped), ctx))
    }
}

#[allow(clippy::result_large_err)]
fn build_contextualizer_config<M, E>(
    state: &AppState<M, E>,
    request: &GenerateServiceGenerateRequestView<'_>,
) -> Result<ContextualizerConfig, ConnectError>
where
    M: MemoryStore,
    E: TextGenerationModelProvider + 'static,
{
    let model_key = &state.config.routing.text.default;
    let model = state.config.models.text.get(model_key).ok_or_else(|| {
        ConnectError::internal(
            ConfigError::MissingKey {
                section: "models.text".into(),
                key: model_key.clone(),
            }
            .to_string(),
        )
    })?;

    let min_score = Score::new(request.min_score)
        .map_err(|_| ConnectError::invalid_argument("min_score must be in [0.0, 1.0]"))?;

    Ok(ContextualizerConfig {
        text_generation_model: model.model.clone(),
        system: request.system.map(|sys| ContextualizerSystemConfig {
            mode: match request.system_mode.as_known() {
                Some(SystemMode::SYSTEM_MODE_REPLACE) => ContextualizerSystemMode::Replace,
                _ => ContextualizerSystemMode::Append,
            },
            system: sys.to_owned(),
        }),
        memory_mode: match request.memory_mode.as_known() {
            Some(MemoryMode::MEMORY_MODE_OFF) => ContextualizationMemoryMode::Off,
            _ => ContextualizationMemoryMode::Auto,
        },
        max_memory_entries: request.max_memory_entries as usize,
        min_score,
        filters: request
            .filters
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        tuning: None,
    })
}

fn token_usage_to_proto(usage: CoreTokenUsage) -> TokenUsage {
    TokenUsage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
        ..Default::default()
    }
}

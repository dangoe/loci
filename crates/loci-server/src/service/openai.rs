// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

use std::collections::HashMap;
use std::sync::Arc;

use loci_config::ConfigError;
use loci_core::contextualization::{
    ContextualizationMemoryMode, Contextualizer, ContextualizerConfig, ContextualizerSystemConfig,
    ContextualizerSystemMode, ContextualizerTuningConfig, ResultStream,
};
use loci_core::error::ContextualizerError;
use loci_core::memory::Score;
use loci_core::memory::store::MemoryStore;
use loci_core::model_provider::text_generation::TextGenerationModelProvider;

use crate::state::AppState;

/// Protocol-agnostic input for a single chat completion request.
pub(crate) struct ChatCompletionInput {
    /// The final user message — used as the memory-retrieval prompt and LLM prompt.
    prompt: String,
    /// Optional system message from the incoming request. Appended to the
    /// loci base template when present.
    system: Option<String>,
    /// Optional generation tuning forwarded from the incoming request.
    tuning: Option<ContextualizerTuningConfig>,
}

impl ChatCompletionInput {
    /// Constructs a new `ChatCompletionInput`.
    pub(crate) fn new(
        prompt: String,
        system: Option<String>,
        tuning: Option<ContextualizerTuningConfig>,
    ) -> Self {
        Self {
            prompt,
            system,
            tuning,
        }
    }

    /// Returns the prompt text.
    pub(crate) fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Returns the optional system message.
    pub(crate) fn system(&self) -> Option<&str> {
        self.system.as_deref()
    }

    /// Returns the optional tuning configuration.
    pub(crate) fn tuning(&self) -> Option<&ContextualizerTuningConfig> {
        self.tuning.as_ref()
    }
}

/// Errors produced by [`OpenAICompletionService::complete`].
#[derive(Debug)]
pub(crate) enum OpenAIServiceError {
    /// The routing config references a text model that does not exist.
    MissingModel(String),
    /// The contextualizer failed (memory store error or model provider error).
    Contextualization(ContextualizerError),
}

impl std::fmt::Display for OpenAIServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingModel(key) => write!(
                f,
                "{}",
                ConfigError::MissingKey {
                    section: "models.text".into(),
                    key: key.clone(),
                }
            ),
            Self::Contextualization(e) => write!(f, "{e}"),
        }
    }
}

/// Service that enriches an OpenAI-compatible chat completion request with
/// retrieved memory entries and forwards it to the configured LLM.
pub(crate) struct OpenAICompletionService<M, E>
where
    M: MemoryStore,
    E: TextGenerationModelProvider + 'static,
{
    state: Arc<AppState<M, E>>,
}

impl<M, E> OpenAICompletionService<M, E>
where
    M: MemoryStore + 'static,
    E: TextGenerationModelProvider + 'static,
{
    pub(crate) fn new(state: Arc<AppState<M, E>>) -> Self {
        Self { state }
    }

    /// Run the full memory-enriched completion pipeline and return a stream of
    /// [`loci_core::model_provider::text_generation::TextGenerationResponse`] chunks.
    pub(crate) async fn complete(
        &self,
        input: ChatCompletionInput,
    ) -> Result<ResultStream, OpenAIServiceError> {
        let prompt = input.prompt().to_owned();
        let config = self.build_contextualizer_config(&input)?;
        let contextualizer = Contextualizer::new(
            Arc::clone(self.state.store()),
            Arc::clone(self.state.llm_provider()),
            config,
        );
        contextualizer
            .contextualize(prompt)
            .await
            .map_err(OpenAIServiceError::Contextualization)
    }

    fn build_contextualizer_config(
        &self,
        input: &ChatCompletionInput,
    ) -> Result<ContextualizerConfig, OpenAIServiceError> {
        let model_key = self.state.config().routing().text().default();
        let model = self
            .state
            .config()
            .models()
            .text()
            .get(model_key)
            .ok_or_else(|| OpenAIServiceError::MissingModel(model_key.to_owned()))?;

        Ok(ContextualizerConfig::new(
            model.model().to_owned(),
            input.system().map(|s| {
                ContextualizerSystemConfig::new(ContextualizerSystemMode::Append, s.to_owned())
            }),
            ContextualizationMemoryMode::Auto,
            std::num::NonZeroUsize::new(5).unwrap(),
            Score::ZERO,
            HashMap::new(),
            input.tuning().cloned(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use loci_core::contextualization::ContextualizerTuningConfig;
    use loci_core::model_provider::common::ModelProviderResult;
    use loci_core::model_provider::text_generation::{
        TextGenerationModelProvider, TextGenerationRequest, TextGenerationResponse,
    };
    use loci_core::testing::{AddEntriesBehavior, MockStore, QueryBehavior};
    use pretty_assertions::assert_eq;

    use crate::state::AppState;
    use crate::testing::mock_config;

    use super::*;

    struct StubProvider;
    impl TextGenerationModelProvider for StubProvider {
        async fn generate(
            &self,
            req: TextGenerationRequest,
        ) -> ModelProviderResult<TextGenerationResponse> {
            Ok(TextGenerationResponse::new_done(
                "ok".into(),
                req.model().to_owned(),
                None,
            ))
        }
    }

    fn make_service() -> OpenAICompletionService<MockStore, StubProvider> {
        let config = mock_config();
        let store = Arc::new(
            MockStore::new()
                .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![]))
                .with_query_behavior(QueryBehavior::Ok(vec![])),
        );
        let provider = Arc::new(StubProvider);
        let state = Arc::new(AppState::new(store, provider, Arc::new(config)));
        OpenAICompletionService::new(state)
    }

    #[test]
    fn test_build_config_uses_default_text_model() {
        let svc = make_service();
        let input = ChatCompletionInput::new("test".into(), None, None);
        let config = svc.build_contextualizer_config(&input).unwrap();
        assert_eq!(config.text_generation_model(), "test-text-model");
    }

    #[test]
    fn test_build_config_sets_system_append_mode_when_system_present() {
        let svc = make_service();
        let input = ChatCompletionInput::new("test".into(), Some("Be concise.".into()), None);
        let config = svc.build_contextualizer_config(&input).unwrap();
        let system = config.system().unwrap();
        assert_eq!(system.system(), "Be concise.");
        assert!(matches!(system.mode(), ContextualizerSystemMode::Append));
    }

    #[test]
    fn test_build_config_returns_error_when_model_key_missing() {
        let mut cfg = mock_config();
        cfg.routing_mut().text_mut().set_default("nonexistent");
        let store = Arc::new(
            MockStore::new()
                .with_add_entries_behavior(AddEntriesBehavior::Ok(vec![]))
                .with_query_behavior(QueryBehavior::Ok(vec![])),
        );
        let state = Arc::new(AppState::new(store, Arc::new(StubProvider), Arc::new(cfg)));
        let svc = OpenAICompletionService::new(state);
        let input = ChatCompletionInput::new("test".into(), None, None);
        let err = svc.build_contextualizer_config(&input).unwrap_err();
        assert!(
            matches!(err, OpenAIServiceError::MissingModel(ref k) if k == "nonexistent"),
            "expected MissingModel, got: {err}"
        );
    }

    #[test]
    fn test_build_config_memory_mode_is_auto() {
        let svc = make_service();
        let input = ChatCompletionInput::new("test".into(), None, None);
        let config = svc.build_contextualizer_config(&input).unwrap();
        assert!(matches!(
            config.memory_mode(),
            ContextualizationMemoryMode::Auto
        ));
    }

    #[test]
    fn test_build_config_tuning_is_forwarded() {
        let svc = make_service();
        let input = ChatCompletionInput::new(
            "test".into(),
            None,
            Some(ContextualizerTuningConfig::new(
                Some(0.5),
                Some(128),
                None,
                None,
                None,
                None,
                None,
                None,
                Default::default(),
            )),
        );
        let config = svc.build_contextualizer_config(&input).unwrap();
        let t = config.tuning().cloned().unwrap();
        assert_eq!(t.temperature(), Some(0.5));
        assert_eq!(t.max_tokens(), Some(128));
    }
}

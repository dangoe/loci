// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-wire.

//! Model-provider builders: constructs concrete provider instances from parsed
//! [`AppConfig`] sections.

use std::pin::Pin;

use futures::Stream;
use log::info;

use loci_config::{AppConfig, ConfigError, ModelProviderConfig, ModelProviderKind};
use loci_core::model_provider::common::ModelProviderResult;
use loci_core::model_provider::text_generation::{
    TextGenerationModelProvider, TextGenerationRequest, TextGenerationResponse,
};
use loci_model_provider_ollama::provider::{OllamaConfig, OllamaModelProvider};
use loci_model_provider_openai::provider::{OpenAIConfig, OpenAIModelProvider};

/// A runtime-selected text-generation provider.
///
/// Allows callers to support multiple provider backends (Ollama,
/// OpenAI-compatible, …) without monomorphising the entire application for each
/// variant.
pub enum AnyModelProvider {
    Ollama(OllamaModelProvider),
    OpenAI(OpenAIModelProvider),
}

impl TextGenerationModelProvider for AnyModelProvider {
    async fn generate(
        &self,
        req: TextGenerationRequest,
    ) -> ModelProviderResult<TextGenerationResponse> {
        match self {
            AnyModelProvider::Ollama(p) => p.generate(req).await,
            AnyModelProvider::OpenAI(p) => p.generate(req).await,
        }
    }

    fn generate_stream(
        &self,
        req: TextGenerationRequest,
    ) -> impl Stream<Item = ModelProviderResult<TextGenerationResponse>> + Send + '_ {
        let boxed: Pin<
            Box<dyn Stream<Item = ModelProviderResult<TextGenerationResponse>> + Send + '_>,
        > = match self {
            AnyModelProvider::Ollama(p) => Box::pin(p.generate_stream(req)),
            AnyModelProvider::OpenAI(p) => Box::pin(p.generate_stream(req)),
        };
        boxed
    }
}

/// Builds an [`AnyModelProvider`] for text generation using the default model's provider.
pub fn build_llm_provider(
    config: &AppConfig,
) -> Result<AnyModelProvider, Box<dyn std::error::Error>> {
    let provider = resolve_llm_provider(config)?;
    match provider.kind() {
        ModelProviderKind::Ollama => {
            info!("Using Ollama model provider at {}", provider.endpoint());
            let cfg = OllamaConfig::new(provider.endpoint());
            Ok(AnyModelProvider::Ollama(OllamaModelProvider::new(cfg)?))
        }
        ModelProviderKind::OpenAI => {
            info!(
                "Using OpenAI-compatible model provider at {}",
                provider.endpoint()
            );
            let mut cfg = OpenAIConfig::new(provider.endpoint());
            if let Some(key) = provider.api_key() {
                cfg = cfg.with_api_key(key.to_owned());
            }
            Ok(AnyModelProvider::OpenAI(OpenAIModelProvider::new(cfg)?))
        }
        ModelProviderKind::Anthropic => Err(Box::new(ConfigError::UnsupportedKind {
            kind: "anthropic".into(),
            context: "provider".into(),
        })),
    }
}

/// Constructs an [`OllamaModelProvider`] from a provider config, failing if the
/// provider kind is not `ollama`.
pub fn build_ollama_provider(
    provider: &ModelProviderConfig,
) -> Result<OllamaModelProvider, Box<dyn std::error::Error>> {
    match provider.kind() {
        ModelProviderKind::Ollama => {
            let cfg = OllamaConfig::new(provider.endpoint());
            info!("Using Ollama model provider at {}", provider.endpoint());
            Ok(OllamaModelProvider::new(cfg)?)
        }
        ModelProviderKind::OpenAI => Err(Box::new(ConfigError::UnsupportedKind {
            kind: "openai".into(),
            context: "provider".into(),
        })),
        ModelProviderKind::Anthropic => Err(Box::new(ConfigError::UnsupportedKind {
            kind: "anthropic".into(),
            context: "provider".into(),
        })),
    }
}

/// Resolves the [`ModelProviderConfig`] for the active embedding profile.
pub fn resolve_embedding_provider(
    config: &AppConfig,
) -> Result<&ModelProviderConfig, Box<dyn std::error::Error>> {
    let profile_name = config.embedding().model();
    let profile = config
        .resources()
        .models()
        .embedding()
        .get(profile_name)
        .ok_or_else(|| ConfigError::MissingKey {
            section: "resources.models.embedding".into(),
            key: profile_name.to_owned(),
        })?;
    config
        .resources()
        .model_providers()
        .get(profile.provider())
        .ok_or_else(|| {
            Box::new(ConfigError::MissingKey {
                section: "resources.model_providers".into(),
                key: profile.provider().to_owned(),
            }) as Box<dyn std::error::Error>
        })
}

/// Resolves the [`ModelProviderConfig`] for the default LLM model.
pub fn resolve_llm_provider(
    config: &AppConfig,
) -> Result<&ModelProviderConfig, Box<dyn std::error::Error>> {
    let model_name = config.generation().text().model();
    let model = config
        .resources()
        .models()
        .text()
        .get(model_name)
        .ok_or_else(|| ConfigError::MissingKey {
            section: "resources.models.text".into(),
            key: model_name.to_owned(),
        })?;
    config
        .resources()
        .model_providers()
        .get(model.provider())
        .ok_or_else(|| {
            Box::new(ConfigError::MissingKey {
                section: "resources.model_providers".into(),
                key: model.provider().to_owned(),
            }) as Box<dyn std::error::Error>
        })
}

#[cfg(test)]
mod tests {
    use loci_config::{ModelProviderConfig, ModelProviderKind};

    use crate::testing::minimal_ollama_config;

    use super::*;

    #[test]
    fn test_resolve_embedding_provider_returns_provider_config() {
        let config = minimal_ollama_config();
        let provider = resolve_embedding_provider(&config).unwrap();
        assert_eq!(provider.endpoint(), "http://localhost:11434");
        assert_eq!(*provider.kind(), ModelProviderKind::Ollama);
    }

    #[test]
    fn test_resolve_embedding_provider_missing_embedding_key_returns_err() {
        let mut config = minimal_ollama_config();
        config.embedding_mut().set_model("nonexistent");

        let err = resolve_embedding_provider(&config).unwrap_err();
        assert!(
            err.to_string().contains("nonexistent"),
            "expected missing key name in error, got: {err}"
        );
    }

    #[test]
    fn test_resolve_embedding_provider_missing_provider_key_returns_err() {
        let mut config = minimal_ollama_config();
        config
            .resources_mut()
            .models_mut()
            .embedding_entries_mut()
            .get_mut("default")
            .unwrap()
            .set_provider("ghost");

        let err = resolve_embedding_provider(&config).unwrap_err();
        assert!(
            err.to_string().contains("ghost"),
            "expected missing provider name in error, got: {err}"
        );
    }

    #[test]
    fn test_resolve_llm_provider_returns_provider_config() {
        let config = minimal_ollama_config();
        let provider = resolve_llm_provider(&config).unwrap();
        assert_eq!(provider.endpoint(), "http://localhost:11434");
    }

    #[test]
    fn test_resolve_llm_provider_missing_model_returns_err() {
        let mut config = minimal_ollama_config();
        config.generation_mut().text_mut().set_model("nonexistent");

        let err = resolve_llm_provider(&config).unwrap_err();
        assert!(
            err.to_string().contains("nonexistent"),
            "expected missing key name in error, got: {err}"
        );
    }

    #[test]
    fn test_resolve_llm_provider_missing_provider_returns_err() {
        let mut config = minimal_ollama_config();
        config
            .resources_mut()
            .models_mut()
            .text_entries_mut()
            .get_mut("default")
            .unwrap()
            .set_provider("ghost");

        let err = resolve_llm_provider(&config).unwrap_err();
        assert!(
            err.to_string().contains("ghost"),
            "expected missing provider name in error, got: {err}"
        );
    }

    #[test]
    fn test_build_ollama_provider_succeeds_for_ollama_kind() {
        let cfg =
            ModelProviderConfig::new(ModelProviderKind::Ollama, "http://localhost:11434", None);
        assert!(build_ollama_provider(&cfg).is_ok());
    }

    #[test]
    fn test_build_ollama_provider_fails_for_openai_kind() {
        let cfg =
            ModelProviderConfig::new(ModelProviderKind::OpenAI, "https://api.openai.com/v1", None);
        let err = build_ollama_provider(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("openai"),
            "expected 'openai' in error, got: {err}"
        );
    }

    #[test]
    fn test_build_ollama_provider_fails_for_anthropic_kind() {
        let cfg = ModelProviderConfig::new(
            ModelProviderKind::Anthropic,
            "https://api.anthropic.com",
            None,
        );
        let err = build_ollama_provider(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("anthropic"),
            "expected 'anthropic' in error, got: {err}"
        );
    }
}

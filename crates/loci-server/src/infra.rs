// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

// TODO: Extract shared infrastructure builders to a dedicated `loci-infra` crate
// when a third consumer of these functions appears.

//! Infrastructure builders: constructs concrete store and provider instances
//! from parsed [`AppConfig`] sections.

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use log::info;

use loci_config::{AppConfig, ConfigError, ModelProviderConfig, ModelProviderKind, StoreConfig};
use loci_core::embedding::DefaultTextEmbedder;
use loci_core::model_provider::common::ModelProviderResult;
use loci_core::model_provider::text_generation::{
    TextGenerationModelProvider, TextGenerationRequest, TextGenerationResponse,
};
use loci_memory_store_qdrant::config::QdrantConfig;
use loci_memory_store_qdrant::store::QdrantMemoryStore;
use loci_model_provider_ollama::provider::{OllamaConfig, OllamaModelProvider};
use loci_model_provider_openai::provider::{OpenAIConfig, OpenAIModelProvider};

/// A runtime-selected text-generation provider.
///
/// Used as the concrete `E` type parameter in [`AppState`] so that loci-server
/// can support multiple provider backends (Ollama, OpenAI-compatible, …)
/// without monomorphising the entire server for each variant.
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

/// Builds a `QdrantMemoryStore<DefaultTextEmbedder<OllamaModelProvider>>` from the active config.
pub async fn build_store(
    config: &AppConfig,
) -> Result<QdrantMemoryStore<DefaultTextEmbedder<OllamaModelProvider>>, Box<dyn std::error::Error>>
{
    let backend_name = &config.memory.config.backend;
    let store_cfg =
        config
            .memory
            .backends
            .get(backend_name)
            .ok_or_else(|| ConfigError::MissingKey {
                section: "memory.backends".into(),
                key: backend_name.clone(),
            })?;

    match store_cfg {
        StoreConfig::Qdrant {
            url, collection, ..
        } => {
            let embed_provider = resolve_embedding_provider(config)?;
            let embed_provider_instance = build_ollama_provider(embed_provider)?;
            let embed_profile_name = &config.routing.embedding.default;
            let embed_profile =
                config
                    .models
                    .embedding
                    .get(embed_profile_name)
                    .ok_or_else(|| ConfigError::MissingKey {
                        section: "models.embedding".into(),
                        key: embed_profile_name.clone(),
                    })?;

            let embedder = DefaultTextEmbedder::new(
                Arc::new(embed_provider_instance),
                &embed_profile.model,
                embed_profile.dimension,
            );

            let qdrant_config = QdrantConfig {
                collection_name: collection.clone(),
                similarity_threshold: config.memory.config.similarity_threshold,
            };

            info!("Connecting to Qdrant at {url}");
            let store = QdrantMemoryStore::new(url, qdrant_config, embedder)?;
            store.initialize().await?;
            info!("Memory store initialized.");
            Ok(store)
        }
        StoreConfig::Markdown { .. } => Err(Box::new(ConfigError::UnsupportedKind {
            kind: "markdown".into(),
            context: "memory store".into(),
        })),
    }
}

/// Builds an [`AnyModelProvider`] for text generation using the default model's provider.
pub fn build_llm_provider(
    config: &AppConfig,
) -> Result<AnyModelProvider, Box<dyn std::error::Error>> {
    let provider = resolve_llm_provider(config)?;
    match provider.kind {
        ModelProviderKind::Ollama => {
            info!("Using Ollama model provider at {}", provider.endpoint);
            let cfg = OllamaConfig {
                base_url: provider.endpoint.clone(),
                timeout: None,
            };
            Ok(AnyModelProvider::Ollama(OllamaModelProvider::new(cfg)?))
        }
        ModelProviderKind::OpenAI => {
            info!(
                "Using OpenAI-compatible model provider at {}",
                provider.endpoint
            );
            let cfg = OpenAIConfig {
                base_url: provider.endpoint.clone(),
                api_key: provider.api_key.clone(),
                timeout: None,
            };
            Ok(AnyModelProvider::OpenAI(OpenAIModelProvider::new(cfg)?))
        }
        ModelProviderKind::Anthropic => Err(Box::new(ConfigError::UnsupportedKind {
            kind: "anthropic".into(),
            context: "provider".into(),
        })),
    }
}

fn resolve_embedding_provider(
    config: &AppConfig,
) -> Result<&ModelProviderConfig, Box<dyn std::error::Error>> {
    let profile_name = &config.routing.embedding.default;
    let profile =
        config
            .models
            .embedding
            .get(profile_name)
            .ok_or_else(|| ConfigError::MissingKey {
                section: "models.embedding".into(),
                key: profile_name.clone(),
            })?;
    config.providers.get(&profile.provider).ok_or_else(|| {
        Box::new(ConfigError::MissingKey {
            section: "providers".into(),
            key: profile.provider.clone(),
        }) as Box<dyn std::error::Error>
    })
}

fn resolve_llm_provider(
    config: &AppConfig,
) -> Result<&ModelProviderConfig, Box<dyn std::error::Error>> {
    let model_name = &config.routing.text.default;
    let model = config
        .models
        .text
        .get(model_name)
        .ok_or_else(|| ConfigError::MissingKey {
            section: "models.text".into(),
            key: model_name.clone(),
        })?;
    config.providers.get(&model.provider).ok_or_else(|| {
        Box::new(ConfigError::MissingKey {
            section: "providers".into(),
            key: model.provider.clone(),
        }) as Box<dyn std::error::Error>
    })
}

fn build_ollama_provider(
    provider: &ModelProviderConfig,
) -> Result<OllamaModelProvider, Box<dyn std::error::Error>> {
    match provider.kind {
        ModelProviderKind::Ollama => {
            let cfg = OllamaConfig {
                base_url: provider.endpoint.clone(),
                timeout: None,
            };
            info!("Using Ollama embedding provider at {}", provider.endpoint);
            Ok(OllamaModelProvider::new(cfg)?)
        }
        ModelProviderKind::OpenAI | ModelProviderKind::Anthropic => {
            Err(Box::new(ConfigError::UnsupportedKind {
                kind: provider.kind.to_string(),
                context: "embedding provider".into(),
            }))
        }
    }
}

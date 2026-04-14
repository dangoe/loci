// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

// TODO: Extract shared infrastructure builders to a dedicated `loci-infra` crate
// when a third consumer of these functions appears.

//! Infrastructure builders: constructs concrete store and provider instances
//! from parsed [`AppConfig`] sections.

use std::sync::Arc;

use log::info;

use loci_config::{AppConfig, ConfigError, ModelProviderConfig, ModelProviderKind, StoreConfig};
use loci_core::embedding::DefaultTextEmbedder;
use loci_memory_store_qdrant::config::QdrantConfig;
use loci_memory_store_qdrant::store::QdrantMemoryStore;
use loci_model_provider_ollama::provider::{OllamaConfig, OllamaModelProvider};

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
                promotion_source_threshold: config.memory.config.promotion_source_threshold,
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

/// Builds an [`OllamaModelProvider`] for text generation using the default model's provider.
pub fn build_llm_provider(
    config: &AppConfig,
) -> Result<OllamaModelProvider, Box<dyn std::error::Error>> {
    let provider = resolve_llm_provider(config)?;
    build_ollama_provider(provider)
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
            info!("Using Ollama model provider at {}", provider.endpoint);
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

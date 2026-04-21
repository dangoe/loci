// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-wire.

//! Memory-store builders: constructs concrete store instances from parsed
//! [`AppConfig`] sections.

use std::sync::Arc;

use log::info;

use loci_config::{AppConfig, ConfigError, StoreConfig};
use loci_core::embedding::DefaultTextEmbedder;
use loci_memory_store_qdrant::config::QdrantConfig;
use loci_memory_store_qdrant::store::QdrantMemoryStore;
use loci_model_provider_ollama::provider::OllamaModelProvider;

use crate::provider::{build_ollama_provider, resolve_embedding_provider};

/// Builds a `QdrantMemoryStore<DefaultTextEmbedder<OllamaModelProvider>>` from the active config.
pub async fn build_store(
    config: &AppConfig,
) -> Result<QdrantMemoryStore<DefaultTextEmbedder<OllamaModelProvider>>, Box<dyn std::error::Error>>
{
    let store_name = config.memory().store();
    let store_cfg = config
        .resources()
        .memory_stores()
        .get(store_name)
        .ok_or_else(|| ConfigError::MissingKey {
            section: "resources.memory_stores".into(),
            key: store_name.to_owned(),
        })?;

    match store_cfg {
        StoreConfig::Qdrant {
            url, collection, ..
        } => {
            let embed_provider = resolve_embedding_provider(config)?;
            let embed_provider_instance = build_ollama_provider(embed_provider)?;
            let embed_profile_name = config.embedding().model();
            let embed_profile = config
                .resources()
                .models()
                .embedding()
                .get(embed_profile_name)
                .ok_or_else(|| ConfigError::MissingKey {
                    section: "resources.models.embedding".into(),
                    key: embed_profile_name.to_owned(),
                })?;

            let embedder = DefaultTextEmbedder::new(
                Arc::new(embed_provider_instance),
                embed_profile.model(),
                embed_profile.dimension(),
            );

            let mut qdrant_config = QdrantConfig::new(collection.clone());
            if let Some(threshold) = config.memory().similarity_threshold() {
                qdrant_config = qdrant_config.with_similarity_threshold(threshold);
            }

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

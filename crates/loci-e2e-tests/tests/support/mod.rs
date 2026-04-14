// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-e2e-tests.

// Each test binary gets its own copy of this module; items not needed by a
// particular binary show up as dead_code warnings even though they are used
// by other binaries.
#![allow(dead_code)]

use std::sync::Arc;

use loci_core::embedding::DefaultTextEmbedder;
use loci_memory_store_qdrant::store::QdrantMemoryStore;
pub use loci_memory_store_qdrant::testing::start_qdrant_container;
use loci_model_provider_ollama::provider::OllamaModelProvider;
use loci_model_provider_ollama::testing::embedding_model;
use testcontainers::{ContainerAsync, GenericImage};

use loci_memory_store_qdrant::config::QdrantConfig;

/// Embedding vector dimension produced by the default Ollama embedding model.
pub const EMBEDDING_DIM: usize = 1024;

pub type RealEmbedder = DefaultTextEmbedder<OllamaModelProvider>;

pub fn create_embedder(provider: Arc<OllamaModelProvider>) -> RealEmbedder {
    DefaultTextEmbedder::new(provider, embedding_model(), EMBEDDING_DIM)
}

/// Starts a Qdrant container and initialises a `QdrantMemoryStore` against it.
///
/// Returns `(store, container)`. Keep `container` alive for the test duration.
pub async fn start_qdrant_store(
    embedder: RealEmbedder,
    similarity_threshold: Option<f64>,
) -> (
    QdrantMemoryStore<RealEmbedder>,
    ContainerAsync<GenericImage>,
) {
    let (container, url) = start_qdrant_container().await;

    let config = QdrantConfig {
        collection_name: "memory_entries".to_string(),
        similarity_threshold,
        promotion_source_threshold: 2,
    };

    let store =
        QdrantMemoryStore::new(&url, config, embedder).expect("failed to create Qdrant client");
    store
        .initialize()
        .await
        .expect("failed to initialize Qdrant collection");

    (store, container)
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-e2e-tests.

//! Shared test infrastructure: Qdrant container helpers and type aliases
//! used across multiple e2e test files.

// Each test binary gets its own copy of this module; items not needed by a
// particular binary show up as dead_code warnings even though they are used
// by other binaries.
#![allow(dead_code)]

use std::sync::Arc;

use loci_core::embedding::DefaultTextEmbedder;
use loci_memory_store_qdrant::config::QdrantConfig;
use loci_memory_store_qdrant::store::QdrantMemoryStore;
use loci_model_provider_ollama::provider::OllamaModelProvider;
use loci_model_provider_ollama::testing::embedding_model;
use testcontainers::core::wait::HttpWaitStrategy;
use testcontainers::core::{ContainerPort, IntoContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage};

/// gRPC port used by Qdrant (used to obtain the mapped host port).
pub const QDRANT_GRPC_PORT: u16 = 6334;

/// Embedding vector dimension produced by the default Ollama embedding model.
pub const EMBEDDING_DIM: usize = 1024;

pub type RealEmbedder = DefaultTextEmbedder<OllamaModelProvider>;

/// Starts a Qdrant Docker container and returns `(container, grpc_url)`.
///
/// The gRPC URL is suitable for passing to `QdrantMemoryStore::new` or to
/// `minimal_app_config`. Keep the returned `container` alive for the duration
/// of the test.
pub async fn start_qdrant_container() -> (ContainerAsync<GenericImage>, String) {
    let image = GenericImage::new("qdrant/qdrant", "latest")
        .with_exposed_port(QDRANT_GRPC_PORT.tcp())
        .with_wait_for(WaitFor::http(
            HttpWaitStrategy::new("/healthz")
                .with_port(ContainerPort::Tcp(6333))
                .with_expected_status_code(200u16),
        ));

    let container: ContainerAsync<GenericImage> = image
        .start()
        .await
        .expect("Docker must be available to run E2E tests");

    let host = container.get_host().await.unwrap();
    let port = container
        .get_host_port_ipv4(QDRANT_GRPC_PORT)
        .await
        .unwrap();
    let url = format!("http://{host}:{port}");

    (container, url)
}

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

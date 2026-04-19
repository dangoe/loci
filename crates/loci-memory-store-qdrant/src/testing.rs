// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-memory-store-qdrant.

//! Shared test infrastructure for Qdrant-backed memory store tests.
//!
//! Available only when the `testing` feature is enabled.

use loci_core::embedding::TextEmbedder;
use testcontainers::core::wait::HttpWaitStrategy;
use testcontainers::core::{ContainerPort, IntoContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage};

use crate::config::QdrantConfig;
use crate::store::QdrantMemoryStore;

/// gRPC port used by Qdrant (used to obtain the mapped host port).
pub const QDRANT_GRPC_PORT: u16 = 6334;

/// Starts a Qdrant Docker container and returns `(container, grpc_url)`.
///
/// The gRPC URL is suitable for passing to [`QdrantMemoryStore::new`] or to
/// config builders. Keep the returned `container` alive for the duration
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
        .expect("Docker must be available to run tests");

    let host = container.get_host().await.unwrap();
    let port = container
        .get_host_port_ipv4(QDRANT_GRPC_PORT)
        .await
        .unwrap();
    let url = format!("http://{host}:{port}");

    (container, url)
}

/// Starts a Qdrant container and initialises a [`QdrantMemoryStore`] against it.
///
/// Returns `(store, container)`. Keep `container` alive for the test duration.
pub async fn start_store<E: TextEmbedder>(
    embedder: E,
    similarity_threshold: Option<f64>,
) -> (QdrantMemoryStore<E>, ContainerAsync<GenericImage>) {
    let (container, url) = start_qdrant_container().await;

    let config = if let Some(threshold) = similarity_threshold {
        QdrantConfig::new("memory_entries").with_similarity_threshold(threshold)
    } else {
        QdrantConfig::new("memory_entries")
    };

    let store =
        QdrantMemoryStore::new(&url, config, embedder).expect("failed to create Qdrant client");
    store
        .initialize()
        .await
        .expect("failed to initialize Qdrant collection");

    (store, container)
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-memory-store-qdrant.

use std::{net::SocketAddr, sync::Arc};

use connectrpc::client::{ClientConfig, HttpClient};
use loci_config::AppConfig;
use loci_core::{
    memory::store::MemoryStore, model_provider::text_generation::TextGenerationModelProvider,
};

use crate::{
    infra::{build_llm_provider, build_store},
    loci::{generate::v1::GenerateServiceClient, memory::v1::MemoryServiceClient},
    routes::build_router,
    state::AppState,
};

/// A test server bound to a random local port.
///
/// Starts an axum server on `127.0.0.1:0` with real infrastructure built from
/// the supplied `AppConfig`. Shut down when this value is dropped.
pub struct TestServer {
    /// The local address the server is listening on.
    addr: SocketAddr,
    _shutdown: tokio::sync::oneshot::Sender<()>,
}

impl TestServer {
    /// Returns the local address the server is listening on.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Start the server using the given configuration.
    ///
    /// Panics if the store or provider cannot be initialised.
    pub async fn start(config: AppConfig) -> Self {
        let store = build_store(&config)
            .await
            .expect("TestServer: failed to build store");
        let llm_provider =
            build_llm_provider(&config).expect("TestServer: failed to build llm provider");

        Self::start_with_components(config, Arc::new(store), Arc::new(llm_provider)).await
    }

    /// Start the server using explicit dependencies.
    pub async fn start_with_components<M, E>(
        config: AppConfig,
        store: Arc<M>,
        llm_provider: Arc<E>,
    ) -> Self
    where
        M: MemoryStore + 'static,
        E: TextGenerationModelProvider + 'static,
    {
        let state = Arc::new(AppState::new(store, llm_provider, Arc::new(config)));

        let router = build_router(Arc::clone(&state));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("TestServer: failed to bind listener");
        let addr = listener.local_addr().unwrap();

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    rx.await.ok();
                })
                .await
                .unwrap();
        });

        Self {
            addr,
            _shutdown: tx,
        }
    }

    /// Returns a `MemoryServiceClient` connected to this server.
    pub fn memory_client(&self) -> MemoryServiceClient<HttpClient> {
        let uri = format!("http://{}", self.addr).parse().unwrap();
        MemoryServiceClient::new(HttpClient::plaintext(), ClientConfig::new(uri))
    }

    /// Returns a `GenerateServiceClient` connected to this server.
    pub fn generate_client(&self) -> GenerateServiceClient<HttpClient> {
        let uri = format!("http://{}", self.addr).parse().unwrap();
        GenerateServiceClient::new(HttpClient::plaintext(), ClientConfig::new(uri))
    }
}

/// Builds a minimal [`AppConfig`] with dummy URLs for tests that use mock
/// stores and providers (no real infrastructure needed).
pub fn mock_config() -> AppConfig {
    minimal_app_config(
        "http://unused-qdrant",
        "http://unused-ollama",
        "test-text-model",
        "test-embedding-model",
        384,
    )
}

/// Builds a minimal [`AppConfig`] suitable for tests.
///
/// Points text and embedding models at the given Ollama instance, and the
/// memory store at the given Qdrant gRPC URL.
pub fn minimal_app_config(
    qdrant_url: &str,
    ollama_url: &str,
    text_model: &str,
    embedding_model: &str,
    embedding_dim: usize,
) -> AppConfig {
    let content = format!(
        r#"
[resources.model_providers.ollama]
kind = "ollama"
endpoint = "{ollama_url}"

[resources.models.text.default]
provider = "ollama"
model = "{text_model}"

[resources.models.embedding.default]
provider = "ollama"
model = "{embedding_model}"
dimension = {embedding_dim}

[resources.memory_stores.qdrant]
kind = "qdrant"
url = "{qdrant_url}"
collection = "memory_entries"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "qdrant"

[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "test-classification-model"
"#
    );
    loci_config::load_config_from_str(&content).expect("failed to parse test config")
}

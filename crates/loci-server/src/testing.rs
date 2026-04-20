// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-memory-store-qdrant.

use std::{collections::HashMap, net::SocketAddr, sync::Arc};

use connectrpc::client::{ClientConfig, HttpClient};
use loci_config::{
    AppConfig, EmbeddingModelConfig, EmbeddingRoutingConfig, MemoryConfig, MemoryExtractionConfig,
    MemoryExtractorConfig, MemoryExtractorSearchResultsConfig, MemoryRoutingConfig, MemorySection,
    ModelProviderConfig, ModelProviderKind, ModelsConfig, RoutingConfig, StoreConfig,
    TextModelConfig, TextRoutingConfig,
};
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
    pub addr: SocketAddr,
    _shutdown: tokio::sync::oneshot::Sender<()>,
}

impl TestServer {
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
        let state = Arc::new(AppState {
            store,
            llm_provider,
            config: Arc::new(config),
        });

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
    AppConfig {
        providers: HashMap::from([(
            "ollama".to_string(),
            ModelProviderConfig {
                kind: ModelProviderKind::Ollama,
                endpoint: ollama_url.to_string(),
                api_key: None,
            },
        )]),
        models: ModelsConfig {
            text: HashMap::from([(
                "default".to_string(),
                TextModelConfig {
                    provider: "ollama".to_string(),
                    model: text_model.to_string(),
                    tuning: None,
                },
            )]),
            embedding: HashMap::from([(
                "default".to_string(),
                EmbeddingModelConfig {
                    provider: "ollama".to_string(),
                    model: embedding_model.to_string(),
                    dimension: embedding_dim,
                },
            )]),
        },
        memory: MemorySection {
            backends: HashMap::from([(
                "qdrant".to_string(),
                StoreConfig::Qdrant {
                    url: qdrant_url.to_string(),
                    collection: "memory_entries".to_string(),
                    api_key: None,
                },
            )]),
            config: MemoryConfig {
                backend: "qdrant".to_string(),
                similarity_threshold: None,
            },
            extraction: MemoryExtractionConfig {
                model: "default".to_string(),
                max_entries: None,
                min_confidence: None,
                guidelines: None,
                thinking: None,
                chunking: None,
                extractor: MemoryExtractorConfig {
                    classification_model: "test-classification-model".to_string(),
                    direct_search: MemoryExtractorSearchResultsConfig {
                        max_results: 5,
                        min_score: 0.70,
                    },
                    inverted_search: MemoryExtractorSearchResultsConfig {
                        max_results: 3,
                        min_score: 0.60,
                    },
                    bayesian_seed_weight: 10.0,
                    max_counter_increment: 5.0,
                    max_counter: 100.0,
                    auto_discard_threshold: 0.1,
                    auto_promotion_threshold: 0.9,
                    min_alpha_for_promotion: 12.0,
                    decay_rate: 0.99,
                },
            },
        },
        routing: RoutingConfig {
            text: TextRoutingConfig {
                default: "default".to_string(),
                fallback: vec![],
            },
            embedding: EmbeddingRoutingConfig {
                default: "default".to_string(),
            },
            memory: MemoryRoutingConfig {
                default: "qdrant".to_string(),
            },
        },
    }
}

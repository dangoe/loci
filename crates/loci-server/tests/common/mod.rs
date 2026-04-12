// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-memory-store-qdrant.

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
                promotion_source_threshold: 2,
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

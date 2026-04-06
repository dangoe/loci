use std::collections::HashMap;

use loci_config::{
    AppConfig, EmbeddingModelConfig, MemoryConfig, MemorySection, ModelsConfig,
    ModelProviderConfig, ModelProviderKind, RoutingConfig, EmbeddingRoutingConfig,
    MemoryRoutingConfig, TextModelConfig, TextRoutingConfig, StoreConfig,
};

/// Builds a minimal `AppConfig` wired to a single Ollama provider.
pub fn minimal_ollama_config() -> AppConfig {
    AppConfig {
        providers: HashMap::from([(
            "ollama".to_string(),
            ModelProviderConfig {
                kind: ModelProviderKind::Ollama,
                endpoint: "http://localhost:11434".to_string(),
                api_key: None,
            },
        )]),
        models: ModelsConfig {
            text: HashMap::from([(
                "default".to_string(),
                TextModelConfig {
                    provider: "ollama".to_string(),
                    model: "qwen3:0.6b".to_string(),
                    tuning: None,
                },
            )]),
            embedding: HashMap::from([(
                "default".to_string(),
                EmbeddingModelConfig {
                    provider: "ollama".to_string(),
                    model: "qwen3-embedding:0.6b".to_string(),
                    dimension: 768,
                },
            )]),
        },
        memory: MemorySection {
            backends: HashMap::from([(
                "qdrant".to_string(),
                StoreConfig::Qdrant {
                    url: "http://localhost:6333".to_string(),
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

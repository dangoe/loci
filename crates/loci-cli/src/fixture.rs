use std::collections::HashMap;

use loci_config::{
    AppConfig, EmbeddingProfileConfig, MemoryConfig, ModelConfig, ModelProviderConfig,
    ModelProviderKind, RoutingConfig, StoreConfig,
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
        models: HashMap::from([(
            "default".to_string(),
            ModelConfig {
                provider: "ollama".to_string(),
                name: "qwen3:0.6b".to_string(),
                tuning: None,
            },
        )]),
        embeddings: HashMap::from([(
            "default".to_string(),
            EmbeddingProfileConfig {
                provider: "ollama".to_string(),
                model: "qwen3-embedding:0.6b".to_string(),
                dimension: 768,
            },
        )]),
        stores: HashMap::from([(
            "qdrant".to_string(),
            StoreConfig::Qdrant {
                url: "http://localhost:6333".to_string(),
                api_key: None,
            },
        )]),
        memory: MemoryConfig {
            store: "qdrant".to_string(),
            collection: "memory_entries".to_string(),
            similarity_threshold: None,
            promotion_source_threshold: 2,
        },
        routing: RoutingConfig {
            default_model: "default".to_string(),
            fallback_models: vec![],
            embedding: "default".to_string(),
        },
    }
}

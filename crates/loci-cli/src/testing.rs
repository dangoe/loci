// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

use std::collections::HashMap;
use std::error::Error as StdError;
use std::path::Path;
use std::sync::Arc;

use loci_config::{
    AppConfig, EmbeddingModelConfig, EmbeddingRoutingConfig, MemoryConfig, MemoryExtractionConfig,
    MemoryExtractorConfig, MemoryRoutingConfig, MemorySection, ModelProviderConfig,
    ModelProviderKind, ModelsConfig, RoutingConfig, StoreConfig, TextModelConfig,
    TextRoutingConfig,
};
use loci_core::memory::store::MemoryStore;
use loci_core::model_provider::text_generation::TextGenerationModelProvider;

use crate::commands::config::ConfigCommand;
use crate::commands::generate::{GenerateArgs, GenerateCommand};
use crate::commands::memory::MemoryCommand;
use crate::handlers::CommandHandler;
use crate::handlers::config::ConfigCommandHandler;
use crate::handlers::generate::GenerateCommandHandler;
use crate::handlers::memory::MemoryCommandHandler;

/// A test harness that dispatches CLI commands to handlers with injected
/// dependencies.
///
/// Analogous to `TestServer` in `loci-server`, but runs entirely in-process
/// without network I/O.
pub struct TestCli<S: MemoryStore, T: TextGenerationModelProvider> {
    store: Arc<S>,
    provider: Arc<T>,
    config: AppConfig,
}

impl<S: MemoryStore + 'static, T: TextGenerationModelProvider + 'static> TestCli<S, T> {
    /// Creates a new test CLI with the given mock store and provider.
    pub fn new(store: S, provider: T) -> Self {
        Self {
            store: Arc::new(store),
            provider: Arc::new(provider),
            config: mock_config(),
        }
    }

    /// Overrides the config used by the generate handler.
    pub fn with_config(mut self, config: AppConfig) -> Self {
        self.config = config;
        self
    }

    /// Executes a memory sub-command and returns stdout as a string.
    pub async fn memory(&self, cmd: MemoryCommand) -> Result<String, Box<dyn StdError>> {
        let mut out = Vec::new();
        let text_model = self
            .config
            .models
            .text
            .get(&self.config.routing.text.default)
            .map(|m| m.model.clone())
            .unwrap_or_default();
        let handler = MemoryCommandHandler::new(
            Arc::clone(&self.store),
            Arc::clone(&self.provider),
            text_model,
            self.config.memory.extraction.clone(),
        );
        handler.handle(cmd, &mut out).await?;
        Ok(String::from_utf8(out)?)
    }

    /// Executes the generate command and returns stdout as a string.
    pub async fn generate(&self, args: GenerateArgs) -> Result<String, Box<dyn StdError>> {
        let mut out = Vec::new();
        let handler = GenerateCommandHandler::new(
            Arc::clone(&self.store),
            Arc::clone(&self.provider),
            &self.config,
        );
        handler
            .handle(GenerateCommand::Execute(args), &mut out)
            .await?;
        Ok(String::from_utf8(out)?)
    }

    /// Executes a config sub-command and returns stdout as a string.
    pub async fn config(
        &self,
        path: &Path,
        cmd: ConfigCommand,
    ) -> Result<String, Box<dyn StdError>> {
        let mut out = Vec::new();
        let handler = ConfigCommandHandler::new(path);
        handler.handle(cmd, &mut out).await?;
        Ok(String::from_utf8(out)?)
    }

    /// Returns a reference to the underlying store for snapshot assertions.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Returns a reference to the underlying provider for snapshot assertions.
    pub fn provider(&self) -> &T {
        &self.provider
    }
}

/// Builds a minimal [`AppConfig`] wired to a single Ollama provider.
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
                    direct_search: loci_config::MemoryExtractorSearchResultsConfig {
                        max_results: 5,
                        min_score: 0.70,
                    },
                    inverted_search: loci_config::MemoryExtractorSearchResultsConfig {
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

/// Builds a minimal [`AppConfig`] with dummy URLs for tests that use mock
/// stores and providers (no real infrastructure needed).
pub fn mock_config() -> AppConfig {
    AppConfig {
        providers: HashMap::from([(
            "ollama".to_string(),
            ModelProviderConfig {
                kind: ModelProviderKind::Ollama,
                endpoint: "http://unused-ollama".to_string(),
                api_key: None,
            },
        )]),
        models: ModelsConfig {
            text: HashMap::from([(
                "default".to_string(),
                TextModelConfig {
                    provider: "ollama".to_string(),
                    model: "test-text-model".to_string(),
                    tuning: None,
                },
            )]),
            embedding: HashMap::from([(
                "default".to_string(),
                EmbeddingModelConfig {
                    provider: "ollama".to_string(),
                    model: "test-embedding-model".to_string(),
                    dimension: 384,
                },
            )]),
        },
        memory: MemorySection {
            backends: HashMap::from([(
                "qdrant".to_string(),
                StoreConfig::Qdrant {
                    url: "http://unused-qdrant".to_string(),
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
                    direct_search: loci_config::MemoryExtractorSearchResultsConfig {
                        max_results: 5,
                        min_score: 0.70,
                    },
                    inverted_search: loci_config::MemoryExtractorSearchResultsConfig {
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

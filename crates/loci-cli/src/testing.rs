// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

use std::error::Error as StdError;
use std::path::Path;
use std::sync::Arc;

use loci_config::AppConfig;
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
        let handler = MemoryCommandHandler::new(
            Arc::clone(&self.store),
            Arc::clone(&self.provider),
            self.config.memory().extraction().clone(),
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
    loci_config::load_config_from_str(
        r#"
[resources.model_providers.ollama]
kind = "ollama"
endpoint = "http://localhost:11434"

[resources.models.text.default]
provider = "ollama"
model = "qwen3:0.6b"

[resources.models.embedding.default]
provider = "ollama"
model = "qwen3-embedding:0.6b"
dimension = 768

[resources.memory_stores.qdrant]
kind = "qdrant"
url = "http://localhost:6333"
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
"#,
    )
    .expect("failed to parse minimal_ollama_config")
}

/// Builds a minimal [`AppConfig`] with dummy URLs for tests that use mock
/// stores and providers (no real infrastructure needed).
pub fn mock_config() -> AppConfig {
    loci_config::load_config_from_str(
        r#"
[resources.model_providers.ollama]
kind = "ollama"
endpoint = "http://unused-ollama"

[resources.models.text.default]
provider = "ollama"
model = "test-text-model"

[resources.models.embedding.default]
provider = "ollama"
model = "test-embedding-model"
dimension = 384

[resources.memory_stores.qdrant]
kind = "qdrant"
url = "http://unused-qdrant"
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
"#,
    )
    .expect("failed to parse mock_config")
}

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

pub use loci_wire::testing::{minimal_app_config, minimal_ollama_config, mock_config};

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

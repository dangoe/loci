// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-cli.

use std::{error::Error as StdError, io::Write, path::Path};

use loci_config::init_config;

use crate::{commands::config::ConfigCommand, handlers::CommandHandler};

pub struct ConfigCommandHandler<'a> {
    path: &'a Path,
}

impl<'a> ConfigCommandHandler<'a> {
    pub fn new(path: &'a Path) -> Self {
        Self { path }
    }
}

impl<'a, W: Write + Send> CommandHandler<'a, ConfigCommand, W> for ConfigCommandHandler<'a> {
    async fn handle(
        &self,
        command: ConfigCommand,
        out: &mut W,
    ) -> Result<(), Box<dyn StdError>> {
        match command {
            ConfigCommand::Init => {
                init_config(self.path)?;
                writeln!(out, "Config file written to {}", self.path.display())?;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handlers::CommandHandler;

    #[tokio::test]
    async fn test_config_init_writes_file_to_new_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut out = Vec::new();

        let handler = ConfigCommandHandler::new(&path);
        handler
            .handle(ConfigCommand::Init, &mut out)
            .await
            .unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(
            content.contains("[providers.ollama]"),
            "expected provider section"
        );
        assert!(
            content.contains("[stores.qdrant]"),
            "expected store section"
        );
        assert!(content.contains("[routing]"), "expected routing section");
    }

    #[tokio::test]
    async fn test_config_init_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("sub").join("config.toml");
        let mut out = Vec::new();

        let handler = ConfigCommandHandler::new(&path);
        handler
            .handle(ConfigCommand::Init, &mut out)
            .await
            .unwrap();

        assert!(path.exists(), "config file should have been created");
    }

    #[tokio::test]
    async fn test_config_init_fails_when_file_already_exists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "existing").unwrap();
        let mut out = Vec::new();

        let handler = ConfigCommandHandler::new(&path);
        let err = handler
            .handle(ConfigCommand::Init, &mut out)
            .await
            .unwrap_err();

        assert!(
            err.to_string().contains("already exists"),
            "expected 'already exists' in error, got: {err}"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "existing");
    }

    #[tokio::test]
    async fn test_config_init_output_reports_config_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut out = Vec::new();

        let handler = ConfigCommandHandler::new(&path);
        handler
            .handle(ConfigCommand::Init, &mut out)
            .await
            .unwrap();

        let output = String::from_utf8(out).unwrap();
        assert!(
            output.contains(path.to_str().unwrap()),
            "output should contain the config path, got: {output}"
        );
    }
}

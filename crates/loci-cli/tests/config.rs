// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

#![cfg(feature = "testing")]

mod common;

use std::fs;

use loci_cli::commands::config::ConfigCommand;

use common::{MockStore, MockTextGenerationModelProvider, TestCli};

#[tokio::test]
async fn test_config_init_creates_file() {
    let dir = tempfile::tempdir().expect("should create temp dir");
    let config_path = dir.path().join("loci").join("config.toml");
    let cli = TestCli::new(MockStore::new(), MockTextGenerationModelProvider::ok());

    let output = cli
        .config(&config_path, ConfigCommand::Init)
        .await
        .expect("config init should succeed");

    assert!(config_path.exists(), "config file should be created");
    assert!(
        output.contains(&config_path.display().to_string()),
        "output should report the config path"
    );

    let contents = fs::read_to_string(&config_path).expect("should read config file");
    assert!(
        contents.contains("[providers"),
        "config should contain providers section"
    );
}

#[tokio::test]
async fn test_config_init_existing_file_errors() {
    let dir = tempfile::tempdir().expect("should create temp dir");
    let config_path = dir.path().join("config.toml");
    fs::write(&config_path, "existing").expect("should create file");

    let cli = TestCli::new(MockStore::new(), MockTextGenerationModelProvider::ok());

    let result = cli.config(&config_path, ConfigCommand::Init).await;

    assert!(result.is_err(), "init should fail when file exists");
}

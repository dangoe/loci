// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

#![cfg(feature = "testing")]

mod common;

use loci_cli::commands::generate::{GenerateArgs, GenerateMemoryMode, GenerateSystemMode};

use common::{MockStore, MockTextGenerationModelProvider, TestCli};

fn default_args(prompt: &str) -> GenerateArgs {
    GenerateArgs {
        prompt: prompt.to_string(),
        system: None,
        system_mode: GenerateSystemMode::Append,
        max_memory_entries: 5,
        min_score: 0.5,
        memory_mode: GenerateMemoryMode::Auto,
        filters: vec![],
        debug_flags: vec![],
    }
}

#[tokio::test]
async fn test_generate_streams_response_to_stdout() {
    let cli = TestCli::new(
        MockStore::new().with_query(vec![]),
        MockTextGenerationModelProvider::with_chunks(vec!["hello", " world"]),
    );

    let output = cli
        .generate(default_args("test prompt"))
        .await
        .expect("generate should succeed");

    assert!(output.contains("hello"), "got: {output:?}");
    assert!(output.contains(" world"), "got: {output:?}");
}

#[tokio::test]
async fn test_generate_with_memory_off_does_not_query_store() {
    let store = MockStore::new(); // query defaults to empty — snapshot will show 0 calls
    let provider = MockTextGenerationModelProvider::with_chunks(vec!["ok"]);
    let cli = TestCli::new(store, provider);

    let mut args = default_args("silent prompt");
    args.memory_mode = GenerateMemoryMode::Off;

    let output = cli.generate(args).await.expect("generate should succeed");

    assert!(output.contains("ok"), "got: {output:?}");
    assert_eq!(
        cli.store().snapshot().query_calls,
        0,
        "store should not be queried when memory mode is off"
    );
}

#[tokio::test]
async fn test_generate_with_system_prompt_replace() {
    let cli = TestCli::new(
        MockStore::new().with_query(vec![]),
        MockTextGenerationModelProvider::ok(),
    );

    let mut args = default_args("prompt");
    args.system = Some("custom system".to_string());
    args.system_mode = GenerateSystemMode::Replace;

    let output = cli.generate(args).await.expect("generate should succeed");

    assert!(!output.is_empty(), "should produce output");

    let snapshot = cli.provider().snapshot();
    let req = snapshot
        .last_request
        .expect("provider should capture request");
    let system = req
        .system
        .as_deref()
        .expect("request should have a system prompt");
    assert!(
        system.starts_with("custom system"),
        "replace mode should start with the custom system prompt, got: {system:?}"
    );
}

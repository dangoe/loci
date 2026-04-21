// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-e2e-tests.

#![cfg(feature = "e2e")]

mod support;

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use futures::StreamExt as _;
use loci_core::contextualization::{
    ContextualizationMemoryMode, Contextualizer, ContextualizerConfig, ContextualizerTuningConfig,
};
use loci_core::memory::store::{MemoryInput, MemoryQuery, MemoryQueryMode, MemoryStore};
use loci_core::memory::{MemoryTrust, Score};
use loci_model_provider_ollama::testing::{ensure_ollama_available, ollama_provider, text_model};

use support::{create_embedder, start_qdrant_store};

fn input(content: &str) -> MemoryInput {
    MemoryInput::new(content.to_string(), MemoryTrust::Fact, HashMap::new())
}

#[tokio::test]
async fn test_add_entry_and_query_with_real_embeddings() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let embedder = create_embedder(Arc::clone(&provider));
    let (store, _container) = start_qdrant_store(embedder, None).await;

    // Save a memory
    store
        .add_entry(&input("The user's favorite color is blue"))
        .await
        .expect("save should succeed");

    // Query with semantically similar text
    let results = store
        .query(
            MemoryQuery::new(
                "what is the user's preferred color".to_string(),
                MemoryQueryMode::Lookup,
            )
            .with_max_results(NonZeroUsize::new(5).unwrap()),
        )
        .await
        .expect("query should succeed");

    assert!(
        !results.is_empty(),
        "query should return at least one result"
    );
    assert!(
        results[0].content().contains("favorite color is blue"),
        "top result should be the saved memory, got: {:?}",
        results[0].content()
    );
}

#[tokio::test]
async fn test_contextualizer_injects_relevant_memory() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let embedder = create_embedder(Arc::clone(&provider));
    let (store, _container) = start_qdrant_store(embedder, None).await;

    // Save a memory the model should reference
    store
        .add_entry(&input("The user's name is Alice"))
        .await
        .expect("save should succeed");

    let config = build_base_contextualizer_config();

    let ctx = Contextualizer::new(Arc::new(store), provider, config);
    let stream = ctx
        .contextualize("What is my name?")
        .await
        .expect("contextualize should succeed");

    let chunks: Vec<_> = stream.collect().await;
    let full_text: String = chunks
        .into_iter()
        .map(|c| c.expect("chunk should not be an error").text().to_string())
        .collect();

    let lower = full_text.to_lowercase();
    assert!(
        lower.contains("alice"),
        "response should mention 'Alice' from memory, got: {full_text:?}"
    );
}

#[tokio::test]
async fn test_contextualizer_with_no_relevant_memory() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let embedder = create_embedder(Arc::clone(&provider));
    let (store, _container) = start_qdrant_store(embedder, None).await;

    // Empty store — no memories saved

    let config = build_base_contextualizer_config();

    let ctx = Contextualizer::new(Arc::new(store), provider, config);

    let (debug_info, stream) = ctx
        .contextualize_with_debug("What is my favorite food?")
        .await
        .expect("contextualize_with_debug should succeed");

    // With an empty store, no memory entries should be injected
    assert!(
        debug_info.memory_entries().is_empty(),
        "debug info should show no memory entries for an empty store"
    );

    // The stream should still complete successfully
    let chunks: Vec<_> = stream.collect().await;
    assert!(
        !chunks.is_empty(),
        "should receive at least one response chunk"
    );
    for chunk in &chunks {
        assert!(chunk.is_ok(), "all chunks should succeed");
    }
}

#[tokio::test]
async fn test_deduplication_with_real_embeddings() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let embedder = create_embedder(Arc::clone(&provider));
    // Enable deduplication with a high threshold
    let (store, _container) = start_qdrant_store(embedder, Some(0.9)).await;

    let first = store
        .add_entry(&input("Paris is the capital of France"))
        .await
        .expect("first save should succeed");

    let second = store
        .add_entry(&input("Paris is the capital of France"))
        .await
        .expect("second save should succeed");

    // With deduplication enabled at 0.9 threshold, identical content should return the same ID
    assert_eq!(
        first.id(),
        second.id(),
        "identical content should be deduplicated (same ID returned)"
    );
}

fn build_base_contextualizer_config() -> ContextualizerConfig {
    ContextualizerConfig::new(
        text_model(),
        None,
        ContextualizationMemoryMode::Auto,
        NonZeroUsize::new(5).unwrap(),
        Score::ZERO,
        HashMap::new(),
        Some(ContextualizerTuningConfig::new(
            Some(0.0),
            Some(100),
            None,
            None,
            None,
            None,
            None,
            None,
            HashMap::new(),
        )),
    )
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-e2e-tests.

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt as _;
use loci_core::contextualization::{
    ContextualizationMemoryMode, Contextualizer, ContextualizerConfig, ContextualizerTuningConfig,
};
use loci_core::embedding::DefaultTextEmbedder;
use loci_core::memory::{MemoryInput, MemoryQuery, MemoryQueryMode, Score};
use loci_core::store::MemoryStore;
use loci_memory_store_qdrant::config::QdrantConfig;
use loci_memory_store_qdrant::store::QdrantMemoryStore;
use loci_model_provider_ollama::provider::OllamaModelProvider;
use loci_model_provider_ollama::testing::{
    embedding_model, ensure_ollama_available, ollama_provider, text_model,
};
use testcontainers::core::wait::HttpWaitStrategy;
use testcontainers::core::{ContainerPort, IntoContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage};

const QDRANT_PORT: u16 = 6334;
const EMBEDDING_DIM: usize = 1024;

type RealEmbedder = DefaultTextEmbedder<OllamaModelProvider>;

async fn start_qdrant_store(
    embedder: RealEmbedder,
    similarity_threshold: Option<f64>,
) -> (
    QdrantMemoryStore<RealEmbedder>,
    ContainerAsync<GenericImage>,
) {
    let image = GenericImage::new("qdrant/qdrant", "latest")
        .with_exposed_port(QDRANT_PORT.tcp())
        .with_wait_for(WaitFor::http(
            HttpWaitStrategy::new("/healthz")
                .with_port(ContainerPort::Tcp(6333))
                .with_expected_status_code(200u16),
        ));

    let container: ContainerAsync<GenericImage> = image
        .start()
        .await
        .expect("Docker must be available to run E2E tests");

    let host = container.get_host().await.unwrap();
    let port = container.get_host_port_ipv4(QDRANT_PORT).await.unwrap();
    let url = format!("http://{host}:{port}");

    let config = QdrantConfig {
        collection_name: "memory_entries".to_string(),
        similarity_threshold,
        promotion_source_threshold: 2,
    };

    let store =
        QdrantMemoryStore::new(&url, config, embedder).expect("failed to create Qdrant client");
    store
        .initialize()
        .await
        .expect("failed to initialize Qdrant collection");

    (store, container)
}

fn create_embedder(provider: Arc<OllamaModelProvider>) -> RealEmbedder {
    DefaultTextEmbedder::new(provider, embedding_model(), EMBEDDING_DIM)
}

fn input(content: &str) -> MemoryInput {
    MemoryInput::new(content.to_string(), HashMap::new())
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_save_and_query_with_real_embeddings() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let embedder = create_embedder(Arc::clone(&provider));
    let (store, _container) = start_qdrant_store(embedder, None).await;

    // Save a memory
    store
        .add_entry(input("The user's favorite color is blue"))
        .await
        .expect("save should succeed");

    // Query with semantically similar text
    let results = store
        .query(MemoryQuery {
            topic: "what is the user's preferred color".to_string(),
            max_results: 5,
            min_score: Score::ZERO,
            filters: HashMap::new(),
            mode: MemoryQueryMode::Lookup,
        })
        .await
        .expect("query should succeed");

    assert!(
        !results.is_empty(),
        "query should return at least one result"
    );
    assert!(
        results[0]
            .memory_entry
            .content
            .contains("favorite color is blue"),
        "top result should be the saved memory, got: {:?}",
        results[0].memory_entry.content
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_contextualizer_injects_relevant_memory() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let embedder = create_embedder(Arc::clone(&provider));
    let (store, _container) = start_qdrant_store(embedder, None).await;

    // Save a memory the model should reference
    store
        .add_entry(input("The user's name is Alice"))
        .await
        .expect("save should succeed");

    let config = ContextualizerConfig {
        text_generation_model: text_model(),
        system: None,
        memory_mode: ContextualizationMemoryMode::Auto,
        max_memory_entries: 5,
        min_score: Score::ZERO,
        filters: HashMap::new(),
        tuning: Some(ContextualizerTuningConfig {
            temperature: Some(0.0),
            max_tokens: Some(100),
            ..Default::default()
        }),
    };

    let ctx = Contextualizer::new(&store, provider.as_ref(), config);
    let stream = ctx
        .contextualize("What is my name?")
        .await
        .expect("contextualize should succeed");

    let chunks: Vec<_> = stream.collect().await;
    let full_text: String = chunks
        .into_iter()
        .map(|c| c.expect("chunk should not be an error").text)
        .collect();

    let lower = full_text.to_lowercase();
    assert!(
        lower.contains("alice"),
        "response should mention 'Alice' from memory, got: {full_text:?}"
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_contextualizer_with_no_relevant_memory() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let embedder = create_embedder(Arc::clone(&provider));
    let (store, _container) = start_qdrant_store(embedder, None).await;

    // Empty store — no memories saved

    let config = ContextualizerConfig {
        text_generation_model: text_model(),
        system: None,
        memory_mode: ContextualizationMemoryMode::Auto,
        max_memory_entries: 5,
        min_score: Score::ZERO,
        filters: HashMap::new(),
        tuning: Some(ContextualizerTuningConfig {
            temperature: Some(0.0),
            max_tokens: Some(100),
            ..Default::default()
        }),
    };

    let ctx = Contextualizer::new(&store, provider.as_ref(), config);

    let (debug_info, stream) = ctx
        .contextualize_with_debug("What is my favorite food?")
        .await
        .expect("contextualize_with_debug should succeed");

    // With an empty store, no memory entries should be injected
    assert!(
        debug_info.memory_entries.is_empty(),
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
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_deduplication_with_real_embeddings() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let embedder = create_embedder(Arc::clone(&provider));
    // Enable deduplication with a high threshold
    let (store, _container) = start_qdrant_store(embedder, Some(0.9)).await;

    let first = store
        .add_entry(input("Paris is the capital of France"))
        .await
        .expect("first save should succeed");

    let second = store
        .add_entry(input("Paris is the capital of France"))
        .await
        .expect("second save should succeed");

    // With deduplication enabled at 0.9 threshold, identical content should return the same ID
    assert_eq!(
        first.memory_entry.id, second.memory_entry.id,
        "identical content should be deduplicated (same ID returned)"
    );
}

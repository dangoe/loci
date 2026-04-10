// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-model-provider-ollama.

use futures::StreamExt as _;
use loci_core::model_provider::{
    embedding::{EmbeddingModelProvider, EmbeddingRequest},
    text_generation::{TextGenerationModelProvider, TextGenerationRequest},
};
use loci_model_provider_ollama::testing::{
    embedding_model, ensure_ollama_available, ollama_provider, text_model,
};

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_embed_returns_vectors_with_correct_count() {
    ensure_ollama_available().await;

    let provider = ollama_provider();
    let req = EmbeddingRequest::new_batch(
        embedding_model(),
        vec![
            "first text".to_string(),
            "second text".to_string(),
            "third text".to_string(),
        ],
    );

    let resp = provider.embed(req).await.expect("embed should succeed");

    assert_eq!(resp.embeddings.len(), 3, "expected one vector per input");
    for (i, vec) in resp.embeddings.iter().enumerate() {
        assert!(!vec.is_empty(), "vector {i} should not be empty");
    }
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_embed_similar_texts_have_high_cosine_similarity() {
    ensure_ollama_available().await;

    let provider = ollama_provider();

    let req = EmbeddingRequest::new_batch(
        embedding_model(),
        vec![
            "the cat sat on the mat".to_string(),
            "a cat was sitting on the mat".to_string(),
        ],
    );

    let resp = provider.embed(req).await.expect("embed should succeed");
    let sim = cosine_similarity(&resp.embeddings[0], &resp.embeddings[1]);

    assert!(
        sim > 0.7,
        "similar texts should have cosine similarity > 0.7, got {sim}"
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_embed_dissimilar_texts_have_low_cosine_similarity() {
    ensure_ollama_available().await;

    let provider = ollama_provider();

    let req = EmbeddingRequest::new_batch(
        embedding_model(),
        vec![
            "quantum physics equations and particle accelerators".to_string(),
            "chocolate cake recipe with vanilla frosting".to_string(),
        ],
    );

    let resp = provider.embed(req).await.expect("embed should succeed");
    let sim = cosine_similarity(&resp.embeddings[0], &resp.embeddings[1]);

    assert!(
        sim < 0.5,
        "dissimilar texts should have cosine similarity < 0.5, got {sim}"
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_embed_nonexistent_model_returns_error() {
    ensure_ollama_available().await;

    let provider = ollama_provider();
    let req = EmbeddingRequest::new("nonexistent-model-xyz-99999", "hello");

    let result = provider.embed(req).await;

    assert!(result.is_err(), "nonexistent model should return an error");
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_generate_returns_nonempty_response() {
    ensure_ollama_available().await;

    let provider = ollama_provider();
    let mut req =
        TextGenerationRequest::new(text_model(), "What is 2+2? Answer with just the number.");
    req.temperature = Some(0.0);
    req.max_tokens = Some(50);

    let resp = provider
        .generate(req)
        .await
        .expect("generate should succeed");

    assert!(!resp.text.is_empty(), "response text should not be empty");
    assert!(resp.done, "response should be marked as done");
    assert!(resp.usage.is_some(), "usage stats should be present");
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_generate_stream_yields_chunks_ending_with_done() {
    ensure_ollama_available().await;

    let provider = ollama_provider();
    let mut req = TextGenerationRequest::new(text_model(), "Count from 1 to 5.");
    req.temperature = Some(0.0);
    req.max_tokens = Some(100);

    let stream = provider.generate_stream(req);
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream should not error");

    assert!(
        chunks.len() >= 2,
        "stream should yield at least 2 chunks, got {}",
        chunks.len()
    );

    let last = chunks.last().expect("should have at least one chunk");
    assert!(last.done, "last chunk should have done=true");

    let full_text: String = chunks.iter().map(|c| c.text.as_str()).collect();
    assert!(
        !full_text.is_empty(),
        "concatenated text should not be empty"
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_generate_with_system_prompt() {
    ensure_ollama_available().await;

    let provider = ollama_provider();
    let mut req = TextGenerationRequest::new(text_model(), "What color is the sky?");
    req.system = Some("You must reply in exactly one word. No punctuation.".to_string());
    req.temperature = Some(0.0);
    req.max_tokens = Some(20);

    let resp = provider
        .generate(req)
        .await
        .expect("generate should succeed");
    let trimmed = resp.text.trim();

    assert!(
        trimmed.len() < 30,
        "response should be very short when instructed to reply in one word, got: {trimmed:?}"
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_generate_respects_temperature_zero() {
    ensure_ollama_available().await;

    let provider = ollama_provider();

    let make_req = || {
        let mut req = TextGenerationRequest::new(text_model(), "What is the capital of France?");
        req.temperature = Some(0.0);
        req.max_tokens = Some(50);
        req
    };

    let resp1 = provider
        .generate(make_req())
        .await
        .expect("first generate should succeed");
    let resp2 = provider
        .generate(make_req())
        .await
        .expect("second generate should succeed");

    assert_eq!(
        resp1.text, resp2.text,
        "temperature=0.0 should produce deterministic output"
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_generate_nonexistent_model_returns_error() {
    ensure_ollama_available().await;

    let provider = ollama_provider();
    let req = TextGenerationRequest::new("nonexistent-model-xyz-99999", "hello");

    let result = provider.generate(req).await;

    assert!(result.is_err(), "nonexistent model should return an error");
}

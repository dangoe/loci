// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-model-provider-ollama.

//! Test utilities for the Ollama model provider.
//!
//! Available only when the `e2e` feature is enabled.

use crate::provider::{OllamaConfig, OllamaModelProvider};

/// Returns the base URL for the Ollama instance.
///
/// Reads from the `OLLAMA_BASE_URL` environment variable, defaulting to
/// `http://localhost:11434`.
pub fn base_url() -> String {
    std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

/// Returns the text generation model name.
///
/// Reads from the `OLLAMA_TEXT_MODEL` environment variable, defaulting to
/// `qwen3:0.6b`.
pub fn text_model() -> String {
    std::env::var("OLLAMA_TEXT_MODEL").unwrap_or_else(|_| "qwen3:0.6b".to_string())
}

/// Returns the classification model name.
///
/// Reads from the `OLLAMA_CLASSIFICATION_MODEL` environment variable, defaulting to
/// `qwen3:0.6b`.
pub fn classification_model() -> String {
    std::env::var("OLLAMA_CLASSIFICATION_MODEL").unwrap_or_else(|_| "qwen3:0.6b".to_string())
}

/// Returns the embedding model name.
///
/// Reads from the `OLLAMA_EMBEDDING_MODEL` environment variable, defaulting to
/// `qwen3-embedding:0.6b`.
pub fn embedding_model() -> String {
    std::env::var("OLLAMA_EMBEDDING_MODEL").unwrap_or_else(|_| "qwen3-embedding:0.6b".to_string())
}

/// Creates an [`OllamaModelProvider`] configured from environment variables.
pub fn ollama_provider() -> OllamaModelProvider {
    OllamaModelProvider::new(OllamaConfig {
        base_url: base_url(),
        timeout: None,
    })
    .expect("Failed to create OllamaModelProvider")
}

/// Panics with a helpful message if Ollama is not reachable.
///
/// Pings the `/api/tags` endpoint. Call this at the start of each E2E test
/// so failures are immediately obvious rather than surfacing as opaque
/// connection errors.
pub async fn ensure_ollama_available() {
    let url = format!("{}/api/tags", base_url());
    let client = reqwest::Client::new();
    let resp = client.get(&url).send().await;
    match resp {
        Ok(r) if r.status().is_success() => {}
        Ok(r) => panic!(
            "Ollama returned status {} at {url}. Is Ollama running?",
            r.status()
        ),
        Err(e) => panic!(
            "Cannot reach Ollama at {url}: {e}\n\
             Ensure Ollama is running: `ollama serve`"
        ),
    }
}

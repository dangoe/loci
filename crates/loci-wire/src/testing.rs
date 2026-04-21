// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-wire.

use loci_config::AppConfig;

/// Builds a minimal [`AppConfig`] with dummy URLs for tests that use mock
/// stores and providers (no real infrastructure needed).
pub fn mock_config() -> AppConfig {
    minimal_app_config(
        "http://unused-qdrant",
        "http://unused-ollama",
        "test-text-model",
        "test-embedding-model",
        384,
    )
}

/// Builds a minimal [`AppConfig`] wired to a single Ollama provider.
pub fn minimal_ollama_config() -> AppConfig {
    minimal_app_config(
        "http://localhost:6333",
        "http://localhost:11434",
        "qwen3:0.6b",
        "qwen3-embedding:0.6b",
        768,
    )
}

/// Builds a minimal [`AppConfig`] suitable for tests with parameterized URLs and models.
pub fn minimal_app_config(
    qdrant_url: &str,
    ollama_url: &str,
    text_model: &str,
    embedding_model: &str,
    embedding_dim: usize,
) -> AppConfig {
    let content = format!(
        r#"
[resources.model_providers.ollama]
kind = "ollama"
endpoint = "{ollama_url}"

[resources.models.text.default]
provider = "ollama"
model = "{text_model}"

[resources.models.embedding.default]
provider = "ollama"
model = "{embedding_model}"
dimension = {embedding_dim}

[resources.memory_stores.qdrant]
kind = "qdrant"
url = "{qdrant_url}"
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
"#
    );
    loci_config::load_config_from_str(&content).expect("failed to parse test config")
}

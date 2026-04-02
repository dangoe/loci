use std::future::Future;

use serde::{Deserialize, Serialize};

use ai_memory_core::{Embedding, EmbeddingError, TextEmbedder};

const DEFAULT_MODEL: &str = "nomic-embed-text";
/// Embedding dimension produced by `nomic-embed-text`.
const DEFAULT_DIMENSION: usize = 768;

/// [`TextEmbedder`] implementation that calls the Ollama `/api/embed` endpoint.
///
/// Defaults to the `nomic-embed-text` model (768 dimensions). Use
/// [`OllamaTextEmbedder::with_model`] to override both the model and its dimension.
pub struct OllamaTextEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dimension: usize,
}

impl OllamaTextEmbedder {
    /// Creates an embedder pointing at `base_url` (e.g. `"http://localhost:11434"`)
    /// using the default `nomic-embed-text` model.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            model: DEFAULT_MODEL.to_string(),
            dimension: DEFAULT_DIMENSION,
        }
    }

    /// Overrides the model name and its embedding dimension.
    pub fn with_model(mut self, model: impl Into<String>, dimension: usize) -> Self {
        self.model = model.into();
        self.dimension = dimension;
        self
    }
}

impl TextEmbedder for OllamaTextEmbedder {
    fn embedding_dimension(&self) -> usize {
        self.dimension
    }

    fn embed(&self, text: &str) -> impl Future<Output = Result<Embedding, EmbeddingError>> + Send + '_ {
        let text = text.to_owned();
        async move {
            log::debug!("embedding {} chars via Ollama ({})", text.len(), self.model);

            let response = self
                .client
                .post(format!("{}/api/embed", self.base_url))
                .json(&EmbedRequest { model: &self.model, input: &text })
                .send()
                .await
                .map_err(|e| EmbeddingError::Http(e.to_string()))?;

            if !response.status().is_success() {
                return Err(EmbeddingError::Http(format!(
                    "Ollama returned HTTP {}",
                    response.status()
                )));
            }

            let body: EmbedResponse = response
                .json()
                .await
                .map_err(|e| EmbeddingError::Parse(e.to_string()))?;

            let raw = body
                .embeddings
                .into_iter()
                .next()
                .ok_or_else(|| EmbeddingError::Parse("embeddings array is empty".to_string()))?;

            let values = raw
                .into_iter()
                .map(|v| v as f32)
                .collect();

            Ok(Embedding::new(values))
        }
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f64>>,
}

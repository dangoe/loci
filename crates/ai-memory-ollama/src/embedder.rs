use std::future::Future;

use serde::{Deserialize, Serialize};

use ai_memory_core::{
    Embedding, EmbeddingError, RemoteModelClient, RemoteModelError, TextEmbedder,
};

use crate::OllamaRemoteModelClient;

const DEFAULT_MODEL: &str = "nomic-embed-text";
/// Embedding dimension produced by `nomic-embed-text`.
const DEFAULT_DIMENSION: usize = 768;

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f64>>,
}

/// [`TextEmbedder`] implementation that calls the Ollama `/api/embed` endpoint.
///
/// Defaults to the `nomic-embed-text` model (768 dimensions). Use
/// [`OllamaTextEmbedder::with_model`] to override both the model and its dimension.
pub struct OllamaTextEmbedder {
    client: OllamaRemoteModelClient,
    dimension: usize,
}

impl OllamaTextEmbedder {
    /// Creates an embedder pointing at `base_url` (e.g. `"http://localhost:11434"`)
    /// using the default `nomic-embed-text` model.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: OllamaRemoteModelClient::new(base_url, model),
            dimension: DEFAULT_DIMENSION,
        }
    }
}

impl TextEmbedder for OllamaTextEmbedder {
    fn embedding_dimension(&self) -> usize {
        self.dimension
    }

    fn embed(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<Embedding, EmbeddingError>> + Send + '_ {
        let request = OllamaRequest::Embed {
            model: self.model.clone(),
            input: text.to_owned(),
        };

        async move {
            let response = self
                .client
                .send(request)
                .await
                .map_err(|e| EmbeddingError::TargetModel(e))?;

            if !response.status().is_success() {
                return Err(EmbeddingError::TargetModel(RemoteModelError::Http(
                    format!("Ollama returned HTTP {}", response.status()),
                )));
            }

            let body: EmbedResponse = response
                .json()
                .await
                .map_err(|e| EmbeddingError::TargetModel(RemoteModelError::Parse(e.to_string())))?;

            let raw = body.embeddings.into_iter().next().ok_or_else(|| {
                EmbeddingError::TargetModel(RemoteModelError::Parse(
                    "embeddings array is empty".to_string(),
                ))
            })?;

            let values = raw.into_iter().map(|v| v as f32).collect();

            Ok(Embedding::new(values))
        }
    }
}

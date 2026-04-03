use std::{pin::Pin, sync::Arc};

use crate::{
    backend::embedding::{EmbeddingBackend, EmbeddingRequest},
    error::EmbeddingError,
};

/// An embedding vector represented as a sequence of `f32` values.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    /// Creates an `Embedding` from the given values.
    pub fn new(values: Vec<f32>) -> Self {
        Self(values)
    }

    /// Returns the raw embedding values.
    pub fn values(&self) -> &[f32] {
        &self.0
    }

    /// Returns the number of dimensions in this embedding.
    pub fn dimension(&self) -> usize {
        self.0.len()
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(values: Vec<f32>) -> Self {
        Self::new(values)
    }
}

/// Computes embedding vectors for text input.
pub trait TextEmbedder: Send + Sync {
    /// Returns the fixed number of dimensions this embedder produces.
    fn embedding_dimension(&self) -> usize;

    fn embed(
        &self,
        text: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Embedding, EmbeddingError>> + Send + '_>>;
}

/// Default [`TextEmbedder`] implementation backed by any [`EmbeddingBackend`].
pub struct DefaultTextEmbedder {
    backend: Arc<dyn EmbeddingBackend>,
    model: String,
    embedding_dimension: usize,
}

impl DefaultTextEmbedder {
    /// Creates a new `DefaultTextEmbedder`.
    pub fn new(
        backend: Arc<dyn EmbeddingBackend>,
        model: impl Into<String>,
        embedding_dimension: usize,
    ) -> Self {
        Self {
            backend,
            model: model.into(),
            embedding_dimension,
        }
    }
}

impl TextEmbedder for DefaultTextEmbedder {
    fn embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }

    fn embed(
        &self,
        text: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Embedding, EmbeddingError>> + Send + '_>> {
        let req = EmbeddingRequest::new(self.model.as_str(), text)
            .with_embedding_dimension(self.embedding_dimension);
        Box::pin(async move {
            self.backend
                .embed(req)
                .await
                .map_err(EmbeddingError::TargetModel)?
                .embeddings
                .into_iter()
                .next()
                .map(Embedding::from)
                .ok_or(EmbeddingError::EmptyResponse)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension() {
        assert_eq!(Embedding::new(vec![1.0, 2.0, 3.0]).dimension(), 3);
    }

    #[test]
    fn test_values() {
        let values = vec![0.1_f32, 0.2, 0.3];
        assert_eq!(Embedding::new(values.clone()).values(), values.as_slice());
    }

    #[test]
    fn test_from_vec() {
        let e: Embedding = vec![1.0_f32, 2.0].into();
        assert_eq!(e.dimension(), 2);
    }
}

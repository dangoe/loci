// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-core.

use std::sync::Arc;

use crate::{
    error::EmbeddingError,
    model_provider::embedding::{EmbeddingModelProvider, EmbeddingRequest},
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
    ) -> impl Future<Output = Result<Embedding, EmbeddingError>> + Send + '_;
}

/// Default [`TextEmbedder`] implementation backed by any [`EmbeddingModelProvider`].
///
/// Delegates to the provider's [`EmbeddingModelProvider::embed`] and extracts
/// the first embedding vector from the batch response.
pub struct DefaultTextEmbedder<P: EmbeddingModelProvider> {
    provider: Arc<P>,
    model: String,
    embedding_dimension: usize,
}

impl<P: EmbeddingModelProvider> DefaultTextEmbedder<P> {
    /// Creates a new `DefaultTextEmbedder`.
    pub fn new(provider: Arc<P>, model: impl Into<String>, embedding_dimension: usize) -> Self {
        Self {
            provider,
            model: model.into(),
            embedding_dimension,
        }
    }
}

impl<P: EmbeddingModelProvider> TextEmbedder for DefaultTextEmbedder<P> {
    fn embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }

    fn embed(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<Embedding, EmbeddingError>> + Send + '_ {
        let req = EmbeddingRequest::new(self.model.as_str(), text)
            .with_embedding_dimension(self.embedding_dimension);
        async move {
            self.provider
                .embed(req)
                .await
                .map_err(EmbeddingError::ModelProvider)?
                .embeddings
                .into_iter()
                .next()
                .map(Embedding::from)
                .ok_or(EmbeddingError::EmptyResponse)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use crate::model_provider::{
        common::ModelProviderResult,
        embedding::{EmbeddingModelProvider, EmbeddingRequest, EmbeddingResponse},
        error::ModelProviderError,
    };

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

    #[test]
    fn test_empty_embedding_has_zero_dimension() {
        assert_eq!(Embedding::new(vec![]).dimension(), 0);
    }

    struct EmptyResponseProvider;

    impl EmbeddingModelProvider for EmptyResponseProvider {
        fn embed(
            &self,
            req: EmbeddingRequest,
        ) -> impl Future<Output = ModelProviderResult<EmbeddingResponse>> + Send + '_ {
            async move {
                Ok(EmbeddingResponse {
                    embeddings: vec![], // empty — should trigger EmptyResponse error
                    model: req.model.clone(),
                    usage: None,
                })
            }
        }
    }

    struct ErrorProvider;

    impl EmbeddingModelProvider for ErrorProvider {
        fn embed(
            &self,
            _req: EmbeddingRequest,
        ) -> impl Future<Output = ModelProviderResult<EmbeddingResponse>> + Send + '_ {
            async move { Err(ModelProviderError::Timeout) }
        }
    }

    struct FixedProvider {
        values: Vec<f32>,
    }

    impl EmbeddingModelProvider for FixedProvider {
        fn embed(
            &self,
            req: EmbeddingRequest,
        ) -> impl Future<Output = ModelProviderResult<EmbeddingResponse>> + Send + '_ {
            let values = self.values.clone();
            async move {
                Ok(EmbeddingResponse {
                    embeddings: vec![values],
                    model: req.model.clone(),
                    usage: None,
                })
            }
        }
    }

    #[tokio::test]
    async fn test_default_text_embedder_returns_embedding_from_provider() {
        let provider = Arc::new(FixedProvider {
            values: vec![0.1, 0.2, 0.3],
        });
        let embedder = DefaultTextEmbedder::new(provider, "test-model", 3);
        let result = embedder.embed("hello").await.unwrap();
        assert_eq!(result.values(), &[0.1_f32, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn test_default_text_embedder_returns_empty_response_error_when_no_vectors() {
        let provider = Arc::new(EmptyResponseProvider);
        let embedder = DefaultTextEmbedder::new(provider, "test-model", 3);
        let result = embedder.embed("hello").await;
        assert!(matches!(result, Err(EmbeddingError::EmptyResponse)));
    }

    #[tokio::test]
    async fn test_default_text_embedder_propagates_provider_error() {
        let provider = Arc::new(ErrorProvider);
        let embedder = DefaultTextEmbedder::new(provider, "test-model", 3);
        let result = embedder.embed("hello").await;
        assert!(matches!(result, Err(EmbeddingError::ModelProvider(_))));
    }

    #[test]
    fn test_default_text_embedder_reports_configured_dimension() {
        let provider = Arc::new(FixedProvider { values: vec![] });
        let embedder = DefaultTextEmbedder::new(provider, "model", 768);
        assert_eq!(embedder.embedding_dimension(), 768);
    }
}

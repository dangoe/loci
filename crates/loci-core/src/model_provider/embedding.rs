// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::collections::HashMap;
use std::future::Future;
use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

use crate::model_provider::common::{ModelProviderParams, ModelProviderResult, TokenUsage};

/// A text-embedding request.
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
    embedding_dimension: Option<NonZeroUsize>,
    extra_params: ModelProviderParams,
}

impl EmbeddingRequest {
    /// Creates a single-input embedding request.
    pub fn new(model: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: vec![input.into()],
            embedding_dimension: None,
            extra_params: HashMap::new(),
        }
    }

    /// Creates a batch embedding request.
    pub fn new_batch(model: impl Into<String>, inputs: Vec<String>) -> Self {
        Self {
            model: model.into(),
            input: inputs,
            embedding_dimension: None,
            extra_params: HashMap::new(),
        }
    }

    /// Returns the model identifier.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the input texts.
    pub fn input(&self) -> &[String] {
        &self.input
    }

    /// Returns the desired output embedding dimension, if set.
    pub fn embedding_dimension(&self) -> Option<NonZeroUsize> {
        self.embedding_dimension
    }

    /// Returns the extra model-provider-specific parameters.
    pub fn extra_params(&self) -> &ModelProviderParams {
        &self.extra_params
    }

    /// Sets the desired output embedding dimension.
    pub fn with_embedding_dimension(mut self, dim: NonZeroUsize) -> Self {
        self.embedding_dimension = Some(dim);
        self
    }

    /// Adds a model-provider-specific extra parameter.
    pub fn with_extra(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}

/// One embedding vector per input text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Parallel to `EmbeddingRequest::input` — one vector per input string.
    embeddings: Vec<Vec<f32>>,

    /// Embedding model name echoed back by the model provider.
    model: String,

    /// Token usage if reported.
    usage: Option<TokenUsage>,
}

impl EmbeddingResponse {
    /// Creates a new `EmbeddingResponse`.
    pub fn new(embeddings: Vec<Vec<f32>>, model: String, usage: Option<TokenUsage>) -> Self {
        Self {
            embeddings,
            model,
            usage,
        }
    }

    /// Returns the embedding vectors, one per input string.
    pub fn embeddings(&self) -> &[Vec<f32>] {
        &self.embeddings
    }

    /// Returns the model name echoed back by the provider.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns token usage, if reported.
    pub fn usage(&self) -> Option<&TokenUsage> {
        self.usage.as_ref()
    }
}

/// Low-level interface to a model provider capable of producing embedding vectors.
///
/// Implementations communicate with a specific inference service (e.g. Ollama,
/// OpenAI) and translate [`EmbeddingRequest`] into the provider's wire format.
/// Higher-level code should use [`crate::embedding::TextEmbedder`] instead.
///
/// # Object Safety
/// This trait uses return-position `impl Trait` (RPITIT) for ergonomics and
/// zero-cost dispatch in generic contexts. It is therefore **not object-safe**
/// and cannot be used as `dyn EmbeddingModelProvider`. If dynamic dispatch is
/// needed in future, convert to `BoxFuture` like `MemoryStore`.
pub trait EmbeddingModelProvider: Send + Sync {
    fn embed(
        &self,
        req: EmbeddingRequest,
    ) -> impl Future<Output = ModelProviderResult<EmbeddingResponse>> + Send + '_;
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    #[test]
    fn test_new_sets_model_and_single_input() {
        let req = EmbeddingRequest::new("my-model", "hello world");
        assert_eq!(req.model(), "my-model");
        assert_eq!(req.input(), &["hello world".to_string()]);
        assert!(req.embedding_dimension().is_none());
        assert!(req.extra_params().is_empty());
    }

    #[test]
    fn test_new_batch_sets_all_inputs() {
        let inputs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let req = EmbeddingRequest::new_batch("model", inputs.clone());
        assert_eq!(req.input(), inputs.as_slice());
    }

    #[test]
    fn test_with_embedding_dimension_sets_field() {
        let dim = NonZeroUsize::new(512).unwrap();
        let req = EmbeddingRequest::new("model", "text").with_embedding_dimension(dim);
        assert_eq!(
            req.embedding_dimension(),
            Some(NonZeroUsize::new(512).unwrap())
        );
    }

    #[test]
    fn test_with_extra_inserts_param() {
        let req = EmbeddingRequest::new("model", "text").with_extra("truncate", json!(true));
        assert_eq!(req.extra_params()["truncate"], json!(true));
    }

    #[test]
    fn test_builder_methods_are_chainable() {
        let dim = NonZeroUsize::new(768).unwrap();
        let req = EmbeddingRequest::new("m", "t")
            .with_embedding_dimension(dim)
            .with_extra("k", json!("v"));
        assert_eq!(
            req.embedding_dimension(),
            Some(NonZeroUsize::new(768).unwrap())
        );
        assert_eq!(req.extra_params()["k"], json!("v"));
    }

    #[test]
    fn test_embedding_response_new_and_accessors() {
        let resp =
            EmbeddingResponse::new(vec![vec![0.1, 0.2, 0.3]], "test-model".to_string(), None);
        assert_eq!(resp.embeddings(), &[vec![0.1_f32, 0.2, 0.3]]);
        assert_eq!(resp.model(), "test-model");
        assert!(resp.usage().is_none());
    }

    #[test]
    fn test_embedding_response_with_usage() {
        let usage = TokenUsage::new(Some(5), None, Some(5));
        let resp = EmbeddingResponse::new(vec![], "m".to_string(), Some(usage));
        assert!(resp.usage().is_some());
        assert_eq!(resp.usage().unwrap().prompt_tokens(), Some(5));
    }
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-core.

use std::{collections::HashMap};

use serde::{Deserialize, Serialize};

use crate::model_provider::{
    common::{ModelProviderParams, ModelProviderResult},
    text_generation::TokenUsage,
};

/// A text-embedding request.
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    /// The embedding model identifier.
    pub model: String,

    /// One or more texts to embed in a single call. Model providers that only
    /// support a single input per request will issue multiple HTTP calls
    /// transparently and concatenate the results.
    pub input: Vec<String>,

    /// Optional output dimension hint forwarded to the model provider when supported.
    pub embedding_dimension: Option<usize>,

    /// Pass-through map for model-provider-specific options.
    pub extra_params: ModelProviderParams,
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

    /// Sets the desired output embedding dimension.
    pub fn with_embedding_dimension(mut self, dim: usize) -> Self {
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
    pub embeddings: Vec<Vec<f32>>,

    /// Embedding model name echoed back by the model provider.
    pub model: String,

    /// Token usage if reported.
    pub usage: Option<TokenUsage>,
}

/// Low-level interface to a model provider capable of producing embedding vectors.
///
/// Implementations communicate with a specific inference service (e.g. Ollama,
/// OpenAI) and translate [`EmbeddingRequest`] into the provider's wire format.
/// Higher-level code should use [`crate::embedding::TextEmbedder`] instead.
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
        assert_eq!(req.model, "my-model");
        assert_eq!(req.input, vec!["hello world".to_string()]);
        assert!(req.embedding_dimension.is_none());
        assert!(req.extra_params.is_empty());
    }

    #[test]
    fn test_new_batch_sets_all_inputs() {
        let inputs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let req = EmbeddingRequest::new_batch("model", inputs.clone());
        assert_eq!(req.input, inputs);
    }

    #[test]
    fn test_with_embedding_dimension_sets_field() {
        let req = EmbeddingRequest::new("model", "text").with_embedding_dimension(512);
        assert_eq!(req.embedding_dimension, Some(512));
    }

    #[test]
    fn test_with_extra_inserts_param() {
        let req = EmbeddingRequest::new("model", "text").with_extra("truncate", json!(true));
        assert_eq!(req.extra_params["truncate"], json!(true));
    }

    #[test]
    fn test_builder_methods_are_chainable() {
        let req = EmbeddingRequest::new("m", "t")
            .with_embedding_dimension(768)
            .with_extra("k", json!("v"));
        assert_eq!(req.embedding_dimension, Some(768));
        assert_eq!(req.extra_params["k"], json!("v"));
    }
}

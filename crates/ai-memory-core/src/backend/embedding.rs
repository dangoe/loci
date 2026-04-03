use std::{collections::HashMap, pin::Pin};

use serde::{Deserialize, Serialize};

use crate::backend::{
    common::{BackendParams, BackendResult},
    text_generation::TokenUsage,
};

/// A text-embedding request.
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    /// The embedding model identifier.
    pub model: String,

    /// One or more texts to embed in a single call.  Backends that only
    /// support a single input per request will issue multiple HTTP calls
    /// transparently and concatenate the results.
    pub input: Vec<String>,

    /// Optional output dimension hint forwarded to the backend when supported.
    pub embedding_dimension: Option<usize>,

    /// Pass-through map for backend-specific options.
    pub extra_params: BackendParams,
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

    /// Adds a backend-specific extra parameter.
    pub fn with_extra(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}

/// One embedding vector per input text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Parallel to `EmbedRequest::input` — one vector per input string.
    pub embeddings: Vec<Vec<f32>>,

    /// Embedding model name echoed back by the backend.
    pub model: String,

    /// Token usage if reported.
    pub usage: Option<TokenUsage>,
}

pub trait EmbeddingBackend: Send + Sync {
    fn embed(
        &self,
        req: EmbeddingRequest,
    ) -> Pin<Box<dyn Future<Output = BackendResult<EmbeddingResponse>> + Send + '_>>;
}

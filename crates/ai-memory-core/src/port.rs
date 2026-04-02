use std::future::Future;

use crate::{Embedding, EmbeddingError};

/// Computes embedding vectors for text input.
pub trait EmbeddingPort: Send + Sync {
    /// Returns the fixed number of dimensions this port produces.
    fn embedding_dimension(&self) -> usize;

    fn embed(&self, text: &str) -> impl Future<Output = Result<Embedding, EmbeddingError>> + Send + '_;
}

use std::fmt;

use uuid::Uuid;

use crate::backend::error::BackendError;

/// Errors produced by a [`crate::ContextEnhancer`].
#[derive(Debug)]
pub enum ContextualizerError {
    /// The memory store returned an error during query.
    MemoryStore(MemoryStoreError),
    /// The remote model call failed.
    RemoteModel(BackendError),
}

impl fmt::Display for ContextualizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryStore(e) => write!(f, "memory store error: {e}"),
            Self::RemoteModel(e) => write!(f, "remote model error: {e}"),
        }
    }
}

impl std::error::Error for ContextualizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MemoryStore(e) => Some(e),
            Self::RemoteModel(e) => Some(e),
        }
    }
}

/// Errors produced by a [`MemoryStore`][crate::MemoryStore] implementation.
#[derive(Debug)]
pub enum MemoryStoreError {
    Connection(String),
    Query(String),
    Embedding(EmbeddingError),
    /// No memory with the given ID exists in the store.
    NotFound(Uuid),
}

impl fmt::Display for MemoryStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connection(msg) => write!(f, "connection error: {msg}"),
            Self::Query(msg) => write!(f, "query error: {msg}"),
            Self::Embedding(e) => write!(f, "embedding error: {e}"),
            Self::NotFound(id) => write!(f, "memory not found: {id}"),
        }
    }
}

impl std::error::Error for MemoryStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Embedding(e) => Some(e),
            _ => None,
        }
    }
}

/// Errors produced by a [`TextEmbedder`][crate::TextEmbedder] implementation.
#[derive(Debug)]
pub enum EmbeddingError {
    TargetModel(BackendError),
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TargetModel(msg) => write!(f, "Tawrget model request error: {msg}"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

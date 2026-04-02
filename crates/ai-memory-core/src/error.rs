use std::fmt;

/// Errors produced by a [`MemoryStore`][crate::MemoryStore] implementation.
#[derive(Debug)]
pub enum MemoryStoreError {
    Connection(String),
    Query(String),
}

impl fmt::Display for MemoryStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connection(msg) => write!(f, "connection error: {msg}"),
            Self::Query(msg) => write!(f, "query error: {msg}"),
        }
    }
}

impl std::error::Error for MemoryStoreError {}

/// Errors produced by an [`EmbeddingPort`][crate::EmbeddingPort] implementation.
#[derive(Debug)]
pub enum EmbeddingError {
    Http(String),
    Parse(String),
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(msg) => write!(f, "HTTP error: {msg}"),
            Self::Parse(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

/// Errors produced by [`MemoryService`][crate::MemoryService].
#[derive(Debug)]
pub enum MemoryServiceError {
    Store(MemoryStoreError),
    Embedding(EmbeddingError),
}

impl fmt::Display for MemoryServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(e) => write!(f, "store error: {e}"),
            Self::Embedding(e) => write!(f, "embedding error: {e}"),
        }
    }
}

impl std::error::Error for MemoryServiceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(e) => Some(e),
            Self::Embedding(e) => Some(e),
        }
    }
}

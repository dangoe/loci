use std::fmt;

use uuid::Uuid;

/// Errors produced by a remote model client.
#[derive(Debug)]
pub enum RemoteModelRequestError {
    /// An HTTP-level error occurred while calling the model API.
    Http(String),
    /// The response from the model could not be parsed.
    Parse(String),
}

impl fmt::Display for RemoteModelRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(msg) => write!(f, "Remote model HTTP error: {msg}"),
            Self::Parse(msg) => write!(f, "Remote model response parse error: {msg}"),
        }
    }
}

impl std::error::Error for RemoteModelRequestError {}

/// Errors produced by a [`crate::MemoryExtractor`] implementation.
#[derive(Debug)]
pub enum MemoryExtractorError {
    /// The underlying remote model call failed.
    TargetModel(RemoteModelRequestError),
    /// The LLM output could not be parsed into memory inputs.
    Parse(String),
}

impl fmt::Display for MemoryExtractorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TargetModel(e) => write!(f, "extractor target model error: {e}"),
            Self::Parse(msg) => write!(f, "extractor parse error: {msg}"),
        }
    }
}

impl std::error::Error for MemoryExtractorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TargetModel(e) => Some(e),
            Self::Parse(_) => None,
        }
    }
}

/// Errors produced by a [`crate::ContextEnhancer`].
#[derive(Debug)]
pub enum ContextEnhancerError {
    /// The memory store returned an error during query.
    MemoryStore(MemoryStoreError),
    /// The remote model call failed.
    TargetModel(RemoteModelRequestError),
}

impl fmt::Display for ContextEnhancerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryStore(e) => write!(f, "memory store error: {e}"),
            Self::Llm(e) => write!(f, "LLM error: {e}"),
        }
    }
}

impl std::error::Error for ContextEnhancerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MemoryStore(e) => Some(e),
            Self::Llm(e) => Some(e),
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
    TargetModel(RemoteModelRequestError),
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TargetModel(msg) => write!(f, "Tawrget model request error: {msg}"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

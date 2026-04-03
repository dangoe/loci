use std::fmt;

use uuid::Uuid;

/// Errors produced by an LLM HTTP client.
#[derive(Debug)]
pub enum LlmError {
    /// An HTTP-level error occurred while calling the LLM API.
    Http(String),
    /// The response from the LLM could not be parsed.
    Parse(String),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(msg) => write!(f, "LLM HTTP error: {msg}"),
            Self::Parse(msg) => write!(f, "LLM parse error: {msg}"),
        }
    }
}

impl std::error::Error for LlmError {}

/// Errors produced by a [`crate::MemoryExtractor`] implementation.
#[derive(Debug)]
pub enum ExtractorError {
    /// The underlying LLM call failed.
    Llm(LlmError),
    /// The LLM output could not be parsed into memory inputs.
    Parse(String),
}

impl fmt::Display for ExtractorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Llm(e) => write!(f, "extractor LLM error: {e}"),
            Self::Parse(msg) => write!(f, "extractor parse error: {msg}"),
        }
    }
}

impl std::error::Error for ExtractorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Llm(e) => Some(e),
            Self::Parse(_) => None,
        }
    }
}

/// Errors produced by a [`crate::ContextEnhancer`].
#[derive(Debug)]
pub enum EnhancerError {
    /// The memory store returned an error during query.
    MemoryStore(MemoryStoreError),
    /// The LLM call failed.
    Llm(LlmError),
}

impl fmt::Display for EnhancerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryStore(e) => write!(f, "memory store error: {e}"),
            Self::Llm(e) => write!(f, "LLM error: {e}"),
        }
    }
}

impl std::error::Error for EnhancerError {
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

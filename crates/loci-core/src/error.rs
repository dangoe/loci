// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::fmt;

use uuid::Uuid;

use crate::model_provider::error::ModelProviderError;

/// Errors produced by a [`crate::contextualization::Contextualizer`].
#[derive(Debug)]
pub enum ContextualizerError {
    /// The memory store returned an error during query.
    MemoryStore(MemoryStoreError),
    /// The model provider call failed.
    ModelProvider(ModelProviderError),
}

impl fmt::Display for ContextualizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryStore(e) => write!(f, "memory store error: {e}"),
            Self::ModelProvider(e) => write!(f, "model provider error: {e}"),
        }
    }
}

impl std::error::Error for ContextualizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MemoryStore(e) => Some(e),
            Self::ModelProvider(e) => Some(e),
        }
    }
}

#[derive(Debug)]
pub enum MemoryExtractionError {
    /// The memory store returned an error while persisting extracted entries.
    MemoryStore(MemoryStoreError),
    /// The model provider call failed during extraction.
    ModelProvider(ModelProviderError),
    /// The model's response could not be parsed into memory entries.
    Parse(String),
    /// An error that does not fit any other variant (e.g. classification failures).
    Other(String),
}

impl fmt::Display for MemoryExtractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryStore(e) => write!(f, "memory store error: {e}"),
            Self::ModelProvider(e) => write!(f, "model provider error: {e}"),
            Self::Parse(msg) => write!(f, "parse error: {msg}"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for MemoryExtractionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MemoryStore(e) => Some(e),
            Self::ModelProvider(e) => Some(e),
            Self::Parse(_) | Self::Other(_) => None,
        }
    }
}

/// Errors produced by a [`MemoryStore`][crate::store::MemoryStore] implementation.
#[derive(Debug)]
pub enum MemoryStoreError {
    /// An error occurred while connecting to the store.
    Connection(String),
    /// An error occurred while querying the store.
    Query(String),
    /// An error occurred while embedding a memory.
    Embedding(EmbeddingError),
    /// The requested memory entry was not found in the store.
    NotFound(Uuid),
    /// An error occurred while saving a memory to the store.
    GenericSave(String),
}

impl fmt::Display for MemoryStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connection(msg) => write!(f, "connection error: {msg}"),
            Self::Query(msg) => write!(f, "query error: {msg}"),
            Self::Embedding(e) => write!(f, "embedding error: {e}"),
            Self::NotFound(id) => write!(f, "memory not found: {id}"),
            Self::GenericSave(msg) => write!(f, "generic save error: {msg}"),
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

/// Errors produced by a [`TextEmbedder`][crate::embedding::TextEmbedder] implementation.
#[derive(Debug)]
pub enum EmbeddingError {
    /// The model provider returned a transport or protocol error.
    ModelProvider(ModelProviderError),
    /// The model provider returned a response containing no vectors.
    EmptyResponse,
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelProvider(e) => write!(f, "model provider error: {e}"),
            Self::EmptyResponse => write!(f, "embedding model provider returned no vectors"),
        }
    }
}

impl std::error::Error for EmbeddingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ModelProvider(e) => Some(e),
            Self::EmptyResponse => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use pretty_assertions::assert_eq;
    use uuid::Uuid;

    use crate::model_provider::error::ModelProviderError;

    use super::*;

    #[test]
    fn test_embedding_error_model_provider_display() {
        let err = EmbeddingError::ModelProvider(ModelProviderError::Timeout);
        assert_eq!(err.to_string(), "model provider error: request timed out");
    }

    #[test]
    fn test_embedding_error_empty_response_display() {
        assert_eq!(
            EmbeddingError::EmptyResponse.to_string(),
            "embedding model provider returned no vectors",
        );
    }

    #[test]
    fn test_embedding_error_model_provider_has_source() {
        let err = EmbeddingError::ModelProvider(ModelProviderError::Timeout);
        assert!(err.source().is_some());
    }

    #[test]
    fn test_embedding_error_empty_response_has_no_source() {
        assert!(EmbeddingError::EmptyResponse.source().is_none());
    }

    #[test]
    fn test_memory_store_error_connection_display() {
        let err = MemoryStoreError::Connection("timeout".to_string());
        assert_eq!(err.to_string(), "connection error: timeout");
    }

    #[test]
    fn test_memory_store_error_query_display() {
        let err = MemoryStoreError::Query("bad syntax".to_string());
        assert_eq!(err.to_string(), "query error: bad syntax");
    }

    #[test]
    fn test_memory_store_error_embedding_display() {
        let err = MemoryStoreError::Embedding(EmbeddingError::EmptyResponse);
        assert_eq!(
            err.to_string(),
            "embedding error: embedding model provider returned no vectors",
        );
    }

    #[test]
    fn test_memory_store_error_not_found_display() {
        let id = Uuid::nil();
        let err = MemoryStoreError::NotFound(id);
        assert_eq!(err.to_string(), format!("memory not found: {id}"));
    }

    #[test]
    fn test_memory_store_error_embedding_has_source() {
        let err = MemoryStoreError::Embedding(EmbeddingError::EmptyResponse);
        assert!(err.source().is_some());
    }

    #[test]
    fn test_memory_store_error_connection_has_no_source() {
        assert!(
            MemoryStoreError::Connection("x".to_string())
                .source()
                .is_none()
        );
    }

    #[test]
    fn test_memory_store_error_query_has_no_source() {
        assert!(MemoryStoreError::Query("x".to_string()).source().is_none());
    }

    #[test]
    fn test_memory_store_error_not_found_has_no_source() {
        assert!(MemoryStoreError::NotFound(Uuid::nil()).source().is_none());
    }

    #[test]
    fn test_contextualizer_error_memory_store_display() {
        let err =
            ContextualizerError::MemoryStore(MemoryStoreError::Connection("db down".to_string()));
        assert_eq!(
            err.to_string(),
            "memory store error: connection error: db down"
        );
    }

    #[test]
    fn test_contextualizer_error_model_provider_display() {
        let err = ContextualizerError::ModelProvider(ModelProviderError::Timeout);
        assert_eq!(err.to_string(), "model provider error: request timed out");
    }

    #[test]
    fn test_contextualizer_error_memory_store_has_source() {
        let err = ContextualizerError::MemoryStore(MemoryStoreError::Connection("x".to_string()));
        assert!(err.source().is_some());
    }

    #[test]
    fn test_contextualizer_error_model_provider_has_source() {
        let err = ContextualizerError::ModelProvider(ModelProviderError::Timeout);
        assert!(err.source().is_some());
    }
}

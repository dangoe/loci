// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
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

/// Errors produced by a [`MemoryStore`][crate::store::MemoryStore] implementation.
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

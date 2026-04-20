// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-memory-store-qdrant.

/// Configuration for [`QdrantMemoryStore`][crate::store::QdrantMemoryStore].
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    /// The Qdrant collection name.
    collection_name: String,
    /// Optional deduplication threshold. When `Some(t)`, a new memory is not stored
    /// if an existing one already has a cosine similarity score ≥ `t`.
    similarity_threshold: Option<f64>,
}

impl QdrantConfig {
    /// Creates a new `QdrantConfig` with the given collection name and no similarity threshold.
    pub fn new(collection_name: impl Into<String>) -> Self {
        Self {
            collection_name: collection_name.into(),
            similarity_threshold: None,
        }
    }

    /// Sets the deduplication similarity threshold.
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = Some(threshold);
        self
    }

    /// Returns the Qdrant collection name.
    pub fn collection_name(&self) -> &str {
        &self.collection_name
    }

    /// Returns the optional deduplication similarity threshold.
    pub fn similarity_threshold(&self) -> Option<f64> {
        self.similarity_threshold
    }
}

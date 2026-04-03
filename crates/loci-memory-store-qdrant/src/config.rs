// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-memory-store-qdrant.

/// Configuration for [`QdrantMemoryStore`][crate::QdrantMemoryStore].
pub struct QdrantConfig {
    /// The Qdrant collection name.
    pub collection_name: String,
    /// Optional deduplication threshold. When `Some(t)`, a new memory is not stored
    /// if an existing one already has a cosine similarity score ≥ `t`.
    pub similarity_threshold: Option<f64>,
}

/// Configuration for [`Neo4jMemoryStore`][crate::Neo4jMemoryStore].
pub struct Neo4jConfig {
    /// The Neo4j database name.
    pub database: String,
    /// The dimensionality of the embedding vectors stored in the vector index.
    pub embedding_dimension: usize,
    /// Optional deduplication threshold. When `Some(t)`, a new memory is not stored
    /// if an existing one already has a cosine similarity score ≥ `t`.
    pub similarity_threshold: Option<f64>,
}

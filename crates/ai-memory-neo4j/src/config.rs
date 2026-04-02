/// Configuration for [`Neo4jMemoryStore`][crate::Neo4jMemoryStore].
pub struct Neo4jConfig {
    /// The Neo4j database name.
    pub database: String,
    /// Optional deduplication threshold. When `Some(t)`, a new memory is not stored
    /// if an existing one already has a cosine similarity score ≥ `t`.
    pub similarity_threshold: Option<f64>,
}

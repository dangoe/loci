use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use neo4rs::Graph;
use uuid::Uuid;

use ai_memory_core::{
    Embedding, Memory, MemoryEntry, MemoryInput, MemoryQuery, MemoryStore, MemoryStoreError, Score,
    TextEmbedder,
};

use crate::config::Neo4jConfig;

const INDEX_NAME: &str = "memory_embedding_idx";
const NODE_LABEL: &str = "Memory";

/// [`MemoryStore`] implementation backed by Neo4j using cosine vector similarity.
///
/// Embeddings are computed internally via the provided [`TextEmbedder`]; callers
/// work with plain text and are not exposed to embedding vectors.
pub struct Neo4jMemoryStore<E> {
    graph: Arc<Graph>,
    config: Neo4jConfig,
    embedder: E,
}

impl<E: TextEmbedder> Neo4jMemoryStore<E> {
    pub fn new(graph: Arc<Graph>, config: Neo4jConfig, embedder: E) -> Self {
        Self { graph, config, embedder }
    }

    /// Creates the vector index in Neo4j. Must be called once before using the store.
    pub async fn initialize(&self) -> Result<(), MemoryStoreError> {
        let dim = self.embedder.embedding_dimension();
        let cypher = format!(
            "CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS \
             FOR (m:{NODE_LABEL}) ON (m.embedding) \
             OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}",
        );
        self.graph
            .run(neo4rs::query(&cypher))
            .await
            .map_err(|e| MemoryStoreError::Connection(e.to_string()))
    }

    async fn do_save(&self, memory: Memory, embedding: Embedding) -> Result<MemoryEntry, MemoryStoreError> {
        let embedding_f64 = to_f64_vec(embedding.values());
        let metadata_json = serde_json::to_string(&memory.metadata)
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let q = neo4rs::query(&format!(
            "CREATE (m:{NODE_LABEL} {{id: $id, content: $content, embedding: $embedding, \
             metadata: $metadata, createdAt: $createdAt}}) \
             RETURN m.id AS id"
        ))
        .param("id", memory.id.to_string())
        .param("content", memory.content.clone())
        .param("embedding", embedding_f64)
        .param("metadata", metadata_json)
        .param("createdAt", memory.created_at.to_rfc3339());

        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        result
            .next()
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?
            .ok_or_else(|| MemoryStoreError::Query("CREATE returned no rows".to_string()))?;

        Ok(MemoryEntry {
            memory,
            score: Score::new(1.0).expect("1.0 is always a valid score"),
        })
    }

    async fn query_by_embedding(
        &self,
        embedding: &Embedding,
        max_results: usize,
        min_score: f64,
        filters: &HashMap<String, String>,
    ) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
        let embedding_f64 = to_f64_vec(embedding.values());
        let q = neo4rs::query(
            "CALL db.index.vector.queryNodes($index, $maxResults, $embedding) \
             YIELD node AS m, score \
             RETURN m.id AS id, m.content AS content, \
                    m.metadata AS metadata, m.createdAt AS createdAt, score",
        )
        .param("index", INDEX_NAME)
        .param("maxResults", max_results as i64)
        .param("embedding", embedding_f64);

        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let mut entries = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            let entry = Self::parse_entry(&row)?;
            if entry.score.value() < min_score {
                continue;
            }
            if !filters.iter().all(|(k, v)| entry.memory.metadata.get(k).map(|s| s == v).unwrap_or(false)) {
                continue;
            }
            entries.push(entry);
        }
        Ok(entries)
    }

    fn parse_entry(row: &neo4rs::Row) -> Result<MemoryEntry, MemoryStoreError> {
        let id_str: String = row.get("id").map_err(|e| MemoryStoreError::Query(e.to_string()))?;
        let content: String = row.get("content").map_err(|e| MemoryStoreError::Query(e.to_string()))?;
        let metadata_json: String = row.get("metadata").unwrap_or_else(|_| "{}".to_string());
        let created_at_str: String = row.get("createdAt").map_err(|e| MemoryStoreError::Query(e.to_string()))?;
        let score_val: f64 = row.get("score").map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        let id = Uuid::parse_str(&id_str).map_err(|e| MemoryStoreError::Query(e.to_string()))?;
        let metadata: HashMap<String, String> =
            serde_json::from_str(&metadata_json).unwrap_or_default();
        let created_at: DateTime<Utc> = DateTime::parse_from_rfc3339(&created_at_str)
            .map_err(|e| MemoryStoreError::Query(e.to_string()))?
            .into();
        let score =
            Score::new(score_val).map_err(|e| MemoryStoreError::Query(e.to_string()))?;

        Ok(MemoryEntry {
            memory: Memory { id, content, metadata, created_at },
            score,
        })
    }
}

impl<E: TextEmbedder> MemoryStore for Neo4jMemoryStore<E> {
    async fn save(&self, input: MemoryInput) -> Result<MemoryEntry, MemoryStoreError> {
        let memory = Memory::new(input.content, input.metadata);
        let embedding = self
            .embedder
            .embed(&memory.content)
            .await
            .map_err(MemoryStoreError::Embedding)?;

        if let Some(threshold) = self.config.similarity_threshold {
            let candidates = self
                .query_by_embedding(&embedding, 1, threshold, &HashMap::new())
                .await?;
            if let Some(entry) = candidates.into_iter().next() {
                log::debug!(
                    "deduplication: reusing memory {} (score {:.4})",
                    entry.memory.id,
                    entry.score.value()
                );
                return Ok(entry);
            }
        }

        self.do_save(memory, embedding).await
    }

    async fn query(&self, query: MemoryQuery) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
        let embedding = self
            .embedder
            .embed(&query.topic)
            .await
            .map_err(MemoryStoreError::Embedding)?;
        self.query_by_embedding(&embedding, query.max_results, query.min_score, &query.filters)
            .await
    }

    async fn delete(&self, id: Uuid) -> Result<(), MemoryStoreError> {
        let q = neo4rs::query(&format!("MATCH (m:{NODE_LABEL} {{id: $id}}) DELETE m"))
            .param("id", id.to_string());
        self.graph
            .run(q)
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))
    }

    async fn clear(&self) -> Result<(), MemoryStoreError> {
        let q = neo4rs::query(&format!("MATCH (m:{NODE_LABEL}) DELETE m"));
        self.graph
            .run(q)
            .await
            .map_err(|e| MemoryStoreError::Query(e.to_string()))
    }
}

fn to_f64_vec(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| v as f64).collect()
}

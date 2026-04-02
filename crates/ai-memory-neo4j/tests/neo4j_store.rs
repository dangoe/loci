use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use ai_memory_core::{
    Embedding, EmbeddingError, MemoryInput, MemoryQuery, MemoryStore, MemoryStoreError, Score,
    TextEmbedder,
};
use ai_memory_neo4j::{Neo4jConfig, Neo4jMemoryStore};
use neo4rs::Graph;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::neo4j::{Neo4j, Neo4jImage};
use uuid::Uuid;

const DIM: usize = 4;

// ── Test double ──────────────────────────────────────────────────────────────

/// Maps specific strings to predetermined unit vectors for predictable cosine similarity.
/// Any unmapped text returns the zero vector.
struct FakeTextEmbedder {
    mappings: HashMap<String, Vec<f32>>,
}

impl FakeTextEmbedder {
    fn new() -> Self {
        Self {
            mappings: HashMap::new(),
        }
    }

    fn with(mut self, text: &str, values: Vec<f32>) -> Self {
        self.mappings.insert(text.to_string(), values);
        self
    }
}

impl TextEmbedder for FakeTextEmbedder {
    fn embedding_dimension(&self) -> usize {
        DIM
    }

    fn embed(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<Embedding, EmbeddingError>> + Send + '_ {
        let values = self
            .mappings
            .get(text)
            .cloned()
            .unwrap_or_else(|| vec![0.0; DIM]);
        async move { Ok(Embedding::new(values)) }
    }
}

// Unit vector along `index` axis.
fn unit_vec(index: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[index] = 1.0;
    v
}

// ── Test setup ───────────────────────────────────────────────────────────────

async fn start_store(
    embedder: FakeTextEmbedder,
    similarity_threshold: Option<f64>,
) -> (
    Neo4jMemoryStore<FakeTextEmbedder>,
    ContainerAsync<Neo4jImage>,
) {
    let container: ContainerAsync<Neo4jImage> = Neo4j::default()
        .start()
        .await
        .expect("Docker must be available to run Neo4j integration tests");

    let host = container.get_host().await.unwrap();
    let port = container.image().bolt_port_ipv4().unwrap();
    let user = container.image().user().unwrap_or("neo4j");
    let pass = container.image().password().unwrap_or("password");

    let graph = Graph::new(format!("bolt://{host}:{port}"), user, pass)
        .await
        .expect("failed to connect to Neo4j");

    let config = Neo4jConfig {
        database: "neo4j".to_string(),
        similarity_threshold,
    };

    let store = Neo4jMemoryStore::new(Arc::new(graph), config, embedder);
    store
        .initialize()
        .await
        .expect("failed to initialize Neo4j vector index");

    (store, container)
}

fn input(content: &str) -> MemoryInput {
    MemoryInput::new(content.to_string(), HashMap::new())
}

fn input_with_metadata(content: &str, metadata: HashMap<String, String>) -> MemoryInput {
    MemoryInput::new(content.to_string(), metadata)
}

fn query(topic: &str, max_results: usize) -> MemoryQuery {
    MemoryQuery {
        topic: topic.to_string(),
        max_results,
        min_score: Score::ZERO,
        filters: HashMap::new(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_saved_memory_is_returned_by_query() {
    let embedder = FakeTextEmbedder::new()
        .with("hello world", unit_vec(0))
        .with("hello world query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let entry = store.save(input("hello world")).await.unwrap();

    let results = store.query(query("hello world query", 1)).await.unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory.id, entry.memory.id);
    assert_eq!(results[0].memory.content, "hello world");
    assert!(results[0].score.value() > 0.9);
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_query_ranks_results_by_similarity() {
    let embedder = FakeTextEmbedder::new()
        .with("x-axis", unit_vec(0))
        .with("y-axis", unit_vec(1))
        .with("near x", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    store.save(input("x-axis")).await.unwrap();
    store.save(input("y-axis")).await.unwrap();

    // "near x" maps to the same vector as "x-axis" so it should rank first
    let results = store.query(query("near x", 2)).await.unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].memory.content, "x-axis");
    assert!(results[0].score.value() > results[1].score.value());
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_query_respects_max_results() {
    let mut embedder = FakeTextEmbedder::new().with("query", unit_vec(0));
    for i in 0..5 {
        embedder = embedder.with(&format!("memory {i}"), unit_vec(i % DIM));
    }
    let (store, _container) = start_store(embedder, None).await;

    for i in 0..5 {
        store.save(input(&format!("memory {i}"))).await.unwrap();
    }

    let results = store.query(query("query", 3)).await.unwrap();

    assert_eq!(results.len(), 3);
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_metadata_is_persisted_and_restored() {
    let embedder = FakeTextEmbedder::new()
        .with("tagged content", unit_vec(0))
        .with("tagged content query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let metadata = HashMap::from([
        ("source".to_string(), "test".to_string()),
        ("language".to_string(), "en".to_string()),
    ]);
    store
        .save(input_with_metadata("tagged content", metadata.clone()))
        .await
        .unwrap();

    let results = store.query(query("tagged content query", 1)).await.unwrap();

    assert_eq!(results[0].memory.metadata, metadata);
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_query_filters_by_metadata() {
    let embedder = FakeTextEmbedder::new()
        .with("doc en", unit_vec(0))
        .with("doc de", unit_vec(0))
        .with("topic", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    store
        .save(input_with_metadata(
            "doc en",
            HashMap::from([("lang".to_string(), "en".to_string())]),
        ))
        .await
        .unwrap();
    store
        .save(input_with_metadata(
            "doc de",
            HashMap::from([("lang".to_string(), "de".to_string())]),
        ))
        .await
        .unwrap();

    let results = store
        .query(MemoryQuery {
            topic: "topic".to_string(),
            max_results: 10,
            min_score: Score::ZERO,
            filters: HashMap::from([("lang".to_string(), "en".to_string())]),
        })
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory.content, "doc en");
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_query_respects_min_score() {
    // x-axis and y-axis are orthogonal → cosine = 0
    let embedder = FakeTextEmbedder::new()
        .with("x-axis", unit_vec(0))
        .with("y-axis", unit_vec(1))
        .with("query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    store.save(input("x-axis")).await.unwrap();
    store.save(input("y-axis")).await.unwrap();

    // min_score of 0.9 should exclude the orthogonal entry
    let results = store
        .query(MemoryQuery {
            topic: "query".to_string(),
            max_results: 10,
            min_score: Score::new(0.9).unwrap(),
            filters: HashMap::new(),
        })
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory.content, "x-axis");
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_deleted_memory_is_not_returned_by_query() {
    let embedder = FakeTextEmbedder::new()
        .with("to be deleted", unit_vec(0))
        .with("query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let entry = store.save(input("to be deleted")).await.unwrap();
    store.delete(entry.memory.id).await.unwrap();

    let results = store.query(query("query", 10)).await.unwrap();

    assert!(results.iter().all(|e| e.memory.id != entry.memory.id));
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_clear_removes_all_memories() {
    let embedder = FakeTextEmbedder::new()
        .with("alpha", unit_vec(0))
        .with("beta", unit_vec(1))
        .with("query", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    store.save(input("alpha")).await.unwrap();
    store.save(input("beta")).await.unwrap();
    store.clear().await.unwrap();

    let results = store.query(query("query", 10)).await.unwrap();

    assert!(results.is_empty());
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_deduplication_reuses_id_when_score_meets_threshold() {
    // Both "original" and "near-duplicate" map to the same vector → cosine = 1.0 ≥ 0.9
    let embedder = FakeTextEmbedder::new()
        .with("original", unit_vec(0))
        .with("near-duplicate", unit_vec(0));
    let (store, _container) = start_store(embedder, Some(0.9)).await;

    let original = store.save(input("original")).await.unwrap();
    let duplicate = store.save(input("near-duplicate")).await.unwrap();

    assert_eq!(duplicate.memory.id, original.memory.id);
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_deduplication_stores_new_memory_when_score_below_threshold() {
    // Orthogonal vectors → cosine ≈ 0 < 0.9 → stored separately
    let embedder = FakeTextEmbedder::new()
        .with("x-axis topic", unit_vec(0))
        .with("y-axis topic", unit_vec(1));
    let (store, _container) = start_store(embedder, Some(0.9)).await;

    let first = store.save(input("x-axis topic")).await.unwrap();
    let second = store.save(input("y-axis topic")).await.unwrap();

    assert_ne!(second.memory.id, first.memory.id);
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_update_changes_content_and_returns_updated_entry() {
    let embedder = FakeTextEmbedder::new()
        .with("original content", unit_vec(0))
        .with("updated content", unit_vec(1));
    let (store, _container) = start_store(embedder, None).await;

    let saved = store.save(input("original content")).await.unwrap();
    let updated = store
        .update(saved.memory.id, input("updated content"))
        .await
        .unwrap();

    assert_eq!(updated.memory.id, saved.memory.id);
    assert_eq!(updated.memory.content, "updated content");
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_update_returns_not_found_for_unknown_id() {
    let embedder = FakeTextEmbedder::new().with("anything", unit_vec(0));
    let (store, _container) = start_store(embedder, None).await;

    let unknown_id = Uuid::new_v4();
    let result = store.update(unknown_id, input("anything")).await;

    assert!(matches!(result, Err(MemoryStoreError::NotFound(id)) if id == unknown_id));
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_deduplication_does_not_collapse_different_metadata() {
    // Same embedding but different metadata → must produce two distinct entries
    let embedder = FakeTextEmbedder::new()
        .with("same content a", unit_vec(0))
        .with("same content b", unit_vec(0));
    let (store, _container) = start_store(embedder, Some(0.9)).await;

    let first = store
        .save(input_with_metadata(
            "same content a",
            HashMap::from([("source".to_string(), "a".to_string())]),
        ))
        .await
        .unwrap();
    let second = store
        .save(input_with_metadata(
            "same content b",
            HashMap::from([("source".to_string(), "b".to_string())]),
        ))
        .await
        .unwrap();

    assert_ne!(second.memory.id, first.memory.id);
}

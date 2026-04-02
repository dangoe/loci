use std::collections::HashMap;
use std::sync::Arc;

use ai_memory_core::{Embedding, Memory, MemoryQuery, MemoryStore};
use ai_memory_neo4j::{Neo4jConfig, Neo4jMemoryStore};
use neo4rs::Graph;
use testcontainers::runners::AsyncRunner;
use testcontainers::ContainerAsync;
use testcontainers_modules::neo4j::{Neo4j, Neo4jImage};

const DIM: usize = 4;

// Creates a Neo4jMemoryStore backed by a fresh container.
// The container must be kept alive (held by the caller) for the duration of the test.
async fn start_store(
    similarity_threshold: Option<f64>,
) -> (Neo4jMemoryStore, ContainerAsync<Neo4jImage>) {
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
        embedding_dimension: DIM,
        similarity_threshold,
    };

    let store = Neo4jMemoryStore::new(Arc::new(graph), config);
    store
        .initialize()
        .await
        .expect("failed to initialize Neo4j vector index");

    (store, container)
}

// Unit vector along `index` axis: all zeros except a 1.0 at position `index`.
fn unit_vec(index: usize) -> Embedding {
    let mut values = vec![0.0_f32; DIM];
    values[index] = 1.0;
    Embedding::new(values)
}

fn memory(content: &str, embedding: Embedding) -> Memory {
    Memory::new(content.to_string(), embedding, HashMap::new())
}

fn memory_with_metadata(
    content: &str,
    embedding: Embedding,
    metadata: HashMap<String, String>,
) -> Memory {
    Memory::new(content.to_string(), embedding, metadata)
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_saved_memory_is_returned_by_query() {
    let (store, _container) = start_store(None).await;

    let id = store.save(memory("hello world", unit_vec(0))).await.unwrap();

    let results = store
        .query(MemoryQuery { embedding: unit_vec(0), max_results: 1 })
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory.id, id);
    assert_eq!(results[0].memory.content, "hello world");
    assert!(results[0].score.value() > 0.9);
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_query_ranks_results_by_similarity() {
    let (store, _container) = start_store(None).await;

    store.save(memory("x-axis", unit_vec(0))).await.unwrap();
    store.save(memory("y-axis", unit_vec(1))).await.unwrap();

    // Query aligned with x-axis — "x-axis" memory must rank first
    let results = store
        .query(MemoryQuery { embedding: unit_vec(0), max_results: 2 })
        .await
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].memory.content, "x-axis");
    assert!(results[0].score.value() > results[1].score.value());
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_query_respects_max_results() {
    let (store, _container) = start_store(None).await;

    for i in 0..5 {
        store
            .save(memory(&format!("memory {i}"), unit_vec(i % DIM)))
            .await
            .unwrap();
    }

    let results = store
        .query(MemoryQuery { embedding: unit_vec(0), max_results: 3 })
        .await
        .unwrap();

    assert_eq!(results.len(), 3);
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_metadata_is_persisted_and_restored() {
    let (store, _container) = start_store(None).await;

    let metadata = HashMap::from([
        ("source".to_string(), "test".to_string()),
        ("language".to_string(), "en".to_string()),
    ]);
    store
        .save(memory_with_metadata("tagged content", unit_vec(0), metadata.clone()))
        .await
        .unwrap();

    let results = store
        .query(MemoryQuery { embedding: unit_vec(0), max_results: 1 })
        .await
        .unwrap();

    assert_eq!(results[0].memory.metadata, metadata);
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_deleted_memory_is_not_returned_by_query() {
    let (store, _container) = start_store(None).await;

    let id = store.save(memory("to be deleted", unit_vec(0))).await.unwrap();
    store.delete(id).await.unwrap();

    let results = store
        .query(MemoryQuery { embedding: unit_vec(0), max_results: 10 })
        .await
        .unwrap();

    assert!(results.iter().all(|e| e.memory.id != id));
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_deduplication_reuses_id_when_score_meets_threshold() {
    // threshold = 0.9; same embedding → cosine similarity = 1.0 → reuse
    let (store, _container) = start_store(Some(0.9)).await;

    let original_id = store.save(memory("original", unit_vec(0))).await.unwrap();
    let duplicate_id = store.save(memory("near-duplicate", unit_vec(0))).await.unwrap();

    assert_eq!(duplicate_id, original_id);
}

#[tokio::test]
#[ignore = "requires Docker"]
async fn test_deduplication_stores_new_memory_when_score_below_threshold() {
    // threshold = 0.9; orthogonal embeddings → cosine similarity ≈ 0.5 → store separately
    let (store, _container) = start_store(Some(0.9)).await;

    let first_id = store.save(memory("x-axis topic", unit_vec(0))).await.unwrap();
    let second_id = store.save(memory("y-axis topic", unit_vec(1))).await.unwrap();

    assert_ne!(second_id, first_id);
}

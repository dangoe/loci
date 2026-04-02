use std::collections::HashMap;

use uuid::Uuid;

use crate::{EmbeddingPort, Memory, MemoryEntry, MemoryQuery, MemoryServiceError, MemoryStore};

/// High-level API composing a [`MemoryStore`] with an [`EmbeddingPort`].
///
/// Text is embedded automatically; callers work with strings, not raw vectors.
pub struct MemoryService<S, E> {
    store: S,
    embedding: E,
}

impl<S: MemoryStore, E: EmbeddingPort> MemoryService<S, E> {
    pub fn new(store: S, embedding: E) -> Self {
        Self { store, embedding }
    }

    /// Embeds `content` and persists it along with `metadata`.
    pub async fn memorize(
        &self,
        content: &str,
        metadata: HashMap<String, String>,
    ) -> Result<Uuid, MemoryServiceError> {
        let embedding = self
            .embedding
            .embed(content)
            .await
            .map_err(MemoryServiceError::Embedding)?;
        let memory = Memory::new(content.to_owned(), embedding, metadata);
        self.store.save(memory).await.map_err(MemoryServiceError::Store)
    }

    /// Embeds `topic` and returns the closest stored memories.
    pub async fn retrieve(
        &self,
        topic: &str,
        max_results: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryServiceError> {
        let embedding = self
            .embedding
            .embed(topic)
            .await
            .map_err(MemoryServiceError::Embedding)?;
        let query = MemoryQuery { embedding, max_results };
        self.store.query(query).await.map_err(MemoryServiceError::Store)
    }

    /// Deletes the memory with the given `id`.
    pub async fn forget(&self, id: Uuid) -> Result<(), MemoryServiceError> {
        self.store.delete(id).await.map_err(MemoryServiceError::Store)
    }
}

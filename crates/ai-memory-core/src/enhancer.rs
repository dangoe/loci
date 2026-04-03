use std::collections::HashMap;
use std::sync::Arc;

use crate::store::MemoryStore;
use crate::{ContextEnhancerError, MemoryEntry, MemoryQuery, Score};

use crate::extractor::MemoryExtractor;
use crate::remote_model::{LlmClient, Message, Role};

/// Configuration for a [`ContextEnhancer`].
#[derive(Debug, Clone)]
pub struct EnhancerConfig {
    /// Maximum number of memories to retrieve and inject into the prompt.
    pub max_memories: usize,
    /// Minimum similarity score a memory must have to be included.
    pub min_score: Score,
    /// Metadata filters applied when querying the memory store.
    pub filters: HashMap<String, String>,
}

impl Default for EnhancerConfig {
    fn default() -> Self {
        Self {
            max_memories: 5,
            min_score: Score::ZERO,
            filters: HashMap::new(),
        }
    }
}

/// Enhances a user prompt with relevant memories before calling an LLM.
///
/// On each call to [`ContextEnhancer::enhance`] the enhancer:
/// 1. Queries the memory store for entries relevant to the prompt.
/// 2. Prepends a `[MEMORY CONTEXT]` block to the prompt when memories exist.
/// 3. Calls the LLM with the enriched prompt.
/// 4. Optionally extracts new memories from the exchange in a background task.
pub struct ContextEnhancer<S: MemoryStore, L: LlmClient> {
    store: Arc<S>,
    llm: Arc<L>,
    extractor: Option<Arc<dyn MemoryExtractor>>,
    config: EnhancerConfig,
}

impl<S, L> ContextEnhancer<S, L>
where
    S: MemoryStore + 'static,
    L: LlmClient + 'static,
{
    /// Creates a new `ContextEnhancer` with default configuration.
    pub fn new(store: Arc<S>, target_llm: Arc<L>) -> Self {
        Self {
            store,
            target_llm,
            extractor: None,
            config: EnhancerConfig::default(),
        }
    }

    /// Applies a custom [`EnhancerConfig`].
    pub fn with_config(mut self, config: EnhancerConfig) -> Self {
        self.config = config;
        self
    }

    /// Attaches a [`MemoryExtractor`] that will save new memories after each
    /// LLM call in a fire-and-forget background task.
    pub fn with_extractor(mut self, extractor: Arc<dyn MemoryExtractor>) -> Self {
        self.extractor = Some(extractor);
        self
    }

    /// Enhances `prompt` with memory context and returns the LLM response.
    ///
    /// # Errors
    ///
    /// Returns [`EnhancerError::MemoryStore`] if the store query fails, or
    /// [`EnhancerError::Llm`] if the LLM call fails.
    pub async fn enhance(&self, prompt: &str) -> Result<String, ContextEnhancerError> {
        let memory_entries = self.query_memory(prompt).await?;

        log::debug!("retrieved {} relevant memories", memory_entries.len());

        let enriched_prompt = self.enrich_prompt(prompt, memory_entries);

        // 3. Call the LLM.
        let messages = vec![Message {
            role: Role::User,
            content: enriched_prompt,
        }];

        let response = self
            .llm
            .complete(&messages)
            .await
            .map_err(ContextEnhancerError::Llm)?;

        // 4. Fire-and-forget memory extraction.
        if let Some(extractor) = &self.extractor {
            let extractor = Arc::clone(extractor);
            let store = Arc::clone(&self.store);
            let prompt_owned = prompt.to_string();
            let response_owned = response.clone();

            tokio::spawn(async move {
                match extractor.extract(&prompt_owned, &response_owned).await {
                    Ok(inputs) => {
                        for input in inputs {
                            if let Err(e) = store.save(input).await {
                                log::warn!("failed to save extracted memory: {e}");
                            }
                        }
                    }
                    Err(e) => log::warn!("memory extraction failed: {e}"),
                }
            });
        }

        Ok(response)
    }

    async fn query_memory(
        &self,
        prompt: &str,
    ) -> Result<Vec<crate::MemoryEntry>, ContextEnhancerError> {
        let query = MemoryQuery {
            topic: prompt.to_string(),
            max_results: self.config.max_memories,
            min_score: self.config.min_score,
            filters: self.config.filters.clone(),
        };

        self.store
            .query(query)
            .await
            .map_err(ContextEnhancerError::MemoryStore)
    }

    fn enrich_prompt(&self, prompt: &str, memory_entries: Vec<MemoryEntry>) -> String {
        if memory_entries.is_empty() {
            prompt.to_string()
        } else {
            let mut buf = String::from("[MEMORY CONTEXT]\n");
            buf.push_str("The following memories are relevant to your request:\n\n");

            for (i, entry) in memory_entries.iter().enumerate() {
                buf.push_str(&format!(
                    "{}. (relevance: {:.2}) {}\n",
                    i + 1,
                    entry.score.value(),
                    entry.memory.content,
                ));
            }

            buf.push_str("\n[USER PROMPT]\n");
            buf.push_str(prompt);
            buf
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::future::Future;

    use uuid::Uuid;

    use crate::{
        Memory, MemoryEntry, MemoryInput, MemoryQuery, MemoryStore, MemoryStoreError, Score,
    };

    use crate::LlmError;
    use crate::remote_model::{LlmClient, Message};

    use super::*;

    // ── Mock store ────────────────────────────────────────────────────────────

    struct MockStore {
        entries: Vec<MemoryEntry>,
    }

    impl MemoryStore for MockStore {
        fn save(
            &self,
            input: MemoryInput,
        ) -> impl Future<Output = Result<MemoryEntry, MemoryStoreError>> + Send + '_ {
            let entry = MemoryEntry {
                memory: Memory::new(input.content, input.metadata),
                score: Score::ZERO,
            };
            async move { Ok(entry) }
        }

        fn query(
            &self,
            _query: MemoryQuery,
        ) -> impl Future<Output = Result<Vec<MemoryEntry>, MemoryStoreError>> + Send + '_ {
            let entries = self.entries.clone();
            async move { Ok(entries) }
        }

        fn update(
            &self,
            _id: Uuid,
            _input: MemoryInput,
        ) -> impl Future<Output = Result<MemoryEntry, MemoryStoreError>> + Send + '_ {
            async move { Err(MemoryStoreError::NotFound(_id)) }
        }

        fn delete(
            &self,
            _id: Uuid,
        ) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_ {
            async move { Ok(()) }
        }

        fn clear(&self) -> impl Future<Output = Result<(), MemoryStoreError>> + Send + '_ {
            async move { Ok(()) }
        }
    }

    // ── Mock LLM ─────────────────────────────────────────────────────────────

    struct MockLlm {
        reply: String,
    }

    impl LlmClient for MockLlm {
        fn complete(
            &self,
            _messages: &[Message],
        ) -> impl Future<Output = Result<String, LlmError>> + Send + '_ {
            let reply = self.reply.clone();
            async move { Ok(reply) }
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn enhance_no_memories_returns_plain_response() {
        let store = Arc::new(MockStore { entries: vec![] });
        let llm = Arc::new(MockLlm {
            reply: "Hello!".to_string(),
        });

        let enhancer = ContextEnhancer::new(store, llm);
        let result = enhancer.enhance("hi").await.unwrap();
        assert_eq!(result, "Hello!");
    }

    #[tokio::test]
    async fn enhance_with_memories_injects_context() {
        use std::collections::HashMap;

        let entry = MemoryEntry {
            memory: Memory::new("user likes Rust".to_string(), HashMap::new()),
            score: Score::new(0.9).unwrap(),
        };

        // Capture the message sent to the LLM via a channel.
        let store = Arc::new(MockStore {
            entries: vec![entry],
        });

        struct CaptureLlm {
            tx: std::sync::Mutex<std::sync::mpsc::SyncSender<String>>,
        }

        impl LlmClient for CaptureLlm {
            fn complete(
                &self,
                messages: &[Message],
            ) -> impl Future<Output = Result<String, LlmError>> + Send + '_ {
                let content = messages[0].content.clone();
                self.tx.lock().unwrap().send(content).unwrap();
                async move { Ok("noted".to_string()) }
            }
        }

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let llm = Arc::new(CaptureLlm {
            tx: std::sync::Mutex::new(tx),
        });

        let enhancer = ContextEnhancer::new(store, llm);
        enhancer.enhance("what do I like?").await.unwrap();

        let sent = rx.try_recv().unwrap();
        assert!(sent.contains("[MEMORY CONTEXT]"), "missing context block");
        assert!(sent.contains("user likes Rust"), "missing memory content");
        assert!(sent.contains("[USER PROMPT]"), "missing user prompt block");
        assert!(sent.contains("what do I like?"), "missing original prompt");
    }
}

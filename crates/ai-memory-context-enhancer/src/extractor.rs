use std::pin::Pin;
use std::sync::Arc;

use ai_memory_core::MemoryInput;

use crate::error::ExtractorError;
use crate::llm::{LlmClient, Message, Role};

/// A type alias for a boxed, heap-allocated future that is `Send`.
///
/// Using a boxed future makes [`MemoryExtractor`] object-safe so it can be
/// stored as `Arc<dyn MemoryExtractor>`.
pub type BoxFuture<'a, T> = Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// Extracts memorable facts from a prompt/response pair.
///
/// The trait is object-safe: implementations are stored as `Arc<dyn
/// MemoryExtractor>`. All methods use [`BoxFuture`] instead of AFIT so that
/// the vtable can be constructed at compile time.
pub trait MemoryExtractor: Send + Sync {
    /// Analyses the given `prompt` and LLM `response` and returns a list of
    /// [`MemoryInput`] entries worth persisting for future retrieval.
    fn extract<'a>(
        &'a self,
        prompt: &'a str,
        response: &'a str,
    ) -> BoxFuture<'a, Result<Vec<MemoryInput>, ExtractorError>>;
}

// ── NoOpExtractor ─────────────────────────────────────────────────────────────

/// A no-operation extractor that always returns an empty list.
///
/// Useful as the default when memory extraction is disabled.
pub struct NoOpExtractor;

impl MemoryExtractor for NoOpExtractor {
    fn extract<'a>(
        &'a self,
        _prompt: &'a str,
        _response: &'a str,
    ) -> BoxFuture<'a, Result<Vec<MemoryInput>, ExtractorError>> {
        Box::pin(async { Ok(vec![]) })
    }
}

// ── LlmMemoryExtractor ────────────────────────────────────────────────────────

const EXTRACTION_SYSTEM_PROMPT: &str = "\
You are a memory extraction assistant. Your job is to identify and extract key \
facts from a conversation that would be useful to remember for future interactions. \
Return ONLY a JSON array of strings. Each string should be a concise, self-contained \
memorable fact. If there are no memorable facts, return an empty JSON array: [].";

/// An extractor that uses an LLM to identify memorable facts in a conversation.
///
/// The LLM is prompted with a system instruction and the raw prompt/response
/// pair. It is expected to reply with a JSON array of fact strings, each of
/// which becomes a [`MemoryInput`] entry.
pub struct LlmMemoryExtractor<L: LlmClient> {
    llm: Arc<L>,
}

impl<L: LlmClient> LlmMemoryExtractor<L> {
    /// Creates a new extractor backed by the given LLM client.
    pub fn new(llm: Arc<L>) -> Self {
        Self { llm }
    }
}

impl<L: LlmClient + 'static> MemoryExtractor for LlmMemoryExtractor<L> {
    fn extract<'a>(
        &'a self,
        prompt: &'a str,
        response: &'a str,
    ) -> BoxFuture<'a, Result<Vec<MemoryInput>, ExtractorError>> {
        Box::pin(async move {
            let conversation = format!("User prompt:\n{prompt}\n\nAssistant response:\n{response}");

            let messages = vec![
                Message {
                    role: Role::System,
                    content: EXTRACTION_SYSTEM_PROMPT.to_string(),
                },
                Message {
                    role: Role::User,
                    content: conversation,
                },
            ];

            let raw = self
                .llm
                .complete(&messages)
                .await
                .map_err(ExtractorError::Llm)?;

            log::debug!("memory extractor LLM response: {raw}");

            // Parse the JSON array of fact strings.
            let facts: Vec<String> = serde_json::from_str(raw.trim())
                .map_err(|e| ExtractorError::Parse(format!("expected JSON array: {e}")))?;

            let inputs = facts
                .into_iter()
                .map(|fact| MemoryInput::new(fact, std::collections::HashMap::new()))
                .collect();

            Ok(inputs)
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn no_op_extractor_returns_empty_vec() {
        let ext = NoOpExtractor;
        let result = ext.extract("hello", "world").await.unwrap();
        assert!(result.is_empty());
    }
}

use std::future::Future;

use crate::LlmError;

/// The role of a participant in a chat conversation.
#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    /// A system-level instruction.
    System,
    /// A message from the human user.
    User,
    /// A message from the AI assistant.
    Assistant,
}

impl Role {
    /// Returns the lowercase string representation expected by OpenAI-compatible APIs.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// A single message in a chat conversation.
#[derive(Debug, Clone)]
pub struct Message {
    /// The role of the message author.
    pub role: Role,
    /// The text content of the message.
    pub content: String,
}

/// A client capable of completing a chat conversation with an LLM.
///
/// This trait uses AFIT (async-fn-in-trait) with an explicit `Send` bound so
/// implementations can be used across await points in multi-threaded runtimes.
pub trait LlmClient: Send + Sync {
    /// Sends the provided messages to the LLM and returns the assistant reply.
    fn complete(
        &self,
        messages: &[Message],
    ) -> impl Future<Output = Result<String, LlmError>> + Send + '_;
}

use std::future::Future;

use crate::LlmError;

/// A single message in a chat conversation.
#[derive(Debug, Clone)]
pub struct Message {
    /// The text content of the message.
    pub content: String,
}

/// A client capable of completing a chat conversation with an LLM.
///
/// This trait uses AFIT (async-fn-in-trait) with an explicit `Send` bound so
/// implementations can be used across await points in multi-threaded runtimes.
pub trait RemoteModelClient: Send + Sync {
    /// Sends the provided messages to the target remote model and returns the reply.
    fn complete(
        &self,
        messages: &[Message],
    ) -> impl Future<Output = Result<String, RemoteModelError>> + Send + '_;
}

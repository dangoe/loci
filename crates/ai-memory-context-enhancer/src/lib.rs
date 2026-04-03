//! Context enhancer for AI memory — enriches LLM prompts with retrieved memories.
//!
//! # Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use ai_memory_context_enhancer::{ContextEnhancer, OpenAiCompatibleClient};
//!
//! # async fn example() {
//! // Assumes a MemoryStore implementation is available as `store`.
//! let llm = Arc::new(OpenAiCompatibleClient::new(
//!     "https://api.openai.com/v1",
//!     "gpt-4o",
//!     Some("sk-...".to_string()),
//! ));
//! # }
//! ```

mod enhancer;
mod error;
mod extractor;
mod llm;
mod openai;

pub use enhancer::{ContextEnhancer, EnhancerConfig};
pub use error::{EnhancerError, ExtractorError, LlmError};
pub use extractor::{BoxFuture, LlmMemoryExtractor, MemoryExtractor, NoOpExtractor};
pub use llm::{LlmClient, Message, Role};
pub use openai::OpenAiCompatibleClient;

mod embedding;
mod enhancer;
mod error;
mod extractor;
mod memory;
mod openai;
mod remote_model;
mod store;

pub use embedding::{Embedding, TextEmbedder};
pub use enhancer::{ContextEnhancer, EnhancerConfig};
pub use error::{
    ContextEnhancerError, EmbeddingError, LlmError, MemoryExtractorError, MemoryStoreError,
};
pub use extractor::{BoxFuture, LlmMemoryExtractor, MemoryExtractor, NoOpExtractor};
pub use memory::{InvalidScore, Memory, MemoryEntry, MemoryInput, MemoryQuery, Score};
pub use openai::OpenAiCompatibleClient;
pub use remote_model::{LlmClient, Message, Role};
pub use store::MemoryStore;

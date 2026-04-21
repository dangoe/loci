// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::model_provider::error::ModelProviderError;

/// A type alias for model provider results, which are `Result<T, ModelProviderError>`.
pub type ModelProviderResult<T> = Result<T, ModelProviderError>;

/// Arbitrary model-provider-specific options forwarded verbatim in the request body
/// (e.g. `top_k`, `repeat_penalty`, vendor-specific flags).
pub type ModelProviderParams = HashMap<String, Value>;

/// Token consumption reported by the model provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

impl TokenUsage {
    /// Creates a new `TokenUsage`.
    pub fn new(
        prompt_tokens: Option<u32>,
        completion_tokens: Option<u32>,
        total_tokens: Option<u32>,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        }
    }

    /// Returns the number of prompt tokens, if reported.
    pub fn prompt_tokens(&self) -> Option<u32> {
        self.prompt_tokens
    }

    /// Returns the number of completion tokens, if reported.
    pub fn completion_tokens(&self) -> Option<u32> {
        self.completion_tokens
    }

    /// Returns the total token count, if reported.
    pub fn total_tokens(&self) -> Option<u32> {
        self.total_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_usage_new_stores_all_fields() {
        let u = TokenUsage::new(Some(10), Some(20), Some(30));
        assert_eq!(u.prompt_tokens(), Some(10));
        assert_eq!(u.completion_tokens(), Some(20));
        assert_eq!(u.total_tokens(), Some(30));
    }

    #[test]
    fn test_token_usage_new_accepts_none_fields() {
        let u = TokenUsage::new(None, None, None);
        assert!(u.prompt_tokens().is_none());
        assert!(u.completion_tokens().is_none());
        assert!(u.total_tokens().is_none());
    }
}

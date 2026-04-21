// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::sync::Arc;

use futures::{StreamExt, future::BoxFuture};
use log::debug;

use crate::{
    error::MemoryExtractionError,
    memory::{MemoryEntry, store::MemoryInput},
    model_provider::text_generation::{
        TextGenerationModelProvider, TextGenerationRequest, ThinkingMode,
    },
};

/// Synthesizes the content for a merged memory entry from a candidate and its
/// matching existing entries.
///
/// The trust level is computed separately by the extractor; this trait is
/// responsible only for deciding what the merged *content string* should be.
pub trait MemoryMergeStrategy<P: Send + Sync>: Send + Sync {
    fn merge<'a>(
        &'a self,
        candidate: &'a MemoryInput,
        matches: &'a [MemoryEntry],
        params: &'a P,
    ) -> BoxFuture<'a, Result<String, MemoryExtractionError>>;
}

/// Fast merge strategy: returns the content of whichever entry has the highest
/// effective score — the candidate or one of the existing matches.
///
/// No LLM call is made. This is appropriate when latency is more important than
/// content quality and the matched entries are close paraphrases.
pub struct BestScoreMergeStrategy;

impl<P: Send + Sync> MemoryMergeStrategy<P> for BestScoreMergeStrategy {
    fn merge<'a>(
        &'a self,
        candidate: &'a MemoryInput,
        matches: &'a [MemoryEntry],
        _params: &'a P,
    ) -> BoxFuture<'a, Result<String, MemoryExtractionError>> {
        let candidate_score = candidate.trust().effective_score();
        let best_match = matches.iter().max_by(|a, b| {
            a.trust()
                .effective_score()
                .partial_cmp(&b.trust().effective_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let content = match best_match {
            Some(entry) if entry.trust().effective_score() > candidate_score => {
                entry.content().to_string()
            }
            _ => candidate.content().to_string(),
        };

        debug!("BestScoreMergeStrategy selected content: \"{content}\".");
        Box::pin(async move { Ok(content) })
    }
}

const MERGE_SYSTEM_PROMPT: &str = "\
You are a memory merge assistant. You are given a set of related memory entries \
and a new candidate entry. Synthesize them into a single, concise, self-contained \
statement that captures all the relevant information.\n\n\
The merged statement MUST:\n\
- Name its subject explicitly (no pronouns, no dangling references)\n\
- Contain only information that is present in the provided inputs\n\
- Be more specific than any single input when the entries are complementary\n\n\
Output ONLY the merged statement as plain text. No JSON, no lists, no explanation, \
no preamble.";

/// LLM-based merge strategy that synthesises the candidate and all matching
/// entries into a single statement via a model call.
///
/// Falls back to the candidate's content if the model returns an empty response.
pub struct LlmMemoryMergeStrategy<P: TextGenerationModelProvider> {
    provider: Arc<P>,
    model: String,
    thinking_mode: Option<ThinkingMode>,
}

impl<P: TextGenerationModelProvider> LlmMemoryMergeStrategy<P> {
    pub fn new(provider: Arc<P>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            thinking_mode: None,
        }
    }

    pub fn with_thinking(mut self, mode: ThinkingMode) -> Self {
        self.thinking_mode = Some(mode);
        self
    }
}

impl<P: TextGenerationModelProvider + Send + Sync, Params: Send + Sync> MemoryMergeStrategy<Params>
    for LlmMemoryMergeStrategy<P>
{
    fn merge<'a>(
        &'a self,
        candidate: &'a MemoryInput,
        matches: &'a [MemoryEntry],
        _params: &'a Params,
    ) -> BoxFuture<'a, Result<String, MemoryExtractionError>> {
        Box::pin(async move {
            let mut parts = Vec::new();
            parts.push(format!("New entry: {}", candidate.content()));
            for (i, entry) in matches.iter().enumerate() {
                parts.push(format!("Existing entry {}: {}", i + 1, entry.content()));
            }
            let prompt = parts.join("\n");

            let mut req = TextGenerationRequest::new(&self.model, &prompt)
                .with_system(MERGE_SYSTEM_PROMPT)
                .with_temperature(0.1);
            if let Some(mode) = self.thinking_mode.clone() {
                req = req.with_thinking(mode);
            }

            let mut stream = Box::pin(self.provider.generate_stream(req));
            let mut merged = String::new();
            while let Some(chunk) = stream.next().await {
                let resp = chunk.map_err(MemoryExtractionError::ModelProvider)?;
                merged.push_str(resp.text());
            }

            let merged = merged.trim().to_string();
            if merged.is_empty() {
                Ok(candidate.content().to_string())
            } else {
                Ok(merged)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use crate::memory::{MemoryTrust, TrustEvidence, store::MemoryInput};
    use crate::model_provider::text_generation::TextGenerationResponse;
    use crate::testing::{
        MockTextGenerationModelProvider, ProviderBehavior, make_extracted_result, make_fact_result,
    };

    use super::{BestScoreMergeStrategy, LlmMemoryMergeStrategy, MemoryMergeStrategy};

    fn done_chunk(text: &str) -> TextGenerationResponse {
        TextGenerationResponse::new_done(text.to_string(), "mock".to_string(), None)
    }

    fn extracted_input(content: &str, confidence: f64) -> MemoryInput {
        MemoryInput::new(
            content.to_string(),
            MemoryTrust::Extracted {
                confidence,
                evidence: TrustEvidence::default(),
            },
            HashMap::new(),
        )
    }

    #[tokio::test]
    async fn test_best_score_returns_candidate_when_no_matches() {
        let candidate = extracted_input("the sky is blue", 0.8);
        let content = BestScoreMergeStrategy
            .merge(&candidate, &[], &())
            .await
            .unwrap();
        assert_eq!(content, "the sky is blue");
    }

    #[tokio::test]
    async fn test_best_score_returns_candidate_when_it_has_highest_score() {
        let candidate = extracted_input("the sky is blue", 0.9);
        // Existing entry has lower confidence.
        let lower = make_extracted_result(uuid::Uuid::new_v4(), "sky is blue", 0.5);
        let content = BestScoreMergeStrategy
            .merge(&candidate, &[lower], &())
            .await
            .unwrap();
        assert_eq!(content, "the sky is blue");
    }

    #[tokio::test]
    async fn test_best_score_returns_match_when_it_has_highest_score() {
        let candidate = extracted_input("sky is blue", 0.5);
        let higher =
            make_extracted_result(uuid::Uuid::new_v4(), "the sky is definitively blue", 0.9);
        let content = BestScoreMergeStrategy
            .merge(&candidate, &[higher], &())
            .await
            .unwrap();
        assert_eq!(content, "the sky is definitively blue");
    }

    #[tokio::test]
    async fn test_best_score_fact_always_wins_over_extracted() {
        // A Fact has effective_score 1.0 — it should always win.
        let candidate = extracted_input("sky is blue", 0.99);
        let fact = make_fact_result(uuid::Uuid::new_v4(), "the sky is unambiguously blue", 1.0);
        let content = BestScoreMergeStrategy
            .merge(&candidate, &[fact], &())
            .await
            .unwrap();
        assert_eq!(content, "the sky is unambiguously blue");
    }

    #[tokio::test]
    async fn test_llm_merge_calls_model_and_returns_trimmed_output() {
        let provider = Arc::new(MockTextGenerationModelProvider::new(
            ProviderBehavior::Stream(vec![done_chunk("  The sky is definitively blue.  ")]),
        ));
        let strategy = LlmMemoryMergeStrategy::new(Arc::clone(&provider), "m");
        let candidate = extracted_input("sky is blue", 0.8);
        let existing =
            make_extracted_result(uuid::Uuid::new_v4(), "the sky has a blue colour", 0.7);

        let content = strategy.merge(&candidate, &[existing], &()).await.unwrap();
        assert_eq!(content, "The sky is definitively blue.");
        assert_eq!(provider.snapshot().request_count, 1);
    }

    #[tokio::test]
    async fn test_llm_merge_falls_back_to_candidate_on_empty_response() {
        let provider = Arc::new(MockTextGenerationModelProvider::new(
            ProviderBehavior::Stream(vec![done_chunk("   ")]),
        ));
        let strategy = LlmMemoryMergeStrategy::new(Arc::clone(&provider), "m");
        let candidate = extracted_input("the sky is blue", 0.8);

        let content = strategy.merge(&candidate, &[], &()).await.unwrap();
        assert_eq!(content, "the sky is blue");
    }
}

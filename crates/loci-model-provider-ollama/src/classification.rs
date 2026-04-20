// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-model-provider-ollama.

use std::sync::Arc;

use futures::future::BoxFuture;
use loci_core::classification::{
    ClassificationError, ClassificationModelProvider, HitClass, parse_hit_class,
};
use loci_core::model_provider::ThinkingMode;
use loci_core::model_provider::text_generation::{
    TextGenerationModelProvider, TextGenerationRequest,
};

const SYSTEM_PROMPT: &str = "\
You are a memory classification assistant. Given a candidate memory entry and an existing memory hit, classify their relationship.

Respond with ONLY a JSON object in this exact format:
{\"class\": \"<value>\"}

Where <value> is exactly one of: \"duplicate\", \"complementary\", \"contradiction\", \"unrelated\"

- duplicate: The hit says essentially the same thing as the candidate
- complementary: The hit adds related information to the candidate
- contradiction: The hit directly contradicts the candidate
- unrelated: The hit is not meaningfully related to the candidate";

/// An [`ClassificationModelProvider`] that delegates to any [`TextGenerationModelProvider`].
///
/// Builds a structured prompt and parses the JSON response returned by the
/// underlying model.
pub struct LlmClassificationModelProvider<P> {
    provider: Arc<P>,
    model: String,
}

impl<P: TextGenerationModelProvider> LlmClassificationModelProvider<P> {
    /// Creates a new [`LlmClassificationModelProvider`].
    pub fn new(provider: Arc<P>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

impl<P: TextGenerationModelProvider + Send + Sync> ClassificationModelProvider
    for LlmClassificationModelProvider<P>
{
    fn classify_hit<'a>(
        &'a self,
        candidate: &'a str,
        hit: &'a str,
    ) -> BoxFuture<'a, Result<HitClass, ClassificationError>> {
        let prompt = format!("Candidate: {candidate}\n\nExisting memory: {hit}");
        let req = TextGenerationRequest::new(self.model.clone(), prompt)
            .with_system(SYSTEM_PROMPT)
            .with_thinking(ThinkingMode::Disabled);
        let provider = Arc::clone(&self.provider);

        Box::pin(async move {
            let response = provider
                .generate(req)
                .await
                .map_err(ClassificationError::ModelProvider)?;

            let text = response.text().to_owned();

            // Strip thinking tokens: find the first `{` in the response.
            let json_str = text
                .find('{')
                .map(|pos| &text[pos..])
                .unwrap_or(text.as_str());

            let parsed: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
                ClassificationError::Parse(format!("failed to parse JSON response: {e}"))
            })?;

            let class_str = parsed["class"].as_str().ok_or_else(|| {
                ClassificationError::Parse(
                    "missing or non-string \"class\" field in response".to_string(),
                )
            })?;

            Ok(parse_hit_class(class_str).unwrap_or_else(|| {
                log::warn!(
                    "classifier returned unknown hit class {class_str:?}; defaulting to Unrelated"
                );
                HitClass::Unrelated
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use loci_core::classification::{ClassificationError, ClassificationModelProvider, HitClass};
    use loci_core::model_provider::common::ModelProviderResult;
    use loci_core::model_provider::error::ModelProviderError;
    use loci_core::model_provider::text_generation::{
        TextGenerationModelProvider, TextGenerationRequest, TextGenerationResponse,
    };
    use pretty_assertions::assert_eq;

    use super::LlmClassificationModelProvider;

    struct MockProvider {
        text: Option<String>,
    }

    impl MockProvider {
        fn returning(text: impl Into<String>) -> Self {
            Self {
                text: Some(text.into()),
            }
        }

        fn timing_out() -> Self {
            Self { text: None }
        }
    }

    impl TextGenerationModelProvider for MockProvider {
        async fn generate(
            &self,
            _req: TextGenerationRequest,
        ) -> ModelProviderResult<TextGenerationResponse> {
            match &self.text {
                Some(text) => Ok(TextGenerationResponse::new_done(
                    text.clone(),
                    "mock".to_string(),
                    None,
                )),
                None => Err(ModelProviderError::Timeout),
            }
        }
    }

    fn provider_with(text: impl Into<String>) -> LlmClassificationModelProvider<MockProvider> {
        LlmClassificationModelProvider::new(Arc::new(MockProvider::returning(text)), "mock-model")
    }

    #[tokio::test]
    async fn test_classify_hit_with_duplicate_response_returns_duplicate() {
        let p = provider_with(r#"{"class": "duplicate"}"#);
        let result = p.classify_hit("a", "b").await;
        assert_eq!(result.unwrap(), HitClass::Duplicate);
    }

    #[tokio::test]
    async fn test_classify_hit_with_uppercase_class_returns_complementary() {
        let p = provider_with(r#"{"class": "COMPLEMENTARY"}"#);
        let result = p.classify_hit("a", "b").await;
        assert_eq!(result.unwrap(), HitClass::Complementary);
    }

    #[tokio::test]
    async fn test_unknown_class_falls_back_to_unrelated() {
        // An unrecognised class label should soft-fail to Unrelated rather
        // than abort the whole extraction pipeline. Parse errors are reserved
        // for responses we cannot read at all.
        let p = provider_with(r#"{"class": "blorp"}"#);
        let result = p.classify_hit("a", "b").await;
        assert_eq!(result.unwrap(), HitClass::Unrelated);
    }

    #[tokio::test]
    async fn test_invalid_json_returns_parse_error() {
        let p = provider_with("not json");
        let result = p.classify_hit("a", "b").await;
        match result {
            Err(ClassificationError::Parse(_)) => {}
            other => panic!("expected Parse error, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_classify_hit_with_think_wrapped_response_returns_correct_class() {
        let p = provider_with("<think>reasoning</think>\n{\"class\": \"contradiction\"}");
        let result = p.classify_hit("a", "b").await;
        assert_eq!(result.unwrap(), HitClass::Contradiction);
    }

    #[tokio::test]
    async fn test_classify_hit_when_provider_errors_returns_model_provider_error() {
        let provider =
            LlmClassificationModelProvider::new(Arc::new(MockProvider::timing_out()), "mock-model");
        let result = provider.classify_hit("a", "b").await;
        match result {
            Err(ClassificationError::ModelProvider(ModelProviderError::Timeout)) => {}
            other => panic!("expected ModelProvider(Timeout) error, got: {other:?}"),
        }
    }
}

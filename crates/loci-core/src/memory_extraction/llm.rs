// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::{collections::HashMap, future::Future, sync::Arc};

use crate::{
    error::MemoryExtractionError,
    memory::{MemoryInput, MemoryTier},
    model_provider::text_generation::{TextGenerationModelProvider, TextGenerationRequest},
};

use super::MemoryExtractionStrategy;

/// Parameters shared by all LLM-based memory extraction strategies.
pub struct LlmMemoryExtractionStrategyParams {
    /// Optional instructions appended to the extraction prompt to guide or
    /// constrain what the model should extract.
    pub guidelines: Option<String>,
    /// Memory tier assigned to every extracted entry.
    pub default_tier: MemoryTier,
    /// Metadata key/value pairs attached to every extracted entry.
    pub metadata: HashMap<String, String>,
    /// If set, at most this many entries are returned (applied both as a
    /// prompt hint and as a hard post-processing cap).
    pub max_entries: Option<usize>,
}

const EXTRACTION_SYSTEM_PROMPT: &str = "\
You are a memory extraction assistant. Extract discrete, self-contained facts, \
preferences, goals, and important details worth remembering from the provided text.\n\n\
Output ONLY a valid JSON array of strings. Each string is one memory entry. \
Be concise but complete. If nothing is worth remembering, return an empty array.\n\n\
Example output:\n\
[\"The user prefers dark mode.\", \"The project targets Rust stable.\", \"Deadline is end of Q3.\"]";

fn build_extraction_prompt(input: &str, params: &LlmMemoryExtractionStrategyParams) -> String {
    let mut parts: Vec<String> = Vec::new();

    if let Some(guidelines) = &params.guidelines {
        parts.push(format!("Additional guidelines: {guidelines}"));
    }
    if let Some(max) = params.max_entries {
        parts.push(format!("Extract at most {max} entries."));
    }

    parts.push("Text to extract memories from:".to_string());
    parts.push(input.to_string());

    parts.join("\n\n")
}

pub(super) fn parse_extraction_response(
    response: &str,
    params: LlmMemoryExtractionStrategyParams,
) -> Result<Vec<MemoryInput>, MemoryExtractionError> {
    let start = response.find('[').ok_or_else(|| {
        MemoryExtractionError::Parse("no JSON array found in model response".to_string())
    })?;
    let end = response.rfind(']').ok_or_else(|| {
        MemoryExtractionError::Parse("no closing bracket found in model response".to_string())
    })?;

    if end < start {
        return Err(MemoryExtractionError::Parse(
            "malformed JSON array in model response".to_string(),
        ));
    }

    let json_slice = &response[start..=end];
    let entries: Vec<String> = serde_json::from_str(json_slice).map_err(|e| {
        MemoryExtractionError::Parse(format!("failed to parse model response as JSON array: {e}"))
    })?;

    let entries = match params.max_entries {
        Some(max) => entries.into_iter().take(max).collect::<Vec<_>>(),
        None => entries,
    };

    Ok(entries
        .into_iter()
        .map(|content| MemoryInput {
            content,
            metadata: params.metadata.clone(),
            tier: Some(params.default_tier),
        })
        .collect())
}

/// LLM-based memory extraction strategy that works with any
/// [`TextGenerationModelProvider`].
///
/// The provider supplies the model connection; this type owns the prompt
/// engineering and response-parsing logic that is identical across all LLM
/// backends.
pub struct LlmMemoryExtractionStrategy<P: TextGenerationModelProvider> {
    provider: Arc<P>,
    /// Model identifier forwarded verbatim to the provider
    /// (e.g. `"llama3.2"`, `"gpt-4o"`, `"claude-opus-4-5"`).
    model: String,
}

impl<P: TextGenerationModelProvider> LlmMemoryExtractionStrategy<P> {
    /// Creates a new strategy backed by `provider` using `model`.
    pub fn new(provider: Arc<P>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

impl<P: TextGenerationModelProvider + Send + Sync>
    MemoryExtractionStrategy<LlmMemoryExtractionStrategyParams> for LlmMemoryExtractionStrategy<P>
{
    fn extract(
        &self,
        input: &str,
        params: LlmMemoryExtractionStrategyParams,
    ) -> impl Future<Output = Result<Vec<MemoryInput>, MemoryExtractionError>> + Send {
        let prompt = build_extraction_prompt(input, &params);
        let req = TextGenerationRequest::new(self.model.clone(), prompt)
            .with_system(EXTRACTION_SYSTEM_PROMPT)
            .with_temperature(0.1); // Low temperature for deterministic, structured output.
        let provider = Arc::clone(&self.provider);

        async move {
            let response = provider
                .generate(req)
                .await
                .map_err(MemoryExtractionError::ModelProvider)?;

            parse_extraction_response(&response.text, params)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::memory::MemoryTier;

    use super::{LlmMemoryExtractionStrategyParams, parse_extraction_response};

    fn default_params() -> LlmMemoryExtractionStrategyParams {
        LlmMemoryExtractionStrategyParams {
            guidelines: None,
            default_tier: MemoryTier::Candidate,
            metadata: HashMap::new(),
            max_entries: None,
        }
    }

    #[test]
    fn test_parse_clean_json_array() {
        let response = r#"["fact one", "fact two", "fact three"]"#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].content, "fact one");
        assert_eq!(result[2].content, "fact three");
    }

    #[test]
    fn test_parse_json_embedded_in_prose() {
        let response =
            r#"Here are the extracted memories: ["fact one", "fact two"] Hope that helps!"#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].content, "fact one");
    }

    #[test]
    fn test_parse_empty_array() {
        let response = "[]";
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_applies_default_tier() {
        let params = LlmMemoryExtractionStrategyParams {
            default_tier: MemoryTier::Core,
            ..default_params()
        };
        let response = r#"["a fact"]"#;
        let result = parse_extraction_response(response, params).unwrap();
        assert_eq!(result[0].tier, Some(MemoryTier::Core));
    }

    #[test]
    fn test_parse_applies_metadata_to_all_entries() {
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "test".to_string());
        let params = LlmMemoryExtractionStrategyParams {
            metadata: meta,
            ..default_params()
        };
        let response = r#"["fact one", "fact two"]"#;
        let result = parse_extraction_response(response, params).unwrap();
        assert_eq!(
            result[0].metadata.get("source").map(|s| s.as_str()),
            Some("test")
        );
        assert_eq!(
            result[1].metadata.get("source").map(|s| s.as_str()),
            Some("test")
        );
    }

    #[test]
    fn test_parse_honours_max_entries_cap() {
        let params = LlmMemoryExtractionStrategyParams {
            max_entries: Some(2),
            ..default_params()
        };
        let response = r#"["one", "two", "three", "four"]"#;
        let result = parse_extraction_response(response, params).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parse_error_on_missing_array() {
        let response = "nothing to see here";
        assert!(parse_extraction_response(response, default_params()).is_err());
    }

    #[test]
    fn test_parse_error_on_non_string_array() {
        let response = r#"[1, 2, 3]"#;
        assert!(parse_extraction_response(response, default_params()).is_err());
    }
}

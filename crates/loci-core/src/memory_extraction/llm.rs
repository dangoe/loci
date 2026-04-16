// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::{collections::HashMap, future::Future, sync::Arc};

use futures::StreamExt as _;

use serde::Deserialize as _;

use crate::{
    error::MemoryExtractionError,
    memory::{MemoryInput, MemoryTier},
    model_provider::text_generation::{
        TextGenerationModelProvider, TextGenerationRequest, ThinkingMode,
    },
};

use super::MemoryExtractionStrategy;

/// Optional chunking configuration for LLM-based extraction strategies.
#[derive(Clone)]
pub struct ChunkingConfig {
    /// If set, the input text is processed in non-overlapping chunks of this
    /// size (in characters) to extract entries from long inputs that exceed
    /// the model context window. `None` means no chunking (i.e. one prompt
    /// for the entire input).
    pub chunk_size: Option<usize>,
    /// If set, the input text is processed in overlapping chunks of this size
    /// (in characters) to extract more entries from long inputs. The overlap
    /// helps ensure that entries spanning chunk boundaries
    pub overlap_size: Option<usize>,
}

/// Parameters shared by all LLM-based memory extraction strategies.
#[derive(Clone)]
pub struct LlmMemoryExtractionStrategyParams {
    /// Optional instructions appended to the extraction prompt to guide or
    /// constrain what the model should extract.
    pub guidelines: Option<String>,
    /// Metadata key/value pairs attached to every extracted entry.
    pub metadata: HashMap<String, String>,
    /// If set, at most this many entries are returned (applied both as a
    /// prompt hint and as a hard post-processing cap, after confidence filtering).
    pub max_entries: Option<usize>,
    /// If set, entries whose LLM-assigned confidence score is below this
    /// threshold are discarded before storing. In [0.0, 1.0].
    pub min_confidence: Option<f64>,
    /// Thinking mode forwarded to the model provider.  Extraction produces
    /// structured JSON output, so thinking is rarely beneficial; callers
    /// should pass [`ThinkingMode::Disabled`] explicitly unless they have a
    /// specific reason to enable it.  `None` defers to the provider default.
    pub thinking_mode: Option<ThinkingMode>,
    /// Optional chunking configuration for long inputs that may exceed the model
    /// context window. If set, the input is split into chunks according to the
    /// configuration and each chunk is processed separately, with results aggregated
    /// into a single output list.
    pub chunking: Option<ChunkingConfig>,
}

const EXTRACTION_SYSTEM_PROMPT: &str = "\
You are a memory extraction assistant. Extract ONLY information that is:\n
- Directly stated facts about a specific, named subject\n
- User preferences, goals, or decisions\n
- Deadlines, constraints, or requirements\n\n
Do NOT extract:\n
- General knowledge or encyclopedia-style facts\n
- Information that can be looked up anywhere\n
- Vague or context-free statements\n\n
Output ONLY a valid JSON array of objects. Each object must have:\n
  \"content\": a string naming its subject explicitly\n
  \"confidence\": a float 0.0-1.0 indicating how confident you are that this is worth remembering\n\n
Example output:\n
[{\"content\": \"The user prefers dark mode.\", \"confidence\": 0.95}, {\"content\": \"Project X targets Rust stable.\", \"confidence\": 0.80}]";

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

    // Use a streaming deserializer so that any prose the model appends after
    // the JSON array (e.g. footnotes) is silently ignored rather than causing
    // a "trailing characters" parse error.
    let mut de = serde_json::Deserializer::from_str(&response[start..]);
    let raw: Vec<serde_json::Value> =
        Vec::deserialize(&mut de).map_err(|e| {
            MemoryExtractionError::Parse(format!(
                "failed to parse model response as JSON array: {e}"
            ))
        })?;

    let mut entries: Vec<(String, f64)> = raw
        .into_iter()
        .filter_map(|v| {
            let content = v.get("content")?.as_str()?.to_owned();
            let confidence = v.get("confidence").and_then(|c| c.as_f64()).unwrap_or(1.0);
            Some((content, confidence))
        })
        .collect();

    if let Some(min) = params.min_confidence {
        entries.retain(|(_, confidence)| *confidence >= min);
    }

    let entries = match params.max_entries {
        Some(max) => entries.into_iter().take(max).collect::<Vec<_>>(),
        None => entries,
    };

    Ok(entries
        .into_iter()
        .map(|(content, confidence)| MemoryInput {
            content,
            metadata: params.metadata.clone(),
            tier: Some(MemoryTier::Stable),
            confidence: Some(confidence),
            review: Default::default(),
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

        let req = match params.thinking_mode.clone() {
            Some(mode) => req.with_thinking(mode),
            None => req,
        };

        async move {
            // Use generate_stream so the connection stays alive while the model
            // generates (important for thinking-capable models like qwen3 that
            // can produce very long outputs before the JSON array).
            let mut stream = Box::pin(provider.generate_stream(req));
            let mut full_text = String::new();
            while let Some(chunk) = stream.next().await {
                let resp = chunk.map_err(MemoryExtractionError::ModelProvider)?;
                full_text.push_str(&resp.text);
            }

            parse_extraction_response(&full_text, params)
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
            metadata: HashMap::new(),
            max_entries: None,
            min_confidence: None,
            chunking: None,
            thinking_mode: None,
        }
    }

    #[test]
    fn test_parse_clean_json_array() {
        let response = r#"[{"content": "fact one", "confidence": 0.9}, {"content": "fact two", "confidence": 0.8}, {"content": "fact three", "confidence": 0.7}]"#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].content, "fact one");
        assert_eq!(result[2].content, "fact three");
    }

    #[test]
    fn test_parse_json_embedded_in_prose() {
        let response = r#"Here are the extracted memories: [{"content": "fact one", "confidence": 0.9}, {"content": "fact two", "confidence": 0.8}] Hope that helps!"#;
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
    fn test_parse_hardcodes_stable_tier() {
        let response = r#"[{"content": "a fact", "confidence": 0.9}]"#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert_eq!(result[0].tier, Some(MemoryTier::Stable));
    }

    #[test]
    fn test_parse_stores_confidence_on_entry() {
        let response = r#"[{"content": "a fact", "confidence": 0.85}]"#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert_eq!(result[0].confidence, Some(0.85));
    }

    #[test]
    fn test_parse_missing_confidence_defaults_to_1_0() {
        let response = r#"[{"content": "a fact"}]"#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert_eq!(result[0].confidence, Some(1.0));
    }

    #[test]
    fn test_parse_applies_metadata_to_all_entries() {
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "test".to_string());
        let params = LlmMemoryExtractionStrategyParams {
            metadata: meta,
            ..default_params()
        };
        let response = r#"[{"content": "fact one", "confidence": 0.9}, {"content": "fact two", "confidence": 0.8}]"#;
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
        let response = r#"[{"content": "one", "confidence": 0.9}, {"content": "two", "confidence": 0.8}, {"content": "three", "confidence": 0.7}, {"content": "four", "confidence": 0.6}]"#;
        let result = parse_extraction_response(response, params).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parse_filters_below_min_confidence() {
        let params = LlmMemoryExtractionStrategyParams {
            min_confidence: Some(0.75),
            ..default_params()
        };
        let response = r#"[{"content": "high", "confidence": 0.9}, {"content": "low", "confidence": 0.5}, {"content": "borderline", "confidence": 0.75}]"#;
        let result = parse_extraction_response(response, params).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].content, "high");
        assert_eq!(result[1].content, "borderline");
    }

    #[test]
    fn test_parse_max_entries_applied_after_confidence_filter() {
        let params = LlmMemoryExtractionStrategyParams {
            min_confidence: Some(0.6),
            max_entries: Some(1),
            ..default_params()
        };
        let response = r#"[{"content": "good", "confidence": 0.9}, {"content": "also good", "confidence": 0.8}, {"content": "bad", "confidence": 0.3}]"#;
        let result = parse_extraction_response(response, params).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "good");
    }

    #[test]
    fn test_parse_error_on_missing_array() {
        let response = "nothing to see here";
        assert!(parse_extraction_response(response, default_params()).is_err());
    }

    #[test]
    fn test_parse_skips_objects_without_content_field() {
        let response = r#"[{"confidence": 0.9}, {"content": "valid", "confidence": 0.8}]"#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "valid");
    }

    #[test]
    fn test_parse_ignores_trailing_prose_after_json_array() {
        // Models sometimes append footnotes after the JSON; the Deserializer
        // should parse the array and silently ignore the trailing text.
        let response = r#"[{"content": "fact", "confidence": 0.9}]
Note: [confidence values] are rough estimates."#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "fact");
    }
}

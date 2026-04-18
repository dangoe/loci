// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::{collections::HashMap, future::Future, sync::Arc};

use futures::StreamExt as _;

use serde::Deserialize as _;

use crate::{
    error::MemoryExtractionError,
    memory::{MemoryInput, MemoryTrust, TrustEvidence},
    memory_extraction::chunker::split_into_chunks,
    model_provider::text_generation::{
        ResponseFormat, TextGenerationModelProvider, TextGenerationRequest, ThinkingMode,
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
You are a memory extraction assistant. Extract discrete, self-contained facts \
from the provided text that someone would plausibly want to retrieve later.\n\n\
Each extracted entry MUST:\n\
- Be a standalone statement understandable without the source context\n\
- Name its subject explicitly (no pronouns, no dangling references)\n\
- Represent a specific, concrete fact\n\n\
DO NOT extract:\n\
- Trivia with no lasting relevance (weather, meals, moods, greetings, small \
  talk) unless the source text explicitly marks them as important\n\
- Generic advice, platitudes, or opinions\n\
- Procedural UI steps in instructions (\"click X\", \"then Y\") — extract only \
  the resulting facts (e.g. the password policy, the timeout duration)\n\n\
The `confidence` field is EPISTEMIC — it measures how clearly the fact is \
stated in the text, NOT how interesting or useful it is. Use this rubric:\n\
- 0.95-0.99: directly and unambiguously stated (1.0 is reserved for promoted \
  facts; never use it here)\n\
- 0.80-0.94: clearly stated with minor paraphrase\n\
- 0.60-0.79: implied from a few sentences, some interpretation needed\n\
- below 0.60: do NOT output — the inference is too weak\n\n\
Output ONLY a valid JSON array of objects. No prose before or after. \
Each object must have:\n\
  \"content\": string, naming its subject explicitly\n\
  \"confidence\": float in [0.0, 1.0] per the rubric above\n\n\
Example:\n\
[\n\
  {\"content\": \"The CI deployment pipeline failed on 2026-04-10 because the runner's IAM credentials expired.\", \"confidence\": 0.97},\n\
  {\"content\": \"Rania Khalil will add a Conftest rule to fail CI on IAM role expiry.\", \"confidence\": 0.90},\n\
  {\"content\": \"The on-call engineer was paged during the incident.\", \"confidence\": 0.70}\n\
]";

/// Reinforcement appended to the prompt on a parse-failure retry.
const REPAIR_REINFORCEMENT: &str = "\
Your previous output was not a valid JSON array. Output ONLY a JSON array \
starting with `[` and ending with `]`. No prose, no markdown fences, no \
explanation. If there are no facts to extract, output `[]`.";

/// Extracts the entries array from a model response, tolerating several
/// shapes that local models produce under structured-output modes:
///
/// 1. A top-level JSON array: `[{...}, {...}]`.
/// 2. A JSON object that wraps the array under any key, e.g.
///    `{"entries": [...]}`, `{"memories": [...]}`, `{"facts": [...]}`.
/// 3. Prose that contains an embedded JSON array somewhere inside.
///
/// Empty objects / unrelated JSON become an empty entry list — the pipeline
/// treats that as "no facts this run" and moves on. A lone entry object
/// (`{"content": "…"}` at the top level) is **not** promoted to a one-entry
/// list: it's a protocol violation that suggests the model was too eager to
/// stop, so we surface it as a parse error to trigger the retry path.
fn locate_entries_array(response: &str) -> Result<Vec<serde_json::Value>, MemoryExtractionError> {
    let trimmed = response.trim();

    // Case 1: full response parses as a JSON value directly.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return coerce_to_entries(value);
    }

    // Case 2: response has an embedded array — scan from the first `[` and let
    // the streaming deserializer tolerate trailing content.
    if let Some(start) = response.find('[') {
        let mut de = serde_json::Deserializer::from_str(&response[start..]);
        if let Ok(arr) = Vec::<serde_json::Value>::deserialize(&mut de) {
            return Ok(arr);
        }
    }

    // Case 3: response has an embedded object — try to parse it and coerce.
    if let Some(start) = response.find('{') {
        let mut de = serde_json::Deserializer::from_str(&response[start..]);
        if let Ok(value) = serde_json::Value::deserialize(&mut de) {
            return coerce_to_entries(value);
        }
    }

    Err(MemoryExtractionError::Parse(
        "no JSON array or entry object found in model response".to_string(),
    ))
}

/// Given a parsed JSON value, returns the list of entry objects.
///
/// - Arrays are returned as-is.
/// - Objects with an array-valued field (any name) surface that array.
/// - Objects that are a lone entry (have a top-level `content` field) are
///   rejected so the retry path kicks in and forces a proper array.
/// - Anything else yields an empty list.
fn coerce_to_entries(
    value: serde_json::Value,
) -> Result<Vec<serde_json::Value>, MemoryExtractionError> {
    match value {
        serde_json::Value::Array(arr) => Ok(arr),
        serde_json::Value::Object(map) => {
            for (_, v) in &map {
                if let serde_json::Value::Array(arr) = v {
                    return Ok(arr.clone());
                }
            }
            if map.contains_key("content") {
                return Err(MemoryExtractionError::Parse(
                    "model returned a single entry object instead of an array".to_string(),
                ));
            }
            Ok(Vec::new())
        }
        _ => Ok(Vec::new()),
    }
}

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
    let raw = locate_entries_array(response)?;

    let mut entries: Vec<(String, f64)> = raw
        .into_iter()
        .filter_map(|v| {
            let content = v.get("content")?.as_str()?.to_owned();
            let confidence = v
                .get("confidence")
                .and_then(|c| c.as_f64())
                .map(MemoryTrust::clamp_confidence)
                .unwrap_or(0.5);
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
            trust: Some(MemoryTrust::Extracted {
                confidence,
                evidence: TrustEvidence::default(),
            }),
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
        let provider = Arc::clone(&self.provider);
        let model = self.model.clone();
        let input = input.to_owned();

        async move {
            let chunks = chunks_for(&input, params.chunking.as_ref());
            let total = chunks.len();

            let mut all_entries: Vec<MemoryInput> = Vec::new();
            for (idx, chunk) in chunks.into_iter().enumerate() {
                match extract_one_chunk(provider.as_ref(), &model, &chunk, params.clone()).await {
                    Ok(entries) => all_entries.extend(entries),
                    // Parse failures are the small-model-output problem, not a
                    // hard failure of the whole extraction. Drop the chunk so
                    // we keep whatever we already extracted from siblings.
                    Err(MemoryExtractionError::Parse(msg)) => {
                        log::warn!(
                            "skipping chunk {}/{} after parse error: {msg}",
                            idx + 1,
                            total,
                        );
                    }
                    Err(other) => return Err(other),
                }
            }

            Ok(all_entries)
        }
    }
}

/// Splits `input` into chunks honouring the caller's [`ChunkingConfig`]. When
/// no config is supplied, returns the entire input as a single chunk.
fn chunks_for(input: &str, config: Option<&ChunkingConfig>) -> Vec<String> {
    let chunk_size = config.and_then(|c| c.chunk_size).unwrap_or(0);
    if chunk_size == 0 {
        return vec![input.to_owned()];
    }
    let overlap = config.and_then(|c| c.overlap_size).unwrap_or(0);
    split_into_chunks(input, chunk_size, overlap)
}

/// Runs a single extraction round for one chunk. On parse failure, retries
/// once with an explicit reinforcement message.
///
/// JSON mode is intentionally disabled: provider-enforced JSON sometimes
/// coerces small models into emitting a bare object (`{"response": "..."}` or
/// `{}`) rather than the JSON array the system prompt requests. Relying on the
/// text instruction alone produces correct arrays from the models tested.
async fn extract_one_chunk<P: TextGenerationModelProvider>(
    provider: &P,
    model: &str,
    chunk: &str,
    params: LlmMemoryExtractionStrategyParams,
) -> Result<Vec<MemoryInput>, MemoryExtractionError> {
    let prompt = build_extraction_prompt(chunk, &params);
    let first = call_model(provider, model, &prompt, &params, false).await?;
    match parse_extraction_response(&first, params.clone()) {
        Ok(entries) => Ok(entries),
        Err(MemoryExtractionError::Parse(_)) => {
            let repair_prompt = format!("{prompt}\n\n{REPAIR_REINFORCEMENT}");
            let second = call_model(provider, model, &repair_prompt, &params, false).await?;
            parse_extraction_response(&second, params)
        }
        Err(e) => Err(e),
    }
}

/// Issues one streaming generate call and returns the concatenated text.
/// `json_mode` toggles provider-side structured-output enforcement.
async fn call_model<P: TextGenerationModelProvider>(
    provider: &P,
    model: &str,
    prompt: &str,
    params: &LlmMemoryExtractionStrategyParams,
    json_mode: bool,
) -> Result<String, MemoryExtractionError> {
    let mut req = TextGenerationRequest::new(model, prompt)
        .with_system(EXTRACTION_SYSTEM_PROMPT)
        .with_temperature(0.1);
    if json_mode {
        req = req.with_response_format(ResponseFormat::Json);
    }
    if let Some(mode) = params.thinking_mode.clone() {
        req = req.with_thinking(mode);
    }

    // Streaming keeps the connection alive for thinking-capable models that
    // emit long prefixes before the JSON array.
    let mut stream = Box::pin(provider.generate_stream(req));
    let mut full_text = String::new();
    while let Some(chunk) = stream.next().await {
        let resp = chunk.map_err(MemoryExtractionError::ModelProvider)?;
        full_text.push_str(&resp.text);
    }
    Ok(full_text)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use crate::memory::MemoryTrust;
    use crate::memory_extraction::MemoryExtractionStrategy;
    use crate::model_provider::text_generation::TextGenerationResponse;
    use crate::testing::{MockTextGenerationModelProvider, ProviderBehavior};

    use super::{
        ChunkingConfig, LlmMemoryExtractionStrategy, LlmMemoryExtractionStrategyParams,
        parse_extraction_response,
    };

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

    fn done_chunk(text: &str) -> TextGenerationResponse {
        TextGenerationResponse::done(text.to_string(), "mock".to_string(), None)
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
    fn test_parse_uses_extracted_memory_kind() {
        let response = r#"[{"content": "a fact", "confidence": 0.9}]"#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert!(matches!(
            result[0].trust,
            Some(MemoryTrust::Extracted { .. })
        ));
    }

    #[test]
    fn test_parse_stores_confidence_on_entry() {
        let response = r#"[{"content": "a fact", "confidence": 0.85}]"#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert!(matches!(
            result[0].trust,
            Some(MemoryTrust::Extracted { confidence, .. }) if (confidence - 0.85).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn test_parse_missing_confidence_defaults_to_0_5() {
        let response = r#"[{"content": "a fact"}]"#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert!(matches!(
            result[0].trust,
            Some(MemoryTrust::Extracted { confidence, .. }) if (confidence - 0.5).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn test_parse_clamps_confidence_at_boundary() {
        let response = r#"[{"content": "fact at one", "confidence": 1.0}, {"content": "fact at zero", "confidence": 0.0}]"#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        let conf0 = match result[0].trust {
            Some(MemoryTrust::Extracted { confidence, .. }) => confidence,
            _ => panic!("expected Extracted"),
        };
        let conf1 = match result[1].trust {
            Some(MemoryTrust::Extracted { confidence, .. }) => confidence,
            _ => panic!("expected Extracted"),
        };
        assert!(conf0 < 1.0);
        assert!(conf1 > 0.0);
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

    #[tokio::test]
    async fn test_extract_retries_once_on_parse_failure() {
        // First call returns prose (no JSON array); second call returns valid JSON.
        let provider = Arc::new(MockTextGenerationModelProvider::new(
            ProviderBehavior::Sequence(vec![
                vec![done_chunk("Sorry, I cannot produce JSON here.")],
                vec![done_chunk(
                    r#"[{"content": "recovered fact", "confidence": 0.9}]"#,
                )],
            ]),
        ));
        let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), "m");

        let result = strategy.extract("input", default_params()).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "recovered fact");
        assert_eq!(provider.snapshot().request_count, 2);
    }

    #[tokio::test]
    async fn test_extract_drops_chunk_after_retry_also_fails() {
        // Both calls return prose with no JSON array. The chunk is dropped
        // and `extract` returns an empty list rather than aborting the whole
        // run — long inputs with many chunks shouldn't lose good sibling
        // chunks because one went sideways.
        let provider = Arc::new(MockTextGenerationModelProvider::new(
            ProviderBehavior::Sequence(vec![
                vec![done_chunk("no json here")],
                vec![done_chunk("still no json")],
            ]),
        ));
        let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), "m");

        let result = strategy.extract("input", default_params()).await.unwrap();
        assert!(result.is_empty());
        assert_eq!(provider.snapshot().request_count, 2);
    }

    #[tokio::test]
    async fn test_extract_does_not_retry_when_first_call_parses() {
        let provider = Arc::new(MockTextGenerationModelProvider::new(
            ProviderBehavior::Sequence(vec![vec![done_chunk(
                r#"[{"content": "ok", "confidence": 0.9}]"#,
            )]]),
        ));
        let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), "m");

        let result = strategy.extract("input", default_params()).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(provider.snapshot().request_count, 1);
    }

    #[tokio::test]
    async fn test_extract_chunks_input_when_config_set() {
        // Two separate JSON responses, one per chunk, concatenated.
        let provider = Arc::new(MockTextGenerationModelProvider::new(
            ProviderBehavior::Sequence(vec![
                vec![done_chunk(
                    r#"[{"content": "from chunk one", "confidence": 0.9}]"#,
                )],
                vec![done_chunk(
                    r#"[{"content": "from chunk two", "confidence": 0.8}]"#,
                )],
            ]),
        ));
        let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), "m");

        // ~500 chars of distinct sentences so split_into_chunks yields ≥ 2 chunks.
        let sentence = "The cat sat on the mat in the quiet library room. ";
        let input = sentence.repeat(10);
        let params = LlmMemoryExtractionStrategyParams {
            chunking: Some(ChunkingConfig {
                chunk_size: Some(200),
                overlap_size: Some(0),
            }),
            ..default_params()
        };

        let result = strategy.extract(&input, params).await.unwrap();

        assert!(
            provider.snapshot().request_count >= 2,
            "expected at least 2 model calls, got {}",
            provider.snapshot().request_count
        );
        let contents: Vec<&str> = result.iter().map(|e| e.content.as_str()).collect();
        assert!(contents.contains(&"from chunk one"));
        assert!(contents.contains(&"from chunk two"));
    }

    #[tokio::test]
    async fn test_extract_without_chunking_issues_single_call() {
        let provider = Arc::new(MockTextGenerationModelProvider::new(
            ProviderBehavior::Stream(vec![done_chunk(
                r#"[{"content": "one", "confidence": 0.9}]"#,
            )]),
        ));
        let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), "m");

        strategy
            .extract("some long text", default_params())
            .await
            .unwrap();

        assert_eq!(provider.snapshot().request_count, 1);
    }

    #[test]
    fn test_parse_object_wrapped_array_key_entries() {
        let response = r#"{"entries": [{"content": "fact", "confidence": 0.9}]}"#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "fact");
    }

    #[test]
    fn test_parse_object_wrapped_array_arbitrary_key() {
        // Key name is not one we special-case — first array-valued field wins.
        let response = r#"{"whatever_key": [{"content": "x", "confidence": 0.8}]}"#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "x");
    }

    #[test]
    fn test_parse_single_object_errors_to_trigger_retry() {
        // Model returned a bare entry object instead of a list — this is a
        // protocol violation that the retry path should fix by forcing the
        // model to emit a real array.
        let response = r#"{"content": "lonely fact", "confidence": 0.85}"#;
        let err = parse_extraction_response(response, default_params())
            .expect_err("bare entry object should error out");
        assert!(matches!(err, crate::error::MemoryExtractionError::Parse(_)));
    }

    #[test]
    fn test_parse_empty_object_yields_empty_list() {
        // `{}` under strict JSON mode → treat as "no facts this run", not error.
        let result = parse_extraction_response("{}", default_params()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_unrelated_json_object_yields_empty_list() {
        let response = r#"{"status": "done", "count": 0}"#;
        let result = parse_extraction_response(response, default_params()).unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_extract_retries_on_single_object_response() {
        // First call returns a bare entry object (JSON-mode artefact); the
        // retry path should fire and return the real array.
        let provider = Arc::new(MockTextGenerationModelProvider::new(
            ProviderBehavior::Sequence(vec![
                vec![done_chunk(r#"{"content": "single", "confidence": 0.9}"#)],
                vec![done_chunk(
                    r#"[{"content": "one", "confidence": 0.9}, {"content": "two", "confidence": 0.8}]"#,
                )],
            ]),
        ));
        let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), "m");

        let result = strategy.extract("input", default_params()).await.unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(provider.snapshot().request_count, 2);
    }

    #[tokio::test]
    async fn test_retry_disables_json_mode() {
        let provider = Arc::new(MockTextGenerationModelProvider::new(
            ProviderBehavior::Sequence(vec![
                vec![done_chunk("no json at all")],
                vec![done_chunk(r#"[{"content": "ok", "confidence": 0.9}]"#)],
            ]),
        ));
        let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), "m");

        strategy.extract("input", default_params()).await.unwrap();

        // Only the last request is captured; it must be the retry, which must
        // have JSON mode turned off.
        let snapshot = provider.snapshot();
        assert_eq!(snapshot.request_count, 2);
        let last = snapshot.last_request.expect("last request captured");
        assert!(
            last.response_format.is_none(),
            "retry call must drop response_format to let the model produce text"
        );
    }

    #[tokio::test]
    async fn test_extract_does_not_set_json_response_format_on_request() {
        // JSON mode is deliberately disabled: provider-enforced JSON causes
        // small models to emit objects (`{}`) instead of arrays.  The text
        // instruction in the system prompt is sufficient.
        let provider = Arc::new(MockTextGenerationModelProvider::new(
            ProviderBehavior::Stream(vec![done_chunk(r#"[{"content": "x", "confidence": 0.9}]"#)]),
        ));
        let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), "m");
        strategy.extract("input", default_params()).await.unwrap();

        let req = provider.snapshot().last_request.expect("request captured");
        assert!(
            req.response_format.is_none(),
            "extraction must not set response_format — JSON mode breaks small models"
        );
    }
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

use std::{
    collections::HashMap,
    error::Error as StdError,
    io::{IsTerminal, Write},
    marker::PhantomData,
    sync::Arc,
};

use loci_config::MemoryExtractionConfig;
use loci_core::{
    memory::{
        MemoryInput as CoreMemoryInput, MemoryQuery as CoreMemoryQuery,
        MemoryQueryMode as CoreMemoryQueryMode, MemoryTrust, Score as CoreScore,
    },
    memory_extraction::{
        LlmMemoryExtractionStrategy, LlmMemoryExtractionStrategyParams, MemoryExtractionStrategy,
        MemoryExtractor, MemoryExtractorConfig, MemoryExtractorSearchResultsConfig,
        llm::ChunkingConfig as CoreChunkingConfig,
    },
    model_provider::text_generation::TextGenerationModelProvider,
    store::MemoryStore as CoreMemoryStore,
};
use loci_model_provider_ollama::classification::LlmClassificationModelProvider;
use log::debug;

use crate::{
    commands::{
        input::read_extraction_input,
        memory::{MemoryCommand, MemoryKind},
    },
    handlers::{CommandHandler, json::entry_to_json, mapping::model_thinking_to_core},
};

impl From<MemoryKind> for MemoryTrust {
    fn from(val: MemoryKind) -> Self {
        match val {
            MemoryKind::Fact => MemoryTrust::Fact,
            MemoryKind::ExtractedMemory => MemoryTrust::Extracted {
                confidence: 0.5,
                evidence: Default::default(),
            },
        }
    }
}

pub struct MemoryCommandHandler<'a, S: CoreMemoryStore, P: TextGenerationModelProvider> {
    store: Arc<S>,
    provider: Arc<P>,
    text_model: String,
    extraction_config: MemoryExtractionConfig,
    /// Ties the `'a` lifetime used in `CommandHandler<'a, …>` to this struct
    /// so that Rust can prove `'a: '_` when borrowing `&'_ Self` — identical to
    /// the pattern used in `GenerateCommandHandler`.
    _marker: PhantomData<&'a ()>,
}

impl<'a, S: CoreMemoryStore, P: TextGenerationModelProvider> MemoryCommandHandler<'a, S, P> {
    pub fn new(
        store: Arc<S>,
        provider: Arc<P>,
        text_model: impl Into<String>,
        extraction_config: MemoryExtractionConfig,
    ) -> Self {
        Self {
            store,
            provider,
            text_model: text_model.into(),
            extraction_config,
            _marker: PhantomData,
        }
    }
}

impl<'a, S, P, W> CommandHandler<'a, MemoryCommand, W> for MemoryCommandHandler<'a, S, P>
where
    S: CoreMemoryStore + 'static,
    P: TextGenerationModelProvider + Send + Sync + 'static,
    W: Write + Send,
{
    async fn handle(&self, command: MemoryCommand, out: &mut W) -> Result<(), Box<dyn StdError>> {
        match command {
            MemoryCommand::Add {
                content,
                metadata,
                kind,
            } => {
                debug!(
                    "add memory entry: content={content}, metadata={:?}, kind={:?}",
                    metadata, kind
                );
                let input = match kind {
                    Some(kind) => CoreMemoryInput::new_with_trust(
                        content,
                        pairs_to_map(metadata),
                        kind.into(),
                    ),
                    None => CoreMemoryInput::new(content, pairs_to_map(metadata)),
                };
                let entry = self.store.add_entry(input).await?;
                writeln!(
                    out,
                    "{}",
                    serde_json::to_string_pretty(&entry_to_json(&entry))?
                )?;
            }
            MemoryCommand::Query {
                topic,
                max_results,
                min_score,
                filters,
            } => {
                debug!(
                    "query memory: topic={topic}, max_results={max_results}, min_score={min_score}, filters={:?}",
                    filters
                );
                let query = CoreMemoryQuery {
                    topic,
                    max_results,
                    min_score: CoreScore::new(min_score)
                        .map_err(|e| format!("invalid min_score: {e}"))?,
                    filters: pairs_to_map(filters),
                    mode: CoreMemoryQueryMode::Lookup,
                };
                let entries = self.store.query(query).await?;
                let json: Vec<_> = entries.iter().map(entry_to_json).collect();
                writeln!(out, "{}", serde_json::to_string_pretty(&json)?)?;
            }
            MemoryCommand::Get { id } => {
                debug!("get memory entry: id={id}");
                let entry = self.store.get_entry(id).await?;
                writeln!(
                    out,
                    "{}",
                    serde_json::to_string_pretty(&entry_to_json(&entry))?
                )?;
            }
            MemoryCommand::Promote { id } => {
                debug!("promote memory entry to Fact: id={id}");
                let entry = self.store.set_entry_trust(id, MemoryTrust::Fact).await?;
                writeln!(
                    out,
                    "{}",
                    serde_json::to_string_pretty(&entry_to_json(&entry))?
                )?;
            }
            MemoryCommand::Delete { id } => {
                debug!("delete memory entry: id={id}");
                self.store.delete_entry(id).await?;
                writeln!(
                    out,
                    "{}",
                    serde_json::to_string_pretty(
                        &serde_json::json!({ "deleted": id.to_string() })
                    )?
                )?;
            }
            MemoryCommand::PruneExpired => {
                debug!("prune expired memory entries");
                self.store.prune_expired().await?;
                writeln!(
                    out,
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({ "expired pruned": true }))?
                )?;
            }
            MemoryCommand::Extract {
                text,
                files,
                metadata,
                max_entries,
                min_confidence,
                guidelines,
                dry_run,
            } => {
                debug!("extract memories: dry_run={dry_run}");
                if text.is_none() && files.is_empty() && std::io::stdin().is_terminal() {
                    return Err(
                        "no input provided: pass text as an argument, pipe via stdin, or use --file"
                            .into(),
                    );
                }
                let input = read_extraction_input(text, &files, std::io::stdin())?;

                let params = LlmMemoryExtractionStrategyParams {
                    guidelines: match (guidelines, &self.extraction_config.guidelines) {
                        (Some(cli), Some(cfg)) => Some(format!("{cfg}\n\n{cli}")),
                        (Some(cli), None) => Some(cli),
                        (None, Some(cfg)) => Some(cfg.clone()),
                        (None, None) => None,
                    },
                    metadata: pairs_to_map(metadata),
                    max_entries: max_entries.or(self.extraction_config.max_entries),
                    min_confidence: min_confidence.or(self.extraction_config.min_confidence),
                    thinking_mode: self
                        .extraction_config
                        .thinking
                        .as_ref()
                        .map(model_thinking_to_core),
                    chunking: self.extraction_config.chunking.as_ref().map(|c| {
                        CoreChunkingConfig {
                            chunk_size: Some(c.chunk_size),
                            overlap_size: Some(c.overlap_size),
                        }
                    }),
                };

                let strategy = LlmMemoryExtractionStrategy::new(
                    Arc::clone(&self.provider),
                    self.text_model.clone(),
                );

                if dry_run {
                    let entries = strategy
                        .extract(&input, params)
                        .await
                        .map_err(|e| Box::new(e) as Box<dyn StdError>)?;
                    let json: Vec<_> = entries
                        .iter()
                        .map(|e| {
                            serde_json::json!({
                                "content": e.content,
                                "confidence": e.trust.as_ref().map(|t| t.effective_confidence()).unwrap_or(0.5),
                                "kind": e.trust.as_ref().map(|t| t.as_str()).unwrap_or("extracted_memory"),
                                "metadata": e.metadata,
                            })
                        })
                        .collect();
                    writeln!(out, "{}", serde_json::to_string_pretty(&json)?)?;
                } else {
                    let extractor_cfg = &self.extraction_config.extractor;
                    let classification_provider = Arc::new(LlmClassificationModelProvider::new(
                        Arc::clone(&self.provider),
                        extractor_cfg.classification_model.clone(),
                    ));
                    let extractor = MemoryExtractor::new(
                        Arc::clone(&self.store),
                        Arc::new(strategy),
                        classification_provider,
                        config_extractor_config_to_core(extractor_cfg),
                    );
                    let result = extractor
                        .extract_and_store(&input, params)
                        .await
                        .map_err(|e| Box::new(e) as Box<dyn StdError>)?;
                    writeln!(
                        out,
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "inserted": result.inserted.len(),
                            "merged": result.merged.len(),
                            "promoted": result.promoted.len(),
                            "discarded": result.discarded.len(),
                        }))?
                    )?;
                }
            }
        }
        Ok(())
    }
}

fn config_extractor_config_to_core(
    cfg: &loci_config::MemoryExtractorConfig,
) -> MemoryExtractorConfig {
    MemoryExtractorConfig {
        direct_search: MemoryExtractorSearchResultsConfig {
            max_results: cfg.direct_search.max_results,
            min_score: cfg.direct_search.min_score,
        },
        inverted_search: MemoryExtractorSearchResultsConfig {
            max_results: cfg.inverted_search.max_results,
            min_score: cfg.inverted_search.min_score,
        },
        bayesian_seed_weight: cfg.bayesian_seed_weight,
        max_counter_increment: cfg.max_counter_increment,
        max_counter: cfg.max_counter,
        auto_discard_threshold: cfg.auto_discard_threshold,
        auto_promotion_threshold: cfg.auto_promotion_threshold,
        min_alpha_for_promotion: cfg.min_alpha_for_promotion,
    }
}

/// Converts a list of `(key, value)` pairs into a [`HashMap`].
fn pairs_to_map(pairs: Vec<(String, String)>) -> HashMap<String, String> {
    pairs.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use std::{collections::HashMap, sync::Arc};

    use loci_config::{MemoryExtractionConfig, MemoryExtractorConfig, MemoryExtractorSearchResultsConfig};
    use loci_core::{
        memory::{
            MemoryEntry as CoreMemoryEntry, MemoryQueryResult as CoreMemoryQueryResult,
            MemoryTrust, Score as CoreScore, TrustEvidence,
        },
        model_provider::text_generation::TextGenerationResponse,
        testing::{
            AddEntriesBehavior, MockStore, MockTextGenerationModelProvider, ProviderBehavior,
        },
    };
    use serde_json::Value;
    use uuid::Uuid;

    use crate::commands::memory::MemoryKind;
    use crate::handlers::CommandHandler;

    use crate::{
        commands::memory::MemoryCommand,
        handlers::memory::{MemoryCommandHandler, pairs_to_map},
    };

    fn make_handler(
        store: MockStore,
    ) -> MemoryCommandHandler<'static, MockStore, MockTextGenerationModelProvider> {
        MemoryCommandHandler::new(
            Arc::new(store),
            Arc::new(MockTextGenerationModelProvider::ok()),
            "test-model",
            MemoryExtractionConfig {
                model: "test-model".to_string(),
                max_entries: None,
                min_confidence: None,
                guidelines: None,
                thinking: None,
                chunking: None,
                extractor: MemoryExtractorConfig {
                    classification_model: "test-classification-model".to_string(),
                    direct_search: MemoryExtractorSearchResultsConfig { max_results: 5, min_score: 0.70 },
                    inverted_search: MemoryExtractorSearchResultsConfig { max_results: 3, min_score: 0.60 },
                    bayesian_seed_weight: 10.0,
                    max_counter_increment: 5.0,
                    max_counter: 100.0,
                    auto_discard_threshold: 0.1,
                    auto_promotion_threshold: 0.9,
                    min_alpha_for_promotion: 12.0,
                    decay_rate: 0.99,
                },
            },
        )
    }

    fn make_handler_with_provider(
        store: MockStore,
        provider: MockTextGenerationModelProvider,
    ) -> MemoryCommandHandler<'static, MockStore, MockTextGenerationModelProvider> {
        MemoryCommandHandler::new(
            Arc::new(store),
            Arc::new(provider),
            "test-model",
            MemoryExtractionConfig {
                model: "test-model".to_string(),
                max_entries: None,
                min_confidence: None,
                guidelines: None,
                thinking: None,
                chunking: None,
                extractor: MemoryExtractorConfig {
                    classification_model: "test-classification-model".to_string(),
                    direct_search: MemoryExtractorSearchResultsConfig { max_results: 5, min_score: 0.70 },
                    inverted_search: MemoryExtractorSearchResultsConfig { max_results: 3, min_score: 0.60 },
                    bayesian_seed_weight: 10.0,
                    max_counter_increment: 5.0,
                    max_counter: 100.0,
                    auto_discard_threshold: 0.1,
                    auto_promotion_threshold: 0.9,
                    min_alpha_for_promotion: 12.0,
                    decay_rate: 0.99,
                },
            },
        )
    }

    fn make_result(content: &str, trust: MemoryTrust) -> CoreMemoryQueryResult {
        CoreMemoryQueryResult {
            memory_entry: CoreMemoryEntry::new_with_trust(
                content.to_string(),
                HashMap::new(),
                trust,
            ),
            score: CoreScore::ZERO,
        }
    }

    fn make_result_with_metadata(
        content: &str,
        metadata: HashMap<String, String>,
    ) -> CoreMemoryQueryResult {
        CoreMemoryQueryResult {
            memory_entry: CoreMemoryEntry::new(content.to_string(), metadata),
            score: CoreScore::new(0.9).unwrap(),
        }
    }

    fn parse_json_output(buf: &[u8]) -> Value {
        serde_json::from_str(std::str::from_utf8(buf).unwrap().trim()).unwrap()
    }

    #[tokio::test]
    async fn test_memory_add_outputs_json() {
        let entry = make_result(
            "hello world",
            MemoryTrust::Extracted {
                confidence: 0.5,
                evidence: TrustEvidence::default(),
            },
        );
        let id = entry.memory_entry.id;
        let handler = make_handler(MockStore::new().with_add(entry));
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Add {
                    content: "hello world".to_string(),
                    metadata: vec![],
                    kind: None,
                },
                &mut out,
            )
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["id"].as_str().unwrap(), id.to_string().as_str());
        assert_eq!(v["content"].as_str().unwrap(), "hello world");
        assert_eq!(v["kind"].as_str().unwrap(), "extracted_memory");
    }

    #[tokio::test]
    async fn test_memory_add_with_kind_outputs_kind_field() {
        let entry = make_result("core fact", MemoryTrust::Fact);
        let handler = make_handler(MockStore::new().with_add(entry));
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Add {
                    content: "core fact".to_string(),
                    metadata: vec![],
                    kind: Some(MemoryKind::Fact),
                },
                &mut out,
            )
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["kind"].as_str().unwrap(), "fact");
    }

    #[tokio::test]
    async fn test_memory_add_propagates_store_error() {
        let handler = make_handler(MockStore::new());
        let mut out = Vec::new();

        let result = handler
            .handle(
                MemoryCommand::Add {
                    content: "x".to_string(),
                    metadata: vec![],
                    kind: None,
                },
                &mut out,
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_memory_query_outputs_json_array() {
        let entries = vec![
            make_result(
                "first",
                MemoryTrust::Extracted {
                    confidence: 0.5,
                    evidence: TrustEvidence::default(),
                },
            ),
            make_result("second", MemoryTrust::Fact),
        ];
        let handler = make_handler(MockStore::new().with_query(entries));
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Query {
                    topic: "something".to_string(),
                    max_results: 10,
                    min_score: 0.0,
                    filters: vec![],
                },
                &mut out,
            )
            .await
            .unwrap();

        let v = parse_json_output(&out);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["content"].as_str().unwrap(), "first");
        assert_eq!(arr[1]["content"].as_str().unwrap(), "second");
    }

    #[tokio::test]
    async fn test_memory_query_invalid_min_score_returns_err() {
        let handler = make_handler(MockStore::new());
        let mut out = Vec::new();

        let result = handler
            .handle(
                MemoryCommand::Query {
                    topic: "t".to_string(),
                    max_results: 5,
                    min_score: 1.5,
                    filters: vec![],
                },
                &mut out,
            )
            .await;

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid min_score")
        );
    }

    #[tokio::test]
    async fn test_memory_get_outputs_json() {
        let entry = make_result(
            "specific entry",
            MemoryTrust::Extracted {
                confidence: 0.5,
                evidence: TrustEvidence::default(),
            },
        );
        let id = entry.memory_entry.id;
        let handler = make_handler(MockStore::new().with_get(entry));
        let mut out = Vec::new();

        handler
            .handle(MemoryCommand::Get { id }, &mut out)
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["id"].as_str().unwrap(), id.to_string().as_str());
        assert_eq!(v["content"].as_str().unwrap(), "specific entry");
        assert_eq!(v["kind"].as_str().unwrap(), "extracted_memory");
    }

    #[tokio::test]
    async fn test_memory_get_not_found_returns_err() {
        let handler = make_handler(MockStore::new());
        let mut out = Vec::new();

        let result = handler
            .handle(MemoryCommand::Get { id: Uuid::new_v4() }, &mut out)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_memory_promote_outputs_json() {
        let entry = make_result("important fact", MemoryTrust::Fact);
        let id = entry.memory_entry.id;
        let handler = make_handler(MockStore::new().with_set_kind(entry));
        let mut out = Vec::new();

        handler
            .handle(MemoryCommand::Promote { id }, &mut out)
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["id"].as_str().unwrap(), id.to_string().as_str());
        assert_eq!(v["content"].as_str().unwrap(), "important fact");
        assert_eq!(v["kind"].as_str().unwrap(), "fact");
    }

    #[tokio::test]
    async fn test_memory_delete_outputs_deleted_id() {
        let id = Uuid::new_v4();
        let handler = make_handler(MockStore::new());
        let mut out = Vec::new();

        handler
            .handle(MemoryCommand::Delete { id }, &mut out)
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["deleted"].as_str().unwrap(), id.to_string().as_str());
    }

    #[tokio::test]
    async fn test_memory_prune_expired_outputs_success() {
        let handler = make_handler(MockStore::new());
        let mut out = Vec::new();

        handler
            .handle(MemoryCommand::PruneExpired, &mut out)
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["expired pruned"].as_bool().unwrap(), true);
    }

    fn extraction_provider(response_json: &str) -> MockTextGenerationModelProvider {
        MockTextGenerationModelProvider::new(ProviderBehavior::Stream(vec![
            TextGenerationResponse::done(response_json.to_string(), "mock".to_string(), None),
        ]))
    }

    fn make_extract_entries(contents: &[&str]) -> Vec<CoreMemoryQueryResult> {
        contents
            .iter()
            .map(|c| {
                make_result(
                    c,
                    MemoryTrust::Extracted {
                        confidence: 0.5,
                        evidence: TrustEvidence::default(),
                    },
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn test_extract_dry_run_prints_candidates_without_persisting() {
        let provider = extraction_provider(r#"[{"content": "extracted fact", "confidence": 0.9}]"#);
        let store = Arc::new(MockStore::new()); // add_entries not configured → would err if called
        let handler = MemoryCommandHandler::new(
            Arc::clone(&store),
            Arc::new(provider),
            "m",
            MemoryExtractionConfig {
                model: "m".to_string(),
                max_entries: None,
                min_confidence: None,
                guidelines: None,
                thinking: None,
                chunking: None,
                extractor: MemoryExtractorConfig {
                    classification_model: "test-classification-model".to_string(),
                    direct_search: MemoryExtractorSearchResultsConfig { max_results: 5, min_score: 0.70 },
                    inverted_search: MemoryExtractorSearchResultsConfig { max_results: 3, min_score: 0.60 },
                    bayesian_seed_weight: 10.0,
                    max_counter_increment: 5.0,
                    max_counter: 100.0,
                    auto_discard_threshold: 0.1,
                    auto_promotion_threshold: 0.9,
                    min_alpha_for_promotion: 12.0,
                    decay_rate: 0.99,
                },
            },
        );
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Extract {
                    text: Some("some input text".to_string()),
                    files: vec![],
                    metadata: vec![],
                    max_entries: None,
                    min_confidence: None,
                    guidelines: None,
                    dry_run: true,
                },
                &mut out,
            )
            .await
            .unwrap();

        // Nothing written to store
        assert!(
            store.snapshot().add_inputs.is_none(),
            "dry_run must not write to store"
        );

        let v = parse_json_output(&out);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["content"].as_str().unwrap(), "extracted fact");
        assert_eq!(arr[0]["kind"].as_str().unwrap(), "extracted_memory");
        assert!((arr[0]["confidence"].as_f64().unwrap() - 0.9).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_extract_persists_entries_by_default() {
        let provider = extraction_provider(
            r#"[{"content": "fact one", "confidence": 0.9}, {"content": "fact two", "confidence": 0.8}]"#,
        );
        let stored = make_extract_entries(&["fact one", "fact two"]);
        let store =
            Arc::new(MockStore::new().with_add_entries_behavior(AddEntriesBehavior::Ok(stored)));
        let handler = MemoryCommandHandler::new(
            Arc::clone(&store),
            Arc::new(provider),
            "m",
            MemoryExtractionConfig {
                model: "m".to_string(),
                max_entries: None,
                min_confidence: None,
                guidelines: None,
                thinking: None,
                chunking: None,
                extractor: MemoryExtractorConfig {
                    classification_model: "test-classification-model".to_string(),
                    direct_search: MemoryExtractorSearchResultsConfig { max_results: 5, min_score: 0.70 },
                    inverted_search: MemoryExtractorSearchResultsConfig { max_results: 3, min_score: 0.60 },
                    bayesian_seed_weight: 10.0,
                    max_counter_increment: 5.0,
                    max_counter: 100.0,
                    auto_discard_threshold: 0.1,
                    auto_promotion_threshold: 0.9,
                    min_alpha_for_promotion: 12.0,
                    decay_rate: 0.99,
                },
            },
        );
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Extract {
                    text: Some("text with two facts".to_string()),
                    files: vec![],
                    metadata: vec![],
                    max_entries: None,
                    min_confidence: None,
                    guidelines: None,
                    dry_run: false,
                },
                &mut out,
            )
            .await
            .unwrap();

        // Store was written (add_entries was called at least once)
        assert!(
            store.snapshot().add_inputs.is_some(),
            "store should have received add_entries"
        );

        let v = parse_json_output(&out);
        assert!(
            v.get("inserted").is_some(),
            "output should have 'inserted' field"
        );
        assert!(
            v.get("merged").is_some(),
            "output should have 'merged' field"
        );
        assert!(
            v.get("promoted").is_some(),
            "output should have 'promoted' field"
        );
        assert!(
            v.get("discarded").is_some(),
            "output should have 'discarded' field"
        );
    }

    #[tokio::test]
    async fn test_extract_entries_have_extracted_memory_kind() {
        let provider = extraction_provider(r#"[{"content": "a fact", "confidence": 0.9}]"#);
        let stored = make_extract_entries(&["a fact"]);
        let store = MockStore::new().with_add_entries_behavior(AddEntriesBehavior::Ok(stored));
        let handler = make_handler_with_provider(store, provider);
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Extract {
                    text: Some("input".to_string()),
                    files: vec![],
                    metadata: vec![],
                    max_entries: None,
                    min_confidence: None,
                    guidelines: None,
                    dry_run: true,
                },
                &mut out,
            )
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v[0]["kind"].as_str().unwrap(), "extracted_memory");
    }

    #[tokio::test]
    async fn test_extract_propagates_metadata_to_entries() {
        let provider = extraction_provider(r#"[{"content": "a fact", "confidence": 0.9}]"#);
        let stored = make_extract_entries(&["a fact"]);
        let store = MockStore::new().with_add_entries_behavior(AddEntriesBehavior::Ok(stored));
        let handler = make_handler_with_provider(store, provider);
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Extract {
                    text: Some("input".to_string()),
                    files: vec![],
                    metadata: vec![("source".to_string(), "readme".to_string())],
                    max_entries: None,
                    min_confidence: None,
                    guidelines: None,
                    dry_run: true,
                },
                &mut out,
            )
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v[0]["metadata"]["source"].as_str().unwrap(), "readme");
    }

    #[tokio::test]
    async fn test_extract_max_entries_caps_result() {
        let provider = extraction_provider(
            r#"[{"content": "one", "confidence": 0.9}, {"content": "two", "confidence": 0.8}, {"content": "three", "confidence": 0.7}]"#,
        );
        let stored = make_extract_entries(&["one"]);
        let store = MockStore::new().with_add_entries_behavior(AddEntriesBehavior::Ok(stored));
        let handler = make_handler_with_provider(store, provider);
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Extract {
                    text: Some("input".to_string()),
                    files: vec![],
                    metadata: vec![],
                    max_entries: Some(1),
                    min_confidence: None,
                    guidelines: None,
                    dry_run: true,
                },
                &mut out,
            )
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v.as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_extract_empty_input_returns_error() {
        let handler = make_handler(MockStore::new());
        let mut out = Vec::new();

        let result = handler
            .handle(
                MemoryCommand::Extract {
                    text: Some("   ".to_string()),
                    files: vec![],
                    metadata: vec![],
                    max_entries: None,
                    min_confidence: None,
                    guidelines: None,
                    dry_run: false,
                },
                &mut out,
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_extract_conflicting_input_returns_error() {
        let handler = make_handler(MockStore::new());
        let mut out = Vec::new();

        let result = handler
            .handle(
                MemoryCommand::Extract {
                    text: Some("positional text".to_string()),
                    files: vec![std::path::PathBuf::from("some_file.txt")],
                    metadata: vec![],
                    max_entries: None,
                    min_confidence: None,
                    guidelines: None,
                    dry_run: false,
                },
                &mut out,
            )
            .await;

        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("conflict"),
            "expected conflict error message"
        );
    }

    #[test]
    fn pairs_to_map_converts_pairs() {
        let pairs = vec![
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "two".to_string()),
        ];
        let map = pairs_to_map(pairs);
        assert_eq!(map.get("a").unwrap(), "1");
        assert_eq!(map.get("b").unwrap(), "two");
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_pairs_to_map_empty_returns_empty_map() {
        let map = pairs_to_map(vec![]);
        assert!(map.is_empty());
    }

    #[test]
    fn test_pairs_to_map_duplicate_key_last_value_wins() {
        let pairs = vec![
            ("k".to_string(), "first".to_string()),
            ("k".to_string(), "last".to_string()),
        ];
        let map = pairs_to_map(pairs);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("k").unwrap(), "last");
    }

    #[rstest]
    #[case(MemoryKind::Fact, MemoryTrust::Fact)]
    #[case(MemoryKind::ExtractedMemory, MemoryTrust::Extracted { confidence: 0.5, evidence: TrustEvidence::default() })]
    fn test_memory_kind_all_variants_convert(
        #[case] input: MemoryKind,
        #[case] expected: MemoryTrust,
    ) {
        let result: MemoryTrust = input.into();
        assert_eq!(result, expected);
    }

    #[tokio::test]
    async fn test_memory_add_with_metadata_outputs_metadata_in_json() {
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "wiki".to_string());
        let entry = make_result_with_metadata("some fact", meta);
        let handler = make_handler(MockStore::new().with_add(entry));
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Add {
                    content: "some fact".to_string(),
                    metadata: vec![("source".to_string(), "wiki".to_string())],
                    kind: None,
                },
                &mut out,
            )
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["metadata"]["source"].as_str().unwrap(), "wiki");
    }

    #[tokio::test]
    async fn test_memory_query_empty_results_outputs_empty_array() {
        let handler = make_handler(MockStore::new());
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Query {
                    topic: "nothing".to_string(),
                    max_results: 10,
                    min_score: 0.0,
                    filters: vec![],
                },
                &mut out,
            )
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_memory_query_forwards_filters_to_store() {
        let store = Arc::new(MockStore::new());
        let handler = MemoryCommandHandler::new(
            Arc::clone(&store),
            Arc::new(MockTextGenerationModelProvider::ok()),
            "test-model",
            MemoryExtractionConfig {
                model: "test-model".to_string(),
                max_entries: None,
                min_confidence: None,
                guidelines: None,
                thinking: None,
                chunking: None,
                extractor: MemoryExtractorConfig {
                    classification_model: "test-classification-model".to_string(),
                    direct_search: MemoryExtractorSearchResultsConfig { max_results: 5, min_score: 0.70 },
                    inverted_search: MemoryExtractorSearchResultsConfig { max_results: 3, min_score: 0.60 },
                    bayesian_seed_weight: 10.0,
                    max_counter_increment: 5.0,
                    max_counter: 100.0,
                    auto_discard_threshold: 0.1,
                    auto_promotion_threshold: 0.9,
                    min_alpha_for_promotion: 12.0,
                    decay_rate: 0.99,
                },
            },
        );
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Query {
                    topic: "topic".to_string(),
                    max_results: 5,
                    min_score: 0.0,
                    filters: vec![
                        ("lang".to_string(), "rust".to_string()),
                        ("env".to_string(), "prod".to_string()),
                    ],
                },
                &mut out,
            )
            .await
            .unwrap();

        let captured = store.snapshot().query.unwrap();
        assert_eq!(captured.filters.get("lang").unwrap(), "rust");
        assert_eq!(captured.filters.get("env").unwrap(), "prod");
    }

    #[tokio::test]
    async fn test_memory_delete_propagates_store_error() {
        let handler = make_handler(MockStore::new().with_delete_behavior(
            loci_core::testing::UnitBehavior::Err(
                loci_core::testing::MockStoreErrorKind::Connection("mock: delete error".into()),
            ),
        ));
        let mut out = Vec::new();

        let result = handler
            .handle(MemoryCommand::Delete { id: Uuid::new_v4() }, &mut out)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_memory_prune_expired_propagates_store_error() {
        let handler = make_handler(MockStore::new().with_prune_behavior(
            loci_core::testing::UnitBehavior::Err(
                loci_core::testing::MockStoreErrorKind::Connection("mock: prune error".into()),
            ),
        ));
        let mut out = Vec::new();

        let result = handler.handle(MemoryCommand::PruneExpired, &mut out).await;
        assert!(result.is_err());
    }
}

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

use loci_core::{
    memory::{
        MemoryInput as CoreMemoryInput, MemoryQuery as CoreMemoryQuery,
        MemoryQueryMode as CoreMemoryQueryMode, MemoryTier as CoreMemoryTier, Score as CoreScore,
    },
    memory_extraction::{
        LlmMemoryExtractionStrategy, LlmMemoryExtractionStrategyParams, MemoryExtractionStrategy,
        MemoryExtractor,
    },
    model_provider::text_generation::{TextGenerationModelProvider, ThinkingMode},
    store::MemoryStore as CoreMemoryStore,
};
use log::debug;

use crate::{
    commands::{
        input::read_extraction_input,
        memory::{MemoryCommand, MemoryTier},
    },
    handlers::{CommandHandler, json::entry_to_json},
};

impl From<MemoryTier> for CoreMemoryTier {
    fn from(val: MemoryTier) -> Self {
        match val {
            MemoryTier::Ephemeral => CoreMemoryTier::Ephemeral,
            MemoryTier::Candidate => CoreMemoryTier::Candidate,
            MemoryTier::Stable => CoreMemoryTier::Stable,
            MemoryTier::Core => CoreMemoryTier::Core,
        }
    }
}

pub struct MemoryCommandHandler<'a, S: CoreMemoryStore, P: TextGenerationModelProvider> {
    store: Arc<S>,
    provider: Arc<P>,
    text_model: String,
    /// Ties the `'a` lifetime used in `CommandHandler<'a, …>` to this struct
    /// so that Rust can prove `'a: '_` when borrowing `&'_ Self` — identical to
    /// the pattern used in `GenerateCommandHandler`.
    _marker: PhantomData<&'a ()>,
}

impl<'a, S: CoreMemoryStore, P: TextGenerationModelProvider> MemoryCommandHandler<'a, S, P> {
    pub fn new(store: Arc<S>, provider: Arc<P>, text_model: impl Into<String>) -> Self {
        Self {
            store,
            provider,
            text_model: text_model.into(),
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
                tier,
            } => {
                debug!(
                    "add memory entry: content={content}, metadata={:?}, tier={:?}",
                    metadata, tier
                );
                let input = match tier {
                    Some(tier) => {
                        CoreMemoryInput::new_with_tier(content, pairs_to_map(metadata), tier.into())
                    }
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
            MemoryCommand::Update {
                id,
                content,
                metadata,
                tier,
            } => {
                debug!(
                    "update memory entry: id={id}, content={:?}, metadata={:?}, tier={:?}",
                    content, metadata, tier
                );
                if content.is_none() && metadata.is_empty() && tier.is_none() {
                    return Err(
                        "nothing to update; provide content, --meta, and/or --tier to change a memory"
                            .into(),
                    );
                }

                let existing = self.store.get_entry(id).await?;
                let content = content.unwrap_or(existing.memory_entry.content);
                let metadata = if metadata.is_empty() {
                    existing.memory_entry.metadata
                } else {
                    pairs_to_map(metadata)
                };
                let tier = tier
                    .map(|tier| tier.into())
                    .unwrap_or(existing.memory_entry.tier);

                let input = CoreMemoryInput::new_with_tier(content, metadata, tier);
                let entry = self.store.update_entry(id, input).await?;
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
                tier,
                metadata,
                max_entries,
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
                    guidelines,
                    default_tier: tier.into(),
                    metadata: pairs_to_map(metadata),
                    max_entries,
                    // Extraction produces structured JSON — thinking adds latency with no benefit.
                    thinking: Some(ThinkingMode::Disabled),
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
                                "tier": e.tier.map(|t: CoreMemoryTier| t.as_str()).unwrap_or("candidate"),
                                "metadata": e.metadata,
                            })
                        })
                        .collect();
                    writeln!(out, "{}", serde_json::to_string_pretty(&json)?)?;
                } else {
                    let extractor =
                        MemoryExtractor::from_arcs(Arc::clone(&self.store), Arc::new(strategy));
                    /* TODO let extractor = match chunk_size {
                        Some(size) => extractor.with_chunker(SentenceAwareChunker {
                            chunk_size: size,
                            overlap,
                        }),
                        None => extractor,
                    }; */
                    let result = extractor
                        .extract_and_store(&input, params)
                        .await
                        .map_err(|e| Box::new(e) as Box<dyn StdError>)?;
                    let json = serde_json::json!({
                        "added": result.added.iter().map(entry_to_json).collect::<Vec<_>>(),
                        "failures": result.failures.iter().map(|f| serde_json::json!({
                            "index": f.index,
                            "error": f.error.to_string(),
                        })).collect::<Vec<_>>(),
                    });
                    writeln!(out, "{}", serde_json::to_string_pretty(&json)?)?;
                }
            }
        }
        Ok(())
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

    use loci_core::{
        memory::{
            MemoryEntry as CoreMemoryEntry, MemoryQueryResult as CoreMemoryQueryResult,
            MemoryTier as CoreMemoryTier, Score as CoreScore,
        },
        model_provider::text_generation::TextGenerationResponse,
        testing::{
            AddEntriesBehavior, MockStore, MockTextGenerationModelProvider, ProviderBehavior,
        },
    };
    use serde_json::Value;
    use uuid::Uuid;

    use crate::commands::memory::MemoryTier;
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
        )
    }

    fn make_handler_with_provider(
        store: MockStore,
        provider: MockTextGenerationModelProvider,
    ) -> MemoryCommandHandler<'static, MockStore, MockTextGenerationModelProvider> {
        MemoryCommandHandler::new(Arc::new(store), Arc::new(provider), "test-model")
    }

    fn make_result(content: &str, tier: loci_core::memory::MemoryTier) -> CoreMemoryQueryResult {
        CoreMemoryQueryResult {
            memory_entry: CoreMemoryEntry::new_with_tier(content.to_string(), HashMap::new(), tier),
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
        let entry = make_result("hello world", CoreMemoryTier::Candidate);
        let id = entry.memory_entry.id;
        let handler = make_handler(MockStore::new().with_add(entry));
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Add {
                    content: "hello world".to_string(),
                    metadata: vec![],
                    tier: None,
                },
                &mut out,
            )
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["id"].as_str().unwrap(), id.to_string().as_str());
        assert_eq!(v["content"].as_str().unwrap(), "hello world");
        assert_eq!(v["tier"].as_str().unwrap(), "candidate");
    }

    #[tokio::test]
    async fn test_memory_add_with_tier_outputs_tier_field() {
        let entry = make_result("core fact", CoreMemoryTier::Core);
        let handler = make_handler(MockStore::new().with_add(entry));
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Add {
                    content: "core fact".to_string(),
                    metadata: vec![],
                    tier: Some(MemoryTier::Core),
                },
                &mut out,
            )
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["tier"].as_str().unwrap(), "core");
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
                    tier: None,
                },
                &mut out,
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_memory_query_outputs_json_array() {
        let entries = vec![
            make_result("first", CoreMemoryTier::Stable),
            make_result("second", CoreMemoryTier::Core),
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
        let entry = make_result("specific entry", CoreMemoryTier::Stable);
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
        assert_eq!(v["tier"].as_str().unwrap(), "stable");
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
    async fn test_memory_update_happy_path() {
        let id = Uuid::new_v4();
        let existing = CoreMemoryQueryResult {
            memory_entry: CoreMemoryEntry::new_with_tier(
                "old".to_string(),
                HashMap::new(),
                CoreMemoryTier::Candidate,
            ),
            score: CoreScore::ZERO,
        };
        let updated = make_result("new content", CoreMemoryTier::Stable);
        let updated_id = updated.memory_entry.id;
        let store = Arc::new(MockStore::new().with_get(existing).with_update(updated));
        let handler = MemoryCommandHandler::new(
            Arc::clone(&store),
            Arc::new(MockTextGenerationModelProvider::ok()),
            "test-model",
        );
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Update {
                    id,
                    content: Some("new content".to_string()),
                    metadata: vec![],
                    tier: Some(MemoryTier::Stable),
                },
                &mut out,
            )
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["id"].as_str().unwrap(), updated_id.to_string().as_str());
        assert_eq!(v["content"].as_str().unwrap(), "new content");
    }

    #[tokio::test]
    async fn test_memory_update_nothing_to_update_returns_err() {
        let handler = make_handler(MockStore::new());
        let mut out = Vec::new();

        let result = handler
            .handle(
                MemoryCommand::Update {
                    id: Uuid::new_v4(),
                    content: None,
                    metadata: vec![],
                    tier: None,
                },
                &mut out,
            )
            .await;

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("nothing to update"),
            "expected 'nothing to update' error message"
        );
    }

    #[tokio::test]
    async fn test_memory_update_preserves_existing_metadata_when_not_provided() {
        let id = Uuid::new_v4();
        let original_meta = HashMap::from([("source".to_string(), "original-source".to_string())]);
        let existing = make_result_with_metadata("original", original_meta.clone());
        let updated = make_result_with_metadata("updated", original_meta.clone());
        let store = Arc::new(MockStore::new().with_get(existing).with_update(updated));
        let handler = MemoryCommandHandler::new(
            Arc::clone(&store),
            Arc::new(MockTextGenerationModelProvider::ok()),
            "test-model",
        );
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Update {
                    id,
                    content: Some("updated".to_string()),
                    metadata: vec![],
                    tier: None,
                },
                &mut out,
            )
            .await
            .unwrap();

        let input = store.snapshot().update_input.unwrap();
        assert_eq!(
            input.metadata.get("source").unwrap(),
            "original-source",
            "existing metadata should be preserved when no --meta flags are given"
        );
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
            .map(|c| make_result(c, CoreMemoryTier::Candidate))
            .collect()
    }

    #[tokio::test]
    async fn test_extract_dry_run_prints_candidates_without_persisting() {
        let provider = extraction_provider(r#"["extracted fact"]"#);
        let store = Arc::new(MockStore::new()); // add_entries not configured → would err if called
        let handler = MemoryCommandHandler::new(Arc::clone(&store), Arc::new(provider), "m");
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Extract {
                    text: Some("some input text".to_string()),
                    files: vec![],
                    tier: MemoryTier::Candidate,
                    metadata: vec![],
                    max_entries: None,
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
        assert_eq!(arr[0]["tier"].as_str().unwrap(), "candidate");
    }

    #[tokio::test]
    async fn test_extract_persists_entries_by_default() {
        let provider = extraction_provider(r#"["fact one", "fact two"]"#);
        let stored = make_extract_entries(&["fact one", "fact two"]);
        let store =
            Arc::new(MockStore::new().with_add_entries_behavior(AddEntriesBehavior::Ok(stored)));
        let handler = MemoryCommandHandler::new(Arc::clone(&store), Arc::new(provider), "m");
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Extract {
                    text: Some("text with two facts".to_string()),
                    files: vec![],
                    tier: MemoryTier::Candidate,
                    metadata: vec![],
                    max_entries: None,
                    guidelines: None,
                    dry_run: false,
                },
                &mut out,
            )
            .await
            .unwrap();

        // Store was written
        let inputs = store
            .snapshot()
            .add_inputs
            .expect("store should have received add_entries");
        assert_eq!(inputs.len(), 2);

        let v = parse_json_output(&out);
        assert!(v.get("added").is_some());
        assert_eq!(v["added"].as_array().unwrap().len(), 2);
        assert_eq!(v["failures"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_extract_propagates_tier_to_entries() {
        let provider = extraction_provider(r#"["a core fact"]"#);
        let stored = make_extract_entries(&["a core fact"]);
        let store = MockStore::new().with_add_entries_behavior(AddEntriesBehavior::Ok(stored));
        let handler = make_handler_with_provider(store, provider);
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Extract {
                    text: Some("input".to_string()),
                    files: vec![],
                    tier: MemoryTier::Core,
                    metadata: vec![],
                    max_entries: None,
                    guidelines: None,
                    dry_run: true,
                },
                &mut out,
            )
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v[0]["tier"].as_str().unwrap(), "core");
    }

    #[tokio::test]
    async fn test_extract_propagates_metadata_to_entries() {
        let provider = extraction_provider(r#"["a fact"]"#);
        let stored = make_extract_entries(&["a fact"]);
        let store = MockStore::new().with_add_entries_behavior(AddEntriesBehavior::Ok(stored));
        let handler = make_handler_with_provider(store, provider);
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Extract {
                    text: Some("input".to_string()),
                    files: vec![],
                    tier: MemoryTier::Candidate,
                    metadata: vec![("source".to_string(), "readme".to_string())],
                    max_entries: None,
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
        let provider = extraction_provider(r#"["one", "two", "three"]"#);
        let stored = make_extract_entries(&["one"]);
        let store = MockStore::new().with_add_entries_behavior(AddEntriesBehavior::Ok(stored));
        let handler = make_handler_with_provider(store, provider);
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Extract {
                    text: Some("input".to_string()),
                    files: vec![],
                    tier: MemoryTier::Candidate,
                    metadata: vec![],
                    max_entries: Some(1),
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
    async fn test_extract_with_chunking_calls_provider_per_chunk() {
        // Two sentences of ~10 words each; chunk_size=10 forces two chunks.
        let input = "The quick brown fox jumps over the lazy dog today. \
                     Another quick brown fox leaps over a sleeping hound now."
            .to_string();
        let provider = Arc::new(extraction_provider(r#"["a fact"]"#));
        let stored: Vec<CoreMemoryQueryResult> = (0..2)
            .map(|_| make_result("a fact", CoreMemoryTier::Candidate))
            .collect();
        let store =
            Arc::new(MockStore::new().with_add_entries_behavior(AddEntriesBehavior::Ok(stored)));
        let handler = MemoryCommandHandler::new(Arc::clone(&store), Arc::clone(&provider), "m");
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Extract {
                    text: Some(input),
                    files: vec![],
                    tier: MemoryTier::Candidate,
                    metadata: vec![],
                    max_entries: None,
                    guidelines: None,
                    dry_run: false,
                },
                &mut out,
            )
            .await
            .unwrap();

        // Provider should have been called once per chunk (2 chunks)
        assert_eq!(provider.snapshot().request_count, 2);
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
                    tier: MemoryTier::Candidate,
                    metadata: vec![],
                    max_entries: None,
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
                    tier: MemoryTier::Candidate,
                    metadata: vec![],
                    max_entries: None,
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
    #[case(MemoryTier::Ephemeral, CoreMemoryTier::Ephemeral)]
    #[case(MemoryTier::Candidate, CoreMemoryTier::Candidate)]
    #[case(MemoryTier::Stable, CoreMemoryTier::Stable)]
    #[case(MemoryTier::Core, CoreMemoryTier::Core)]
    fn test_memory_tier_all_variants_convert(
        #[case] input: MemoryTier,
        #[case] expected: CoreMemoryTier,
    ) {
        let result: CoreMemoryTier = input.into();
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
                    tier: None,
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
    async fn test_memory_update_replaces_metadata_when_provided() {
        let id = Uuid::new_v4();
        let original_meta = HashMap::from([("old_key".to_string(), "old_val".to_string())]);
        let existing = make_result_with_metadata("original", original_meta);
        let updated = make_result_with_metadata(
            "original",
            HashMap::from([("new_key".to_string(), "new_val".to_string())]),
        );
        let store = Arc::new(MockStore::new().with_get(existing).with_update(updated));
        let handler = MemoryCommandHandler::new(
            Arc::clone(&store),
            Arc::new(MockTextGenerationModelProvider::ok()),
            "test-model",
        );
        let mut out = Vec::new();

        handler
            .handle(
                MemoryCommand::Update {
                    id,
                    content: None,
                    metadata: vec![("new_key".to_string(), "new_val".to_string())],
                    tier: None,
                },
                &mut out,
            )
            .await
            .unwrap();

        let input = store.snapshot().update_input.unwrap();
        assert!(
            !input.metadata.contains_key("old_key"),
            "old metadata should be replaced when --meta is provided"
        );
        assert_eq!(input.metadata.get("new_key").unwrap(), "new_val");
    }

    #[tokio::test]
    async fn test_memory_update_get_not_found_propagates_error() {
        let handler = make_handler(MockStore::new());
        let mut out = Vec::new();

        let result = handler
            .handle(
                MemoryCommand::Update {
                    id: Uuid::new_v4(),
                    content: Some("new".to_string()),
                    metadata: vec![],
                    tier: None,
                },
                &mut out,
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_memory_update_store_write_fails_propagates_error() {
        let existing = make_result("existing", CoreMemoryTier::Candidate);
        let handler = make_handler(MockStore::new().with_get(existing));
        let mut out = Vec::new();

        let result = handler
            .handle(
                MemoryCommand::Update {
                    id: Uuid::new_v4(),
                    content: Some("updated".to_string()),
                    metadata: vec![],
                    tier: None,
                },
                &mut out,
            )
            .await;

        assert!(result.is_err());
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

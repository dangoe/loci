// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

use std::{collections::HashMap, error::Error as StdError, io::Write};

use loci_core::{
    memory::{
        MemoryInput as CoreMemoryInput, MemoryQuery as CoreMemoryQuery,
        MemoryQueryMode as CoreMemoryQueryMode, MemoryTier as CoreMemoryTier, Score as CoreScore,
    },
    store::MemoryStore as CoreMemoryStore,
};
use log::debug;

use crate::{
    commands::memory::{MemoryCommand, MemoryTier},
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

pub struct MemoryCommandHandler<'a, S: CoreMemoryStore> {
    store: &'a S,
}

impl<'a, S: CoreMemoryStore> MemoryCommandHandler<'a, S> {
    pub fn new(store: &'a S) -> Self {
        Self { store }
    }
}

impl<'a, S: CoreMemoryStore, W: Write + Send> CommandHandler<'a, MemoryCommand, W>
    for MemoryCommandHandler<'a, S>
{
    async fn handle(&self, command: MemoryCommand, out: &mut W) -> Result<(), Box<dyn StdError>> {
        match command {
            MemoryCommand::Save {
                content,
                metadata,
                tier,
            } => {
                debug!(
                    "save memory entry: content={content}, metadata={:?}, tier={:?}",
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
                debug!("clear memory");
                self.store.prune_expired().await?;
                writeln!(
                    out,
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({ "expired pruned": true }))?
                )?;
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
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use loci_core::memory::{
        MemoryEntry as CoreMemoryEntry, MemoryInput as CoreMemoryInput,
        MemoryQueryResult as CoreMemoryQueryResult, MemoryTier as CoreMemoryTier,
        Score as CoreScore,
    };
    use serde_json::Value;
    use uuid::Uuid;

    use crate::commands::memory::MemoryTier;
    use crate::handlers::CommandHandler;

    use crate::{
        commands::memory::MemoryCommand,
        handlers::memory::{MemoryCommandHandler, pairs_to_map},
        mock::MockStore,
    };

    #[tokio::test]
    async fn test_memory_save_outputs_json() {
        let entry = make_result("hello world", CoreMemoryTier::Candidate);
        let id = entry.memory_entry.id;
        let store = MockStore::new().with_save(entry);
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);

        handler
            .handle(
                MemoryCommand::Save {
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
    async fn test_memory_save_with_tier_outputs_tier_field() {
        let entry = make_result("core fact", CoreMemoryTier::Core);
        let store = MockStore::new().with_save(entry);
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);

        handler
            .handle(
                MemoryCommand::Save {
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
    async fn test_memory_save_propagates_store_error() {
        let store = MockStore::new(); // save_entry = None → Connection error
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);

        let result = handler
            .handle(
                MemoryCommand::Save {
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
        let store = MockStore::new().with_query(entries);
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);

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
        let store = MockStore::new();
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);

        let result = handler
            .handle(
                MemoryCommand::Query {
                    topic: "t".to_string(),
                    max_results: 5,
                    min_score: 1.5, // invalid — outside [0.0, 1.0]
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
        let store = MockStore::new().with_get(entry);
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);

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
        let store = MockStore::new(); // get_entry = None → NotFound
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);

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
        let store = MockStore::new().with_get(existing).with_update(updated);
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);

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
        let store = MockStore::new();
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);

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
        let captured = Arc::new(Mutex::new(None::<CoreMemoryInput>));
        let store = MockStore {
            save_entry: None,
            get_entry: Some(existing),
            query_entries: vec![],
            update_entry: Some(updated),
            captured_update_input: captured.clone(),
            captured_query: Arc::new(Mutex::new(None)),
            delete_error: false,
            prune_error: false,
        };
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);

        handler
            .handle(
                MemoryCommand::Update {
                    id,
                    content: Some("updated".to_string()),
                    metadata: vec![], // no --meta flag → should preserve existing
                    tier: None,
                },
                &mut out,
            )
            .await
            .unwrap();

        let input = captured.lock().unwrap().take().unwrap();
        assert_eq!(
            input.metadata.get("source").unwrap(),
            "original-source",
            "existing metadata should be preserved when no --meta flags are given"
        );
    }

    #[tokio::test]
    async fn test_memory_delete_outputs_deleted_id() {
        let id = Uuid::new_v4();
        let store = MockStore::new();
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);

        handler
            .handle(MemoryCommand::Delete { id }, &mut out)
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["deleted"].as_str().unwrap(), id.to_string().as_str());
    }

    #[tokio::test]
    async fn test_memory_prune_expired_outputs_success() {
        let store = MockStore::new();
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);

        handler
            .handle(MemoryCommand::PruneExpired, &mut out)
            .await
            .unwrap();

        let v = parse_json_output(&out);
        assert_eq!(v["expired pruned"].as_bool().unwrap(), true);
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

    #[tokio::test]
    async fn test_memory_save_with_metadata_outputs_metadata_in_json() {
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "wiki".to_string());
        let entry = make_result_with_metadata("some fact", meta);
        let store = MockStore::new().with_save(entry);
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);
        handler
            .handle(
                MemoryCommand::Save {
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
        let store = MockStore::new(); // query_entries defaults to vec![]
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);
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
        let store = MockStore::new();
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);
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

        let captured = store.captured_query.lock().unwrap().take().unwrap();
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
        let captured = Arc::new(Mutex::new(None::<CoreMemoryInput>));
        let store = MockStore {
            save_entry: None,
            get_entry: Some(existing),
            query_entries: vec![],
            update_entry: Some(updated),
            captured_update_input: captured.clone(),
            captured_query: Arc::new(Mutex::new(None)),
            delete_error: false,
            prune_error: false,
        };
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);
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

        let input = captured.lock().unwrap().take().unwrap();
        assert!(
            !input.metadata.contains_key("old_key"),
            "old metadata should be replaced when --meta is provided"
        );
        assert_eq!(input.metadata.get("new_key").unwrap(), "new_val");
    }

    #[tokio::test]
    async fn test_memory_update_get_not_found_propagates_error() {
        let store = MockStore::new(); // get_entry = None → NotFound
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);
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
        let store = MockStore::new().with_get(existing); // update_entry = None → NotFound
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);
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
        let store = MockStore::new().with_delete_error();
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);
        let result = handler
            .handle(MemoryCommand::Delete { id: Uuid::new_v4() }, &mut out)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_memory_prune_expired_propagates_store_error() {
        let store = MockStore::new().with_prune_error();
        let mut out = Vec::new();

        let handler = MemoryCommandHandler::new(&store);
        let result = handler.handle(MemoryCommand::PruneExpired, &mut out).await;

        assert!(result.is_err());
    }
}

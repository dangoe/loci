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

impl Into<CoreMemoryTier> for MemoryTier {
    fn into(self) -> CoreMemoryTier {
        match self {
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
                let entry = self.store.save(input).await?;
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
                let entry = self.store.get(id).await?;
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

                let existing = self.store.get(id).await?;
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
                let entry = self.store.update(id, input).await?;
                writeln!(
                    out,
                    "{}",
                    serde_json::to_string_pretty(&entry_to_json(&entry))?
                )?;
            }
            MemoryCommand::Delete { id } => {
                debug!("delete memory entry: id={id}");
                self.store.delete(id).await?;
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
}

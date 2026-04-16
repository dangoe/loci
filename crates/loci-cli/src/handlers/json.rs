// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

/// Serialises a [`loci_core::memory::MemoryEntry`] to a [`serde_json::Value`].
pub fn entry_to_json(e: &loci_core::memory::MemoryQueryResult) -> serde_json::Value {
    serde_json::json!({
        "id": e.memory_entry.id.to_string(),
        "content": e.memory_entry.content,
        "metadata": e.memory_entry.metadata,
        "tier": e.memory_entry.tier.as_str(),
        "confidence": e.memory_entry.confidence,
        "seen_count": e.memory_entry.seen_count,
        "sources": e.memory_entry.sources,
        "first_seen": e.memory_entry.first_seen.to_rfc3339(),
        "last_seen": e.memory_entry.last_seen.to_rfc3339(),
        "expires_at": e.memory_entry.expires_at.map(|dt| dt.to_rfc3339()),
        "created_at": e.memory_entry.created_at.to_rfc3339(),
        "score": e.score.value(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use loci_core::memory::{
        MemoryEntry as CoreMemoryEntry, MemoryQueryResult as CoreMemoryQueryResult,
        Score as CoreScore,
    };
    use serde_json::Value as JsonValue;

    use crate::handlers::json::entry_to_json;

    #[test]
    fn entry_to_json_serializes_fields() {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "unit-test".to_string());

        let entry = CoreMemoryEntry::new_with_tier(
            "my content".to_string(),
            metadata.clone(),
            loci_core::memory::MemoryTier::Core,
        );
        let mq = CoreMemoryQueryResult {
            memory_entry: entry.clone(),
            score: CoreScore::new(0.75).unwrap(),
        };

        let v: JsonValue = entry_to_json(&mq);

        assert_eq!(v["id"].as_str().unwrap(), entry.id.to_string().as_str());
        assert_eq!(v["content"].as_str().unwrap(), "my content");
        assert_eq!(v["metadata"]["source"].as_str().unwrap(), "unit-test");
        assert_eq!(v["tier"].as_str().unwrap(), "core");
        assert_eq!(v["seen_count"].as_u64().unwrap(), entry.seen_count as u64);
        assert!(v.get("expires_at").unwrap().is_null());
        assert_eq!(v["score"].as_f64().unwrap(), 0.75);
        assert!(v["created_at"].as_str().is_some());
        assert!(v["first_seen"].as_str().is_some());
        assert!(v["last_seen"].as_str().is_some());
    }
}

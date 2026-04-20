// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

/// Serialises a [`loci_core::memory::MemoryEntry`] to a [`serde_json::Value`].
pub fn entry_to_json(e: &loci_core::memory::MemoryEntry) -> serde_json::Value {
    use loci_core::memory::MemoryTrust;
    let (kind, confidence, trust_evidence) = match e.trust() {
        MemoryTrust::Fact => ("fact", 1.0_f64, serde_json::Value::Null),
        MemoryTrust::Extracted {
            confidence,
            evidence,
        } => (
            "extracted_memory",
            evidence.bayesian_confidence().unwrap_or(*confidence),
            serde_json::json!({
                "alpha": evidence.alpha,
                "beta": evidence.beta,
            }),
        ),
    };
    serde_json::json!({
        "id": e.id().to_string(),
        "content": e.content(),
        "metadata": e.metadata(),
        "kind": kind,
        "confidence": confidence,
        "trust_evidence": trust_evidence,
        "seen_count": e.seen_count(),
        "first_seen": e.first_seen().map(|dt| dt.to_rfc3339()),
        "last_seen": e.last_seen().map(|dt| dt.to_rfc3339()),
        "expires_at": e.expires_at().map(|dt| dt.to_rfc3339()),
        "created_at": e.created_at().to_rfc3339(),
        "score": e.trust().effective_score().value(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use loci_core::memory::{MemoryEntry as CoreMemoryEntry, MemoryTrust};
    use serde_json::Value as JsonValue;

    use crate::handlers::json::entry_to_json;

    #[test]
    fn test_entry_to_json_serializes_fields() {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "unit-test".to_string());

        let entry = CoreMemoryEntry::new_with_trust(
            "my content".to_string(),
            metadata.clone(),
            MemoryTrust::Fact,
        );

        let v: JsonValue = entry_to_json(&entry);

        assert_eq!(v["id"].as_str().unwrap(), entry.id().to_string().as_str());
        assert_eq!(v["content"].as_str().unwrap(), "my content");
        assert_eq!(v["metadata"]["source"].as_str().unwrap(), "unit-test");
        assert_eq!(v["kind"].as_str().unwrap(), "fact");
        assert_eq!(v["seen_count"].as_u64().unwrap(), entry.seen_count() as u64);
        assert!(v.get("expires_at").unwrap().is_null());
        assert_eq!(v["score"].as_f64().unwrap(), 1.0);
        assert!(v["created_at"].as_str().is_some());
        assert!(v["first_seen"].as_str().is_none()); // first_seen is None by default
        assert!(v["last_seen"].as_str().is_none()); // last_seen is None by default
    }
}

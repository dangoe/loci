// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-e2e-tests.

mod support;

use std::collections::HashMap;
use std::sync::Arc;

use loci_core::memory::{MemoryQuery, MemoryQueryMode, MemoryTrust, Score};
use loci_core::memory_extraction::{
    LlmMemoryExtractionStrategy, LlmMemoryExtractionStrategyParams, MemoryExtractionStrategy,
    MemoryExtractor, MemoryExtractorConfig,
};
use loci_core::store::MemoryStore;
use loci_model_provider_ollama::classification::LlmClassificationModelProvider;
use loci_model_provider_ollama::testing::{
    classification_model, ensure_ollama_available, ollama_provider, text_model,
};

use support::{create_embedder, start_qdrant_store};

const RICH_INPUT: &str = "Alice is a senior Rust developer with ten years of experience. \
    She prefers functional programming paradigms and uses Neovim as her primary editor. \
    Her current project is a distributed key-value store written in async Rust. \
    She works remotely from Berlin and her preferred communication tool is Zulip.";

fn base_params() -> LlmMemoryExtractionStrategyParams {
    LlmMemoryExtractionStrategyParams {
        guidelines: None,
        metadata: HashMap::new(),
        max_entries: None,
        min_confidence: None,
        thinking_mode: None,
        chunking: None,
    }
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_extract_yields_entries_from_fact_rich_text() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), text_model());

    let entries = strategy
        .extract(RICH_INPUT, base_params())
        .await
        .expect("extraction should succeed");

    assert!(
        !entries.is_empty(),
        "a fact-rich paragraph should yield at least one extracted entry"
    );
    for entry in &entries {
        assert!(
            !entry.content.is_empty(),
            "every extracted entry must have non-empty content"
        );
    }

    // The input contains several proper nouns that any small model reliably
    // preserves verbatim.  At least one extracted entry must surface one of them.
    let key_terms = ["alice", "rust", "neovim", "berlin", "zulip"];
    let mentions_key_term = entries.iter().any(|e| {
        let lower = e.content.to_lowercase();
        key_terms.iter().any(|kw| lower.contains(kw))
    });
    assert!(
        mentions_key_term,
        "at least one extracted entry should reference a key term from the input \
         (checked: {key_terms:?}); got: {:?}",
        entries.iter().map(|e| &e.content).collect::<Vec<_>>()
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_extracted_entries_carry_extracted_memory_kind() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), text_model());

    let entries = strategy
        .extract(RICH_INPUT, base_params())
        .await
        .expect("extraction should succeed");

    // If the model returned nothing we cannot validate kinds, so fail loudly.
    assert!(
        !entries.is_empty(),
        "extraction should yield entries to validate kind propagation"
    );
    for entry in &entries {
        assert!(
            matches!(&entry.trust, Some(MemoryTrust::Extracted { .. })),
            "every extracted entry must carry the hardcoded ExtractedMemory kind"
        );
    }
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_extracted_entries_carry_configured_metadata() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), text_model());

    let mut meta = HashMap::new();
    meta.insert("source".to_string(), "conversation".to_string());
    meta.insert("session_id".to_string(), "abc-123".to_string());

    let params = LlmMemoryExtractionStrategyParams {
        metadata: meta,
        ..base_params()
    };

    let entries = strategy
        .extract(RICH_INPUT, params)
        .await
        .expect("extraction should succeed");

    assert!(
        !entries.is_empty(),
        "extraction should yield entries to validate metadata propagation"
    );
    for entry in &entries {
        assert_eq!(
            entry.metadata.get("source").map(String::as_str),
            Some("conversation"),
            "every entry should carry the configured 'source' metadata key"
        );
        assert_eq!(
            entry.metadata.get("session_id").map(String::as_str),
            Some("abc-123"),
            "every entry should carry the configured 'session_id' metadata key"
        );
    }
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_max_entries_cap_is_honoured() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), text_model());

    let params = LlmMemoryExtractionStrategyParams {
        max_entries: Some(1),
        ..base_params()
    };

    let entries = strategy
        .extract(RICH_INPUT, params)
        .await
        .expect("extraction should succeed");

    assert!(
        entries.len() <= 1,
        "max_entries: Some(1) should cap the result at one entry, got {}",
        entries.len()
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_extract_from_sparse_text_does_not_error() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), text_model());

    // Very little semantic content — the model should return an empty array,
    // not an error.
    let result = strategy.extract("ok", base_params()).await;

    assert!(
        result.is_ok(),
        "extraction must not error on sparse input, got: {:?}",
        result.unwrap_err()
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_extract_and_store_persists_entries() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let embedder = create_embedder(Arc::clone(&provider));
    let (store, _container) = start_qdrant_store(embedder, None).await;
    let store = Arc::new(store);

    let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), text_model());
    let classifier = Arc::new(LlmClassificationModelProvider::new(
        Arc::clone(&provider),
        classification_model(),
    ));
    let extractor = MemoryExtractor::new(
        Arc::clone(&store),
        Arc::new(strategy),
        classifier,
        MemoryExtractorConfig::default(),
    );

    let result = extractor
        .extract_and_store(RICH_INPUT, base_params())
        .await
        .expect("extract_and_store should succeed");

    let stored: Vec<_> = result
        .inserted
        .iter()
        .chain(result.merged.iter())
        .chain(result.promoted.iter())
        .collect();

    assert!(
        !stored.is_empty(),
        "at least one entry should be stored from a fact-rich paragraph"
    );

    // The combined content of all stored entries should reference at least one
    // proper noun that was present in the input.
    let key_terms = ["alice", "rust", "neovim", "berlin", "zulip"];
    let all_content = stored
        .iter()
        .map(|r| r.memory_entry.content.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    let mentions_key_term = key_terms.iter().any(|kw| all_content.contains(kw));
    assert!(
        mentions_key_term,
        "stored entries should reference key terms from the input \
         (checked: {key_terms:?}); stored: {:?}",
        stored
            .iter()
            .map(|r| &r.memory_entry.content)
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_extracted_entries_are_semantically_retrievable() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let embedder = create_embedder(Arc::clone(&provider));
    let (store, _container) = start_qdrant_store(embedder, None).await;
    let store = Arc::new(store);

    let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), text_model());
    let classifier = Arc::new(LlmClassificationModelProvider::new(
        Arc::clone(&provider),
        classification_model(),
    ));
    let extractor = MemoryExtractor::new(
        Arc::clone(&store),
        Arc::new(strategy),
        classifier,
        MemoryExtractorConfig::default(),
    );

    extractor
        .extract_and_store(RICH_INPUT, base_params())
        .await
        .expect("extract_and_store should succeed");

    // A semantically related but differently phrased query should surface at
    // least one of the stored facts.
    let results = store
        .query(MemoryQuery {
            topic: "what text editor does the developer use".to_string(),
            max_results: 5,
            min_score: Score::ZERO,
            filters: HashMap::new(),
            mode: MemoryQueryMode::Lookup,
        })
        .await
        .expect("semantic query should succeed");

    assert!(
        !results.is_empty(),
        "a semantic query for 'text editor' should find stored entries"
    );

    // The editor-specific query should surface the Neovim entry.  We also
    // accept "vim" and "editor" in case the model slightly paraphrased.
    let mentions_editor = results.iter().any(|r| {
        let lower = r.memory_entry.content.to_lowercase();
        lower.contains("neovim") || lower.contains("vim") || lower.contains("editor")
    });
    assert!(
        mentions_editor,
        "at least one result for the 'text editor' query should mention the editor \
         from the input (checked: neovim / vim / editor); got: {:?}",
        results
            .iter()
            .map(|r| &r.memory_entry.content)
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_stored_entries_have_configured_metadata() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let embedder = create_embedder(Arc::clone(&provider));
    let (store, _container) = start_qdrant_store(embedder, None).await;
    let store = Arc::new(store);

    let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), text_model());
    let classifier = Arc::new(LlmClassificationModelProvider::new(
        Arc::clone(&provider),
        classification_model(),
    ));
    let extractor = MemoryExtractor::new(
        Arc::clone(&store),
        Arc::new(strategy),
        classifier,
        MemoryExtractorConfig::default(),
    );

    let mut meta = HashMap::new();
    meta.insert("source".to_string(), "e2e-test".to_string());

    let params = LlmMemoryExtractionStrategyParams {
        metadata: meta,
        ..base_params()
    };

    let result = extractor
        .extract_and_store(RICH_INPUT, params)
        .await
        .expect("extract_and_store should succeed");

    let stored: Vec<_> = result
        .inserted
        .iter()
        .chain(result.merged.iter())
        .chain(result.promoted.iter())
        .collect();

    assert!(
        !stored.is_empty(),
        "extraction should yield entries to validate metadata persistence"
    );
    for entry in &stored {
        assert_eq!(
            entry
                .memory_entry
                .metadata
                .get("source")
                .map(String::as_str),
            Some("e2e-test"),
            "stored entry should carry the configured metadata, entry id: {}",
            entry.memory_entry.id
        );
    }
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_stored_entries_have_extracted_memory_kind() {
    ensure_ollama_available().await;

    let provider = Arc::new(ollama_provider());
    let embedder = create_embedder(Arc::clone(&provider));
    let (store, _container) = start_qdrant_store(embedder, None).await;
    let store = Arc::new(store);

    let strategy = LlmMemoryExtractionStrategy::new(Arc::clone(&provider), text_model());
    let classifier = Arc::new(LlmClassificationModelProvider::new(
        Arc::clone(&provider),
        classification_model(),
    ));
    let extractor = MemoryExtractor::new(
        Arc::clone(&store),
        Arc::new(strategy),
        classifier,
        MemoryExtractorConfig::default(),
    );

    let result = extractor
        .extract_and_store(RICH_INPUT, base_params())
        .await
        .expect("extract_and_store should succeed");

    assert!(
        !result.inserted.is_empty(),
        "extraction should yield inserted entries to validate kind persistence"
    );
    // In a fresh store there are no prior entries so all candidates are inserted,
    // not promoted. Each inserted entry must carry the Extracted trust kind.
    for entry in &result.inserted {
        assert!(
            matches!(entry.memory_entry.trust, MemoryTrust::Extracted { .. }),
            "inserted entry must carry the Extracted trust kind, entry id: {}",
            entry.memory_entry.id
        );
    }
}

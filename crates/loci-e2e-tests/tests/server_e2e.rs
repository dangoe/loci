// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-e2e-tests.

mod support;

use loci_model_provider_ollama::testing::{
    base_url, embedding_model, ensure_ollama_available, text_model,
};
use loci_server::loci::memory::v1::{
    MemoryServiceAddEntryRequest, MemoryServiceDeleteEntryRequest, MemoryServiceGetEntryRequest,
    MemoryServiceQueryRequest,
};
use loci_server::testing::{TestServer, minimal_app_config};

use support::{EMBEDDING_DIM, start_qdrant_container};

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_add_and_get_entry_via_server() {
    ensure_ollama_available().await;

    let (_container, qdrant_url) = start_qdrant_container().await;
    let config = minimal_app_config(
        &qdrant_url,
        &base_url(),
        &text_model(),
        &embedding_model(),
        EMBEDDING_DIM,
    );
    let server = TestServer::start(config).await;
    let client = server.memory_client();

    let add_resp = client
        .add_entry(MemoryServiceAddEntryRequest {
            content: "The user's favorite programming language is Rust".to_string(),
            ..Default::default()
        })
        .await
        .expect("add_entry should succeed");

    let entry = add_resp
        .view()
        .entry
        .as_option()
        .expect("response should contain entry");
    let id = entry.id.to_string();
    assert!(!id.is_empty(), "entry should have a non-empty id");
    assert_eq!(
        entry.content,
        "The user's favorite programming language is Rust"
    );

    let get_resp = client
        .get_entry(MemoryServiceGetEntryRequest {
            id: id.clone(),
            ..Default::default()
        })
        .await
        .expect("get_entry should succeed");

    let fetched = get_resp
        .view()
        .entry
        .as_option()
        .expect("response should contain entry");
    assert_eq!(fetched.id, id);
    assert_eq!(
        fetched.content,
        "The user's favorite programming language is Rust"
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_add_and_query_entry_via_server() {
    ensure_ollama_available().await;

    let (_container, qdrant_url) = start_qdrant_container().await;
    let config = minimal_app_config(
        &qdrant_url,
        &base_url(),
        &text_model(),
        &embedding_model(),
        EMBEDDING_DIM,
    );
    let server = TestServer::start(config).await;
    let client = server.memory_client();

    client
        .add_entry(MemoryServiceAddEntryRequest {
            content: "The user's name is Bob".to_string(),
            ..Default::default()
        })
        .await
        .expect("add_entry should succeed");

    let query_resp = client
        .query(MemoryServiceQueryRequest {
            topic: "what is the user's name".to_string(),
            max_results: 5,
            ..Default::default()
        })
        .await
        .expect("query should succeed");

    let entries = &query_resp.view().entries;
    assert!(
        !entries.is_empty(),
        "query should return at least one result"
    );
    assert!(
        entries[0].content.contains("Bob"),
        "top result should contain 'Bob', got: {:?}",
        entries[0].content
    );
}

#[tokio::test]
#[cfg_attr(not(feature = "e2e"), ignore)]
async fn test_delete_entry_via_server() {
    ensure_ollama_available().await;

    let (_container, qdrant_url) = start_qdrant_container().await;
    let config = minimal_app_config(
        &qdrant_url,
        &base_url(),
        &text_model(),
        &embedding_model(),
        EMBEDDING_DIM,
    );
    let server = TestServer::start(config).await;
    let client = server.memory_client();

    let add_resp = client
        .add_entry(MemoryServiceAddEntryRequest {
            content: "Temporary memory entry".to_string(),
            ..Default::default()
        })
        .await
        .expect("add_entry should succeed");

    let id = add_resp
        .view()
        .entry
        .as_option()
        .expect("response should contain entry")
        .id
        .to_string();

    let del_resp = client
        .delete_entry(MemoryServiceDeleteEntryRequest {
            id: id.clone(),
            ..Default::default()
        })
        .await
        .expect("delete_entry should succeed");

    assert!(del_resp.view().deleted, "deleted flag should be true");

    let get_result = client
        .get_entry(MemoryServiceGetEntryRequest {
            id: id.clone(),
            ..Default::default()
        })
        .await;

    assert!(
        get_result.is_err(),
        "get_entry after delete should return an error"
    );
}

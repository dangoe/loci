// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use loci_core::memory::{MemoryInput, MemoryQueryResult, MemoryTier as CoreMemoryTier};
use loci_core::store::MemoryStore;

use crate::error::RpcError;
use crate::state::AppState;

// ─── Param structs ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SaveMemoryParams {
    content: String,
    #[serde(default)]
    metadata: HashMap<String, String>,
    tier: Option<ApiMemoryTier>,
}

#[derive(Deserialize)]
struct GetMemoryParams {
    id: Uuid,
}

#[derive(Deserialize)]
struct QueryMemoryParams {
    topic: String,
    #[serde(default = "default_max_results")]
    max_results: usize,
    #[serde(default)]
    min_score: f64,
    #[serde(default)]
    filters: HashMap<String, String>,
}

fn default_max_results() -> usize {
    10
}

#[derive(Deserialize)]
struct UpdateMemoryParams {
    id: Uuid,
    content: String,
    #[serde(default)]
    metadata: HashMap<String, String>,
    tier: Option<ApiMemoryTier>,
}

#[derive(Deserialize)]
struct SetTierParams {
    id: Uuid,
    tier: ApiMemoryTier,
}

#[derive(Deserialize)]
struct DeleteMemoryParams {
    id: Uuid,
}

// ─── Tier enum ───────────────────────────────────────────────────────────────

#[derive(Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ApiMemoryTier {
    Ephemeral,
    Candidate,
    Stable,
    Core,
}

impl From<ApiMemoryTier> for CoreMemoryTier {
    fn from(t: ApiMemoryTier) -> Self {
        match t {
            ApiMemoryTier::Ephemeral => CoreMemoryTier::Ephemeral,
            ApiMemoryTier::Candidate => CoreMemoryTier::Candidate,
            ApiMemoryTier::Stable => CoreMemoryTier::Stable,
            ApiMemoryTier::Core => CoreMemoryTier::Core,
        }
    }
}

impl From<CoreMemoryTier> for ApiMemoryTier {
    fn from(t: CoreMemoryTier) -> Self {
        match t {
            CoreMemoryTier::Ephemeral => ApiMemoryTier::Ephemeral,
            CoreMemoryTier::Candidate => ApiMemoryTier::Candidate,
            CoreMemoryTier::Stable => ApiMemoryTier::Stable,
            CoreMemoryTier::Core => ApiMemoryTier::Core,
        }
    }
}

// ─── Response struct ─────────────────────────────────────────────────────────

#[derive(Serialize)]
struct MemoryEntryResponse {
    id: String,
    content: String,
    metadata: HashMap<String, String>,
    tier: String,
    seen_count: u32,
    sources: Vec<String>,
    first_seen: String,
    last_seen: String,
    expires_at: Option<String>,
    created_at: String,
    score: f64,
}

impl From<&MemoryQueryResult> for MemoryEntryResponse {
    fn from(r: &MemoryQueryResult) -> Self {
        Self {
            id: r.memory_entry.id.to_string(),
            content: r.memory_entry.content.clone(),
            metadata: r.memory_entry.metadata.clone(),
            tier: r.memory_entry.tier.as_str().to_string(),
            seen_count: r.memory_entry.seen_count,
            sources: r.memory_entry.sources.clone(),
            first_seen: r.memory_entry.first_seen.to_rfc3339(),
            last_seen: r.memory_entry.last_seen.to_rfc3339(),
            expires_at: r.memory_entry.expires_at.map(|dt| dt.to_rfc3339()),
            created_at: r.memory_entry.created_at.to_rfc3339(),
            score: r.score.value(),
        }
    }
}

fn to_value<S: Serialize>(v: S) -> Result<serde_json::Value, RpcError> {
    serde_json::to_value(v).map_err(RpcError::internal_error)
}

// ─── Handlers ────────────────────────────────────────────────────────────────

pub(crate) async fn handle_save(
    params: serde_json::Value,
    state: &AppState,
) -> Result<serde_json::Value, RpcError> {
    let p: SaveMemoryParams = serde_json::from_value(params).map_err(RpcError::invalid_params)?;

    let tier = p
        .tier
        .map(CoreMemoryTier::from)
        .unwrap_or(CoreMemoryTier::Candidate);
    let input = MemoryInput::new_with_tier(p.content, p.metadata, tier);
    let result = state.store.save(input).await.map_err(RpcError::from)?;

    to_value(MemoryEntryResponse::from(&result))
}

pub(crate) async fn handle_get(
    params: serde_json::Value,
    state: &AppState,
) -> Result<serde_json::Value, RpcError> {
    let p: GetMemoryParams = serde_json::from_value(params).map_err(RpcError::invalid_params)?;

    let result = state.store.get(p.id).await.map_err(RpcError::from)?;
    to_value(MemoryEntryResponse::from(&result))
}

pub(crate) async fn handle_query(
    params: serde_json::Value,
    state: &AppState,
) -> Result<serde_json::Value, RpcError> {
    use loci_core::memory::{MemoryQuery, MemoryQueryMode, Score};

    let p: QueryMemoryParams = serde_json::from_value(params).map_err(RpcError::invalid_params)?;

    let min_score = Score::new(p.min_score).map_err(RpcError::invalid_params)?;

    let query = MemoryQuery {
        topic: p.topic,
        max_results: p.max_results,
        min_score,
        filters: p.filters,
        mode: MemoryQueryMode::Lookup,
    };

    let results = state.store.query(query).await.map_err(RpcError::from)?;
    let responses: Vec<MemoryEntryResponse> =
        results.iter().map(MemoryEntryResponse::from).collect();
    to_value(responses)
}

pub(crate) async fn handle_update(
    params: serde_json::Value,
    state: &AppState,
) -> Result<serde_json::Value, RpcError> {
    let p: UpdateMemoryParams = serde_json::from_value(params).map_err(RpcError::invalid_params)?;

    let tier = p.tier.map(CoreMemoryTier::from);
    let input = if let Some(t) = tier {
        MemoryInput::new_with_tier(p.content, p.metadata, t)
    } else {
        MemoryInput::new(p.content, p.metadata)
    };
    let result = state
        .store
        .update(p.id, input)
        .await
        .map_err(RpcError::from)?;

    to_value(MemoryEntryResponse::from(&result))
}

pub(crate) async fn handle_set_tier(
    params: serde_json::Value,
    state: &AppState,
) -> Result<serde_json::Value, RpcError> {
    let p: SetTierParams = serde_json::from_value(params).map_err(RpcError::invalid_params)?;

    let result = state
        .store
        .set_tier(p.id, p.tier.into())
        .await
        .map_err(RpcError::from)?;

    to_value(MemoryEntryResponse::from(&result))
}

pub(crate) async fn handle_delete(
    params: serde_json::Value,
    state: &AppState,
) -> Result<serde_json::Value, RpcError> {
    let p: DeleteMemoryParams = serde_json::from_value(params).map_err(RpcError::invalid_params)?;

    state.store.delete(p.id).await.map_err(RpcError::from)?;
    to_value(serde_json::json!({ "deleted": true }))
}

pub(crate) async fn handle_prune_expired(state: &AppState) -> Result<serde_json::Value, RpcError> {
    state.store.prune_expired().await.map_err(RpcError::from)?;
    to_value(serde_json::json!({ "pruned": true }))
}

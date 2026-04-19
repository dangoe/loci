// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

use std::collections::HashMap;
use std::sync::Arc;

use buffa::view::OwnedView;
use buffa::{EnumValue, MessageField};
use buffa_types::google::protobuf::Timestamp;
use chrono::{DateTime, Utc};
use connectrpc::{ConnectError, Context};
use uuid::Uuid;

use loci_core::error::MemoryStoreError;
use loci_core::memory::{
    MemoryInput, MemoryQuery, MemoryQueryMode, MemoryQueryResult, MemoryTrust, Score,
};
use loci_core::memory_store::MemoryStore;
use loci_core::model_provider::text_generation::TextGenerationModelProvider;

use crate::loci::memory::v1::{
    MemoryEntry, MemoryKind, MemoryServiceAddEntryRequestView, MemoryServiceAddEntryResponse,
    MemoryServiceDeleteEntryRequestView, MemoryServiceDeleteEntryResponse,
    MemoryServiceGetEntryRequestView, MemoryServiceGetEntryResponse,
    MemoryServicePruneExpiredRequestView, MemoryServicePruneExpiredResponse,
    MemoryServiceQueryRequestView, MemoryServiceQueryResponse,
    MemoryServiceSetEntryKindRequestView, MemoryServiceSetEntryKindResponse,
    MemoryServiceUpdateEntryRequestView, MemoryServiceUpdateEntryResponse,
};
use crate::state::AppState;

pub(crate) struct MemoryServiceImpl<M, E>
where
    M: MemoryStore,
    E: TextGenerationModelProvider + 'static,
{
    state: Arc<AppState<M, E>>,
}

impl<M, E> MemoryServiceImpl<M, E>
where
    M: MemoryStore,
    E: TextGenerationModelProvider + 'static,
{
    pub(crate) fn new(state: Arc<AppState<M, E>>) -> Self {
        Self { state }
    }
}

impl<M, E> crate::loci::memory::v1::MemoryService for MemoryServiceImpl<M, E>
where
    M: MemoryStore + 'static,
    E: TextGenerationModelProvider + 'static,
{
    async fn add_entry(
        &self,
        ctx: Context,
        request: OwnedView<MemoryServiceAddEntryRequestView<'static>>,
    ) -> Result<(MemoryServiceAddEntryResponse, Context), ConnectError> {
        let trust = proto_kind_to_trust(request.kind.as_known());
        let metadata = map_view_to_hashmap(&request.metadata);
        let input = MemoryInput::new_with_trust(request.content.to_owned(), metadata, trust);

        let result = self.state.store.add_entry(input).await.map_err(store_err)?;
        Ok((
            MemoryServiceAddEntryResponse {
                entry: MessageField::some(query_result_to_proto(&result)),
                ..Default::default()
            },
            ctx,
        ))
    }

    async fn get_entry(
        &self,
        ctx: Context,
        request: OwnedView<MemoryServiceGetEntryRequestView<'static>>,
    ) -> Result<(MemoryServiceGetEntryResponse, Context), ConnectError> {
        let id = parse_uuid(request.id)?;
        let result = self.state.store.get_entry(id).await.map_err(store_err)?;
        Ok((
            MemoryServiceGetEntryResponse {
                entry: MessageField::some(query_result_to_proto(&result)),
                ..Default::default()
            },
            ctx,
        ))
    }

    async fn query(
        &self,
        ctx: Context,
        request: OwnedView<MemoryServiceQueryRequestView<'static>>,
    ) -> Result<(MemoryServiceQueryResponse, Context), ConnectError> {
        let min_score = Score::new(request.min_score)
            .map_err(|_| ConnectError::invalid_argument("min_score must be in [0.0, 1.0]"))?;
        let query = MemoryQuery {
            topic: request.topic.to_owned(),
            max_results: request.max_results as usize,
            min_score,
            filters: map_view_to_hashmap(&request.filters),
            mode: MemoryQueryMode::Lookup,
        };
        let results = self.state.store.query(query).await.map_err(store_err)?;
        Ok((
            MemoryServiceQueryResponse {
                entries: results.iter().map(query_result_to_proto).collect(),
                ..Default::default()
            },
            ctx,
        ))
    }

    async fn update_entry(
        &self,
        ctx: Context,
        request: OwnedView<MemoryServiceUpdateEntryRequestView<'static>>,
    ) -> Result<(MemoryServiceUpdateEntryResponse, Context), ConnectError> {
        let id = parse_uuid(request.id)?;
        let trust = proto_kind_to_trust(request.kind.as_known());
        let input = MemoryInput::new_with_trust(
            request.content.to_owned(),
            map_view_to_hashmap(&request.metadata),
            trust,
        );
        let result = self
            .state
            .store
            .update_entry(id, input)
            .await
            .map_err(store_err)?;
        Ok((
            MemoryServiceUpdateEntryResponse {
                entry: MessageField::some(query_result_to_proto(&result)),
                ..Default::default()
            },
            ctx,
        ))
    }

    async fn set_entry_kind(
        &self,
        ctx: Context,
        request: OwnedView<MemoryServiceSetEntryKindRequestView<'static>>,
    ) -> Result<(MemoryServiceSetEntryKindResponse, Context), ConnectError> {
        let id = parse_uuid(request.id)?;
        let trust = proto_kind_to_trust(request.kind.as_known());
        let result = self
            .state
            .store
            .set_entry_trust(id, trust)
            .await
            .map_err(store_err)?;
        Ok((
            MemoryServiceSetEntryKindResponse {
                entry: MessageField::some(query_result_to_proto(&result)),
                ..Default::default()
            },
            ctx,
        ))
    }

    async fn delete_entry(
        &self,
        ctx: Context,
        request: OwnedView<MemoryServiceDeleteEntryRequestView<'static>>,
    ) -> Result<(MemoryServiceDeleteEntryResponse, Context), ConnectError> {
        let id = parse_uuid(request.id)?;
        self.state.store.delete_entry(id).await.map_err(store_err)?;
        Ok((
            MemoryServiceDeleteEntryResponse {
                deleted: true,
                ..Default::default()
            },
            ctx,
        ))
    }

    async fn prune_expired(
        &self,
        ctx: Context,
        _request: OwnedView<MemoryServicePruneExpiredRequestView<'static>>,
    ) -> Result<(MemoryServicePruneExpiredResponse, Context), ConnectError> {
        self.state.store.prune_expired().await.map_err(store_err)?;
        Ok((
            MemoryServicePruneExpiredResponse {
                pruned: true,
                ..Default::default()
            },
            ctx,
        ))
    }
}

#[allow(clippy::result_large_err)]
fn parse_uuid(string: &str) -> Result<Uuid, ConnectError> {
    Uuid::parse_str(string)
        .map_err(|_| ConnectError::invalid_argument(format!("invalid id: {string}")))
}

fn store_err(e: MemoryStoreError) -> ConnectError {
    match e {
        MemoryStoreError::NotFound(id) => {
            ConnectError::not_found(format!("memory entry not found: {id}"))
        }
        other => ConnectError::internal(other.to_string()),
    }
}

fn proto_kind_to_trust(kind: Option<MemoryKind>) -> MemoryTrust {
    match kind {
        Some(MemoryKind::MEMORY_KIND_FACT) => MemoryTrust::Fact,
        _ => MemoryTrust::Extracted {
            confidence: 0.5,
            evidence: Default::default(),
        },
    }
}

fn trust_to_proto_kind(trust: &MemoryTrust) -> MemoryKind {
    match trust {
        MemoryTrust::Fact => MemoryKind::MEMORY_KIND_FACT,
        MemoryTrust::Extracted { .. } => MemoryKind::MEMORY_KIND_EXTRACTED_MEMORY,
    }
}

fn datetime_to_timestamp(date_time: DateTime<Utc>) -> Timestamp {
    Timestamp {
        seconds: date_time.timestamp(),
        nanos: date_time.timestamp_subsec_nanos() as i32,
        ..Default::default()
    }
}

fn map_view_to_hashmap(map: &buffa::MapView<'_, &str, &str>) -> HashMap<String, String> {
    map.iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn query_result_to_proto(r: &MemoryQueryResult) -> MemoryEntry {
    let e = &r.memory_entry;
    MemoryEntry {
        id: e.id.to_string(),
        content: e.content.clone(),
        metadata: e.metadata.clone(),
        kind: EnumValue::from(trust_to_proto_kind(&e.trust)),
        seen_count: e.seen_count,
        first_seen: MessageField::some(datetime_to_timestamp(e.first_seen)),
        last_seen: MessageField::some(datetime_to_timestamp(e.last_seen)),
        expires_at: e
            .expires_at
            .map(|dt| MessageField::some(datetime_to_timestamp(dt)))
            .unwrap_or_default(),
        created_at: MessageField::some(datetime_to_timestamp(e.created_at)),
        score: r.score.value(),
        ..Default::default()
    }
}

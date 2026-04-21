// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use buffa::view::OwnedView;
use buffa::{EnumValue, MessageField};
use buffa_types::google::protobuf::Timestamp;
use chrono::{DateTime, Utc};
use connectrpc::{ConnectError, Context};
use uuid::Uuid;

use loci_config::{
    MemoryExtractorConfig as ConfigMemoryExtractorConfig, MergeStrategyConfig, ModelThinkingConfig,
    ModelThinkingEffortLevel,
};
use loci_core::error::MemoryStoreError;
use loci_core::memory::Score;
use loci_core::memory::extraction::{
    BestScoreMergeStrategy, DiscardReason, LlmMemoryExtractionStrategy,
    LlmMemoryExtractionStrategyParams, LlmMemoryMergeStrategy, MemoryExtractor,
    MemoryExtractorConfig, MemoryQueryOptions,
};
use loci_core::memory::store::{MemoryInput, MemoryQuery, MemoryQueryMode, MemoryStore};
use loci_core::memory::{MemoryEntry as CoreMemoryEntry, MemoryTrust};
use loci_core::memory_extraction::llm::ChunkingStrategy;
use loci_core::model_provider::text_generation::{
    TextGenerationModelProvider, ThinkingEffortLevel, ThinkingMode,
};
use loci_model_provider_ollama::classification::LlmClassificationModelProvider;

use crate::loci::memory::v1::{
    MemoryEntry, MemoryExtractionDiscardReason, MemoryExtractionDiscardedEntry, MemoryKind,
    MemoryServiceAddEntryRequestView, MemoryServiceAddEntryResponse,
    MemoryServiceDeleteEntryRequestView, MemoryServiceDeleteEntryResponse,
    MemoryServiceExtractRequestView, MemoryServiceExtractResponse,
    MemoryServiceGetEntryRequestView, MemoryServiceGetEntryResponse,
    MemoryServicePromoteRequestView, MemoryServicePromoteResponse,
    MemoryServicePruneExpiredRequestView, MemoryServicePruneExpiredResponse,
    MemoryServiceQueryRequestView, MemoryServiceQueryResponse,
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
        let input = MemoryInput::new(request.content.to_owned(), trust, metadata);

        let result = self
            .state
            .store()
            .add_entry(&input)
            .await
            .map_err(store_err)?;
        Ok((
            MemoryServiceAddEntryResponse {
                entry: MessageField::some(entry_to_proto(&result)),
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
        let result = self
            .state
            .store()
            .get_entry(&id)
            .await
            .map_err(store_err)?
            .ok_or_else(|| ConnectError::not_found(format!("memory entry not found: {id}")))?;
        Ok((
            MemoryServiceGetEntryResponse {
                entry: MessageField::some(entry_to_proto(&result)),
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
        let min_score = Score::try_new(request.min_score)
            .map_err(|_| ConnectError::invalid_argument("min_score must be in [0.0, 1.0]"))?;
        let max_results = NonZeroUsize::new(request.max_results as usize)
            .unwrap_or(NonZeroUsize::new(10).unwrap());
        let query = MemoryQuery::new(request.topic.to_owned(), MemoryQueryMode::Lookup)
            .with_max_results(max_results)
            .with_min_score(min_score)
            .with_filters(map_view_to_hashmap(&request.filters));
        let results = self.state.store().query(query).await.map_err(store_err)?;
        Ok((
            MemoryServiceQueryResponse {
                entries: results.iter().map(entry_to_proto).collect(),
                ..Default::default()
            },
            ctx,
        ))
    }

    async fn promote(
        &self,
        ctx: Context,
        request: OwnedView<MemoryServicePromoteRequestView<'static>>,
    ) -> Result<(MemoryServicePromoteResponse, Context), ConnectError> {
        let id = parse_uuid(request.id)?;
        let result = self
            .state
            .store()
            .promote(&id)
            .await
            .map_err(store_err)?
            .ok_or_else(|| ConnectError::not_found(format!("memory entry not found: {id}")))?;
        Ok((
            MemoryServicePromoteResponse {
                entry: MessageField::some(entry_to_proto(&result)),
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
        self.state
            .store()
            .delete_entry(&id)
            .await
            .map_err(store_err)?;
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
        self.state
            .store()
            .prune_expired()
            .await
            .map_err(store_err)?;
        Ok((
            MemoryServicePruneExpiredResponse {
                pruned: true,
                ..Default::default()
            },
            ctx,
        ))
    }

    async fn extract(
        &self,
        ctx: Context,
        request: OwnedView<MemoryServiceExtractRequestView<'static>>,
    ) -> Result<(MemoryServiceExtractResponse, Context), ConnectError> {
        let extraction_config = self.state.config().memory().extraction();
        let strategy = LlmMemoryExtractionStrategy::new(
            Arc::clone(self.state.llm_provider()),
            extraction_config.model().to_owned(),
        );
        let extractor_cfg = extraction_config.extractor();
        let classifier = Arc::new(LlmClassificationModelProvider::new(
            Arc::clone(self.state.llm_provider()),
            extractor_cfg.classification_model().to_owned(),
        ));
        let params = build_extraction_params(&request, extraction_config)?;
        let core_cfg = config_extractor_config_to_core(extractor_cfg);
        let result = match extractor_cfg.merge_strategy() {
            MergeStrategyConfig::BestScore => {
                MemoryExtractor::new(
                    Arc::clone(self.state.store()),
                    Arc::new(strategy),
                    Arc::new(BestScoreMergeStrategy),
                    Arc::clone(&classifier),
                    core_cfg,
                )
                .extract_memory_entries(request.text, &params)
                .await
            }
            MergeStrategyConfig::Llm { model } => {
                MemoryExtractor::new(
                    Arc::clone(self.state.store()),
                    Arc::new(strategy),
                    Arc::new(LlmMemoryMergeStrategy::new(
                        Arc::clone(self.state.llm_provider()),
                        model.clone(),
                    )),
                    Arc::clone(&classifier),
                    core_cfg,
                )
                .extract_memory_entries(request.text, &params)
                .await
            }
        };
        let result = result.map_err(|e| ConnectError::internal(e.to_string()))?;

        Ok((
            MemoryServiceExtractResponse {
                inserted: result.inserted().iter().map(entry_to_proto).collect(),
                merged: result.merged().iter().map(entry_to_proto).collect(),
                discarded: result
                    .discarded()
                    .iter()
                    .map(discarded_entry_to_proto)
                    .collect(),
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

#[allow(clippy::result_large_err)]
fn build_extraction_params(
    request: &MemoryServiceExtractRequestView<'_>,
    extraction_config: &loci_config::MemoryExtractionConfig,
) -> Result<LlmMemoryExtractionStrategyParams, ConnectError> {
    if let Some(min_confidence) = request.min_confidence {
        Score::try_new(min_confidence)
            .map_err(|_| ConnectError::invalid_argument("min_confidence must be in [0.0, 1.0]"))?;
    }

    Ok(LlmMemoryExtractionStrategyParams::new(
        match (
            request.guidelines.map(str::to_owned),
            extraction_config.guidelines(),
        ) {
            (Some(request), Some(config)) => Some(format!("{config}\n\n{request}")),
            (Some(request), None) => Some(request),
            (None, Some(config)) => Some(config.to_owned()),
            (None, None) => None,
        },
        map_view_to_hashmap(&request.metadata),
        request
            .max_entries
            .map(|value| value as usize)
            .or(extraction_config.max_entries()),
        request
            .min_confidence
            .or(extraction_config.min_confidence()),
        extraction_config.thinking().map(model_thinking_to_core),
        extraction_config
            .chunking()
            .map(|chunking| ChunkingStrategy::SentenceAware {
                chunk_size: NonZeroUsize::new(chunking.chunk_size())
                    .expect("chunk_size must be > 0"),
                overlap_size: chunking.overlap_size(),
            })
            .unwrap_or(ChunkingStrategy::WholeInput),
    ))
}

fn model_thinking_to_core(thinking: &ModelThinkingConfig) -> ThinkingMode {
    match thinking {
        ModelThinkingConfig::Enabled => ThinkingMode::Enabled,
        ModelThinkingConfig::Disabled => ThinkingMode::Disabled,
        ModelThinkingConfig::Effort { level } => ThinkingMode::Effort {
            level: match level {
                ModelThinkingEffortLevel::Low => ThinkingEffortLevel::Low,
                ModelThinkingEffortLevel::Medium => ThinkingEffortLevel::Medium,
                ModelThinkingEffortLevel::High => ThinkingEffortLevel::High,
            },
        },
        ModelThinkingConfig::Budgeted { max_tokens } => ThinkingMode::Budgeted {
            max_tokens: *max_tokens,
        },
    }
}

fn config_extractor_config_to_core(cfg: &ConfigMemoryExtractorConfig) -> MemoryExtractorConfig {
    MemoryExtractorConfig::new(
        MemoryQueryOptions::try_new(
            cfg.direct_search().max_results(),
            cfg.direct_search().min_score(),
        )
        .expect("invalid extractor config: direct_search"),
        MemoryQueryOptions::try_new(
            cfg.inverted_search().max_results(),
            cfg.inverted_search().min_score(),
        )
        .expect("invalid extractor config: inverted_search"),
        cfg.bayesian_seed_weight(),
        cfg.max_counter_increment(),
        cfg.max_counter(),
        cfg.auto_discard_threshold(),
    )
}

fn entry_to_proto(e: &CoreMemoryEntry) -> MemoryEntry {
    MemoryEntry {
        id: e.id().to_string(),
        content: e.content().to_owned(),
        metadata: e.metadata().clone(),
        kind: EnumValue::from(trust_to_proto_kind(e.trust())),
        seen_count: e.seen_count(),
        first_seen: e
            .first_seen()
            .map(|dt| MessageField::some(datetime_to_timestamp(dt)))
            .unwrap_or_default(),
        last_seen: e
            .last_seen()
            .map(|dt| MessageField::some(datetime_to_timestamp(dt)))
            .unwrap_or_default(),
        expires_at: e
            .expires_at()
            .map(|dt| MessageField::some(datetime_to_timestamp(dt)))
            .unwrap_or_default(),
        created_at: MessageField::some(datetime_to_timestamp(e.created_at())),
        score: e.trust().effective_score().value(),
        ..Default::default()
    }
}

fn discarded_entry_to_proto(
    discarded: &loci_core::memory::extraction::DiscardedEntry,
) -> MemoryExtractionDiscardedEntry {
    MemoryExtractionDiscardedEntry {
        content: discarded.content().to_owned(),
        reason: EnumValue::from(match discarded.reason() {
            DiscardReason::LowScore => {
                MemoryExtractionDiscardReason::MEMORY_EXTRACTION_DISCARD_REASON_LOW_SCORE
            }
            DiscardReason::ContradictsAFact => {
                MemoryExtractionDiscardReason::MEMORY_EXTRACTION_DISCARD_REASON_CONTRADICTS_A_FACT
            }
        }),
        ..Default::default()
    }
}

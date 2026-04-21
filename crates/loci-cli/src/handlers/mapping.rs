// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

use loci_config::{ModelThinkingConfig, ModelThinkingEffortLevel};
use loci_core::model_provider::text_generation::{
    ThinkingEffortLevel as CoreThinkingEffortLevel, ThinkingMode as CoreThinkingMode,
};

/// Maps a [`ModelThinkingConfig`] from the config layer to a
/// [`CoreThinkingMode`] understood by the core model-provider layer.
pub(crate) fn model_thinking_to_core(thinking: &ModelThinkingConfig) -> CoreThinkingMode {
    match thinking {
        ModelThinkingConfig::Enabled => CoreThinkingMode::Enabled,
        ModelThinkingConfig::Disabled => CoreThinkingMode::Disabled,
        ModelThinkingConfig::Effort { level } => CoreThinkingMode::Effort {
            level: match level {
                ModelThinkingEffortLevel::Low => CoreThinkingEffortLevel::Low,
                ModelThinkingEffortLevel::Medium => CoreThinkingEffortLevel::Medium,
                ModelThinkingEffortLevel::High => CoreThinkingEffortLevel::High,
            },
        },
        ModelThinkingConfig::Budgeted { max_tokens } => CoreThinkingMode::Budgeted {
            max_tokens: *max_tokens,
        },
    }
}

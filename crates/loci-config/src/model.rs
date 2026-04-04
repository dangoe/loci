// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-config.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

/// A named model alias referencing a provider.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// The provider name this model is served by.
    pub provider: String,

    /// The model identifier as understood by the provider (e.g. `"gpt-4.1"`).
    pub name: String,

    /// Optional generation tuning parameters for this model alias.
    #[serde(default)]
    pub tuning: Option<ModelTuningConfig>,
}

/// Provider-agnostic text generation tuning knobs.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ModelTuningConfig {
    /// Sampling temperature in `[0.0, 2.0]`.
    pub temperature: Option<f32>,
    /// Maximum number of output tokens.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling cutoff.
    pub top_p: Option<f32>,
    /// Repetition penalty.
    pub repeat_penalty: Option<f32>,
    /// Number of recent tokens to consider for repetition penalty.
    pub repeat_last_n: Option<u32>,
    /// Optional stop sequences.
    pub stop: Option<Vec<String>>,
    /// Keep-alive duration in seconds.
    pub keep_alive_secs: Option<u64>,
    /// Optional reasoning/thinking mode.
    pub thinking: Option<ModelThinkingConfig>,
    /// Provider-specific passthrough options.
    #[serde(default)]
    pub extra_params: HashMap<String, Value>,
}

/// Configuration for model reasoning/thinking behavior.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ModelThinkingConfig {
    Enabled,
    Disabled,
    Effort { level: ModelThinkingEffortLevel },
    Budgeted { max_tokens: u32 },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelThinkingEffortLevel {
    Low,
    Medium,
    High,
}

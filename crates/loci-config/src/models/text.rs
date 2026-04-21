// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

/// A named text-generation model config, nested under `[models.text.<name>]`.
#[derive(Debug, Clone, Deserialize)]
pub struct TextModelConfig {
    /// The provider name this model is served by.
    provider: String,

    /// The model identifier as understood by the provider (e.g. `"qwen3:0.6b"`).
    model: String,

    /// Optional generation tuning parameters for this model.
    #[serde(default)]
    tuning: Option<ModelTuningConfig>,
}

impl TextModelConfig {
    /// Constructs a new `TextModelConfig`.
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        tuning: Option<ModelTuningConfig>,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            tuning,
        }
    }

    /// Returns the provider name.
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// Returns the model identifier.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the optional tuning configuration.
    pub fn tuning(&self) -> Option<&ModelTuningConfig> {
        self.tuning.as_ref()
    }

    /// Sets the provider name.
    pub fn set_provider(&mut self, val: impl Into<String>) {
        self.provider = val.into();
    }
}

/// Provider-agnostic text generation tuning knobs.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ModelTuningConfig {
    /// Sampling temperature in `[0.0, 2.0]`.
    temperature: Option<f32>,
    /// Maximum number of output tokens.
    max_tokens: Option<u32>,
    /// Nucleus sampling cutoff.
    top_p: Option<f32>,
    /// Repetition penalty.
    repeat_penalty: Option<f32>,
    /// Number of recent tokens to consider for repetition penalty.
    repeat_last_n: Option<u32>,
    /// Optional stop sequences.
    stop: Option<Vec<String>>,
    /// Keep-alive duration in seconds.
    keep_alive_secs: Option<u64>,
    /// Optional reasoning/thinking mode.
    thinking: Option<ModelThinkingConfig>,
    /// Provider-specific passthrough options.
    #[serde(default)]
    extra: HashMap<String, Value>,
}

impl ModelTuningConfig {
    /// Constructs a new `ModelTuningConfig` with all fields specified.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        top_p: Option<f32>,
        repeat_penalty: Option<f32>,
        repeat_last_n: Option<u32>,
        stop: Option<Vec<String>>,
        keep_alive_secs: Option<u64>,
        thinking: Option<ModelThinkingConfig>,
        extra: HashMap<String, Value>,
    ) -> Self {
        Self {
            temperature,
            max_tokens,
            top_p,
            repeat_penalty,
            repeat_last_n,
            stop,
            keep_alive_secs,
            thinking,
            extra,
        }
    }

    /// Returns the sampling temperature.
    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// Returns the maximum number of output tokens.
    pub fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    /// Returns the top-p sampling cutoff.
    pub fn top_p(&self) -> Option<f32> {
        self.top_p
    }

    /// Returns the repetition penalty.
    pub fn repeat_penalty(&self) -> Option<f32> {
        self.repeat_penalty
    }

    /// Returns the repeat last-n window.
    pub fn repeat_last_n(&self) -> Option<u32> {
        self.repeat_last_n
    }

    /// Returns the stop sequences.
    pub fn stop(&self) -> Option<&[String]> {
        self.stop.as_deref()
    }

    /// Returns the keep-alive duration in seconds.
    pub fn keep_alive_secs(&self) -> Option<u64> {
        self.keep_alive_secs
    }

    /// Returns the optional thinking mode configuration.
    pub fn thinking(&self) -> Option<&ModelThinkingConfig> {
        self.thinking.as_ref()
    }

    /// Returns the extra provider-specific options.
    pub fn extra(&self) -> &HashMap<String, Value> {
        &self.extra
    }
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

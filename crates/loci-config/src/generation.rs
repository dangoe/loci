// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use serde::Deserialize;

/// Text-generation capability configuration, deserialized from
/// `[generation.text]`.
#[derive(Debug, Clone, Deserialize)]
pub struct TextGenerationConfig {
    /// Key in `[resources.models.text]` that selects the active model.
    model: String,
}

impl TextGenerationConfig {
    /// Returns the active text model key.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Sets the active text model key.
    pub fn set_model(&mut self, val: impl Into<String>) {
        self.model = val.into();
    }
}

/// Generation capability configuration, deserialized from `[generation]`.
#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    /// Text-generation settings, under `[generation.text]`.
    text: TextGenerationConfig,
}

impl GenerationConfig {
    /// Returns the text-generation configuration.
    pub fn text(&self) -> &TextGenerationConfig {
        &self.text
    }

    /// Returns a mutable reference to the text-generation configuration.
    pub fn text_mut(&mut self) -> &mut TextGenerationConfig {
        &mut self.text
    }
}

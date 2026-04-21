// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use std::collections::HashMap;

use serde::Deserialize;

use crate::memory::store::StoreConfig;
use crate::models::ModelsConfig;
pub use crate::providers::{ModelProviderConfig, ModelProviderKind};

/// All infrastructure registries, deserialized from `[resources]`.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ResourcesConfig {
    /// Named model provider definitions, each under
    /// `[resources.model_providers.<name>]`.
    #[serde(default)]
    model_providers: HashMap<String, ModelProviderConfig>,

    /// Text and embedding model registries, under `[resources.models.*]`.
    #[serde(default)]
    models: ModelsConfig,

    /// Named memory store definitions, each under
    /// `[resources.memory_stores.<name>]`.
    #[serde(default)]
    memory_stores: HashMap<String, StoreConfig>,
}

impl ResourcesConfig {
    /// Returns the named model provider definitions.
    pub fn model_providers(&self) -> &HashMap<String, ModelProviderConfig> {
        &self.model_providers
    }

    /// Returns a mutable reference to the model provider definitions.
    pub fn model_providers_mut(&mut self) -> &mut HashMap<String, ModelProviderConfig> {
        &mut self.model_providers
    }

    /// Returns the model registries.
    pub fn models(&self) -> &ModelsConfig {
        &self.models
    }

    /// Returns a mutable reference to the model registries.
    pub fn models_mut(&mut self) -> &mut ModelsConfig {
        &mut self.models
    }

    /// Returns the named memory store definitions.
    pub fn memory_stores(&self) -> &HashMap<String, StoreConfig> {
        &self.memory_stores
    }

    /// Returns a mutable reference to the memory store definitions.
    pub fn memory_stores_mut(&mut self) -> &mut HashMap<String, StoreConfig> {
        &mut self.memory_stores
    }
}

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-config.

use thiserror::Error;

/// Errors that can occur while loading or resolving configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The config file could not be read.
    #[error("could not read config file '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// The config file contains invalid TOML or does not match the schema.
    #[error("could not parse config file '{path}': {source}")]
    Parse {
        path: String,
        #[source]
        source: toml::de::Error,
    },

    /// An `env:VAR_NAME` reference could not be resolved.
    #[error("environment variable '{var}' referenced in config is not set")]
    EnvVar { var: String },

    /// A named key (provider, model, embedding, store) is missing from the config.
    #[error("config references '{key}' under [{section}] but no such entry is defined")]
    MissingKey { section: String, key: String },

    /// A `kind` value is recognised by the schema but not yet implemented.
    #[error("'{kind}' is not yet supported as a {context}")]
    UnsupportedKind { kind: String, context: String },
}

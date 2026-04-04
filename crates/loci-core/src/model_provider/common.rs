// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-core.

use std::collections::HashMap;

use serde_json::Value;

use crate::model_provider::error::ModelProviderError;

/// A type alias for model provider results, which are `Result<T, ModelProviderError>`.
pub type ModelProviderResult<T> = Result<T, ModelProviderError>;

/// Arbitrary model-provider-specific options forwarded verbatim in the request body
/// (e.g. `top_k`, `repeat_penalty`, vendor-specific flags).
pub type ModelProviderParams = HashMap<String, Value>;

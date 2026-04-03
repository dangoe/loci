// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-core.

use std::collections::HashMap;

use serde_json::Value;

use crate::backend::error::BackendError;

/// A type alias for backend results, which are `Result<T, BackendError>`.
pub type BackendResult<T> = Result<T, BackendError>;

/// Arbitrary backend-specific options forwarded verbatim in the request body
/// (e.g. `top_k`, `repeat_penalty`, vendor-specific flags).
pub type BackendParams = HashMap<String, Value>;

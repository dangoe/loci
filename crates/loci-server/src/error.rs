// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

use std::fmt;

use serde::Serialize;
use uuid::Uuid;

use loci_core::error::{ContextualizerError, MemoryStoreError};

/// A JSON-RPC 2.0 error object.
#[derive(Debug, Serialize)]
pub(crate) struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl RpcError {
    pub fn parse_error() -> Self {
        Self {
            code: -32700,
            message: "Parse error".into(),
            data: None,
        }
    }

    pub fn invalid_request() -> Self {
        Self {
            code: -32600,
            message: "Invalid Request".into(),
            data: None,
        }
    }

    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("Method not found: {method}"),
            data: None,
        }
    }

    pub fn invalid_params(detail: impl fmt::Display) -> Self {
        Self {
            code: -32602,
            message: format!("Invalid params: {detail}"),
            data: None,
        }
    }

    pub fn internal_error(detail: impl fmt::Display) -> Self {
        Self {
            code: -32603,
            message: format!("Internal error: {detail}"),
            data: None,
        }
    }

    /// Memory entry not found.
    pub fn memory_not_found(id: Uuid) -> Self {
        Self {
            code: -32000,
            message: format!("Memory not found: {id}"),
            data: None,
        }
    }

    /// Memory store error.
    pub fn store_error(detail: impl fmt::Display) -> Self {
        Self {
            code: -32001,
            message: format!("Memory store error: {detail}"),
            data: None,
        }
    }

    /// Model provider / contextualizer error.
    pub fn provider_error(detail: impl fmt::Display) -> Self {
        Self {
            code: -32002,
            message: format!("Provider error: {detail}"),
            data: None,
        }
    }
}

impl From<MemoryStoreError> for RpcError {
    fn from(e: MemoryStoreError) -> Self {
        match e {
            MemoryStoreError::NotFound(id) => Self::memory_not_found(id),
            other => Self::store_error(other),
        }
    }
}

impl From<ContextualizerError> for RpcError {
    fn from(e: ContextualizerError) -> Self {
        Self::provider_error(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loci_core::error::{ContextualizerError, MemoryStoreError};
    use loci_core::model_provider::error::ModelProviderError;

    #[test]
    fn test_rpc_error_parse_error_code() {
        assert_eq!(RpcError::parse_error().code, -32700);
    }

    #[test]
    fn test_rpc_error_invalid_request_code() {
        assert_eq!(RpcError::invalid_request().code, -32600);
    }

    #[test]
    fn test_rpc_error_method_not_found_code_and_message() {
        let e = RpcError::method_not_found("memory.foo");
        assert_eq!(e.code, -32601);
        assert!(e.message.contains("memory.foo"));
    }

    #[test]
    fn test_rpc_error_invalid_params_code() {
        assert_eq!(RpcError::invalid_params("bad").code, -32602);
    }

    #[test]
    fn test_rpc_error_internal_error_code() {
        assert_eq!(RpcError::internal_error("oops").code, -32603);
    }

    #[test]
    fn test_rpc_error_memory_not_found_code() {
        assert_eq!(RpcError::memory_not_found(Uuid::nil()).code, -32000);
    }

    #[test]
    fn test_rpc_error_store_error_code() {
        assert_eq!(RpcError::store_error("timeout").code, -32001);
    }

    #[test]
    fn test_rpc_error_provider_error_code() {
        assert_eq!(RpcError::provider_error("timeout").code, -32002);
    }

    #[test]
    fn test_from_memory_store_error_not_found_maps_to_minus_32000() {
        let e: RpcError = MemoryStoreError::NotFound(Uuid::nil()).into();
        assert_eq!(e.code, -32000);
    }

    #[test]
    fn test_from_memory_store_error_connection_maps_to_minus_32001() {
        let e: RpcError = MemoryStoreError::Connection("boom".into()).into();
        assert_eq!(e.code, -32001);
    }

    #[test]
    fn test_from_contextualizer_error_maps_to_minus_32002() {
        let e: RpcError = ContextualizerError::ModelProvider(ModelProviderError::Timeout).into();
        assert_eq!(e.code, -32002);
    }
}

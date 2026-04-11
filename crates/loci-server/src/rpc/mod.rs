// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

pub(crate) mod memory;

use serde::{Deserialize, Serialize};

use crate::error::RpcError;
use crate::state::AppState;

/// JSON-RPC 2.0 request envelope.
#[derive(Debug, Deserialize)]
pub(crate) struct RpcRequest {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
    pub id: Option<serde_json::Value>,
}

/// JSON-RPC 2.0 response envelope.
#[derive(Debug, Serialize)]
pub(crate) struct RpcResponse {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
    pub id: serde_json::Value,
}

impl RpcResponse {
    pub fn ok(id: serde_json::Value, result: impl Serialize) -> Self {
        Self {
            jsonrpc: "2.0",
            result: Some(serde_json::to_value(result).unwrap_or(serde_json::Value::Null)),
            error: None,
            id,
        }
    }

    pub fn err(id: serde_json::Value, error: RpcError) -> Self {
        Self {
            jsonrpc: "2.0",
            result: None,
            error: Some(error),
            id,
        }
    }
}

/// Dispatches a validated JSON-RPC request to the appropriate handler.
pub(crate) async fn dispatch(req: RpcRequest, state: &AppState) -> RpcResponse {
    let id = req.id.clone().unwrap_or(serde_json::Value::Null);

    let result = match req.method.as_str() {
        "memory.save" => memory::handle_save(req.params, state).await,
        "memory.get" => memory::handle_get(req.params, state).await,
        "memory.query" => memory::handle_query(req.params, state).await,
        "memory.update" => memory::handle_update(req.params, state).await,
        "memory.set_tier" => memory::handle_set_tier(req.params, state).await,
        "memory.delete" => memory::handle_delete(req.params, state).await,
        "memory.prune_expired" => memory::handle_prune_expired(state).await,
        _ => Err(RpcError::method_not_found(&req.method)),
    };

    match result {
        Ok(value) => RpcResponse::ok(id, value),
        Err(e) => RpcResponse::err(id, e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpc_response_ok_sets_jsonrpc_version() {
        let resp = RpcResponse::ok(serde_json::Value::Number(1.into()), "data");
        assert_eq!(resp.jsonrpc, "2.0");
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_rpc_response_err_sets_jsonrpc_version() {
        let resp = RpcResponse::err(
            serde_json::Value::Number(1.into()),
            RpcError::internal_error("boom"),
        );
        assert_eq!(resp.jsonrpc, "2.0");
        assert!(resp.result.is_none());
        assert!(resp.error.is_some());
    }
}

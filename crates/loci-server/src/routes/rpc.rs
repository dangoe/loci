// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::IntoResponse;

use crate::error::RpcError;
use crate::rpc::{RpcRequest, RpcResponse, dispatch};
use crate::state::AppState;

pub(crate) async fn rpc_handler(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> impl IntoResponse {
    // Parse the raw body as JSON; return a JSON-RPC parse error on failure.
    let req: RpcRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(_) => {
            return Json(RpcResponse::err(
                serde_json::Value::Null,
                RpcError::parse_error(),
            ))
            .into_response();
        }
    };

    // Validate jsonrpc version.
    if req.jsonrpc != "2.0" {
        let id = req.id.clone().unwrap_or(serde_json::Value::Null);
        return Json(RpcResponse::err(id, RpcError::invalid_request())).into_response();
    }

    let response = dispatch(req, &state).await;
    Json(response).into_response()
}

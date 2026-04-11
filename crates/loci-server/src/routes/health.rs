// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

use axum::Json;

pub(crate) async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_handler_returns_ok_status() {
        let Json(body) = health_handler().await;
        assert_eq!(body["status"], "ok");
        assert!(body["version"].as_str().is_some());
    }
}

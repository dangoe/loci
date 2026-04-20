// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

mod health;
mod openai;

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::routing::get;
use connectrpc::Router as ConnectRouter;
use log::info;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use loci_config::load_config;
use loci_core::memory::store::MemoryStore;
use loci_core::model_provider::text_generation::TextGenerationModelProvider;

use crate::cli::ServerArgs;
use crate::infra::{build_llm_provider, build_store};
use crate::loci::generate::v1::GenerateServiceExt as _;
use crate::loci::memory::v1::MemoryServiceExt as _;
use crate::service::generate::GenerateServiceImpl;
use crate::service::memory::MemoryServiceImpl;
use crate::state::AppState;

pub(crate) fn build_router<M, E>(state: Arc<AppState<M, E>>) -> Router
where
    M: MemoryStore + 'static,
    E: TextGenerationModelProvider + 'static,
{
    let connect =
        Arc::new(MemoryServiceImpl::new(Arc::clone(&state))).register(ConnectRouter::new());
    let connect = Arc::new(GenerateServiceImpl::new(Arc::clone(&state))).register(connect);

    Router::new()
        .route("/v1/health", get(health::health_handler))
        .nest("/openai", openai::openai_router::<M, E>())
        .fallback_service(connect.into_axum_router())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Entry point called from `main.rs`. Builds infrastructure, constructs the axum
/// router, and starts the TCP listener.
pub(crate) async fn run_server(args: ServerArgs) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = resolve_config_path(args.config);
    info!("Loading config from {}", config_path.display());
    let config = load_config(&config_path)?;

    let store = Arc::new(build_store(&config).await?);
    let llm_provider = Arc::new(build_llm_provider(&config)?);
    let state = Arc::new(AppState {
        store,
        llm_provider,
        config: Arc::new(config),
    });

    let router = build_router(state);
    let addr: std::net::SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    info!("loci-server listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

fn resolve_config_path(cli_value: Option<PathBuf>) -> PathBuf {
    if let Some(path) = cli_value {
        return path;
    }
    dirs::config_dir()
        .unwrap_or_else(|| {
            log::warn!("could not determine system config directory, falling back to '.'");
            PathBuf::from(".")
        })
        .join("loci")
        .join("config.toml")
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_resolve_config_path_uses_cli_value_when_set() {
        let p = resolve_config_path(Some(PathBuf::from("/tmp/my-config.toml")));
        assert_eq!(p, PathBuf::from("/tmp/my-config.toml"));
    }

    #[test]
    fn test_resolve_config_path_falls_back_to_xdg_when_empty() {
        let p = resolve_config_path(None);
        assert!(p.ends_with("loci/config.toml"));
    }
}

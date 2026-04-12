// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

include!(concat!(env!("OUT_DIR"), "/_connectrpc.rs"));

pub mod cli;
pub(crate) mod infra;
pub(crate) mod routes;
pub(crate) mod service;
pub(crate) mod state;
#[cfg(any(feature = "testing", test))]
pub mod testing;

pub use cli::ServerArgs;

pub async fn run(args: ServerArgs) -> Result<(), Box<dyn std::error::Error>> {
    routes::run_server(args).await
}

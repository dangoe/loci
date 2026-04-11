// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

mod cli;
mod error;
mod infra;
mod routes;
mod rpc;
mod state;

use clap::Parser;
use log::{LevelFilter, error, info};

use crate::cli::ServerArgs;

#[tokio::main]
async fn main() {
    let args = ServerArgs::parse();
    setup_logging(args.verbose);
    info!("Starting loci-server");
    if let Err(e) = crate::routes::run_server(args).await {
        error!("fatal: {e}");
        std::process::exit(1);
    }
}

fn setup_logging(verbose: bool) {
    env_logger::init();
    log::set_max_level(if verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    });
}

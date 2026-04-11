// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

use clap::Parser;
use log::{LevelFilter, error, info};

use loci_server::ServerArgs;

#[tokio::main]
async fn main() {
    let args = ServerArgs::parse();

    setup_logging(args.verbose);

    info!("Starting loci-server");

    if let Err(e) = loci_server::run(args).await {
        error!("fatal: {e}");
        std::process::exit(1);
    }
}

fn setup_logging(verbose: bool) {
    let default_level = if verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };
    env_logger::Builder::new()
        .filter_level(default_level)
        .parse_default_env()
        .init();
}

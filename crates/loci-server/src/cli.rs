// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(name = "loci-server", about = "loci REST JSON-RPC server")]
pub(crate) struct ServerArgs {
    /// Path to the loci configuration file.
    #[arg(long, short, env = "LOCI_CONFIG", hide_env_values = true)]
    pub config: Option<PathBuf>,

    /// Host address to listen on.
    #[arg(long, default_value = "127.0.0.1", env = "LOCI_SERVER_HOST")]
    pub host: String,

    /// Port to listen on.
    #[arg(long, default_value_t = 8080, env = "LOCI_SERVER_PORT")]
    pub port: u16,

    /// Enable verbose (debug-level) logging.
    #[arg(long, short)]
    pub verbose: bool,
}

use std::path::PathBuf;

use clap::{Parser, command};

use crate::commands::Command;

/// Top-level CLI arguments and global options.
#[derive(Parser)]
#[command(name = "loci", about = "loci CLI")]
pub struct Cli {
    /// Path to the TOML configuration file.
    #[arg(long, short, env = "LOCI_CONFIG", hide_env_values = true)]
    config: Option<PathBuf>,

    /// Verbose output.
    #[arg(long, short)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

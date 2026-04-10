use std::path::PathBuf;

use clap::Parser;

use crate::commands::Command;

/// Top-level CLI arguments and global options.
#[derive(Parser)]
#[command(name = "loci", about = "loci CLI")]
pub struct Cli {
    /// Path to the TOML configuration file.
    #[arg(long, short, env = "LOCI_CONFIG", hide_env_values = true)]
    pub config: Option<PathBuf>,

    /// Verbose output.
    #[arg(long, short)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Command,
}

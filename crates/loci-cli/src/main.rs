// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

//! Binary entry-point for the `loci` CLI.
//!
//! All infrastructure configuration (providers, stores, models, embeddings) is
//! read from a TOML config file (default: `~/.config/loci/config.toml`).
//! The CLI itself only exposes operational flags and sub-commands.

mod cli;
mod commands;
#[cfg(test)]
mod fixture;
mod handlers;
mod infra;
#[cfg(test)]
mod mock;

use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "systemd-journal-logger")]
use systemd_journal_logger::JournalLog;

use clap::Parser;
use log::{LevelFilter, error, info};

use loci_config::load_config;

use crate::cli::Cli;
use crate::commands::{Command, GenerateCommand};
use crate::handlers::CommandHandler;
use crate::handlers::config::ConfigCommandHandler;
use crate::handlers::generate::GenerateCommandHandler;
use crate::handlers::memory::MemoryCommandHandler;
use crate::infra::{build_llm_provider, build_store};

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    setup_logging(cli.verbose);

    info!("Starting loci CLI");

    if let Err(e) = run(cli).await {
        error!("error: {e}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = resolve_config_path(cli.config);

    // Config commands are handled before loading the config file,
    // as the file may not exist yet.
    if let Command::Config { command } = cli.command {
        let handler = ConfigCommandHandler::new(&config_path);
        return handler.handle(command, &mut std::io::stdout()).await;
    }

    info!("Loading config from {}", config_path.display());
    let config = load_config(&config_path)?;

    match cli.command {
        Command::Memory { command } => {
            let store = build_store(&config).await?;
            let handler = MemoryCommandHandler::new(&store);
            handler.handle(command, &mut std::io::stdout()).await
        }
        Command::Generate { args } => {
            let store = Arc::new(build_store(&config).await?);
            let llm_provider = Arc::new(build_llm_provider(&config)?);
            let handler =
                GenerateCommandHandler::new(Arc::clone(&store), Arc::clone(&llm_provider), &config);
            handler
                .handle(GenerateCommand::Execute(args), &mut std::io::stdout())
                .await
        }
        Command::Config { .. } => unreachable!("handled above"),
    }
}

/// Resolves the config file path: uses the provided value when set,
/// otherwise falls back to `~/.config/loci/config.toml`.
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

fn setup_logging(verbose: bool) {
    #[cfg(feature = "systemd-journal-logger")]
    {
        let journald_ok = JournalLog::new()
            .and_then(|journal| journal.install())
            .is_ok();
        if !journald_ok {
            eprintln!("warning: failed to connect to journald, falling back to env_logger");
            env_logger::init();
        }
    }

    #[cfg(not(feature = "systemd-journal-logger"))]
    {
        env_logger::init();
    }

    log::set_max_level(if verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_config_path_uses_cli_value_when_set() {
        let p = resolve_config_path(Some(PathBuf::from("/tmp/my-config.toml")));
        assert_eq!(p, PathBuf::from("/tmp/my-config.toml"));
    }

    #[test]
    fn resolve_config_path_falls_back_to_xdg_when_empty() {
        let p = resolve_config_path(None);
        assert!(p.ends_with("loci/config.toml"));
    }
}

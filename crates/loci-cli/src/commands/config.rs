// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

use clap::Subcommand;

/// Config sub-commands.
#[derive(Subcommand)]
pub enum ConfigCommand {
    /// Scaffold a default configuration file at the config path.
    Init,
}

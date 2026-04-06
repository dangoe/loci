pub mod memory;

use clap::{Subcommand, command};

use crate::commands::memory::MemoryCommand;

/// Available sub-commands.
#[derive(Subcommand)]
pub enum Command {
    /// Memory store operations.
    Memory {
        #[command(subcommand)]
        command: MemoryCommand,
    },
}

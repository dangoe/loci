pub mod config;
pub mod generate;
pub mod memory;
mod parse;

use clap::Subcommand;

pub use config::ConfigCommand;
pub use generate::{GenerateArgs, GenerateCommand};
pub use memory::MemoryCommand;

/// Available sub-commands.
#[derive(Subcommand)]
pub enum Command {
    /// Memory store operations.
    #[command(name = "memory", alias = "mem")]
    Memory {
        #[command(subcommand)]
        command: MemoryCommand,
    },
    /// Generate text from a prompt.
    #[command(name = "generate", alias = "gen")]
    Generate {
        #[command(flatten)]
        args: GenerateArgs,
    },
    /// Configuration management.
    #[command(name = "config")]
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
}

pub mod generate;
pub mod memory;

use clap::Subcommand;

pub use generate::GenerateArgs;
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
}

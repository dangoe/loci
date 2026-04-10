use clap::Args;

use crate::commands::parse::parse_key_value;

/// Domain-level dispatch type for the `generate` command.
///
/// This is not a clap type — the CLI still parses directly into [`GenerateArgs`].
/// The enum exists for structural parity with [`super::MemoryCommand`] and
/// [`super::ConfigCommand`], allowing [`GenerateCommandHandler`] to implement
/// [`crate::handlers::CommandHandler<GenerateCommand, _>`].
///
/// [`GenerateCommandHandler`]: crate::handlers::generate::GenerateCommandHandler
pub enum GenerateCommand {
    Execute(GenerateArgs),
}

/// Arguments for the `generate` command, controlling prompt generation and memory injection behavior.
#[derive(Args)]
pub struct GenerateArgs {
    /// The prompt to process.
    pub prompt: String,

    /// Optional override for the system prompt used for generation. If not set, the default system prompt will be used.
    #[arg(long)]
    pub system: Option<String>,

    /// System prompt mode, which controls how the provided system prompt interacts with the default system prompt.
    #[arg(long, value_enum, default_value_t = GenerateSystemMode::Append)]
    pub system_mode: GenerateSystemMode,

    /// Maximum number of memory entries to inject into the prompt.
    #[arg(long, default_value_t = 5)]
    pub max_memory_entries: usize,

    /// Minimum similarity score for memory retrieval (0.0–1.0).
    #[arg(long, default_value_t = 0.5)]
    pub min_score: f64,

    /// Memory mode for generation, which controls whether and how memory is retrieved and injected into the prompt.
    #[arg(long, value_enum, default_value_t = GenerateMemoryMode::Auto)]
    pub memory_mode: GenerateMemoryMode,

    /// Memory meta data filter criteria
    #[arg(long = "filters", value_parser = parse_key_value)]
    pub filters: Vec<(String, String)>,

    /// Print debug info about the contextualization process, such as retrieved memory entries.
    #[arg(long)]
    pub debug_flags: Vec<GenerateDebugFlags>,
}

/// Memory mode for the `gen` command, controlling memory retrieval and injection behavior.
#[derive(clap::ValueEnum, PartialEq, Eq, Clone, Debug)]
pub enum GenerateMemoryMode {
    /// Retrieves and injects memory entries into the prompt based on the configured contextualization settings.
    Auto,
    /// Skips memory retrieval and injection, generating a response based solely on the prompt.
    Off,
}

/// Debug flags for the `gen` command, which prints additional info about the contextualization process when set.
#[derive(clap::ValueEnum, PartialEq, Eq, Clone, Debug)]
pub enum GenerateDebugFlags {
    /// Print the memory entries that were injected into the model provider prompt.
    Memory,
}

/// Memory mode for generation, which controls whether and how memory is retrieved and injected into the prompt.
#[derive(clap::ValueEnum, PartialEq, Eq, Clone, Debug)]
pub enum GenerateSystemMode {
    /// Append given system prompt to the default system prompt.
    Append,
    /// Replace default system prompt with the given system prompt.
    Replace,
}

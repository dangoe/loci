use clap::Subcommand;

/// Config sub-commands.
#[derive(Subcommand)]
pub enum ConfigCommand {
    /// Scaffold a default configuration file at the config path.
    Init,
}

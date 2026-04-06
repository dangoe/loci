use std::error::Error;

pub mod memory;

/// A trait for handling commands and returning a result.
pub trait CommandHandler<'a, C, W: std::io::Write> {
    /// Handles the given command and returns a result.
    fn handle(
        &self,
        command: C,
        out: &'a mut W,
    ) -> impl Future<Output = Result<(), Box<dyn Error>>> + Send + '_;
}

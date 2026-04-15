// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

use std::{error::Error as StdError, io::Write};

pub mod config;
pub mod generate;
mod json;
pub(crate) mod mapping;
pub mod memory;

/// A trait for handling commands and returning a result.
pub trait CommandHandler<'a, C, W: Write> {
    /// Handles the given command and returns a result.
    fn handle(
        &self,
        command: C,
        out: &'a mut W,
    ) -> impl Future<Output = Result<(), Box<dyn StdError>>> + Send + '_;
}

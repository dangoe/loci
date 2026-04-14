// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

//! Library crate for the `loci` CLI.
//!
//! Exposes commands, handlers, and shared test infrastructure so that
//! integration tests can construct handlers with mock dependencies.

pub mod cli;
pub mod commands;
pub mod handlers;

#[cfg(any(feature = "testing", test))]
pub mod testing;

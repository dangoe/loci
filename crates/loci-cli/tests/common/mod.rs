// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

//! Shared test infrastructure re-exported from upstream crates.

// Each test binary gets its own copy of this module; items not needed by a
// particular binary show up as dead_code warnings even though they are used
// by other binaries.
#![allow(dead_code, unused_imports)]

pub use loci_cli::testing::{TestCli, minimal_ollama_config, mock_config};
pub use loci_core::testing::{
    EntryBehavior, MockStore, MockStoreErrorKind, MockTextGenerationModelProvider,
    ProviderBehavior, QueryBehavior, UnitBehavior, make_result,
};

// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-wire.

mod provider;
mod store;
#[cfg(any(feature = "testing", test))]
pub mod testing;

pub use provider::{
    AnyModelProvider, build_llm_provider, build_ollama_provider, resolve_embedding_provider,
    resolve_llm_provider,
};
pub use store::build_store;

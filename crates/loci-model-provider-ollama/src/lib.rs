// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-model-provider-ollama.

pub mod classification;
pub mod provider;

#[cfg(any(feature = "testing", test))]
pub mod testing;

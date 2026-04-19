// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::pin::Pin;

pub mod classification;
pub mod contextualization;
pub mod embedding;
pub mod error;
pub mod memory;
pub mod memory_extraction;
pub mod model_provider;
pub mod store;
#[cfg(any(feature = "testing", test))]
pub mod testing;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

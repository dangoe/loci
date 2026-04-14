// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

//! Shared test infrastructure re-exported from upstream crates.

// Each test binary gets its own copy of this module; items not needed by a
// particular binary show up as dead_code warnings even though they are used
// by other binaries.
#![allow(dead_code, unused_imports)]

use std::sync::Arc;

use connectrpc::{ConnectError, ErrorCode};
use loci_server::loci::generate::v1::GenerateServiceGenerateRequest;

pub use loci_core::testing::{
    EntryBehavior, MockStore, MockStoreErrorKind, MockTextGenerationModelProvider,
    ProviderBehavior, QueryBehavior, UnitBehavior, make_result,
};
pub use loci_server::testing::{TestServer, mock_config};

/// Sends a generate request and extracts the trailing error from the stream.
pub async fn generate_error(
    server: &TestServer,
    request: GenerateServiceGenerateRequest,
) -> ConnectError {
    match server.generate_client().generate(request).await {
        Err(error) => error,
        Ok(mut stream) => {
            let message = stream
                .message()
                .await
                .expect("stream should decode the terminal frame");
            assert!(message.is_none());
            stream
                .error()
                .cloned()
                .expect("stream should surface a trailing connect error")
        }
    }
}

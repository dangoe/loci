// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-server.

fn main() {
    connectrpc_build::Config::new()
        .files(&[
            "proto/loci/memory/v1/memory.proto",
            "proto/loci/generate/v1/generate.proto",
        ])
        .includes(&["proto"])
        .include_file("_connectrpc.rs")
        .compile()
        .unwrap();
}

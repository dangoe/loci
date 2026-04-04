// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT
// This file is part of loci-core.

use std::{error::Error, fmt};

/// Errors returned by a model provider (embedding or text generation).
#[derive(Debug)]
pub enum ModelProviderError {
    Http {
        message: String,
        status: Option<u16>,
    },
    Transport {
        message: String,
    },
    Parse {
        message: String,
    },
    Timeout,
    RateLimited,
    InvalidRequest {
        message: String,
    },
    Other {
        message: String,
    },
}

impl fmt::Display for ModelProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http { message, status } => {
                if let Some(code) = status {
                    write!(f, "HTTP error ({code}): {message}")
                } else {
                    write!(f, "HTTP error: {message}")
                }
            }
            Self::Transport { message } => write!(f, "transport error: {message}"),
            Self::Parse { message } => write!(f, "parse error: {message}"),
            Self::Timeout => write!(f, "request timed out"),
            Self::RateLimited => write!(f, "rate limited by model provider"),
            Self::InvalidRequest { message } => write!(f, "invalid request: {message}"),
            Self::Other { message } => write!(f, "{message}"),
        }
    }
}

impl Error for ModelProviderError {}

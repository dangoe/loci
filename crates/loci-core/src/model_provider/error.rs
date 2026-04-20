// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::{error::Error, fmt};

/// Errors returned by a model provider (embedding or text generation).
#[non_exhaustive]
#[derive(Debug)]
pub enum ModelProviderError {
    Http {
        message: String,
        status: Option<u16>,
    },
    Transport(String),
    Parse(String),
    Timeout,
    RateLimited,
    InvalidRequest(String),
    Other(String),
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
            Self::Transport(msg) => write!(f, "transport error: {msg}"),
            Self::Parse(msg) => write!(f, "parse error: {msg}"),
            Self::Timeout => write!(f, "request timed out"),
            Self::RateLimited => write!(f, "rate limited by model provider"),
            Self::InvalidRequest(msg) => write!(f, "invalid request: {msg}"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl Error for ModelProviderError {}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(ModelProviderError::Timeout, "request timed out")]
    #[case(ModelProviderError::RateLimited, "rate limited by model provider")]
    #[case(
        ModelProviderError::Http { message: "not found".to_string(), status: Some(404) },
        "HTTP error (404): not found"
    )]
    #[case(
        ModelProviderError::Http { message: "unknown".to_string(), status: None },
        "HTTP error: unknown"
    )]
    #[case(
        ModelProviderError::Transport("conn refused".to_string()),
        "transport error: conn refused"
    )]
    #[case(
        ModelProviderError::Parse("invalid json".to_string()),
        "parse error: invalid json"
    )]
    #[case(
        ModelProviderError::InvalidRequest("bad model".to_string()),
        "invalid request: bad model"
    )]
    #[case(
        ModelProviderError::Other("something unexpected".to_string()),
        "something unexpected"
    )]
    fn test_display_formats_error_message_for_each_variant(
        #[case] err: ModelProviderError,
        #[case] expected: &str,
    ) {
        assert_eq!(err.to_string(), expected);
    }
}

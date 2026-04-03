use std::{error::Error, fmt};

#[derive(Debug)]
pub enum BackendError {
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

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http { message, status } => {
                if let Some(code) = status {
                    write!(f, "HTTP error ({code}): {message}")
                } else {
                    write!(f, "HTTP error: {message}")
                }
            }
            Self::Transport { message } => write!(f, "Transport error: {message}"),
            Self::Parse { message } => write!(f, "Parse error: {message}"),
            Self::Timeout => write!(f, "Request timed out"),
            Self::RateLimited => write!(f, "Rate limited by backend"),
            Self::InvalidRequest { message } => write!(f, "Invalid request: {message}"),
            Self::Other { message } => write!(f, "{message}"),
        }
    }
}

impl Error for BackendError {}

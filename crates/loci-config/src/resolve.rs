// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use crate::ConfigError;

/// Resolves a secret value that may be a literal string or an `env:VAR_NAME`
/// reference.
///
/// - If `value` starts with `env:`, the remainder is treated as an environment
///   variable name and its value is returned.
/// - Otherwise the value is returned as-is.
pub(crate) fn resolve_secret(value: &str) -> Result<String, ConfigError> {
    if let Some(var_name) = value.strip_prefix("env:") {
        std::env::var(var_name).map_err(|_| ConfigError::EnvVar {
            var: var_name.to_string(),
        })
    } else {
        Ok(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_value_is_returned_unchanged() {
        assert_eq!(resolve_secret("sk-abc123").unwrap(), "sk-abc123");
    }

    #[test]
    fn test_env_prefix_resolves_variable() {
        // SAFETY: single-threaded test process; no other threads read this var.
        unsafe { std::env::set_var("LOCI_RESOLVE_TEST", "hello") };
        assert_eq!(resolve_secret("env:LOCI_RESOLVE_TEST").unwrap(), "hello");
        // SAFETY: same as above.
        unsafe { std::env::remove_var("LOCI_RESOLVE_TEST") };
    }

    #[test]
    fn test_missing_env_var_is_an_error() {
        // SAFETY: single-threaded test process; no other threads read this var.
        unsafe { std::env::remove_var("LOCI_RESOLVE_MISSING") };
        let err = resolve_secret("env:LOCI_RESOLVE_MISSING").unwrap_err();
        assert!(matches!(err, ConfigError::EnvVar { var } if var == "LOCI_RESOLVE_MISSING"));
    }
}

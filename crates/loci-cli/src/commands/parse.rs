// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

/// Parses a `key=value` string into a `(String, String)` pair.
pub fn parse_key_value(s: &str) -> Result<(String, String), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("expected KEY=VALUE, got: {s:?}"))?;
    Ok((k.to_string(), v.to_string()))
}

#[cfg(test)]
mod tests {
    use crate::commands::parse::parse_key_value;

    #[test]
    fn parse_key_value_valid() {
        let (k, v) = parse_key_value("project=ai-memory").unwrap();
        assert_eq!(k, "project");
        assert_eq!(v, "ai-memory");
    }

    #[test]
    fn parse_key_value_when_value_contains_equals() {
        let (k, v) = parse_key_value("url=http://host:1234").unwrap();
        assert_eq!(k, "url");
        assert_eq!(v, "http://host:1234");
    }

    #[test]
    fn parse_key_value_missing_equals_returns_err() {
        assert!(parse_key_value("noequalssign").is_err());
    }

    #[test]
    fn parse_key_value_empty_value_is_allowed() {
        let (k, v) = parse_key_value("key=").unwrap();
        assert_eq!(k, "key");
        assert_eq!(v, "");
    }
}

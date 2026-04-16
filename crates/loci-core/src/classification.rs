// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-core.

use std::future::Future;

use crate::model_provider::error::ModelProviderError;

/// The four possible classifications a pipeline hit can receive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitClass {
    Duplicate,
    Complementary,
    Contradiction,
    Unrelated,
}

/// Errors returned by a classification model provider.
#[derive(Debug)]
pub enum ClassificationError {
    ModelProvider(ModelProviderError),
    Parse(String),
}

impl std::fmt::Display for ClassificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelProvider(e) => write!(f, "model provider error: {e}"),
            Self::Parse(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

impl std::error::Error for ClassificationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ModelProvider(e) => Some(e),
            Self::Parse(_) => None,
        }
    }
}

/// A model provider capable of classifying a hit against a candidate memory entry.
pub trait ClassificationModelProvider: Send + Sync {
    fn classify_hit(
        &self,
        candidate: &str,
        hit: &str,
    ) -> impl Future<Output = Result<HitClass, ClassificationError>> + Send;
}

/// Maps a string label to a [`HitClass`] variant (case-insensitive, trimmed).
pub fn parse_hit_class(s: &str) -> Option<HitClass> {
    match s.trim().to_ascii_lowercase().as_str() {
        "duplicate" => Some(HitClass::Duplicate),
        "complementary" => Some(HitClass::Complementary),
        "contradiction" => Some(HitClass::Contradiction),
        "unrelated" => Some(HitClass::Unrelated),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_hit_class_parse_all_variants() {
        assert_eq!(parse_hit_class("duplicate"), Some(HitClass::Duplicate));
        assert_eq!(
            parse_hit_class("complementary"),
            Some(HitClass::Complementary)
        );
        assert_eq!(
            parse_hit_class("contradiction"),
            Some(HitClass::Contradiction)
        );
        assert_eq!(parse_hit_class("unrelated"), Some(HitClass::Unrelated));
    }

    #[test]
    fn test_hit_class_parse_case_insensitive() {
        assert_eq!(parse_hit_class("DUPLICATE"), Some(HitClass::Duplicate));
        assert_eq!(
            parse_hit_class("COMPLEMENTARY"),
            Some(HitClass::Complementary)
        );
        assert_eq!(
            parse_hit_class("CONTRADICTION"),
            Some(HitClass::Contradiction)
        );
        assert_eq!(parse_hit_class("UNRELATED"), Some(HitClass::Unrelated));
    }

    #[test]
    fn test_hit_class_parse_unknown_returns_none() {
        assert_eq!(parse_hit_class("unknown"), None);
        assert_eq!(parse_hit_class(""), None);
        assert_eq!(parse_hit_class("partial"), None);
    }

    #[test]
    fn test_classification_error_model_provider_display() {
        let err = ClassificationError::ModelProvider(ModelProviderError::Timeout);
        assert!(err.to_string().contains("model provider"));
    }

    #[test]
    fn test_classification_error_parse_display() {
        let err = ClassificationError::Parse("unexpected token".to_string());
        assert!(err.to_string().contains("parse error"));
    }

    #[test]
    fn test_classification_error_model_provider_has_source() {
        let err = ClassificationError::ModelProvider(ModelProviderError::Timeout);
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn test_classification_error_parse_has_no_source() {
        let err = ClassificationError::Parse("bad output".to_string());
        assert!(std::error::Error::source(&err).is_none());
    }
}

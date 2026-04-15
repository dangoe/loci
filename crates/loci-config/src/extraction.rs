// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use serde::Deserialize;

use crate::model::ModelThinkingConfig;

/// Memory tier assigned to extracted entries, deserialized from
/// `[memory.extraction]`.
///
/// Mirrors `loci_core::MemoryTier` without `Ephemeral`, which is not
/// meaningful for persisted extraction output.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionTierConfig {
    /// New persisted memory with shorter TTL and lower retrieval priority.
    #[default]
    Candidate,
    /// Promoted memory with longer TTL and higher retrieval priority.
    Stable,
    /// Manually curated long-term memory that does not expire.
    Core,
}

/// Chunking settings for splitting input text before LLM extraction,
/// deserialized from `[memory.extraction.chunking]`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkingConfig {
    /// Maximum number of characters per chunk. The splitter finishes the
    /// current word before cutting, so the actual chunk may be slightly
    /// longer.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,

    /// Characters of overlap between consecutive chunks.
    #[serde(default = "default_overlap_size")]
    pub overlap_size: usize,
}

fn default_chunk_size() -> usize {
    2500
}

fn default_overlap_size() -> usize {
    200
}

/// LLM-based memory extraction configuration, deserialized from
/// `[memory.extraction]`.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryExtractionConfig {
    /// Name of the text model in `[models.text]` used to perform extraction.
    pub model: String,

    /// Memory tier assigned to every extracted entry.
    /// Defaults to `candidate` when absent.
    #[serde(default)]
    pub default_tier: ExtractionTierConfig,

    /// Optional hard cap on the number of entries extracted per run.
    /// Applied both as a prompt hint and as a post-processing limit.
    pub max_entries: Option<usize>,

    /// Optional additional guidelines appended to the extraction prompt to
    /// guide or constrain what the model should extract.
    pub guidelines: Option<String>,

    /// Optional thinking mode override for the extraction model. When absent
    /// the model's default thinking behaviour is used. Extraction produces
    /// structured JSON, so `disabled` is usually the right choice.
    pub thinking: Option<ModelThinkingConfig>,

    /// Optional text chunking settings. When absent, the full input is sent
    /// to the model as a single call.
    pub chunking: Option<ChunkingConfig>,
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{AppConfig, load_config};

    use super::*;

    fn write_temp_config(content: &str) -> tempfile::NamedTempFile {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    const BASE: &str = r#"
[memory.backends.qdrant]
kind = "qdrant"
url  = "http://localhost:6334"
collection = "mem"

[memory.config]
backend = "qdrant"

[routing.text]
default = "x"

[routing.embedding]
default = "x"

[routing.memory]
default = "qdrant"
"#;

    fn config_with_extraction(extraction_toml: &str) -> AppConfig {
        let raw = format!("{BASE}\n{extraction_toml}");
        let f = write_temp_config(&raw);
        load_config(f.path()).unwrap()
    }

    #[test]
    fn minimal_extraction_block_is_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"
"#,
        );
        assert_eq!(cfg.memory.extraction.model, "default");
        assert!(matches!(
            cfg.memory.extraction.default_tier,
            ExtractionTierConfig::Candidate
        ));
        assert!(cfg.memory.extraction.max_entries.is_none());
        assert!(cfg.memory.extraction.guidelines.is_none());
        assert!(cfg.memory.extraction.thinking.is_none());
        assert!(cfg.memory.extraction.chunking.is_none());
    }

    #[test]
    fn default_tier_candidate_when_absent() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"
"#,
        );
        assert!(matches!(
            cfg.memory.extraction.default_tier,
            ExtractionTierConfig::Candidate
        ));
    }

    #[test]
    fn default_tier_stable_is_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"
default_tier = "stable"
"#,
        );
        assert!(matches!(
            cfg.memory.extraction.default_tier,
            ExtractionTierConfig::Stable
        ));
    }

    #[test]
    fn default_tier_core_is_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"
default_tier = "core"
"#,
        );
        assert!(matches!(
            cfg.memory.extraction.default_tier,
            ExtractionTierConfig::Core
        ));
    }

    #[test]
    fn max_entries_is_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"
max_entries = 20
"#,
        );
        assert_eq!(cfg.memory.extraction.max_entries, Some(20));
    }

    #[test]
    fn guidelines_is_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"
guidelines = "Focus on technical facts only."
"#,
        );
        assert_eq!(
            cfg.memory.extraction.guidelines.as_deref(),
            Some("Focus on technical facts only.")
        );
    }

    #[test]
    fn thinking_disabled_is_parsed() {
        use crate::ModelThinkingConfig;
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.thinking]
mode = "disabled"
"#,
        );
        assert!(matches!(
            cfg.memory.extraction.thinking,
            Some(ModelThinkingConfig::Disabled)
        ));
    }

    #[test]
    fn thinking_effort_is_parsed() {
        use crate::{ModelThinkingConfig, ModelThinkingEffortLevel};
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.thinking]
mode  = "effort"
level = "low"
"#,
        );
        assert!(matches!(
            cfg.memory.extraction.thinking,
            Some(ModelThinkingConfig::Effort {
                level: ModelThinkingEffortLevel::Low
            })
        ));
    }

    #[test]
    fn chunking_defaults_when_section_present_but_fields_absent() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.chunking]
"#,
        );
        let chunking = cfg.memory.extraction.chunking.as_ref().unwrap();
        assert_eq!(chunking.chunk_size, 2500);
        assert_eq!(chunking.overlap_size, 200);
    }

    #[test]
    fn chunking_explicit_values_are_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.chunking]
chunk_size   = 3000
overlap_size = 300
"#,
        );
        let chunking = cfg.memory.extraction.chunking.as_ref().unwrap();
        assert_eq!(chunking.chunk_size, 3000);
        assert_eq!(chunking.overlap_size, 300);
    }

    #[test]
    fn missing_extraction_block_returns_parse_error() {
        let f = write_temp_config(BASE);
        let err = crate::load_config(f.path()).unwrap_err();
        assert!(
            matches!(err, crate::ConfigError::Parse { .. }),
            "expected Parse error when [memory.extraction] is absent, got: {err:?}"
        );
    }

    #[test]
    fn missing_model_field_returns_parse_error() {
        let f = write_temp_config(&format!(
            "{BASE}\n[memory.extraction]\ndefault_tier = \"stable\"\n"
        ));
        let err = crate::load_config(f.path()).unwrap_err();
        assert!(
            matches!(err, crate::ConfigError::Parse { .. }),
            "expected Parse error when `model` is missing, got: {err:?}"
        );
    }
}

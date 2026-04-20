// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use serde::Deserialize;

use crate::ModelThinkingConfig;

mod extractor;
pub use extractor::{MemoryExtractorConfig, MemoryExtractorSearchResultsConfig};

/// Chunking settings for splitting input text before LLM extraction,
/// deserialized from `[memory.extraction.chunking]`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkingConfig {
    /// Maximum number of characters per chunk. The splitter finishes the
    /// current word before cutting, so the actual chunk may be slightly
    /// longer.
    #[serde(default = "default_chunk_size")]
    chunk_size: usize,

    /// Characters of overlap between consecutive chunks.
    #[serde(default = "default_overlap_size")]
    overlap_size: usize,
}

impl ChunkingConfig {
    /// Returns the chunk size.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Returns the overlap size.
    pub fn overlap_size(&self) -> usize {
        self.overlap_size
    }
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
    model: String,

    /// Optional hard cap on the number of entries extracted per run.
    /// Applied both as a prompt hint and as a post-processing limit.
    max_entries: Option<usize>,

    /// Minimum LLM-assigned confidence score required to keep an extracted
    /// entry. Entries below this threshold are discarded before storing.
    /// In [0.0, 1.0]. When absent, all entries are kept.
    min_confidence: Option<f64>,

    /// Optional additional guidelines appended to the extraction prompt to
    /// guide or constrain what the model should extract.
    guidelines: Option<String>,

    /// Optional thinking mode override for the extraction model. When absent
    /// the model's default thinking behaviour is used. Extraction produces
    /// structured JSON, so `disabled` is usually the right choice.
    thinking: Option<ModelThinkingConfig>,

    /// Optional text chunking settings. When absent, the full input is sent
    /// to the model as a single call.
    chunking: Option<ChunkingConfig>,

    /// Optional pipeline-stage settings. When absent, the legacy single-stage
    /// extraction path is used.
    extractor: MemoryExtractorConfig,
}

impl MemoryExtractionConfig {
    /// Constructs a new config with all fields specified.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: impl Into<String>,
        max_entries: Option<usize>,
        min_confidence: Option<f64>,
        guidelines: Option<String>,
        thinking: Option<ModelThinkingConfig>,
        chunking: Option<ChunkingConfig>,
        extractor: MemoryExtractorConfig,
    ) -> Self {
        Self {
            model: model.into(),
            max_entries,
            min_confidence,
            guidelines,
            thinking,
            chunking,
            extractor,
        }
    }

    /// Returns the extraction model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the maximum number of entries to extract.
    pub fn max_entries(&self) -> Option<usize> {
        self.max_entries
    }

    /// Returns the minimum confidence threshold.
    pub fn min_confidence(&self) -> Option<f64> {
        self.min_confidence
    }

    /// Returns the optional extraction guidelines.
    pub fn guidelines(&self) -> Option<&str> {
        self.guidelines.as_deref()
    }

    /// Returns the optional thinking mode configuration.
    pub fn thinking(&self) -> Option<&ModelThinkingConfig> {
        self.thinking.as_ref()
    }

    /// Returns the optional chunking configuration.
    pub fn chunking(&self) -> Option<&ChunkingConfig> {
        self.chunking.as_ref()
    }

    /// Returns the extractor configuration.
    pub fn extractor(&self) -> &MemoryExtractorConfig {
        &self.extractor
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{AppConfig, load_config};

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

    const EXTRACTOR_SECTION: &str = r#"
[memory.extraction.extractor]
classification_model = "x"
"#;

    fn config_with_extraction(extraction_toml: &str) -> AppConfig {
        let raw = format!("{BASE}\n{extraction_toml}\n{EXTRACTOR_SECTION}");
        let f = write_temp_config(&raw);
        load_config(f.path()).unwrap()
    }

    #[test]
    fn test_minimal_extraction_block_is_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"
"#,
        );
        assert_eq!(cfg.memory().extraction().model(), "default");
        assert!(cfg.memory().extraction().max_entries().is_none());
        assert!(cfg.memory().extraction().min_confidence().is_none());
        assert!(cfg.memory().extraction().guidelines().is_none());
        assert!(cfg.memory().extraction().thinking().is_none());
        assert!(cfg.memory().extraction().chunking().is_none());
    }

    #[test]
    fn test_min_confidence_is_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"
min_confidence = 0.7
"#,
        );
        assert_eq!(cfg.memory().extraction().min_confidence(), Some(0.7));
    }

    #[test]
    fn test_max_entries_is_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"
max_entries = 20
"#,
        );
        assert_eq!(cfg.memory().extraction().max_entries(), Some(20));
    }

    #[test]
    fn test_guidelines_is_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"
guidelines = "Focus on technical facts only."
"#,
        );
        assert_eq!(
            cfg.memory().extraction().guidelines(),
            Some("Focus on technical facts only.")
        );
    }

    #[test]
    fn test_thinking_disabled_is_parsed() {
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
            cfg.memory().extraction().thinking(),
            Some(ModelThinkingConfig::Disabled)
        ));
    }

    #[test]
    fn test_thinking_effort_is_parsed() {
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
            cfg.memory().extraction().thinking(),
            Some(ModelThinkingConfig::Effort {
                level: ModelThinkingEffortLevel::Low
            })
        ));
    }

    #[test]
    fn test_chunking_defaults_when_section_present_but_fields_absent() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.chunking]
"#,
        );
        let chunking = cfg.memory().extraction().chunking().unwrap();
        assert_eq!(chunking.chunk_size(), 2500);
        assert_eq!(chunking.overlap_size(), 200);
    }

    #[test]
    fn test_chunking_explicit_values_are_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.chunking]
chunk_size   = 3000
overlap_size = 300
"#,
        );
        let chunking = cfg.memory().extraction().chunking().unwrap();
        assert_eq!(chunking.chunk_size(), 3000);
        assert_eq!(chunking.overlap_size(), 300);
    }

    #[test]
    fn test_missing_extraction_block_returns_parse_error() {
        let f = write_temp_config(BASE);
        let err = crate::load_config(f.path()).unwrap_err();
        assert!(
            matches!(err, crate::ConfigError::Parse { .. }),
            "expected Parse error when [memory.extraction] is absent, got: {err:?}"
        );
    }

    #[test]
    fn test_missing_extractor_section_returns_parse_error() {
        let f = write_temp_config(&format!(
            "{BASE}\n[memory.extraction]\nmodel = \"default\"\n"
        ));
        let err = crate::load_config(f.path()).unwrap_err();
        assert!(
            matches!(err, crate::ConfigError::Parse { .. }),
            "expected Parse error when [memory.extraction.extractor] is absent, got: {err:?}"
        );
    }

    #[test]
    fn test_missing_model_field_returns_parse_error() {
        let f = write_temp_config(&format!(
            "{BASE}\n[memory.extraction]\nmin_confidence = 0.7\n{EXTRACTOR_SECTION}"
        ));
        let err = crate::load_config(f.path()).unwrap_err();
        assert!(
            matches!(err, crate::ConfigError::Parse { .. }),
            "expected Parse error when `model` is missing, got: {err:?}"
        );
    }
}

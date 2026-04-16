// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use serde::Deserialize;

/// Configuration for the multi-stage memory extraction pipeline, deserialized from `[memory.extraction.pipeline]`.
#[derive(Debug, Clone, Deserialize)]
pub struct PipelineExtractionSearchResultsConfig {
    /// Maximum results for the semantic search.
    pub max_results: usize,
    /// Minimum score for semantic search results.
    pub min_score: f64,
}

/// Pipeline-stage configuration, deserialized from `[memory.extraction.pipeline]`.
///
/// When this section is absent, the legacy single-stage extraction path is used.
#[derive(Debug, Clone, Deserialize)]
pub struct PipelineExtractionConfig {
    /// Model name (key in `[models.text]`) for the hit-classification step.
    /// Intended to be a small, fast model (e.g. `"qwen2.5:0.5b"`).
    pub classification_model: String,

    /// Configuration for the direct (supporting evidence) semantic search stage.
    #[serde(default = "default_direct_search_config")]
    pub direct_search: PipelineExtractionSearchResultsConfig,

    /// Configuration for the inverted (contradiction evidence) semantic search stage.
    #[serde(default = "default_inverted_search_config")]
    pub inverted_search: PipelineExtractionSearchResultsConfig,

    /// Alpha increment for Duplicate classification. Default: 3.0
    #[serde(default = "default_duplicate_alpha_weight")]
    pub duplicate_alpha_weight: f64,

    /// Alpha increment for Complementary classification. Default: 1.0
    #[serde(default = "default_complementary_alpha_weight")]
    pub complementary_alpha_weight: f64,

    /// Beta increment for Contradiction classification. Default: 3.0
    #[serde(default = "default_contradiction_beta_weight")]
    pub contradiction_beta_weight: f64,

    /// Per-day exponential decay rate for alpha. Default: 0.99
    #[serde(default = "default_decay_rate")]
    pub decay_rate: f64,
}

fn default_direct_search_config() -> PipelineExtractionSearchResultsConfig {
    PipelineExtractionSearchResultsConfig {
        max_results: 5,
        min_score: 0.70,
    }
}

fn default_inverted_search_config() -> PipelineExtractionSearchResultsConfig {
    PipelineExtractionSearchResultsConfig {
        max_results: 3,
        min_score: 0.60,
    }
}
fn default_duplicate_alpha_weight() -> f64 {
    3.0
}
fn default_complementary_alpha_weight() -> f64 {
    1.0
}
fn default_contradiction_beta_weight() -> f64 {
    3.0
}
fn default_decay_rate() -> f64 {
    0.99
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{AppConfig, load_config};

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
        use std::io::Write as _;
        let raw = format!("{BASE}\n{extraction_toml}");
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(raw.as_bytes()).unwrap();
        crate::load_config(f.path()).unwrap()
    }

    #[test]
    fn pipeline_absent_gives_none() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"
"#,
        );
        assert!(cfg.memory.extraction.pipeline.is_none());
    }

    #[test]
    fn pipeline_minimal_is_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.pipeline]
classification_model = "qwen2.5:0.5b"
"#,
        );
        let pipeline = cfg.memory.extraction.pipeline.as_ref().unwrap();
        assert_eq!(pipeline.classification_model, "qwen2.5:0.5b");
        assert_eq!(pipeline.direct_search.max_results, 5);
        assert_eq!(pipeline.direct_search.min_score, 0.70);
        assert_eq!(pipeline.inverted_search.max_results, 3);
        assert_eq!(pipeline.inverted_search.min_score, 0.60);
        assert_eq!(pipeline.duplicate_alpha_weight, 3.0);
        assert_eq!(pipeline.complementary_alpha_weight, 1.0);
        assert_eq!(pipeline.contradiction_beta_weight, 3.0);
        assert_eq!(pipeline.decay_rate, 0.99);
    }

    #[test]
    fn pipeline_explicit_values_are_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.pipeline]
classification_model       = "qwen2.5:1.5b"
duplicate_alpha_weight     = 4.0
complementary_alpha_weight = 2.0
contradiction_beta_weight  = 5.0
decay_rate                 = 0.95

[memory.extraction.pipeline.direct_search]
max_results = 10
min_score   = 0.80

[memory.extraction.pipeline.inverted_search]
max_results = 6
min_score   = 0.55
"#,
        );
        let pipeline = cfg.memory.extraction.pipeline.as_ref().unwrap();
        assert_eq!(pipeline.classification_model, "qwen2.5:1.5b");
        assert_eq!(pipeline.direct_search.max_results, 10);
        assert_eq!(pipeline.direct_search.min_score, 0.80);
        assert_eq!(pipeline.inverted_search.max_results, 6);
        assert_eq!(pipeline.inverted_search.min_score, 0.55);
        assert_eq!(pipeline.duplicate_alpha_weight, 4.0);
        assert_eq!(pipeline.complementary_alpha_weight, 2.0);
        assert_eq!(pipeline.contradiction_beta_weight, 5.0);
        assert_eq!(pipeline.decay_rate, 0.95);
    }

    #[test]
    fn pipeline_defaults() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.pipeline]
classification_model = "x"
"#,
        );
        let pipeline = cfg.memory.extraction.pipeline.as_ref().unwrap();
        assert_eq!(pipeline.classification_model, "x");
        assert_eq!(pipeline.direct_search.max_results, 5);
        assert_eq!(pipeline.direct_search.min_score, 0.70);
        assert_eq!(pipeline.inverted_search.max_results, 3);
        assert_eq!(pipeline.inverted_search.min_score, 0.60);
        assert_eq!(pipeline.duplicate_alpha_weight, 3.0);
        assert_eq!(pipeline.complementary_alpha_weight, 1.0);
        assert_eq!(pipeline.contradiction_beta_weight, 3.0);
        assert_eq!(pipeline.decay_rate, 0.99);

        // Smoke-check the file round-trip produces a valid config overall.
        let f = {
            use std::io::Write as _;
            let raw = format!(
                "{BASE}\n[memory.extraction]\nmodel = \"default\"\n[memory.extraction.pipeline]\nclassification_model = \"x\"\n"
            );
            let mut tmp = tempfile::NamedTempFile::new().unwrap();
            tmp.write_all(raw.as_bytes()).unwrap();
            tmp
        };
        let cfg2 = load_config(f.path()).unwrap();
        assert_eq!(
            cfg2.memory.extraction.pipeline.as_ref().unwrap().decay_rate,
            0.99
        );
    }
}

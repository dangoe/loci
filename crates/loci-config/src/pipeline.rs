// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use serde::Deserialize;

/// Pipeline-stage configuration, deserialized from `[memory.extraction.pipeline]`.
///
/// When this section is absent, the legacy single-stage extraction path is used.
#[derive(Debug, Clone, Deserialize)]
pub struct PipelineExtractionConfig {
    /// Model name (key in `[models.text]`) for the hit-classification step.
    /// Intended to be a small, fast model (e.g. `"qwen2.5:0.5b"`).
    pub classification_model: String,

    /// Maximum results for the direct semantic search. Default: 5
    #[serde(default = "default_direct_max")]
    pub direct_search_max: usize,

    /// Minimum score for direct search results. Default: 0.70
    #[serde(default = "default_direct_min_score")]
    pub direct_search_min_score: f64,

    /// Maximum results for the inverted (contradiction) semantic search. Default: 3
    #[serde(default = "default_inverted_max")]
    pub inverted_search_max: usize,

    /// Minimum score for inverted search results. Default: 0.60
    #[serde(default = "default_inverted_min_score")]
    pub inverted_search_min_score: f64,

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

fn default_direct_max() -> usize {
    5
}
fn default_direct_min_score() -> f64 {
    0.70
}
fn default_inverted_max() -> usize {
    3
}
fn default_inverted_min_score() -> f64 {
    0.60
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
        assert_eq!(pipeline.direct_search_max, 5);
        assert_eq!(pipeline.direct_search_min_score, 0.70);
        assert_eq!(pipeline.inverted_search_max, 3);
        assert_eq!(pipeline.inverted_search_min_score, 0.60);
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
direct_search_max          = 10
direct_search_min_score    = 0.80
inverted_search_max        = 6
inverted_search_min_score  = 0.55
duplicate_alpha_weight     = 4.0
complementary_alpha_weight = 2.0
contradiction_beta_weight  = 5.0
decay_rate                 = 0.95
"#,
        );
        let pipeline = cfg.memory.extraction.pipeline.as_ref().unwrap();
        assert_eq!(pipeline.classification_model, "qwen2.5:1.5b");
        assert_eq!(pipeline.direct_search_max, 10);
        assert_eq!(pipeline.direct_search_min_score, 0.80);
        assert_eq!(pipeline.inverted_search_max, 6);
        assert_eq!(pipeline.inverted_search_min_score, 0.55);
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
        assert_eq!(pipeline.direct_search_max, 5);
        assert_eq!(pipeline.direct_search_min_score, 0.70);
        assert_eq!(pipeline.inverted_search_max, 3);
        assert_eq!(pipeline.inverted_search_min_score, 0.60);
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

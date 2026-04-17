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

    /// Seed weight `W` used to initialise Bayesian counters from LLM confidence.
    /// `α = confidence × W`, `β = (1 − confidence) × W`. Default: 10.0
    #[serde(default = "default_bayesian_seed_weight")]
    pub bayesian_seed_weight: f64,

    /// Maximum increment applied to a counter per evidence event. Default: 5.0
    #[serde(default = "default_max_counter_increment")]
    pub max_counter_increment: f64,

    /// Upper bound for each Bayesian counter (α and β). Default: 100.0
    #[serde(default = "default_max_counter")]
    pub max_counter: f64,

    /// Score at or below which an entry is automatically discarded. Default: 0.1
    #[serde(default = "default_auto_discard_threshold")]
    pub auto_discard_threshold: f64,

    /// Score at or above which an entry is automatically promoted to Fact. Default: 0.9
    #[serde(default = "default_auto_promotion_threshold")]
    pub auto_promotion_threshold: f64,

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
fn default_bayesian_seed_weight() -> f64 {
    10.0
}
fn default_max_counter_increment() -> f64 {
    5.0
}
fn default_max_counter() -> f64 {
    100.0
}
fn default_auto_discard_threshold() -> f64 {
    0.1
}
fn default_auto_promotion_threshold() -> f64 {
    0.9
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
        assert_eq!(pipeline.bayesian_seed_weight, 10.0);
        assert_eq!(pipeline.max_counter_increment, 5.0);
        assert_eq!(pipeline.max_counter, 100.0);
        assert_eq!(pipeline.auto_discard_threshold, 0.1);
        assert_eq!(pipeline.auto_promotion_threshold, 0.9);
        assert_eq!(pipeline.decay_rate, 0.99);
    }

    #[test]
    fn pipeline_explicit_values_are_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.pipeline]
classification_model      = "qwen2.5:1.5b"
bayesian_seed_weight      = 20.0
max_counter_increment     = 3.0
max_counter               = 50.0
auto_discard_threshold    = 0.05
auto_promotion_threshold  = 0.95
decay_rate                = 0.95

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
        assert_eq!(pipeline.bayesian_seed_weight, 20.0);
        assert_eq!(pipeline.max_counter_increment, 3.0);
        assert_eq!(pipeline.max_counter, 50.0);
        assert_eq!(pipeline.auto_discard_threshold, 0.05);
        assert_eq!(pipeline.auto_promotion_threshold, 0.95);
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
        assert_eq!(pipeline.bayesian_seed_weight, 10.0);
        assert_eq!(pipeline.max_counter_increment, 5.0);
        assert_eq!(pipeline.max_counter, 100.0);
        assert_eq!(pipeline.auto_discard_threshold, 0.1);
        assert_eq!(pipeline.auto_promotion_threshold, 0.9);
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

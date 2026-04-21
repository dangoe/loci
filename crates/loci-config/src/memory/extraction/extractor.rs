// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-config.

use serde::Deserialize;

/// Selects the strategy used to merge a new candidate with existing matching
/// memory entries, deserialized from
/// `[memory.extraction.extractor.merge_strategy]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MergeStrategyConfig {
    /// Pick the entry with the highest effective score — no LLM call.
    BestScore,
    /// Synthesise the candidate and all matching entries into a single
    /// statement via an LLM call.
    Llm {
        /// Text model (key in `[models.text]`) used for the merge call.
        model: String,
    },
}

fn default_merge_strategy() -> MergeStrategyConfig {
    MergeStrategyConfig::BestScore
}

/// Configuration for the memory extractor search stages,
/// deserialized from `[memory.extraction.extractor]`.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryExtractorSearchResultsConfig {
    /// Maximum results for the semantic search.
    max_results: usize,
    /// Minimum score for semantic search results.
    min_score: f64,
}

impl MemoryExtractorSearchResultsConfig {
    /// Constructs a new config with explicit values.
    pub fn new(max_results: usize, min_score: f64) -> Self {
        Self {
            max_results,
            min_score,
        }
    }

    /// Returns the maximum number of results.
    pub fn max_results(&self) -> usize {
        self.max_results
    }

    /// Returns the minimum score threshold.
    pub fn min_score(&self) -> f64 {
        self.min_score
    }
}

/// Memory extractor configuration, deserialized from `[memory.extraction.extractor]`.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryExtractorConfig {
    /// Model name (key in `[models.text]`) for the hit-classification step.
    /// Intended to be a small, fast model (e.g. `"qwen2.5:0.5b"`).
    classification_model: String,

    /// Configuration for the direct (supporting evidence) semantic search stage.
    #[serde(default = "default_direct_search_config")]
    direct_search: MemoryExtractorSearchResultsConfig,

    /// Configuration for the inverted (contradiction evidence) semantic search stage.
    #[serde(default = "default_inverted_search_config")]
    inverted_search: MemoryExtractorSearchResultsConfig,

    /// Seed weight `W` used to initialise Bayesian counters from LLM confidence.
    /// `α = confidence × W`, `β = (1 − confidence) × W`. Default: 10.0
    #[serde(default = "default_bayesian_seed_weight")]
    bayesian_seed_weight: f64,

    /// Maximum increment applied to a counter per evidence event. Default: 5.0
    #[serde(default = "default_max_counter_increment")]
    max_counter_increment: f64,

    /// Upper bound for each Bayesian counter (α and β). Default: 100.0
    #[serde(default = "default_max_counter")]
    max_counter: f64,

    /// Score at or below which an entry is automatically discarded. Default: 0.1
    #[serde(default = "default_auto_discard_threshold")]
    auto_discard_threshold: f64,

    /// Strategy used to merge a candidate with existing matching entries.
    /// Defaults to `BestScore` (no LLM call).
    #[serde(default = "default_merge_strategy")]
    merge_strategy: MergeStrategyConfig,
}

impl MemoryExtractorConfig {
    /// Returns the classification model name.
    pub fn classification_model(&self) -> &str {
        &self.classification_model
    }

    /// Returns the direct search configuration.
    pub fn direct_search(&self) -> &MemoryExtractorSearchResultsConfig {
        &self.direct_search
    }

    /// Returns the inverted search configuration.
    pub fn inverted_search(&self) -> &MemoryExtractorSearchResultsConfig {
        &self.inverted_search
    }

    /// Returns the Bayesian seed weight.
    pub fn bayesian_seed_weight(&self) -> f64 {
        self.bayesian_seed_weight
    }

    /// Returns the maximum counter increment.
    pub fn max_counter_increment(&self) -> f64 {
        self.max_counter_increment
    }

    /// Returns the maximum counter value.
    pub fn max_counter(&self) -> f64 {
        self.max_counter
    }

    /// Returns the auto-discard threshold.
    pub fn auto_discard_threshold(&self) -> f64 {
        self.auto_discard_threshold
    }

    /// Returns the merge strategy configuration.
    pub fn merge_strategy(&self) -> &MergeStrategyConfig {
        &self.merge_strategy
    }
}

fn default_direct_search_config() -> MemoryExtractorSearchResultsConfig {
    MemoryExtractorSearchResultsConfig {
        max_results: 5,
        min_score: 0.70,
    }
}

fn default_inverted_search_config() -> MemoryExtractorSearchResultsConfig {
    MemoryExtractorSearchResultsConfig {
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{AppConfig, load_config};

    const BASE: &str = r#"
[resources.memory_stores.qdrant]
kind = "qdrant"
url  = "http://localhost:6334"
collection = "mem"

[generation.text]
model = "x"

[embedding]
model = "x"

[memory]
store = "qdrant"
"#;

    fn config_with_extraction(extraction_toml: &str) -> AppConfig {
        use std::io::Write as _;
        let raw = format!("{BASE}\n{extraction_toml}");
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(raw.as_bytes()).unwrap();
        crate::load_config(f.path()).unwrap()
    }

    #[test]
    fn test_extractor_section_required() {
        use std::io::Write as _;
        let raw = format!("{BASE}\n[memory.extraction]\nmodel = \"default\"\n");
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(raw.as_bytes()).unwrap();
        let err = crate::load_config(f.path()).unwrap_err();
        assert!(
            matches!(err, crate::ConfigError::Parse { .. }),
            "expected Parse error when [memory.extraction.extractor] is absent, got: {err:?}"
        );
    }

    #[test]
    fn test_extractor_minimal_is_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "qwen2.5:0.5b"
"#,
        );
        let extractor = cfg.memory().extraction().extractor();
        assert_eq!(extractor.classification_model(), "qwen2.5:0.5b");
        assert_eq!(extractor.direct_search().max_results(), 5);
        assert_eq!(extractor.direct_search().min_score(), 0.70);
        assert_eq!(extractor.inverted_search().max_results(), 3);
        assert_eq!(extractor.inverted_search().min_score(), 0.60);
        assert_eq!(extractor.bayesian_seed_weight(), 10.0);
        assert_eq!(extractor.max_counter_increment(), 5.0);
        assert_eq!(extractor.max_counter(), 100.0);
        assert_eq!(extractor.auto_discard_threshold(), 0.1);
        assert!(matches!(
            extractor.merge_strategy(),
            crate::MergeStrategyConfig::BestScore
        ));
    }

    #[test]
    fn test_extractor_explicit_values_are_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model      = "qwen2.5:1.5b"
bayesian_seed_weight      = 20.0
max_counter_increment     = 3.0
max_counter               = 50.0
auto_discard_threshold    = 0.05

[memory.extraction.extractor.direct_search]
max_results = 10
min_score   = 0.80

[memory.extraction.extractor.inverted_search]
max_results = 6
min_score   = 0.55
"#,
        );
        let extractor = cfg.memory().extraction().extractor();
        assert_eq!(extractor.classification_model(), "qwen2.5:1.5b");
        assert_eq!(extractor.direct_search().max_results(), 10);
        assert_eq!(extractor.direct_search().min_score(), 0.80);
        assert_eq!(extractor.inverted_search().max_results(), 6);
        assert_eq!(extractor.inverted_search().min_score(), 0.55);
        assert_eq!(extractor.bayesian_seed_weight(), 20.0);
        assert_eq!(extractor.max_counter_increment(), 3.0);
        assert_eq!(extractor.max_counter(), 50.0);
        assert_eq!(extractor.auto_discard_threshold(), 0.05);
    }

    #[test]
    fn test_extractor_defaults() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
"#,
        );
        let extractor = cfg.memory().extraction().extractor();
        assert_eq!(extractor.classification_model(), "x");
        assert_eq!(extractor.direct_search().max_results(), 5);
        assert_eq!(extractor.direct_search().min_score(), 0.70);
        assert_eq!(extractor.inverted_search().max_results(), 3);
        assert_eq!(extractor.inverted_search().min_score(), 0.60);
        assert_eq!(extractor.bayesian_seed_weight(), 10.0);
        assert_eq!(extractor.max_counter_increment(), 5.0);
        assert_eq!(extractor.max_counter(), 100.0);
        assert_eq!(extractor.auto_discard_threshold(), 0.1);
        assert!(matches!(
            extractor.merge_strategy(),
            crate::MergeStrategyConfig::BestScore
        ));

        // Smoke-check the file round-trip produces a valid config overall.
        let f = {
            use std::io::Write as _;
            let raw = format!(
                "{BASE}\n[memory.extraction]\nmodel = \"default\"\n[memory.extraction.extractor]\nclassification_model = \"x\"\n"
            );
            let mut tmp = tempfile::NamedTempFile::new().unwrap();
            tmp.write_all(raw.as_bytes()).unwrap();
            tmp
        };
        let cfg2 = load_config(f.path()).unwrap();
        assert_eq!(
            cfg2.memory()
                .extraction()
                .extractor()
                .auto_discard_threshold(),
            0.1
        );
    }

    #[test]
    fn test_merge_strategy_best_score_is_default() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"
"#,
        );
        assert!(matches!(
            cfg.memory().extraction().extractor().merge_strategy(),
            crate::MergeStrategyConfig::BestScore
        ));
    }

    #[test]
    fn test_merge_strategy_llm_is_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"

[memory.extraction.extractor.merge_strategy]
kind  = "llm"
model = "llama3.2"
"#,
        );
        let strategy = cfg.memory().extraction().extractor().merge_strategy();
        match strategy {
            crate::MergeStrategyConfig::Llm { model } => {
                assert_eq!(model, "llama3.2");
            }
            other => panic!("expected Llm variant, got: {other:?}"),
        }
    }

    #[test]
    fn test_merge_strategy_best_score_explicit_is_parsed() {
        let cfg = config_with_extraction(
            r#"
[memory.extraction]
model = "default"

[memory.extraction.extractor]
classification_model = "x"

[memory.extraction.extractor.merge_strategy]
kind = "best_score"
"#,
        );
        assert!(matches!(
            cfg.memory().extraction().extractor().merge_strategy(),
            crate::MergeStrategyConfig::BestScore
        ));
    }
}

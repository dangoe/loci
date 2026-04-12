// Copyright (c) 2026 Daniel Götten
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is part of loci-cli.

use std::sync::Arc;
use std::{error::Error as StdError, io::Write};

use loci_config::{
    AppConfig, ConfigError, ModelThinkingConfig, ModelThinkingEffortLevel, ModelTuningConfig,
};
use loci_core::{
    contextualization::{
        ContextualizationMemoryMode as CoreContextualizationMemoryMode,
        Contextualizer as CoreContextualizer, ContextualizerConfig as CoreContextualizerConfig,
        ContextualizerSystemConfig as CoreContextualizerSystemConfig,
        ContextualizerSystemMode as CoreContextualizerSystemMode,
        ContextualizerTuningConfig as CoreContextualizerTuningConfig,
    },
    error::ContextualizerError,
    memory::Score as CoreScore,
    model_provider::text_generation::{
        TextGenerationModelProvider as CoreTextGenerationModelProvider,
        TextGenerationResponse as CoreTextGenerationResponse,
        ThinkingEffortLevel as CoreThinkingEffortLevel, ThinkingMode as CoreThinkingMode,
    },
    store::MemoryStore as CoreMemoryStore,
};

use crate::{
    commands::{
        GenerateCommand,
        generate::{GenerateDebugFlags, GenerateMemoryMode, GenerateSystemMode},
    },
    handlers::{CommandHandler, json::entry_to_json},
};

impl From<GenerateMemoryMode> for CoreContextualizationMemoryMode {
    fn from(val: GenerateMemoryMode) -> Self {
        match val {
            GenerateMemoryMode::Auto => CoreContextualizationMemoryMode::Auto,
            GenerateMemoryMode::Off => CoreContextualizationMemoryMode::Off,
        }
    }
}

pub struct GenerateCommandHandler<
    'a,
    S: CoreMemoryStore,
    T: CoreTextGenerationModelProvider + 'static,
> {
    store: Arc<S>,
    text_generation_model_provider: Arc<T>,
    config: &'a AppConfig,
}

impl<'a, S: CoreMemoryStore, T: CoreTextGenerationModelProvider + 'static>
    GenerateCommandHandler<'a, S, T>
{
    pub fn new(
        store: Arc<S>,
        text_generation_model_provider: Arc<T>,
        config: &'a AppConfig,
    ) -> Self {
        Self {
            store,
            text_generation_model_provider,
            config,
        }
    }
}

impl<'a, S: CoreMemoryStore, T: CoreTextGenerationModelProvider + 'static, W: Write + Send>
    CommandHandler<'a, GenerateCommand, W> for GenerateCommandHandler<'a, S, T>
{
    async fn handle(&self, command: GenerateCommand, out: &mut W) -> Result<(), Box<dyn StdError>> {
        let GenerateCommand::Execute(command) = command;
        let model = {
            let model_key = &self.config.routing.text.default;
            self.config
                .models
                .text
                .get(model_key)
                .ok_or_else(|| ConfigError::MissingKey {
                    section: "models.text".into(),
                    key: model_key.clone(),
                })?
                .clone()
        };
        let min_score =
            CoreScore::new(command.min_score).map_err(|e| format!("invalid min_score: {e}"))?;

        let ctx_config = CoreContextualizerConfig {
            system: command.system.map(|system| CoreContextualizerSystemConfig {
                mode: match command.system_mode {
                    GenerateSystemMode::Append => CoreContextualizerSystemMode::Append,
                    GenerateSystemMode::Replace => CoreContextualizerSystemMode::Replace,
                },
                system,
            }),
            max_memory_entries: command.max_memory_entries,
            min_score,
            memory_mode: command.memory_mode.into(),
            filters: command.filters.into_iter().collect(),
            text_generation_model: model.model,
            tuning: model.tuning.as_ref().map(model_tuning_to_contextualizer),
        };

        let contextualizer = CoreContextualizer::new(
            Arc::clone(&self.store),
            Arc::clone(&self.text_generation_model_provider),
            ctx_config,
        );

        if command.debug_flags.contains(&GenerateDebugFlags::Memory) {
            let (debug_info, stream) = contextualizer
                .contextualize_with_debug(&command.prompt)
                .await?;

            eprintln!("Debug info:\n");
            eprintln!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                      "retrieved_memory": debug_info.memory_entries.iter().map(entry_to_json).collect::<Vec<_>>(),
                }))?
            );

            writeln!(out, "\nResponse:\n")?;

            stream_text_generation(stream, out).await?;
        } else {
            let stream = contextualizer.contextualize(&command.prompt).await?;
            stream_text_generation(stream, out).await?;
        }
        Ok(())
    }
}

fn model_tuning_to_contextualizer(tuning: &ModelTuningConfig) -> CoreContextualizerTuningConfig {
    CoreContextualizerTuningConfig {
        temperature: tuning.temperature,
        max_tokens: tuning.max_tokens,
        top_p: tuning.top_p,
        repeat_penalty: tuning.repeat_penalty,
        repeat_last_n: tuning.repeat_last_n,
        thinking: tuning.thinking.as_ref().map(model_thinking_to_core),
        stop: tuning.stop.clone(),
        keep_alive: tuning.keep_alive_secs.map(std::time::Duration::from_secs),
        extra_params: tuning.extra.clone(),
    }
}

fn model_thinking_to_core(thinking: &ModelThinkingConfig) -> CoreThinkingMode {
    match thinking {
        ModelThinkingConfig::Enabled => CoreThinkingMode::Enabled,
        ModelThinkingConfig::Disabled => CoreThinkingMode::Disabled,
        ModelThinkingConfig::Effort { level } => CoreThinkingMode::Effort {
            level: match level {
                ModelThinkingEffortLevel::Low => CoreThinkingEffortLevel::Low,
                ModelThinkingEffortLevel::Medium => CoreThinkingEffortLevel::Medium,
                ModelThinkingEffortLevel::High => CoreThinkingEffortLevel::High,
            },
        },
        ModelThinkingConfig::Budgeted { max_tokens } => CoreThinkingMode::Budgeted {
            max_tokens: *max_tokens,
        },
    }
}

/// Consumes a text-generation stream, printing each chunk to stdout.
///
/// A newline is printed after the final chunk (when `chunk.done` is `true`).
async fn stream_text_generation<W: std::io::Write>(
    mut stream: impl futures::Stream<Item = Result<CoreTextGenerationResponse, ContextualizerError>>
    + Unpin,
    out: &mut W,
) -> Result<(), Box<dyn std::error::Error>> {
    use futures::StreamExt as _;
    while let Some(result) = stream.next().await {
        let chunk = result.map_err(|e| e.to_string())?;
        write!(out, "{}", chunk.text)?;
        out.flush()?;
        if chunk.done {
            writeln!(out)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{
        commands::generate::{
            GenerateArgs, GenerateCommand, GenerateDebugFlags, GenerateMemoryMode,
            GenerateSystemMode,
        },
        fixture,
        handlers::{
            CommandHandler,
            generate::{
                GenerateCommandHandler, model_thinking_to_core, model_tuning_to_contextualizer,
                stream_text_generation,
            },
        },
    };
    use loci_core::testing::{MockStore, MockTextGenerationModelProvider};

    fn default_generate_args(prompt: &str) -> GenerateArgs {
        GenerateArgs {
            prompt: prompt.to_string(),
            system: None,
            system_mode: GenerateSystemMode::Append,
            max_memory_entries: 5,
            min_score: 0.5,
            memory_mode: GenerateMemoryMode::Auto,
            filters: vec![],
            debug_flags: vec![],
        }
    }

    #[test]
    fn test_gen_memory_mode_auto_converts_to_contextualizer_auto() {
        use loci_core::contextualization::ContextualizationMemoryMode;
        let mode: ContextualizationMemoryMode = GenerateMemoryMode::Auto.into();
        assert_eq!(mode, ContextualizationMemoryMode::Auto);
    }

    #[test]
    fn test_gen_memory_mode_off_converts_to_contextualizer_off() {
        use loci_core::contextualization::ContextualizationMemoryMode;
        let mode: ContextualizationMemoryMode = GenerateMemoryMode::Off.into();
        assert_eq!(mode, ContextualizationMemoryMode::Off);
    }

    #[test]
    fn test_model_tuning_to_contextualizer_maps_all_fields() {
        use loci_config::{ModelThinkingConfig, ModelTuningConfig};
        use loci_core::model_provider::text_generation::ThinkingMode;
        use std::time::Duration;

        let tuning = ModelTuningConfig {
            temperature: Some(0.7),
            max_tokens: Some(512),
            top_p: Some(0.9),
            repeat_penalty: Some(1.1),
            repeat_last_n: Some(64),
            stop: Some(vec!["<END>".to_string()]),
            keep_alive_secs: Some(300),
            thinking: Some(ModelThinkingConfig::Enabled),
            extra: HashMap::new(),
        };

        let ctx = model_tuning_to_contextualizer(&tuning);

        assert_eq!(ctx.temperature, Some(0.7));
        assert_eq!(ctx.max_tokens, Some(512));
        assert_eq!(ctx.top_p, Some(0.9));
        assert_eq!(ctx.repeat_penalty, Some(1.1));
        assert_eq!(ctx.repeat_last_n, Some(64));
        assert_eq!(ctx.stop.as_deref(), Some(["<END>".to_string()].as_slice()));
        assert_eq!(ctx.keep_alive, Some(Duration::from_secs(300)));
        assert!(matches!(ctx.thinking, Some(ThinkingMode::Enabled)));
    }

    #[test]
    fn test_model_tuning_to_contextualizer_maps_none_fields() {
        use loci_config::ModelTuningConfig;

        let tuning = ModelTuningConfig::default();
        let ctx = model_tuning_to_contextualizer(&tuning);

        assert_eq!(ctx.temperature, None);
        assert_eq!(ctx.max_tokens, None);
        assert_eq!(ctx.top_p, None);
        assert_eq!(ctx.repeat_penalty, None);
        assert_eq!(ctx.repeat_last_n, None);
        assert_eq!(ctx.stop, None);
        assert_eq!(ctx.keep_alive, None);
        assert!(ctx.thinking.is_none());
    }

    #[rstest]
    #[case(loci_config::ModelThinkingConfig::Enabled, "enabled")]
    #[case(loci_config::ModelThinkingConfig::Disabled, "disabled")]
    #[case(
        loci_config::ModelThinkingConfig::Effort { level: loci_config::ModelThinkingEffortLevel::Low },
        "effort_low"
    )]
    #[case(
        loci_config::ModelThinkingConfig::Effort { level: loci_config::ModelThinkingEffortLevel::Medium },
        "effort_medium"
    )]
    #[case(
        loci_config::ModelThinkingConfig::Effort { level: loci_config::ModelThinkingEffortLevel::High },
        "effort_high"
    )]
    #[case(
        loci_config::ModelThinkingConfig::Budgeted { max_tokens: 256 },
        "budgeted"
    )]
    fn test_model_thinking_to_core_all_variants(
        #[case] input: loci_config::ModelThinkingConfig,
        #[case] label: &str,
    ) {
        use loci_core::model_provider::text_generation::{ThinkingEffortLevel, ThinkingMode};

        let result = model_thinking_to_core(&input);
        match (label, &result) {
            ("enabled", ThinkingMode::Enabled) => {}
            ("disabled", ThinkingMode::Disabled) => {}
            (
                "effort_low",
                ThinkingMode::Effort {
                    level: ThinkingEffortLevel::Low,
                },
            ) => {}
            (
                "effort_medium",
                ThinkingMode::Effort {
                    level: ThinkingEffortLevel::Medium,
                },
            ) => {}
            (
                "effort_high",
                ThinkingMode::Effort {
                    level: ThinkingEffortLevel::High,
                },
            ) => {}
            ("budgeted", ThinkingMode::Budgeted { max_tokens: 256 }) => {}
            _ => panic!("unexpected mapping for label '{label}': {result:?}"),
        }
    }

    #[tokio::test]
    async fn test_stream_text_generation_writes_all_chunks() {
        use futures::stream;
        use loci_core::error::ContextualizerError;
        use loci_core::model_provider::text_generation::TextGenerationResponse;

        let chunks: Vec<Result<TextGenerationResponse, ContextualizerError>> = vec![
            Ok(TextGenerationResponse {
                text: "hello ".to_string(),
                model: "m".to_string(),
                usage: None,
                done: false,
            }),
            Ok(TextGenerationResponse {
                text: "world".to_string(),
                model: "m".to_string(),
                usage: None,
                done: true,
            }),
        ];

        let mut out = Vec::new();
        stream_text_generation(stream::iter(chunks), &mut out)
            .await
            .unwrap();

        let output = String::from_utf8(out).unwrap();
        // Text chunks are concatenated; a newline is appended after the done chunk.
        assert!(output.starts_with("hello world"), "got: {output:?}");
        assert!(
            output.ends_with('\n'),
            "final newline missing, got: {output:?}"
        );
    }

    #[tokio::test]
    async fn test_stream_text_generation_propagates_stream_error() {
        use futures::stream;
        use loci_core::error::{ContextualizerError, MemoryStoreError};
        use loci_core::model_provider::text_generation::TextGenerationResponse;

        let chunks: Vec<Result<TextGenerationResponse, ContextualizerError>> = vec![
            Ok(TextGenerationResponse {
                text: "partial".to_string(),
                model: "m".to_string(),
                usage: None,
                done: false,
            }),
            Err(ContextualizerError::MemoryStore(
                MemoryStoreError::Connection("boom".to_string()),
            )),
        ];

        let mut out = Vec::new();
        let result = stream_text_generation(stream::iter(chunks), &mut out).await;

        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("boom"),
            "error message should include the underlying cause"
        );
    }

    #[tokio::test]
    async fn test_stream_text_generation_empty_stream_writes_nothing() {
        use futures::stream;
        use loci_core::error::ContextualizerError;
        use loci_core::model_provider::text_generation::TextGenerationResponse;

        let chunks: Vec<Result<TextGenerationResponse, ContextualizerError>> = vec![];
        let mut out = Vec::new();
        stream_text_generation(stream::iter(chunks), &mut out)
            .await
            .unwrap();

        assert!(
            out.is_empty(),
            "no bytes should be written for an empty stream"
        );
    }

    #[tokio::test]
    async fn test_stream_text_generation_single_done_chunk_appends_newline() {
        use futures::stream;
        use loci_core::error::ContextualizerError;
        use loci_core::model_provider::text_generation::TextGenerationResponse;

        let chunks: Vec<Result<TextGenerationResponse, ContextualizerError>> =
            vec![Ok(TextGenerationResponse {
                text: "hi".to_string(),
                model: "m".to_string(),
                usage: None,
                done: true,
            })];
        let mut out = Vec::new();
        stream_text_generation(stream::iter(chunks), &mut out)
            .await
            .unwrap();

        let output = String::from_utf8(out).unwrap();
        assert_eq!(output, "hi\n");
    }

    #[tokio::test]
    async fn test_stream_text_generation_no_done_chunk_omits_trailing_newline() {
        use futures::stream;
        use loci_core::error::ContextualizerError;
        use loci_core::model_provider::text_generation::TextGenerationResponse;

        let chunks: Vec<Result<TextGenerationResponse, ContextualizerError>> = vec![
            Ok(TextGenerationResponse {
                text: "a".to_string(),
                model: "m".to_string(),
                usage: None,
                done: false,
            }),
            Ok(TextGenerationResponse {
                text: "b".to_string(),
                model: "m".to_string(),
                usage: None,
                done: false,
            }),
        ];
        let mut out = Vec::new();
        stream_text_generation(stream::iter(chunks), &mut out)
            .await
            .unwrap();

        let output = String::from_utf8(out).unwrap();
        assert_eq!(output, "ab");
    }

    #[test]
    fn test_model_tuning_to_contextualizer_maps_extra_params() {
        use loci_config::ModelTuningConfig;
        use serde_json::json;

        let mut extra = HashMap::new();
        extra.insert("top_k".to_string(), json!(40));
        extra.insert("seed".to_string(), json!(42));

        let tuning = ModelTuningConfig {
            extra,
            ..ModelTuningConfig::default()
        };

        let ctx = model_tuning_to_contextualizer(&tuning);

        assert_eq!(ctx.extra_params.get("top_k").unwrap(), &json!(40));
        assert_eq!(ctx.extra_params.get("seed").unwrap(), &json!(42));
    }

    #[tokio::test]
    async fn test_generate_handle_streams_response() {
        let store = MockStore::new().with_query(vec![]);
        let provider = MockTextGenerationModelProvider::new().with_chunks(vec!["hello", " world"]);
        let config = fixture::minimal_ollama_config();
        let mut out = Vec::new();

        let handler = GenerateCommandHandler::new(Arc::new(store), Arc::new(provider), &config);
        handler
            .handle(
                GenerateCommand::Execute(default_generate_args("test prompt")),
                &mut out,
            )
            .await
            .unwrap();

        let output = String::from_utf8(out).unwrap();
        assert!(output.contains("hello"), "got: {output:?}");
        assert!(output.contains(" world"), "got: {output:?}");
    }

    #[tokio::test]
    async fn test_generate_handle_missing_model_key_returns_err() {
        let store = MockStore::new();
        let provider = MockTextGenerationModelProvider::new();
        let mut config = fixture::minimal_ollama_config();
        config.routing.text.default = "nonexistent".to_string();
        let mut out = Vec::new();

        let handler = GenerateCommandHandler::new(Arc::new(store), Arc::new(provider), &config);
        let result = handler
            .handle(
                GenerateCommand::Execute(default_generate_args("hi")),
                &mut out,
            )
            .await;

        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("nonexistent"),
            "error should mention the missing key, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_generate_handle_invalid_min_score_returns_err() {
        let store = MockStore::new();
        let provider = MockTextGenerationModelProvider::new();
        let config = fixture::minimal_ollama_config();
        let mut out = Vec::new();

        let mut args = default_generate_args("hi");
        args.min_score = 1.5; // outside [0.0, 1.0]

        let handler = GenerateCommandHandler::new(Arc::new(store), Arc::new(provider), &config);
        let result = handler
            .handle(GenerateCommand::Execute(args), &mut out)
            .await;

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid min_score"),
            "error should mention invalid min_score"
        );
    }

    #[tokio::test]
    async fn test_generate_handle_debug_memory_flag_writes_response_header() {
        let store = MockStore::new().with_query(vec![]);
        let provider = MockTextGenerationModelProvider::new().with_chunks(vec!["debug output"]);
        let config = fixture::minimal_ollama_config();
        let mut out = Vec::new();

        let mut args = default_generate_args("debug prompt");
        args.debug_flags = vec![GenerateDebugFlags::Memory];

        let handler = GenerateCommandHandler::new(Arc::new(store), Arc::new(provider), &config);
        handler
            .handle(GenerateCommand::Execute(args), &mut out)
            .await
            .unwrap();

        let output = String::from_utf8(out).unwrap();
        assert!(
            output.contains("Response:"),
            "debug path should write response header, got: {output:?}"
        );
        assert!(
            output.contains("debug output"),
            "debug path should stream model output, got: {output:?}"
        );
    }

    #[tokio::test]
    async fn test_generate_handle_memory_mode_off_succeeds_without_store_entries() {
        let store = MockStore::new(); // query_entries empty — would fail if queried and result expected
        let provider = MockTextGenerationModelProvider::new().with_chunks(vec!["ok"]);
        let config = fixture::minimal_ollama_config();
        let mut out = Vec::new();

        let mut args = default_generate_args("silent prompt");
        args.memory_mode = GenerateMemoryMode::Off;

        let handler = GenerateCommandHandler::new(Arc::new(store), Arc::new(provider), &config);
        handler
            .handle(GenerateCommand::Execute(args), &mut out)
            .await
            .unwrap();

        let output = String::from_utf8(out).unwrap();
        assert!(output.contains("ok"), "got: {output:?}");
    }

    #[tokio::test]
    async fn test_generate_handle_system_mode_append_succeeds() {
        let store = MockStore::new().with_query(vec![]);
        let provider = MockTextGenerationModelProvider::new();
        let config = fixture::minimal_ollama_config();
        let mut out = Vec::new();

        let mut args = default_generate_args("prompt");
        args.system = Some("be brief".to_string());
        args.system_mode = GenerateSystemMode::Append;

        let handler = GenerateCommandHandler::new(Arc::new(store), Arc::new(provider), &config);
        handler
            .handle(GenerateCommand::Execute(args), &mut out)
            .await
            .unwrap();

        assert!(!out.is_empty(), "should produce output");
    }

    #[tokio::test]
    async fn test_generate_handle_system_mode_replace_succeeds() {
        let store = MockStore::new().with_query(vec![]);
        let provider = MockTextGenerationModelProvider::new();
        let config = fixture::minimal_ollama_config();
        let mut out = Vec::new();

        let mut args = default_generate_args("prompt");
        args.system = Some("you are a pirate".to_string());
        args.system_mode = GenerateSystemMode::Replace;

        let handler = GenerateCommandHandler::new(Arc::new(store), Arc::new(provider), &config);
        handler
            .handle(GenerateCommand::Execute(args), &mut out)
            .await
            .unwrap();

        assert!(!out.is_empty(), "should produce output");
    }
}

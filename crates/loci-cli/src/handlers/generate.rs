use std::{collections::HashMap, error::Error as StdError, io::Write};

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

impl Into<CoreContextualizationMemoryMode> for GenerateMemoryMode {
    fn into(self) -> CoreContextualizationMemoryMode {
        match self {
            GenerateMemoryMode::Auto => CoreContextualizationMemoryMode::Auto,
            GenerateMemoryMode::Off => CoreContextualizationMemoryMode::Off,
        }
    }
}

pub struct GenerateCommandHandler<'a, S: CoreMemoryStore, T: CoreTextGenerationModelProvider> {
    store: &'a S,
    text_generation_model_provider: &'a T,
    config: &'a AppConfig,
}

impl<'a, S: CoreMemoryStore, T: CoreTextGenerationModelProvider> GenerateCommandHandler<'a, S, T> {
    pub fn new(
        store: &'a S,
        text_generation_model_provider: &'a T,
        config: &'a AppConfig,
    ) -> Self {
        Self {
            store,
            text_generation_model_provider,
            config,
        }
    }
}

impl<'a, S: CoreMemoryStore, T: CoreTextGenerationModelProvider, W: Write + Send>
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
            filters: HashMap::new(),
            text_generation_model: model.model,
            tuning: model.tuning.as_ref().map(model_tuning_to_contextualizer),
        };

        let contextualizer = CoreContextualizer::new(
            self.store,
            self.text_generation_model_provider,
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

    use rstest::rstest;

    use crate::{
        commands::generate::GenerateMemoryMode,
        handlers::generate::{model_tuning_to_contextualizer, stream_text_generation},
    };

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

        use crate::handlers::generate::model_thinking_to_core;

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
}

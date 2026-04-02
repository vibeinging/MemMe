//! Unified structured LLM generation with automatic retry on parse failure.
//!
//! Instead of each call site manually calling `llm.generate()` + parse + retry,
//! this module provides a single `generate_structured()` function that handles
//! the retry loop, temperature bumping, and error context feedback automatically.
//!
//! TODO: Unify embedding retry logic. `OpenAiEmbedder::embed_async()` has its own
//! 3-attempt exponential backoff for network errors. Consider extracting a shared
//! retry helper that both structured generation and embedding calls can use.

use crate::{GenerateOptions, LlmError, LlmProvider, Message, MessageRole, ResponseFormat};

/// Configuration for structured generation with retries.
#[derive(Debug, Clone)]
pub struct StructuredGenConfig {
    /// Maximum number of retry attempts on parse failure (default: 3).
    pub max_retries: u32,
    /// Base temperature for generation. `None` keeps it unset (for reasoning models).
    pub base_temperature: Option<f32>,
    /// Temperature increment per retry attempt (default: 0.1).
    pub temperature_increment: f32,
    /// Maximum tokens to generate.
    pub max_tokens: Option<usize>,
    /// Desired response format.
    pub response_format: Option<ResponseFormat>,
}

impl Default for StructuredGenConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_temperature: Some(0.1),
            temperature_increment: 0.1,
            max_tokens: None,
            response_format: Some(ResponseFormat::Json),
        }
    }
}

impl StructuredGenConfig {
    /// Build `GenerateOptions` for a given attempt number (0-based).
    fn options_for_attempt(&self, attempt: u32) -> GenerateOptions {
        let temperature = self
            .base_temperature
            .map(|base| base + self.temperature_increment * attempt as f32);
        GenerateOptions {
            temperature,
            max_tokens: self.max_tokens,
            response_format: self.response_format,
        }
    }
}

/// Generate a structured response from an LLM with automatic retry on parse failure.
///
/// On each parse failure, the assistant's raw output and a user message describing
/// the error are appended to the conversation, giving the model context about what
/// went wrong. Temperature is also bumped to encourage different output.
///
/// # Arguments
/// * `llm` — the LLM provider to call
/// * `messages` — initial conversation messages (will be cloned for retries)
/// * `config` — retry and generation configuration
/// * `parse` — closure that attempts to parse the raw LLM output into `T`
///
/// # Errors
/// * `LlmError` from `generate()` propagates immediately (network/provider errors)
/// * After `max_retries` exhausted, returns `LlmError::ParseError`
pub fn generate_structured<T>(
    llm: &dyn LlmProvider,
    messages: &[Message],
    config: &StructuredGenConfig,
    parse: impl Fn(&str) -> Result<T, String>,
) -> Result<T, LlmError> {
    let mut conversation: Vec<Message> = messages.to_vec();
    let max_attempts = config.max_retries + 1; // first attempt + retries
    let mut last_error = String::new();

    for attempt in 0..max_attempts {
        let options = config.options_for_attempt(attempt);

        // LlmError from generate() propagates immediately (network errors)
        let raw = llm.generate(&conversation, &options)?;

        match parse(&raw) {
            Ok(value) => return Ok(value),
            Err(err) => {
                last_error = err.clone();

                if attempt + 1 < max_attempts {
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = config.max_retries,
                        error = %err,
                        provider = llm.name(),
                        "Structured generation parse failed, retrying"
                    );

                    // Append the failed output so the model sees what it produced.
                    // Truncate to 500 chars to avoid context explosion with large outputs.
                    let truncated_raw = if raw.chars().count() > 500 {
                        let prefix: String = raw.chars().take(500).collect();
                        format!("{prefix}... [truncated, {} total chars]", raw.len())
                    } else {
                        raw
                    };
                    conversation.push(Message {
                        role: MessageRole::Assistant,
                        content: truncated_raw,
                    });

                    // Append error context and retry instruction
                    conversation.push(Message {
                        role: MessageRole::User,
                        content: format!(
                            "Your previous response could not be parsed. Error: {err}\n\
                             Please fix the output and try again. Return ONLY valid JSON."
                        ),
                    });
                }
            }
        }
    }

    Err(LlmError::ParseError(format!(
        "Failed to parse after {} attempts. Last error: {last_error}",
        max_attempts,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct MockLlm {
        responses: Mutex<Vec<String>>,
    }

    impl MockLlm {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    impl LlmProvider for MockLlm {
        fn generate(
            &self,
            _messages: &[Message],
            _options: &GenerateOptions,
        ) -> Result<String, LlmError> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                Err(LlmError::NotAvailable("no more mock responses".into()))
            } else {
                Ok(responses.remove(0))
            }
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    #[test]
    fn test_success_on_first_attempt() {
        let llm = MockLlm::new(vec![r#"{"facts": ["a"]}"#.into()]);
        let messages = vec![Message {
            role: MessageRole::User,
            content: "test".into(),
        }];
        let config = StructuredGenConfig::default();

        let result: Result<String, _> =
            generate_structured(&llm, &messages, &config, |raw| Ok(raw.to_string()));
        assert!(result.is_ok());
    }

    #[test]
    fn test_retry_on_parse_failure() {
        let llm = MockLlm::new(vec![
            "bad json".into(),
            "still bad".into(),
            r#"{"ok": true}"#.into(),
        ]);
        let messages = vec![Message {
            role: MessageRole::User,
            content: "test".into(),
        }];
        let config = StructuredGenConfig {
            max_retries: 3,
            ..Default::default()
        };

        let result = generate_structured(&llm, &messages, &config, |raw| {
            if raw.contains("ok") {
                Ok(raw.to_string())
            } else {
                Err("not valid".into())
            }
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_exhausts_retries() {
        let llm = MockLlm::new(vec!["bad".into(), "bad".into(), "bad".into(), "bad".into()]);
        let messages = vec![Message {
            role: MessageRole::User,
            content: "test".into(),
        }];
        let config = StructuredGenConfig {
            max_retries: 3,
            ..Default::default()
        };

        let result = generate_structured(&llm, &messages, &config, |_raw| {
            Err::<String, _>("parse error".into())
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            LlmError::ParseError(msg) => {
                assert!(msg.contains("4 attempts"));
            }
            other => panic!("Expected ParseError, got: {other:?}"),
        }
    }

    #[test]
    fn test_network_error_propagates_immediately() {
        let llm = MockLlm::new(vec![]); // will return NotAvailable
        let messages = vec![Message {
            role: MessageRole::User,
            content: "test".into(),
        }];
        let config = StructuredGenConfig::default();

        let result = generate_structured(&llm, &messages, &config, |raw| Ok(raw.to_string()));
        assert!(result.is_err());
        match result.unwrap_err() {
            LlmError::NotAvailable(_) => {}
            other => panic!("Expected NotAvailable, got: {other:?}"),
        }
    }

    #[test]
    fn test_none_temperature_stays_none() {
        let config = StructuredGenConfig {
            base_temperature: None,
            ..Default::default()
        };
        let opts = config.options_for_attempt(2);
        assert!(opts.temperature.is_none());
    }

    #[test]
    fn test_temperature_increments() {
        let config = StructuredGenConfig {
            base_temperature: Some(0.1),
            temperature_increment: 0.1,
            ..Default::default()
        };
        assert_eq!(config.options_for_attempt(0).temperature, Some(0.1));
        assert!((config.options_for_attempt(2).temperature.unwrap() - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_conversation_grows_on_retry() {
        // Verify the retry messages are appended by checking the mock sees them
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct SpyLlm {
            call_count: AtomicUsize,
            message_counts: Mutex<Vec<usize>>,
        }
        impl LlmProvider for SpyLlm {
            fn generate(
                &self,
                messages: &[Message],
                _options: &GenerateOptions,
            ) -> Result<String, LlmError> {
                self.message_counts.lock().unwrap().push(messages.len());
                let n = self.call_count.fetch_add(1, Ordering::Relaxed);
                if n < 2 {
                    Ok("bad".into())
                } else {
                    Ok("good".into())
                }
            }
            fn name(&self) -> &str {
                "spy"
            }
        }

        let llm = SpyLlm {
            call_count: AtomicUsize::new(0),
            message_counts: Mutex::new(vec![]),
        };
        let messages = vec![Message {
            role: MessageRole::User,
            content: "test".into(),
        }];
        let config = StructuredGenConfig {
            max_retries: 3,
            ..Default::default()
        };

        let _ = generate_structured(&llm, &messages, &config, |raw| {
            if raw == "good" {
                Ok(raw.to_string())
            } else {
                Err("bad".into())
            }
        });

        let counts = llm.message_counts.lock().unwrap();
        // 1st call: 1 message, 2nd call: 1 + 2 (assistant + user) = 3, 3rd: 3 + 2 = 5
        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 3);
        assert_eq!(counts[2], 5);
    }

    #[test]
    fn test_long_output_truncated_in_retry_context() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct SpyLlm {
            call_count: AtomicUsize,
            captured_messages: Mutex<Vec<Vec<Message>>>,
        }

        impl LlmProvider for SpyLlm {
            fn generate(
                &self,
                messages: &[Message],
                _options: &GenerateOptions,
            ) -> Result<String, LlmError> {
                self.captured_messages
                    .lock()
                    .unwrap()
                    .push(messages.to_vec());
                let n = self.call_count.fetch_add(1, Ordering::Relaxed);
                if n == 0 {
                    // Return a very long string on first attempt
                    Ok("x".repeat(1000))
                } else {
                    Ok("good".into())
                }
            }
            fn name(&self) -> &str {
                "spy"
            }
        }

        let llm = SpyLlm {
            call_count: AtomicUsize::new(0),
            captured_messages: Mutex::new(vec![]),
        };
        let messages = vec![Message {
            role: MessageRole::User,
            content: "test".into(),
        }];
        let config = StructuredGenConfig {
            max_retries: 2,
            ..Default::default()
        };

        let _ = generate_structured(&llm, &messages, &config, |raw| {
            if raw == "good" {
                Ok(raw.to_string())
            } else {
                Err("bad".into())
            }
        });

        let captured = llm.captured_messages.lock().unwrap();
        // Second call should have the truncated assistant message
        assert!(captured.len() >= 2);
        let assistant_msg = &captured[1][1]; // index 1 = assistant message appended after failure
        assert_eq!(assistant_msg.role, MessageRole::Assistant);
        assert!(assistant_msg.content.len() < 1000);
        assert!(assistant_msg.content.contains("[truncated"));
    }
}

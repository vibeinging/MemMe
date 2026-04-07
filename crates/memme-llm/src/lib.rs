//! # memme-llm
//!
//! LLM abstraction layer for MemMe — an edge-first AI memory engine.
//!
//! This crate provides:
//! - A trait-object-safe [`LlmProvider`] abstraction for text generation
//! - Prompt templates for fact extraction and memory management (adapted from mem0)
//! - Concrete providers: Ollama (default), OpenAI (feature-gated), and NoOp (fallback)

pub mod error;
pub mod generate;
pub mod noop;
pub mod prompts;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "gemini")]
pub mod gemini;

pub use error::LlmError;
pub use generate::{generate_structured, StructuredGenConfig};

// ---------------------------------------------------------------------------
// JSON repair utility (used by prompts and providers)
// ---------------------------------------------------------------------------

/// Try to repair common JSON formatting issues from LLM output.
pub fn try_repair_json(raw: &str) -> String {
    let mut s = raw.trim().to_string();

    // Strip markdown code fences
    if s.starts_with("```") {
        if let Some(first_newline) = s.find('\n') {
            s = s[first_newline + 1..].to_string();
        }
        if s.ends_with("```") {
            s = s[..s.len() - 3].trim().to_string();
        }
    }

    // Strip <think>...</think> blocks (Qwen think mode)
    while let Some(start) = s.find("<think>") {
        if let Some(end) = s.find("</think>") {
            s = format!("{}{}", &s[..start], &s[end + 8..])
                .trim()
                .to_string();
        } else {
            break;
        }
    }

    // Fix Python-style literals using word boundaries (best-effort heuristic;
    // does not track JSON string boundaries, so values like "True Stories" may be affected).
    use once_cell::sync::Lazy;
    static RE_TRUE: Lazy<regex_lite::Regex> =
        Lazy::new(|| regex_lite::Regex::new(r"\bTrue\b").unwrap());
    static RE_FALSE: Lazy<regex_lite::Regex> =
        Lazy::new(|| regex_lite::Regex::new(r"\bFalse\b").unwrap());
    static RE_NONE: Lazy<regex_lite::Regex> =
        Lazy::new(|| regex_lite::Regex::new(r"\bNone\b").unwrap());
    static RE_TRAILING: Lazy<regex_lite::Regex> =
        Lazy::new(|| regex_lite::Regex::new(r",\s*([}\]])").unwrap());

    s = RE_TRUE.replace_all(&s, "true").to_string();
    s = RE_FALSE.replace_all(&s, "false").to_string();
    s = RE_NONE.replace_all(&s, "null").to_string();

    // Fix single quotes → double quotes (simple cases only)
    if !s.contains('"') && s.contains('\'') {
        s = s.replace('\'', "\"");
    }

    // Remove trailing commas before } or ]
    s = RE_TRAILING.replace_all(&s, "$1").to_string();

    s
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A chat message to send to the LLM.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

/// The role of a chat message participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

/// Options controlling LLM generation behavior.
#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    /// Sampling temperature (0.0 = deterministic, higher = more creative).
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<usize>,
    /// Desired response format.
    pub response_format: Option<ResponseFormat>,
}

impl GenerateOptions {
    /// Create options requesting a JSON response with low temperature
    /// (suitable for structured extraction).
    pub fn json() -> Self {
        Self {
            temperature: Some(0.0),
            max_tokens: None,
            response_format: Some(ResponseFormat::Json),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repair_json_strips_markdown_fences() {
        let input = "```json\n{\"key\": \"value\"}\n```";
        assert_eq!(try_repair_json(input), r#"{"key": "value"}"#);
    }

    #[test]
    fn test_repair_json_strips_think_blocks() {
        let input = "<think>reasoning here</think>{\"a\": 1}";
        assert_eq!(try_repair_json(input), r#"{"a": 1}"#);
    }

    #[test]
    fn test_repair_json_fixes_python_literals() {
        let input = r#"{"active": True, "count": None, "flag": False}"#;
        assert_eq!(
            try_repair_json(input),
            r#"{"active": true, "count": null, "flag": false}"#
        );
    }

    #[test]
    fn test_repair_json_fixes_single_quotes() {
        let input = "{'key': 'value'}";
        assert_eq!(try_repair_json(input), r#"{"key": "value"}"#);
    }

    #[test]
    fn test_repair_json_removes_trailing_commas() {
        let input = r#"{"a": 1, "b": 2, }"#;
        assert_eq!(try_repair_json(input), r#"{"a": 1, "b": 2}"#);

        let input2 = r#"[1, 2, 3, ]"#;
        assert_eq!(try_repair_json(input2), r#"[1, 2, 3]"#);
    }

    #[test]
    fn test_repair_json_noop_on_valid() {
        let input = r#"{"key": "value"}"#;
        assert_eq!(try_repair_json(input), input);
    }

    #[test]
    fn test_generate_options_json() {
        let opts = GenerateOptions::json();
        assert_eq!(opts.temperature, Some(0.0));
        assert_eq!(opts.response_format, Some(ResponseFormat::Json));
        assert!(opts.max_tokens.is_none());
    }

    #[test]
    fn test_message_role_debug() {
        let msg = Message {
            role: MessageRole::System,
            content: "hello".into(),
        };
        assert_eq!(msg.role, MessageRole::System);
    }
}

/// Desired format for the LLM response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseFormat {
    /// Plain text response.
    Text,
    /// JSON response (provider will enforce JSON output if supported).
    Json,
}

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

/// Trait for LLM providers.
///
/// This trait is object-safe so it can be used as `Arc<dyn LlmProvider>`.
/// Providers use synchronous HTTP (ureq) — no tokio runtime required.
pub trait LlmProvider: Send + Sync {
    /// Generate a text response from the given messages.
    fn generate(&self, messages: &[Message], options: &GenerateOptions)
        -> Result<String, LlmError>;

    /// Get the name of this provider (e.g., "ollama", "openai", "noop").
    fn name(&self) -> &str;
}

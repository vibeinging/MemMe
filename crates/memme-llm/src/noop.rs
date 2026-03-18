//! NoOp LLM provider (fallback when no LLM is configured).
//!
//! Returns an error (or empty string) when called. Used when no LLM is
//! configured — the system operates with only vector similarity
//! deduplication without LLM-based fact extraction.

use crate::error::LlmError;
use crate::{GenerateOptions, LlmProvider, Message};

/// A no-operation LLM provider that always returns an error or empty response.
pub struct NoOpProvider {
    /// If true, return an empty string instead of an error.
    pub silent: bool,
}

impl NoOpProvider {
    /// Create a new NoOp provider that returns errors when invoked.
    pub fn new() -> Self {
        Self { silent: false }
    }

    /// Create a new NoOp provider that silently returns empty strings.
    pub fn silent() -> Self {
        Self { silent: true }
    }
}

impl Default for NoOpProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmProvider for NoOpProvider {
    fn generate(
        &self,
        _messages: &[Message],
        _options: &GenerateOptions,
    ) -> Result<String, LlmError> {
        if self.silent {
            Ok(String::new())
        } else {
            Err(LlmError::NotAvailable(
                "No LLM provider configured. \
                 Enable the 'ollama' or 'openai' feature to use LLM capabilities."
                    .to_string(),
            ))
        }
    }

    fn name(&self) -> &str {
        "noop"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MessageRole;

    #[test]
    fn test_noop_returns_error() {
        let provider = NoOpProvider::new();
        let msgs = vec![Message {
            role: MessageRole::User,
            content: "hello".to_string(),
        }];
        let result = provider.generate(&msgs, &GenerateOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_noop_silent_returns_empty() {
        let provider = NoOpProvider::silent();
        let msgs = vec![Message {
            role: MessageRole::User,
            content: "hello".to_string(),
        }];
        let result = provider.generate(&msgs, &GenerateOptions::default());
        assert_eq!(result.unwrap(), "");
    }
}

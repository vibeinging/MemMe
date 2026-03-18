use thiserror::Error;

/// Errors that can occur in the LLM subsystem.
#[derive(Debug, Error)]
pub enum LlmError {
    /// The LLM provider returned an HTTP or network error.
    #[error("LLM request failed: {0}")]
    RequestFailed(String),

    /// The LLM response could not be parsed.
    #[error("Failed to parse LLM response: {0}")]
    ParseError(String),

    /// The LLM provider is not configured or not available.
    #[error("LLM provider not available: {0}")]
    NotAvailable(String),

    /// The response JSON did not match the expected schema.
    #[error("Invalid response format: {0}")]
    InvalidFormat(String),

    /// Configuration error (missing API key, bad URL, etc.).
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Timeout waiting for LLM response.
    #[error("LLM request timed out")]
    Timeout,

    /// Generic internal error.
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

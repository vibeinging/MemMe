use thiserror::Error;

/// Errors that can occur during embedding operations.
#[derive(Debug, Error)]
pub enum EmbedError {
    /// Model initialization failed (download, load, etc.)
    #[error("model initialization failed: {0}")]
    InitError(String),

    /// Embedding inference failed
    #[error("embedding inference failed: {0}")]
    InferenceError(String),

    /// Invalid input (empty text, too long, etc.)
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Network/API error for remote embedders
    #[error("API error: {0}")]
    ApiError(String),

    /// Dimension mismatch
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Generic wrapped error
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

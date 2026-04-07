use thiserror::Error;

/// All errors that can occur during memory operations.
///
/// Each variant wraps a specific failure domain so callers can match
/// on the error kind and decide how to recover.
#[derive(Error, Debug)]
pub enum MemoryError {
    /// Low-level DuckDB storage error (connection, query, constraint violation).
    #[error("DuckDB error: {0}")]
    DuckDb(#[from] duckdb::Error),

    /// Embedding provider failed (network timeout, model not found, dimension mismatch).
    #[error("Embedding error: {0}")]
    Embedding(#[from] memme_embeddings::EmbedError),

    /// The requested memory ID does not exist in the store.
    #[error("Memory not found: {0}")]
    NotFound(String),

    /// JSON serialization or deserialization failed (metadata, export, filter values).
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Invalid configuration value detected during [`crate::MemoryConfig::validate`].
    #[error("Invalid configuration: {0}")]
    Config(String),

    /// LLM provider error (API failure, token limit, unsupported model).
    #[error("LLM error: {0}")]
    Llm(String),

    /// Attempted to modify or delete a memory marked as immutable.
    #[error("Memory is immutable and cannot be modified or deleted: {0}")]
    ImmutableMemory(String),
}

/// Convenience alias used throughout `memme-core`.
pub type Result<T> = std::result::Result<T, MemoryError>;

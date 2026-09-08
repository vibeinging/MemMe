use thiserror::Error;

/// All errors that can occur during memory operations.
///
/// Each variant wraps a specific failure domain so callers can match
/// on the error kind and decide how to recover.
#[derive(Error, Debug)]
pub enum MemoryError {
    /// Low-level storage engine error (connection, query, constraint violation).
    #[error("Storage error: {0}")]
    Storage(String),

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

    /// A database restore failed and the previous primary could not be put
    /// back into a verified, openable state. The process must stop so SQLite
    /// cannot create a new empty database at the primary path.
    #[error(
        "Database restore rollback failed; manual recovery is required from '{rollback_path}': {message}"
    )]
    RollbackFailed {
        rollback_path: String,
        message: String,
    },

    /// Attempted to modify or delete a memory marked as immutable.
    #[error("Memory is immutable and cannot be modified or deleted: {0}")]
    ImmutableMemory(String),
}

/// Convenience alias used throughout `memme-core`.
pub type Result<T> = std::result::Result<T, MemoryError>;

//! memme-embeddings — Embedding model abstraction and implementations for MemMe.
//!
//! Provides a unified `Embedder` trait with multiple backends:
//! - **onnx** (default): Local ONNX inference via fastembed (all-MiniLM-L6-v2, BGE, etc.)
//! - **openai**: OpenAI embeddings API (text-embedding-3-small, etc.)
//! - **ollama**: Ollama local server embeddings API
//! - **mock**: Deterministic fake embeddings for testing
//!
//! # Example
//! ```no_run
//! use memme_embeddings::{Embedder, mock::MockEmbedder};
//!
//! let embedder = MockEmbedder::new(384);
//! let vec = embedder.embed("hello world").unwrap();
//! assert_eq!(vec.len(), 384);
//! ```

mod error;
pub mod mock;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "ollama")]
pub mod ollama;

pub use error::EmbedError;

/// Core embedding trait.
///
/// All embedding backends implement this trait, making them interchangeable
/// via `Arc<dyn Embedder>`. The trait is object-safe and requires `Send + Sync`
/// for safe sharing across threads.
pub trait Embedder: Send + Sync {
    /// Embed a single text string into a float vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;

    /// Embed multiple texts in a batch.
    ///
    /// The default implementation calls `embed` in a loop. Backends that
    /// support native batching (ONNX, OpenAI) override this for efficiency.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Return the dimensionality of the embedding vectors produced.
    fn dimensions(&self) -> usize;

    /// Return the model name / identifier.
    fn model_name(&self) -> &str;
}

/// Convenience function: create the default embedder.
///
/// With the `onnx` feature enabled, this returns an `OnnxEmbedder` with
/// the default model (all-MiniLM-L6-v2). Otherwise, returns a `MockEmbedder`.
pub fn default_embedder() -> Result<Box<dyn Embedder>, EmbedError> {
    #[cfg(feature = "onnx")]
    {
        Ok(Box::new(onnx::OnnxEmbedder::new()?))
    }
    #[cfg(not(feature = "onnx"))]
    {
        Ok(Box::new(mock::MockEmbedder::default_mini()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn trait_is_object_safe() {
        // Verify that Embedder can be used as a trait object
        let embedder: Arc<dyn Embedder> = Arc::new(mock::MockEmbedder::new(128));
        let vec = embedder.embed("test").unwrap();
        assert_eq!(vec.len(), 128);
        assert_eq!(embedder.dimensions(), 128);
        assert_eq!(embedder.model_name(), "mock-128d");
    }

    #[test]
    fn default_batch_implementation() {
        let embedder: Box<dyn Embedder> = Box::new(mock::MockEmbedder::new(64));
        let results = embedder.embed_batch(&["a", "b"]).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 64);
    }
}

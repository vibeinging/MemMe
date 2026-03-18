use crate::{EmbedError, Embedder};

/// A mock embedder that returns deterministic vectors for testing.
///
/// Each dimension is computed as a simple hash of the input text,
/// making results deterministic but unique per input.
pub struct MockEmbedder {
    dims: usize,
    model: String,
}

impl MockEmbedder {
    /// Create a new mock embedder with the given dimensions.
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            model: format!("mock-{dims}d"),
        }
    }

    /// Create a mock embedder that mimics all-MiniLM-L6-v2 dimensions (384).
    pub fn default_mini() -> Self {
        Self::new(384)
    }

    /// Simple deterministic hash for generating fake embeddings.
    fn text_to_vec(&self, text: &str) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.dims);
        // Use a simple hash-like approach: each dimension is derived from
        // the text bytes with a dimension-specific seed.
        let bytes = text.as_bytes();
        for i in 0..self.dims {
            let mut val: u32 = (i as u32).wrapping_mul(2654435761); // Knuth multiplicative hash
            for (j, &b) in bytes.iter().enumerate() {
                val = val
                    .wrapping_add((b as u32).wrapping_mul((j as u32).wrapping_add(i as u32 + 1)));
                val ^= val >> 16;
                val = val.wrapping_mul(0x45d9f3b);
            }
            // Normalize to [-1.0, 1.0] range
            vec.push(((val as f32) / (u32::MAX as f32)) * 2.0 - 1.0);
        }
        // L2 normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vec
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if text.is_empty() {
            return Err(EmbedError::InvalidInput("text is empty".into()));
        }
        Ok(self.text_to_vec(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_produces_correct_dimensions() {
        let embedder = MockEmbedder::new(128);
        let vec = embedder.embed("hello world").unwrap();
        assert_eq!(vec.len(), 128);
    }

    #[test]
    fn mock_is_deterministic() {
        let embedder = MockEmbedder::new(64);
        let v1 = embedder.embed("test input").unwrap();
        let v2 = embedder.embed("test input").unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn mock_different_inputs_differ() {
        let embedder = MockEmbedder::new(64);
        let v1 = embedder.embed("hello").unwrap();
        let v2 = embedder.embed("world").unwrap();
        assert_ne!(v1, v2);
    }

    #[test]
    fn mock_is_normalized() {
        let embedder = MockEmbedder::new(384);
        let vec = embedder.embed("normalize me").unwrap();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn mock_empty_input_errors() {
        let embedder = MockEmbedder::new(64);
        assert!(embedder.embed("").is_err());
    }

    #[test]
    fn mock_batch() {
        let embedder = MockEmbedder::new(64);
        let results = embedder.embed_batch(&["a", "b", "c"]).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].len(), 64);
    }
}

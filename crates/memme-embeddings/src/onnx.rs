use crate::{EmbedError, Embedder};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// Supported ONNX embedding models via fastembed.
#[derive(Debug, Clone, Copy, Default)]
pub enum OnnxModel {
    /// all-MiniLM-L6-v2 — 384 dimensions, fast and lightweight.
    #[default]
    AllMiniLmL6V2,
    /// BGE-small-en-v1.5 — 384 dimensions.
    BgeSmallEnV15,
    /// BGE-base-en-v1.5 — 768 dimensions.
    BgeBaseEnV15,
    /// BGE-large-en-v1.5 — 1024 dimensions.
    BgeLargeEnV15,
    /// BGE-small-zh-v1.5 — 512 dimensions, Chinese.
    BgeSmallZhV15,
    /// BGE-large-zh-v1.5 — 1024 dimensions, Chinese.
    BgeLargeZhV15,
    /// multilingual-e5-small — 384 dimensions, 100+ languages.
    MultilingualE5Small,
    /// multilingual-e5-base — 768 dimensions, 100+ languages.
    MultilingualE5Base,
    /// multilingual-e5-large — 1024 dimensions, 100+ languages.
    MultilingualE5Large,
}

impl OnnxModel {
    fn to_fastembed(self) -> EmbeddingModel {
        match self {
            OnnxModel::AllMiniLmL6V2 => EmbeddingModel::AllMiniLML6V2,
            OnnxModel::BgeSmallEnV15 => EmbeddingModel::BGESmallENV15,
            OnnxModel::BgeBaseEnV15 => EmbeddingModel::BGEBaseENV15,
            OnnxModel::BgeLargeEnV15 => EmbeddingModel::BGELargeENV15,
            OnnxModel::BgeSmallZhV15 => EmbeddingModel::BGESmallZHV15,
            OnnxModel::BgeLargeZhV15 => EmbeddingModel::BGELargeZHV15,
            OnnxModel::MultilingualE5Small => EmbeddingModel::MultilingualE5Small,
            OnnxModel::MultilingualE5Base => EmbeddingModel::MultilingualE5Base,
            OnnxModel::MultilingualE5Large => EmbeddingModel::MultilingualE5Large,
        }
    }

    fn dimensions(self) -> usize {
        match self {
            OnnxModel::AllMiniLmL6V2 => 384,
            OnnxModel::BgeSmallEnV15 => 384,
            OnnxModel::BgeBaseEnV15 => 768,
            OnnxModel::BgeLargeEnV15 => 1024,
            OnnxModel::BgeSmallZhV15 => 512,
            OnnxModel::BgeLargeZhV15 => 1024,
            OnnxModel::MultilingualE5Small => 384,
            OnnxModel::MultilingualE5Base => 768,
            OnnxModel::MultilingualE5Large => 1024,
        }
    }

    fn name(self) -> &'static str {
        match self {
            OnnxModel::AllMiniLmL6V2 => "all-MiniLM-L6-v2",
            OnnxModel::BgeSmallEnV15 => "bge-small-en-v1.5",
            OnnxModel::BgeBaseEnV15 => "bge-base-en-v1.5",
            OnnxModel::BgeLargeEnV15 => "bge-large-en-v1.5",
            OnnxModel::BgeSmallZhV15 => "bge-small-zh-v1.5",
            OnnxModel::BgeLargeZhV15 => "bge-large-zh-v1.5",
            OnnxModel::MultilingualE5Small => "multilingual-e5-small",
            OnnxModel::MultilingualE5Base => "multilingual-e5-base",
            OnnxModel::MultilingualE5Large => "multilingual-e5-large",
        }
    }
}

/// Local ONNX embedding using the fastembed crate.
///
/// This wraps `fastembed::TextEmbedding` and handles model download/caching
/// automatically. The default model is all-MiniLM-L6-v2 (384 dimensions).
pub struct OnnxEmbedder {
    model: TextEmbedding,
    dims: usize,
    name: String,
}

impl OnnxEmbedder {
    /// Create a new ONNX embedder with the default model (all-MiniLM-L6-v2).
    pub fn new() -> Result<Self, EmbedError> {
        Self::with_model(OnnxModel::default())
    }

    /// Create a new ONNX embedder with the specified model.
    pub fn with_model(onnx_model: OnnxModel) -> Result<Self, EmbedError> {
        let options = InitOptions::new(onnx_model.to_fastembed()).with_show_download_progress(true);

        let model = TextEmbedding::try_new(options).map_err(|e| {
            EmbedError::InitError(format!(
                "failed to initialize ONNX model '{}': {}",
                onnx_model.name(),
                e
            ))
        })?;

        Ok(Self {
            model,
            dims: onnx_model.dimensions(),
            name: onnx_model.name().to_string(),
        })
    }

    /// Create with custom cache directory for model files.
    pub fn with_cache_dir(
        onnx_model: OnnxModel,
        cache_dir: impl Into<std::path::PathBuf>,
    ) -> Result<Self, EmbedError> {
        let options = InitOptions::new(onnx_model.to_fastembed())
            .with_show_download_progress(true)
            .with_cache_dir(cache_dir.into());

        let model = TextEmbedding::try_new(options).map_err(|e| {
            EmbedError::InitError(format!(
                "failed to initialize ONNX model '{}': {}",
                onnx_model.name(),
                e
            ))
        })?;

        Ok(Self {
            model,
            dims: onnx_model.dimensions(),
            name: onnx_model.name().to_string(),
        })
    }
}

impl Embedder for OnnxEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if text.is_empty() {
            return Err(EmbedError::InvalidInput("text is empty".into()));
        }

        let results = self
            .model
            .embed(vec![text.to_string()], None)
            .map_err(|e| EmbedError::InferenceError(e.to_string()))?;

        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::InferenceError("model returned no embeddings".into()))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        for (i, text) in texts.iter().enumerate() {
            if text.is_empty() {
                return Err(EmbedError::InvalidInput(format!(
                    "text at index {i} is empty"
                )));
            }
        }

        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let results = self
            .model
            .embed(owned, None)
            .map_err(|e| EmbedError::InferenceError(e.to_string()))?;

        Ok(results)
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn model_name(&self) -> &str {
        &self.name
    }
}

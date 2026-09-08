use crate::{EmbedError, Embedder};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use hf_hub::{api::sync::ApiBuilder, Cache};

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
    /// multilingual-e5-small — 384 dimensions, 100+ languages.
    MultilingualE5Small,
    /// multilingual-e5-base — 768 dimensions, 100+ languages.
    MultilingualE5Base,
    /// multilingual-e5-large — 1024 dimensions, 100+ languages.
    MultilingualE5Large,
}

impl OnnxModel {
    fn to_fastembed(self) -> Result<EmbeddingModel, EmbedError> {
        let model = match self {
            OnnxModel::AllMiniLmL6V2 => EmbeddingModel::AllMiniLML6V2,
            OnnxModel::BgeSmallEnV15 => EmbeddingModel::BGESmallENV15,
            OnnxModel::BgeBaseEnV15 => EmbeddingModel::BGEBaseENV15,
            OnnxModel::BgeLargeEnV15 => EmbeddingModel::BGELargeENV15,
            OnnxModel::BgeSmallZhV15 => EmbeddingModel::BGESmallZHV15,
            OnnxModel::MultilingualE5Small => EmbeddingModel::MultilingualE5Small,
            OnnxModel::MultilingualE5Base => EmbeddingModel::MultilingualE5Base,
            OnnxModel::MultilingualE5Large => EmbeddingModel::MultilingualE5Large,
        };
        Ok(model)
    }

    fn dimensions(self) -> usize {
        match self {
            OnnxModel::AllMiniLmL6V2 => 384,
            OnnxModel::BgeSmallEnV15 => 384,
            OnnxModel::BgeBaseEnV15 => 768,
            OnnxModel::BgeLargeEnV15 => 1024,
            OnnxModel::BgeSmallZhV15 => 512,
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
        let options = init_options(onnx_model.to_fastembed()?, None);
        Self::from_options(onnx_model, options)
    }

    /// Create with custom cache directory for model files.
    pub fn with_cache_dir(
        onnx_model: OnnxModel,
        cache_dir: impl Into<std::path::PathBuf>,
    ) -> Result<Self, EmbedError> {
        let options = init_options(onnx_model.to_fastembed()?, Some(cache_dir.into()));
        Self::from_options(onnx_model, options)
    }

    fn from_options(onnx_model: OnnxModel, options: InitOptions) -> Result<Self, EmbedError> {
        prepare_model_files(
            &options,
            ApiBuilder::from_cache(Cache::new(options.cache_dir.clone()))
                .with_progress(options.show_download_progress)
                .with_retries(2),
        )
        .map_err(|e| {
            EmbedError::InitError(format!(
                "failed to download ONNX model '{}': {e}",
                onnx_model.name()
            ))
        })?;

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

// fastembed 3.6 uses hf-hub 0.3, which cannot resolve relative HTTP redirects.
// Populate every file it reads using the compatible Hub cache layout. Inference
// still opens the ONNX file from disk, including external weights for E5-large.
fn prepare_model_files(options: &InitOptions, builder: ApiBuilder) -> anyhow::Result<()> {
    let model = TextEmbedding::get_model_info(&options.model_name);
    let api = builder.build()?;
    let repo = api.model(model.model_code);
    let mut files = vec![
        model.model_file.as_str(),
        "tokenizer.json",
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ];
    if options.model_name == EmbeddingModel::MultilingualE5Large {
        files.push("model.onnx_data");
    }
    for file in files {
        repo.get(file)?;
    }
    Ok(())
}

fn init_options(model: EmbeddingModel, cache_dir: Option<std::path::PathBuf>) -> InitOptions {
    let mut options = InitOptions {
        model_name: model,
        show_download_progress: true,
        ..Default::default()
    };
    if let Some(cache_dir) = cache_dir {
        options.cache_dir = cache_dir;
    }
    options
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    use std::thread;
    use std::time::Duration;

    #[test]
    fn downloads_relative_redirects_then_reuses_cache_offline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let server_stop = stop.clone();
        let server = thread::spawn(move || {
            while !server_stop.load(Ordering::Relaxed) {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(5));
                    continue;
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut request = Vec::new();
                let mut buf = [0; 1024];
                while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                    let count = stream.read(&mut buf).unwrap();
                    if count == 0 {
                        break;
                    }
                    request.extend_from_slice(&buf[..count]);
                }
                let request = String::from_utf8(request).unwrap();
                let path = request.split_whitespace().nth(1).unwrap();
                let name = path.rsplit('/').next().unwrap();
                let response = if !path.starts_with("/files/") {
                    format!("HTTP/1.1 307 Temporary Redirect\r\nLocation: /files/{name}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                } else {
                    let body = format!("fixture-{name}");
                    let length = body.len();
                    let range_request = request
                        .to_ascii_lowercase()
                        .contains("range: bytes=0-0\r\n");
                    let data = if range_request { &body[..1] } else { &body };
                    format!("HTTP/1.1 206 Partial Content\r\nx-repo-commit: 0123456789abcdef\r\nETag: {name}\r\nContent-Range: bytes 0-{}/{length}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{data}", data.len() - 1, data.len())
                };
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        let dir = tempfile::tempdir().unwrap();
        let options = init_options(EmbeddingModel::BGESmallZHV15, Some(dir.path().into()));
        let builder = || {
            ApiBuilder::from_cache(Cache::new(dir.path().into()))
                .with_endpoint(endpoint.clone())
                .with_token(None)
                .with_progress(false)
        };
        let result = prepare_model_files(&options, builder());
        stop.store(true, Ordering::Relaxed);
        server.join().unwrap();
        result.unwrap();

        // The endpoint is now closed. The same initialization must need no HTTP.
        prepare_model_files(&options, builder()).unwrap();
        let repo = Cache::new(dir.path().into()).model("Xenova/bge-small-zh-v1.5".into());
        assert_eq!(
            std::fs::read(repo.get("config.json").unwrap()).unwrap(),
            b"fixture-config.json"
        );
    }
}

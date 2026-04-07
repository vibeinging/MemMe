use crate::{EmbedError, Embedder};
use serde::{Deserialize, Serialize};

/// Ollama embedding model variants.
#[derive(Debug, Clone)]
pub enum OllamaModel {
    /// all-minilm — 384 dimensions
    AllMiniLm,
    /// nomic-embed-text — 768 dimensions
    NomicEmbedText,
    /// mxbai-embed-large — 1024 dimensions
    MxbaiEmbedLarge,
    /// Custom model name with explicit dimensions
    Custom { name: String, dims: usize },
}

impl OllamaModel {
    fn name(&self) -> &str {
        match self {
            OllamaModel::AllMiniLm => "all-minilm",
            OllamaModel::NomicEmbedText => "nomic-embed-text",
            OllamaModel::MxbaiEmbedLarge => "mxbai-embed-large",
            OllamaModel::Custom { name, .. } => name,
        }
    }

    fn dimensions(&self) -> usize {
        match self {
            OllamaModel::AllMiniLm => 384,
            OllamaModel::NomicEmbedText => 768,
            OllamaModel::MxbaiEmbedLarge => 1024,
            OllamaModel::Custom { dims, .. } => *dims,
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for OllamaModel {
    fn default() -> Self {
        OllamaModel::AllMiniLm
    }
}

/// Ollama embeddings client.
///
/// Calls the Ollama `POST /api/embed` endpoint.
/// Exposes a synchronous API by using `tokio::runtime::Handle::block_on`.
pub struct OllamaEmbedder {
    host: String,
    model: OllamaModel,
    client: reqwest::Client,
}

impl OllamaEmbedder {
    /// Create a new Ollama embedder with default settings.
    /// Default host: `http://localhost:11434`, default model: `all-minilm`.
    pub fn new() -> Self {
        Self {
            host: "http://localhost:11434".to_string(),
            model: OllamaModel::default(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Set a custom host URL (e.g., `http://192.168.1.100:11434`).
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set the embedding model.
    pub fn with_model(mut self, model: OllamaModel) -> Self {
        self.model = model;
        self
    }

    /// Create from environment variable `OLLAMA_HOST`, falling back to localhost.
    pub fn from_env() -> Self {
        let host =
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
        Self {
            host,
            model: OllamaModel::default(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Internal async embed call using Ollama's /api/embed endpoint.
    /// Retries up to 3 times with exponential backoff on transient errors.
    async fn embed_async(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
        let request = OllamaEmbedRequest {
            model: self.model.name().to_string(),
            input,
        };

        let url = format!("{}/api/embed", self.host);
        let max_retries = 3u32;
        let mut last_err = EmbedError::ApiError("no attempts made".into());

        for attempt in 0..max_retries {
            if attempt > 0 {
                let wait = std::time::Duration::from_millis(1000 * 2u64.pow(attempt));
                tracing::warn!(attempt, "Retrying Ollama embed request after: {last_err}");
                tokio::time::sleep(wait).await;
            }

            let result = self.client.post(&url).json(&request).send().await;

            match result {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        let resp: OllamaEmbedResponse = response.json().await.map_err(|e| {
                            EmbedError::ApiError(format!("failed to parse Ollama response: {e}"))
                        })?;
                        return Ok(resp.embeddings);
                    }

                    let body = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "failed to read body".into());

                    if status.as_u16() == 429 || status.is_server_error() {
                        last_err =
                            EmbedError::ApiError(format!("Ollama API returned {status}: {body}"));
                        continue;
                    }

                    // Non-retriable error — fail immediately
                    return Err(EmbedError::ApiError(format!(
                        "Ollama API returned {status}: {body}"
                    )));
                }
                Err(e) => {
                    // Network/timeout errors are retriable
                    last_err = EmbedError::ApiError(format!("request to Ollama failed: {e}"));
                    continue;
                }
            }
        }

        Err(last_err)
    }

    /// Run the async embed within the current tokio runtime.
    fn block_embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(self.embed_async(input))),
            Err(_) => {
                let rt = tokio::runtime::Runtime::new().map_err(|e| EmbedError::Other(e.into()))?;
                rt.block_on(self.embed_async(input))
            }
        }
    }
}

impl Default for OllamaEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder for OllamaEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if text.is_empty() {
            return Err(EmbedError::InvalidInput("text is empty".into()));
        }
        let results = self.block_embed(vec![text.to_string()])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::ApiError("Ollama returned no embeddings".into()))
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
        self.block_embed(owned)
    }

    fn dimensions(&self) -> usize {
        self.model.dimensions()
    }

    fn model_name(&self) -> &str {
        self.model.name()
    }
}

// --- API request/response types ---

#[derive(Debug, Serialize)]
struct OllamaEmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

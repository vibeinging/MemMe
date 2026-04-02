use crate::{EmbedError, Embedder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Protocol adapter for custom embedding API formats.
///
/// Implement this trait to support non-OpenAI embedding API formats.
/// The HTTP transport layer (retry, concurrency, auth) is handled by `OpenAiEmbedder`;
/// you only need to define how to build the request body and parse the response.
///
/// # Example
/// ```no_run
/// use memme_embeddings::openai::EmbedProtocol;
/// use memme_embeddings::EmbedError;
///
/// struct DashScopeEmbed;
///
/// impl EmbedProtocol for DashScopeEmbed {
///     fn build_request(&self, model: &str, texts: &[String], _dims: Option<usize>) -> serde_json::Value {
///         serde_json::json!({
///             "model": model,
///             "input": { "texts": texts },
///         })
///     }
///
///     fn parse_response(&self, raw: &str) -> Result<Vec<Vec<f32>>, EmbedError> {
///         let v: serde_json::Value = serde_json::from_str(raw)
///             .map_err(|e| EmbedError::ApiError(e.to_string()))?;
///         // Extract embeddings from DashScope format...
///         Ok(vec![])
///     }
/// }
/// ```
pub trait EmbedProtocol: Send + Sync {
    /// Build the JSON request body for the embedding API.
    fn build_request(
        &self,
        model: &str,
        texts: &[String],
        dimensions: Option<usize>,
    ) -> serde_json::Value;

    /// Parse the raw response body into embedding vectors.
    fn parse_response(&self, raw: &str) -> Result<Vec<Vec<f32>>, EmbedError>;
}

/// OpenAI embedding model variants.
#[derive(Debug, Clone)]
pub enum OpenAiModel {
    /// text-embedding-3-small — 1536 dimensions (default)
    TextEmbedding3Small,
    /// text-embedding-3-large — 3072 dimensions
    TextEmbedding3Large,
    /// text-embedding-ada-002 — 1536 dimensions (legacy)
    TextEmbeddingAda002,
    /// Custom model name with explicit dimensions.
    /// Set `send_dims = false` for providers that don't accept the `dimensions` parameter.
    Custom { name: String, dims: usize, send_dims: bool },
}

impl OpenAiModel {
    fn name(&self) -> &str {
        match self {
            OpenAiModel::TextEmbedding3Small => "text-embedding-3-small",
            OpenAiModel::TextEmbedding3Large => "text-embedding-3-large",
            OpenAiModel::TextEmbeddingAda002 => "text-embedding-ada-002",
            OpenAiModel::Custom { name, .. } => name,
        }
    }

    fn dimensions(&self) -> usize {
        match self {
            OpenAiModel::TextEmbedding3Small => 1536,
            OpenAiModel::TextEmbedding3Large => 3072,
            OpenAiModel::TextEmbeddingAda002 => 1536,
            OpenAiModel::Custom { dims, .. } => *dims,
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for OpenAiModel {
    fn default() -> Self {
        OpenAiModel::TextEmbedding3Small
    }
}

/// Configuration for concurrent embedding requests.
#[derive(Debug, Clone)]
pub struct ConcurrencyConfig {
    /// Maximum texts per batch request (API limit per request)
    pub batch_size: usize,
    /// Maximum concurrent API requests (rate limit consideration)
    pub max_concurrent: usize,
    /// Delay between concurrent request launches (to avoid burst rate limits)
    pub stagger_ms: u64,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,   // Safe for most providers
            max_concurrent: 5, // Conservative default
            stagger_ms: 100,   // 100ms between launches
        }
    }
}

/// OpenAI embeddings API client.
///
/// Calls the OpenAI `/v1/embeddings` endpoint using reqwest.
/// Exposes a synchronous API by using `tokio::runtime::Handle::block_on`.
pub struct OpenAiEmbedder {
    api_key: String,
    base_url: String,
    model: OpenAiModel,
    client: reqwest::Client,
    /// Concurrency configuration
    concurrency: ConcurrencyConfig,
    /// Custom protocol adapter. Arc (not Box) because block_embed_concurrent clones Self.
    protocol: Option<Arc<dyn EmbedProtocol>>,
}

impl OpenAiEmbedder {
    /// Create a new OpenAI embedder with the given API key and full endpoint URL.
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: OpenAiModel::default(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            concurrency: ConcurrencyConfig::default(),
            protocol: None,
        }
    }

    /// Set the embedding model.
    pub fn with_model(mut self, model: OpenAiModel) -> Self {
        self.model = model;
        self
    }

    /// Set the batch size for embed_batch calls.
    /// Different API providers have different limits:
    /// - OpenAI: ~2048 texts per request (but smaller batches are safer)
    /// - Some providers: as low as 10 texts per request
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.concurrency.batch_size = batch_size.max(1);
        self
    }

    /// Set the maximum concurrent API requests.
    /// Consider your API rate limits:
    /// - OpenAI: varies by tier, typically 3-10 concurrent
    /// - Azure OpenAI: depends on PTU allocation
    /// - Local models: can be higher (10-20)
    pub fn with_max_concurrent(mut self, max_concurrent: usize) -> Self {
        self.concurrency.max_concurrent = max_concurrent.max(1);
        self
    }

    /// Set the stagger delay between concurrent request launches (milliseconds).
    /// Helps avoid burst rate limits. Default: 100ms.
    pub fn with_stagger_ms(mut self, stagger_ms: u64) -> Self {
        self.concurrency.stagger_ms = stagger_ms;
        self
    }

    /// Set full concurrency configuration.
    pub fn with_concurrency(mut self, config: ConcurrencyConfig) -> Self {
        self.concurrency = config;
        self
    }

    /// Set a custom protocol adapter for non-OpenAI API formats.
    ///
    /// When set, the default OpenAI request/response format is bypassed entirely.
    /// The HTTP transport (retry, concurrency, auth) remains unchanged.
    pub fn with_protocol(mut self, protocol: impl EmbedProtocol + 'static) -> Self {
        self.protocol = Some(Arc::new(protocol));
        self
    }

    /// Create from environment variables `OPENAI_API_KEY` and `EMBEDDING_URL`.
    pub fn from_env() -> Result<Self, EmbedError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            EmbedError::InitError("OPENAI_API_KEY environment variable not set".into())
        })?;
        let base_url = std::env::var("EMBEDDING_URL").map_err(|_| {
            EmbedError::InitError("EMBEDDING_URL environment variable not set".into())
        })?;
        Ok(Self::new(api_key, base_url))
    }

    /// Internal async embed call with exponential backoff retry.
    ///
    /// Retries up to 3 times on transient errors (429 rate limit, 5xx server errors,
    /// network timeouts). Non-retriable errors (400 bad request, auth errors) fail immediately.
    async fn embed_async(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
        use std::time::Duration;

        // For Custom models, pass the dimensions parameter to truncate server-side
        let dimensions = match &self.model {
            OpenAiModel::Custom { dims, send_dims, .. } => {
                if *send_dims { Some(*dims) } else { None }
            }
            _ => None,
        };

        // Build request body: custom protocol or default OpenAI format
        let request_body = if let Some(ref proto) = self.protocol {
            proto.build_request(self.model.name(), &texts, dimensions)
        } else {
            let request = EmbeddingRequest {
                model: self.model.name().to_string(),
                input: texts,
                dimensions,
            };
            serde_json::to_value(&request)
                .map_err(|e| EmbedError::ApiError(format!("failed to serialize request: {e}")))?
        };

        let url: &str = &self.base_url;
        let max_retries = 3u32;
        let mut last_err = EmbedError::ApiError("no attempts made".into());

        for attempt in 0..max_retries {
            if attempt > 0 {
                let wait = Duration::from_millis(1000 * 2u64.pow(attempt));
                tracing::warn!(attempt, "Retrying embedding request after: {last_err}");
                tokio::time::sleep(wait).await;
            }

            let result = self
                .client
                .post(url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&request_body)
                .send()
                .await;

            match result {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        if let Some(ref proto) = self.protocol {
                            let raw = response.text().await.map_err(|e| {
                                EmbedError::ApiError(format!("failed to read response: {e}"))
                            })?;
                            return proto.parse_response(&raw);
                        }

                        let resp: EmbeddingResponse = response.json().await.map_err(|e| {
                            EmbedError::ApiError(format!("failed to parse response: {e}"))
                        })?;

                        // Sort by index to ensure correct ordering
                        let mut data = resp.data;
                        data.sort_by_key(|d| d.index);

                        return Ok(data.into_iter().map(|d| d.embedding).collect());
                    }

                    let body = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "failed to read body".into());

                    if status.as_u16() == 429 || status.is_server_error() {
                        // Retriable error
                        last_err =
                            EmbedError::ApiError(format!("OpenAI API returned {status}: {body}"));
                        continue;
                    }

                    // Non-retriable error (400, 401, 403, etc.) — fail immediately
                    return Err(EmbedError::ApiError(format!(
                        "OpenAI API returned {status}: {body}"
                    )));
                }
                Err(e) => {
                    // Network/timeout errors are retriable
                    last_err = EmbedError::ApiError(format!("request failed: {e}"));
                    continue;
                }
            }
        }

        Err(last_err)
    }

    /// Run the async embed within the current tokio runtime.
    fn block_embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
        // Try to use the current tokio runtime handle
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                // We're inside an async context — we need to spawn a blocking task
                // to avoid blocking the executor. Use block_in_place if available.
                tokio::task::block_in_place(|| handle.block_on(self.embed_async(texts)))
            }
            Err(_) => {
                // No runtime — create a temporary one
                let rt = tokio::runtime::Runtime::new().map_err(|e| EmbedError::Other(e.into()))?;
                rt.block_on(self.embed_async(texts))
            }
        }
    }

    /// Execute multiple embedding batches concurrently with rate limiting.
    fn block_embed_concurrent(
        &self,
        batches: Vec<Vec<String>>,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let max_concurrent = self.concurrency.max_concurrent;
        let stagger_ms = self.concurrency.stagger_ms;
        let total_count: usize = batches.iter().map(|b| b.len()).sum();

        let this = Arc::new(Self {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            client: self.client.clone(),
            concurrency: self.concurrency.clone(),
            protocol: self.protocol.clone(),
        });

        let rt_handle = match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle,
            Err(_) => {
                let rt = tokio::runtime::Runtime::new().map_err(|e| EmbedError::Other(e.into()))?;
                let mut all_embeddings = Vec::with_capacity(total_count);
                for batch in batches {
                    let result = rt.block_on(this.embed_async(batch))?;
                    all_embeddings.extend(result);
                }
                return Ok(all_embeddings);
            }
        };

        tokio::task::block_in_place(|| {
            rt_handle.block_on(async {
                use tokio::task::JoinSet;

                let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
                let mut join_set = JoinSet::new();
                let mut all_embeddings: Vec<(usize, Vec<Vec<f32>>)> = Vec::new();

                for (batch_idx, batch) in batches.into_iter().enumerate() {
                    let permit = semaphore
                        .clone()
                        .acquire_owned()
                        .await
                        .map_err(|e| EmbedError::Other(e.into()))?;

                    if batch_idx > 0 && stagger_ms > 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(stagger_ms)).await;
                    }

                    let this_clone = Arc::clone(&this);
                    join_set.spawn(async move {
                        let result = this_clone.embed_async(batch).await;
                        drop(permit);
                        (batch_idx, result)
                    });
                }

                while let Some(res) = join_set.join_next().await {
                    match res {
                        Ok((idx, Ok(embeddings))) => {
                            all_embeddings.push((idx, embeddings));
                        }
                        Ok((idx, Err(e))) => {
                            return Err(EmbedError::ApiError(format!(
                                "Batch {} failed: {}",
                                idx, e
                            )));
                        }
                        Err(e) => {
                            return Err(EmbedError::Other(e.into()));
                        }
                    }
                }

                all_embeddings.sort_by_key(|(idx, _)| *idx);
                let mut result = Vec::with_capacity(total_count);
                for (_, embeddings) in all_embeddings {
                    result.extend(embeddings);
                }
                Ok(result)
            })
        })
    }
}

impl Embedder for OpenAiEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if text.is_empty() {
            return Err(EmbedError::InvalidInput("text is empty".into()));
        }
        let results = self.block_embed(vec![text.to_string()])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::ApiError("API returned no embeddings".into()))
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

        // Split into batches
        let batches: Vec<Vec<String>> = texts
            .chunks(self.concurrency.batch_size)
            .map(|chunk| chunk.iter().map(|s| s.to_string()).collect())
            .collect();

        // If only one batch, no need for concurrency
        if batches.len() == 1 {
            return self.block_embed(batches.into_iter().next().unwrap());
        }

        // Execute batches concurrently
        self.block_embed_concurrent(batches)
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
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

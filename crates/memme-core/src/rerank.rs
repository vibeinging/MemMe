//! Reranker abstraction for re-scoring search results.

use crate::error::Result;
use crate::types::MemoryResult;

/// Trait for reranking search results.
/// Implementations can use cross-encoders, LLM-based scoring, or external APIs.
pub trait Reranker: Send + Sync {
    /// Re-score and reorder results based on relevance to query.
    /// Returns results sorted by relevance (best first).
    fn rerank(
        &self,
        query: &str,
        results: Vec<MemoryResult>,
        top_k: usize,
    ) -> Result<Vec<MemoryResult>>;
}

/// Simple pass-through reranker (no-op). Returns results as-is.
pub struct NoOpReranker;

impl Reranker for NoOpReranker {
    fn rerank(
        &self,
        _query: &str,
        mut results: Vec<MemoryResult>,
        top_k: usize,
    ) -> Result<Vec<MemoryResult>> {
        results.truncate(top_k);
        Ok(results)
    }
}

/// LLM-based reranker using the OpenAI-format API.
/// Asks the LLM to score each result's relevance to the query on a 0-10 scale.
pub struct LlmReranker {
    llm: std::sync::Arc<dyn memme_llm::LlmProvider>,
}

impl LlmReranker {
    pub fn new(llm: std::sync::Arc<dyn memme_llm::LlmProvider>) -> Self {
        Self { llm }
    }
}

impl Reranker for LlmReranker {
    fn rerank(
        &self,
        query: &str,
        results: Vec<MemoryResult>,
        top_k: usize,
    ) -> Result<Vec<MemoryResult>> {
        use crate::error::MemoryError;

        if results.is_empty() {
            return Ok(results);
        }

        // Build prompt asking LLM to score each result
        let docs: Vec<String> = results
            .iter()
            .enumerate()
            .map(|(i, r)| format!("[{}] {}", i, r.content))
            .collect();

        let system = "You are a relevance scorer. Given a query and a list of documents, \
            score each document's relevance to the query on a scale of 0-10 \
            (10 = most relevant). Return JSON: {\"scores\": [score0, score1, ...]}";
        let user_msg = format!("Query: {}\n\nDocuments:\n{}", query, docs.join("\n"));

        let messages = vec![
            memme_llm::Message {
                role: memme_llm::MessageRole::System,
                content: system.to_string(),
            },
            memme_llm::Message {
                role: memme_llm::MessageRole::User,
                content: user_msg,
            },
        ];
        let config = memme_llm::StructuredGenConfig {
            base_temperature: Some(0.0),
            max_tokens: Some(200),
            response_format: Some(memme_llm::ResponseFormat::Json),
            ..Default::default()
        };

        let scores_vec =
            memme_llm::generate_structured(self.llm.as_ref(), &messages, &config, |raw| {
                let parsed: serde_json::Value = serde_json::from_str(raw)
                    .map_err(|e| format!("Failed to parse reranker response: {e}"))?;
                let scores = parsed["scores"]
                    .as_array()
                    .ok_or_else(|| "Reranker response missing 'scores' array".to_string())?;
                Ok(scores
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0))
                    .collect::<Vec<f64>>())
            })
            .map_err(|e| MemoryError::Llm(e.to_string()))?;

        let mut scored: Vec<(f64, MemoryResult)> = results
            .into_iter()
            .enumerate()
            .map(|(i, mut r)| {
                let score = scores_vec.get(i).copied().unwrap_or(0.0);
                r.score = Some(score as f32);
                (score, r)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        Ok(scored.into_iter().map(|(_, r)| r).collect())
    }
}

/// Protocol adapter for custom rerank API formats.
///
/// Implement this trait to support non-Jina/Cohere/DashScope rerank API formats.
/// The HTTP transport layer (retry, backoff, auth) is handled by `ApiReranker`;
/// you only define request/response format.
///
/// # Example
/// ```no_run
/// use memme_core::rerank::RerankProtocol;
/// use memme_core::error::MemoryError;
///
/// struct CustomRerank;
///
/// impl RerankProtocol for CustomRerank {
///     fn build_request(&self, model: &str, query: &str, documents: &[&str], top_n: usize) -> serde_json::Value {
///         serde_json::json!({ "model": model, "query": query, "passages": documents, "top_k": top_n })
///     }
///
///     fn parse_response(&self, raw: &str) -> Result<Vec<(usize, f64)>, MemoryError> {
///         // Parse custom format into (index, score) pairs...
///         Ok(vec![])
///     }
/// }
/// ```
#[cfg(feature = "api-rerank")]
pub trait RerankProtocol: Send + Sync {
    /// Build the JSON request body for the rerank API.
    fn build_request(
        &self,
        model: &str,
        query: &str,
        documents: &[&str],
        top_n: usize,
    ) -> serde_json::Value;

    /// Parse the raw response body into `(index, score)` pairs.
    fn parse_response(
        &self,
        raw: &str,
    ) -> std::result::Result<Vec<(usize, f64)>, crate::error::MemoryError>;
}

/// API-based cross-encoder reranker compatible with Jina / Cohere rerank endpoints.
///
/// Uses `reqwest::blocking` for synchronous HTTP — no tokio runtime required,
/// safe to call from Python/Node FFI contexts.
///
/// Requires the `api-rerank` feature.
#[cfg(feature = "api-rerank")]
pub struct ApiReranker {
    config: RerankConfig,
    client: reqwest::blocking::Client,
    /// Custom protocol adapter for non-standard rerank API formats.
    protocol: Option<Box<dyn RerankProtocol>>,
}

/// Configuration for the API-based reranker.
#[cfg(feature = "api-rerank")]
#[derive(Clone)]
pub struct RerankConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

#[cfg(feature = "api-rerank")]
impl std::fmt::Debug for RerankConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RerankConfig")
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .finish()
    }
}

#[cfg(feature = "api-rerank")]
impl RerankConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://dashscope.aliyuncs.com".to_string(),
            model: "gte-rerank-v2".to_string(),
        }
    }
}

#[cfg(feature = "api-rerank")]
impl ApiReranker {
    pub fn new(config: RerankConfig) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        Self {
            config,
            client,
            protocol: None,
        }
    }

    /// Set a custom protocol adapter for non-standard rerank API formats.
    ///
    /// When set, the default DashScope/Jina/Cohere format detection is bypassed entirely.
    /// The HTTP transport (retry, backoff, auth) remains unchanged.
    pub fn with_protocol(mut self, protocol: impl RerankProtocol + 'static) -> Self {
        self.protocol = Some(Box::new(protocol));
        self
    }

    fn do_request(
        &self,
        query: &str,
        documents: &[&str],
        top_n: usize,
    ) -> std::result::Result<Vec<RerankApiResult>, crate::error::MemoryError> {
        use crate::error::MemoryError;

        // Detect DashScope native API vs standard Jina/Cohere format
        let is_dashscope = self.config.base_url.contains("dashscope");

        let url = if is_dashscope {
            format!(
                "{}/api/v1/services/rerank/text-rerank/text-rerank",
                self.config.base_url.trim_end_matches('/')
            )
        } else if self.config.base_url.contains("/rerank") {
            self.config.base_url.clone()
        } else {
            format!("{}/v1/rerank", self.config.base_url)
        };

        let body = if let Some(ref proto) = self.protocol {
            proto.build_request(&self.config.model, query, documents, top_n)
        } else if is_dashscope {
            // DashScope native format: nested input/parameters
            serde_json::json!({
                "model": self.config.model,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {
                    "top_n": top_n,
                },
            })
        } else {
            // Standard Jina/Cohere format: flat structure
            serde_json::json!({
                "model": self.config.model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            })
        };

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .json(&body)
            .send()
            .map_err(|e| MemoryError::Config(format!("Rerank API request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(MemoryError::Config(format!(
                    "Rerank API rate limited: {body_text}"
                )));
            }
            return Err(MemoryError::Config(format!(
                "Rerank API error ({status}): {body_text}"
            )));
        }

        if let Some(ref proto) = self.protocol {
            let raw = response
                .text()
                .map_err(|e| MemoryError::Config(format!("failed to read response: {e}")))?;
            let pairs = proto.parse_response(&raw)?;
            return Ok(pairs
                .into_iter()
                .map(|(index, score)| RerankApiResult { index, score })
                .collect());
        }

        let resp_body: serde_json::Value = response
            .json()
            .map_err(|e| MemoryError::Config(format!("Rerank API invalid JSON: {e}")))?;

        // Support both standard (Jina/Cohere) and DashScope (output.results) formats
        let results = resp_body["results"]
            .as_array()
            .or_else(|| resp_body["output"]["results"].as_array())
            .ok_or_else(|| {
                MemoryError::Config(format!(
                    "Rerank API: missing 'results' array in response: {}",
                    serde_json::to_string(&resp_body).unwrap_or_default()
                ))
            })?;

        let mut parsed = Vec::with_capacity(results.len());
        for item in results {
            let index = item["index"].as_u64().unwrap_or(0) as usize;
            let score = item["relevance_score"]
                .as_f64()
                .or_else(|| item["score"].as_f64())
                .unwrap_or(0.0);
            parsed.push(RerankApiResult { index, score });
        }

        Ok(parsed)
    }
}

#[cfg(feature = "api-rerank")]
struct RerankApiResult {
    index: usize,
    score: f64,
}

#[cfg(feature = "api-rerank")]
impl Reranker for ApiReranker {
    fn rerank(
        &self,
        query: &str,
        results: Vec<MemoryResult>,
        top_k: usize,
    ) -> Result<Vec<MemoryResult>> {
        if results.is_empty() {
            return Ok(results);
        }

        let documents: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();
        let effective_top_k = top_k.min(results.len());

        // Retry with exponential backoff (3 attempts)
        let mut last_err = None;
        for attempt in 0..3u64 {
            if attempt > 0 {
                let wait = std::time::Duration::from_secs(2u64.pow(attempt as u32));
                tracing::warn!(attempt, "Retrying rerank API request");
                std::thread::sleep(wait);
            }

            match self.do_request(query, &documents, effective_top_k) {
                Ok(api_results) => {
                    let mut reranked: Vec<MemoryResult> = api_results
                        .into_iter()
                        .filter_map(|rr| {
                            if rr.index < results.len() {
                                let mut mem = results[rr.index].clone();
                                mem.score = Some(rr.score as f32);
                                Some(mem)
                            } else {
                                None
                            }
                        })
                        .collect();
                    reranked.truncate(effective_top_k);
                    return Ok(reranked);
                }
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            crate::error::MemoryError::Config("Rerank API: no attempts made".into())
        }))
    }
}

/// ONNX cross-encoder reranker using `fastembed::TextRerank` for local offline reranking.
///
/// When the `onnx-rerank` feature is enabled and a model is available, this reranker
/// uses a cross-encoder model to re-score query-document pairs. If model loading fails,
/// it falls back to the NoOp strategy (preserving original order) with a warning.
#[cfg(feature = "onnx-rerank")]
pub struct OnnxReranker {
    model: std::result::Result<fastembed::TextRerank, String>,
    top_n: usize,
}

#[cfg(feature = "onnx-rerank")]
impl OnnxReranker {
    /// Create a new ONNX reranker.
    ///
    /// `model_name` is descriptive only (the default cross-encoder model is used).
    /// `top_n` controls how many results to return from the reranker.
    /// If model initialization fails, the reranker degrades gracefully to NoOp behavior.
    pub fn new(model_name: &str, top_n: usize) -> Self {
        let model = fastembed::TextRerank::try_new(Default::default())
            .map_err(|e| format!("Failed to load rerank model '{}': {}", model_name, e));

        if model.is_err() {
            tracing::warn!(
                "OnnxReranker: model '{}' not available, falling back to NoOp. Error: {}",
                model_name,
                model.as_ref().unwrap_err()
            );
        }

        Self { model, top_n }
    }

    /// Create a reranker with a specific fastembed init options.
    pub fn with_options(options: fastembed::RerankInitOptions, top_n: usize) -> Self {
        let model = fastembed::TextRerank::try_new(options)
            .map_err(|e| format!("Failed to load rerank model: {}", e));

        Self { model, top_n }
    }

    /// Returns true if the underlying model was loaded successfully.
    pub fn is_available(&self) -> bool {
        self.model.is_ok()
    }
}

#[cfg(feature = "onnx-rerank")]
impl Reranker for OnnxReranker {
    fn rerank(
        &self,
        query: &str,
        mut results: Vec<MemoryResult>,
        top_k: usize,
    ) -> Result<Vec<MemoryResult>> {
        if results.is_empty() {
            return Ok(results);
        }

        let model = match &self.model {
            Ok(m) => m,
            Err(msg) => {
                tracing::warn!("OnnxReranker fallback to NoOp: {}", msg);
                results.truncate(top_k);
                return Ok(results);
            }
        };

        let effective_top_k = top_k.min(self.top_n).min(results.len());

        // Build documents list for the reranker
        let documents: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();

        // Run reranking
        let rerank_results = model
            .rerank(query, documents, true, Some(effective_top_k))
            .map_err(|e| crate::error::MemoryError::Config(format!("ONNX rerank failed: {}", e)))?;

        // Map reranker output back to MemoryResults, preserving order by score
        let mut reranked: Vec<MemoryResult> = rerank_results
            .into_iter()
            .filter_map(|rr| {
                let idx = rr.index;
                if idx < results.len() {
                    let mut mem = results[idx].clone();
                    mem.score = Some(rr.score as f32);
                    Some(mem)
                } else {
                    None
                }
            })
            .collect();

        reranked.truncate(effective_top_k);
        Ok(reranked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(id: &str, content: &str, score: Option<f32>) -> MemoryResult {
        MemoryResult {
            id: id.to_string(),
            content: content.to_string(),
            user_id: "user1".to_string(),
            agent_id: None,
            app_id: None,
            run_id: None,
            score,
            created_at: "2024-01-01".to_string(),
            updated_at: "2024-01-01".to_string(),
            metadata: None,
            access_count: None,
            importance: None,
            immutable: false,
            expiration_date: None,
            categories: None,
            memory_type: None,
            retention: None,
            stability: None,
            privacy: "syncable".to_string(),
            event_time: None,
            episode_id: None,
            session_id: None,
            resolution: crate::types::Resolution::Granular,
        }
    }

    #[test]
    fn test_noop_reranker() {
        let reranker = NoOpReranker;
        let results = vec![
            make_result("a", "alpha", Some(0.5)),
            make_result("b", "beta", Some(0.3)),
            make_result("c", "gamma", Some(0.9)),
        ];
        let reranked = reranker.rerank("query", results, 10).unwrap();
        assert_eq!(reranked.len(), 3);
        // Order should be preserved
        assert_eq!(reranked[0].id, "a");
        assert_eq!(reranked[1].id, "b");
        assert_eq!(reranked[2].id, "c");
    }

    #[test]
    fn test_noop_reranker_empty() {
        let reranker = NoOpReranker;
        let results: Vec<MemoryResult> = vec![];
        let reranked = reranker.rerank("query", results, 10).unwrap();
        assert!(reranked.is_empty());
    }

    #[test]
    fn test_reranker_top_k() {
        let reranker = NoOpReranker;
        let results = vec![
            make_result("a", "alpha", Some(0.5)),
            make_result("b", "beta", Some(0.3)),
            make_result("c", "gamma", Some(0.9)),
        ];
        let reranked = reranker.rerank("query", results, 2).unwrap();
        assert_eq!(reranked.len(), 2);
        assert_eq!(reranked[0].id, "a");
        assert_eq!(reranked[1].id, "b");
    }

    mod smart_tests {
        use super::*;
        use memme_llm::{GenerateOptions, LlmError, LlmProvider, Message};
        use std::sync::Arc;

        /// Mock LLM that returns preset scores for reranking.
        struct MockLlm {
            response: String,
        }

        impl MockLlm {
            fn with_scores(scores: &[f64]) -> Self {
                let json = serde_json::json!({ "scores": scores });
                Self {
                    response: json.to_string(),
                }
            }
        }

        impl LlmProvider for MockLlm {
            fn generate(
                &self,
                _messages: &[Message],
                _options: &GenerateOptions,
            ) -> std::result::Result<String, LlmError> {
                Ok(self.response.clone())
            }

            fn name(&self) -> &str {
                "mock"
            }
        }

        #[test]
        fn test_llm_reranker_with_mock() {
            // LLM returns scores [3, 8, 1] — so doc "b" (score 8) should come first
            let llm = Arc::new(MockLlm::with_scores(&[3.0, 8.0, 1.0]));
            let reranker = LlmReranker::new(llm);

            let results = vec![
                make_result("a", "alpha", None),
                make_result("b", "beta", None),
                make_result("c", "gamma", None),
            ];

            let reranked = reranker.rerank("test query", results, 3).unwrap();
            assert_eq!(reranked.len(), 3);
            // Should be reordered by score descending: b(8), a(3), c(1)
            assert_eq!(reranked[0].id, "b");
            assert_eq!(reranked[0].score, Some(8.0));
            assert_eq!(reranked[1].id, "a");
            assert_eq!(reranked[1].score, Some(3.0));
            assert_eq!(reranked[2].id, "c");
            assert_eq!(reranked[2].score, Some(1.0));
        }
    }

    #[cfg(feature = "onnx-rerank")]
    mod onnx_tests {
        use super::*;

        #[test]
        fn test_onnx_reranker_construction() {
            // The model may not be downloaded in CI, so we test that construction
            // doesn't panic and the struct is usable regardless.
            let reranker = OnnxReranker::new("BAAI/bge-reranker-base", 10);
            // is_available may be true or false depending on environment
            let _ = reranker.is_available();
        }

        #[test]
        fn test_onnx_reranker_empty_input() {
            let reranker = OnnxReranker::new("nonexistent-model", 10);
            let results: Vec<MemoryResult> = vec![];
            let reranked = reranker.rerank("query", results, 10).unwrap();
            assert!(reranked.is_empty());
        }

        #[test]
        fn test_onnx_reranker_fallback_on_unavailable_model() {
            // Use a bogus model name so initialization will fail and fallback to NoOp
            let reranker = OnnxReranker::new("nonexistent-model-xyz", 10);
            assert!(!reranker.is_available());

            let results = vec![
                make_result("a", "alpha", Some(0.5)),
                make_result("b", "beta", Some(0.3)),
                make_result("c", "gamma", Some(0.9)),
            ];
            let reranked = reranker.rerank("query", results, 10).unwrap();
            // Fallback preserves original order (NoOp behavior)
            assert_eq!(reranked.len(), 3);
            assert_eq!(reranked[0].id, "a");
            assert_eq!(reranked[1].id, "b");
            assert_eq!(reranked[2].id, "c");
        }

        #[test]
        fn test_onnx_reranker_fallback_respects_top_k() {
            let reranker = OnnxReranker::new("nonexistent-model-xyz", 10);
            let results = vec![
                make_result("a", "alpha", Some(0.5)),
                make_result("b", "beta", Some(0.3)),
                make_result("c", "gamma", Some(0.9)),
            ];
            let reranked = reranker.rerank("query", results, 2).unwrap();
            assert_eq!(reranked.len(), 2);
        }
    }
}

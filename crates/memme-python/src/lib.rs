//! Python bindings for MemMe — the edge-first AI memory engine.

mod convert;

use std::sync::{Arc, OnceLock};

use memme_embeddings::Embedder;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use convert::{
    episode_to_dict, event_to_dict, graph_result_to_dict, history_record_to_dict,
    memory_result_to_dict,
};

/// Shared tokio runtime — still needed for embeddings (reqwest-based).
fn shared_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime")
    })
}

/// Python wrapper for the MemMe memory store.
///
/// MemoryStore is Send+Sync via internal ConnectionPool.
/// Python's GIL ensures single-threaded access anyway.
///
/// LLM provider is managed internally by the Rust MemoryStore and persisted
/// in the SQLite database — no separate Python-side caching needed.
#[pyclass]
struct MemoryStore {
    inner: memme_core::memory::MemoryStore,
}

#[pymethods]
impl MemoryStore {
    // =====================================================================
    // Constructor
    // =====================================================================

    /// Create a new MemoryStore.
    ///
    /// Args:
    ///     db_path: Path to SQLite file, or ":memory:" for in-memory.
    ///     embedder: "onnx" (default, local), "openai", or "mock" (testing).
    ///     api_key: API key for OpenAI embedder.
    ///     base_url: Custom API base URL (OpenAI-compatible).
    ///     embed_model: Embedding model name.
    ///     dims: Embedding dimensions (auto-detected if not specified).
    ///     llm_provider: LLM provider ("openai" (default), "anthropic", "gemini", "ollama").
    ///     llm_api_key: API key for the LLM provider (persisted in DB).
    ///     llm_model: LLM model name (default depends on provider).
    ///     llm_base_url: Custom LLM API base URL.
    #[new]
    #[pyo3(signature = (db_path=":memory:", *, embedder="onnx", api_key=None, base_url=None, embed_model=None, dims=None, llm_provider="openai", llm_api_key=None, llm_model=None, llm_base_url=None, llm_max_tokens=None, llm_temperature=None, enable_forgetting_curve=None, rrf_vector_weight=None, rrf_fts_weight=None, rrf_entity_weight=None, rrf_k=None, rrf_temporal_weight=None, rrf_event_weight=None, event_memory_threshold=None, rerank_api_key=None, rerank_base_url=None, rerank_model=None))]
    fn new(
        db_path: &str,
        embedder: &str,
        api_key: Option<&str>,
        base_url: Option<&str>,
        embed_model: Option<&str>,
        dims: Option<usize>,
        llm_provider: &str,
        llm_api_key: Option<&str>,
        llm_model: Option<&str>,
        llm_base_url: Option<&str>,
        llm_max_tokens: Option<usize>,
        llm_temperature: Option<f32>,
        enable_forgetting_curve: Option<bool>,
        rrf_vector_weight: Option<f64>,
        rrf_fts_weight: Option<f64>,
        rrf_entity_weight: Option<f64>,
        rrf_k: Option<usize>,
        rrf_temporal_weight: Option<f64>,
        rrf_event_weight: Option<f64>,
        event_memory_threshold: Option<usize>,
        rerank_api_key: Option<&str>,
        rerank_base_url: Option<&str>,
        rerank_model: Option<&str>,
    ) -> PyResult<Self> {
        let _rt_guard = shared_runtime().enter();

        let (emb, detected_dims): (Arc<dyn Embedder>, usize) = match embedder {
            "onnx" => {
                let e = memme_embeddings::onnx::OnnxEmbedder::new()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let d = e.dimensions();
                (Arc::new(e), d)
            }
            "openai" => {
                let key = api_key.ok_or_else(|| {
                    PyRuntimeError::new_err("api_key required for openai embedder")
                })?;
                let url = base_url.ok_or_else(|| {
                    PyRuntimeError::new_err("base_url required for openai embedder (full endpoint URL, e.g. https://api.openai.com/v1/embeddings)")
                })?;
                let mut e =
                    memme_embeddings::openai::OpenAiEmbedder::new(key, url).with_batch_size(10);
                if let Some(model) = embed_model {
                    let d = dims.unwrap_or(1536);
                    e = e.with_model(memme_embeddings::openai::OpenAiModel::Custom {
                        name: model.to_string(),
                        dims: d,
                        send_dims: false,
                    });
                    (Arc::new(e) as Arc<dyn Embedder>, d)
                } else {
                    let d = e.dimensions();
                    (Arc::new(e) as Arc<dyn Embedder>, d)
                }
            }
            "mock" => {
                let d = dims.unwrap_or(384);
                (Arc::new(memme_embeddings::mock::MockEmbedder::new(d)), d)
            }
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown embedder: {embedder}. Use 'onnx', 'openai', or 'mock'."
                )));
            }
        };

        let final_dims = dims.unwrap_or(detected_dims);
        let mut config = memme_core::config::MemoryConfig::new(db_path, final_dims);
        config.enable_graph = true;
        if let Some(mt) = llm_max_tokens {
            config.llm_max_tokens = mt;
        }
        if let Some(t) = llm_temperature {
            config.llm_temperature = if t < 0.0 { None } else { Some(t) };
        }
        if let Some(fc) = enable_forgetting_curve {
            config.enable_forgetting_curve = fc;
        }
        if let Some(w) = rrf_vector_weight {
            config.rrf_vector_weight = w;
        }
        if let Some(w) = rrf_fts_weight {
            config.rrf_fts_weight = w;
        }
        if let Some(w) = rrf_entity_weight {
            config.rrf_entity_weight = w;
        }
        if let Some(k) = rrf_k {
            config.rrf_k = k;
        }
        if let Some(w) = rrf_temporal_weight {
            config.rrf_temporal_weight = w;
        }
        if let Some(w) = rrf_event_weight {
            config.rrf_event_weight = w;
        }
        if let Some(t) = event_memory_threshold {
            config.event_memory_threshold = t;
        }
        if rerank_api_key.is_some() {
            config.enable_rerank = true;
        }

        let mut store = memme_core::memory::MemoryStore::new(config, emb)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Configure LLM if provided
        if let Some(key) = llm_api_key {
            let llm: Arc<dyn memme_llm::LlmProvider> = match llm_provider {
                "anthropic" => {
                    let mut cfg = memme_llm::anthropic::AnthropicConfig::new(key);
                    if let Some(m) = llm_model {
                        cfg = cfg.with_model(m);
                    }
                    if let Some(u) = llm_base_url {
                        cfg = cfg.with_base_url(u);
                    }
                    Arc::new(memme_llm::anthropic::AnthropicProvider::new(cfg))
                }
                "gemini" => {
                    let mut cfg = memme_llm::gemini::GeminiConfig::new(key);
                    if let Some(m) = llm_model {
                        cfg = cfg.with_model(m);
                    }
                    if let Some(u) = llm_base_url {
                        cfg = cfg.with_base_url(u);
                    }
                    Arc::new(memme_llm::gemini::GeminiProvider::new(cfg))
                }
                "ollama" => {
                    let mut cfg = memme_llm::ollama::OllamaConfig::default();
                    if let Some(m) = llm_model {
                        cfg.model = m.to_string();
                    }
                    if let Some(u) = llm_base_url {
                        cfg.host = u.to_string();
                    }
                    Arc::new(memme_llm::ollama::OllamaProvider::new(cfg))
                }
                "openai" | _ => {
                    let model = llm_model.unwrap_or("gpt-4o-mini");
                    let url = llm_base_url.ok_or_else(|| {
                        PyRuntimeError::new_err("llm_base_url required (full endpoint URL, e.g. https://api.openai.com/v1/chat/completions)")
                    })?;
                    Arc::new(memme_llm::openai::OpenAIProvider::new(
                        memme_llm::openai::OpenAIConfig {
                            api_key: key.to_string(),
                            base_url: url.to_string(),
                            model: model.to_string(),
                        },
                    ))
                }
            };
            store.set_llm_provider(llm);
            store
                .save_llm_config(
                    key,
                    llm_model.unwrap_or("default"),
                    llm_base_url.unwrap_or(""),
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }

        // Configure reranker if provided
        if let Some(key) = rerank_api_key {
            let mut rc = memme_core::rerank::RerankConfig::new(key);
            if let Some(url) = rerank_base_url {
                rc.base_url = url.to_string();
            }
            if let Some(model) = rerank_model {
                rc.model = model.to_string();
            }
            store.set_reranker(Arc::new(memme_core::rerank::ApiReranker::new(rc)));
        }

        Ok(Self { inner: store })
    }

    // =====================================================================
    // CRUD
    // =====================================================================

    /// Add a memory (vector dedup, no LLM).
    #[pyo3(signature = (content, *, user_id, agent_id=None, run_id=None, metadata=None))]
    fn add(
        &self,
        py: Python<'_>,
        content: &str,
        user_id: &str,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        metadata: Option<&str>,
    ) -> PyResult<PyObject> {
        let mut opts = memme_core::types::AddOptions::new(user_id);
        if let Some(aid) = agent_id {
            opts = opts.agent_id(aid);
        }
        if let Some(rid) = run_id {
            opts = opts.run_id(rid);
        }
        if let Some(meta_str) = metadata {
            let val: serde_json::Value = serde_json::from_str(meta_str)
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid metadata JSON: {e}")))?;
            opts = opts.metadata(val);
        }
        let content = content.to_string();
        let result = py
            .allow_threads(|| self.inner.add(&content, opts).map_err(|e| e.to_string()))
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        memory_result_to_dict(py, &result)
    }

    /// Get a memory by ID.
    fn get(&self, py: Python<'_>, id: &str) -> PyResult<Option<PyObject>> {
        let id = id.to_string();
        let result = py
            .allow_threads(|| self.inner.get_trace(&id).map_err(|e| e.to_string()))
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        result.map(|r| memory_result_to_dict(py, &r)).transpose()
    }

    /// Update a memory's content.
    fn update(&self, py: Python<'_>, id: &str, content: &str) -> PyResult<PyObject> {
        let id = id.to_string();
        let content = content.to_string();
        let result = py
            .allow_threads(|| {
                self.inner
                    .update_trace(&id, &content, None)
                    .map_err(|e| e.to_string())
            })
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        memory_result_to_dict(py, &result)
    }

    /// Delete a memory by ID.
    fn delete(&self, py: Python<'_>, id: &str) -> PyResult<()> {
        let id = id.to_string();
        py.allow_threads(|| self.inner.delete_trace(&id).map_err(|e| e.to_string()))
            .map_err(|e: String| PyRuntimeError::new_err(e))
    }

    /// List memories with optional filters.
    #[pyo3(signature = (*, user_id, agent_id=None, run_id=None, limit=10))]
    fn list(
        &self,
        py: Python<'_>,
        user_id: &str,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let mut opts = memme_core::types::ListOptions::new(user_id).limit(limit);
        if let Some(aid) = agent_id {
            opts = opts.agent_id(aid);
        }
        if let Some(rid) = run_id {
            opts = opts.run_id(rid);
        }
        let results = py
            .allow_threads(|| self.inner.list_traces(opts).map_err(|e| e.to_string()))
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        results
            .iter()
            .map(|r| memory_result_to_dict(py, r))
            .collect()
    }

    /// Delete all memories matching filters.
    #[pyo3(signature = (*, user_id, agent_id=None, run_id=None))]
    fn delete_all(
        &self,
        py: Python<'_>,
        user_id: &str,
        agent_id: Option<&str>,
        run_id: Option<&str>,
    ) -> PyResult<u64> {
        let user_id = user_id.to_string();
        let agent_id = agent_id.map(|s| s.to_string());
        let run_id = run_id.map(|s| s.to_string());
        py.allow_threads(|| {
            self.inner
                .delete_all_traces(&user_id, agent_id.as_deref(), run_id.as_deref(), None)
                .map_err(|e| e.to_string())
        })
        .map_err(|e: String| PyRuntimeError::new_err(e))
    }

    /// Get change history for a memory.
    fn history(&self, py: Python<'_>, memory_id: &str) -> PyResult<Vec<PyObject>> {
        let memory_id = memory_id.to_string();
        let records = py
            .allow_threads(|| {
                self.inner
                    .trace_history(&memory_id)
                    .map_err(|e| e.to_string())
            })
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        records
            .iter()
            .map(|r| history_record_to_dict(py, r))
            .collect()
    }

    /// Reset the entire store — delete ALL data.
    fn reset(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.reset().map_err(|e| e.to_string()))
            .map_err(|e: String| PyRuntimeError::new_err(e))
    }

    // =====================================================================
    // Search
    // =====================================================================

    /// Search memories by semantic similarity.
    #[pyo3(signature = (query, *, user_id, agent_id=None, run_id=None, limit=10, threshold=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        user_id: &str,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        limit: usize,
        threshold: Option<f32>,
    ) -> PyResult<Vec<PyObject>> {
        let mut opts = memme_core::types::SearchOptions::new(user_id)
            .limit(limit)
            .keyword_search(true);
        if let Some(aid) = agent_id {
            opts = opts.agent_id(aid);
        }
        if let Some(rid) = run_id {
            opts = opts.run_id(rid);
        }
        if let Some(t) = threshold {
            opts = opts.threshold(t);
        }
        let query = query.to_string();
        let results = py
            .allow_threads(|| self.inner.search(&query, opts).map_err(|e| e.to_string()))
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        results
            .iter()
            .map(|r| memory_result_to_dict(py, r))
            .collect()
    }

    /// Hybrid search (vector + FTS with RRF fusion). Delegates to search().
    ///
    /// RRF weights are configured at store construction time via
    /// `rrf_vector_weight` / `rrf_fts_weight` constructor parameters.
    #[pyo3(signature = (query, *, user_id, agent_id=None, run_id=None, limit=10))]
    fn hybrid_search(
        &self,
        py: Python<'_>,
        query: &str,
        user_id: &str,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        self.search(py, query, user_id, agent_id, run_id, limit, None)
    }

    /// Rebuild the FTS index.
    fn rebuild_fts_index(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.rebuild_fts_index().map_err(|e| e.to_string()))
            .map_err(|e: String| PyRuntimeError::new_err(e))
    }

    // =====================================================================
    // Smart (LLM-powered)
    // =====================================================================

    /// Configure the LLM provider (persisted in database).
    #[pyo3(signature = (api_key, base_url, *, model="gpt-4o-mini"))]
    fn set_llm(&self, py: Python<'_>, api_key: &str, base_url: &str, model: &str) -> PyResult<()> {
        let (api_key, model, base_url) =
            (api_key.to_string(), model.to_string(), base_url.to_string());
        let llm = Arc::new(memme_llm::openai::OpenAIProvider::new(
            memme_llm::openai::OpenAIConfig {
                api_key: api_key.clone(),
                base_url: base_url.clone(),
                model: model.clone(),
            },
        ));
        self.inner.set_llm_provider(llm);
        py.allow_threads(|| {
            self.inner
                .save_llm_config(&api_key, &model, &base_url)
                .map_err(|e| e.to_string())
        })
        .map_err(|e: String| PyRuntimeError::new_err(e))
    }

    /// Run diagnostic checks on storage, embedder, and LLM (if configured).
    /// Returns a dict with check results.
    fn diagnose(&self, py: Python<'_>) -> PyResult<PyObject> {
        let report = py.allow_threads(|| self.inner.diagnose());
        let json_str =
            serde_json::to_string(&report).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        let result = json_mod.call_method1("loads", (json_str,))?;
        Ok(result.into())
    }

    // =====================================================================
    // Pipeline: append_events → compact → meditate
    // =====================================================================

    /// Append chat messages as events to a session.
    #[pyo3(signature = (messages, *, session_id, user_id, metadata=None))]
    fn append_events(
        &self,
        py: Python<'_>,
        messages: Vec<(String, String)>,
        session_id: &str,
        user_id: &str,
        metadata: Option<&str>,
    ) -> PyResult<PyObject> {
        let meta = parse_metadata(metadata)?;
        let chat_messages: Vec<memme_core::types::ChatMessage> = messages
            .into_iter()
            .map(|(role, content)| memme_core::types::ChatMessage {
                role,
                content,
                image_url: None,
                image_type: None,
                timestamp: None,
            })
            .collect();
        let (session_id, user_id) = (session_id.to_string(), user_id.to_string());
        let result = py
            .allow_threads(|| {
                self.inner
                    .append_events(&session_id, &chat_messages, &user_id, meta)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("session_id", &result.session_id)?;
        dict.set_item("events_appended", result.events_appended)?;
        dict.set_item("total_unprocessed", result.total_unprocessed)?;
        dict.set_item("compact_needed", result.compact_needed)?;
        Ok(dict.into())
    }

    /// Compact a session: purify events, create episode narrative.
    #[pyo3(signature = (session_id))]
    fn compact(&self, py: Python<'_>, session_id: &str) -> PyResult<PyObject> {
        let session_id = session_id.to_string();
        let result = py
            .allow_threads(|| self.inner.compact(&session_id))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("session_id", &result.session_id)?;
        dict.set_item("episode_id", &result.episode_id)?;
        dict.set_item("events_processed", result.events_processed)?;
        Ok(dict.into())
    }

    /// Meditate: extract facts from episodes, reconcile with existing memories,
    /// build entity graph. Requires LLM to be configured.
    #[pyo3(signature = (*, user_id, triggered_by="python"))]
    fn meditate(&self, py: Python<'_>, user_id: &str, triggered_by: &str) -> PyResult<PyObject> {
        let opts = memme_core::types::MeditateOptions {
            user_id: user_id.to_string(),
            triggered_by: triggered_by.to_string(),
            since: None,
        };
        let record = py
            .allow_threads(|| self.inner.meditate(opts))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let json_str =
            serde_json::to_string(&record).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        let result = json_mod.call_method1("loads", (json_str,))?;
        Ok(result.into())
    }

    // =====================================================================
    // Episode & Session
    // =====================================================================

    /// List episodes for a user.
    #[pyo3(signature = (*, user_id, limit=20, offset=0, since=None, until=None))]
    fn list_episodes(
        &self,
        py: Python<'_>,
        user_id: &str,
        limit: usize,
        offset: usize,
        since: Option<&str>,
        until: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let (user_id, since, until) = (
            user_id.to_string(),
            since.map(|s| s.to_string()),
            until.map(|s| s.to_string()),
        );
        let episodes = py
            .allow_threads(move || {
                let mut opts = memme_core::types::ListEpisodesOptions::new(&user_id)
                    .limit(limit)
                    .offset(offset);
                if let Some(s) = since {
                    opts = opts.since(s);
                }
                if let Some(u) = until {
                    opts = opts.until(u);
                }
                self.inner.list_episodes(opts).map_err(|e| e.to_string())
            })
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        episodes.iter().map(|ep| episode_to_dict(py, ep)).collect()
    }

    /// Get an episode by ID.
    fn get_episode(&self, py: Python<'_>, episode_id: &str) -> PyResult<Option<PyObject>> {
        let episode_id = episode_id.to_string();
        let result = py
            .allow_threads(|| {
                self.inner
                    .get_episode(&episode_id)
                    .map_err(|e| e.to_string())
            })
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        result.map(|ep| episode_to_dict(py, &ep)).transpose()
    }

    /// Get messages within an episode.
    #[pyo3(signature = (episode_id, *, limit=50, offset=0))]
    fn get_episode_messages(
        &self,
        py: Python<'_>,
        episode_id: &str,
        limit: usize,
        offset: usize,
    ) -> PyResult<Vec<PyObject>> {
        let episode_id = episode_id.to_string();
        let events = py
            .allow_threads(move || {
                let opts = memme_core::types::EpisodeMessagesOptions::new()
                    .limit(limit)
                    .offset(offset);
                self.inner
                    .get_episode_messages(&episode_id, opts)
                    .map_err(|e| e.to_string())
            })
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        events.iter().map(|ev| event_to_dict(py, ev)).collect()
    }

    /// Search episodes by semantic similarity.
    #[pyo3(signature = (query, *, user_id, limit=10))]
    fn search_episodes(
        &self,
        py: Python<'_>,
        query: &str,
        user_id: &str,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let (query, user_id) = (query.to_string(), user_id.to_string());
        let episodes = py
            .allow_threads(move || {
                let opts = memme_core::types::SearchEpisodesOptions::new(&user_id).limit(limit);
                self.inner
                    .search_episodes(&query, opts)
                    .map_err(|e| e.to_string())
            })
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        episodes.iter().map(|ep| episode_to_dict(py, ep)).collect()
    }

    /// Search messages within an episode.
    #[pyo3(signature = (episode_id, query, *, limit=10))]
    fn search_episode_messages(
        &self,
        py: Python<'_>,
        episode_id: &str,
        query: &str,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let (episode_id, query) = (episode_id.to_string(), query.to_string());
        let events = py
            .allow_threads(move || {
                self.inner
                    .search_episode_messages(&episode_id, &query, limit)
                    .map_err(|e| e.to_string())
            })
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        events.iter().map(|ev| event_to_dict(py, ev)).collect()
    }

    /// Delete an episode by ID.
    fn delete_episode(&self, py: Python<'_>, episode_id: &str) -> PyResult<()> {
        let episode_id = episode_id.to_string();
        py.allow_threads(|| {
            self.inner
                .delete_episode(&episode_id)
                .map_err(|e| e.to_string())
        })
        .map_err(|e: String| PyRuntimeError::new_err(e))
    }

    /// Get session context within a token budget.
    #[pyo3(signature = (session_id, *, token_budget=2000, include_summary=true, max_events=100))]
    fn get_session_context(
        &self,
        py: Python<'_>,
        session_id: &str,
        token_budget: usize,
        include_summary: bool,
        max_events: usize,
    ) -> PyResult<PyObject> {
        let session_id = session_id.to_string();
        let ctx = py
            .allow_threads(|| {
                let opts = memme_core::types::GetSessionContextOptions::new()
                    .token_budget(token_budget)
                    .include_summary(include_summary)
                    .max_events(max_events);
                self.inner
                    .get_session_context(&session_id, opts)
                    .map_err(|e| e.to_string())
            })
            .map_err(|e: String| PyRuntimeError::new_err(e))?;

        let dict = PyDict::new(py);
        dict.set_item("session_id", &ctx.session_id)?;
        dict.set_item("tokens_used", ctx.tokens_used)?;
        dict.set_item("token_budget", ctx.token_budget)?;
        dict.set_item("episode_summary", &ctx.episode_summary)?;
        let events: Vec<PyObject> = ctx
            .events
            .iter()
            .map(|ev| event_to_dict(py, ev))
            .collect::<PyResult<Vec<_>>>()?;
        dict.set_item("events", events)?;
        Ok(dict.into())
    }

    // =====================================================================
    // Analytics
    // =====================================================================

    /// Get summary statistics for a user.
    fn user_stats(&self, py: Python<'_>, user_id: &str) -> PyResult<PyObject> {
        let user_id = user_id.to_string();
        let stats = py
            .allow_threads(|| self.inner.user_stats(&user_id).map_err(|e| e.to_string()))
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        let dict = PyDict::new(py);
        dict.set_item("user_id", &stats.user_id)?;
        dict.set_item("total_memories", stats.total_memories)?;
        dict.set_item("total_entities", stats.total_entities)?;
        dict.set_item("total_relationships", stats.total_relationships)?;
        dict.set_item("earliest_memory", &stats.earliest_memory)?;
        dict.set_item("latest_memory", &stats.latest_memory)?;
        dict.set_item("unique_agents", stats.unique_agents)?;
        Ok(dict.into())
    }

    /// Get memory creation frequency by time period.
    #[pyo3(signature = (user_id, granularity, limit=30))]
    fn memory_frequency(
        &self,
        py: Python<'_>,
        user_id: &str,
        granularity: &str,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let (user_id, granularity) = (user_id.to_string(), granularity.to_string());
        let buckets = py
            .allow_threads(|| {
                self.inner
                    .memory_frequency(&user_id, &granularity, limit)
                    .map_err(|e| e.to_string())
            })
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        buckets
            .iter()
            .map(|b| {
                let dict = PyDict::new(py);
                dict.set_item("period", &b.period)?;
                dict.set_item("count", b.count)?;
                Ok(dict.into())
            })
            .collect()
    }

    /// Get top entities by relationship count.
    #[pyo3(signature = (user_id, limit=10))]
    fn top_entities(&self, py: Python<'_>, user_id: &str, limit: usize) -> PyResult<Vec<PyObject>> {
        let user_id = user_id.to_string();
        let entities = py
            .allow_threads(|| {
                self.inner
                    .top_entities(&user_id, limit)
                    .map_err(|e| e.to_string())
            })
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        entities
            .iter()
            .map(|e| {
                let dict = PyDict::new(py);
                dict.set_item("name", &e.name)?;
                dict.set_item("entity_type", &e.entity_type)?;
                dict.set_item("relationship_count", e.relationship_count)?;
                Ok(dict.into())
            })
            .collect()
    }

    // =====================================================================
    // Knowledge Graph
    // =====================================================================

    /// Extract entities/relationships from text using LLM.
    #[pyo3(signature = (text, *, user_id))]
    fn add_graph(&self, py: Python<'_>, text: &str, user_id: &str) -> PyResult<PyObject> {
        let (text, user_id) = (text.to_string(), user_id.to_string());
        let result = py
            .allow_threads(|| {
                self.inner
                    .add_graph_auto(&text, &user_id)
                    .map_err(|e| e.to_string())
            })
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        graph_result_to_dict(py, &result)
    }

    /// Search the knowledge graph.
    #[pyo3(signature = (query, *, user_id, depth=2))]
    fn search_graph(
        &self,
        py: Python<'_>,
        query: &str,
        user_id: &str,
        depth: usize,
    ) -> PyResult<PyObject> {
        let (query, user_id) = (query.to_string(), user_id.to_string());
        let result = py
            .allow_threads(|| {
                self.inner
                    .search_graph(&query, &user_id, depth)
                    .map_err(|e| e.to_string())
            })
            .map_err(|e: String| PyRuntimeError::new_err(e))?;
        graph_result_to_dict(py, &result)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_metadata(metadata: Option<&str>) -> PyResult<Option<serde_json::Value>> {
    metadata
        .map(|s| serde_json::from_str(s))
        .transpose()
        .map_err(|e| PyRuntimeError::new_err(format!("Invalid metadata JSON: {e}")))
}

/// MemMe: Edge-first AI memory engine powered by SQLite.
#[pymodule]
fn memme(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MemoryStore>()?;
    Ok(())
}

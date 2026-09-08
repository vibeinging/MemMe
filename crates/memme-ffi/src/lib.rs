//! UniFFI bindings for MemMe — exposes the memory engine to Swift, Kotlin, etc.
//!
//! This crate does NOT embed an HTTP client. Instead it exposes an `HttpClient`
//! callback interface that the host app implements (e.g. URLSession on iOS,
//! OkHttp on Android). This keeps reqwest/tokio out of the FFI binary.
//!
//! Mobile store construction is intentionally blocked until VexDB-Lite is
//! statically registered against the same SQLite instance used by MemMe.

uniffi::setup_scaffolding!();

use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// HttpClient callback interface
// ---------------------------------------------------------------------------

/// HTTP transport callback — implemented by the host app.
///
/// The host provides this via `newWithHttpClient`. All LLM and embedding
/// API calls are routed through this single callback.
#[uniffi::export(callback_interface)]
pub trait HttpClient: Send + Sync {
    /// Perform a synchronous HTTP POST.
    ///
    /// - `url`: full URL (e.g. `https://api.openai.com/v1/chat/completions`)
    /// - `headers`: alternating key/value pairs: `["Authorization", "Bearer sk-...", "Content-Type", "application/json"]`
    /// - `body`: JSON request body as a string.
    ///
    /// Returns the response body as a string, or throws on network/HTTP error.
    fn post(&self, url: String, headers: Vec<String>, body: String) -> Result<String, MemmeError>;
}

// ---------------------------------------------------------------------------
// FfiLlmProvider — implements LlmProvider via HttpClient callback
// ---------------------------------------------------------------------------

/// LLM provider that delegates HTTP to the host-app `HttpClient`.
struct FfiLlmProvider {
    http_client: Arc<dyn HttpClient>,
    api_key: String,
    model: String,
    base_url: String,
}

impl memme_llm::LlmProvider for FfiLlmProvider {
    fn generate(
        &self,
        messages: &[memme_llm::Message],
        options: &memme_llm::GenerateOptions,
    ) -> std::result::Result<String, memme_llm::LlmError> {
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": match m.role {
                        memme_llm::MessageRole::System => "system",
                        memme_llm::MessageRole::User => "user",
                        memme_llm::MessageRole::Assistant => "assistant",
                    },
                    "content": m.content,
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": msgs,
        });
        if let Some(t) = options.temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(max) = options.max_tokens {
            body["max_tokens"] = serde_json::json!(max);
        }
        if let Some(ref fmt) = options.response_format {
            match fmt {
                memme_llm::ResponseFormat::Json => {
                    body["response_format"] = serde_json::json!({"type": "json_object"});
                }
                memme_llm::ResponseFormat::Text => {}
            }
        }

        let url = self.base_url.clone();
        let headers = vec![
            "Content-Type".to_string(),
            "application/json".to_string(),
            "Authorization".to_string(),
            format!("Bearer {}", self.api_key),
        ];

        let resp_str = self
            .http_client
            .post(url, headers, body.to_string())
            .map_err(|e| memme_llm::LlmError::RequestFailed(e.to_string()))?;

        let resp: serde_json::Value = serde_json::from_str(&resp_str)
            .map_err(|e| memme_llm::LlmError::ParseError(format!("Invalid JSON response: {e}")))?;

        // Check for API error
        if let Some(err) = resp.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown API error");
            return Err(memme_llm::LlmError::RequestFailed(msg.to_string()));
        }

        resp["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| {
                memme_llm::LlmError::ParseError("Missing choices[0].message.content".to_string())
            })
    }

    fn name(&self) -> &str {
        "ffi-openai"
    }
}

// ---------------------------------------------------------------------------
// FfiEmbedder — implements Embedder via HttpClient callback
// ---------------------------------------------------------------------------

/// Embedder that delegates HTTP to the host-app `HttpClient`.
struct FfiEmbedder {
    http_client: Arc<dyn HttpClient>,
    api_key: String,
    model: String,
    dims: usize,
    base_url: String,
}

impl memme_embeddings::Embedder for FfiEmbedder {
    fn embed(&self, text: &str) -> std::result::Result<Vec<f32>, memme_embeddings::EmbedError> {
        let body = serde_json::json!({
            "model": self.model,
            "input": text,
        });

        let url = self.base_url.clone();
        let headers = vec![
            "Content-Type".to_string(),
            "application/json".to_string(),
            "Authorization".to_string(),
            format!("Bearer {}", self.api_key),
        ];

        let resp_str = self
            .http_client
            .post(url, headers, body.to_string())
            .map_err(|e| memme_embeddings::EmbedError::ApiError(e.to_string()))?;

        let resp: serde_json::Value = serde_json::from_str(&resp_str)
            .map_err(|e| memme_embeddings::EmbedError::ApiError(format!("Invalid JSON: {e}")))?;

        if let Some(err) = resp.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown API error");
            return Err(memme_embeddings::EmbedError::ApiError(msg.to_string()));
        }

        let embedding = &resp["data"][0]["embedding"];
        embedding
            .as_array()
            .ok_or_else(|| {
                memme_embeddings::EmbedError::ApiError(
                    "Missing data[0].embedding array".to_string(),
                )
            })?
            .iter()
            .map(|v| {
                v.as_f64().map(|f| f as f32).ok_or_else(|| {
                    memme_embeddings::EmbedError::ApiError(
                        "Non-numeric embedding value".to_string(),
                    )
                })
            })
            .collect()
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ---------------------------------------------------------------------------
// FFI-safe data types
// ---------------------------------------------------------------------------

#[derive(uniffi::Record)]
pub struct MemoryResult {
    pub id: String,
    pub content: String,
    pub user_id: String,
    pub agent_id: Option<String>,
    pub app_id: Option<String>,
    pub run_id: Option<String>,
    pub score: Option<f32>,
    pub created_at: String,
    pub updated_at: String,
    pub metadata: Option<String>,
    pub importance: Option<f32>,
    pub access_count: Option<u32>,
    pub immutable: bool,
    pub expiration_date: Option<String>,
    pub categories: Option<Vec<String>>,
    pub retention: Option<f32>,
    pub stability: Option<f32>,
    pub event_time: Option<String>,
}

#[derive(uniffi::Record)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: Option<String>,
    pub user_id: String,
}

#[derive(uniffi::Record)]
pub struct DiagnoseCheck {
    pub name: String,
    pub ok: bool,
    pub latency_ms: u64,
    pub detail: String,
}

#[derive(uniffi::Record)]
pub struct DiagnoseReport {
    pub all_ok: bool,
    pub checks: Vec<DiagnoseCheck>,
}

#[derive(uniffi::Record)]
pub struct FfiBackupInfo {
    pub source_path: String,
    pub backup_path: String,
    pub size_bytes: u64,
    pub created_at: String,
    pub memory_count: u64,
    pub schema_version: String,
}

impl From<memme_core::types::BackupInfo> for FfiBackupInfo {
    fn from(info: memme_core::types::BackupInfo) -> Self {
        Self {
            source_path: info.source_path,
            backup_path: info.backup_path,
            size_bytes: info.size_bytes,
            created_at: info.created_at,
            memory_count: info.memory_count,
            schema_version: info.schema_version,
        }
    }
}

#[derive(uniffi::Record)]
pub struct GraphRelation {
    pub id: String,
    pub source: String,
    pub source_id: String,
    pub target: String,
    pub target_id: String,
    pub relation_type: String,
    pub user_id: String,
    pub description: Option<String>,
}

#[derive(uniffi::Record)]
pub struct HistoryRecord {
    pub id: String,
    pub memory_id: String,
    pub old_memory: Option<String>,
    pub new_memory: String,
    pub event: String,
    pub created_at: String,
}

#[derive(uniffi::Record)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// An event in the stream layer.
#[derive(uniffi::Record)]
pub struct Event {
    pub event_id: String,
    pub session_id: Option<String>,
    pub event_type: String,
    pub content: String,
    pub purified_content: Option<String>,
    pub timestamp: String,
    pub event_time: Option<String>,
    pub location: Option<String>,
}

/// Session context for retrieval — purified events within a token budget.
#[derive(uniffi::Record)]
pub struct SessionContext {
    pub session_id: String,
    pub events: Vec<Event>,
    pub tokens_used: u32,
    pub token_budget: u32,
    pub episode_summary: Option<String>,
}

#[derive(uniffi::Record)]
pub struct UserStats {
    pub user_id: String,
    pub total_memories: u64,
    pub total_entities: u64,
    pub total_relationships: u64,
    pub earliest_memory: Option<String>,
    pub latest_memory: Option<String>,
    pub unique_agents: u64,
}

#[derive(uniffi::Record)]
pub struct TimeBucket {
    pub period: String,
    pub count: u64,
}

#[derive(uniffi::Record)]
pub struct EntityStat {
    pub name: String,
    pub entity_type: Option<String>,
    pub relationship_count: u64,
}

#[derive(uniffi::Record)]
pub struct GraphSearchResult {
    pub entities: Vec<Entity>,
    pub relations: Vec<GraphRelation>,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum MemmeError {
    #[error("{msg}")]
    Runtime { msg: String },
}

impl From<memme_core::error::MemoryError> for MemmeError {
    fn from(e: memme_core::error::MemoryError) -> Self {
        MemmeError::Runtime { msg: e.to_string() }
    }
}

fn mobile_vexdb_unavailable() -> MemmeError {
    MemmeError::Runtime {
        msg: "MemMe mobile bindings are not available yet: VexDB-Lite static registration is required"
            .to_string(),
    }
}

// ---------------------------------------------------------------------------
// Main object
// ---------------------------------------------------------------------------

/// The main MemMe memory store, exposed as a UniFFI Object.
///
/// iOS and Android construction currently returns a clear error until
/// VexDB-Lite static registration is integrated into the mobile package.
#[derive(uniffi::Object)]
pub struct MemoryStore {
    inner: Mutex<memme_core::memory::MemoryStore>,
    /// Kept alive so FfiLlmProvider / FfiEmbedder can reference it.
    _http_client: Option<Arc<dyn HttpClient>>,
}

#[uniffi::export]
impl MemoryStore {
    // -- Constructors -------------------------------------------------------

    /// Create a MemoryStore with a mock embedder (useful for testing, no API needed).
    #[uniffi::constructor]
    pub fn new_mock(db_path: String, dims: u32) -> Result<Arc<Self>, MemmeError> {
        if cfg!(any(target_os = "ios", target_os = "android")) {
            return Err(mobile_vexdb_unavailable());
        }
        let embedder = Arc::new(memme_embeddings::mock::MockEmbedder::new(dims as usize));
        let config = memme_core::config::MemoryConfig::new(&db_path, dims as usize);
        let store = memme_core::memory::MemoryStore::new(config, embedder)?;
        Ok(Arc::new(Self {
            inner: Mutex::new(store),
            _http_client: None,
        }))
    }

    /// Create a MemoryStore with an HTTP client callback for OpenAI-compatible APIs.
    ///
    /// The host app implements `HttpClient` (e.g. URLSession on iOS, OkHttp on Android).
    /// Both embedding and LLM calls are routed through this single callback.
    ///
    /// Note: `http_client` is `Box<dyn HttpClient>` — UniFFI callback interfaces
    /// use `Box`, not `Arc`, on the foreign side.
    #[uniffi::constructor]
    pub fn new_with_http_client(
        db_path: String,
        http_client: Box<dyn HttpClient>,
        api_key: String,
        embedding_model: Option<String>,
        embedding_dims: Option<u32>,
        llm_base_url: Option<String>,
    ) -> Result<Arc<Self>, MemmeError> {
        if cfg!(any(target_os = "ios", target_os = "android")) {
            return Err(mobile_vexdb_unavailable());
        }
        let http_arc: Arc<dyn HttpClient> = Arc::from(http_client);
        let base_url = llm_base_url.ok_or(MemmeError::Runtime { msg: "llm_base_url required (full endpoint URL, e.g. https://api.openai.com/v1/chat/completions)".to_string() })?;

        let embed_model = embedding_model.unwrap_or_else(|| "text-embedding-3-small".to_string());
        let dims = embedding_dims.unwrap_or(1536) as usize;

        let embedder = Arc::new(FfiEmbedder {
            http_client: Arc::clone(&http_arc),
            api_key: api_key.clone(),
            model: embed_model,
            dims,
            base_url: base_url.clone(),
        });

        let config = memme_core::config::MemoryConfig::new(&db_path, dims);
        let store = memme_core::memory::MemoryStore::new(
            config,
            embedder as Arc<dyn memme_embeddings::Embedder>,
        )?;

        // Set up default LLM provider using same HTTP client
        let llm: Arc<dyn memme_llm::LlmProvider> = Arc::new(FfiLlmProvider {
            http_client: Arc::clone(&http_arc),
            api_key,
            model: "gpt-4o-mini".to_string(),
            base_url,
        });
        store.set_llm_provider(llm);

        Ok(Arc::new(Self {
            inner: Mutex::new(store),
            _http_client: Some(http_arc),
        }))
    }

    // -- CRUD ---------------------------------------------------------------

    pub fn add(
        &self,
        content: String,
        user_id: String,
        agent_id: Option<String>,
        run_id: Option<String>,
        metadata: Option<String>,
    ) -> Result<MemoryResult, MemmeError> {
        let mut opts = memme_core::types::AddOptions::new(&user_id);
        if let Some(aid) = &agent_id {
            opts = opts.agent_id(aid);
        }
        if let Some(rid) = &run_id {
            opts = opts.run_id(rid);
        }
        if let Some(meta_str) = &metadata {
            let val: serde_json::Value =
                serde_json::from_str(meta_str).map_err(|e| MemmeError::Runtime {
                    msg: format!("Invalid metadata JSON: {e}"),
                })?;
            opts = opts.metadata(val);
        }
        let store = self.lock_store()?;
        let r = store.add(&content, opts)?;
        Ok(convert_result(&r))
    }

    pub fn get(&self, id: String) -> Result<Option<MemoryResult>, MemmeError> {
        let store = self.lock_store()?;
        Ok(store.get_trace(&id)?.map(|r| convert_result(&r)))
    }

    pub fn update(&self, id: String, content: String) -> Result<MemoryResult, MemmeError> {
        let store = self.lock_store()?;
        Ok(convert_result(&store.update_trace(&id, &content, None)?))
    }

    pub fn delete(&self, id: String) -> Result<(), MemmeError> {
        self.lock_store()?.delete_trace(&id)?;
        Ok(())
    }

    pub fn list(
        &self,
        user_id: String,
        agent_id: Option<String>,
        run_id: Option<String>,
        limit: Option<u32>,
    ) -> Result<Vec<MemoryResult>, MemmeError> {
        let mut opts = memme_core::types::ListOptions::new(&user_id);
        if let Some(aid) = &agent_id {
            opts = opts.agent_id(aid);
        }
        if let Some(rid) = &run_id {
            opts = opts.run_id(rid);
        }
        if let Some(l) = limit {
            opts = opts.limit(l as usize);
        }
        let store = self.lock_store()?;
        Ok(store
            .list_traces(opts)?
            .iter()
            .map(convert_result)
            .collect())
    }

    // -- Search -------------------------------------------------------------

    pub fn search(
        &self,
        query: String,
        user_id: String,
        agent_id: Option<String>,
        run_id: Option<String>,
        limit: Option<u32>,
        threshold: Option<f32>,
    ) -> Result<Vec<MemoryResult>, MemmeError> {
        let mut opts = memme_core::types::SearchOptions::new(&user_id);
        if let Some(aid) = &agent_id {
            opts = opts.agent_id(aid);
        }
        if let Some(rid) = &run_id {
            opts = opts.run_id(rid);
        }
        if let Some(l) = limit {
            opts = opts.limit(l as usize);
        }
        if let Some(t) = threshold {
            opts = opts.threshold(t);
        }
        let store = self.lock_store()?;
        Ok(store
            .search(&query, opts)?
            .iter()
            .map(convert_result)
            .collect())
    }

    /// Hybrid search (vector + FTS with RRF fusion).
    ///
    /// RRF weights are configured at store construction time.
    pub fn hybrid_search(
        &self,
        query: String,
        user_id: String,
        agent_id: Option<String>,
        run_id: Option<String>,
        limit: Option<u32>,
    ) -> Result<Vec<MemoryResult>, MemmeError> {
        let mut opts = memme_core::types::SearchOptions::new(&user_id).keyword_search(true);
        if let Some(aid) = &agent_id {
            opts = opts.agent_id(aid);
        }
        if let Some(rid) = &run_id {
            opts = opts.run_id(rid);
        }
        if let Some(l) = limit {
            opts = opts.limit(l as usize);
        }
        let store = self.lock_store()?;
        Ok(store
            .search(&query, opts)?
            .iter()
            .map(convert_result)
            .collect())
    }

    pub fn rebuild_fts_index(&self) -> Result<(), MemmeError> {
        self.lock_store()?.rebuild_fts_index()?;
        Ok(())
    }

    // -- Backup / Restore ---------------------------------------------------

    pub fn backup_to_path(&self, path: String) -> Result<FfiBackupInfo, MemmeError> {
        let store = self.lock_store()?;
        Ok(store.backup_to_path(&path)?.into())
    }

    /// Restore the database from a backup file.
    ///
    /// **Warning**: The caller must re-create the MemoryStore after calling this.
    pub fn restore_from_backup(&self, backup_path: String) -> Result<(), MemmeError> {
        let store = self.lock_store()?;
        let config = store.config().clone();
        drop(store);
        memme_core::MemoryStore::restore_from_backup(&backup_path, &config)?;
        Ok(())
    }

    // -- Diagnostics ---------------------------------------------------------

    pub fn diagnose(&self) -> Result<DiagnoseReport, MemmeError> {
        let store = self.lock_store()?;
        let report = store.diagnose();
        Ok(DiagnoseReport {
            all_ok: report.all_ok,
            checks: report
                .checks
                .into_iter()
                .map(|c| DiagnoseCheck {
                    name: c.name.to_string(),
                    ok: c.ok,
                    latency_ms: c.latency_ms,
                    detail: c.detail,
                })
                .collect(),
        })
    }

    /// Extract knowledge-graph entities and relationships via LLM.
    pub fn add_graph(
        &self,
        text: String,
        user_id: String,
        llm_model: Option<String>,
        llm_base_url: Option<String>,
    ) -> Result<GraphSearchResult, MemmeError> {
        let llm = self.make_llm_provider(llm_model, llm_base_url)?;
        let store = self.lock_store()?;
        Ok(convert_graph_result(
            &store.add_graph(&text, &user_id, llm)?,
        ))
    }

    pub fn search_graph(
        &self,
        query: String,
        user_id: String,
        depth: Option<u32>,
    ) -> Result<GraphSearchResult, MemmeError> {
        let store = self.lock_store()?;
        Ok(convert_graph_result(&store.search_graph(
            &query,
            &user_id,
            depth.unwrap_or(2) as usize,
        )?))
    }

    // -- Bulk / stats -------------------------------------------------------

    pub fn delete_all(
        &self,
        user_id: String,
        agent_id: Option<String>,
        run_id: Option<String>,
    ) -> Result<u64, MemmeError> {
        let store = self.lock_store()?;
        Ok(store.delete_all_traces(&user_id, agent_id.as_deref(), run_id.as_deref(), None)?)
    }

    pub fn history(&self, memory_id: String) -> Result<Vec<HistoryRecord>, MemmeError> {
        let store = self.lock_store()?;
        Ok(store
            .trace_history(&memory_id)?
            .iter()
            .map(convert_history)
            .collect())
    }

    pub fn reset(&self) -> Result<(), MemmeError> {
        self.lock_store()?.reset()?;
        Ok(())
    }

    pub fn user_stats(&self, user_id: String) -> Result<UserStats, MemmeError> {
        let store = self.lock_store()?;
        let s = store.user_stats(&user_id)?;
        Ok(UserStats {
            user_id: s.user_id,
            total_memories: s.total_memories,
            total_entities: s.total_entities,
            total_relationships: s.total_relationships,
            earliest_memory: s.earliest_memory,
            latest_memory: s.latest_memory,
            unique_agents: s.unique_agents,
        })
    }

    pub fn memory_frequency(
        &self,
        user_id: String,
        granularity: String,
        limit: u32,
    ) -> Result<Vec<TimeBucket>, MemmeError> {
        let store = self.lock_store()?;
        Ok(store
            .memory_frequency(&user_id, &granularity, limit as usize)?
            .iter()
            .map(|b| TimeBucket {
                period: b.period.clone(),
                count: b.count,
            })
            .collect())
    }

    pub fn top_entities(&self, user_id: String, limit: u32) -> Result<Vec<EntityStat>, MemmeError> {
        let store = self.lock_store()?;
        Ok(store
            .top_entities(&user_id, limit as usize)?
            .iter()
            .map(|e| EntityStat {
                name: e.name.clone(),
                entity_type: e.entity_type.clone(),
                relationship_count: e.relationship_count,
            })
            .collect())
    }

    /// Get session context for retrieval — purified events within a token budget.
    pub fn get_session_context(
        &self,
        session_id: String,
        token_budget: u32,
    ) -> Result<SessionContext, MemmeError> {
        let store = self.lock_store()?;
        let opts =
            memme_core::types::GetSessionContextOptions::new().token_budget(token_budget as usize);
        let ctx = store.get_session_context(&session_id, opts)?;
        Ok(SessionContext {
            session_id: ctx.session_id,
            events: ctx
                .events
                .iter()
                .map(|e| Event {
                    event_id: e.event_id.clone(),
                    session_id: e.session_id.clone(),
                    timestamp: e.timestamp.clone(),
                    event_type: e.event_type.as_str().to_string(),
                    content: e.content.clone(),
                    purified_content: e.purified_content.clone(),
                    event_time: e.event_time.clone(),
                    location: e.location.clone(),
                })
                .collect(),
            tokens_used: ctx.tokens_used as u32,
            token_budget: ctx.token_budget as u32,
            episode_summary: ctx.episode_summary,
        })
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

impl MemoryStore {
    fn lock_store(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, memme_core::memory::MemoryStore>, MemmeError> {
        self.inner.lock().map_err(|e| MemmeError::Runtime {
            msg: format!("Lock poisoned: {e}"),
        })
    }

    /// Get or create an LLM provider, optionally overriding model/base_url.
    fn make_llm_provider(
        &self,
        model_override: Option<String>,
        base_url_override: Option<String>,
    ) -> Result<Arc<dyn memme_llm::LlmProvider>, MemmeError> {
        // If no overrides, reuse the store's configured LLM
        if model_override.is_none() && base_url_override.is_none() {
            let store = self.lock_store()?;
            if let Some(llm) = store.llm() {
                return Ok(llm);
            }
        }
        Err(MemmeError::Runtime {
            msg: "No LLM provider configured. Create the store with new_with_http_client()."
                .to_string(),
        })
    }
}

fn parse_metadata(metadata: Option<String>) -> Result<Option<serde_json::Value>, MemmeError> {
    metadata
        .map(|s| serde_json::from_str(&s))
        .transpose()
        .map_err(|e| MemmeError::Runtime {
            msg: format!("Invalid metadata JSON: {e}"),
        })
}

fn convert_result(r: &memme_core::types::MemoryResult) -> MemoryResult {
    MemoryResult {
        id: r.id.clone(),
        content: r.content.clone(),
        user_id: r.user_id.clone(),
        agent_id: r.agent_id.clone(),
        app_id: r.app_id.clone(),
        run_id: r.run_id.clone(),
        score: r.score,
        created_at: r.created_at.clone(),
        updated_at: r.updated_at.clone(),
        metadata: r.metadata.as_ref().map(|v| v.to_string()),
        importance: r.importance,
        access_count: r.access_count,
        immutable: r.immutable,
        expiration_date: r.expiration_date.clone(),
        categories: r.categories.clone(),
        retention: r.retention,
        stability: r.stability,
        event_time: r.event_time.clone(),
    }
}

fn convert_history(r: &memme_core::types::HistoryRecord) -> HistoryRecord {
    HistoryRecord {
        id: r.id.clone(),
        memory_id: r.memory_id.clone(),
        old_memory: r.old_memory.clone(),
        new_memory: r.new_memory.clone(),
        event: r.event.clone(),
        created_at: r.created_at.clone(),
    }
}

fn convert_graph_result(r: &memme_core::types::GraphSearchResult) -> GraphSearchResult {
    GraphSearchResult {
        entities: r
            .entities
            .iter()
            .map(|e| Entity {
                id: e.id.clone(),
                name: e.name.clone(),
                entity_type: e.entity_type.clone(),
                user_id: e.user_id.clone(),
            })
            .collect(),
        relations: r
            .relations
            .iter()
            .map(|rel| GraphRelation {
                id: rel.id.clone(),
                source: rel.source.clone(),
                source_id: rel.source_id.clone(),
                target: rel.target.clone(),
                target_id: rel.target_id.clone(),
                relation_type: rel.relation_type.clone(),
                user_id: rel.user_id.clone(),
                description: rel.description.clone(),
            })
            .collect(),
    }
}

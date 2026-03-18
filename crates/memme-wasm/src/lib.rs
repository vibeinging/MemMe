//! WASM bindings for MemMe.
//!
//! **Limitations**: DuckDB's bundled C++ library does not compile to
//! `wasm32-unknown-unknown`. These bindings currently target native
//! WASM runtimes (wasmtime, wasmer) via `wasm32-wasip1`, or serve
//! as a reference for future DuckDB-WASM integration.
//!
//! For browser usage, consider using DuckDB-WASM on the JavaScript side
//! and only delegating memory management logic from this crate.

use std::sync::Mutex;

use wasm_bindgen::prelude::*;

/// WASM wrapper for MemMe memory store.
///
/// Wraps `memme_core::memory::MemoryStore` with a `Mutex` because
/// DuckDB's `Connection` is not `Sync`. JavaScript is single-threaded
/// so the lock is never contended in practice.
#[wasm_bindgen]
pub struct MemoryStore {
    inner: Mutex<memme_core::memory::MemoryStore>,
}

#[wasm_bindgen]
impl MemoryStore {
    /// Create an in-memory store with mock embedder.
    ///
    /// # Arguments
    /// * `dims` - Embedding dimensions (default: 384).
    #[wasm_bindgen(constructor)]
    pub fn new(dims: Option<u32>) -> Result<MemoryStore, JsError> {
        let d = dims.unwrap_or(384) as usize;
        let embedder = std::sync::Arc::new(memme_embeddings::mock::MockEmbedder::new(d));
        let config = memme_core::config::MemoryConfig::new(":memory:", d);
        let store = memme_core::memory::MemoryStore::new(config, embedder)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self {
            inner: Mutex::new(store),
        })
    }

    /// Add a memory.
    ///
    /// # Arguments
    /// * `content` - Text content to store.
    /// * `user_id` - User identifier.
    /// * `metadata` - Optional JSON string of metadata.
    ///
    /// # Returns
    /// A JS object with `id`, `content`, `user_id`, `score`,
    /// `created_at`, `updated_at`, `metadata`.
    pub fn add(
        &self,
        content: &str,
        user_id: &str,
        metadata: Option<String>,
    ) -> Result<JsValue, JsError> {
        let mut opts = memme_core::types::AddOptions::new(user_id);
        if let Some(meta_str) = metadata {
            let val: serde_json::Value = serde_json::from_str(&meta_str)
                .map_err(|e| JsError::new(&format!("Invalid metadata JSON: {e}")))?;
            opts = opts.metadata(val);
        }

        let store = self
            .inner
            .lock()
            .map_err(|e| JsError::new(&e.to_string()))?;
        let result = store
            .add(content, opts)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&to_js_memory(&result))
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Search memories by semantic similarity.
    ///
    /// # Arguments
    /// * `query` - Search query text.
    /// * `user_id` - User identifier.
    /// * `limit` - Maximum number of results (default: 10).
    ///
    /// # Returns
    /// An array of JS objects ordered by similarity.
    pub fn search(
        &self,
        query: &str,
        user_id: &str,
        limit: Option<u32>,
    ) -> Result<JsValue, JsError> {
        let mut opts = memme_core::types::SearchOptions::new(user_id);
        if let Some(l) = limit {
            opts = opts.limit(l as usize);
        }

        let store = self
            .inner
            .lock()
            .map_err(|e| JsError::new(&e.to_string()))?;
        let results = store
            .search(query, opts)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let js_results: Vec<JsMemoryResult> = results.iter().map(to_js_memory).collect();
        serde_wasm_bindgen::to_value(&js_results).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get a memory by ID.
    ///
    /// # Arguments
    /// * `id` - Memory identifier.
    ///
    /// # Returns
    /// A JS object with memory data, or `null` if not found.
    pub fn get(&self, id: &str) -> Result<JsValue, JsError> {
        let store = self
            .inner
            .lock()
            .map_err(|e| JsError::new(&e.to_string()))?;
        let result = store
            .get_trace(id)
            .map_err(|e| JsError::new(&e.to_string()))?;
        match result {
            Some(r) => serde_wasm_bindgen::to_value(&to_js_memory(&r))
                .map_err(|e| JsError::new(&e.to_string())),
            None => Ok(JsValue::NULL),
        }
    }

    /// Update a memory's content (re-embeds automatically).
    ///
    /// # Arguments
    /// * `id` - Memory identifier.
    /// * `content` - New text content.
    ///
    /// # Returns
    /// A JS object with updated memory data.
    pub fn update(&self, id: &str, content: &str) -> Result<JsValue, JsError> {
        let store = self
            .inner
            .lock()
            .map_err(|e| JsError::new(&e.to_string()))?;
        let r = store
            .update_trace(id, content, None)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&to_js_memory(&r)).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Delete a memory by ID.
    ///
    /// # Arguments
    /// * `id` - Memory identifier.
    pub fn delete(&self, id: &str) -> Result<(), JsError> {
        let store = self
            .inner
            .lock()
            .map_err(|e| JsError::new(&e.to_string()))?;
        store
            .delete_trace(id)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// List memories for a user.
    ///
    /// # Arguments
    /// * `user_id` - User identifier.
    /// * `limit` - Maximum number of results (default: 10).
    ///
    /// # Returns
    /// An array of JS objects.
    pub fn list(&self, user_id: &str, limit: Option<u32>) -> Result<JsValue, JsError> {
        let mut opts = memme_core::types::ListOptions::new(user_id);
        if let Some(l) = limit {
            opts = opts.limit(l as usize);
        }

        let store = self
            .inner
            .lock()
            .map_err(|e| JsError::new(&e.to_string()))?;
        let results = store
            .list_traces(opts)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let js_results: Vec<JsMemoryResult> = results.iter().map(to_js_memory).collect();
        serde_wasm_bindgen::to_value(&js_results).map_err(|e| JsError::new(&e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Serializable memory result for WASM transport.
///
/// Mirrors `memme_core::types::MemoryResult` but with `serde_json::Value`
/// metadata serialized to a JSON string so it passes through
/// `serde-wasm-bindgen` cleanly.
#[derive(serde::Serialize)]
struct JsMemoryResult {
    id: String,
    content: String,
    user_id: String,
    agent_id: Option<String>,
    app_id: Option<String>,
    run_id: Option<String>,
    score: Option<f32>,
    created_at: String,
    updated_at: String,
    metadata: Option<String>,
    importance: Option<f32>,
    access_count: Option<u32>,
    immutable: bool,
    expiration_date: Option<String>,
    categories: Option<Vec<String>>,
    retention: Option<f32>,
    stability: Option<f32>,
    event_time: Option<String>,
}

fn to_js_memory(r: &memme_core::types::MemoryResult) -> JsMemoryResult {
    JsMemoryResult {
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

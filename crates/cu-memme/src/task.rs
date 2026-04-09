//! CuTask implementation for MemMe memory engine.
//!
//! NOTE: This is a skeleton for Copper-rs integration. The exact trait
//! signatures depend on the cu29 version and may need adjustment.
//! Copper uses compile-time task graph generation via proc macros,
//! so this task must be used with `#[copper_runtime(config = "...")]`.

use std::sync::Arc;

use cu29::prelude::*;
use tracing::{debug, error, info};

use memme_core::config::MemoryConfig;
use memme_core::memory::MemoryStore;
use memme_core::types::*;
use memme_embeddings::Embedder;

use crate::messages::{MemMeRequest, MemMeResponse, MemoryHit};

/// Copper task that provides persistent memory via MemMe.
///
/// This task should run on the **background** path (not time-critical) because
/// embedding inference and SQLite I/O have variable latency.
///
/// ```ron
/// (id: "memme", type: "cu_memme::MemMeTask", background: true, config: {
///     "db_path": "robot_memory.db",
///     "embedding_dims": "384",
/// })
/// ```
pub struct MemMeTask {
    store: MemoryStore,
    #[allow(dead_code)]
    rt: tokio::runtime::Runtime,
}

impl Freezable for MemMeTask {}

impl CuTask for MemMeTask {
    type Input<'m> = &'m CuMsg<MemMeRequest>;
    type Output<'m> = &'m mut CuMsg<MemMeResponse>;
    type Resources<'r> = ();

    fn new(config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| CuError::new_with_cause("failed to create tokio runtime", e))?;
        let _guard = rt.enter();

        let (db_path, dims) = if let Some(cfg) = config {
            let p = cfg
                .get::<String>("db_path")
                .unwrap_or_else(|_| "robot_memory.db".into());
            let d: usize = cfg
                .get::<String>("embedding_dims")
                .unwrap_or_else(|_| "384".into())
                .parse()
                .unwrap_or(384);
            (p, d)
        } else {
            ("robot_memory.db".into(), 384)
        };

        let mem_config = MemoryConfig {
            db_path,
            embedding_dims: dims,
            ..Default::default()
        };

        let embedder: Arc<dyn Embedder> =
            Arc::new(memme_embeddings::onnx::OnnxEmbedder::new().map_err(|e| {
                CuError::new_with_cause("failed to initialize ONNX embedder", e)
            })?);

        let store = MemoryStore::new(mem_config, embedder)
            .map_err(|e| CuError::new_with_cause("failed to initialize MemMe store", e))?;

        info!("cu-memme task initialized");
        Ok(Self { store, rt })
    }

    fn process<'i, 'o>(
        &mut self,
        _ctx: &CuContext,
        input: &Self::Input<'i>,
        output: &mut Self::Output<'o>,
    ) -> CuResult<()> {
        let request = input.payload().unwrap_or(&MemMeRequest::Noop);

        let response = match request {
            MemMeRequest::Store {
                content,
                user_id,
                metadata,
            } => handle_store(&self.store, content, user_id, metadata.as_deref()),

            MemMeRequest::Search {
                query,
                user_id,
                limit,
            } => handle_search(&self.store, query, user_id, *limit),

            MemMeRequest::IngestEvent {
                content,
                user_id,
                session_id,
                event_type,
            } => handle_ingest(
                &self.store,
                content,
                user_id,
                session_id.as_deref(),
                event_type.as_deref(),
            ),

            MemMeRequest::Compact { session_id } => handle_compact(&self.store, session_id),

            MemMeRequest::Noop => MemMeResponse::Empty,
        };

        output.set_payload(response);
        Ok(())
    }
}

// --- Handler functions ---

fn handle_store(
    store: &MemoryStore,
    content: &str,
    user_id: &str,
    metadata: Option<&str>,
) -> MemMeResponse {
    let mut opts = AddOptions::new(user_id);
    if let Some(meta_str) = metadata {
        if let Ok(val) = serde_json::from_str(meta_str) {
            opts = opts.metadata(val);
        }
    }

    match store.add(content, opts) {
        Ok(result) => {
            debug!("stored memory: {}", result.id);
            MemMeResponse::Stored {
                id: result.id.clone(),
                content: result.content.clone(),
            }
        }
        Err(e) => {
            error!("store error: {e}");
            MemMeResponse::Error {
                message: e.to_string(),
            }
        }
    }
}

fn handle_search(store: &MemoryStore, query: &str, user_id: &str, limit: u32) -> MemMeResponse {
    let opts = SearchOptions::new(user_id).limit(limit as usize);

    match store.search(query, opts) {
        Ok(results) => {
            let hits: Vec<MemoryHit> = results
                .iter()
                .map(|r| MemoryHit {
                    id: r.id.clone(),
                    content: r.content.clone(),
                    score: r.score.unwrap_or(0.0),
                    created_at: r.created_at.clone(),
                })
                .collect();
            debug!("search returned {} results", hits.len());
            MemMeResponse::SearchResults { results: hits }
        }
        Err(e) => {
            error!("search error: {e}");
            MemMeResponse::Error {
                message: e.to_string(),
            }
        }
    }
}

fn handle_ingest(
    store: &MemoryStore,
    content: &str,
    user_id: &str,
    session_id: Option<&str>,
    event_type: Option<&str>,
) -> MemMeResponse {
    let mut opts = IngestEventOptions::new(user_id);
    if let Some(sid) = session_id {
        opts = opts.session_id(sid);
    }
    if let Some(et) = event_type {
        opts = opts.event_type(et);
    }

    match store.ingest_event(content, opts) {
        Ok(event) => {
            debug!("ingested event: {}", event.event_id);
            MemMeResponse::EventAck {
                event_id: event.event_id.clone(),
            }
        }
        Err(e) => {
            error!("ingest error: {e}");
            MemMeResponse::Error {
                message: e.to_string(),
            }
        }
    }
}

fn handle_compact(store: &MemoryStore, session_id: &str) -> MemMeResponse {
    match store.compact(session_id) {
        Ok(result) => {
            debug!(
                "compacted: episode={}, memories={}",
                result.episode_id,
                result.memories.len()
            );
            MemMeResponse::Compacted {
                episode_id: result.episode_id.clone(),
                memory_count: result.memories.len() as u32,
            }
        }
        Err(e) => {
            error!("compact error: {e}");
            MemMeResponse::Error {
                message: e.to_string(),
            }
        }
    }
}

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use memme_embeddings::Embedder;
use tracing::{debug, info};
use uuid::Uuid;

use crate::config::MemoryConfig;
use crate::dedup::{self, DedupResult};
use crate::error::{MemoryError, Result};
use crate::storage::{InsertMemoryParams, Storage};
use crate::types::*;
#[cfg(feature = "webhooks")]
use crate::webhook::{WebhookEvent, WebhookManager};

mod helpers;
pub(crate) use helpers::{content_hash, recover_lock, row_to_result};

mod analytics;
mod battery;
mod compact_ops;
mod episode_ops;
mod graph;
mod identity_ops;
mod lifecycle;
mod procedural;
mod search;
mod session_ops;
mod smart;
mod stream_ops;
mod sync;
// mod recall_ops; // removed: use search() with SearchOptions instead
mod meditation_ops;

use battery::DeferredOp;
use helpers::{compute_retention, initial_stability_for_tier};

/// Current time as milliseconds since UNIX epoch.
fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// Static assertion: MemoryStore must be Send + Sync for concurrent use.
const _: () = {
    fn _assert_send_sync<T: Send + Sync>() {}
    fn _check() {
        _assert_send_sync::<MemoryStore>();
    }
};

/// Deferred write operations queued during reads (access tracking).
enum DeferredWrite {
    IncrementAccess(String),
    ReinforceStability(String, f32),
}

/// Main entry point for all memory operations.
///
/// `MemoryStore` is the public API surface of `memme-core`. It manages the full
/// memory lifecycle: adding, searching, updating, deleting, and consolidating
/// memories backed by DuckDB storage and vector embeddings.
///
/// # Creating a store
///
/// ```no_run
/// use std::sync::Arc;
/// use memme_core::{MemoryConfig, MemoryStore};
/// use memme_embeddings::mock::MockEmbedder;
///
/// let config = MemoryConfig::new(":memory:", 384);
/// let embedder = Arc::new(MockEmbedder::new(384));
/// let store = MemoryStore::new(config, embedder).unwrap();
/// ```
///
/// # Thread safety
///
/// `MemoryStore` is `Send + Sync` and can be shared across threads via `Arc`.
/// Internally it uses a connection pool with separate read/write connections
/// for file-backed databases, or a `Mutex<Connection>` for in-memory databases.
pub struct MemoryStore {
    pub(crate) storage: Storage,
    embedder: Arc<dyn Embedder>,
    config: MemoryConfig,
    llm: Mutex<Option<Arc<dyn memme_llm::LlmProvider>>>,
    #[cfg(feature = "webhooks")]
    webhook_manager: Option<WebhookManager>,
    /// Battery level stored as u32 (level * 100). Lock-free atomic.
    battery_level: AtomicU32,
    /// Whether device is currently charging.
    battery_charging: AtomicU32,
    /// Queue of deferred operations (when battery is critical).
    deferred_ops: Mutex<Vec<DeferredOp>>,
    /// Queue of deferred write operations from read paths (access tracking).
    deferred_writes: Mutex<Vec<DeferredWrite>>,
    /// Timestamp (millis since UNIX epoch) of the last deferred write flush.
    last_flush_millis: AtomicU64,
    /// Optional reranker for post-fusion re-scoring.
    reranker: Option<Arc<dyn crate::rerank::Reranker>>,
}

impl MemoryStore {
    /// Create a new MemoryStore, opening (or creating) the database and
    /// initializing the schema.
    pub fn new(config: MemoryConfig, embedder: Arc<dyn Embedder>) -> Result<Self> {
        config.validate()?;

        // Validate that embedder dims match config
        if embedder.dimensions() != config.embedding_dims {
            return Err(MemoryError::Config(format!(
                "Embedder dimension {} does not match config dimension {}",
                embedder.dimensions(),
                config.embedding_dims,
            )));
        }

        let storage = Storage::open(config.clone())?;
        info!(db_path = %config.db_path, collection = %config.collection_name, "MemoryStore initialized");

        #[cfg(feature = "webhooks")]
        let webhook_manager = config
            .webhooks
            .as_ref()
            .map(|hooks| WebhookManager::from_configs(hooks.clone()));

        Ok(Self {
            storage,
            embedder,
            config,
            llm: Mutex::new(None),
            #[cfg(feature = "webhooks")]
            webhook_manager,
            battery_level: AtomicU32::new(100), // default: full battery
            battery_charging: AtomicU32::new(0),
            deferred_ops: Mutex::new(Vec::new()),
            deferred_writes: Mutex::new(Vec::new()),
            last_flush_millis: AtomicU64::new(now_millis()),
            reranker: None,
        })
    }

    /// Flush deferred write operations (access count bumps, stability reinforcement).
    /// Called opportunistically during write operations and by time-based auto-flush.
    pub fn flush_deferred_writes(&self) {
        let ops: Vec<DeferredWrite> = {
            let mut queue = recover_lock(&self.deferred_writes, "deferred_writes");
            std::mem::take(&mut *queue)
        };
        if ops.is_empty() {
            return;
        }
        self.last_flush_millis
            .store(now_millis(), Ordering::Relaxed);
        for op in ops {
            match op {
                DeferredWrite::IncrementAccess(id) => {
                    let _ = self.storage.increment_access_count(&id);
                }
                DeferredWrite::ReinforceStability(id, factor) => {
                    let _ = self.storage.reinforce_stability(&id, factor);
                }
            }
        }
    }

    /// Check whether enough time has elapsed to warrant a deferred write flush.
    fn should_time_flush(&self) -> bool {
        let interval_secs = self.config.deferred_flush_interval_secs;
        if interval_secs == 0 {
            return false;
        }
        let last = self.last_flush_millis.load(Ordering::Relaxed);
        now_millis().saturating_sub(last) >= interval_secs * 1000
    }

    /// Check whether an LLM provider is configured.
    pub fn has_llm(&self) -> bool {
        recover_lock(&self.llm, "llm").is_some()
    }

    /// Return the embedding dimensions.
    pub fn embedding_dims(&self) -> usize {
        self.config.embedding_dims
    }

    /// Set the LLM provider for smart operations (builder pattern).
    pub fn with_llm(self, llm: Arc<dyn memme_llm::LlmProvider>) -> Self {
        *recover_lock(&self.llm, "llm") = Some(llm);
        self
    }

    /// Replace the LLM provider at runtime.
    pub fn set_llm_provider(&self, llm: Arc<dyn memme_llm::LlmProvider>) {
        *recover_lock(&self.llm, "llm") = Some(llm);
    }

    /// Set the reranker for post-search re-scoring (builder pattern).
    pub fn with_reranker(mut self, reranker: Arc<dyn crate::rerank::Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Replace the reranker at runtime.
    pub fn set_reranker(&mut self, reranker: Arc<dyn crate::rerank::Reranker>) {
        self.reranker = Some(reranker);
    }

    /// Persist LLM config to the database (for auto-restore on next open).
    /// Does NOT create or change the current LLM provider.
    pub fn save_llm_config(&self, api_key: &str, model: &str, base_url: &str) -> Result<()> {
        self.storage.set_config("llm_api_key", api_key)?;
        self.storage.set_config("llm_model", model)?;
        self.storage.set_config("llm_base_url", base_url)?;
        Ok(())
    }

    /// Read persisted LLM config from DB / env vars. Returns `(api_key, model, base_url)`.
    pub fn load_llm_config(&self) -> Option<(String, String, String)> {
        let api_key = std::env::var("MEMME_LLM_API_KEY")
            .ok()
            .or_else(|| std::env::var("MEMME_API_KEY").ok())
            .or_else(|| self.storage.get_config("llm_api_key").ok().flatten())?;
        let model = self
            .storage
            .get_config("llm_model")
            .ok()
            .flatten()
            .unwrap_or_else(|| "gpt-4o-mini".to_string());
        let base_url = self
            .storage
            .get_config("llm_base_url")
            .ok()
            .flatten()
            .unwrap_or_else(|| "https://api.openai.com".to_string());
        Some((api_key, model, base_url))
    }

    /// Get a clone of the internal LLM provider, if configured.
    pub fn llm(&self) -> Option<Arc<dyn memme_llm::LlmProvider>> {
        recover_lock(&self.llm, "llm").clone()
    }

    /// Add a memory. Performs dedup: if a similar memory exists within
    /// `dedup_threshold`, updates it instead of creating a new one.
    ///
    /// When battery is critical and `defer_when_critical` is enabled,
    /// the operation is queued instead of executed immediately.
    ///
    /// When `auto_prune` is enabled and the user exceeds `max_memories_per_user`,
    /// older memories are pruned after insertion.
    ///
    /// Returns the resulting memory (newly created or updated).
    pub fn add(&self, content: &str, options: AddOptions) -> Result<MemoryResult> {
        self.flush_deferred_writes();
        if content.trim().is_empty() {
            return Err(MemoryError::Config("content cannot be empty".into()));
        }

        // Battery-aware: defer if critical power
        if self.is_critical_power() {
            if let Some(ref pc) = self.config.power_config {
                if pc.defer_when_critical {
                    let mut queue = recover_lock(&self.deferred_ops, "deferred_ops");
                    queue.push(DeferredOp {
                        content: content.to_string(),
                        options: options.clone(),
                    });
                    debug!("Deferred add due to critical battery");
                    return Ok(MemoryResult {
                        id: "deferred".to_string(),
                        content: content.to_string(),
                        user_id: options.user_id.clone(),
                        agent_id: options.agent_id.clone(),
                        app_id: options.app_id.clone(),
                        run_id: options.run_id.clone(),
                        score: None,
                        created_at: String::new(),
                        updated_at: String::new(),
                        metadata: options.metadata.clone(),
                        importance: options.importance,
                        access_count: Some(0),
                        immutable: options.immutable,
                        expiration_date: options.expiration_date.clone(),
                        categories: options.categories.clone(),
                        memory_type: options.memory_type.clone(),
                        retention: None,
                        stability: None,
                        privacy: options.privacy.as_str().to_string(),
                        event_time: options.event_time.clone(),
                        episode_id: options.episode_id.clone(),
                        session_id: options.session_id.clone(),
                        resolution: Resolution::Granular,
                    });
                }
            }
        }

        let embedding = self
            .embedder
            .embed(content)
            .map_err(MemoryError::Embedding)?;

        let hash = content_hash(content);

        // Check for duplicates
        let dedup = dedup::check_dedup(
            &self.storage,
            &embedding,
            &hash,
            content,
            &options.user_id,
            options.agent_id.as_deref(),
            self.config.dedup_threshold,
        )?;

        match dedup {
            DedupResult::Duplicate {
                existing_id,
                existing_content,
            } => {
                info!(id = %existing_id, "Dedup hit — updating existing memory");

                // Update existing memory, passing along the new metadata
                let meta_update = Some(options.metadata.as_ref());
                self.storage.update_memory(
                    &existing_id,
                    content,
                    &embedding,
                    &hash,
                    meta_update,
                    None,
                )?;

                // Record history
                let history_id = Uuid::new_v4().to_string();
                self.storage.record_history(
                    &history_id,
                    &existing_id,
                    &options.user_id,
                    Some(&existing_content),
                    content,
                    HistoryEvent::Update.as_str(),
                )?;

                // Bump sync version for change tracking
                let _ = self.storage.bump_sync_version(&existing_id, None);

                // Fire webhook
                #[cfg(feature = "webhooks")]
                self.fire_webhook(
                    WebhookEvent::MemoryUpdate,
                    &existing_id,
                    serde_json::json!({"content": content, "dedup": true}),
                );

                // Return updated memory
                self.get_trace(&existing_id)?
                    .ok_or_else(|| MemoryError::NotFound(existing_id))
            }
            DedupResult::New => {
                let id = Uuid::new_v4().to_string();
                let metadata_str = options
                    .metadata
                    .as_ref()
                    .map(serde_json::to_string)
                    .transpose()?;

                debug!(id = %id, user_id = %options.user_id, "Inserting new memory");

                let initial_stability = if self.config.enable_forgetting_curve {
                    let imp = options.importance.unwrap_or(0.5);
                    Some(initial_stability_for_tier(imp, 0))
                } else {
                    None
                };

                self.storage.insert_memory(
                    &id,
                    content,
                    &embedding,
                    &options.user_id,
                    &hash,
                    &InsertMemoryParams {
                        agent_id: options.agent_id.clone(),
                        run_id: options.run_id.clone(),
                        app_id: options.app_id.clone(),
                        actor_id: options.actor_id.clone(),
                        metadata: metadata_str,
                        importance: options.importance,
                        immutable: options.immutable,
                        expiration_date: options.expiration_date.clone(),
                        categories: options.categories.clone(),
                        memory_type: options.memory_type.clone(),
                        stability: initial_stability,
                        privacy: Some(options.privacy.as_str().to_string()),
                        event_time: options.event_time.clone(),
                        episode_id: options.episode_id.clone(),
                        session_id: options.session_id.clone(),
                        ..Default::default()
                    },
                )?;

                // Record history
                let history_id = Uuid::new_v4().to_string();
                self.storage.record_history(
                    &history_id,
                    &id,
                    &options.user_id,
                    None,
                    content,
                    HistoryEvent::Add.as_str(),
                )?;

                // Bump sync version for change tracking
                let _ = self.storage.bump_sync_version(&id, None);

                // Fire webhook
                #[cfg(feature = "webhooks")]
                self.fire_webhook(
                    WebhookEvent::MemoryAdd,
                    &id,
                    serde_json::json!({"content": content}),
                );

                let result = self
                    .get_trace(&id)?
                    .ok_or_else(|| MemoryError::NotFound(id))?;

                // Auto-prune if enabled and over limit
                if self.config.auto_prune {
                    if let Some(max) = self.config.max_memories_per_user {
                        let count = self.storage.count_user_memories(&options.user_id)?;
                        if count > max {
                            let to_remove = count - max;
                            let _ = self.storage.prune_memories(
                                &options.user_id,
                                &self.config.pruning_strategy,
                                to_remove,
                            );
                        }
                    }
                }

                Ok(result)
            }
        }
    }

    /// Get a single trace by its ID.
    /// Increments the access_count for the trace if found.
    pub fn get_trace(&self, id: &str) -> Result<Option<MemoryResult>> {
        let row = self.storage.get_memory(id)?;
        if row.is_some() {
            // Defer access tracking to avoid acquiring write lock during read
            let mut queue = recover_lock(&self.deferred_writes, "deferred_writes");
            if self.config.enable_forgetting_curve {
                queue.push(DeferredWrite::ReinforceStability(
                    id.to_string(),
                    self.config.stability_growth_factor,
                ));
            } else {
                queue.push(DeferredWrite::IncrementAccess(id.to_string()));
            }
            drop(queue);
            if self.should_time_flush() {
                self.flush_deferred_writes();
            }
        }
        Ok(row.map(|r| {
            let mut result = row_to_result(r);
            if self.config.enable_forgetting_curve {
                let stability = result.stability.unwrap_or(1.0);
                result.retention = Some(compute_retention(&result.updated_at, stability));
            }
            result
        }))
    }

    /// Update a trace's importance score without changing content or embedding.
    pub fn update_importance(&self, id: &str, importance: f32) -> Result<()> {
        self.storage.update_importance(id, importance)
    }

    /// Update a trace's content (re-embeds and re-dedup-hashes).
    /// Optionally pass UpdateOptions to set a custom timestamp or metadata.
    pub fn update_trace(
        &self,
        id: &str,
        content: &str,
        options: Option<UpdateOptions>,
    ) -> Result<MemoryResult> {
        self.flush_deferred_writes();
        if content.trim().is_empty() {
            return Err(MemoryError::Config("content cannot be empty".into()));
        }

        // Ensure it exists
        let (old_content, user_id) = self
            .storage
            .get_content(id)?
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;

        let embedding = self
            .embedder
            .embed(content)
            .map_err(MemoryError::Embedding)?;
        let hash = content_hash(content);

        self.storage
            .update_memory(id, content, &embedding, &hash, None, options.as_ref())?;

        // Record history
        let history_id = Uuid::new_v4().to_string();
        self.storage.record_history(
            &history_id,
            id,
            &user_id,
            Some(&old_content),
            content,
            HistoryEvent::Update.as_str(),
        )?;

        // Fire webhook
        #[cfg(feature = "webhooks")]
        self.fire_webhook(
            WebhookEvent::MemoryUpdate,
            id,
            serde_json::json!({"content": content, "old_content": old_content}),
        );

        self.get_trace(id)?
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))
    }

    /// Delete a trace by ID.
    pub fn delete_trace(&self, id: &str) -> Result<()> {
        let (old_content, user_id) = self
            .storage
            .get_content(id)?
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;

        self.storage.delete_memory(id)?;

        // Record history
        let history_id = Uuid::new_v4().to_string();
        self.storage.record_history(
            &history_id,
            id,
            &user_id,
            Some(&old_content),
            "",
            HistoryEvent::Delete.as_str(),
        )?;

        // Fire webhook
        #[cfg(feature = "webhooks")]
        self.fire_webhook(
            WebhookEvent::MemoryDelete,
            id,
            serde_json::json!({"old_content": old_content}),
        );

        info!(id, "Memory deleted");
        Ok(())
    }

    /// Delete all traces matching the given filters.
    /// At least user_id must be provided.
    pub fn delete_all_traces(
        &self,
        user_id: &str,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        app_id: Option<&str>,
    ) -> Result<u64> {
        let count = self
            .storage
            .delete_all_memories(user_id, agent_id, run_id, app_id)?;
        info!(
            user_id,
            ?agent_id,
            ?run_id,
            ?app_id,
            count,
            "Deleted all matching memories"
        );
        Ok(count)
    }

    /// Get the change history for a specific trace.
    pub fn trace_history(&self, memory_id: &str) -> Result<Vec<HistoryRecord>> {
        self.storage.get_history(memory_id)
    }

    /// Reset the entire store — delete ALL data.
    /// For production use, simply delete the .duckdb file instead.
    pub fn reset(&self) -> Result<()> {
        self.storage.reset()?;
        info!("Store reset — all data deleted");
        Ok(())
    }

    /// Expose storage reference for tests.
    #[cfg(test)]
    pub(crate) fn storage(&self) -> &Storage {
        &self.storage
    }

    #[cfg(feature = "webhooks")]
    fn fire_webhook(&self, event: WebhookEvent, memory_id: &str, data: serde_json::Value) {
        if let Some(ref manager) = self.webhook_manager {
            manager.fire(event, memory_id, data);
        }
    }
}

#[cfg(test)]
mod tests;

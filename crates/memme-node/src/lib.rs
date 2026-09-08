#![deny(clippy::all)]

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

// ---------------------------------------------------------------------------
// NAPI Types
// ---------------------------------------------------------------------------

/// Memory result returned to JavaScript.
#[napi(object)]
pub struct MemoryResult {
    pub id: String,
    pub content: String,
    pub user_id: String,
    pub agent_id: Option<String>,
    pub app_id: Option<String>,
    pub run_id: Option<String>,
    pub score: Option<f64>,
    pub created_at: String,
    pub updated_at: String,
    pub metadata: Option<String>,
    pub importance: Option<f64>,
    pub access_count: Option<u32>,
    pub immutable: bool,
    pub expiration_date: Option<String>,
    pub categories: Option<Vec<String>>,
    pub memory_type: Option<String>,
    pub retention: Option<f64>,
    pub stability: Option<f64>,
    pub privacy: String,
    pub event_time: Option<String>,
    pub episode_id: Option<String>,
    pub session_id: Option<String>,
}

/// A chat message for smart operations.
#[napi(object)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// A chat message with a stable caller-provided event ID for safe replay.
#[napi(object)]
pub struct IdentifiedChatMessage {
    pub event_id: String,
    pub role: String,
    pub content: String,
    pub timestamp: Option<String>,
}

/// Entity in the knowledge graph.
#[napi(object)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: Option<String>,
    pub user_id: String,
}

/// Relationship in the knowledge graph.
#[napi(object)]
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

/// A history record.
#[napi(object)]
pub struct HistoryRecord {
    pub id: String,
    pub memory_id: String,
    pub old_memory: Option<String>,
    pub new_memory: String,
    pub event: String,
    pub created_at: String,
}

/// User statistics.
#[napi(object)]
pub struct UserStats {
    pub user_id: String,
    pub total_memories: i64,
    pub total_entities: i64,
    pub total_relationships: i64,
    pub earliest_memory: Option<String>,
    pub latest_memory: Option<String>,
    pub unique_agents: i64,
}

/// Memory count per time period.
#[napi(object)]
pub struct TimeBucket {
    pub period: String,
    pub count: i64,
}

/// Top entity by relationship count.
#[napi(object)]
pub struct EntityStat {
    pub name: String,
    pub entity_type: Option<String>,
    pub relationship_count: i64,
}

/// An episode (compressed understanding of a session).
#[napi(object)]
pub struct Episode {
    pub episode_id: String,
    pub title: String,
    pub summary: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub significance: f64,
    pub outcome: Option<String>,
    pub source_id: Option<String>,
    pub event_ids: Vec<String>,
    pub session_ids: Vec<String>,
    pub user_id: String,
    pub created_at: String,
    pub last_recalled: Option<String>,
    pub recall_count: u32,
    pub storage_strength: f64,
    pub retrieval_strength: f64,
    pub score: Option<f64>,
}

/// A stream event (single event in a session).
#[napi(object)]
pub struct StreamEvent {
    pub event_id: String,
    pub event_type: String,
    pub content: String,
    pub timestamp: String,
    pub session_id: Option<String>,
    pub source_id: Option<String>,
    pub user_id: String,
    pub parent_id: Option<String>,
    pub metadata: Option<String>,
    pub processed: bool,
    pub purified_content: Option<String>,
    pub location: Option<String>,
}

/// A session (immutable event container).
#[napi(object)]
pub struct Session {
    pub session_id: String,
    pub user_id: String,
    pub source_id: Option<String>,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub metadata: Option<String>,
    pub created_at: String,
    pub event_count: u32,
}

/// Session context assembled for retrieval-augmented generation.
#[napi(object)]
pub struct SessionContext {
    pub session_id: String,
    pub events: Vec<StreamEvent>,
    pub tokens_used: u32,
    pub token_budget: u32,
    pub episode_summary: Option<String>,
    pub purified_count: u32,
    pub raw_count: u32,
}

/// An identity trait extracted from memories.
#[napi(object)]
pub struct IdentityTrait {
    pub trait_id: String,
    pub trait_type: String,
    pub content: String,
    pub confidence: f64,
    pub evidence_ids: Vec<String>,
    pub user_id: String,
    pub created_at: String,
    pub updated_at: Option<String>,
}

/// A meditation (deep processing) record.
#[napi(object)]
pub struct MeditationRecord {
    pub meditation_id: String,
    pub triggered_by: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub status: String,
    pub user_id: String,
    pub events_processed: u32,
    pub episodes_created: u32,
    pub memories_created: u32,
    pub memories_updated: u32,
    pub memories_decayed: u32,
    pub entities_created: u32,
    pub relations_created: u32,
    pub conflicts_found: u32,
    pub journal: Option<String>,
    pub metadata: Option<String>,
}

/// Result of appending events to a session.
#[napi(object)]
pub struct AppendEventsResult {
    pub session_id: String,
    pub events_appended: u32,
    pub events_replayed: u32,
    pub embedding_pending: u32,
    pub total_unprocessed: u32,
    pub compact_needed: bool,
}

/// Result of compacting a session into an episode.
#[napi(object)]
pub struct CompactResult {
    pub session_id: String,
    pub episode_id: String,
    pub memories: Vec<MemoryResult>,
    pub events_processed: u32,
}

/// Graph search result.
#[napi(object)]
pub struct GraphSearchResult {
    pub entities: Vec<Entity>,
    pub relations: Vec<GraphRelation>,
}

// ---------------------------------------------------------------------------
// Converters
// ---------------------------------------------------------------------------

fn convert_result(r: &memme_core::types::MemoryResult) -> MemoryResult {
    MemoryResult {
        id: r.id.clone(),
        content: r.content.clone(),
        user_id: r.user_id.clone(),
        agent_id: r.agent_id.clone(),
        app_id: r.app_id.clone(),
        run_id: r.run_id.clone(),
        score: r.score.map(|s| s as f64),
        created_at: r.created_at.clone(),
        updated_at: r.updated_at.clone(),
        metadata: r.metadata.as_ref().map(|v| v.to_string()),
        importance: r.importance.map(|v| v as f64),
        access_count: r.access_count,
        immutable: r.immutable,
        expiration_date: r.expiration_date.clone(),
        categories: r.categories.clone(),
        memory_type: r.memory_type.clone(),
        retention: r.retention.map(|v| v as f64),
        stability: r.stability.map(|v| v as f64),
        privacy: r.privacy.clone(),
        event_time: r.event_time.clone(),
        episode_id: r.episode_id.clone(),
        session_id: r.session_id.clone(),
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

fn convert_graph(r: &memme_core::types::GraphSearchResult) -> GraphSearchResult {
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

fn convert_episode(ep: &memme_core::types::Episode) -> Episode {
    Episode {
        episode_id: ep.episode_id.clone(),
        title: ep.title.clone(),
        summary: ep.summary.clone(),
        started_at: ep.started_at.clone(),
        ended_at: ep.ended_at.clone(),
        significance: ep.significance as f64,
        outcome: ep.outcome.clone(),
        source_id: ep.source_id.clone(),
        event_ids: ep.event_ids.clone(),
        session_ids: ep.session_ids.clone(),
        user_id: ep.user_id.clone(),
        created_at: ep.created_at.clone(),
        last_recalled: ep.last_recalled.clone(),
        recall_count: ep.recall_count,
        storage_strength: ep.storage_strength as f64,
        retrieval_strength: ep.retrieval_strength as f64,
        score: ep.score.map(|s| s as f64),
    }
}

fn convert_event(ev: &memme_core::types::Event) -> StreamEvent {
    StreamEvent {
        event_id: ev.event_id.clone(),
        event_type: ev.event_type.as_str().to_string(),
        content: ev.content.clone(),
        timestamp: ev.timestamp.clone(),
        session_id: ev.session_id.clone(),
        source_id: ev.source_id.clone(),
        user_id: ev.user_id.clone(),
        parent_id: ev.parent_id.clone(),
        metadata: ev.metadata.as_ref().map(|v| v.to_string()),
        processed: ev.processed,
        purified_content: ev.purified_content.clone(),
        location: ev.location.clone(),
    }
}

fn convert_session(s: &memme_core::types::Session) -> Session {
    Session {
        session_id: s.session_id.clone(),
        user_id: s.user_id.clone(),
        source_id: s.source_id.clone(),
        started_at: s.started_at.clone(),
        ended_at: s.ended_at.clone(),
        metadata: s.metadata.as_ref().map(|v| v.to_string()),
        created_at: s.created_at.clone(),
        event_count: s.event_count,
    }
}

fn convert_identity_trait(t: &memme_core::types::IdentityTrait) -> IdentityTrait {
    IdentityTrait {
        trait_id: t.trait_id.clone(),
        trait_type: t.trait_type.as_str().to_string(),
        content: t.content.clone(),
        confidence: t.confidence as f64,
        evidence_ids: t.evidence_ids.clone(),
        user_id: t.user_id.clone(),
        created_at: t.created_at.clone(),
        updated_at: t.updated_at.clone(),
    }
}

fn convert_meditation(m: &memme_core::types::MeditationRecord) -> MeditationRecord {
    MeditationRecord {
        meditation_id: m.meditation_id.clone(),
        triggered_by: m.triggered_by.clone(),
        started_at: m.started_at.clone(),
        finished_at: m.finished_at.clone(),
        status: m.status.as_str().to_string(),
        user_id: m.user_id.clone(),
        events_processed: m.events_processed,
        episodes_created: m.episodes_created,
        memories_created: m.memories_created,
        memories_updated: m.memories_updated,
        memories_decayed: m.memories_decayed,
        entities_created: m.entities_created,
        relations_created: m.relations_created,
        conflicts_found: m.conflicts_found,
        journal: m.journal.clone(),
        metadata: m.metadata.as_ref().map(|v| v.to_string()),
    }
}

fn convert_chat_messages(messages: Vec<ChatMessage>) -> Vec<memme_core::types::ChatMessage> {
    messages
        .into_iter()
        .map(|m| memme_core::types::ChatMessage {
            role: m.role,
            content: m.content,
            image_url: None,
            image_type: None,
            timestamp: None,
        })
        .collect()
}

fn create_llm(
    api_key: &str,
    model: &str,
    base_url: Option<&str>,
) -> Result<std::sync::Arc<dyn memme_llm::LlmProvider>> {
    let config = memme_llm::openai::OpenAIConfig {
        api_key: api_key.to_string(),
        base_url: base_url.ok_or_else(|| napi::Error::from_reason("base_url required (full endpoint URL, e.g. https://api.openai.com/v1/chat/completions)"))?.to_string(),
        model: model.to_string(),
    };
    Ok(std::sync::Arc::new(memme_llm::openai::OpenAIProvider::new(
        config,
    )))
}

fn open_core_store(
    config: memme_core::config::MemoryConfig,
    embedder: Arc<dyn memme_embeddings::Embedder>,
    vexdb_extension_path: Option<String>,
) -> Result<memme_core::memory::MemoryStore> {
    let store = match vexdb_extension_path {
        Some(path) => memme_core::memory::MemoryStore::new_with_vexdb_lite(config, embedder, path),
        None => memme_core::memory::MemoryStore::new(config, embedder),
    };
    store.map_err(|e| Error::from_reason(e.to_string()))
}

// ---------------------------------------------------------------------------
// MemoryStore
// ---------------------------------------------------------------------------

/// The main MemMe memory store.
#[napi]
pub struct MemoryStore {
    inner: Arc<memme_core::memory::MemoryStore>,
}

#[napi]
impl MemoryStore {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create a new MemoryStore with mock embedder (for testing).
    /// Pass a trusted VexDB-Lite extension path explicitly, or omit it to use
    /// MEMME_VEXDB_LITE_EXTENSION.
    #[napi(factory)]
    pub fn new_mock(
        db_path: Option<String>,
        dims: Option<u32>,
        vexdb_extension_path: Option<String>,
    ) -> Result<Self> {
        let path = db_path.unwrap_or_else(|| ":memory:".to_string());
        let d = dims.unwrap_or(384) as usize;
        let embedder = Arc::new(memme_embeddings::mock::MockEmbedder::new(d));
        let config = memme_core::config::MemoryConfig::new(&path, d);
        let store = open_core_store(config, embedder, vexdb_extension_path)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    /// Create a new MemoryStore with mock embedder AND mock LLM (for E2E testing).
    /// The mock LLM returns pre-scripted responses in order.
    /// Pass an array of JSON strings that the LLM should return.
    #[napi(factory)]
    pub fn new_mock_with_llm(
        responses: Vec<String>,
        db_path: Option<String>,
        dims: Option<u32>,
        vexdb_extension_path: Option<String>,
    ) -> Result<Self> {
        let path = db_path.unwrap_or_else(|| ":memory:".to_string());
        let d = dims.unwrap_or(384) as usize;
        let embedder = Arc::new(memme_embeddings::mock::MockEmbedder::new(d));
        let config = memme_core::config::MemoryConfig::new(&path, d);
        let llm = Arc::new(ScriptedMockLlm::new(responses)) as Arc<dyn memme_llm::LlmProvider>;
        let store = open_core_store(config, embedder, vexdb_extension_path)?.with_llm(llm);
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    /// Create a new MemoryStore with OpenAI-compatible embedder.
    #[napi(factory)]
    pub fn new_openai(
        api_key: String,
        db_path: Option<String>,
        base_url: Option<String>,
        model: Option<String>,
        dims: Option<u32>,
        vexdb_extension_path: Option<String>,
    ) -> Result<Self> {
        let path = db_path.unwrap_or_else(|| ":memory:".to_string());
        let embed_url = base_url.ok_or_else(|| {
            napi::Error::from_reason(
                "base_url required (full endpoint URL, e.g. https://api.openai.com/v1/embeddings)",
            )
        })?;
        let mut embedder = memme_embeddings::openai::OpenAiEmbedder::new(&api_key, embed_url);
        let final_dims = if let Some(m) = model {
            let d = dims.unwrap_or(1536) as usize;
            embedder = embedder.with_model(memme_embeddings::openai::OpenAiModel::Custom {
                name: m,
                dims: d,
                send_dims: true,
            });
            d
        } else {
            use memme_embeddings::Embedder;
            embedder.dimensions()
        };
        let config = memme_core::config::MemoryConfig::new(&path, final_dims);
        let store = open_core_store(
            config,
            Arc::new(embedder) as Arc<dyn memme_embeddings::Embedder>,
            vexdb_extension_path,
        )?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    // -----------------------------------------------------------------------
    // LLM Configuration (NEW)
    // -----------------------------------------------------------------------

    /// Configure LLM provider at runtime. Call once after construction.
    #[napi]
    pub fn set_llm_provider(
        &self,
        api_key: String,
        model: String,
        base_url: Option<String>,
    ) -> Result<()> {
        let llm = create_llm(&api_key, &model, base_url.as_deref())?;
        self.inner.set_llm_provider(llm);
        Ok(())
    }

    // -- Backup / Restore ---------------------------------------------------

    /// Backup the database to a file path.
    /// Returns metadata about the backup (size, memory count, schema version).
    #[napi]
    pub fn backup_to_path(&self, path: String) -> Result<serde_json::Value> {
        let info = self
            .inner
            .backup_to_path(&path)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        serde_json::to_value(&info).map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Restore the database from a backup file.
    /// **Warning**: The caller must re-create the MemoryStore after calling this.
    #[napi]
    pub fn restore_from_backup(&self, backup_path: String) -> Result<()> {
        let config = self.inner.config().clone();
        memme_core::MemoryStore::restore_from_backup(&backup_path, &config)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Run diagnostic checks on storage, embedder, and LLM.
    #[napi]
    pub fn diagnose(&self) -> Result<serde_json::Value> {
        let report = self.inner.diagnose();
        serde_json::to_value(&report).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Check if an LLM provider is configured.
    #[napi]
    pub fn has_llm(&self) -> bool {
        self.inner.has_llm()
    }

    /// Persist LLM configuration to the database.
    #[napi]
    pub fn save_llm_config(&self, api_key: String, model: String, base_url: String) -> Result<()> {
        self.inner
            .save_llm_config(&api_key, &model, &base_url)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Load persisted LLM configuration. Returns [apiKey, model, baseUrl] or null.
    #[napi]
    pub fn load_llm_config(&self) -> Option<Vec<String>> {
        self.inner.load_llm_config().map(|(k, m, u)| vec![k, m, u])
    }

    // -----------------------------------------------------------------------
    // Core CRUD
    // -----------------------------------------------------------------------

    /// Add a memory (vector dedup, no LLM).
    #[napi]
    pub async fn add(
        &self,
        content: String,
        user_id: String,
        agent_id: Option<String>,
        run_id: Option<String>,
        metadata: Option<String>,
    ) -> Result<MemoryResult> {
        let store = self.inner.clone();
        let meta_val = metadata
            .map(|s| {
                serde_json::from_str::<serde_json::Value>(&s)
                    .map_err(|e| Error::from_reason(format!("Invalid JSON: {e}")))
            })
            .transpose()?;
        tokio::task::spawn_blocking(move || {
            let mut opts = memme_core::types::AddOptions::new(&user_id);
            if let Some(aid) = &agent_id {
                opts = opts.agent_id(aid);
            }
            if let Some(rid) = &run_id {
                opts = opts.run_id(rid);
            }
            if let Some(val) = meta_val {
                opts = opts.metadata(val);
            }
            let r = store
                .add(&content, opts)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(convert_result(&r))
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Search memories.
    #[napi]
    pub async fn search(
        &self,
        query: String,
        user_id: String,
        agent_id: Option<String>,
        run_id: Option<String>,
        limit: Option<u32>,
        threshold: Option<f64>,
    ) -> Result<Vec<MemoryResult>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
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
                opts = opts.threshold(t as f32);
            }
            let results = store
                .search(&query, opts)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(results.iter().map(convert_result).collect())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Get a memory by ID.
    #[napi]
    pub async fn get(&self, id: String) -> Result<Option<MemoryResult>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let r = store
                .get_trace(&id)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(r.map(|r| convert_result(&r)))
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Update a memory.
    #[napi]
    pub async fn update(&self, id: String, content: String) -> Result<MemoryResult> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let r = store
                .update_trace(&id, &content, None)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(convert_result(&r))
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Delete a memory.
    #[napi]
    pub async fn delete(&self, id: String) -> Result<()> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            store
                .delete_trace(&id)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// List memories.
    #[napi]
    pub async fn list(
        &self,
        user_id: String,
        agent_id: Option<String>,
        run_id: Option<String>,
        limit: Option<u32>,
    ) -> Result<Vec<MemoryResult>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
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
            let results = store
                .list_traces(opts)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(results.iter().map(convert_result).collect())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Hybrid search (vector + FTS with RRF fusion).
    ///
    /// RRF weights are configured at store construction time.
    #[napi]
    pub async fn hybrid_search(
        &self,
        query: String,
        user_id: String,
        agent_id: Option<String>,
        run_id: Option<String>,
        limit: Option<u32>,
    ) -> Result<Vec<MemoryResult>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
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
            let results = store
                .search(&query, opts)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(results.iter().map(convert_result).collect())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Rebuild FTS index.
    #[napi]
    pub async fn rebuild_fts_index(&self) -> Result<()> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            store
                .rebuild_fts_index()
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Delete all memories matching filters.
    #[napi]
    pub async fn delete_all(
        &self,
        user_id: String,
        agent_id: Option<String>,
        run_id: Option<String>,
        app_id: Option<String>,
    ) -> Result<i64> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let count = store
                .delete_all_traces(
                    &user_id,
                    agent_id.as_deref(),
                    run_id.as_deref(),
                    app_id.as_deref(),
                )
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(count as i64)
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Get change history for a memory.
    #[napi]
    pub async fn history(&self, memory_id: String) -> Result<Vec<HistoryRecord>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let records = store
                .trace_history(&memory_id)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(records.iter().map(convert_history).collect())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Reset the entire store — delete ALL data. Destructive.
    #[napi]
    pub async fn reset(&self) -> Result<()> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            store
                .reset()
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    // -----------------------------------------------------------------------
    // Session Management (NEW)
    // -----------------------------------------------------------------------

    /// Append chat messages as events to a session. Returns whether compact is needed.
    #[napi]
    pub async fn append_events(
        &self,
        session_id: String,
        messages: Vec<ChatMessage>,
        user_id: String,
        metadata: Option<String>,
    ) -> Result<AppendEventsResult> {
        let store = self.inner.clone();
        let meta = metadata
            .map(|s| serde_json::from_str(&s))
            .transpose()
            .map_err(|e| Error::from_reason(format!("Invalid JSON: {e}")))?;
        let core_messages = convert_chat_messages(messages);
        tokio::task::spawn_blocking(move || {
            let r = store
                .append_events(&session_id, &core_messages, &user_id, meta)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(AppendEventsResult {
                session_id: r.session_id,
                events_appended: r.events_appended as u32,
                events_replayed: r.events_replayed as u32,
                embedding_pending: r.embedding_pending as u32,
                total_unprocessed: r.total_unprocessed as u32,
                compact_needed: r.compact_needed,
            })
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Append events with stable IDs. Exact replays are accepted; conflicting
    /// reuse of an ID is rejected.
    #[napi]
    pub async fn append_events_idempotent(
        &self,
        session_id: String,
        messages: Vec<IdentifiedChatMessage>,
        user_id: String,
        metadata: Option<String>,
    ) -> Result<AppendEventsResult> {
        let store = self.inner.clone();
        let meta = metadata
            .map(|value| serde_json::from_str(&value))
            .transpose()
            .map_err(|error| Error::from_reason(format!("Invalid JSON: {error}")))?;
        let messages: Vec<memme_core::types::IdentifiedChatMessage> = messages
            .into_iter()
            .map(|message| memme_core::types::IdentifiedChatMessage {
                event_id: message.event_id,
                message: memme_core::types::ChatMessage {
                    role: message.role,
                    content: message.content,
                    image_url: None,
                    image_type: None,
                    timestamp: message.timestamp,
                },
            })
            .collect();
        tokio::task::spawn_blocking(move || {
            let result = store
                .append_events_idempotent(&session_id, &messages, &user_id, meta)
                .map_err(|error| Error::from_reason(error.to_string()))?;
            Ok(AppendEventsResult {
                session_id: result.session_id,
                events_appended: result.events_appended as u32,
                events_replayed: result.events_replayed as u32,
                embedding_pending: result.embedding_pending as u32,
                total_unprocessed: result.total_unprocessed as u32,
                compact_needed: result.compact_needed,
            })
        })
        .await
        .map_err(|error| Error::from_reason(error.to_string()))?
    }

    /// Compact a session: extract memories + create episode from unprocessed events.
    #[napi]
    pub async fn compact(&self, session_id: String) -> Result<CompactResult> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let r = store
                .compact(&session_id)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(CompactResult {
                session_id: r.session_id,
                episode_id: r.episode_id,
                memories: r.memories.iter().map(convert_result).collect(),
                events_processed: r.events_processed as u32,
            })
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Get a session by ID.
    #[napi]
    pub async fn get_session(&self, session_id: String) -> Result<Option<Session>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let r = store
                .get_session(&session_id)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(r.map(|s| convert_session(&s)))
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// List sessions for a user.
    #[napi]
    pub async fn list_sessions(
        &self,
        user_id: String,
        source_id: Option<String>,
        since: Option<String>,
        until: Option<String>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<Session>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut opts = memme_core::types::ListSessionsOptions::new(&user_id);
            if let Some(s) = source_id {
                opts.source_id = Some(s);
            }
            if let Some(s) = since {
                opts.since = Some(s);
            }
            if let Some(u) = until {
                opts.until = Some(u);
            }
            if let Some(l) = limit {
                opts.limit = Some(l as usize);
            }
            if let Some(o) = offset {
                opts.offset = Some(o as usize);
            }
            let sessions = store
                .list_sessions(opts)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(sessions.iter().map(convert_session).collect())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Delete a session and its events.
    #[napi]
    pub async fn delete_session(&self, session_id: String) -> Result<()> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            store
                .delete_session(&session_id)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Get events in a session (paginated).
    #[napi]
    pub async fn get_session_events(
        &self,
        session_id: String,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<StreamEvent>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let events = store
                .get_session_events(
                    &session_id,
                    limit.map(|l| l as usize),
                    offset.map(|o| o as usize),
                )
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(events.iter().map(convert_event).collect())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Assemble session context for RAG with token budget.
    #[napi]
    pub async fn get_session_context(
        &self,
        session_id: String,
        token_budget: Option<u32>,
        include_summary: Option<bool>,
        max_events: Option<u32>,
    ) -> Result<SessionContext> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut opts = memme_core::types::GetSessionContextOptions::default();
            if let Some(b) = token_budget {
                opts.token_budget = b as usize;
            }
            if let Some(s) = include_summary {
                opts.include_summary = s;
            }
            if let Some(m) = max_events {
                opts.max_events = m as usize;
            }
            let ctx = store
                .get_session_context(&session_id, opts)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(SessionContext {
                session_id: ctx.session_id,
                events: ctx.events.iter().map(convert_event).collect(),
                tokens_used: ctx.tokens_used as u32,
                token_budget: ctx.token_budget as u32,
                episode_summary: ctx.episode_summary,
                purified_count: ctx.purified_count as u32,
                raw_count: ctx.raw_count as u32,
            })
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    // -----------------------------------------------------------------------
    // Episode Management
    // -----------------------------------------------------------------------

    /// List episodes for a user.
    #[napi]
    pub async fn list_episodes(
        &self,
        user_id: String,
        limit: Option<u32>,
        offset: Option<u32>,
        since: Option<String>,
        until: Option<String>,
    ) -> Result<Vec<Episode>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut opts = memme_core::types::ListEpisodesOptions::new(&user_id);
            if let Some(l) = limit {
                opts = opts.limit(l as usize);
            }
            if let Some(o) = offset {
                opts = opts.offset(o as usize);
            }
            if let Some(s) = since {
                opts = opts.since(s);
            }
            if let Some(u) = until {
                opts = opts.until(u);
            }
            let episodes = store
                .list_episodes(opts)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(episodes.iter().map(convert_episode).collect())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Get an episode by ID.
    #[napi]
    pub async fn get_episode(&self, episode_id: String) -> Result<Option<Episode>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let ep = store
                .get_episode(&episode_id)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(ep.map(|e| convert_episode(&e)))
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Get messages (events) in an episode.
    #[napi]
    pub async fn get_episode_messages(
        &self,
        episode_id: String,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<StreamEvent>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut opts = memme_core::types::EpisodeMessagesOptions::new();
            if let Some(l) = limit {
                opts = opts.limit(l as usize);
            }
            if let Some(o) = offset {
                opts = opts.offset(o as usize);
            }
            let events = store
                .get_episode_messages(&episode_id, opts)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(events.iter().map(convert_event).collect())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Search episodes by semantic similarity.
    #[napi]
    pub async fn search_episodes(
        &self,
        query: String,
        user_id: String,
        limit: Option<u32>,
    ) -> Result<Vec<Episode>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let opts = memme_core::types::SearchEpisodesOptions::new(&user_id)
                .limit(limit.unwrap_or(10) as usize);
            let episodes = store
                .search_episodes(&query, opts)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(episodes.iter().map(convert_episode).collect())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Search messages within an episode by semantic similarity.
    #[napi]
    pub async fn search_episode_messages(
        &self,
        episode_id: String,
        query: String,
        limit: Option<u32>,
    ) -> Result<Vec<StreamEvent>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let events = store
                .search_episode_messages(&episode_id, &query, limit.unwrap_or(10) as usize)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(events.iter().map(convert_event).collect())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Delete an episode.
    #[napi]
    pub async fn delete_episode(&self, episode_id: String) -> Result<()> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            store
                .delete_episode(&episode_id)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    // -----------------------------------------------------------------------
    // Identity (NEW)
    // -----------------------------------------------------------------------

    /// List all identity traits for a user.
    #[napi]
    pub async fn list_identity_traits(&self, user_id: String) -> Result<Vec<IdentityTrait>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let traits = store
                .list_identity_traits(&user_id)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(traits.iter().map(convert_identity_trait).collect())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Add or update an identity trait.
    #[napi]
    pub async fn add_identity_trait(
        &self,
        trait_type: String,
        content: String,
        user_id: String,
        confidence: Option<f64>,
        evidence_ids: Option<Vec<String>>,
    ) -> Result<IdentityTrait> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let opts = memme_core::types::AddIdentityTraitOptions {
                trait_type,
                content,
                user_id,
                confidence: confidence.map(|c| c as f32),
                evidence_ids: evidence_ids.unwrap_or_default(),
            };
            let t = store
                .add_identity_trait(opts)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(convert_identity_trait(&t))
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    // -----------------------------------------------------------------------
    // Meditation (NEW)
    // -----------------------------------------------------------------------

    /// Trigger deep processing (meditation): decay, extraction, graph, identity.
    #[napi]
    pub async fn meditate(
        &self,
        user_id: String,
        triggered_by: String,
        since: Option<String>,
    ) -> Result<MeditationRecord> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let opts = memme_core::types::MeditateOptions {
                user_id,
                triggered_by,
                since,
            };
            let r = store
                .meditate(opts)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(convert_meditation(&r))
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    // -----------------------------------------------------------------------
    // Graph
    // -----------------------------------------------------------------------

    /// Add graph with LLM.
    #[napi]
    pub async fn add_graph(
        &self,
        text: String,
        user_id: String,
        llm_api_key: String,
        llm_model: Option<String>,
        llm_base_url: Option<String>,
    ) -> Result<GraphSearchResult> {
        let store = self.inner.clone();
        let llm = create_llm(
            &llm_api_key,
            &llm_model.unwrap_or("gpt-4o-mini".into()),
            llm_base_url.as_deref(),
        )?;
        tokio::task::spawn_blocking(move || {
            let r = store
                .add_graph(&text, &user_id, llm)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(convert_graph(&r))
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Search graph (no LLM).
    #[napi]
    pub async fn search_graph(
        &self,
        query: String,
        user_id: String,
        depth: Option<u32>,
    ) -> Result<GraphSearchResult> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let r = store
                .search_graph(&query, &user_id, depth.unwrap_or(2) as usize)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(convert_graph(&r))
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    // -----------------------------------------------------------------------
    // Analytics
    // -----------------------------------------------------------------------

    /// Get summary statistics for a user.
    #[napi]
    pub async fn user_stats(&self, user_id: String) -> Result<UserStats> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let stats = store
                .user_stats(&user_id)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(UserStats {
                user_id: stats.user_id,
                total_memories: stats.total_memories as i64,
                total_entities: stats.total_entities as i64,
                total_relationships: stats.total_relationships as i64,
                earliest_memory: stats.earliest_memory,
                latest_memory: stats.latest_memory,
                unique_agents: stats.unique_agents as i64,
            })
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Get memory creation frequency by time period.
    #[napi]
    pub async fn memory_frequency(
        &self,
        user_id: String,
        granularity: String,
        limit: Option<u32>,
    ) -> Result<Vec<TimeBucket>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let buckets = store
                .memory_frequency(&user_id, &granularity, limit.unwrap_or(30) as usize)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(buckets
                .iter()
                .map(|b| TimeBucket {
                    period: b.period.clone(),
                    count: b.count as i64,
                })
                .collect())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }

    /// Get top entities by relationship count.
    #[napi]
    pub async fn top_entities(
        &self,
        user_id: String,
        limit: Option<u32>,
    ) -> Result<Vec<EntityStat>> {
        let store = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let entities = store
                .top_entities(&user_id, limit.unwrap_or(10) as usize)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(entities
                .iter()
                .map(|e| EntityStat {
                    name: e.name.clone(),
                    entity_type: e.entity_type.clone(),
                    relationship_count: e.relationship_count as i64,
                })
                .collect())
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?
    }
}

// ---------------------------------------------------------------------------
// ScriptedMockLlm — returns pre-scripted responses for testing
// ---------------------------------------------------------------------------

struct ScriptedMockLlm {
    responses: std::sync::Mutex<Vec<String>>,
}

impl ScriptedMockLlm {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
        }
    }
}

impl memme_llm::LlmProvider for ScriptedMockLlm {
    fn generate(
        &self,
        _messages: &[memme_llm::Message],
        _options: &memme_llm::GenerateOptions,
    ) -> std::result::Result<String, memme_llm::LlmError> {
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            // Return a safe fallback instead of erroring — this allows
            // compact/meditate to use their fallback paths gracefully.
            Ok(r#"{"error": "no more mock responses"}"#.to_string())
        } else {
            Ok(responses.remove(0))
        }
    }

    fn name(&self) -> &str {
        "scripted-mock"
    }
}

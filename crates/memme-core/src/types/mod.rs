//! Core data types for the MemMe memory engine.
//!
//! This module defines the option structs (builders), result structs, and enums
//! used across the public API. Types are organized into sub-modules by domain:
//!
//! - [`stream`] -- raw events and data sources
//! - [`session`] -- conversation sessions and context retrieval
//! - [`episode`] -- consolidated episodes derived from sessions
//! - [`identity`] -- high-level personality traits
//! - [`meditation`] -- batch consolidation records
//! - [`filter`] -- advanced filter expressions for search/list queries

pub mod episode;
mod filter;
pub mod identity;
pub mod meditation;
pub(crate) mod recall;
pub mod session;
mod sql_param;
pub mod stream;
pub use episode::*;
pub use filter::*;
pub use identity::*;
pub use meditation::*;
pub(crate) use recall::*;
pub use session::*;
pub use sql_param::SqlParam;
pub use stream::*;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Options for adding a new memory.
///
/// Use the builder pattern via [`AddOptions::new`] followed by chained setters.
#[derive(Debug, Clone, Default)]
pub struct AddOptions {
    /// Owner of this memory (required).
    pub user_id: String,
    /// Agent that generated or relates to this memory.
    pub agent_id: Option<String>,
    /// Application context for multi-app scoping.
    pub app_id: Option<String>,
    /// Run/conversation ID for grouping related memories.
    pub run_id: Option<String>,
    /// Arbitrary JSON metadata attached to the memory.
    pub metadata: Option<serde_json::Value>,
    /// Importance score (0.0-1.0). Affects forgetting curve initial stability.
    pub importance: Option<f32>,
    /// The actor (user or agent) that created this memory.
    pub actor_id: Option<String>,
    /// If true, the memory cannot be updated or deleted.
    pub immutable: bool,
    /// Optional expiration date (ISO 8601 string). Memory auto-expires after this time.
    pub expiration_date: Option<String>,
    /// Optional categories to assign. If empty, auto-categorization may apply.
    pub categories: Option<Vec<String>>,
    /// Memory type for scoping: "session", "long_term", or "shared".
    /// Default: None (treated as "long_term").
    pub memory_type: Option<String>,
    /// Privacy level for this memory.
    pub privacy: Privacy,
    /// When the event described by this memory happened (not when stored).
    /// ISO 8601 string (e.g. "2025-03-15", "2025-03-15T14:30:00Z").
    pub event_time: Option<String>,
    /// Episode ID this memory was extracted from.
    pub episode_id: Option<String>,
    /// Session ID this memory was extracted from.
    pub session_id: Option<String>,
}

impl AddOptions {
    /// Create new add options with the given user ID.
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            ..Default::default()
        }
    }

    /// Set the agent ID.
    pub fn agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Set the application ID.
    pub fn app_id(mut self, app_id: impl Into<String>) -> Self {
        self.app_id = Some(app_id.into());
        self
    }

    /// Set the run/conversation ID.
    pub fn run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Attach arbitrary JSON metadata.
    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set the importance score (0.0-1.0).
    pub fn importance(mut self, importance: f32) -> Self {
        self.importance = Some(importance);
        self
    }

    /// Set the actor (user or agent) that created this memory.
    pub fn actor_id(mut self, actor_id: impl Into<String>) -> Self {
        self.actor_id = Some(actor_id.into());
        self
    }

    /// Mark this memory as immutable (cannot be updated or deleted).
    pub fn immutable(mut self, immutable: bool) -> Self {
        self.immutable = immutable;
        self
    }

    /// Set an expiration date (ISO 8601). Memory auto-expires after this time.
    pub fn expiration_date(mut self, date: impl Into<String>) -> Self {
        self.expiration_date = Some(date.into());
        self
    }

    /// Assign categories for filtering and organization.
    pub fn categories(mut self, categories: Vec<String>) -> Self {
        self.categories = Some(categories);
        self
    }

    /// Set memory type scope: "session", "long_term", or "shared".
    pub fn memory_type(mut self, memory_type: impl Into<String>) -> Self {
        self.memory_type = Some(memory_type.into());
        self
    }

    /// Set the privacy level.
    pub fn privacy(mut self, privacy: Privacy) -> Self {
        self.privacy = privacy;
        self
    }

    /// Set when the described event happened (ISO 8601), distinct from storage time.
    pub fn event_time(mut self, event_time: impl Into<String>) -> Self {
        self.event_time = Some(event_time.into());
        self
    }

    /// Link this memory to an episode.
    pub fn episode_id(mut self, episode_id: impl Into<String>) -> Self {
        self.episode_id = Some(episode_id.into());
        self
    }

    /// Link this memory to a session.
    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
}

/// Options for searching memories via vector similarity and optional keyword (FTS) search.
///
/// Use the builder pattern via [`SearchOptions::new`] followed by chained setters.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Owner whose memories to search (required).
    pub user_id: String,
    /// Restrict results to a specific agent.
    pub agent_id: Option<String>,
    /// Restrict results to a specific application.
    pub app_id: Option<String>,
    /// Restrict results to a specific run/conversation.
    pub run_id: Option<String>,
    /// Maximum number of results to return. Falls back to [`MemoryConfig::default_limit`].
    pub limit: Option<usize>,
    /// Minimum similarity threshold; results below this score are discarded.
    pub threshold: Option<f32>,
    /// Advanced filter expression for metadata, categories, importance, etc.
    pub filter: Option<FilterExpression>,
    /// If true, also perform keyword (FTS) search alongside vector search
    /// and fuse results via RRF. Default: false (vector-only).
    pub keyword_search: bool,
    /// Specific fields to include in results. If None, all fields returned.
    /// Supported: "id", "content", "user_id", "score", "metadata", "categories",
    /// "created_at", "updated_at", "importance", etc.
    pub fields: Option<Vec<String>>,
}

impl SearchOptions {
    /// Create new search options with the given user ID.
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            ..Default::default()
        }
    }

    /// Restrict results to a specific agent.
    pub fn agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Restrict results to a specific application.
    pub fn app_id(mut self, app_id: impl Into<String>) -> Self {
        self.app_id = Some(app_id.into());
        self
    }

    /// Restrict results to a specific run/conversation.
    pub fn run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Set the maximum number of results.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set the minimum similarity threshold.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Apply an advanced filter expression.
    pub fn filter(mut self, filter: FilterExpression) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Enable keyword (FTS) search alongside vector search.
    pub fn keyword_search(mut self, enabled: bool) -> Self {
        self.keyword_search = enabled;
        self
    }

    /// Specify which fields to include in results.
    pub fn fields(mut self, fields: Vec<String>) -> Self {
        self.fields = Some(fields);
        self
    }

    /// Legacy: simple key-value metadata filter (backwards compatibility).
    pub fn metadata_filter(mut self, filter: HashMap<String, serde_json::Value>) -> Self {
        self.filter = Some(FilterExpression::from_simple_map(filter));
        self
    }
}

/// Options for listing memories (no vector search, just filtered retrieval).
///
/// Use the builder pattern via [`ListOptions::new`] followed by chained setters.
#[derive(Debug, Clone, Default)]
pub struct ListOptions {
    /// Owner whose memories to list (required).
    pub user_id: String,
    /// Restrict to a specific agent.
    pub agent_id: Option<String>,
    /// Restrict to a specific application.
    pub app_id: Option<String>,
    /// Restrict to a specific run/conversation.
    pub run_id: Option<String>,
    /// Maximum number of results. Falls back to [`MemoryConfig::default_limit`].
    pub limit: Option<usize>,
    /// Advanced filter expression for metadata, categories, importance, etc.
    pub filter: Option<FilterExpression>,
}

impl ListOptions {
    /// Create new list options with the given user ID.
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            ..Default::default()
        }
    }

    /// Restrict to a specific agent.
    pub fn agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Restrict to a specific application.
    pub fn app_id(mut self, app_id: impl Into<String>) -> Self {
        self.app_id = Some(app_id.into());
        self
    }

    /// Restrict to a specific run/conversation.
    pub fn run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Set the maximum number of results.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Apply an advanced filter expression.
    pub fn filter(mut self, filter: FilterExpression) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Legacy: simple key-value metadata filter (backwards compatibility).
    pub fn metadata_filter(mut self, filter: HashMap<String, serde_json::Value>) -> Self {
        self.filter = Some(FilterExpression::from_simple_map(filter));
        self
    }
}

/// Options for updating a memory.
#[derive(Debug, Clone, Default)]
pub struct UpdateOptions {
    /// Optional custom timestamp to set (ISO 8601 string).
    /// If not set, uses current time.
    pub timestamp: Option<String>,
    /// Optional metadata to set or merge.
    pub metadata: Option<serde_json::Value>,
}

impl UpdateOptions {
    /// Create new update options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a custom update timestamp (ISO 8601).
    pub fn timestamp(mut self, ts: impl Into<String>) -> Self {
        self.timestamp = Some(ts.into());
        self
    }

    /// Set or merge metadata on the updated memory.
    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Result of a memory operation (read, search, add, update).
///
/// Returned by [`MemoryStore::add`], [`MemoryStore::get_trace`],
/// [`MemoryStore::search`], and related methods.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryResult {
    /// Unique memory ID (UUID v4).
    pub id: String,
    /// The memory content text.
    pub content: String,
    /// Owner of this memory.
    pub user_id: String,
    /// Agent associated with this memory, if any.
    pub agent_id: Option<String>,
    /// Application this memory belongs to, if any.
    pub app_id: Option<String>,
    /// Run/conversation this memory belongs to, if any.
    pub run_id: Option<String>,
    /// Cosine distance score: 0.0 = identical, 2.0 = opposite direction.
    /// Only populated for search results.
    pub score: Option<f32>,
    /// When this memory was first created (ISO 8601).
    pub created_at: String,
    /// When this memory was last updated (ISO 8601).
    pub updated_at: String,
    /// Arbitrary JSON metadata.
    pub metadata: Option<serde_json::Value>,
    /// Importance score for the memory (0.0 to 1.0).
    pub importance: Option<f32>,
    /// Number of times this memory has been accessed via get/search.
    pub access_count: Option<u32>,
    /// Whether this memory is immutable (cannot be updated or deleted).
    pub immutable: bool,
    /// Expiration date (ISO 8601). Null means never expires.
    pub expiration_date: Option<String>,
    /// Categories assigned to this memory.
    pub categories: Option<Vec<String>>,
    /// Memory type: "session", "long_term", or "shared".
    pub memory_type: Option<String>,
    /// Current retention (0.0-1.0) based on forgetting curve.
    /// Only populated when enable_forgetting_curve is true.
    pub retention: Option<f32>,
    /// Memory stability in days. Higher = slower decay.
    pub stability: Option<f32>,
    /// Privacy level of this memory.
    pub privacy: String,
    /// When the event described by this memory happened (not when stored).
    /// ISO 8601 string. None if unknown.
    pub event_time: Option<String>,
    /// Episode ID this memory was extracted from.
    pub episode_id: Option<String>,
    /// Session ID this memory was extracted from.
    pub session_id: Option<String>,
    /// Resolution: granular (fact), narrative (episode summary), identity (trait).
    pub resolution: Resolution,
}

/// Result of a consolidation operation (forgetting curve decay + pruning + expiration cleanup).
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidateResult {
    /// Number of memories whose retention was recalculated.
    pub decayed_count: u64,
    /// Number of memories pruned because retention fell below the threshold.
    pub deleted_count: u64,
    /// Number of expired memories that were cleaned up.
    pub expired_count: u64,
}

/// Pruning strategy for auto-pruning when memory limits are exceeded.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum PruningStrategy {
    /// Least recently accessed (by updated_at)
    #[default]
    LRU,
    /// Lowest importance first
    Importance,
    /// Lowest importance after decay
    Decay,
}

/// Resolution level for a memory, representing its abstraction tier.
///
/// Higher resolution levels are distilled from many lower-level memories:
/// events become granular facts, facts become episode narratives, and
/// recurring patterns become identity traits.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum Resolution {
    /// Atomic extracted fact (finest grain).
    #[default]
    Granular,
    /// Episode-level narrative summary.
    Narrative,
    /// High-level personality / identity trait.
    Identity,
}

impl Resolution {
    /// Return the string representation of this resolution level.
    pub fn as_str(&self) -> &'static str {
        match self {
            Resolution::Granular => "granular",
            Resolution::Narrative => "narrative",
            Resolution::Identity => "identity",
        }
    }

    /// Parse a resolution level from its string representation.
    pub fn parse(s: &str) -> Self {
        match s {
            "narrative" => Resolution::Narrative,
            "identity" => Resolution::Identity,
            _ => Resolution::Granular,
        }
    }
}

/// Privacy level marker. Note: `EncryptedSync` is a placeholder indicating
/// the memory *should* be encrypted before sync. Actual encryption is the
/// responsibility of the sync layer / application. MemMe does not perform
/// encryption itself.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum Privacy {
    /// Never leaves device.
    LocalOnly,
    /// Can be synced to cloud as-is.
    #[default]
    Syncable,
    /// Marker: should be encrypted before sync. MemMe stores this tag but
    /// does **not** encrypt the content — the sync / transport layer must
    /// check this flag and apply encryption before transmitting.
    EncryptedSync,
}

impl Privacy {
    /// Return the string representation of this privacy level.
    pub fn as_str(&self) -> &'static str {
        match self {
            Privacy::LocalOnly => "local_only",
            Privacy::Syncable => "syncable",
            Privacy::EncryptedSync => "encrypted_sync",
        }
    }

    /// Parse a privacy level from its string representation.
    pub fn parse(s: &str) -> Self {
        match s {
            "local_only" => Privacy::LocalOnly,
            "encrypted_sync" => Privacy::EncryptedSync,
            _ => Privacy::Syncable,
        }
    }
}

/// A chat message used as input for smart memory operations
/// (fact extraction, episode creation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message role: "user", "assistant", "system", or "tool".
    pub role: String,
    /// Text content of the message.
    pub content: String,
    /// Optional image URL or base64-encoded image data.
    pub image_url: Option<String>,
    /// Image type: "url" or "base64".
    pub image_type: Option<String>,
    /// Optional ISO 8601 timestamp for when this message was sent.
    pub timestamp: Option<String>,
}

/// An entity extracted from text (knowledge graph node).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique entity ID (UUID v4).
    pub id: String,
    /// Canonical entity name (e.g., "Alice", "Project X").
    pub name: String,
    /// Optional type label (e.g., "person", "organization", "location").
    pub entity_type: Option<String>,
    /// Owner of this entity.
    pub user_id: String,
}

/// A relationship between two entities (knowledge graph edge).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRelation {
    /// Unique relation ID (UUID v4).
    pub id: String,
    /// Source entity name.
    pub source: String,
    /// Source entity ID.
    pub source_id: String,
    /// Target entity name.
    pub target: String,
    /// Target entity ID.
    pub target_id: String,
    /// Relationship label (e.g., "works_at", "friend_of").
    pub relation_type: String,
    /// Owner of this relation.
    pub user_id: String,
    /// Natural language description of the relationship.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Result of a graph search, containing matched entities and their relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSearchResult {
    /// Entities found by the graph search.
    pub entities: Vec<Entity>,
    /// Relationships connecting the found entities.
    pub relations: Vec<GraphRelation>,
}

/// Result of a compact operation (session events consolidated into an episode + memories).
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactResult {
    /// Session that was compacted.
    pub session_id: String,
    /// Episode created from the compacted events.
    pub episode_id: String,
    /// Memories extracted from the episode.
    pub memories: Vec<MemoryResult>,
    /// Graph data extracted alongside memories, if graph is enabled.
    pub graph: Option<GraphSearchResult>,
    /// Number of raw events that were processed.
    pub events_processed: usize,
}

/// Result of appending events to a session via [`MemoryStore::append_events`].
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEventsResult {
    /// Session the events were appended to.
    pub session_id: String,
    /// Number of events successfully appended.
    pub events_appended: usize,
    /// Total unprocessed events in the session after appending.
    pub total_unprocessed: u64,
    /// True when unprocessed events exceed `compact_threshold`.
    /// The caller should call `compact()` when ready.
    pub compact_needed: bool,
}

/// A single record from the history table, tracking memory mutations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryRecord {
    /// Unique history record ID.
    pub id: String,
    /// ID of the memory that was changed.
    pub memory_id: String,
    /// Previous content before the change (None for ADD events).
    pub old_memory: Option<String>,
    /// Content after the change (empty string for DELETE events).
    pub new_memory: String,
    /// Event type: "ADD", "UPDATE", or "DELETE".
    pub event: String,
    /// When this change occurred (ISO 8601).
    pub created_at: String,
}

/// Event types recorded in the history table for audit trailing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryEvent {
    /// A new memory was created.
    Add,
    /// An existing memory was modified (content, metadata, or dedup merge).
    Update,
    /// A memory was deleted.
    Delete,
}

impl HistoryEvent {
    /// Return the string representation of this history event type.
    pub fn as_str(&self) -> &'static str {
        match self {
            HistoryEvent::Add => "ADD",
            HistoryEvent::Update => "UPDATE",
            HistoryEvent::Delete => "DELETE",
        }
    }
}

/// Export format for a single memory record (used by full export/import).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryExport {
    /// Unique memory ID.
    pub id: String,
    /// Memory content text.
    pub content: String,
    /// Owner of this memory.
    pub user_id: String,
    /// Agent associated with this memory.
    pub agent_id: Option<String>,
    /// Application this memory belongs to.
    pub app_id: Option<String>,
    /// Run/conversation this memory belongs to.
    pub run_id: Option<String>,
    /// Arbitrary JSON metadata.
    pub metadata: Option<serde_json::Value>,
    /// Importance score (0.0-1.0).
    pub importance: f32,
    /// Whether this memory is immutable.
    pub immutable: bool,
    /// Expiration date (ISO 8601), if set.
    pub expiration_date: Option<String>,
    /// Assigned categories.
    pub categories: Option<Vec<String>>,
    /// When this memory was first created (ISO 8601).
    pub created_at: String,
    /// When this memory was last updated (ISO 8601).
    pub updated_at: String,
    /// Forgetting curve stability in days.
    pub stability: Option<f32>,
}

/// Result of syncing the replica.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaSyncResult {
    /// Path of the primary database file.
    pub primary_path: String,
    /// Path of the replica file.
    pub replica_path: String,
    /// Size of the copied file in bytes.
    pub size_bytes: u64,
    /// When the sync completed (ISO 8601).
    pub synced_at: String,
}

/// Status of the primary + replica pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaStatus {
    /// Path of the primary database file.
    pub primary_path: String,
    /// Whether the primary file is accessible.
    pub primary_ok: bool,
    /// Primary file size in bytes.
    pub primary_size_bytes: u64,
    /// Path of the replica file (None for :memory: databases).
    pub replica_path: Option<String>,
    /// Whether the replica file is accessible.
    pub replica_ok: bool,
    /// Replica file size in bytes.
    pub replica_size_bytes: Option<u64>,
    /// When the replica was last synced (ISO 8601, from memme_config).
    pub last_synced_at: Option<String>,
}

/// Metadata returned after a successful backup operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupInfo {
    /// Path of the source (primary) database.
    pub source_path: String,
    /// Path where the backup was written.
    pub backup_path: String,
    /// Backup file size in bytes.
    pub size_bytes: u64,
    /// When the backup was created (ISO 8601).
    pub created_at: String,
    /// Number of memories in the database at backup time.
    pub memory_count: u64,
    /// Schema version of the database.
    pub schema_version: String,
}

/// Full export structure for backup and migration, including all data layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullExport {
    /// Export format version (for forward compatibility).
    pub version: String,
    /// Collection name that was exported.
    pub collection: String,
    /// When the export was created (ISO 8601).
    pub exported_at: String,
    /// All memories in the collection.
    pub memories: Vec<MemoryExport>,
    /// All entities from the knowledge graph.
    pub entities: Vec<Entity>,
    /// All relationships from the knowledge graph.
    pub relations: Vec<GraphRelation>,
    /// All sessions.
    #[serde(default)]
    pub sessions: Vec<Session>,
    /// All episodes.
    #[serde(default)]
    pub episodes: Vec<Episode>,
    /// All raw events.
    #[serde(default)]
    pub events: Vec<Event>,
    /// All identity traits.
    #[serde(default)]
    pub identity_traits: Vec<IdentityTrait>,
    /// All data sources.
    #[serde(default)]
    pub sources: Vec<Source>,
}

/// Result of a full import operation, with counts for each data layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullImportResult {
    /// Number of sources imported.
    pub sources: u64,
    /// Number of sessions imported.
    pub sessions: u64,
    /// Number of events imported.
    pub events: u64,
    /// Number of episodes imported.
    pub episodes: u64,
    /// Number of memories imported.
    pub memories: u64,
    /// Number of entities imported.
    pub entities: u64,
    /// Number of relations imported.
    pub relations: u64,
    /// Number of identity traits imported.
    pub identity_traits: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_options_builder() {
        let opts = AddOptions::new("user1")
            .agent_id("agent1")
            .app_id("myapp")
            .run_id("run1")
            .metadata(serde_json::json!({"key": "value"}))
            .immutable(true)
            .expiration_date("2026-12-31T23:59:59Z")
            .categories(vec!["work".into(), "tech".into()]);
        assert_eq!(opts.user_id, "user1");
        assert_eq!(opts.agent_id.as_deref(), Some("agent1"));
        assert_eq!(opts.app_id.as_deref(), Some("myapp"));
        assert_eq!(opts.run_id.as_deref(), Some("run1"));
        assert_eq!(opts.metadata.unwrap()["key"], "value");
        assert!(opts.immutable);
        assert_eq!(
            opts.expiration_date.as_deref(),
            Some("2026-12-31T23:59:59Z")
        );
        assert_eq!(opts.categories.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_search_options_builder() {
        let opts = SearchOptions::new("user1")
            .app_id("myapp")
            .run_id("run1")
            .limit(5)
            .threshold(0.8);
        assert_eq!(opts.user_id, "user1");
        assert_eq!(opts.app_id.as_deref(), Some("myapp"));
        assert_eq!(opts.run_id.as_deref(), Some("run1"));
        assert_eq!(opts.limit, Some(5));
        assert_eq!(opts.threshold, Some(0.8));
    }

    #[test]
    fn test_list_options_builder() {
        let opts = ListOptions::new("user1")
            .agent_id("agent1")
            .app_id("app1")
            .run_id("run1")
            .limit(20);
        assert_eq!(opts.user_id, "user1");
        assert_eq!(opts.agent_id.as_deref(), Some("agent1"));
        assert_eq!(opts.app_id.as_deref(), Some("app1"));
        assert_eq!(opts.run_id.as_deref(), Some("run1"));
        assert_eq!(opts.limit, Some(20));
    }

    #[test]
    fn test_update_options_builder() {
        let opts = UpdateOptions::new()
            .timestamp("2026-06-15T10:00:00Z")
            .metadata(serde_json::json!({"tag": "important"}));
        assert_eq!(opts.timestamp.as_deref(), Some("2026-06-15T10:00:00Z"));
        assert!(opts.metadata.is_some());
    }

    #[test]
    fn test_history_event_as_str() {
        assert_eq!(HistoryEvent::Add.as_str(), "ADD");
        assert_eq!(HistoryEvent::Update.as_str(), "UPDATE");
        assert_eq!(HistoryEvent::Delete.as_str(), "DELETE");
    }

    #[test]
    fn test_memory_result_with_new_fields() {
        let result = MemoryResult {
            id: "test-id".into(),
            content: "hello".into(),
            user_id: "user1".into(),
            agent_id: Some("agent1".into()),
            app_id: Some("myapp".into()),
            run_id: None,
            score: Some(0.5),
            created_at: "2024-01-01".into(),
            updated_at: "2024-01-02".into(),
            metadata: None,
            importance: Some(0.5),
            access_count: Some(0),
            immutable: true,
            expiration_date: Some("2025-12-31".into()),
            categories: Some(vec!["test".into()]),
            memory_type: None,
            retention: None,
            stability: Some(1.0),
            privacy: "syncable".to_string(),
            event_time: None,
            episode_id: None,
            session_id: None,
            resolution: Resolution::Granular,
        };
        assert_eq!(result.id, "test-id");
        assert!(result.immutable);
        assert_eq!(result.expiration_date.as_deref(), Some("2025-12-31"));
        assert_eq!(result.categories.as_ref().unwrap()[0], "test");
        assert_eq!(result.app_id.as_deref(), Some("myapp"));
    }

    #[test]
    fn test_consolidate_result_with_expired() {
        let r = ConsolidateResult {
            decayed_count: 5,
            deleted_count: 2,
            expired_count: 3,
        };
        assert_eq!(r.expired_count, 3);
    }

    #[test]
    fn test_export_serialization() {
        let export = FullExport {
            version: "2.0".into(),
            collection: "default".into(),
            exported_at: "2026-03-17".into(),
            memories: vec![],
            entities: vec![],
            relations: vec![],
            sessions: vec![],
            episodes: vec![],
            events: vec![],
            identity_traits: vec![],
            sources: vec![],
        };
        let json = serde_json::to_string(&export).unwrap();
        assert!(json.contains("\"version\":\"2.0\""));
        // Verify backwards compat: deserialize without new fields
        let legacy_json = r#"{"version":"1.0","collection":"test","exported_at":"2026-01-01","memories":[],"entities":[],"relations":[]}"#;
        let parsed: FullExport = serde_json::from_str(legacy_json).unwrap();
        assert_eq!(parsed.version, "1.0");
        assert!(parsed.sessions.is_empty());
        assert!(parsed.events.is_empty());
    }
}

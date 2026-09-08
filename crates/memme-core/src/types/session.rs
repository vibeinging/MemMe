use serde::{Deserialize, Serialize};

use super::Event;

/// A conversation session -- a container for events within a time window.
///
/// Sessions group related events (e.g., one chat conversation) and serve
/// as the unit of compaction: when a session is compacted, its events are
/// consolidated into an [`Episode`](super::Episode).
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session ID (UUID v4).
    pub session_id: String,
    /// Owner of this session.
    pub user_id: String,
    /// Data source that created this session (e.g., an assistant or IDE plugin).
    pub source_id: Option<String>,
    /// When the session started (ISO 8601).
    pub started_at: String,
    /// When the session ended (ISO 8601). None if still active.
    pub ended_at: Option<String>,
    /// Arbitrary JSON metadata.
    pub metadata: Option<serde_json::Value>,
    /// When this session record was created (ISO 8601).
    pub created_at: String,
    /// Number of events in this session.
    pub event_count: u32,
    /// Structured notes accumulated during append_events for LLM-free compact summary.
    pub structured_notes: Option<String>,
    /// Number of searches that touched this session.
    #[serde(default)]
    pub queried_count: u32,
    /// Last time a search touched this session.
    #[serde(default)]
    pub last_queried_at: Option<String>,
}

/// Options for listing sessions with optional time range and pagination.
#[derive(Debug, Clone, Default)]
pub struct ListSessionsOptions {
    /// Owner whose sessions to list (required).
    pub user_id: String,
    /// Filter by data source.
    pub source_id: Option<String>,
    /// Only include sessions started at or after this timestamp (ISO 8601).
    pub since: Option<String>,
    /// Only include sessions started before this timestamp (ISO 8601).
    pub until: Option<String>,
    /// Maximum number of sessions to return.
    pub limit: Option<usize>,
    /// Number of sessions to skip (for pagination).
    pub offset: Option<usize>,
}

impl ListSessionsOptions {
    /// Create new list-sessions options with the given user ID.
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            ..Default::default()
        }
    }

    /// Filter by data source.
    pub fn source_id(mut self, v: impl Into<String>) -> Self {
        self.source_id = Some(v.into());
        self
    }

    /// Only include sessions started at or after this timestamp.
    pub fn since(mut self, v: impl Into<String>) -> Self {
        self.since = Some(v.into());
        self
    }

    /// Only include sessions started before this timestamp.
    pub fn until(mut self, v: impl Into<String>) -> Self {
        self.until = Some(v.into());
        self
    }

    /// Set the maximum number of results.
    pub fn limit(mut self, v: usize) -> Self {
        self.limit = Some(v);
        self
    }

    /// Set the pagination offset.
    pub fn offset(mut self, v: usize) -> Self {
        self.offset = Some(v);
        self
    }
}

/// Session context for retrieval -- purified events within a token budget.
///
/// Returned by [`MemoryStore::get_session_context`]. Contains the most
/// recent events that fit within the requested token budget, preferring
/// purified (post-compact) content when available.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContext {
    /// Session ID
    pub session_id: String,
    /// Events within token budget (preferring purified content when available)
    pub events: Vec<Event>,
    /// Total tokens used (approximate)
    pub tokens_used: usize,
    /// Token budget that was requested
    pub token_budget: usize,
    /// Episode summary if available
    pub episode_summary: Option<String>,
    /// Number of events with purified content
    pub purified_count: usize,
    /// Number of events still using raw content (not yet compacted)
    pub raw_count: usize,
}

/// Options for getting session context.
#[derive(Debug, Clone)]
pub struct GetSessionContextOptions {
    /// Maximum tokens to include (default: 2000)
    pub token_budget: usize,
    /// Include episode summary if available (default: true)
    pub include_summary: bool,
    /// Maximum number of events to consider (default: 100)
    pub max_events: usize,
    /// How to handle unprocessed (not yet compacted) events:
    /// - true (default): Include unprocessed events with raw content
    /// - false: Only include events that have been purified
    pub include_unprocessed: bool,
}

impl Default for GetSessionContextOptions {
    fn default() -> Self {
        Self {
            token_budget: 2000,
            include_summary: true,
            max_events: 100,
            include_unprocessed: true,
        }
    }
}

impl GetSessionContextOptions {
    /// Create new session-context options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum token budget for the returned context.
    pub fn token_budget(mut self, tokens: usize) -> Self {
        self.token_budget = tokens;
        self
    }

    /// Whether to include the episode summary in the context.
    pub fn include_summary(mut self, include: bool) -> Self {
        self.include_summary = include;
        self
    }

    /// Set the maximum number of events to consider.
    pub fn max_events(mut self, max: usize) -> Self {
        self.max_events = max;
        self
    }

    /// Only return events that have been purified (high-quality context).
    /// Use this when you want guaranteed high-quality context and can tolerate
    /// fewer events or waiting for compact to complete.
    pub fn purified_only(mut self) -> Self {
        self.include_unprocessed = false;
        self
    }

    /// Include all events, using raw content for unprocessed ones (default).
    /// Use this for real-time conversations where context completeness matters
    /// more than purification quality.
    pub fn include_all(mut self) -> Self {
        self.include_unprocessed = true;
        self
    }
}

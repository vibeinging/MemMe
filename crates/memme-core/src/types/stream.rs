use serde::{Deserialize, Serialize};

/// A data source that writes events into MemMe (e.g., an AI assistant, IDE, or bot).
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Unique source ID (UUID v4).
    pub source_id: String,
    /// Source category: "assistant", "ide", "bot", or "manual".
    pub source_type: String,
    /// Human-readable source name.
    pub name: Option<String>,
    /// When this source was registered (ISO 8601).
    pub registered_at: String,
    /// Arbitrary JSON metadata about this source.
    pub metadata: Option<serde_json::Value>,
}

/// Type of event in the stream layer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventType {
    /// A message from the user.
    UserMessage,
    /// A response from an AI model.
    AiResponse,
    /// A tool/function call made by the AI.
    ToolCall,
    /// The result returned by a tool/function.
    ToolResult,
    /// An error that occurred during processing.
    Error,
    /// A system-level event (catch-all).
    System,
}

impl EventType {
    /// Return the string representation of this event type.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UserMessage => "user_message",
            Self::AiResponse => "ai_response",
            Self::ToolCall => "tool_call",
            Self::ToolResult => "tool_result",
            Self::Error => "error",
            Self::System => "system",
        }
    }

    /// Parse an event type from its string representation.
    pub fn parse(s: &str) -> Self {
        match s {
            "user_message" => Self::UserMessage,
            "ai_response" => Self::AiResponse,
            "tool_call" => Self::ToolCall,
            "tool_result" => Self::ToolResult,
            "error" => Self::Error,
            _ => Self::System,
        }
    }
}

/// A raw event in the stream layer -- the atomic unit of data ingestion.
///
/// Events are ingested via [`MemoryStore::ingest_event`] or [`MemoryStore::append_events`],
/// grouped into sessions, and later compacted into episodes and memories.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Unique event ID (UUID v4).
    pub event_id: String,
    /// Data source that produced this event.
    pub source_id: Option<String>,
    /// Session this event belongs to.
    pub session_id: Option<String>,
    /// When this event occurred (ISO 8601).
    pub timestamp: String,
    /// Classification of this event.
    pub event_type: EventType,
    /// Raw text content of the event.
    pub content: String,
    /// Parent event ID for threading (e.g., a tool result referencing its tool call).
    pub parent_id: Option<String>,
    /// Arbitrary JSON metadata.
    pub metadata: Option<serde_json::Value>,
    /// Owner of this event.
    pub user_id: String,
    /// Whether this event has been processed by compact().
    pub processed: bool,
    /// When this event was processed (ISO 8601).
    pub processed_at: Option<String>,

    /// Purified content after coreference resolution and temporal/spatial grounding.
    /// Set by compact() when purification is enabled.
    pub purified_content: Option<String>,

    /// Whether this event has been purified by compact().
    pub purified: bool,

    /// Event time extracted from content (e.g., "tomorrow" -> "2026-03-29").
    pub event_time: Option<String>,

    /// Location mentioned in content (e.g., "there" -> "cafe on Main St").
    pub location: Option<String>,
}

/// Options for ingesting a single event into the stream.
#[derive(Debug, Clone, Default)]
pub struct IngestEventOptions {
    /// Data source producing this event.
    pub source_id: Option<String>,
    /// Session to attach this event to. Auto-created if it does not exist.
    pub session_id: Option<String>,
    /// Event type string. Defaults to "system".
    pub event_type: Option<String>,
    /// Parent event ID for threading.
    pub parent_id: Option<String>,
    /// Arbitrary JSON metadata.
    pub metadata: Option<serde_json::Value>,
    /// Owner of this event (required).
    pub user_id: String,
    /// Event timestamp (ISO 8601). Defaults to current time.
    pub timestamp: Option<String>,
}

impl IngestEventOptions {
    /// Create new ingest options with the given user ID.
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            ..Default::default()
        }
    }

    /// Set the data source.
    pub fn source_id(mut self, v: impl Into<String>) -> Self {
        self.source_id = Some(v.into());
        self
    }
    /// Set the session to attach this event to.
    pub fn session_id(mut self, v: impl Into<String>) -> Self {
        self.session_id = Some(v.into());
        self
    }
    /// Set the event type (e.g., "user_message", "ai_response").
    pub fn event_type(mut self, v: impl Into<String>) -> Self {
        self.event_type = Some(v.into());
        self
    }
    /// Set the parent event ID for threading.
    pub fn parent_id(mut self, v: impl Into<String>) -> Self {
        self.parent_id = Some(v.into());
        self
    }
    /// Attach arbitrary JSON metadata.
    pub fn metadata(mut self, v: serde_json::Value) -> Self {
        self.metadata = Some(v);
        self
    }
    /// Set a custom timestamp (ISO 8601).
    pub fn timestamp(mut self, v: impl Into<String>) -> Self {
        self.timestamp = Some(v.into());
        self
    }
}

/// Options for listing events with optional filters and time range.
#[derive(Debug, Clone, Default)]
pub struct ListEventsOptions {
    /// Owner whose events to list (required).
    pub user_id: String,
    /// Filter by data source.
    pub source_id: Option<String>,
    /// Filter by session.
    pub session_id: Option<String>,
    /// Only include events at or after this timestamp (ISO 8601).
    pub since: Option<String>,
    /// Only include events before this timestamp (ISO 8601).
    pub until: Option<String>,
    /// Filter by processed status (true = processed only, false = unprocessed only).
    pub processed: Option<bool>,
    /// Maximum number of events to return.
    pub limit: Option<usize>,
}

impl ListEventsOptions {
    /// Create new list-events options with the given user ID.
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
    /// Filter by session.
    pub fn session_id(mut self, v: impl Into<String>) -> Self {
        self.session_id = Some(v.into());
        self
    }
    /// Only include events at or after this timestamp.
    pub fn since(mut self, v: impl Into<String>) -> Self {
        self.since = Some(v.into());
        self
    }
    /// Only include events before this timestamp.
    pub fn until(mut self, v: impl Into<String>) -> Self {
        self.until = Some(v.into());
        self
    }
    /// Only return events that have not yet been processed by compact().
    pub fn unprocessed_only(mut self) -> Self {
        self.processed = Some(false);
        self
    }
    /// Set the maximum number of results.
    pub fn limit(mut self, v: usize) -> Self {
        self.limit = Some(v);
        self
    }
}

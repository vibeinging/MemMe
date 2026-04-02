use serde::{Deserialize, Serialize};

/// Status of a meditation session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MeditationStatus {
    Running,
    Completed,
    Failed,
    Interrupted,
}

impl MeditationStatus {
    /// Return the string representation of this meditation status.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Interrupted => "interrupted",
        }
    }
    /// Parse a meditation status from its string representation.
    pub fn parse(s: &str) -> Self {
        match s {
            "running" => Self::Running,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            _ => Self::Interrupted,
        }
    }
}

/// Record of a meditation (batch consolidation) session.
///
/// A meditation processes unprocessed events across all sessions for a user,
/// creating episodes, extracting memories and graph data, and running
/// forgetting curve decay -- similar to how sleep consolidates memories.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeditationRecord {
    /// Unique meditation ID (UUID v4).
    pub meditation_id: String,
    /// Source that triggered this meditation (e.g., a cron job or user action).
    pub triggered_by: String,
    /// When the meditation started (ISO 8601).
    pub started_at: String,
    /// When the meditation finished (ISO 8601). None if still running.
    pub finished_at: Option<String>,
    /// Current status of the meditation.
    pub status: MeditationStatus,
    /// Owner whose data was processed.
    pub user_id: String,
    /// Number of raw events processed.
    pub events_processed: u32,
    /// Number of episodes created.
    pub episodes_created: u32,
    /// Number of new memories extracted.
    pub memories_created: u32,
    /// Number of existing memories updated (dedup merges).
    pub memories_updated: u32,
    /// Number of memories that decayed below threshold.
    pub memories_decayed: u32,
    /// Number of entities added to the knowledge graph.
    pub entities_created: u32,
    /// Number of relationships added to the knowledge graph.
    pub relations_created: u32,
    /// Number of conflicting facts detected.
    pub conflicts_found: u32,
    /// Human-readable journal of what happened during meditation.
    pub journal: Option<String>,
    /// Arbitrary JSON metadata.
    pub metadata: Option<serde_json::Value>,
}

impl Default for MeditationRecord {
    fn default() -> Self {
        Self {
            meditation_id: String::new(),
            triggered_by: String::new(),
            started_at: String::new(),
            finished_at: None,
            status: MeditationStatus::Running,
            user_id: String::new(),
            events_processed: 0,
            episodes_created: 0,
            memories_created: 0,
            memories_updated: 0,
            memories_decayed: 0,
            entities_created: 0,
            relations_created: 0,
            conflicts_found: 0,
            journal: None,
            metadata: None,
        }
    }
}

/// Options for starting a meditation (batch consolidation).
#[derive(Debug, Clone)]
pub struct MeditateOptions {
    /// Owner whose events to process (required).
    pub user_id: String,
    /// Source that triggered this meditation (required).
    pub triggered_by: String,
    /// Only process events since this timestamp (ISO 8601). Default: last 24h.
    pub since: Option<String>,
}

impl MeditateOptions {
    /// Create new meditation options with user ID and trigger source.
    pub fn new(user_id: impl Into<String>, triggered_by: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            triggered_by: triggered_by.into(),
            since: None,
        }
    }
    /// Only process events since this timestamp.
    pub fn since(mut self, v: impl Into<String>) -> Self {
        self.since = Some(v.into());
        self
    }
}

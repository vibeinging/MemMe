use serde::{Deserialize, Serialize};

/// An episode -- a consolidated experience derived from compacting session events.
///
/// Episodes sit between raw events and extracted memories in the data pipeline.
/// They carry a narrative summary, significance score, and Bjork dual-strength
/// values for spaced retrieval.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique episode ID (UUID v4).
    pub episode_id: String,
    /// Short title summarizing the episode.
    pub title: String,
    /// Narrative summary of what happened.
    pub summary: String,
    /// When the episode started (ISO 8601).
    pub started_at: String,
    /// When the episode ended (ISO 8601). None if ongoing.
    pub ended_at: Option<String>,
    /// How significant this episode is (0.0-1.0).
    pub significance: f32,
    /// Outcome of the episode: "success", "failure", "partial", or "ongoing".
    pub outcome: Option<String>,
    /// Data source that produced the events.
    pub source_id: Option<String>,
    /// IDs of events that were compacted into this episode.
    pub event_ids: Vec<String>,
    /// Sessions this episode was derived from.
    pub session_ids: Vec<String>,
    /// Owner of this episode.
    pub user_id: String,
    /// When this episode record was created (ISO 8601).
    pub created_at: String,
    /// When this episode was last retrieved (ISO 8601).
    pub last_recalled: Option<String>,
    /// Number of times this episode has been recalled.
    pub recall_count: u32,
    /// Bjork storage strength -- monotonically increases with each recall.
    pub storage_strength: f32,
    /// Bjork retrieval strength -- decays over time, boosted by recall.
    pub retrieval_strength: f32,
    /// Relevance score, populated only during search results.
    pub score: Option<f32>,
}

/// Options for creating an episode manually (outside the compact pipeline).
#[derive(Debug, Clone)]
pub struct CreateEpisodeOptions {
    /// Short title for the episode.
    pub title: String,
    /// Narrative summary.
    pub summary: String,
    /// Owner of this episode (required).
    pub user_id: String,
    /// When the episode started (ISO 8601, required).
    pub started_at: String,
    /// When the episode ended (ISO 8601).
    pub ended_at: Option<String>,
    /// Significance score (0.0-1.0). Defaults to 0.5 if not set.
    pub significance: Option<f32>,
    /// Outcome: "success", "failure", "partial", or "ongoing".
    pub outcome: Option<String>,
    /// Data source that produced the events.
    pub source_id: Option<String>,
    /// Event IDs that make up this episode.
    pub event_ids: Vec<String>,
    /// Session IDs this episode was derived from.
    pub session_ids: Vec<String>,
}

impl CreateEpisodeOptions {
    /// Create new episode options with required fields.
    pub fn new(
        title: impl Into<String>,
        summary: impl Into<String>,
        user_id: impl Into<String>,
        started_at: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            summary: summary.into(),
            user_id: user_id.into(),
            started_at: started_at.into(),
            ended_at: None,
            significance: None,
            outcome: None,
            source_id: None,
            event_ids: Vec::new(),
            session_ids: Vec::new(),
        }
    }
    /// Set the end time of the episode.
    pub fn ended_at(mut self, v: impl Into<String>) -> Self {
        self.ended_at = Some(v.into());
        self
    }
    /// Set the significance score (0.0-1.0).
    pub fn significance(mut self, v: f32) -> Self {
        self.significance = Some(v);
        self
    }
    /// Set the episode outcome.
    pub fn outcome(mut self, v: impl Into<String>) -> Self {
        self.outcome = Some(v.into());
        self
    }
    /// Set the data source.
    pub fn source_id(mut self, v: impl Into<String>) -> Self {
        self.source_id = Some(v.into());
        self
    }
    /// Set the event IDs that make up this episode.
    pub fn event_ids(mut self, v: Vec<String>) -> Self {
        self.event_ids = v;
        self
    }
    /// Set the session IDs this episode was derived from.
    pub fn session_ids(mut self, v: Vec<String>) -> Self {
        self.session_ids = v;
        self
    }
}

/// Options for searching episodes via vector similarity.
#[derive(Debug, Clone, Default)]
pub struct SearchEpisodesOptions {
    /// Owner whose episodes to search (required).
    pub user_id: String,
    /// Only include episodes started at or after this timestamp (ISO 8601).
    pub since: Option<String>,
    /// Only include episodes started before this timestamp (ISO 8601).
    pub until: Option<String>,
    /// Minimum significance threshold (0.0-1.0).
    pub min_significance: Option<f32>,
    /// Maximum number of results.
    pub limit: Option<usize>,
}

impl SearchEpisodesOptions {
    /// Create new search-episodes options with the given user ID.
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            ..Default::default()
        }
    }
    /// Only include episodes started at or after this timestamp.
    pub fn since(mut self, v: impl Into<String>) -> Self {
        self.since = Some(v.into());
        self
    }
    /// Only include episodes started before this timestamp.
    pub fn until(mut self, v: impl Into<String>) -> Self {
        self.until = Some(v.into());
        self
    }
    /// Set the minimum significance threshold.
    pub fn min_significance(mut self, v: f32) -> Self {
        self.min_significance = Some(v);
        self
    }
    /// Set the maximum number of results.
    pub fn limit(mut self, v: usize) -> Self {
        self.limit = Some(v);
        self
    }
}

/// Options for listing episodes without vector search (filtered retrieval with pagination).
#[derive(Debug, Clone, Default)]
pub struct ListEpisodesOptions {
    /// Owner whose episodes to list (required).
    pub user_id: String,
    /// Only include episodes started at or after this timestamp (ISO 8601).
    pub since: Option<String>,
    /// Only include episodes started before this timestamp (ISO 8601).
    pub until: Option<String>,
    /// Minimum significance threshold (0.0-1.0).
    pub min_significance: Option<f32>,
    /// Maximum number of results.
    pub limit: Option<usize>,
    /// Number of episodes to skip (for pagination).
    pub offset: Option<usize>,
}

impl ListEpisodesOptions {
    /// Create new list-episodes options with the given user ID.
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            ..Default::default()
        }
    }
    /// Only include episodes started at or after this timestamp.
    pub fn since(mut self, v: impl Into<String>) -> Self {
        self.since = Some(v.into());
        self
    }
    /// Only include episodes started before this timestamp.
    pub fn until(mut self, v: impl Into<String>) -> Self {
        self.until = Some(v.into());
        self
    }
    /// Set the minimum significance threshold.
    pub fn min_significance(mut self, v: f32) -> Self {
        self.min_significance = Some(v);
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

/// Options for getting the original events (messages) of an episode, with pagination.
#[derive(Debug, Clone, Default)]
pub struct EpisodeMessagesOptions {
    /// Maximum number of events to return.
    pub limit: Option<usize>,
    /// Number of events to skip (for pagination).
    pub offset: Option<usize>,
}

impl EpisodeMessagesOptions {
    /// Create new episode-messages options with defaults.
    pub fn new() -> Self {
        Self::default()
    }
    /// Set the maximum number of events to return.
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

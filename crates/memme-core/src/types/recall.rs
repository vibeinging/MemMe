use serde::{Deserialize, Serialize};

/// A cross-layer association between any two items.
#[allow(dead_code)] // planned API: cross-layer associations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Association {
    pub assoc_id: String,
    pub from_id: String,
    pub from_layer: String, // "event", "episode", "memory", "entity", "identity"
    pub to_id: String,
    pub to_layer: String,
    pub assoc_type: String, // "caused", "reminds_of", "contradicts", "supports", "temporal_next"
    pub strength: f32,
    pub created_at: String,
}

/// A record of a recall (retrieval) operation.
#[allow(dead_code)] // planned API: recall tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallRecord {
    pub recall_id: String,
    pub query: String,
    pub timestamp: String,
    pub source_id: Option<String>,
    pub user_id: String,
    pub results: Option<serde_json::Value>, // [{layer, id, score, used}]
    pub feedback: Option<String>,           // "helpful", "irrelevant", "outdated"
}

/// A unified recall result across all layers.
#[allow(dead_code)] // planned API: multi-layer recall
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    pub memories: Vec<crate::types::MemoryResult>,
    pub episodes: Vec<super::episode::Episode>,
    pub identity_traits: Vec<super::identity::IdentityTrait>,
    pub graph: Option<crate::types::GraphSearchResult>,
    pub recall_id: String,
}

/// Options for a multi-layer recall query.
#[allow(dead_code)] // planned API: multi-layer recall
#[derive(Debug, Clone)]
pub struct RecallOptions {
    pub user_id: String,
    pub limit: Option<usize>,
    pub include_episodes: bool,
    pub include_identity: bool,
    pub include_graph: bool,
    pub source_id: Option<String>,
}

impl Default for RecallOptions {
    fn default() -> Self {
        Self {
            user_id: String::new(),
            limit: None,
            include_episodes: true,
            include_identity: true,
            include_graph: true,
            source_id: None,
        }
    }
}

#[allow(dead_code)] // planned API: multi-layer recall
impl RecallOptions {
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            ..Default::default()
        }
    }
    pub fn limit(mut self, v: usize) -> Self {
        self.limit = Some(v);
        self
    }
    pub fn source_id(mut self, v: impl Into<String>) -> Self {
        self.source_id = Some(v.into());
        self
    }
    pub fn without_episodes(mut self) -> Self {
        self.include_episodes = false;
        self
    }
    pub fn without_identity(mut self) -> Self {
        self.include_identity = false;
        self
    }
    pub fn without_graph(mut self) -> Self {
        self.include_graph = false;
        self
    }
}

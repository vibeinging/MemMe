use crate::error::MemoryError;
use crate::types::PruningStrategy;
#[cfg(feature = "webhooks")]
use crate::webhook::WebhookConfig;

/// Configuration for the memory store.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Path to the DuckDB database file (e.g. "memory.duckdb").
    /// Use ":memory:" for an in-memory database.
    pub db_path: String,

    /// Collection name used as table name prefix / index suffix.
    pub collection_name: String,

    /// Dimensionality of embedding vectors (e.g. 384 for MiniLM).
    pub embedding_dims: usize,

    /// Cosine distance threshold for deduplication.
    /// Memories with distance below this are considered duplicates.
    /// Default: 0.15
    pub dedup_threshold: f32,

    /// Default number of results returned by search/list.
    /// Default: 10
    pub default_limit: usize,

    /// Optional custom system prompt for fact extraction.
    /// When set, replaces the default fact retrieval system prompt.
    pub custom_fact_extraction_prompt: Option<String>,

    /// Optional custom system prompt for the update-memory operation.
    /// When set, replaces the default update-memory system prompt.
    pub custom_update_memory_prompt: Option<String>,

    /// Whether to automatically extract and search knowledge graph alongside vector memories.
    /// Requires an LLM provider to be configured.
    pub enable_graph: bool,

    /// How many hops the entity spreading activation traverses in the knowledge graph.
    /// depth=1: direct neighbors only. depth=2: friends-of-friends. Default: 2.
    pub graph_spreading_depth: usize,

    /// Inclusion prompt: guides what types of information TO extract from conversations.
    /// Example: "Extract work-related tasks, deadlines, and project names"
    pub inclusion_prompt: Option<String>,

    /// Exclusion prompt: guides what types of information to IGNORE.
    /// Example: "Do not extract financial details, passwords, or personal ID numbers"
    pub exclusion_prompt: Option<String>,

    /// Custom categories for auto-tagging. Each entry is (name, description).
    /// When set, the LLM will automatically assign categories from this list to new memories.
    /// Example: vec![("work", "Work-related tasks"), ("personal", "Personal notes")]
    pub custom_categories: Option<Vec<(String, String)>>,

    /// Webhook configurations for event notifications.
    #[cfg(feature = "webhooks")]
    pub webhooks: Option<Vec<WebhookConfig>>,

    /// Enable Ebbinghaus forgetting curve for memory decay.
    /// When enabled, search results are scored with retention factor
    /// and accessed memories get stability reinforcement.
    /// Default: true.
    pub enable_forgetting_curve: bool,

    /// Weight of retention in search scoring (0.0-1.0).
    /// final_score = similarity × (retention_weight × retention + (1 - retention_weight) × importance)
    /// Default: 0.7
    pub retention_weight: f32,

    /// Growth factor for stability reinforcement on access.
    /// new_stability = stability × (1 + growth_factor × (1 - retention))
    /// Default: 2.5
    pub stability_growth_factor: f32,

    /// Minimum retention threshold for auto-pruning in consolidate().
    /// Memories with retention below this AND older than 30 days may be pruned.
    /// Default: 0.05
    pub prune_retention_threshold: f32,

    /// Maximum number of memories per user. When exceeded, auto-pruning kicks in.
    pub max_memories_per_user: Option<usize>,

    /// Maximum database size in megabytes.
    pub max_db_size_mb: Option<usize>,

    /// Strategy for pruning when limits are exceeded.
    pub pruning_strategy: PruningStrategy,

    /// Whether to auto-prune on add() when over limit.
    pub auto_prune: bool,

    /// Power/battery configuration for mobile-aware processing.
    pub power_config: Option<PowerConfig>,

    /// Maximum number of texts per embedding API call.
    /// Some providers limit batch size (e.g., some allow only 1-10).
    /// Default: 10
    pub embed_batch_size: usize,

    /// Number of unprocessed events in a session before auto-triggering compact.
    /// Set to 0 to disable auto-compact. Default: 20.
    pub compact_threshold: usize,

    /// Minimum hours between meditation runs for the same user.
    /// Set to 0 to disable cooldown. Default: 1.
    pub meditation_cooldown_hours: u64,

    /// Maximum number of episodes to process per meditation run.
    /// Set to 0 to use the internal cap (100). Default: 20.
    pub meditation_batch_size: usize,

    /// Minimum significance threshold for episodes to be processed by meditation.
    /// Episodes below this threshold are skipped. Default: 0.0 (process all).
    pub meditation_min_significance: f32,

    /// Minimum estimated token count before using LLM for compact.
    /// Sessions with total content below this threshold use fallback (no LLM calls).
    /// Token estimation: content_bytes / 4 + 10 per event.
    /// Set to 0 to always use LLM. Default: 200.
    pub compact_fallback_token_threshold: usize,

    /// RRF weight for the vector search channel. Default: 0.5.
    pub rrf_vector_weight: f64,

    /// RRF weight for the BM25/FTS search channel. Default: 0.3.
    pub rrf_fts_weight: f64,

    /// RRF weight for the entity spreading activation channel. Default: 0.2.
    pub rrf_entity_weight: f64,

    /// RRF smoothing parameter k. Lower values give more weight to top-ranked results.
    /// Default: 30 (original RRF paper uses 60, but smaller k is better for our scale).
    pub rrf_k: usize,

    /// Multiplier for candidate retrieval in each channel.
    /// Each channel retrieves `limit * rrf_candidate_multiplier` candidates before fusion.
    /// Default: 3.
    pub rrf_candidate_multiplier: usize,

    /// RRF weight for the temporal search channel. Default: 0.15.
    /// Activated when queries contain temporal intent ("When did...", date references).
    pub rrf_temporal_weight: f64,

    /// Maximum tokens for LLM generation calls.
    /// Reasoning models (gpt-5, kimi-k2.5) need 8000-16384 due to thinking tokens.
    /// Default: 2048.
    pub llm_max_tokens: usize,

    /// Temperature for LLM generation calls.
    /// `None` omits the parameter (required by reasoning models like o1, kimi-k2.5).
    /// Default: Some(0.1).
    pub llm_temperature: Option<f32>,

    /// Interval in seconds for time-based deferred write flush.
    /// When > 0, read operations (search, get_trace) will flush the deferred write
    /// queue if this many seconds have elapsed since the last flush, even if the
    /// queue has not reached the 500-item cap.
    /// Set to 0 to disable time-based flushing (cap-only mode).
    /// Default: 30.
    pub deferred_flush_interval_secs: u64,

    /// Score multiplier for Granular resolution memories in forgetting curve scoring.
    /// Default: 1.0 (no adjustment).
    pub resolution_weight_granular: f32,

    /// Score multiplier for Narrative resolution memories in forgetting curve scoring.
    /// Narrative memories (episode summaries) are semantically broad and match many queries,
    /// so a slight penalty reduces noise. Default: 0.85.
    pub resolution_weight_narrative: f32,

    /// Score multiplier for Identity resolution memories in forgetting curve scoring.
    /// Identity traits are the coarsest grain, penalized more. Default: 0.7.
    pub resolution_weight_identity: f32,

    /// Alpha for adaptive RRF weight scaling (0.0-1.0).
    /// 0.0 = fixed weights (current behavior), 1.0 = fully adaptive.
    /// When > 0, each channel's weight is scaled by its confidence score
    /// before RRF fusion. Default: 0.0 (disabled).
    pub adaptive_rrf_alpha: f32,

    /// Enable cross-encoder reranking after RRF fusion. Default: false.
    /// When true and a reranker is configured, search() will fetch more candidates
    /// and re-score them with the reranker before returning.
    pub enable_rerank: bool,

    /// Multiplier for candidate retrieval when reranking is enabled.
    /// search() fetches `limit * rerank_candidate_multiplier` candidates for the reranker.
    /// Default: 3.
    pub rerank_candidate_multiplier: usize,
}

/// Configuration for battery-aware processing.
#[derive(Debug, Clone)]
pub struct PowerConfig {
    /// Battery level threshold for full processing power (default: 0.5)
    pub full_power_threshold: f32,
    /// Battery level threshold for power-saving mode (default: 0.2)
    pub power_save_threshold: f32,
    /// If true, defer operations when battery is critical (below power_save_threshold)
    pub defer_when_critical: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            db_path: "memory.duckdb".to_string(),
            collection_name: "default".to_string(),
            embedding_dims: 384,
            dedup_threshold: 0.15,
            default_limit: 10,
            custom_fact_extraction_prompt: None,
            custom_update_memory_prompt: None,
            enable_graph: true,
            graph_spreading_depth: 2,
            inclusion_prompt: None,
            exclusion_prompt: None,
            custom_categories: None,
            #[cfg(feature = "webhooks")]
            webhooks: None,
            enable_forgetting_curve: true,
            retention_weight: 0.7,
            stability_growth_factor: 2.5,
            prune_retention_threshold: 0.05,
            max_memories_per_user: None,
            max_db_size_mb: None,
            pruning_strategy: PruningStrategy::default(),
            auto_prune: false,
            power_config: None,
            embed_batch_size: 10,
            compact_threshold: 20,
            meditation_cooldown_hours: 1,
            meditation_batch_size: 20,
            meditation_min_significance: 0.0,
            compact_fallback_token_threshold: 200,
            rrf_vector_weight: 0.5,
            rrf_fts_weight: 0.3,
            rrf_entity_weight: 0.2,
            rrf_k: 30,
            rrf_candidate_multiplier: 3,
            rrf_temporal_weight: 0.15,
            llm_max_tokens: 2048,
            llm_temperature: Some(0.1),
            deferred_flush_interval_secs: 30,
            resolution_weight_granular: 1.0,
            resolution_weight_narrative: 0.85,
            resolution_weight_identity: 0.7,
            adaptive_rrf_alpha: 0.0,
            enable_rerank: false,
            rerank_candidate_multiplier: 3,
        }
    }
}

impl MemoryConfig {
    /// Create a new config with required fields, using defaults for the rest.
    pub fn new(db_path: impl Into<String>, embedding_dims: usize) -> Self {
        Self {
            db_path: db_path.into(),
            embedding_dims,
            ..Default::default()
        }
    }

    /// Validate all configuration values.
    pub fn validate(&self) -> Result<(), MemoryError> {
        if self.embedding_dims == 0 {
            return Err(MemoryError::Config(
                "embedding_dims must be greater than 0".to_string(),
            ));
        }
        if self.dedup_threshold <= 0.0
            || self.dedup_threshold > 1.0
            || self.dedup_threshold.is_nan()
        {
            return Err(MemoryError::Config(
                "dedup_threshold must be in the range (0.0, 1.0]".to_string(),
            ));
        }
        if self.default_limit == 0 {
            return Err(MemoryError::Config(
                "default_limit must be greater than 0".to_string(),
            ));
        }
        if self.collection_name.is_empty() {
            return Err(MemoryError::Config(
                "collection_name must not be empty".to_string(),
            ));
        }
        if !self
            .collection_name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_')
        {
            return Err(MemoryError::Config(
                "collection_name must contain only alphanumeric characters and underscores"
                    .to_string(),
            ));
        }
        if self.db_path.is_empty() {
            return Err(MemoryError::Config("db_path must not be empty".to_string()));
        }
        if self.retention_weight < 0.0 || self.retention_weight > 1.0 {
            return Err(MemoryError::Config(
                "retention_weight must be in the range [0.0, 1.0]".to_string(),
            ));
        }
        if self.stability_growth_factor < 0.0 {
            return Err(MemoryError::Config(
                "stability_growth_factor must be non-negative".to_string(),
            ));
        }
        if self.prune_retention_threshold < 0.0 || self.prune_retention_threshold > 1.0 {
            return Err(MemoryError::Config(
                "prune_retention_threshold must be in the range [0.0, 1.0]".to_string(),
            ));
        }
        // Validate custom_categories names
        if let Some(ref cats) = self.custom_categories {
            for (name, _) in cats {
                if name.is_empty() {
                    return Err(MemoryError::Config(
                        "Category name must not be empty".to_string(),
                    ));
                }
                if !name
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
                {
                    return Err(MemoryError::Config(format!(
                        "Category name '{}' contains invalid characters (use alphanumeric, _ or -)",
                        name
                    )));
                }
            }
        }
        if self.meditation_min_significance < 0.0 || self.meditation_min_significance > 1.0 {
            return Err(MemoryError::Config(
                "meditation_min_significance must be in the range [0.0, 1.0]".to_string(),
            ));
        }
        // Prevent division-by-zero in RRF and candidate retrieval
        if self.rrf_k == 0 {
            return Err(MemoryError::Config("rrf_k must be > 0".to_string()));
        }
        if self.rrf_candidate_multiplier == 0 {
            return Err(MemoryError::Config(
                "rrf_candidate_multiplier must be > 0".to_string(),
            ));
        }
        if self.rerank_candidate_multiplier == 0 {
            return Err(MemoryError::Config(
                "rerank_candidate_multiplier must be > 0".to_string(),
            ));
        }
        for (name, val) in [
            (
                "resolution_weight_granular",
                self.resolution_weight_granular,
            ),
            (
                "resolution_weight_narrative",
                self.resolution_weight_narrative,
            ),
            (
                "resolution_weight_identity",
                self.resolution_weight_identity,
            ),
        ] {
            if !(0.0..=2.0).contains(&val) {
                return Err(MemoryError::Config(format!(
                    "{name} must be in the range [0.0, 2.0]"
                )));
            }
        }
        if !(0.0..=1.0).contains(&self.adaptive_rrf_alpha) {
            return Err(MemoryError::Config(
                "adaptive_rrf_alpha must be in the range [0.0, 1.0]".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = MemoryConfig::default();
        assert_eq!(cfg.db_path, "memory.duckdb");
        assert_eq!(cfg.collection_name, "default");
        assert_eq!(cfg.embedding_dims, 384);
        assert_eq!(cfg.dedup_threshold, 0.15);
        assert_eq!(cfg.default_limit, 10);
        assert!(cfg.inclusion_prompt.is_none());
        assert!(cfg.exclusion_prompt.is_none());
        assert!(cfg.custom_categories.is_none());
        assert!(cfg.enable_forgetting_curve);
        assert_eq!(cfg.retention_weight, 0.7);
        assert_eq!(cfg.stability_growth_factor, 2.5);
        assert_eq!(cfg.prune_retention_threshold, 0.05);
    }

    #[test]
    fn test_custom_config() {
        let cfg = MemoryConfig {
            db_path: ":memory:".into(),
            collection_name: "custom".into(),
            embedding_dims: 768,
            dedup_threshold: 0.2,
            default_limit: 20,
            inclusion_prompt: Some("Extract work tasks".into()),
            exclusion_prompt: Some("Ignore passwords".into()),
            custom_categories: Some(vec![
                ("work".into(), "Work tasks".into()),
                ("personal".into(), "Personal notes".into()),
            ]),
            ..Default::default()
        };
        assert_eq!(cfg.inclusion_prompt.as_deref(), Some("Extract work tasks"));
        assert_eq!(cfg.exclusion_prompt.as_deref(), Some("Ignore passwords"));
        assert_eq!(cfg.custom_categories.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_validate_valid() {
        let cfg = MemoryConfig::new(":memory:", 384);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_zero_dims() {
        let cfg = MemoryConfig {
            embedding_dims: 0,
            ..MemoryConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, MemoryError::Config(_)));
    }

    #[test]
    fn test_validate_bad_threshold() {
        let cfg = MemoryConfig {
            dedup_threshold: 0.0,
            ..MemoryConfig::default()
        };
        assert!(cfg.validate().is_err());

        let cfg2 = MemoryConfig {
            dedup_threshold: 1.5,
            ..MemoryConfig::default()
        };
        assert!(cfg2.validate().is_err());

        let cfg3 = MemoryConfig {
            dedup_threshold: 1.0,
            ..MemoryConfig::default()
        };
        assert!(cfg3.validate().is_ok());
    }

    #[test]
    fn test_validate_empty_collection() {
        let cfg = MemoryConfig {
            collection_name: "".into(),
            ..MemoryConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_bad_collection_chars() {
        let cfg = MemoryConfig {
            collection_name: "my-collection".into(),
            ..MemoryConfig::default()
        };
        assert!(cfg.validate().is_err());

        let cfg2 = MemoryConfig {
            collection_name: "my collection".into(),
            ..MemoryConfig::default()
        };
        assert!(cfg2.validate().is_err());

        let cfg3 = MemoryConfig {
            collection_name: "drop;table".into(),
            ..MemoryConfig::default()
        };
        assert!(cfg3.validate().is_err());

        let cfg4 = MemoryConfig {
            collection_name: "my_collection_01".into(),
            ..MemoryConfig::default()
        };
        assert!(cfg4.validate().is_ok());
    }

    #[test]
    fn test_validate_bad_category_name() {
        let cfg = MemoryConfig {
            custom_categories: Some(vec![("".into(), "Empty".into())]),
            ..MemoryConfig::default()
        };
        assert!(cfg.validate().is_err());

        let cfg2 = MemoryConfig {
            custom_categories: Some(vec![("has space".into(), "Bad".into())]),
            ..MemoryConfig::default()
        };
        assert!(cfg2.validate().is_err());

        let cfg3 = MemoryConfig {
            custom_categories: Some(vec![
                ("work-tasks".into(), "Work".into()),
                ("personal_notes".into(), "Personal".into()),
            ]),
            ..MemoryConfig::default()
        };
        assert!(cfg3.validate().is_ok());
    }
}

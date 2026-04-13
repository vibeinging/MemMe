use crate::error::MemoryError;
use crate::types::PruningStrategy;
#[cfg(feature = "webhooks")]
use crate::webhook::WebhookConfig;

/// User-facing configuration for the memory store.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub db_path: String,
    pub collection_name: String,
    pub embedding_dims: usize,
    pub enable_graph: bool,
    pub locale: String,
    pub tuning: TuningConfig,
}

/// Internal tuning parameters with sensible defaults.
#[derive(Debug, Clone)]
pub struct TuningConfig {
    pub dedup_threshold: f32,
    pub default_limit: usize,
    pub custom_fact_extraction_prompt: Option<String>,
    pub custom_update_memory_prompt: Option<String>,
    pub graph_spreading_depth: usize,
    pub inclusion_prompt: Option<String>,
    pub exclusion_prompt: Option<String>,
    pub custom_categories: Option<Vec<(String, String)>>,
    #[cfg(feature = "webhooks")]
    pub webhooks: Option<Vec<WebhookConfig>>,
    pub enable_forgetting_curve: bool,
    pub retention_weight: f32,
    pub stability_growth_factor: f32,
    pub prune_retention_threshold: f32,
    pub max_memories_per_user: Option<usize>,
    pub max_db_size_mb: Option<usize>,
    pub pruning_strategy: PruningStrategy,
    pub auto_prune: bool,
    pub power_config: Option<PowerConfig>,
    pub embed_batch_size: usize,
    pub compact_threshold: usize,
    pub meditation_cooldown_hours: u64,
    pub meditation_batch_size: usize,
    pub meditation_min_significance: f32,
    pub compact_fallback_token_threshold: usize,
    pub rrf_vector_weight: f64,
    pub rrf_fts_weight: f64,
    pub rrf_entity_weight: f64,
    pub rrf_k: usize,
    pub rrf_candidate_multiplier: usize,
    pub rrf_temporal_weight: f64,
    pub rrf_word_overlap_weight: f64,
    pub rrf_event_weight: f64,
    pub event_memory_threshold: usize,
    pub llm_max_tokens: usize,
    pub llm_temperature: Option<f32>,
    pub deferred_flush_interval_secs: u64,
    pub resolution_weight_granular: f32,
    pub resolution_weight_narrative: f32,
    pub resolution_weight_identity: f32,
    pub adaptive_rrf_alpha: f32,
    pub enable_rerank: bool,
    pub rerank_candidate_multiplier: usize,
    pub reader_pool_size: Option<usize>,
    /// Max results from the same session in final output (0 = unlimited).
    /// Ensures cross-session diversity for multi-session queries.
    pub max_per_session: usize,
    /// Number of graph-augmented results to add after initial recall.
    /// Set to 0 to disable post-recall graph augmentation.
    pub graph_augmentation_limit: usize,
}

#[derive(Debug, Clone)]
pub struct PowerConfig {
    pub full_power_threshold: f32,
    pub power_save_threshold: f32,
    pub defer_when_critical: bool,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            dedup_threshold: 0.15,
            default_limit: 10,
            custom_fact_extraction_prompt: None,
            custom_update_memory_prompt: None,
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
            compact_threshold: 0,
            meditation_cooldown_hours: 1,
            meditation_batch_size: 20,
            meditation_min_significance: 0.3,
            compact_fallback_token_threshold: 200,
            rrf_vector_weight: 0.3,
            rrf_fts_weight: 0.35,
            rrf_entity_weight: 0.2,
            rrf_k: 30,
            rrf_candidate_multiplier: 3,
            rrf_temporal_weight: 0.15,
            rrf_word_overlap_weight: 0.15,
            rrf_event_weight: 0.35,
            event_memory_threshold: 100,
            llm_max_tokens: 2048,
            llm_temperature: Some(0.1),
            deferred_flush_interval_secs: 30,
            resolution_weight_granular: 1.0,
            resolution_weight_narrative: 0.65,
            resolution_weight_identity: 0.7,
            adaptive_rrf_alpha: 0.0,
            enable_rerank: false,
            rerank_candidate_multiplier: 3,
            reader_pool_size: None,
            max_per_session: 0,
            graph_augmentation_limit: 10,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            db_path: "memory.db".to_string(),
            collection_name: "default".to_string(),
            embedding_dims: 384,
            enable_graph: true,
            locale: "auto".to_string(),
            tuning: TuningConfig::default(),
        }
    }
}

impl MemoryConfig {
    pub fn new(db_path: impl Into<String>, embedding_dims: usize) -> Self {
        Self {
            db_path: db_path.into(),
            embedding_dims,
            ..Default::default()
        }
    }

    pub fn validate(&self) -> Result<(), MemoryError> {
        if self.embedding_dims == 0 {
            return Err(MemoryError::Config(
                "embedding_dims must be greater than 0".into(),
            ));
        }
        let t = &self.tuning;
        if t.dedup_threshold <= 0.0 || t.dedup_threshold > 1.0 || t.dedup_threshold.is_nan() {
            return Err(MemoryError::Config(
                "dedup_threshold must be in the range (0.0, 1.0]".into(),
            ));
        }
        if t.default_limit == 0 {
            return Err(MemoryError::Config(
                "default_limit must be greater than 0".into(),
            ));
        }
        if self.collection_name.is_empty() {
            return Err(MemoryError::Config(
                "collection_name must not be empty".into(),
            ));
        }
        if !self
            .collection_name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_')
        {
            return Err(MemoryError::Config(
                "collection_name must contain only alphanumeric characters and underscores".into(),
            ));
        }
        if self.db_path.is_empty() {
            return Err(MemoryError::Config("db_path must not be empty".into()));
        }
        if t.retention_weight < 0.0 || t.retention_weight > 1.0 {
            return Err(MemoryError::Config(
                "retention_weight must be in the range [0.0, 1.0]".into(),
            ));
        }
        if t.stability_growth_factor < 0.0 {
            return Err(MemoryError::Config(
                "stability_growth_factor must be non-negative".into(),
            ));
        }
        if t.prune_retention_threshold < 0.0 || t.prune_retention_threshold > 1.0 {
            return Err(MemoryError::Config(
                "prune_retention_threshold must be in the range [0.0, 1.0]".into(),
            ));
        }
        if let Some(ref cats) = t.custom_categories {
            for (name, _) in cats {
                if name.is_empty() {
                    return Err(MemoryError::Config(
                        "Category name must not be empty".into(),
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
        if t.meditation_min_significance < 0.0 || t.meditation_min_significance > 1.0 {
            return Err(MemoryError::Config(
                "meditation_min_significance must be in the range [0.0, 1.0]".into(),
            ));
        }
        if t.rrf_k == 0 {
            return Err(MemoryError::Config("rrf_k must be > 0".into()));
        }
        if t.rrf_candidate_multiplier == 0 {
            return Err(MemoryError::Config(
                "rrf_candidate_multiplier must be > 0".into(),
            ));
        }
        if t.rerank_candidate_multiplier == 0 {
            return Err(MemoryError::Config(
                "rerank_candidate_multiplier must be > 0".into(),
            ));
        }
        for (name, val) in [
            ("resolution_weight_granular", t.resolution_weight_granular),
            ("resolution_weight_narrative", t.resolution_weight_narrative),
            ("resolution_weight_identity", t.resolution_weight_identity),
        ] {
            if !(0.0..=2.0).contains(&val) {
                return Err(MemoryError::Config(format!(
                    "{name} must be in the range [0.0, 2.0]"
                )));
            }
        }
        if !(0.0..=1.0).contains(&t.adaptive_rrf_alpha) {
            return Err(MemoryError::Config(
                "adaptive_rrf_alpha must be in the range [0.0, 1.0]".into(),
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
        assert_eq!(cfg.db_path, "memory.db");
        assert_eq!(cfg.collection_name, "default");
        assert_eq!(cfg.embedding_dims, 384);
        assert_eq!(cfg.locale, "auto");
        assert_eq!(cfg.tuning.dedup_threshold, 0.15);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_zero_dims() {
        let cfg = MemoryConfig {
            embedding_dims: 0,
            ..MemoryConfig::default()
        };
        assert!(matches!(
            cfg.validate().unwrap_err(),
            MemoryError::Config(_)
        ));
    }

    #[test]
    fn test_validate_bad_threshold() {
        let cfg = MemoryConfig {
            tuning: TuningConfig {
                dedup_threshold: 0.0,
                ..Default::default()
            },
            ..MemoryConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_empty_collection() {
        assert!(MemoryConfig {
            collection_name: "".into(),
            ..MemoryConfig::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn test_validate_bad_collection_chars() {
        assert!(MemoryConfig {
            collection_name: "my-collection".into(),
            ..MemoryConfig::default()
        }
        .validate()
        .is_err());
        assert!(MemoryConfig {
            collection_name: "my_collection_01".into(),
            ..MemoryConfig::default()
        }
        .validate()
        .is_ok());
    }
}

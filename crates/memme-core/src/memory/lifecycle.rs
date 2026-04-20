use crate::error::{MemoryError, Result};
use crate::types::*;

use super::helpers::row_to_result;

impl super::MemoryStore {
    /// Apply time-based importance decay to all memories for a user.
    /// Memories that haven't been accessed recently lose importance.
    /// This is designed for edge/offline scenarios where background
    /// maintenance runs periodically.
    ///
    /// `decay_rate`: how much importance to subtract per day since last access (e.g., 0.01)
    /// `min_importance`: minimum importance threshold -- memories below this can be auto-deleted
    /// `delete_below`: if true, delete memories that fall below min_importance
    pub fn consolidate(
        &self,
        user_id: &str,
        decay_rate: f32,
        min_importance: f32,
        delete_below: bool,
    ) -> Result<ConsolidateResult> {
        let mut result =
            self.storage
                .consolidate(user_id, decay_rate, min_importance, delete_below)?;

        if self.config.tuning.enable_forgetting_curve {
            let pruned = self.storage.consolidate_forgetting_curve(
                user_id,
                self.config.tuning.prune_retention_threshold,
                30.0,
            )?;
            result.deleted_count += pruned;
        }

        Ok(result)
    }

    /// Clean up expired memories (past their expiration_date).
    pub fn cleanup_expired(&self) -> Result<u64> {
        self.storage.cleanup_expired()
    }

    /// Manually prune memories for a user. Returns the number of deleted memories.
    #[allow(dead_code)]
    pub(crate) fn prune(&self, user_id: &str, count: usize) -> Result<u64> {
        self.storage
            .prune_memories(user_id, &self.config.tuning.pruning_strategy, count)
    }

    /// Batch update multiple traces. Returns updated results.
    /// Skips immutable traces (logs warning, doesn't error).
    #[allow(dead_code)]
    pub(crate) fn batch_update_traces(
        &self,
        updates: &[(String, String)],
    ) -> Result<Vec<MemoryResult>> {
        let mut results = Vec::new();
        for (id, content) in updates {
            // Check immutable — skip if immutable
            match self.storage.check_immutable(id) {
                Err(MemoryError::ImmutableMemory(ref mid)) => {
                    tracing::warn!(id = %mid, "Skipping batch update: memory is immutable");
                    continue;
                }
                Err(e) => return Err(e),
                Ok(()) => {}
            }
            match self.update_trace(id, content, None) {
                Ok(r) => results.push(r),
                Err(MemoryError::NotFound(ref nid)) => {
                    tracing::warn!(id = %nid, "Skipping batch update: memory not found");
                }
                Err(e) => return Err(e),
            }
        }
        Ok(results)
    }

    /// Batch delete multiple traces by ID. Returns count of successfully deleted.
    /// Skips immutable traces (logs warning, doesn't error).
    #[allow(dead_code)]
    pub(crate) fn batch_delete_traces(&self, ids: &[String]) -> Result<u64> {
        let mut count = 0u64;
        for id in ids {
            // Check immutable — skip if immutable
            match self.storage.check_immutable(id) {
                Err(MemoryError::ImmutableMemory(ref mid)) => {
                    tracing::warn!(id = %mid, "Skipping batch delete: memory is immutable");
                    continue;
                }
                Err(e) => return Err(e),
                Ok(()) => {}
            }
            match self.delete_trace(id) {
                Ok(()) => count += 1,
                Err(MemoryError::NotFound(ref nid)) => {
                    tracing::warn!(id = %nid, "Skipping batch delete: memory not found");
                }
                Err(e) => return Err(e),
            }
        }
        Ok(count)
    }

    /// List traces with optional filters.
    pub fn list_traces(&self, options: ListOptions) -> Result<Vec<MemoryResult>> {
        let limit = options.limit.unwrap_or(self.config.tuning.default_limit);
        let rows = self.storage.list_memories(
            &options.user_id,
            options.agent_id.as_deref(),
            options.run_id.as_deref(),
            options.app_id.as_deref(),
            options.filter.as_ref(),
            limit,
            options.min_importance,
            options.since.as_deref(),
            options.pinned_only,
        )?;

        Ok(rows.into_iter().map(row_to_result).collect())
    }

    /// Get count of traces for a user.
    pub fn count_traces(&self, user_id: &str) -> Result<usize> {
        self.storage.count_user_memories(user_id)
    }

    /// Pin or unpin a memory trace. Pinned memories are exempt from forgetting curve decay
    /// and are prioritized in HOT-tier queries.
    pub fn pin_trace(&self, memory_id: &str, pinned: bool) -> Result<()> {
        let sql = "UPDATE memories SET pinned = ?1 WHERE id = ?2";
        self.storage.backend.execute(
            sql,
            &[
                crate::types::SqlParam::Int(if pinned { 1 } else { 0 }),
                crate::types::SqlParam::Text(memory_id.to_string()),
            ],
        )?;
        Ok(())
    }

    /// List all pinned memories for a user.
    pub fn list_pinned_traces(&self, user_id: &str) -> Result<Vec<MemoryResult>> {
        self.list_traces(
            ListOptions::new(user_id)
                .pinned_only()
                .limit(100),
        )
    }

    /// Recall old memories for nostalgia / proactive bubbles ("还记得那天...").
    /// Returns random memories older than `min_age_days` with importance >= `min_importance`.
    pub fn recall_nostalgia(
        &self,
        user_id: &str,
        min_age_days: u32,
        min_importance: f32,
        limit: usize,
    ) -> Result<Vec<MemoryResult>> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(min_age_days as i64);
        let cutoff_str = cutoff.format("%Y-%m-%dT%H:%M:%S").to_string();

        // Use storage query directly for RANDOM() ordering
        let sql = format!(
            "SELECT {} FROM memories WHERE user_id = ?1 AND created_at <= ?2 AND importance >= ?3 ORDER BY RANDOM() LIMIT ?4",
            crate::storage::query::memory_select_cols(None, "")
        );
        let params = vec![
            crate::types::SqlParam::Text(user_id.to_string()),
            crate::types::SqlParam::Text(cutoff_str),
            crate::types::SqlParam::Float(min_importance as f64),
            crate::types::SqlParam::Int(limit as i64),
        ];
        let rows = self.storage.backend.query_read(&sql, &params, crate::storage::query::map_memory_row)?;
        Ok(rows.into_iter().map(super::helpers::row_to_result).collect())
    }

    /// Get the approximate database size in bytes.
    pub fn db_size_bytes(&self) -> Result<u64> {
        self.storage.db_size_bytes()
    }
}

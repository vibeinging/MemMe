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
        )?;

        Ok(rows.into_iter().map(row_to_result).collect())
    }

    /// Get count of traces for a user.
    pub fn count_traces(&self, user_id: &str) -> Result<usize> {
        self.storage.count_user_memories(user_id)
    }

    /// Get the approximate database size in bytes.
    pub fn db_size_bytes(&self) -> Result<u64> {
        self.storage.db_size_bytes()
    }
}

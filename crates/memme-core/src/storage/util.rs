use tracing::debug;

use crate::error::{MemoryError, Result};
use crate::types::SqlParam;

use super::Storage;

/// Convert an `Option<impl AsRef<str>>` to a SQL parameter value.
/// `Some("text")` → `SqlParam::Text`, `None` → `SqlParam::Null`.
pub(crate) fn opt_text(opt: Option<impl AsRef<str>>) -> SqlParam {
    match opt {
        Some(s) => SqlParam::Text(s.as_ref().to_string()),
        None => SqlParam::Null,
    }
}

impl Storage {
    /// Execute SQL that might fail (e.g. if MemMe-DB extension is not loaded),
    /// logging the error but not propagating it.
    pub(crate) fn execute_ignore_error(&self, sql: &str) {
        self.backend.execute_batch_ignore(sql);
    }

    // ── Helper: format a Vec<f32> as a SQL array literal ──

    /// Convert an embedding vector to a database-compatible array literal string.
    /// Delegates to the configured SQL dialect.
    pub(crate) fn format_embedding(&self, embedding: &[f32], dims: usize) -> Result<String> {
        self.dialect().format_embedding_literal(embedding, dims)
    }

    /// Format categories as a SQL list literal string.
    /// Delegates validation and formatting to the configured SQL dialect.
    pub(crate) fn format_categories(&self, categories: Option<&[String]>) -> Result<String> {
        match categories {
            Some(cats) if !cats.is_empty() => self.dialect().format_categories_literal(cats),
            _ => Ok("NULL".to_string()),
        }
    }

    /// Parse a list string like `[cat1, cat2]` or `["cat1","cat2"]` into a Vec<String>.
    pub(crate) fn parse_categories(raw: Option<String>) -> Option<Vec<String>> {
        let raw = raw?;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed == "[]" {
            return None;
        }
        // Parse list as `[val1, val2]` or JSON array — strip brackets and split
        let inner = trimmed.trim_start_matches('[').trim_end_matches(']');
        if inner.is_empty() {
            return None;
        }
        let cats: Vec<String> = inner
            .split(',')
            .map(|s| s.trim().trim_matches('\'').trim_matches('"').to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if cats.is_empty() {
            None
        } else {
            Some(cats)
        }
    }

    /// Check if a memory is immutable, returning an error if it is.
    pub(crate) fn check_immutable(&self, id: &str) -> Result<()> {
        let immutable = self.backend.query_one(
            "SELECT immutable FROM memories WHERE id = $1",
            &[SqlParam::Text(id.to_string())],
            |row| row.get_opt_bool(0).map(|v| v.unwrap_or(false)),
        )?;
        if immutable == Some(true) {
            return Err(MemoryError::ImmutableMemory(id.to_string()));
        }
        Ok(())
    }

    /// Count the number of memories for a given user.
    pub(crate) fn count_user_memories(&self, user_id: &str) -> Result<usize> {
        let count = self.backend.query_count(
            "SELECT COUNT(*) FROM memories WHERE user_id = $1",
            &[SqlParam::Text(user_id.to_string())],
        )?;
        Ok(count as usize)
    }

    /// Get approximate database size in bytes.
    /// Uses PRAGMA database_size for on-disk databases, returns 0 for in-memory.
    pub(crate) fn db_size_bytes(&self) -> Result<u64> {
        if self.config.db_path == ":memory:" {
            // For in-memory, estimate from row count
            let count = self
                .backend
                .query_count("SELECT COUNT(*) FROM memories", &[])?;
            // Rough estimate: ~1KB per memory
            return Ok(count as u64 * 1024);
        }
        // Try file size
        match std::fs::metadata(&self.config.db_path) {
            Ok(meta) => Ok(meta.len()),
            Err(_) => Ok(0),
        }
    }

    /// Prune memories for a user using the given strategy.
    /// Returns the number of deleted memories.
    pub(crate) fn prune_memories(
        &self,
        user_id: &str,
        strategy: &crate::types::PruningStrategy,
        count_to_remove: usize,
    ) -> Result<u64> {
        if count_to_remove == 0 {
            return Ok(0);
        }

        let order_clause = match strategy {
            crate::types::PruningStrategy::LRU => "updated_at ASC",
            crate::types::PruningStrategy::Importance => "importance ASC",
            crate::types::PruningStrategy::Decay => {
                // Order by importance after decay (lower = prune first)
                "importance ASC, updated_at ASC"
            }
        };

        let sql = format!(
            "DELETE FROM memories WHERE id IN (
                SELECT id FROM memories
                WHERE user_id = $1 AND (immutable IS NULL OR immutable = false)
                ORDER BY {order_clause}
                LIMIT {count_to_remove}
            )"
        );
        let count = self
            .backend
            .execute(&sql, &[SqlParam::Text(user_id.to_string())])? as u64;
        Ok(count)
    }

    /// Delete memories past their expiration_date. Returns the number of deleted memories.
    pub(crate) fn cleanup_expired(&self) -> Result<u64> {
        let now_ts = self.dialect().current_timestamp_expr();
        let sql = format!(
            "DELETE FROM memories WHERE expiration_date IS NOT NULL AND expiration_date < {now_ts}"
        );
        let count = self.backend.execute(&sql, &[])? as u64;
        Ok(count)
    }

    /// Apply time-based importance decay to all memories for a user.
    /// Returns the number of decayed and (optionally) deleted memories, plus expired count.
    pub(crate) fn consolidate(
        &self,
        user_id: &str,
        decay_rate: f32,
        min_importance: f32,
        delete_below: bool,
    ) -> Result<crate::types::ConsolidateResult> {
        // 1. Decay importance based on days since last access (updated_at)
        let now_epoch = self
            .dialect()
            .epoch_seconds_expr(self.dialect().current_timestamp_expr());
        let updated_epoch = self.dialect().epoch_seconds_expr("updated_at");
        let greatest = self.dialect().greatest_expr(
            "0.0",
            &format!("importance - $1 * ({now_epoch} - {updated_epoch}) / 86400.0"),
        );
        let decay_sql = format!("UPDATE memories SET importance = {greatest} WHERE user_id = $2");
        let decay_params = vec![
            SqlParam::Float(decay_rate as f64),
            SqlParam::Text(user_id.to_string()),
        ];
        let decayed = self.backend.execute(&decay_sql, &decay_params)? as u64;

        // 2. Optionally delete memories below min_importance
        let deleted = if delete_below {
            let delete_sql = "DELETE FROM memories WHERE user_id = $1 AND importance < $2";
            let del_params = vec![
                SqlParam::Text(user_id.to_string()),
                SqlParam::Float(min_importance as f64),
            ];
            self.backend.execute(delete_sql, &del_params)? as u64
        } else {
            0
        };

        // 3. Cleanup expired memories
        let now_ts = self.dialect().current_timestamp_expr();
        let expired_sql = format!(
            "DELETE FROM memories WHERE expiration_date IS NOT NULL AND expiration_date < {now_ts}"
        );
        let expired_count = self.backend.execute(&expired_sql, &[])? as u64;

        Ok(crate::types::ConsolidateResult {
            decayed_count: decayed,
            deleted_count: deleted,
            expired_count,
        })
    }

    // ── Access tracking ──

    /// Increment the access_count for a memory by ID.
    pub(crate) fn increment_access_count(&self, id: &str) -> Result<()> {
        self.backend.execute(
            "UPDATE memories SET access_count = access_count + 1 WHERE id = $1",
            &[SqlParam::Text(id.to_string())],
        )?;
        Ok(())
    }

    /// Reinforce a memory's stability when it is accessed.
    /// Uses FSRS-inspired formula: new_S = S × (1 + growth_factor × (1 - R))
    /// where R is current retention computed from the power-law forgetting curve.
    pub(crate) fn reinforce_stability(&self, id: &str, growth_factor: f32) -> Result<()> {
        // Note: updated_at is NOT reset here — it reflects last content update, not access.
        // Retention is computed from updated_at, so resetting it would make R≈1 always,
        // defeating the forgetting curve.
        let now_epoch = self
            .dialect()
            .epoch_seconds_expr(self.dialect().current_timestamp_expr());
        let updated_epoch = self.dialect().epoch_seconds_expr("updated_at");
        let stability_floor = self
            .dialect()
            .greatest_expr("COALESCE(stability, 1.0)", "0.01");
        let power = self.dialect().power_expr(
            &format!("1.0 + ({now_epoch} - {updated_epoch}) / 86400.0 / (5.0 * {stability_floor})"),
            "-0.5",
        );
        let least = self.dialect().least_expr(
            "365.0",
            &format!("COALESCE(stability, 1.0) * (1.0 + $1 * (1.0 - {power}))"),
        );
        let sql = format!(
            "UPDATE memories \
            SET stability = {least}, \
            access_count = access_count + 1 \
            WHERE id = $2"
        );
        self.backend.execute(
            &sql,
            &[
                SqlParam::Float(growth_factor as f64),
                SqlParam::Text(id.to_string()),
            ],
        )?;
        Ok(())
    }

    /// Consolidation using forgetting curve: prune memories with low retention.
    /// Deletes memories where retention < threshold AND last accessed > min_age_days ago.
    pub(crate) fn consolidate_forgetting_curve(
        &self,
        user_id: &str,
        retention_threshold: f32,
        min_age_days: f32,
    ) -> Result<u64> {
        let now_epoch = self
            .dialect()
            .epoch_seconds_expr(self.dialect().current_timestamp_expr());
        let updated_epoch = self.dialect().epoch_seconds_expr("updated_at");
        let stability_floor = self
            .dialect()
            .greatest_expr("COALESCE(stability, 1.0)", "0.01");
        let power = self.dialect().power_expr(
            &format!("1.0 + ({now_epoch} - {updated_epoch}) / 86400.0 / (5.0 * {stability_floor})"),
            "-0.5",
        );
        let sql = format!(
            "DELETE FROM memories \
            WHERE user_id = $1 \
            AND {power} < $2 \
            AND ({now_epoch} - {updated_epoch}) / 86400.0 > $3"
        );
        let count = self.backend.execute(
            &sql,
            &[
                SqlParam::Text(user_id.to_string()),
                SqlParam::Float(retention_threshold as f64),
                SqlParam::Float(min_age_days as f64),
            ],
        )? as u64;
        Ok(count)
    }

    /// Incrementally insert a single memory into the memories_fts index.
    pub(crate) fn fts_insert_memory(&self, id: &str, content: &str) {
        let _ = self.backend.execute(
            "INSERT OR REPLACE INTO memories_fts(id, content) VALUES ($1, $2)",
            &[
                SqlParam::Text(id.to_string()),
                SqlParam::Text(content.to_string()),
            ],
        );
    }

    /// Incrementally insert a single event into the events_fts index.
    pub(crate) fn fts_insert_event(&self, event_id: &str, purified_content: &str) {
        let _ = self.backend.execute(
            "INSERT OR REPLACE INTO events_fts(event_id, purified_content) VALUES ($1, $2)",
            &[
                SqlParam::Text(event_id.to_string()),
                SqlParam::Text(purified_content.to_string()),
            ],
        );
    }

    /// Incrementally delete a memory from the memories_fts index.
    pub(crate) fn fts_delete_memory(&self, id: &str) {
        let _ = self.backend.execute(
            "INSERT INTO memories_fts(memories_fts, id, content) VALUES ('delete', $1, (SELECT content FROM memories WHERE id = $1))",
            &[SqlParam::Text(id.to_string())],
        );
    }

    /// Create or rebuild the FTS index on the memories table.
    pub(crate) fn create_fts_index(&self) -> Result<()> {
        // Ensure FTS extension is loaded
        for sql in self.dialect().load_fts_extension_sql() {
            self.execute_ignore_error(sql);
        }

        // Create FTS index with overwrite to handle existing index
        let fts_sql = self
            .dialect()
            .create_fts_index_sql("memories", "id", &["content"]);
        self.backend.execute_batch(&fts_sql)?;
        debug!("Created/rebuilt FTS index on memories table");
        Ok(())
    }

    /// Create or rebuild the FTS index on the episodes table (title + summary).
    #[allow(dead_code)] // planned API: episode FTS indexing
    pub(crate) fn create_fts_index_episodes(&self) -> Result<()> {
        for sql in self.dialect().load_fts_extension_sql() {
            self.execute_ignore_error(sql);
        }
        let fts_sql =
            self.dialect()
                .create_fts_index_sql("episodes", "episode_id", &["title", "summary"]);
        self.backend.execute_batch(&fts_sql)?;
        debug!("Created/rebuilt FTS index on episodes table");
        Ok(())
    }

    /// Create or rebuild the FTS index on the events table (purified_content).
    /// Note: This should only be called after compact() has populated purified_content.
    pub(crate) fn create_fts_index_events(&self) -> Result<()> {
        for sql in self.dialect().load_fts_extension_sql() {
            self.execute_ignore_error(sql);
        }
        let fts_sql =
            self.dialect()
                .create_fts_index_sql("events", "event_id", &["purified_content"]);
        // FTS on purified_content (not raw content) for better keyword matching
        self.backend.execute_batch(&fts_sql)?;
        debug!("Created/rebuilt FTS index on events.purified_content");
        Ok(())
    }
}

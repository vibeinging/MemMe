use duckdb::params;
use tracing::debug;

use crate::error::{MemoryError, Result};

use super::Storage;

/// Convert an `Option<impl AsRef<str>>` to a DuckDB parameter value.
/// `Some("text")` → `Value::Text`, `None` → `Value::Null`.
pub(crate) fn opt_text(opt: Option<impl AsRef<str>>) -> duckdb::types::Value {
    match opt {
        Some(s) => duckdb::types::Value::Text(s.as_ref().to_string()),
        None => duckdb::types::Value::Null,
    }
}

impl Storage {
    /// Execute SQL that might fail (e.g. if MemMe-DB extension is not loaded),
    /// logging the error but not propagating it.
    pub(crate) fn execute_ignore_error(&self, sql: &str) {
        let conn = self.write_conn();
        if let Err(e) = conn.execute_batch(sql) {
            tracing::warn!("SQL ignored (extension may not be loaded): {e}");
            debug!("Failed SQL: {sql}");
        }
    }

    // ── Helper: format a Vec<f32> as DuckDB array literal ──

    /// Convert an embedding vector to a DuckDB-compatible array literal string.
    /// Example output: `[0.1, 0.2, 0.3]::FLOAT[384]`
    pub(crate) fn format_embedding(embedding: &[f32], dims: usize) -> Result<String> {
        if embedding.iter().any(|v| v.is_nan() || v.is_infinite()) {
            return Err(MemoryError::Config(
                "embedding contains NaN or Infinity".into(),
            ));
        }
        let values: Vec<String> = embedding.iter().map(|v| format!("{v}")).collect();
        Ok(format!("[{}]::FLOAT[{dims}]", values.join(",")))
    }

    /// Format categories as a DuckDB list literal string, e.g. `['cat1', 'cat2']`
    ///
    /// Category names are validated to only allow alphanumeric, underscore,
    /// hyphen, and space characters.
    pub(crate) fn format_categories(
        categories: Option<&[String]>,
    ) -> std::result::Result<String, MemoryError> {
        match categories {
            Some(cats) if !cats.is_empty() => {
                for cat in cats {
                    if !cat
                        .chars()
                        .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == ' ')
                    {
                        return Err(MemoryError::Config(format!(
                            "Invalid category name '{}': only alphanumeric, underscore, hyphen, and space allowed",
                            cat
                        )));
                    }
                    if cat.is_empty() {
                        return Err(MemoryError::Config("Category name cannot be empty".into()));
                    }
                }
                let items: Vec<String> = cats
                    .iter()
                    .map(|c| format!("'{}'", c.replace('\'', "''")))
                    .collect();
                Ok(format!("[{}]", items.join(", ")))
            }
            _ => Ok("NULL".to_string()),
        }
    }

    /// Parse a DuckDB list string like `[cat1, cat2]` into a Vec<String>.
    pub(crate) fn parse_categories(raw: Option<String>) -> Option<Vec<String>> {
        let raw = raw?;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed == "[]" {
            return None;
        }
        // DuckDB returns list as `[val1, val2]` — strip brackets and split
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
        let conn = self.read_conn();
        let mut stmt = conn.prepare("SELECT immutable FROM memories WHERE id = $1")?;
        let mut rows = stmt.query_map(params![id], |row| row.get::<_, Option<bool>>(0))?;
        if let Some(row) = rows.next() {
            let immutable = row?.unwrap_or(false);
            if immutable {
                return Err(MemoryError::ImmutableMemory(id.to_string()));
            }
        }
        Ok(())
    }

    /// Count the number of memories for a given user.
    pub(crate) fn count_user_memories(&self, user_id: &str) -> Result<usize> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare("SELECT COUNT(*) FROM memories WHERE user_id = $1")?;
        let count: i64 = stmt
            .query_map(params![user_id], |row| row.get(0))?
            .next()
            .expect("COUNT always returns a row")?;
        Ok(count as usize)
    }

    /// Get approximate database size in bytes.
    /// Uses PRAGMA database_size for on-disk databases, returns 0 for in-memory.
    pub(crate) fn db_size_bytes(&self) -> Result<u64> {
        if self.config.db_path == ":memory:" {
            // For in-memory, estimate from row count
            let conn = self.read_conn();
            let mut stmt = conn.prepare("SELECT COUNT(*) FROM memories")?;
            let count: i64 = stmt
                .query_map([], |row| row.get(0))?
                .next()
                .expect("COUNT always returns a row")?;
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
        let conn = self.write_conn();
        let count = conn.execute(&sql, params![user_id])? as u64;
        Ok(count)
    }

    /// Delete memories past their expiration_date. Returns the number of deleted memories.
    pub(crate) fn cleanup_expired(&self) -> Result<u64> {
        let conn = self.write_conn();
        let sql = "DELETE FROM memories WHERE expiration_date IS NOT NULL AND expiration_date < now()::TIMESTAMP";
        let count = conn.execute(sql, [])? as u64;
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
        // Use a single write connection for all operations to avoid Mutex re-entry deadlock.
        let conn = self.write_conn();

        // 1. Decay importance based on days since last access (updated_at)
        let decay_sql = r#"UPDATE memories
            SET importance = GREATEST(0.0, importance - $1 * (epoch(now()::TIMESTAMP) - epoch(updated_at)) / 86400.0)
            WHERE user_id = $2"#;
        let decayed = conn.execute(decay_sql, params![decay_rate as f64, user_id])?;

        // 2. Optionally delete memories below min_importance
        let deleted = if delete_below {
            let delete_sql = "DELETE FROM memories WHERE user_id = $1 AND importance < $2";
            conn.execute(delete_sql, params![user_id, min_importance as f64])? as u64
        } else {
            0
        };

        // 3. Cleanup expired memories (inline to avoid re-acquiring write_conn)
        let expired_sql = "DELETE FROM memories WHERE expiration_date IS NOT NULL AND expiration_date < now()::TIMESTAMP";
        let expired_count = conn.execute(expired_sql, [])? as u64;

        Ok(crate::types::ConsolidateResult {
            decayed_count: decayed as u64,
            deleted_count: deleted,
            expired_count,
        })
    }

    // ── Access tracking ──

    /// Increment the access_count for a memory by ID.
    pub(crate) fn increment_access_count(&self, id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            "UPDATE memories SET access_count = access_count + 1 WHERE id = $1",
            params![id],
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
        let sql = r#"UPDATE memories
            SET stability = LEAST(365.0,
                COALESCE(stability, 1.0) * (1.0 + $1 *
                    (1.0 - POWER(
                        1.0 + (epoch(now()::TIMESTAMP) - epoch(updated_at)) / 86400.0
                              / (5.0 * GREATEST(COALESCE(stability, 1.0), 0.01)),
                        -0.5
                    ))
                )
            ),
            access_count = access_count + 1
            WHERE id = $2"#;
        let conn = self.write_conn();
        conn.execute(sql, params![growth_factor as f64, id])?;
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
        let sql = r#"DELETE FROM memories
            WHERE user_id = $1
            AND POWER(
                1.0 + (epoch(now()::TIMESTAMP) - epoch(updated_at)) / 86400.0
                      / (5.0 * GREATEST(COALESCE(stability, 1.0), 0.01)),
                -0.5
            ) < $2
            AND (epoch(now()::TIMESTAMP) - epoch(updated_at)) / 86400.0 > $3"#;
        let conn = self.write_conn();
        let count = conn.execute(
            sql,
            params![user_id, retention_threshold as f64, min_age_days as f64],
        )? as u64;
        Ok(count)
    }

    /// Create or rebuild the FTS index on the memories table.
    /// DuckDB FTS extension must be available (bundled in most builds).
    pub(crate) fn create_fts_index(&self) -> Result<()> {
        // Ensure FTS extension is loaded
        self.execute_ignore_error("INSTALL fts");
        self.execute_ignore_error("LOAD fts");

        // Create FTS index with overwrite to handle existing index
        let conn = self.write_conn();
        conn.execute_batch("PRAGMA create_fts_index('memories', 'id', 'content', overwrite=1)")?;
        debug!("Created/rebuilt FTS index on memories table");
        Ok(())
    }

    /// Create or rebuild the FTS index on the episodes table (title + summary).
    #[allow(dead_code)] // planned API: episode FTS indexing
    pub(crate) fn create_fts_index_episodes(&self) -> Result<()> {
        self.execute_ignore_error("INSTALL fts");
        self.execute_ignore_error("LOAD fts");
        let conn = self.write_conn();
        conn.execute_batch(
            "PRAGMA create_fts_index('episodes', 'episode_id', 'title', 'summary', overwrite=1)",
        )?;
        debug!("Created/rebuilt FTS index on episodes table");
        Ok(())
    }

    /// Create or rebuild the FTS index on the events table (purified_content).
    /// Note: This should only be called after compact() has populated purified_content.
    pub(crate) fn create_fts_index_events(&self) -> Result<()> {
        self.execute_ignore_error("INSTALL fts");
        self.execute_ignore_error("LOAD fts");
        let conn = self.write_conn();
        // FTS on purified_content (not raw content) for better keyword matching
        conn.execute_batch(
            "PRAGMA create_fts_index('events', 'event_id', 'purified_content', overwrite=1)",
        )?;
        debug!("Created/rebuilt FTS index on events.purified_content");
        Ok(())
    }
}

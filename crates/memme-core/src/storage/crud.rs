use crate::error::{MemoryError, Result};
use crate::types::{Resolution, SqlParam, UpdateOptions};

use super::util::opt_text;
use super::{MemoryRow, Storage};

/// Parameters for inserting a new memory (internal storage layer).
/// Core fields (id, content, embedding, user_id, hash) are passed directly;
/// all other fields go through this struct with sensible defaults.
#[derive(Debug, Clone, Default)]
pub(crate) struct InsertMemoryParams {
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
    pub app_id: Option<String>,
    pub actor_id: Option<String>,
    pub metadata: Option<String>,
    pub importance: Option<f32>,
    pub immutable: bool,
    pub expiration_date: Option<String>,
    pub categories: Option<Vec<String>>,
    pub memory_type: Option<String>,
    pub stability: Option<f32>,
    pub privacy: Option<String>,
    pub event_time: Option<String>,
    pub episode_id: Option<String>,
    pub session_id: Option<String>,
    pub resolution: Resolution,
}

impl Storage {
    // ── Insert ──

    pub(crate) fn insert_memory(
        &self,
        id: &str,
        content: &str,
        embedding: &[f32],
        user_id: &str,
        hash: &str,
        params_: &InsertMemoryParams,
    ) -> Result<()> {
        if embedding.len() != self.config.embedding_dims {
            return Err(MemoryError::Config(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dims,
                embedding.len()
            )));
        }
        let emb_literal = self.format_embedding(embedding, self.config.embedding_dims)?;
        let agent_val = opt_text(params_.agent_id.as_deref());
        let run_val = opt_text(params_.run_id.as_deref());
        let app_val = opt_text(params_.app_id.as_deref());
        let actor_val = opt_text(params_.actor_id.as_deref());
        let meta_val = opt_text(params_.metadata.as_deref());
        let imp_val = params_.importance.unwrap_or(0.5) as f64;
        let exp_val = opt_text(params_.expiration_date.as_deref());
        let cats_literal = self.format_categories(params_.categories.as_deref())?;
        let mtype_val = opt_text(params_.memory_type.as_deref());
        let stab_val = params_.stability.unwrap_or(1.0) as f64;
        let privacy_val = params_.privacy.as_deref().unwrap_or("syncable");
        // Normalize partial dates for TIMESTAMP compatibility:
        // "2020" → "2020-01-01", "2023-05" → "2023-05-01"
        let event_time_val = opt_text(params_.event_time.as_ref().map(|t| {
            if t.len() == 4 && t.chars().all(|c| c.is_ascii_digit()) {
                format!("{}-01-01", t)
            } else if t.len() == 7 && t.chars().nth(4) == Some('-') {
                format!("{}-01", t)
            } else {
                t.clone()
            }
        }));
        let episode_val = opt_text(params_.episode_id.as_deref());
        let session_val = opt_text(params_.session_id.as_deref());
        let resolution_val = params_.resolution.as_str();

        let sql = format!(
            r#"INSERT INTO memories (id, content, embedding, user_id, agent_id, run_id, app_id, actor_id, hash, metadata, importance, immutable, expiration_date, categories, memory_type, stability, privacy, event_time, episode_id, session_id, resolution)
               VALUES ($1, $2, {emb_literal}, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, {cats_literal}, $13, $14, $15, $16, $17, $18, $19)"#
        );
        let memory_params = vec![
            SqlParam::Text(id.to_string()),
            SqlParam::Text(content.to_string()),
            SqlParam::Text(user_id.to_string()),
            agent_val.clone(),
            run_val.clone(),
            app_val.clone(),
            actor_val,
            SqlParam::Text(hash.to_string()),
            meta_val,
            SqlParam::Float(imp_val),
            SqlParam::Bool(params_.immutable),
            exp_val,
            mtype_val,
            SqlParam::Float(stab_val),
            SqlParam::Text(privacy_val.to_string()),
            event_time_val,
            episode_val,
            session_val,
            SqlParam::Text(resolution_val.to_string()),
        ];
        let vector_sql = self.dialect().vector_insert_sql("$1", &emb_literal);
        self.backend.transaction(|tx| {
            tx.execute(&sql, &memory_params)?;
            if let Some(vector_sql) = vector_sql.as_deref() {
                tx.execute(
                    vector_sql,
                    &[
                        SqlParam::Text(id.to_string()),
                        SqlParam::Text(user_id.to_string()),
                        agent_val,
                        run_val,
                        app_val,
                    ],
                )?;
            }
            if tx.table_exists("memories_fts")? {
                tx.execute(
                    "INSERT OR REPLACE INTO memories_fts(id, content) VALUES ($1, $2)",
                    &[
                        SqlParam::Text(id.to_string()),
                        SqlParam::Text(content.to_string()),
                    ],
                )?;
            }
            Ok(())
        })
    }

    /// Set the episode_id on a memory (for linking extracted memories to their source episode).
    #[allow(dead_code)] // planned API: memory-episode linking
    pub(crate) fn set_memory_episode_id(&self, memory_id: &str, episode_id: &str) -> Result<()> {
        self.backend.execute(
            "UPDATE memories SET episode_id = $1 WHERE id = $2",
            &[
                SqlParam::Text(episode_id.to_string()),
                SqlParam::Text(memory_id.to_string()),
            ],
        )?;
        Ok(())
    }

    /// Set the session_id on a memory (for linking extracted memories to their source session).
    #[allow(dead_code)] // planned API: memory-session linking
    pub(crate) fn set_memory_session_id(&self, memory_id: &str, session_id: &str) -> Result<()> {
        self.backend.execute(
            "UPDATE memories SET session_id = $1 WHERE id = $2",
            &[
                SqlParam::Text(session_id.to_string()),
                SqlParam::Text(memory_id.to_string()),
            ],
        )?;
        Ok(())
    }

    /// Batch-update session_id and episode_id for multiple memories in a single query.
    #[allow(dead_code)] // planned API: batch memory tagging
    pub(crate) fn tag_memories_batch(
        &self,
        memory_ids: &[&str],
        session_id: &str,
        episode_id: &str,
    ) -> Result<()> {
        if memory_ids.is_empty() {
            return Ok(());
        }
        // Build IN clause: $3, $4, ... for memory IDs (after $1=session_id, $2=episode_id)
        let placeholders: Vec<String> =
            (3..3 + memory_ids.len()).map(|i| format!("${i}")).collect();
        let sql = format!(
            "UPDATE memories SET session_id = $1, episode_id = $2 WHERE id IN ({})",
            placeholders.join(", ")
        );
        let mut param_vals: Vec<SqlParam> = vec![
            SqlParam::Text(session_id.to_string()),
            SqlParam::Text(episode_id.to_string()),
        ];
        for id in memory_ids {
            param_vals.push(SqlParam::Text(id.to_string()));
        }
        self.backend.execute(&sql, &param_vals)?;
        Ok(())
    }

    // ── Update content + embedding ──

    /// Update a memory's content, embedding, and hash.
    ///
    /// `metadata` uses `Option<Option<&serde_json::Value>>`:
    /// - `None` — do not touch the metadata column
    /// - `Some(None)` — set metadata to NULL
    /// - `Some(Some(val))` — set metadata to the given JSON value
    ///
    /// If `options` is provided:
    /// - `options.timestamp` overrides the updated_at to a custom value
    /// - `options.metadata` overrides the `metadata` parameter
    ///
    /// Returns error if the memory is immutable.
    pub(crate) fn update_memory(
        &self,
        id: &str,
        content: &str,
        embedding: &[f32],
        hash: &str,
        metadata: Option<Option<&serde_json::Value>>,
        options: Option<&UpdateOptions>,
    ) -> Result<()> {
        if embedding.len() != self.config.embedding_dims {
            return Err(MemoryError::Config(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dims,
                embedding.len()
            )));
        }
        let emb_literal = self.format_embedding(embedding, self.config.embedding_dims)?;

        // Determine the effective metadata: options.metadata overrides the metadata param
        let effective_metadata = if let Some(opts) = options {
            if opts.metadata.is_some() {
                Some(opts.metadata.as_ref())
            } else {
                metadata
            }
        } else {
            metadata
        };

        // Determine if we have a custom timestamp
        let custom_timestamp: Option<&str> = options.and_then(|opts| opts.timestamp.as_deref());

        let (update_sql, update_params) = match (&effective_metadata, custom_timestamp) {
            (Some(Some(val)), Some(ts)) => {
                let json_str = serde_json::to_string(val).unwrap_or_else(|_| "null".to_string());
                let sql = format!(
                    r#"UPDATE memories
                       SET content = $1,
                           embedding = {emb_literal},
                           hash = $2,
                           updated_at = $3,
                           metadata = $4
                       WHERE id = $5"#
                );
                (
                    sql,
                    vec![
                        SqlParam::Text(content.to_string()),
                        SqlParam::Text(hash.to_string()),
                        SqlParam::Text(ts.to_string()),
                        SqlParam::Text(json_str),
                        SqlParam::Text(id.to_string()),
                    ],
                )
            }
            (Some(Some(val)), None) => {
                let json_str = serde_json::to_string(val).unwrap_or_else(|_| "null".to_string());
                let now_ts = self.dialect().current_timestamp_expr();
                let sql = format!(
                    r#"UPDATE memories
                       SET content = $1,
                           embedding = {emb_literal},
                           hash = $2,
                           updated_at = {now_ts},
                           metadata = $3
                       WHERE id = $4"#
                );
                (
                    sql,
                    vec![
                        SqlParam::Text(content.to_string()),
                        SqlParam::Text(hash.to_string()),
                        SqlParam::Text(json_str),
                        SqlParam::Text(id.to_string()),
                    ],
                )
            }
            (Some(None), Some(ts)) => {
                let sql = format!(
                    r#"UPDATE memories
                       SET content = $1,
                           embedding = {emb_literal},
                           hash = $2,
                           updated_at = $3,
                           metadata = NULL
                       WHERE id = $4"#
                );
                (
                    sql,
                    vec![
                        SqlParam::Text(content.to_string()),
                        SqlParam::Text(hash.to_string()),
                        SqlParam::Text(ts.to_string()),
                        SqlParam::Text(id.to_string()),
                    ],
                )
            }
            (Some(None), None) => {
                let now_ts = self.dialect().current_timestamp_expr();
                let sql = format!(
                    r#"UPDATE memories
                       SET content = $1,
                           embedding = {emb_literal},
                           hash = $2,
                           updated_at = {now_ts},
                           metadata = NULL
                       WHERE id = $3"#
                );
                (
                    sql,
                    vec![
                        SqlParam::Text(content.to_string()),
                        SqlParam::Text(hash.to_string()),
                        SqlParam::Text(id.to_string()),
                    ],
                )
            }
            (None, Some(ts)) => {
                let sql = format!(
                    r#"UPDATE memories
                       SET content = $1,
                           embedding = {emb_literal},
                           hash = $2,
                           updated_at = $3
                       WHERE id = $4"#
                );
                (
                    sql,
                    vec![
                        SqlParam::Text(content.to_string()),
                        SqlParam::Text(hash.to_string()),
                        SqlParam::Text(ts.to_string()),
                        SqlParam::Text(id.to_string()),
                    ],
                )
            }
            (None, None) => {
                let now_ts = self.dialect().current_timestamp_expr();
                let sql = format!(
                    r#"UPDATE memories
                       SET content = $1,
                           embedding = {emb_literal},
                           hash = $2,
                           updated_at = {now_ts}
                       WHERE id = $3"#
                );
                (
                    sql,
                    vec![
                        SqlParam::Text(content.to_string()),
                        SqlParam::Text(hash.to_string()),
                        SqlParam::Text(id.to_string()),
                    ],
                )
            }
        };

        let delete_sql = self.dialect().vector_delete_sql();
        let insert_sql = self.dialect().vector_insert_sql("$1", &emb_literal);
        self.backend.transaction(|tx| {
            let scope = tx.query_one(
                "SELECT immutable, user_id, agent_id, run_id, app_id FROM memories WHERE id = $1",
                &[SqlParam::Text(id.to_string())],
                |row| {
                    Ok((
                        row.get_opt_bool(0)?.unwrap_or(false),
                        row.get_string(1)?,
                        row.get_opt_string(2)?,
                        row.get_opt_string(3)?,
                        row.get_opt_string(4)?,
                    ))
                },
            )?
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;
            if scope.0 {
                return Err(MemoryError::ImmutableMemory(id.to_string()));
            }

            tx.execute(&update_sql, &update_params)?;
            if let Some(delete_sql) = delete_sql {
                tx.execute(delete_sql, &[SqlParam::Text(id.to_string())])?;
            }
            if let Some(insert_sql) = insert_sql.as_deref() {
                tx.execute(
                    insert_sql,
                    &[
                        SqlParam::Text(id.to_string()),
                        SqlParam::Text(scope.1),
                        opt_text(scope.2.as_deref()),
                        opt_text(scope.3.as_deref()),
                        opt_text(scope.4.as_deref()),
                    ],
                )?;
            }
            if tx.table_exists("memories_fts")? {
                tx.execute(
                    "INSERT OR REPLACE INTO memories_fts(id, content) VALUES ($1, $2)",
                    &[
                        SqlParam::Text(id.to_string()),
                        SqlParam::Text(content.to_string()),
                    ],
                )?;
            }
            Ok(())
        })
    }

    // ── Update metadata only (dedup fast path) ──

    /// Update only the metadata and updated_at timestamp of a memory,
    /// without re-writing content or embedding. Used when content hash
    /// matches exactly (no embedding recomputation needed).
    pub(crate) fn update_metadata_on_dedup(
        &self,
        id: &str,
        metadata: Option<&serde_json::Value>,
    ) -> Result<()> {
        let now_ts = self.dialect().current_timestamp_expr();
        match metadata {
            Some(val) => {
                let json_str = serde_json::to_string(val).unwrap_or_else(|_| "null".to_string());
                self.backend.execute(
                    &format!(
                        "UPDATE memories SET metadata = $1, updated_at = {now_ts} WHERE id = $2"
                    ),
                    &[SqlParam::Text(json_str), SqlParam::Text(id.to_string())],
                )?;
            }
            None => {
                self.backend.execute(
                    &format!("UPDATE memories SET updated_at = {now_ts} WHERE id = $1"),
                    &[SqlParam::Text(id.to_string())],
                )?;
            }
        }
        Ok(())
    }

    // ── Mark superseded ──

    /// Mark an old memory as superseded by a new memory.
    /// Sets `superseded_by` to the new memory's ID and `valid_until` to the current timestamp.
    pub(crate) fn mark_superseded(&self, old_id: &str, new_id: &str) -> Result<()> {
        let now_ts = self.dialect().current_timestamp_expr();
        self.backend.execute(
            &format!(
                "UPDATE memories SET superseded_by = $1, valid_until = {now_ts} WHERE id = $2"
            ),
            &[
                SqlParam::Text(new_id.to_string()),
                SqlParam::Text(old_id.to_string()),
            ],
        )?;
        Ok(())
    }

    // ── Update importance ──

    /// Update a memory's importance score without changing content or embedding.
    pub(crate) fn update_importance(&self, id: &str, importance: f32) -> Result<()> {
        let now_ts = self.dialect().current_timestamp_expr();
        self.backend.execute(
            &format!("UPDATE memories SET importance = $1, updated_at = {now_ts} WHERE id = $2"),
            &[
                SqlParam::Float(importance as f64),
                SqlParam::Text(id.to_string()),
            ],
        )?;
        Ok(())
    }

    // ── Delete ──

    /// Delete a memory by ID. Returns error if the memory is immutable.
    pub(crate) fn delete_memory(&self, id: &str) -> Result<()> {
        let vector_sql = self.dialect().vector_delete_sql();
        self.backend.transaction(|tx| {
            let immutable = tx.query_one(
                "SELECT immutable FROM memories WHERE id = $1",
                &[SqlParam::Text(id.to_string())],
                |row| row.get_opt_bool(0).map(|value| value.unwrap_or(false)),
            )?;
            match immutable {
                Some(true) => return Err(MemoryError::ImmutableMemory(id.to_string())),
                None => return Err(MemoryError::NotFound(id.to_string())),
                Some(false) => {}
            }

            if tx.table_exists("memories_fts")? {
                tx.execute(
                    "DELETE FROM memories_fts WHERE id = $1",
                    &[SqlParam::Text(id.to_string())],
                )?;
            }
            if let Some(vector_sql) = vector_sql {
                tx.execute(vector_sql, &[SqlParam::Text(id.to_string())])?;
            }
            tx.execute(
                "DELETE FROM associations WHERE from_id = $1 OR to_id = $1",
                &[SqlParam::Text(id.to_string())],
            )?;
            tx.execute(
                "DELETE FROM memory_entities WHERE memory_id = $1",
                &[SqlParam::Text(id.to_string())],
            )?;
            tx.execute(
                "DELETE FROM memories WHERE id = $1",
                &[SqlParam::Text(id.to_string())],
            )?;
            Ok(())
        })
    }

    // ── Get by ID ──

    pub(crate) fn get_memory(&self, id: &str) -> Result<Option<MemoryRow>> {
        let cols = super::query::memory_select_cols(None, "");
        let sql = format!("SELECT {cols} FROM memories WHERE id = $1");
        self.backend.query_one(
            &sql,
            &[SqlParam::Text(id.to_string())],
            super::query::map_memory_row,
        )
    }

    // ── Get content by ID (for history recording) ──

    pub(crate) fn get_content(&self, id: &str) -> Result<Option<(String, String)>> {
        self.backend.query_one(
            "SELECT content, user_id FROM memories WHERE id = $1",
            &[SqlParam::Text(id.to_string())],
            |row| Ok((row.get_string(0)?, row.get_string(1)?)),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::config::MemoryConfig;
    use crate::error::MemoryError;
    use crate::types::UpdateOptions;

    use super::{InsertMemoryParams, Storage};

    fn test_config(dims: usize) -> MemoryConfig {
        MemoryConfig::new(":memory:", dims)
    }

    fn open_storage(dims: usize) -> Storage {
        Storage::open(test_config(dims)).unwrap()
    }

    fn dummy_embedding(dims: usize, seed: f32) -> Vec<f32> {
        (0..dims).map(|i| (i as f32 * 0.01 + seed).sin()).collect()
    }

    #[test]
    fn test_insert_and_get() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);
        storage
            .insert_memory(
                "id1",
                "hello world",
                &emb,
                "user1",
                "hash1",
                &InsertMemoryParams {
                    agent_id: Some("agent1".into()),
                    metadata: Some(r#"{"key":"val"}"#.into()),
                    ..Default::default()
                },
            )
            .unwrap();

        let row = storage.get_memory("id1").unwrap().unwrap();
        assert_eq!(row.id, "id1");
        assert_eq!(row.content, "hello world");
        assert_eq!(row.user_id, "user1");
        assert_eq!(row.agent_id.as_deref(), Some("agent1"));
        assert!(row.metadata.is_some());
        assert!(row.metadata.unwrap().contains("key"));
    }

    #[test]
    fn test_insert_with_new_fields() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);
        storage
            .insert_memory(
                "id1",
                "hello world",
                &emb,
                "user1",
                "hash1",
                &InsertMemoryParams {
                    agent_id: Some("agent1".into()),
                    run_id: Some("run1".into()),
                    app_id: Some("app1".into()),
                    actor_id: Some("actor1".into()),
                    metadata: Some(r#"{"key":"val"}"#.into()),
                    importance: Some(0.8),
                    immutable: true,
                    expiration_date: Some("2027-12-31T23:59:59".into()),
                    categories: Some(vec!["work".to_string(), "tech".to_string()]),
                    ..Default::default()
                },
            )
            .unwrap();

        let row = storage.get_memory("id1").unwrap().unwrap();
        assert_eq!(row.id, "id1");
        assert_eq!(row.agent_id.as_deref(), Some("agent1"));
        assert_eq!(row.app_id.as_deref(), Some("app1"));
        assert_eq!(row.run_id.as_deref(), Some("run1"));
        assert!(row.immutable);
        assert!(row.expiration_date.is_some());
        // Categories are stored and returned as JSON array string
        assert!(row.categories.is_some());
    }

    #[test]
    fn test_update_memory() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);
        storage
            .insert_memory(
                "id1",
                "original",
                &emb,
                "user1",
                "hash1",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        let emb2 = dummy_embedding(384, 2.0);
        storage
            .update_memory("id1", "updated", &emb2, "hash2", None, None)
            .unwrap();

        let row = storage.get_memory("id1").unwrap().unwrap();
        assert_eq!(row.content, "updated");
    }

    #[test]
    fn test_update_memory_with_metadata() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);
        storage
            .insert_memory(
                "id1",
                "content",
                &emb,
                "user1",
                "hash1",
                &InsertMemoryParams {
                    metadata: Some(r#"{"old":"data"}"#.into()),
                    ..Default::default()
                },
            )
            .unwrap();

        // Some(Some(val)) — set metadata to new value
        let new_meta = serde_json::json!({"new": "data"});
        storage
            .update_memory("id1", "content", &emb, "hash1", Some(Some(&new_meta)), None)
            .unwrap();
        let row = storage.get_memory("id1").unwrap().unwrap();
        assert!(row.metadata.as_ref().unwrap().contains("new"));

        // Some(None) — set metadata to NULL
        storage
            .update_memory("id1", "content", &emb, "hash1", Some(None), None)
            .unwrap();
        let row = storage.get_memory("id1").unwrap().unwrap();
        assert!(row.metadata.is_none());

        // None — do not touch metadata (stays NULL)
        storage
            .update_memory("id1", "content2", &emb, "hash1", None, None)
            .unwrap();
        let row = storage.get_memory("id1").unwrap().unwrap();
        assert!(row.metadata.is_none());
    }

    #[test]
    fn test_update_memory_with_options() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);
        storage
            .insert_memory(
                "id1",
                "content",
                &emb,
                "user1",
                "hash1",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        // Update with custom timestamp via UpdateOptions
        let opts = UpdateOptions::new().timestamp("2025-06-15T10:00:00");
        storage
            .update_memory("id1", "updated", &emb, "hash2", None, Some(&opts))
            .unwrap();
        let row = storage.get_memory("id1").unwrap().unwrap();
        assert_eq!(row.content, "updated");
        assert!(row.updated_at.contains("2025-06-15"));

        // Update with metadata override via UpdateOptions
        let meta = serde_json::json!({"from": "options"});
        let opts2 = UpdateOptions::new().metadata(meta);
        storage
            .update_memory("id1", "updated2", &emb, "hash3", None, Some(&opts2))
            .unwrap();
        let row = storage.get_memory("id1").unwrap().unwrap();
        assert!(row.metadata.as_ref().unwrap().contains("options"));
    }

    #[test]
    fn test_update_immutable_memory_fails() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);
        storage
            .insert_memory(
                "id1",
                "content",
                &emb,
                "user1",
                "hash1",
                &InsertMemoryParams {
                    immutable: true,
                    ..Default::default()
                },
            )
            .unwrap();

        let result = storage.update_memory("id1", "new", &emb, "hash2", None, None);
        assert!(result.is_err());
        match result.unwrap_err() {
            MemoryError::ImmutableMemory(_) => {}
            other => panic!("Expected ImmutableMemory error, got: {other}"),
        }
    }

    #[test]
    fn test_delete_memory() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);
        storage
            .insert_memory(
                "id1",
                "content",
                &emb,
                "user1",
                "hash1",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        storage.delete_memory("id1").unwrap();
        let row = storage.get_memory("id1").unwrap();
        assert!(row.is_none());
    }

    #[test]
    fn test_delete_immutable_memory_fails() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);
        storage
            .insert_memory(
                "id1",
                "content",
                &emb,
                "user1",
                "hash1",
                &InsertMemoryParams {
                    immutable: true,
                    ..Default::default()
                },
            )
            .unwrap();

        let result = storage.delete_memory("id1");
        assert!(result.is_err());
        match result.unwrap_err() {
            MemoryError::ImmutableMemory(_) => {}
            other => panic!("Expected ImmutableMemory error, got: {other}"),
        }
    }

    #[test]
    fn test_sql_injection_resistance() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);
        let malicious_user = "'; DROP TABLE memories; --";

        // Insert with malicious user_id — should not break anything
        storage
            .insert_memory(
                "id1",
                "content",
                &emb,
                malicious_user,
                "hash1",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        // Verify memories table still exists and has the row
        let row = storage.get_memory("id1").unwrap().unwrap();
        assert_eq!(row.user_id, malicious_user);

        // Verify we can still insert more
        storage
            .insert_memory(
                "id2",
                "more content",
                &emb,
                "normal_user",
                "hash2",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        let row2 = storage.get_memory("id2").unwrap();
        assert!(row2.is_some());
    }
}

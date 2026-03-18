use duckdb::params;

use crate::error::{MemoryError, Result};
use crate::types::{Resolution, UpdateOptions};

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
        let emb_literal = Self::format_embedding(embedding, self.config.embedding_dims)?;
        let agent_val: duckdb::types::Value = match &params_.agent_id {
            Some(a) => duckdb::types::Value::Text(a.clone()),
            None => duckdb::types::Value::Null,
        };
        let run_val: duckdb::types::Value = match &params_.run_id {
            Some(r) => duckdb::types::Value::Text(r.clone()),
            None => duckdb::types::Value::Null,
        };
        let app_val: duckdb::types::Value = match &params_.app_id {
            Some(a) => duckdb::types::Value::Text(a.clone()),
            None => duckdb::types::Value::Null,
        };
        let actor_val: duckdb::types::Value = match &params_.actor_id {
            Some(a) => duckdb::types::Value::Text(a.clone()),
            None => duckdb::types::Value::Null,
        };
        let meta_val: duckdb::types::Value = match &params_.metadata {
            Some(m) => duckdb::types::Value::Text(m.clone()),
            None => duckdb::types::Value::Null,
        };
        let imp_val = params_.importance.unwrap_or(0.5) as f64;
        let exp_val: duckdb::types::Value = match &params_.expiration_date {
            Some(d) => duckdb::types::Value::Text(d.clone()),
            None => duckdb::types::Value::Null,
        };
        let cats_literal = Self::format_categories(params_.categories.as_deref())?;
        let mtype_val: duckdb::types::Value = match &params_.memory_type {
            Some(m) => duckdb::types::Value::Text(m.clone()),
            None => duckdb::types::Value::Null,
        };
        let stab_val = params_.stability.unwrap_or(1.0) as f64;
        let privacy_val = params_.privacy.as_deref().unwrap_or("syncable");
        let event_time_val: duckdb::types::Value = match &params_.event_time {
            Some(t) => {
                // Normalize partial dates for DuckDB TIMESTAMP compatibility:
                // "2020" → "2020-01-01", "2023-05" → "2023-05-01"
                let normalized = if t.len() == 4 && t.chars().all(|c| c.is_ascii_digit()) {
                    format!("{}-01-01", t)
                } else if t.len() == 7 && t.chars().nth(4) == Some('-') {
                    format!("{}-01", t)
                } else {
                    t.clone()
                };
                duckdb::types::Value::Text(normalized)
            }
            None => duckdb::types::Value::Null,
        };
        let episode_val: duckdb::types::Value = match &params_.episode_id {
            Some(e) => duckdb::types::Value::Text(e.clone()),
            None => duckdb::types::Value::Null,
        };
        let session_val: duckdb::types::Value = match &params_.session_id {
            Some(s) => duckdb::types::Value::Text(s.clone()),
            None => duckdb::types::Value::Null,
        };
        let resolution_val = params_.resolution.as_str();

        let sql = format!(
            r#"INSERT INTO memories (id, content, embedding, user_id, agent_id, run_id, app_id, actor_id, hash, metadata, importance, immutable, expiration_date, categories, memory_type, stability, privacy, event_time, episode_id, session_id, resolution)
               VALUES ($1, $2, {emb_literal}, $3, $4, $5, $6, $7, $8, $9, $10, $11, CASE WHEN $12 IS NULL THEN NULL ELSE CAST($12 AS TIMESTAMP) END, {cats_literal}, $13, $14, $15, CASE WHEN $16 IS NULL THEN NULL ELSE CAST($16 AS TIMESTAMP) END, $17, $18, $19)"#
        );
        let conn = self.write_conn();
        conn.execute(
            &sql,
            params![
                id,
                content,
                user_id,
                agent_val,
                run_val,
                app_val,
                actor_val,
                hash,
                meta_val,
                imp_val,
                params_.immutable,
                exp_val,
                mtype_val,
                stab_val,
                privacy_val,
                event_time_val,
                episode_val,
                session_val,
                resolution_val
            ],
        )?;
        Ok(())
    }

    /// Set the episode_id on a memory (for linking extracted memories to their source episode).
    #[allow(dead_code)] // planned API: memory-episode linking
    pub(crate) fn set_memory_episode_id(&self, memory_id: &str, episode_id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            "UPDATE memories SET episode_id = $1 WHERE id = $2",
            params![episode_id, memory_id],
        )?;
        Ok(())
    }

    /// Set the session_id on a memory (for linking extracted memories to their source session).
    #[allow(dead_code)] // planned API: memory-session linking
    pub(crate) fn set_memory_session_id(&self, memory_id: &str, session_id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            "UPDATE memories SET session_id = $1 WHERE id = $2",
            params![session_id, memory_id],
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
        let conn = self.write_conn();
        let mut param_vals: Vec<duckdb::types::Value> = vec![
            duckdb::types::Value::Text(session_id.to_string()),
            duckdb::types::Value::Text(episode_id.to_string()),
        ];
        for id in memory_ids {
            param_vals.push(duckdb::types::Value::Text(id.to_string()));
        }
        let param_refs: Vec<&dyn duckdb::ToSql> =
            param_vals.iter().map(|p| p as &dyn duckdb::ToSql).collect();
        conn.execute(&sql, param_refs.as_slice())?;
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
        // Check immutable flag atomically (same write_conn used for the update below)
        self.check_immutable(id)?;

        if embedding.len() != self.config.embedding_dims {
            return Err(MemoryError::Config(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dims,
                embedding.len()
            )));
        }
        let emb_literal = Self::format_embedding(embedding, self.config.embedding_dims)?;

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

        let conn = self.write_conn();
        match (&effective_metadata, custom_timestamp) {
            (Some(Some(val)), Some(ts)) => {
                let json_str = serde_json::to_string(val).unwrap_or_else(|_| "null".to_string());
                let meta_val = duckdb::types::Value::Text(json_str);
                let sql = format!(
                    r#"UPDATE memories
                       SET content = $1,
                           embedding = {emb_literal},
                           hash = $2,
                           updated_at = CAST($3 AS TIMESTAMP),
                           metadata = $4
                       WHERE id = $5"#
                );
                conn.execute(&sql, params![content, hash, ts, meta_val, id])?;
            }
            (Some(Some(val)), None) => {
                let json_str = serde_json::to_string(val).unwrap_or_else(|_| "null".to_string());
                let meta_val = duckdb::types::Value::Text(json_str);
                let sql = format!(
                    r#"UPDATE memories
                       SET content = $1,
                           embedding = {emb_literal},
                           hash = $2,
                           updated_at = now()::TIMESTAMP,
                           metadata = $3
                       WHERE id = $4"#
                );
                conn.execute(&sql, params![content, hash, meta_val, id])?;
            }
            (Some(None), Some(ts)) => {
                let sql = format!(
                    r#"UPDATE memories
                       SET content = $1,
                           embedding = {emb_literal},
                           hash = $2,
                           updated_at = CAST($3 AS TIMESTAMP),
                           metadata = NULL
                       WHERE id = $4"#
                );
                conn.execute(&sql, params![content, hash, ts, id])?;
            }
            (Some(None), None) => {
                let sql = format!(
                    r#"UPDATE memories
                       SET content = $1,
                           embedding = {emb_literal},
                           hash = $2,
                           updated_at = now()::TIMESTAMP,
                           metadata = NULL
                       WHERE id = $3"#
                );
                conn.execute(&sql, params![content, hash, id])?;
            }
            (None, Some(ts)) => {
                let sql = format!(
                    r#"UPDATE memories
                       SET content = $1,
                           embedding = {emb_literal},
                           hash = $2,
                           updated_at = CAST($3 AS TIMESTAMP)
                       WHERE id = $4"#
                );
                conn.execute(&sql, params![content, hash, ts, id])?;
            }
            (None, None) => {
                let sql = format!(
                    r#"UPDATE memories
                       SET content = $1,
                           embedding = {emb_literal},
                           hash = $2,
                           updated_at = now()::TIMESTAMP
                       WHERE id = $3"#
                );
                conn.execute(&sql, params![content, hash, id])?;
            }
        }
        Ok(())
    }

    // ── Update importance ──

    /// Update a memory's importance score without changing content or embedding.
    pub(crate) fn update_importance(&self, id: &str, importance: f32) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            "UPDATE memories SET importance = $1, updated_at = now()::TIMESTAMP WHERE id = $2",
            params![importance, id],
        )?;
        Ok(())
    }

    // ── Delete ──

    /// Delete a memory by ID. Returns error if the memory is immutable.
    pub(crate) fn delete_memory(&self, id: &str) -> Result<()> {
        // Check immutable flag
        self.check_immutable(id)?;

        let conn = self.write_conn();
        conn.execute("DELETE FROM memories WHERE id = $1", params![id])?;
        Ok(())
    }

    // ── Get by ID ──

    pub(crate) fn get_memory(&self, id: &str) -> Result<Option<MemoryRow>> {
        // Use write_conn for read-after-write consistency (file-backed DuckDB
        // uses snapshot isolation, so read_conn may not see recent writes).
        let conn = self.write_conn();
        let mut stmt = conn.prepare(
            "SELECT id, content, user_id,
                    CAST(created_at AS VARCHAR) AS created_at,
                    CAST(updated_at AS VARCHAR) AS updated_at,
                    CAST(metadata AS VARCHAR) AS metadata,
                    importance,
                    access_count,
                    agent_id,
                    app_id,
                    run_id,
                    immutable,
                    CAST(expiration_date AS VARCHAR) AS expiration_date,
                    CAST(categories AS VARCHAR) AS categories,
                    memory_type,
                    stability,
                    privacy,
                    CAST(event_time AS VARCHAR) AS event_time,
                    episode_id,
                    session_id,
                    resolution
             FROM memories WHERE id = $1",
        )?;

        let mut rows = stmt.query_map(params![id], |row| {
            Ok(MemoryRow {
                id: row.get(0)?,
                content: row.get(1)?,
                user_id: row.get(2)?,
                created_at: row.get::<_, String>(3)?,
                updated_at: row.get::<_, String>(4)?,
                metadata: row.get::<_, Option<String>>(5)?,
                score: None,
                importance: row.get::<_, Option<f64>>(6)?.map(|v| v as f32),
                access_count: row.get::<_, Option<i32>>(7)?.map(|v| v as u32),
                agent_id: row.get::<_, Option<String>>(8)?,
                app_id: row.get::<_, Option<String>>(9)?,
                run_id: row.get::<_, Option<String>>(10)?,
                immutable: row.get::<_, Option<bool>>(11)?.unwrap_or(false),
                expiration_date: row.get::<_, Option<String>>(12)?,
                categories: row.get::<_, Option<String>>(13)?,
                memory_type: row.get::<_, Option<String>>(14)?,
                stability: row.get::<_, Option<f64>>(15)?.map(|v| v as f32),
                privacy: row.get::<_, Option<String>>(16)?,
                event_time: row.get::<_, Option<String>>(17)?,
                episode_id: row.get::<_, Option<String>>(18)?,
                session_id: row.get::<_, Option<String>>(19)?,
                resolution: row.get::<_, Option<String>>(20)?,
            })
        })?;

        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    // ── Get content by ID (for history recording) ──

    pub(crate) fn get_content(&self, id: &str) -> Result<Option<(String, String)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare("SELECT content, user_id FROM memories WHERE id = $1")?;
        let mut rows = stmt.query_map(params![id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::MemoryConfig;
    use crate::error::MemoryError;
    use crate::types::UpdateOptions;

    use super::{InsertMemoryParams, Storage};

    fn test_config(dims: usize) -> MemoryConfig {
        MemoryConfig {
            db_path: ":memory:".into(),
            collection_name: "test".into(),
            embedding_dims: dims,
            dedup_threshold: 0.15,
            default_limit: 10,
            ..Default::default()
        }
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
        // Categories are stored and returned as DuckDB list string
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

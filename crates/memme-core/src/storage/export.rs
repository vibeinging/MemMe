use duckdb::params;

use crate::error::Result;
use crate::types::MemoryExport;

use super::Storage;

impl Storage {
    /// Export all memories, optionally filtered by user_id.
    /// By default, skips `local_only` privacy memories.
    #[allow(dead_code)]
    pub(crate) fn export_memories(&self, user_id: Option<&str>) -> Result<Vec<MemoryExport>> {
        self.export_memories_with_privacy(user_id, false)
    }

    /// Export memories with control over including local_only privacy memories.
    #[allow(dead_code)]
    pub(crate) fn export_memories_with_privacy(
        &self,
        user_id: Option<&str>,
        include_local: bool,
    ) -> Result<Vec<MemoryExport>> {
        let privacy_filter = if include_local {
            ""
        } else {
            " AND (privacy IS NULL OR privacy != 'local_only')"
        };
        let (sql, has_user) = if user_id.is_some() {
            (
                format!(
                    "SELECT id, content, user_id, agent_id, app_id, run_id,
                        CAST(metadata AS VARCHAR) AS metadata,
                        importance, immutable,
                        CAST(expiration_date AS VARCHAR) AS expiration_date,
                        CAST(categories AS VARCHAR) AS categories,
                        CAST(created_at AS VARCHAR) AS created_at,
                        CAST(updated_at AS VARCHAR) AS updated_at,
                        stability
                 FROM memories WHERE user_id = $1{privacy_filter}
                 ORDER BY created_at"
                ),
                true,
            )
        } else {
            (
                format!(
                    "SELECT id, content, user_id, agent_id, app_id, run_id,
                        CAST(metadata AS VARCHAR) AS metadata,
                        importance, immutable,
                        CAST(expiration_date AS VARCHAR) AS expiration_date,
                        CAST(categories AS VARCHAR) AS categories,
                        CAST(created_at AS VARCHAR) AS created_at,
                        CAST(updated_at AS VARCHAR) AS updated_at,
                        stability
                 FROM memories WHERE 1=1{privacy_filter}
                 ORDER BY created_at"
                ),
                false,
            )
        };

        let map_export = |row: &duckdb::Row<'_>| -> duckdb::Result<MemoryExport> {
            let metadata_str: Option<String> = row.get(6)?;
            let metadata = metadata_str.and_then(|s| serde_json::from_str(&s).ok());
            let categories_raw: Option<String> = row.get(10)?;
            Ok(MemoryExport {
                id: row.get(0)?,
                content: row.get(1)?,
                user_id: row.get(2)?,
                agent_id: row.get(3)?,
                app_id: row.get(4)?,
                run_id: row.get(5)?,
                metadata,
                importance: row.get::<_, Option<f64>>(7)?.unwrap_or(0.5) as f32,
                immutable: row.get::<_, Option<bool>>(8)?.unwrap_or(false),
                expiration_date: row.get(9)?,
                categories: Self::parse_categories(categories_raw),
                created_at: row.get(11)?,
                updated_at: row.get(12)?,
                stability: row.get::<_, Option<f64>>(13)?.map(|v| v as f32),
            })
        };

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = if has_user {
            stmt.query_map(params![user_id.unwrap()], map_export)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map([], map_export)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };

        Ok(rows)
    }

    /// Import memories from export records. Returns the number of successfully imported memories.
    #[allow(dead_code)]
    pub(crate) fn import_memories(&self, memories: &[MemoryExport]) -> Result<u64> {
        let mut count: u64 = 0;
        let conn = self.write_conn();
        for mem in memories {
            let meta_str = mem
                .metadata
                .as_ref()
                .map(|m| serde_json::to_string(m).unwrap_or_default());
            let meta_val: duckdb::types::Value = match &meta_str {
                Some(s) => duckdb::types::Value::Text(s.clone()),
                None => duckdb::types::Value::Null,
            };
            let agent_val: duckdb::types::Value = match &mem.agent_id {
                Some(a) => duckdb::types::Value::Text(a.clone()),
                None => duckdb::types::Value::Null,
            };
            let run_val: duckdb::types::Value = match &mem.run_id {
                Some(r) => duckdb::types::Value::Text(r.clone()),
                None => duckdb::types::Value::Null,
            };
            let app_val: duckdb::types::Value = match &mem.app_id {
                Some(a) => duckdb::types::Value::Text(a.clone()),
                None => duckdb::types::Value::Null,
            };
            let exp_val: duckdb::types::Value = match &mem.expiration_date {
                Some(d) => duckdb::types::Value::Text(d.clone()),
                None => duckdb::types::Value::Null,
            };
            let cats_literal = Self::format_categories(mem.categories.as_deref())?;

            // Use ON CONFLICT to skip duplicates
            let sql = format!(
                r#"INSERT INTO memories (id, content, user_id, agent_id, run_id, app_id, metadata, importance, immutable, expiration_date, categories, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9,
                           CASE WHEN $10 IS NULL THEN NULL ELSE CAST($10 AS TIMESTAMP) END,
                           {cats_literal},
                           CAST($11 AS TIMESTAMP), CAST($12 AS TIMESTAMP))
                   ON CONFLICT (id) DO NOTHING"#
            );
            let affected = conn.execute(
                &sql,
                params![
                    mem.id,
                    mem.content,
                    mem.user_id,
                    agent_val,
                    run_val,
                    app_val,
                    meta_val,
                    mem.importance as f64,
                    mem.immutable,
                    exp_val,
                    mem.created_at,
                    mem.updated_at
                ],
            )?;
            count += affected as u64;
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use crate::config::MemoryConfig;
    use crate::storage::InsertMemoryParams;

    use super::Storage;

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
    fn test_export_import() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);

        storage
            .insert_memory(
                "id1",
                "hello",
                &emb,
                "user1",
                "h1",
                &InsertMemoryParams {
                    agent_id: Some("agent1".into()),
                    app_id: Some("app1".into()),
                    metadata: Some(r#"{"k":"v"}"#.into()),
                    importance: Some(0.8),
                    ..Default::default()
                },
            )
            .unwrap();
        storage
            .insert_memory(
                "id2",
                "world",
                &emb,
                "user1",
                "h2",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        // Export
        let exports = storage.export_memories(Some("user1")).unwrap();
        assert_eq!(exports.len(), 2);
        assert_eq!(exports[0].id, "id1");
        assert_eq!(exports[0].app_id.as_deref(), Some("app1"));

        // Import into a new storage
        let storage2 = open_storage(384);
        let imported = storage2.import_memories(&exports).unwrap();
        assert_eq!(imported, 2);

        // Verify imported data
        let row = storage2.get_memory("id1").unwrap().unwrap();
        assert_eq!(row.content, "hello");
        assert_eq!(row.app_id.as_deref(), Some("app1"));
    }

    #[test]
    fn test_cleanup_expired() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);

        storage
            .insert_memory(
                "id1",
                "expired",
                &emb,
                "user1",
                "h1",
                &InsertMemoryParams {
                    expiration_date: Some("2020-01-01T00:00:00".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        storage
            .insert_memory(
                "id2",
                "future",
                &emb,
                "user1",
                "h2",
                &InsertMemoryParams {
                    expiration_date: Some("2099-01-01T00:00:00".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        storage
            .insert_memory(
                "id3",
                "no_exp",
                &emb,
                "user1",
                "h3",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        let expired_count = storage.cleanup_expired().unwrap();
        assert_eq!(expired_count, 1);

        assert!(storage.get_memory("id1").unwrap().is_none());
        assert!(storage.get_memory("id2").unwrap().is_some());
        assert!(storage.get_memory("id3").unwrap().is_some());
    }
}

use duckdb::params;

use crate::error::Result;

use super::Storage;

impl Storage {
    /// Get the maximum sync_version across all memories.
    pub(crate) fn get_max_sync_version(&self) -> Result<u64> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare("SELECT COALESCE(MAX(sync_version), 0) FROM memories")?;
        let version: i64 = stmt
            .query_map([], |row| row.get(0))?
            .next()
            .expect("aggregate always returns a row")?;
        Ok(version as u64)
    }

    /// Get all changes (memories) with sync_version > since_version,
    /// ordered by sync_version ascending.
    /// Filters out `local_only` privacy memories and includes DELETE tombstones
    /// from the history table.
    pub(crate) fn get_changes_since(
        &self,
        since_version: u64,
    ) -> Result<Vec<crate::sync::SyncChange>> {
        // 1. Query existing memories (CREATE/UPDATE) — exclude local_only
        let sql = r#"
            SELECT m.id, m.content, CAST(m.updated_at AS VARCHAR),
                   m.sync_version, m.device_id,
                   CAST(m.metadata AS VARCHAR),
                   COALESCE(
                       (SELECT h.event FROM history h
                        WHERE h.memory_id = m.id
                        ORDER BY h.created_at DESC LIMIT 1),
                       'ADD'
                   ) AS last_event
            FROM memories m
            WHERE m.sync_version > $1
              AND (m.privacy IS NULL OR m.privacy != 'local_only')
            ORDER BY m.sync_version ASC
        "#;

        let conn = self.read_conn();
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt
            .query_map(params![since_version as i64], |row| {
                let id: String = row.get(0)?;
                let content: String = row.get(1)?;
                let timestamp: String = row.get(2)?;
                let version: i64 = row.get(3)?;
                let device_id: Option<String> = row.get(4)?;
                let metadata_str: Option<String> = row.get(5)?;
                let event: String = row.get(6)?;

                Ok((
                    id,
                    content,
                    timestamp,
                    version,
                    device_id,
                    metadata_str,
                    event,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut changes: Vec<crate::sync::SyncChange> = rows
            .into_iter()
            .map(
                |(id, content, timestamp, version, device_id, metadata_str, event)| {
                    let operation = match event.as_str() {
                        "ADD" => crate::sync::SyncOperation::Create,
                        "UPDATE" => crate::sync::SyncOperation::Update,
                        "DELETE" => crate::sync::SyncOperation::Delete,
                        _ => crate::sync::SyncOperation::Create,
                    };
                    let metadata = metadata_str.and_then(|s| serde_json::from_str(&s).ok());

                    crate::sync::SyncChange {
                        memory_id: id,
                        operation,
                        content: Some(content),
                        metadata,
                        timestamp,
                        device_id: device_id.unwrap_or_default(),
                        sync_version: version as u64,
                    }
                },
            )
            .collect();

        // 2. Query history table for DELETE events whose memory no longer exists
        //    (tombstones). These won't appear in the memories table query above.
        let delete_sql = r#"
            SELECT h.memory_id, h.old_memory, CAST(h.created_at AS VARCHAR)
            FROM history h
            WHERE h.event = 'DELETE'
              AND h.created_at > (
                  SELECT COALESCE(MAX(m2.updated_at), '1970-01-01'::TIMESTAMP)
                  FROM memories m2
                  WHERE m2.sync_version = $1
              )
              AND NOT EXISTS (SELECT 1 FROM memories m3 WHERE m3.id = h.memory_id)
        "#;

        let mut del_stmt = conn.prepare(delete_sql)?;
        let del_rows = del_stmt
            .query_map(params![since_version as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Use the max sync_version + 1 for tombstone entries
        let max_version = changes
            .iter()
            .map(|c| c.sync_version)
            .max()
            .unwrap_or(since_version);
        for (i, (memory_id, _old_content, timestamp)) in del_rows.into_iter().enumerate() {
            changes.push(crate::sync::SyncChange {
                memory_id,
                operation: crate::sync::SyncOperation::Delete,
                content: None,
                metadata: None,
                timestamp,
                device_id: String::new(),
                sync_version: max_version + 1 + i as u64,
            });
        }

        Ok(changes)
    }

    /// Get storage statistics: counts of memories, entities, relationships,
    /// estimated size, and embedding dimensions.
    pub(crate) fn get_storage_stats(&self) -> Result<crate::sync::StorageStats> {
        let collection = &self.config.collection_name;

        let conn = self.read_conn();
        let mem_count: i64 = {
            let mut stmt = conn.prepare("SELECT COUNT(*) FROM memories")?;
            stmt.query_map([], |row| row.get(0))?
                .next()
                .expect("aggregate always returns a row")?
        };

        let entity_count: i64 = {
            let sql = format!("SELECT COUNT(*) FROM entities_{collection}");
            let mut stmt = conn.prepare(&sql)?;
            stmt.query_map([], |row| row.get(0))?
                .next()
                .expect("aggregate always returns a row")?
        };

        let rel_count: i64 = {
            let sql = format!("SELECT COUNT(*) FROM relationships_{collection}");
            let mut stmt = conn.prepare(&sql)?;
            stmt.query_map([], |row| row.get(0))?
                .next()
                .expect("aggregate always returns a row")?
        };

        // Estimate size using the already-held connection to avoid Mutex re-entry deadlock.
        let estimated_size = if self.config.db_path == ":memory:" {
            mem_count as u64 * 1024
        } else {
            match std::fs::metadata(&self.config.db_path) {
                Ok(meta) => meta.len(),
                Err(_) => 0,
            }
        };

        Ok(crate::sync::StorageStats {
            total_memories: mem_count as u64,
            total_entities: entity_count as u64,
            total_relationships: rel_count as u64,
            estimated_size_bytes: estimated_size,
            embedding_dims: self.config.embedding_dims,
        })
    }

    /// Assign the next sync_version to a memory by ID.
    /// Called after insert or update to track changes for sync.
    ///
    /// Uses a single write_conn to read max version and update atomically,
    /// avoiding TOCTOU race between read_conn and write_conn.
    pub(crate) fn bump_sync_version(
        &self,
        memory_id: &str,
        device_id: Option<&str>,
    ) -> Result<u64> {
        let dev_val: duckdb::types::Value = match device_id {
            Some(d) => duckdb::types::Value::Text(d.to_string()),
            None => duckdb::types::Value::Null,
        };
        let conn = self.write_conn();
        let mut stmt = conn.prepare("SELECT COALESCE(MAX(sync_version), 0) FROM memories")?;
        let max_version: i64 = stmt
            .query_map([], |row| row.get(0))?
            .next()
            .expect("aggregate always returns a row")?;
        let next_version = (max_version as u64) + 1;
        conn.execute(
            "UPDATE memories SET sync_version = $1, device_id = COALESCE($2, device_id), sync_status = 'pending' WHERE id = $3",
            params![next_version as i64, dev_val, memory_id],
        )?;
        Ok(next_version)
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
    fn test_sync_version_starts_at_zero() {
        let storage = open_storage(384);
        let version = storage.get_max_sync_version().unwrap();
        assert_eq!(version, 0);
    }

    #[test]
    fn test_bump_sync_version() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);
        storage
            .insert_memory(
                "id1",
                "hello",
                &emb,
                "user1",
                "h1",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        let v1 = storage.bump_sync_version("id1", Some("device-a")).unwrap();
        assert_eq!(v1, 1);

        let v2 = storage.bump_sync_version("id1", Some("device-a")).unwrap();
        assert_eq!(v2, 2);

        let max = storage.get_max_sync_version().unwrap();
        assert_eq!(max, 2);
    }

    #[test]
    fn test_get_changes_since() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);
        storage
            .insert_memory(
                "id1",
                "hello",
                &emb,
                "user1",
                "h1",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        storage
            .record_history("hist1", "id1", "user1", None, "hello", "ADD")
            .unwrap();
        storage.bump_sync_version("id1", Some("dev-a")).unwrap();

        let emb2 = dummy_embedding(384, 2.0);
        storage
            .insert_memory(
                "id2",
                "world",
                &emb2,
                "user1",
                "h2",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        storage
            .record_history("hist2", "id2", "user1", None, "world", "ADD")
            .unwrap();
        storage.bump_sync_version("id2", Some("dev-a")).unwrap();

        let changes = storage.get_changes_since(0).unwrap();
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].memory_id, "id1");
        assert_eq!(changes[0].operation, crate::sync::SyncOperation::Create);
        assert_eq!(changes[1].memory_id, "id2");

        let changes = storage.get_changes_since(1).unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].memory_id, "id2");
    }

    #[test]
    fn test_get_storage_stats() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);
        storage
            .insert_memory(
                "id1",
                "hello",
                &emb,
                "user1",
                "h1",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        storage
            .insert_memory(
                "id2",
                "world",
                &dummy_embedding(384, 2.0),
                "user1",
                "h2",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        let stats = storage.get_storage_stats().unwrap();
        assert_eq!(stats.total_memories, 2);
        assert_eq!(stats.total_entities, 0);
        assert_eq!(stats.total_relationships, 0);
        assert_eq!(stats.embedding_dims, 384);
    }

    #[test]
    fn test_sync_columns_exist_after_migration() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);
        storage
            .insert_memory(
                "id1",
                "hello",
                &emb,
                "user1",
                "h1",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        let conn = storage.read_conn();
        let mut stmt = conn
            .prepare("SELECT sync_version, device_id, sync_status FROM memories WHERE id = 'id1'")
            .unwrap();
        let mut rows = stmt
            .query_map([], |row| {
                let version: i64 = row.get(0)?;
                let device_id: Option<String> = row.get(1)?;
                let status: String = row.get(2)?;
                Ok((version, device_id, status))
            })
            .unwrap();
        let (version, device_id, status) = rows.next().unwrap().unwrap();
        assert_eq!(version, 0);
        assert!(device_id.is_none());
        assert_eq!(status, "pending");
    }
}

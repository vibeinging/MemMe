use crate::error::Result;
use crate::types::SqlParam;

use super::util::opt_text;
use super::Storage;

impl Storage {
    /// Get the maximum sync_version across all memories.
    pub(crate) fn get_max_sync_version(&self) -> Result<u64> {
        let version = self.backend.query_count(
            "SELECT COALESCE(MAX(sync_version), 0) FROM memories",
            &[],
        )?;
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
            SELECT m.id, m.content, m.updated_at,
                   m.sync_version, m.device_id,
                   m.metadata,
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

        let rows = self.backend.query_read(
            sql,
            &[SqlParam::Int(since_version as i64)],
            |row| {
                Ok((
                    row.get_string(0)?,
                    row.get_string(1)?,
                    row.get_string(2)?,
                    row.get_i64(3)?,
                    row.get_opt_string(4)?,
                    row.get_opt_string(5)?,
                    row.get_string(6)?,
                ))
            },
        )?;

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
            SELECT h.memory_id, h.old_memory, h.created_at
            FROM history h
            WHERE h.event = 'DELETE'
              AND h.created_at > (
                  SELECT COALESCE(MAX(m2.updated_at), '1970-01-01')
                  FROM memories m2
                  WHERE m2.sync_version = $1
              )
              AND NOT EXISTS (SELECT 1 FROM memories m3 WHERE m3.id = h.memory_id)
        "#;

        let del_rows = self.backend.query_read(
            delete_sql,
            &[SqlParam::Int(since_version as i64)],
            |row| {
                Ok((
                    row.get_string(0)?,
                    row.get_opt_string(1)?,
                    row.get_string(2)?,
                ))
            },
        )?;

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

        let mem_count = self.backend.query_count("SELECT COUNT(*) FROM memories", &[])?;

        let entity_sql = format!("SELECT COUNT(*) FROM entities_{collection}");
        let entity_count = self.backend.query_count(&entity_sql, &[])?;

        let rel_sql = format!("SELECT COUNT(*) FROM relationships_{collection}");
        let rel_count = self.backend.query_count(&rel_sql, &[])?;

        // Estimate size
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
    /// Reads max version and then executes update,
    /// ensuring both operations go through the write path.
    pub(crate) fn bump_sync_version(
        &self,
        memory_id: &str,
        device_id: Option<&str>,
    ) -> Result<u64> {
        let max_version = self.backend.query_count(
            "SELECT COALESCE(MAX(sync_version), 0) FROM memories",
            &[],
        )?;
        let next_version = (max_version as u64) + 1;
        let dev_val = opt_text(device_id);
        self.backend.execute(
            "UPDATE memories SET sync_version = $1, device_id = COALESCE($2, device_id), sync_status = 'pending' WHERE id = $3",
            &[
                SqlParam::Int(next_version as i64),
                dev_val,
                SqlParam::Text(memory_id.to_string()),
            ],
        )?;
        Ok(next_version)
    }
}

#[cfg(test)]
mod tests {
    use crate::config::MemoryConfig;
    use crate::storage::InsertMemoryParams;
    use crate::types::SqlParam;

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

        let rows = storage.backend.query_read(
            "SELECT sync_version, device_id, sync_status FROM memories WHERE id = $1",
            &[SqlParam::Text("id1".to_string())],
            |row| {
                let version = row.get_i64(0)?;
                let device_id = row.get_opt_string(1)?;
                let status = row.get_string(2)?;
                Ok((version, device_id, status))
            },
        ).unwrap();
        let (version, device_id, status) = rows.into_iter().next().unwrap();
        assert_eq!(version, 0);
        assert!(device_id.is_none());
        assert_eq!(status, "pending");
    }
}

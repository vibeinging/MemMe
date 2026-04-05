use duckdb::params;

use crate::error::Result;

use super::util::opt_text;
use super::Storage;

impl Storage {
    pub(crate) fn record_history(
        &self,
        id: &str,
        memory_id: &str,
        user_id: &str,
        old_memory: Option<&str>,
        new_memory: &str,
        event: &str,
    ) -> Result<()> {
        let old_val = opt_text(old_memory);

        let conn = self.write_conn();
        conn.execute(
            "INSERT INTO history (id, memory_id, user_id, old_memory, new_memory, event) VALUES ($1, $2, $3, $4, $5, $6)",
            params![id, memory_id, user_id, old_val, new_memory, event],
        )?;
        Ok(())
    }

    /// Get change history for a specific memory.
    pub(crate) fn get_history(&self, memory_id: &str) -> Result<Vec<crate::types::HistoryRecord>> {
        let sql = "SELECT id, memory_id, old_memory, new_memory, event, CAST(created_at AS VARCHAR) FROM history WHERE memory_id = $1 ORDER BY created_at";
        let conn = self.read_conn();
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt
            .query_map(params![memory_id], |row| {
                Ok(crate::types::HistoryRecord {
                    id: row.get(0)?,
                    memory_id: row.get(1)?,
                    old_memory: row.get(2)?,
                    new_memory: row.get(3)?,
                    event: row.get(4)?,
                    created_at: row.get(5)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Delete all memories for a user (optionally scoped by agent_id/run_id/app_id).
    /// Returns the number of deleted memories.
    pub(crate) fn delete_all_memories(
        &self,
        user_id: &str,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        app_id: Option<&str>,
    ) -> Result<u64> {
        // Build WHERE clause dynamically
        let mut conditions = vec!["user_id = $1".to_string()];
        let mut dynamic_params: Vec<duckdb::types::Value> =
            vec![duckdb::types::Value::Text(user_id.to_string())];
        let mut param_idx: usize = 1;

        if let Some(aid) = agent_id {
            param_idx += 1;
            conditions.push(format!("agent_id = ${param_idx}"));
            dynamic_params.push(duckdb::types::Value::Text(aid.to_string()));
        }

        if let Some(rid) = run_id {
            param_idx += 1;
            conditions.push(format!("run_id = ${param_idx}"));
            dynamic_params.push(duckdb::types::Value::Text(rid.to_string()));
        }

        if let Some(appid) = app_id {
            param_idx += 1;
            conditions.push(format!("app_id = ${param_idx}"));
            dynamic_params.push(duckdb::types::Value::Text(appid.to_string()));
        }
        let _ = param_idx;

        let where_clause = conditions.join(" AND ");

        // First, collect the ids and content of memories to be deleted (for history)
        let select_sql = format!("SELECT id, content FROM memories WHERE {where_clause}");
        let delete_sql = format!("DELETE FROM memories WHERE {where_clause}");

        let conn = self.write_conn();
        let mut stmt = conn.prepare(&select_sql)?;
        let param_refs: Vec<&dyn duckdb::ToSql> = dynamic_params
            .iter()
            .map(|p| p as &dyn duckdb::ToSql)
            .collect();
        let rows: Vec<(String, String)> = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let count = rows.len() as u64;

        // Delete the memories
        let mut del_stmt = conn.prepare(&delete_sql)?;
        let param_refs2: Vec<&dyn duckdb::ToSql> = dynamic_params
            .iter()
            .map(|p| p as &dyn duckdb::ToSql)
            .collect();
        del_stmt.execute(param_refs2.as_slice())?;

        // Record history for each deleted memory (inline to avoid re-locking conn)
        for (mem_id, content) in &rows {
            let history_id = uuid::Uuid::new_v4().to_string();
            conn.execute(
                "INSERT INTO history (id, memory_id, user_id, old_memory, new_memory, event) VALUES ($1, $2, $3, $4, $5, $6)",
                params![history_id, mem_id, user_id, content, "", "DELETE"],
            )?;
        }

        Ok(count)
    }

    /// Delete all history records for a user.
    #[allow(dead_code)] // planned API: user data cleanup
    pub(crate) fn delete_user_history(&self, user_id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute("DELETE FROM history WHERE user_id = $1", params![user_id])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use duckdb::params;

    use crate::config::MemoryConfig;

    use super::Storage;

    fn open_storage() -> Storage {
        let config = MemoryConfig {
            db_path: ":memory:".into(),
            collection_name: "test".into(),
            embedding_dims: 384,
            dedup_threshold: 0.15,
            default_limit: 10,
            ..Default::default()
        };
        Storage::open(config).unwrap()
    }

    #[test]
    fn test_record_history() {
        let storage = open_storage();
        storage
            .record_history("hist1", "mem1", "user1", None, "new content", "ADD")
            .unwrap();
        storage
            .record_history(
                "hist2",
                "mem1",
                "user1",
                Some("new content"),
                "updated",
                "UPDATE",
            )
            .unwrap();

        // Verify history entries exist by querying
        let conn = storage.read_conn();
        let mut stmt = conn
            .prepare("SELECT COUNT(*) FROM history WHERE memory_id = $1")
            .unwrap();
        let count: i64 = stmt
            .query_map(params!["mem1"], |row| row.get(0))
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(count, 2);
    }
}

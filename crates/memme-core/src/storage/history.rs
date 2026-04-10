use crate::error::Result;
use crate::types::SqlParam;

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

        self.backend.execute(
            "INSERT INTO history (id, memory_id, user_id, old_memory, new_memory, event) VALUES ($1, $2, $3, $4, $5, $6)",
            &[
                SqlParam::Text(id.to_string()),
                SqlParam::Text(memory_id.to_string()),
                SqlParam::Text(user_id.to_string()),
                old_val,
                SqlParam::Text(new_memory.to_string()),
                SqlParam::Text(event.to_string()),
            ],
        )?;
        Ok(())
    }

    /// Get change history for a specific memory.
    pub(crate) fn get_history(&self, memory_id: &str) -> Result<Vec<crate::types::HistoryRecord>> {
        let sql = "SELECT id, memory_id, old_memory, new_memory, event, created_at FROM history WHERE memory_id = $1 ORDER BY created_at";
        self.backend
            .query_read(sql, &[SqlParam::Text(memory_id.to_string())], |row| {
                Ok(crate::types::HistoryRecord {
                    id: row.get_string(0)?,
                    memory_id: row.get_string(1)?,
                    old_memory: row.get_opt_string(2)?,
                    new_memory: row.get_string(3)?,
                    event: row.get_string(4)?,
                    created_at: row.get_string(5)?,
                })
            })
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
        let mut dynamic_params: Vec<SqlParam> = vec![SqlParam::Text(user_id.to_string())];
        let mut param_idx: usize = 1;

        if let Some(aid) = agent_id {
            param_idx += 1;
            conditions.push(format!("agent_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(aid.to_string()));
        }

        if let Some(rid) = run_id {
            param_idx += 1;
            conditions.push(format!("run_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(rid.to_string()));
        }

        if let Some(appid) = app_id {
            param_idx += 1;
            conditions.push(format!("app_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(appid.to_string()));
        }
        let _ = param_idx;

        let where_clause = conditions.join(" AND ");

        // First, collect the ids and content of memories to be deleted (for history)
        let select_sql = format!("SELECT id, content FROM memories WHERE {where_clause}");
        let delete_sql = format!("DELETE FROM memories WHERE {where_clause}");

        let rows: Vec<(String, String)> =
            self.backend
                .query_read(&select_sql, &dynamic_params, |row| {
                    Ok((row.get_string(0)?, row.get_string(1)?))
                })?;

        let count = rows.len() as u64;

        // Delete the memories
        self.backend.execute(&delete_sql, &dynamic_params)?;

        // Record history for each deleted memory
        for (mem_id, content) in &rows {
            let history_id = uuid::Uuid::new_v4().to_string();
            self.backend.execute(
                "INSERT INTO history (id, memory_id, user_id, old_memory, new_memory, event) VALUES ($1, $2, $3, $4, $5, $6)",
                &[
                    SqlParam::Text(history_id),
                    SqlParam::Text(mem_id.clone()),
                    SqlParam::Text(user_id.to_string()),
                    SqlParam::Text(content.clone()),
                    SqlParam::Text(String::new()),
                    SqlParam::Text("DELETE".to_string()),
                ],
            )?;
        }

        Ok(count)
    }

    /// Delete all history records for a user.
    #[allow(dead_code)] // planned API: user data cleanup
    pub(crate) fn delete_user_history(&self, user_id: &str) -> Result<()> {
        self.backend.execute(
            "DELETE FROM history WHERE user_id = $1",
            &[SqlParam::Text(user_id.to_string())],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::config::MemoryConfig;
    use crate::types::SqlParam;

    use super::Storage;

    fn open_storage() -> Storage {
        let config = MemoryConfig::new(":memory:", 384);
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
        let count: Vec<i64> = storage
            .backend
            .query_read(
                "SELECT COUNT(*) FROM history WHERE memory_id = $1",
                &[SqlParam::Text("mem1".to_string())],
                |row| row.get_i64(0),
            )
            .unwrap();
        assert_eq!(count[0], 2);
    }
}

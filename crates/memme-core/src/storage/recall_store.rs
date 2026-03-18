use duckdb::params;

use crate::error::Result;
use crate::types::RecallRecord;

use super::Storage;

impl Storage {
    /// Insert a recall record.
    #[allow(dead_code)] // planned API: recall tracking
    pub(crate) fn insert_recall(
        &self,
        recall_id: &str,
        query: &str,
        query_vec: &[f32],
        source_id: Option<&str>,
        user_id: &str,
        results: Option<&serde_json::Value>,
    ) -> Result<()> {
        let emb_literal = Self::format_embedding(query_vec, self.config.embedding_dims)?;
        let source_val: duckdb::types::Value = match source_id {
            Some(s) => duckdb::types::Value::Text(s.to_string()),
            None => duckdb::types::Value::Null,
        };
        let results_val: duckdb::types::Value = match results {
            Some(r) => duckdb::types::Value::Text(serde_json::to_string(r).unwrap_or_default()),
            None => duckdb::types::Value::Null,
        };

        let sql = format!(
            r#"INSERT INTO recalls (recall_id, query, query_vec, source_id, user_id, results)
               VALUES ($1, $2, {emb_literal}, $3, $4, $5)"#
        );
        let conn = self.write_conn();
        conn.execute(
            &sql,
            params![recall_id, query, source_val, user_id, results_val],
        )?;
        Ok(())
    }

    /// Update recall feedback.
    #[allow(dead_code)] // planned API: recall feedback
    pub(crate) fn update_recall_feedback(&self, recall_id: &str, feedback: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            "UPDATE recalls SET feedback = $1 WHERE recall_id = $2",
            params![feedback, recall_id],
        )?;
        Ok(())
    }

    /// List recall records for a user.
    #[allow(dead_code)] // planned API: recall history
    pub(crate) fn list_recalls(&self, user_id: &str, limit: usize) -> Result<Vec<RecallRecord>> {
        let sql = format!(
            r#"SELECT recall_id, query, CAST(timestamp AS VARCHAR), source_id, user_id, results, feedback
               FROM recalls WHERE user_id = $1
               ORDER BY timestamp DESC
               LIMIT {limit}"#
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![user_id], |row| {
                Ok(RecallRecord {
                    recall_id: row.get(0)?,
                    query: row.get(1)?,
                    timestamp: row.get::<_, String>(2)?,
                    source_id: row.get::<_, Option<String>>(3)?,
                    user_id: row.get(4)?,
                    results: row
                        .get::<_, Option<String>>(5)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                    feedback: row.get::<_, Option<String>>(6)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }
}

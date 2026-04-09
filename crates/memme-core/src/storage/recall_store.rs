use crate::error::Result;
use crate::types::{RecallRecord, SqlParam};

use super::util::opt_text;
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
        let emb_literal = self.format_embedding(query_vec, self.config.embedding_dims)?;
        let source_val = opt_text(source_id);
        let results_val = opt_text(results.map(|r| serde_json::to_string(r).unwrap_or_default()));

        let sql = format!(
            r#"INSERT INTO recalls (recall_id, query, query_vec, source_id, user_id, results)
               VALUES ($1, $2, {emb_literal}, $3, $4, $5)"#
        );
        self.backend.execute(
            &sql,
            &[
                SqlParam::Text(recall_id.to_string()),
                SqlParam::Text(query.to_string()),
                source_val,
                SqlParam::Text(user_id.to_string()),
                results_val,
            ],
        )?;
        Ok(())
    }

    /// Update recall feedback.
    #[allow(dead_code)] // planned API: recall feedback
    pub(crate) fn update_recall_feedback(&self, recall_id: &str, feedback: &str) -> Result<()> {
        self.backend.execute(
            "UPDATE recalls SET feedback = $1 WHERE recall_id = $2",
            &[
                SqlParam::Text(feedback.to_string()),
                SqlParam::Text(recall_id.to_string()),
            ],
        )?;
        Ok(())
    }

    /// List recall records for a user.
    #[allow(dead_code)] // planned API: recall history
    pub(crate) fn list_recalls(&self, user_id: &str, limit: usize) -> Result<Vec<RecallRecord>> {
        let sql = format!(
            r#"SELECT recall_id, query, timestamp, source_id, user_id, results, feedback
               FROM recalls WHERE user_id = $1
               ORDER BY timestamp DESC
               LIMIT {limit}"#
        );
        self.backend.query_read(
            &sql,
            &[SqlParam::Text(user_id.to_string())],
            |row| {
                let results_str = row.get_opt_string(5)?;
                Ok(RecallRecord {
                    recall_id: row.get_string(0)?,
                    query: row.get_string(1)?,
                    timestamp: row.get_string(2)?,
                    source_id: row.get_opt_string(3)?,
                    user_id: row.get_string(4)?,
                    results: results_str.and_then(|s| serde_json::from_str(&s).ok()),
                    feedback: row.get_opt_string(6)?,
                })
            },
        )
    }
}

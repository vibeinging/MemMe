use duckdb::params;

use crate::error::Result;

use super::{ProcedureRow, Storage};

impl Storage {
    /// Insert a new procedure.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn insert_procedure(
        &self,
        id: &str,
        name: &str,
        description: &str,
        steps_json: &str,
        user_id: &str,
        trigger_pattern: Option<&str>,
        confidence: f32,
    ) -> Result<()> {
        let trigger_val: duckdb::types::Value = match trigger_pattern {
            Some(t) => duckdb::types::Value::Text(t.to_string()),
            None => duckdb::types::Value::Null,
        };
        let conn = self.write_conn();
        conn.execute(
            "INSERT INTO procedures (id, name, description, steps, user_id, trigger_pattern, confidence)
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
            params![id, name, description, steps_json, user_id, trigger_val, confidence as f64],
        )?;
        Ok(())
    }

    /// Get a procedure by ID.
    pub(crate) fn get_procedure(&self, id: &str) -> Result<Option<ProcedureRow>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, name, description, steps, user_id, trigger_pattern, confidence, usage_count,
                    CAST(created_at AS VARCHAR) AS created_at,
                    CAST(updated_at AS VARCHAR) AS updated_at
             FROM procedures WHERE id = $1"
        )?;
        let mut rows = stmt.query_map(params![id], |row| {
            Ok(ProcedureRow {
                id: row.get(0)?,
                name: row.get(1)?,
                description: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                steps: row.get::<_, Option<String>>(3)?,
                user_id: row.get(4)?,
                trigger_pattern: row.get(5)?,
                confidence: row.get::<_, Option<f64>>(6)?.unwrap_or(0.5) as f32,
                usage_count: row.get::<_, Option<i32>>(7)?.unwrap_or(0) as u32,
                created_at: row.get(8)?,
                updated_at: row.get(9)?,
            })
        })?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// List all procedures for a user.
    pub(crate) fn list_procedures(&self, user_id: &str) -> Result<Vec<ProcedureRow>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, name, description, steps, user_id, trigger_pattern, confidence, usage_count,
                    CAST(created_at AS VARCHAR) AS created_at,
                    CAST(updated_at AS VARCHAR) AS updated_at
             FROM procedures WHERE user_id = $1
             ORDER BY updated_at DESC"
        )?;
        let rows = stmt
            .query_map(params![user_id], |row| {
                Ok(ProcedureRow {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    description: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    steps: row.get::<_, Option<String>>(3)?,
                    user_id: row.get(4)?,
                    trigger_pattern: row.get(5)?,
                    confidence: row.get::<_, Option<f64>>(6)?.unwrap_or(0.5) as f32,
                    usage_count: row.get::<_, Option<i32>>(7)?.unwrap_or(0) as u32,
                    created_at: row.get(8)?,
                    updated_at: row.get(9)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Delete a procedure by ID.
    pub(crate) fn delete_procedure(&self, id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute("DELETE FROM procedures WHERE id = $1", params![id])?;
        Ok(())
    }
}

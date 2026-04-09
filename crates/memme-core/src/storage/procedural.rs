use crate::error::Result;
use crate::types::SqlParam;

use super::util::opt_text;
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
        let trigger_val = opt_text(trigger_pattern);
        self.backend.execute(
            "INSERT INTO procedures (id, name, description, steps, user_id, trigger_pattern, confidence)
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
            &[
                SqlParam::Text(id.to_string()),
                SqlParam::Text(name.to_string()),
                SqlParam::Text(description.to_string()),
                SqlParam::Text(steps_json.to_string()),
                SqlParam::Text(user_id.to_string()),
                trigger_val,
                SqlParam::Float(confidence as f64),
            ],
        )?;
        Ok(())
    }

    /// Get a procedure by ID.
    pub(crate) fn get_procedure(&self, id: &str) -> Result<Option<ProcedureRow>> {
        self.backend.query_one(
            "SELECT id, name, description, steps, user_id, trigger_pattern, confidence, usage_count,
                    created_at,
                    updated_at
             FROM procedures WHERE id = $1",
            &[SqlParam::Text(id.to_string())],
            |row| {
                Ok(ProcedureRow {
                    id: row.get_string(0)?,
                    name: row.get_string(1)?,
                    description: row.get_opt_string(2)?.unwrap_or_default(),
                    steps: row.get_opt_string(3)?,
                    user_id: row.get_string(4)?,
                    trigger_pattern: row.get_opt_string(5)?,
                    confidence: row.get_opt_f64(6)?.unwrap_or(0.5) as f32,
                    usage_count: row.get_opt_i64(7)?.unwrap_or(0) as u32,
                    created_at: row.get_string(8)?,
                    updated_at: row.get_string(9)?,
                })
            },
        )
    }

    /// List all procedures for a user.
    pub(crate) fn list_procedures(&self, user_id: &str) -> Result<Vec<ProcedureRow>> {
        self.backend.query_read(
            "SELECT id, name, description, steps, user_id, trigger_pattern, confidence, usage_count,
                    created_at,
                    updated_at
             FROM procedures WHERE user_id = $1
             ORDER BY updated_at DESC",
            &[SqlParam::Text(user_id.to_string())],
            |row| {
                Ok(ProcedureRow {
                    id: row.get_string(0)?,
                    name: row.get_string(1)?,
                    description: row.get_opt_string(2)?.unwrap_or_default(),
                    steps: row.get_opt_string(3)?,
                    user_id: row.get_string(4)?,
                    trigger_pattern: row.get_opt_string(5)?,
                    confidence: row.get_opt_f64(6)?.unwrap_or(0.5) as f32,
                    usage_count: row.get_opt_i64(7)?.unwrap_or(0) as u32,
                    created_at: row.get_string(8)?,
                    updated_at: row.get_string(9)?,
                })
            },
        )
    }

    /// Delete a procedure by ID.
    pub(crate) fn delete_procedure(&self, id: &str) -> Result<()> {
        self.backend.execute(
            "DELETE FROM procedures WHERE id = $1",
            &[SqlParam::Text(id.to_string())],
        )?;
        Ok(())
    }
}

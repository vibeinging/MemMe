use crate::error::{MemoryError, Result};
use crate::procedural::Procedure;

use super::helpers::procedure_row_to_result;

impl super::MemoryStore {
    /// Add a new procedure (skill/habit/workflow).
    pub fn add_procedure(
        &self,
        name: &str,
        description: &str,
        steps: Vec<crate::procedural::ProcedureStep>,
        user_id: &str,
    ) -> Result<Procedure> {
        let id = uuid::Uuid::new_v4().to_string();
        let steps_json = serde_json::to_string(&steps)?;
        self.storage
            .insert_procedure(&id, name, description, &steps_json, user_id, None, 0.5)?;
        let row = self
            .storage
            .get_procedure(&id)?
            .ok_or_else(|| MemoryError::NotFound(id))?;
        Ok(procedure_row_to_result(row))
    }

    /// Get a procedure by ID.
    pub fn get_procedure(&self, id: &str) -> Result<Option<Procedure>> {
        Ok(self.storage.get_procedure(id)?.map(procedure_row_to_result))
    }

    /// List all procedures for a user.
    pub fn list_procedures(&self, user_id: &str) -> Result<Vec<Procedure>> {
        let rows = self.storage.list_procedures(user_id)?;
        Ok(rows.into_iter().map(procedure_row_to_result).collect())
    }

    /// Delete a procedure by ID.
    pub fn delete_procedure(&self, id: &str) -> Result<()> {
        self.storage.delete_procedure(id)
    }
}

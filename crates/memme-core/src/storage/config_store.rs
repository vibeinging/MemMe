//! Key-value config persistence in the `memme_config` table.

use crate::error::Result;
use crate::types::SqlParam;

use super::Storage;

impl Storage {
    /// Set a config key-value pair (upsert).
    pub(crate) fn set_config(&self, key: &str, value: &str) -> Result<()> {
        self.backend.execute(
            "INSERT INTO memme_config (key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            &[SqlParam::Text(key.to_string()), SqlParam::Text(value.to_string())],
        )?;
        Ok(())
    }

    /// Get a config value by key.
    pub(crate) fn get_config(&self, key: &str) -> Result<Option<String>> {
        self.backend.query_one(
            "SELECT value FROM memme_config WHERE key = $1",
            &[SqlParam::Text(key.to_string())],
            |row| row.get_string(0),
        )
    }

    /// Get all config key-value pairs.
    #[allow(dead_code)] // planned API: config inspection
    pub(crate) fn get_all_config(&self) -> Result<std::collections::HashMap<String, String>> {
        let pairs = self
            .backend
            .query_read("SELECT key, value FROM memme_config", &[], |row| {
                let k = row.get_string(0)?;
                let v = row.get_string(1)?;
                Ok((k, v))
            })?;
        Ok(pairs.into_iter().collect())
    }
}

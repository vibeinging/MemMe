//! Key-value config persistence in the `memme_config` table.

use crate::error::Result;

use super::Storage;

impl Storage {
    /// Set a config key-value pair (upsert).
    pub(crate) fn set_config(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            "INSERT INTO memme_config (key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            duckdb::params![key, value],
        )?;
        Ok(())
    }

    /// Get a config value by key.
    pub(crate) fn get_config(&self, key: &str) -> Result<Option<String>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare("SELECT value FROM memme_config WHERE key = $1")?;
        let mut rows = stmt.query_map(duckdb::params![key], |row| row.get::<_, String>(0))?;
        match rows.next() {
            Some(Ok(v)) => Ok(Some(v)),
            _ => Ok(None),
        }
    }

    /// Get all config key-value pairs.
    #[allow(dead_code)] // planned API: config inspection
    pub(crate) fn get_all_config(&self) -> Result<std::collections::HashMap<String, String>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare("SELECT key, value FROM memme_config")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut map = std::collections::HashMap::new();
        for (k, v) in rows.flatten() {
            map.insert(k, v);
        }
        Ok(map)
    }
}

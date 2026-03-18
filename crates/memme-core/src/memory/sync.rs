use crate::error::Result;
use crate::types::MemoryExport;

use super::helpers::chrono_now;

impl super::MemoryStore {
    /// Export memories with optional user_id filter.
    pub fn export(&self, user_id: Option<&str>) -> Result<Vec<MemoryExport>> {
        self.storage.export_memories(user_id)
    }

    /// Import memories from an export. Returns the number of imported memories.
    pub fn import_memories(&self, memories: &[MemoryExport]) -> Result<u64> {
        self.storage.import_memories(memories)
    }

    /// Export memories with control over including local_only privacy memories.
    pub fn export_with_privacy(
        &self,
        user_id: Option<&str>,
        include_local: bool,
    ) -> Result<Vec<MemoryExport>> {
        self.storage
            .export_memories_with_privacy(user_id, include_local)
    }

    /// Export all changes since the given sync version as a `SyncDelta`.
    ///
    /// `device_id` identifies the device requesting the delta (included in
    /// the returned envelope for bookkeeping).
    pub fn export_changes_since(
        &self,
        since_version: u64,
        device_id: &str,
    ) -> Result<crate::sync::SyncDelta> {
        let changes = self.storage.get_changes_since(since_version)?;
        let to_version = self.storage.get_max_sync_version()?;

        Ok(crate::sync::SyncDelta {
            device_id: device_id.to_string(),
            from_version: since_version,
            to_version,
            changes,
            exported_at: chrono_now(),
        })
    }

    /// Return the current (maximum) sync version in the database.
    pub fn current_sync_version(&self) -> Result<u64> {
        self.storage.get_max_sync_version()
    }

    /// Return high-level storage statistics.
    pub fn storage_stats(&self) -> Result<crate::sync::StorageStats> {
        self.storage.get_storage_stats()
    }
}

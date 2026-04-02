use crate::error::Result;
use crate::types::{FullExport, FullImportResult, MemoryExport};

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

    /// Export all data layers (memories, sessions, events, episodes, entities,
    /// relations, identity traits, sources) as a single `FullExport`.
    pub fn full_export(&self, user_id: Option<&str>) -> Result<FullExport> {
        self.storage
            .full_export(user_id, &self.config.collection_name)
    }

    /// Import all data layers from a `FullExport`.
    /// Inserts in dependency order: sources -> sessions -> events -> episodes ->
    /// memories -> entities -> relations -> identity traits.
    /// Existing records (by primary key) are skipped.
    pub fn full_import(&self, export: &FullExport) -> Result<FullImportResult> {
        let sources = self.storage.import_sources(&export.sources)?;
        let sessions = self.storage.import_sessions(&export.sessions)?;
        let events = self.storage.import_events(&export.events)?;
        let episodes = self.storage.import_episodes(&export.episodes)?;
        let memories = self.storage.import_memories(&export.memories)?;
        let entities = self.storage.import_entities(&export.entities)?;
        let relations = self.storage.import_relations(&export.relations)?;
        let identity_traits = self.storage.import_identity_traits(&export.identity_traits)?;
        Ok(FullImportResult {
            sources,
            sessions,
            events,
            episodes,
            memories,
            entities,
            relations,
            identity_traits,
        })
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

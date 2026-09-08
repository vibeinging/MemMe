use crate::error::{MemoryError, Result};
use crate::storage::portable_import::{validate_full_import_envelope, FullImportEmbeddings};
use crate::types::{FullExport, FullImportResult, MemoryExport};

use super::helpers::chrono_now;

fn validate_embedding_batch(
    layer: &str,
    text_count: usize,
    embeddings: &[Vec<f32>],
    expected_dimensions: usize,
) -> Result<()> {
    if embeddings.len() != text_count {
        return Err(MemoryError::Config(format!(
            "embedder returned {} {layer} embeddings for {text_count} texts",
            embeddings.len()
        )));
    }
    if let Some((index, embedding)) = embeddings
        .iter()
        .enumerate()
        .find(|(_, embedding)| embedding.len() != expected_dimensions)
    {
        return Err(MemoryError::Config(format!(
            "embedder returned a {layer} embedding with {} dimensions at index {index}; expected {expected_dimensions}",
            embedding.len()
        )));
    }
    Ok(())
}

fn embed_in_batches(
    embedder: &dyn memme_embeddings::Embedder,
    layer: &str,
    texts: &[&str],
) -> Result<Vec<Vec<f32>>> {
    const BATCH_SIZE: usize = 128;
    let mut result = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(BATCH_SIZE) {
        let embeddings = embedder.embed_batch(chunk)?;
        validate_embedding_batch(layer, chunk.len(), &embeddings, embedder.dimensions())?;
        result.extend(embeddings);
    }
    Ok(result)
}

impl super::MemoryStore {
    /// Export memories with optional user_id filter.
    pub fn export(&self, user_id: Option<&str>) -> Result<Vec<MemoryExport>> {
        self.storage.export_memories(user_id)
    }

    /// Import memories from an export. Returns the number of imported memories.
    pub fn import_memories(&self, memories: &[MemoryExport]) -> Result<u64> {
        let texts: Vec<&str> = memories
            .iter()
            .map(|memory| memory.content.as_str())
            .collect();
        let embeddings = embed_in_batches(self.embedder.as_ref(), "memory", &texts)?;
        self.storage.import_memories_atomic(memories, &embeddings)
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
    /// The export version and collection must match. Any existing primary-key ID
    /// rejects the whole import so no authoritative fields are silently skipped.
    pub fn full_import(&self, export: &FullExport) -> Result<FullImportResult> {
        validate_full_import_envelope(
            export,
            self.embedder.dimensions(),
            &self.config.collection_name,
        )?;
        self.storage.preflight_full_import_target(export)?;
        let memory_texts: Vec<&str> = export
            .memories
            .iter()
            .map(|memory| memory.content.as_str())
            .collect();
        let event_texts: Vec<&str> = export
            .events
            .iter()
            .map(|event| {
                event
                    .purified_content
                    .as_deref()
                    .unwrap_or(event.content.as_str())
            })
            .collect();
        let episode_texts: Vec<&str> = export
            .episodes
            .iter()
            .map(|episode| episode.summary.as_str())
            .collect();
        let identity_texts: Vec<&str> = export
            .identity_traits
            .iter()
            .map(|identity| identity.content.as_str())
            .collect();
        // Finish all embedding calls before writing any imported rows. A remote
        // embedding failure therefore cannot leave a half-imported data set.
        let memory_embeddings = embed_in_batches(self.embedder.as_ref(), "memory", &memory_texts)?;
        let event_embeddings = embed_in_batches(self.embedder.as_ref(), "event", &event_texts)?;
        let episode_embeddings =
            embed_in_batches(self.embedder.as_ref(), "episode", &episode_texts)?;
        let identity_embeddings =
            embed_in_batches(self.embedder.as_ref(), "identity", &identity_texts)?;

        self.storage.import_full_atomic(
            export,
            FullImportEmbeddings {
                memories: &memory_embeddings,
                events: &event_embeddings,
                episodes: &episode_embeddings,
                identity_traits: &identity_embeddings,
            },
        )
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

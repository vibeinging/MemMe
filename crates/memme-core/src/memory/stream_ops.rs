use uuid::Uuid;

use crate::error::{MemoryError, Result};
use crate::types::*;

impl super::MemoryStore {
    /// Register a data source.
    #[allow(dead_code)] // planned API: stream source management
    pub(crate) fn register_source(
        &self,
        source_id: &str,
        source_type: &str,
        name: Option<&str>,
        metadata: Option<serde_json::Value>,
        user_id: &str,
    ) -> Result<Source> {
        self.storage
            .register_source(source_id, source_type, name, metadata.as_ref(), user_id)?;
        self.storage
            .get_source(source_id)?
            .ok_or_else(|| MemoryError::NotFound(source_id.to_string()))
    }

    /// Ingest a raw event into the stream layer.
    /// Note: `append_events()` batch-embeds before calling this. Direct callers
    /// should provide embedding via `insert_event` if search is needed.
    pub fn ingest_event(&self, content: &str, options: IngestEventOptions) -> Result<Event> {
        let event_id = Uuid::new_v4().to_string();

        self.storage
            .insert_event(&event_id, content, &[], &options)?;

        self.storage
            .get_event(&event_id)?
            .ok_or_else(|| MemoryError::NotFound(event_id))
    }

    /// List events with filters.
    pub fn list_events(&self, options: ListEventsOptions) -> Result<Vec<Event>> {
        self.storage.list_events(&options)
    }

    /// Get a single event.
    pub fn get_event(&self, event_id: &str) -> Result<Option<Event>> {
        self.storage.get_event(event_id)
    }
}

use std::sync::atomic::Ordering;

use crate::error::Result;
use crate::storage::InsertMemoryParams;
use crate::types::AddOptions;

use super::helpers::{content_hash, recover_lock};

/// A deferred operation queued when battery is critical.
pub(super) struct DeferredOp {
    pub(super) content: String,
    pub(super) options: AddOptions,
}

impl super::MemoryStore {
    /// Set the current battery level and charging state.
    /// `level` should be between 0.0 (empty) and 1.0 (full).
    pub fn set_battery_level(&self, level: f32, charging: bool) {
        let clamped = level.clamp(0.0, 1.0);
        self.battery_level
            .store((clamped * 100.0) as u32, Ordering::Relaxed);
        self.battery_charging
            .store(if charging { 1 } else { 0 }, Ordering::Relaxed);
    }

    /// Check if the device is in power-saving mode.
    pub fn is_power_save(&self) -> bool {
        if self.battery_charging.load(Ordering::Relaxed) == 1 {
            return false;
        }
        let level = self.battery_level.load(Ordering::Relaxed) as f32 / 100.0;
        if let Some(ref pc) = self.config.tuning.power_config {
            level < pc.full_power_threshold
        } else {
            false
        }
    }

    /// Check if the device is at critical power level.
    pub fn is_critical_power(&self) -> bool {
        if self.battery_charging.load(Ordering::Relaxed) == 1 {
            return false;
        }
        let level = self.battery_level.load(Ordering::Relaxed) as f32 / 100.0;
        if let Some(ref pc) = self.config.tuning.power_config {
            level < pc.power_save_threshold
        } else {
            false
        }
    }

    /// Process all deferred operations. Returns the number successfully processed.
    /// Failed operations are logged but not re-queued (data was already accepted by the caller).
    pub fn process_deferred(&self) -> Result<usize> {
        let ops: Vec<DeferredOp> = {
            let mut queue = recover_lock(&self.deferred_ops, "deferred_ops");
            std::mem::take(&mut *queue)
        };
        let mut processed = 0;
        for op in ops {
            match self.add_internal(&op.content, op.options) {
                Ok(_) => processed += 1,
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to process deferred battery op, skipping")
                }
            }
        }
        Ok(processed)
    }

    /// Get the number of deferred operations waiting in the queue.
    pub fn deferred_count(&self) -> usize {
        recover_lock(&self.deferred_ops, "deferred_ops").len()
    }

    /// Internal add that does not check battery state (used for processing deferred ops).
    fn add_internal(
        &self,
        content: &str,
        options: AddOptions,
    ) -> Result<crate::types::MemoryResult> {
        use crate::dedup::{self, DedupResult};
        use crate::error::MemoryError;
        use uuid::Uuid;

        if content.trim().is_empty() {
            return Err(MemoryError::Config("content cannot be empty".into()));
        }
        let embedding = self
            .embedder
            .embed(content)
            .map_err(MemoryError::Embedding)?;
        let hash = content_hash(content);
        let dedup_result = dedup::check_dedup(
            &self.storage,
            &embedding,
            &hash,
            content,
            &options.user_id,
            options.agent_id.as_deref(),
            self.config.tuning.dedup_threshold,
        )?;
        match dedup_result {
            DedupResult::Duplicate { existing_id, .. } => {
                let meta_update = Some(options.metadata.as_ref());
                self.storage.update_memory(
                    &existing_id,
                    content,
                    &embedding,
                    &hash,
                    meta_update,
                    None,
                )?;
                self.get_trace(&existing_id)?
                    .ok_or_else(|| MemoryError::NotFound(existing_id))
            }
            DedupResult::New => {
                let id = Uuid::new_v4().to_string();
                let metadata_str = options
                    .metadata
                    .as_ref()
                    .map(serde_json::to_string)
                    .transpose()?;
                self.storage.insert_memory(
                    &id,
                    content,
                    &embedding,
                    &options.user_id,
                    &hash,
                    &InsertMemoryParams {
                        agent_id: options.agent_id.clone(),
                        run_id: options.run_id.clone(),
                        app_id: options.app_id.clone(),
                        actor_id: options.actor_id.clone(),
                        metadata: metadata_str,
                        importance: options.importance,
                        immutable: options.immutable,
                        expiration_date: options.expiration_date.clone(),
                        categories: options.categories.clone(),
                        memory_type: options.memory_type.clone(),
                        privacy: Some(options.privacy.as_str().to_string()),
                        event_time: options.event_time.clone(),
                        episode_id: options.episode_id.clone(),
                        session_id: options.session_id.clone(),
                        ..Default::default()
                    },
                )?;
                self.get_trace(&id)?
                    .ok_or_else(|| MemoryError::NotFound(id))
            }
        }
    }
}

//! Public API for replica management on MemoryStore.

use crate::error::Result;
use crate::storage::replica;
use crate::types::{ReplicaStatus, ReplicaSyncResult};

impl super::MemoryStore {
    /// Sync the primary database to its replica.
    ///
    /// Performs CHECKPOINT (flush WAL) then atomic file copy.
    /// For `:memory:` databases, returns `Ok(None)`.
    pub fn sync_replica(&self) -> Result<Option<ReplicaSyncResult>> {
        self.storage.sync_replica()
    }

    /// Restore the primary database from its replica.
    ///
    /// **Warning**: The caller must re-open the MemoryStore after calling this.
    pub fn restore_from_replica(&self) -> Result<()> {
        replica::restore_primary_from_replica(&self.storage.config.db_path)
    }

    /// Promote the replica to primary (swap roles).
    ///
    /// **Warning**: The caller must re-open the MemoryStore after calling this.
    pub fn promote_replica(&self) -> Result<()> {
        replica::promote_replica(&self.storage.config.db_path)
    }

    /// Get the status of the primary and replica files.
    pub fn replica_status(&self) -> Result<ReplicaStatus> {
        self.storage.replica_status()
    }
}

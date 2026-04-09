//! Public API for replica management on MemoryStore.

use crate::error::Result;
use crate::storage::replica;
use crate::types::{BackupInfo, ReplicaStatus, ReplicaSyncResult};

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

    /// Backup the database to a user-specified path.
    ///
    /// Performs CHECKPOINT (flush WAL) then atomic file copy.
    /// The resulting file is a complete, self-contained SQLite database
    /// that can be uploaded to cloud storage by the host application.
    ///
    /// Returns metadata about the backup (size, memory count, schema version).
    /// For `:memory:` databases, returns an error.
    pub fn backup_to_path(&self, path: &str) -> Result<BackupInfo> {
        self.storage.backup_to_path(path)
    }

    /// Restore the primary database from a backup file.
    ///
    /// Validates the backup is a readable database file, then performs
    /// an atomic copy to the primary database path.
    ///
    /// **Warning**: The caller must re-open the `MemoryStore` after calling this,
    /// as the underlying database file has been replaced.
    pub fn restore_from_backup(
        backup_path: &str,
        config: &crate::config::MemoryConfig,
    ) -> Result<()> {
        replica::restore_from_backup(backup_path, &config.db_path)
    }
}

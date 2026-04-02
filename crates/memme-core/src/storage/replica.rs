//! Replica management: CHECKPOINT + atomic file copy for data safety.
//!
//! Guarantees that at least one complete copy of the database exists on disk
//! at all times. The replica is a full binary copy of the primary .duckdb file.

use std::fs;
use std::path::Path;

use tracing::{info, warn};

use crate::error::{MemoryError, Result};
use crate::types::{ReplicaStatus, ReplicaSyncResult};

use super::Storage;

/// Derive the replica path from the primary db path: `{path}.replica`.
pub(crate) fn replica_path_for(db_path: &str) -> String {
    format!("{db_path}.replica")
}

impl Storage {
    /// Whether this is a file-backed database (not :memory:).
    fn is_file_backed(&self) -> bool {
        self.config.db_path != ":memory:"
    }

    /// Sync primary → replica using CHECKPOINT + atomic file copy.
    pub(crate) fn sync_replica(&self) -> Result<Option<ReplicaSyncResult>> {
        if !self.is_file_backed() {
            return Ok(None);
        }

        let primary = &self.config.db_path;
        let replica = replica_path_for(primary);

        {
            let conn = self.write_conn();
            conn.execute_batch("CHECKPOINT")
                .map_err(MemoryError::DuckDb)?;
        }
        // Lock released — file copy does not block reads/writes

        atomic_copy(primary, &replica)?;

        let synced_at = chrono::Utc::now().to_rfc3339();
        let _ = self.set_config("replica_last_synced_at", &synced_at);

        let size_bytes = fs::metadata(primary)
            .map(|m| m.len())
            .unwrap_or(0);

        info!(primary = %primary, replica = %replica, size_bytes, "Replica synced");

        Ok(Some(ReplicaSyncResult {
            primary_path: primary.clone(),
            replica_path: replica,
            size_bytes,
            synced_at,
        }))
    }

    /// Get the status of primary and replica files.
    pub(crate) fn replica_status(&self) -> Result<ReplicaStatus> {
        if !self.is_file_backed() {
            return Ok(ReplicaStatus {
                primary_path: self.config.db_path.clone(),
                primary_ok: true, // :memory: is always ok if we got here
                primary_size_bytes: 0,
                replica_path: None,
                replica_ok: false,
                replica_size_bytes: None,
                last_synced_at: None,
            });
        }

        let primary = &self.config.db_path;
        let replica = replica_path_for(primary);

        let primary_meta = fs::metadata(primary);
        let replica_meta = fs::metadata(&replica);
        let last_synced = self.get_config("replica_last_synced_at").ok().flatten();

        Ok(ReplicaStatus {
            primary_path: primary.clone(),
            primary_ok: primary_meta.as_ref().map(|m| m.is_file()).unwrap_or(false),
            primary_size_bytes: primary_meta.map(|m| m.len()).unwrap_or(0),
            replica_path: Some(replica),
            replica_ok: replica_meta.as_ref().map(|m| m.is_file()).unwrap_or(false),
            replica_size_bytes: replica_meta.as_ref().ok().map(|m| m.len()),
            last_synced_at: last_synced,
        })
    }
}

/// Attempt to recover from a corrupted primary by copying the replica over it.
/// Called during startup before the connection pool is created.
///
/// Returns `true` if recovery was performed.
pub(crate) fn try_recover_from_replica(db_path: &str) -> bool {
    let replica = replica_path_for(db_path);
    if !Path::new(&replica).exists() {
        return false;
    }

    // Try opening primary to see if it's valid
    if duckdb::Connection::open(db_path).is_ok() {
        return false; // primary is fine
    }

    // Primary is broken — try to restore from replica
    warn!(
        primary = %db_path,
        replica = %replica,
        "Primary database corrupted, attempting recovery from replica"
    );

    match fs::copy(&replica, db_path) {
        Ok(_) => {
            info!("Successfully recovered primary from replica");
            true
        }
        Err(e) => {
            warn!("Failed to recover from replica: {e}");
            false
        }
    }
}

/// Restore primary from replica (explicit user action).
/// Uses atomic copy to avoid leaving a corrupted primary if interrupted.
pub(crate) fn restore_primary_from_replica(db_path: &str) -> Result<()> {
    if db_path == ":memory:" {
        return Ok(());
    }
    let replica = replica_path_for(db_path);
    if !Path::new(&replica).exists() {
        return Err(MemoryError::Config(
            "No replica file found to restore from".into(),
        ));
    }
    atomic_copy(&replica, db_path)?;
    info!(primary = %db_path, "Primary restored from replica");
    Ok(())
}

/// Promote replica to primary: move primary aside, copy replica in, clean up.
pub(crate) fn promote_replica(db_path: &str) -> Result<()> {
    if db_path == ":memory:" {
        return Ok(());
    }
    let replica = replica_path_for(db_path);
    if !Path::new(&replica).exists() {
        return Err(MemoryError::Config(
            "No replica file found to promote".into(),
        ));
    }

    let old_primary = format!("{db_path}.old");

    if Path::new(db_path).exists() {
        fs::rename(db_path, &old_primary).map_err(|e| {
            MemoryError::Config(format!("Failed to move primary aside: {e}"))
        })?;
    }

    match atomic_copy(&replica, db_path) {
        Ok(_) => {
            let _ = fs::remove_file(&old_primary);
            info!(primary = %db_path, "Replica promoted to primary");
            Ok(())
        }
        Err(e) => {
            if Path::new(&old_primary).exists() {
                let _ = fs::rename(&old_primary, db_path);
            }
            Err(e)
        }
    }
}

/// Copy src → dst atomically: write to .tmp then rename.
fn atomic_copy(src: &str, dst: &str) -> Result<()> {
    let tmp = format!("{dst}.tmp");
    fs::copy(src, &tmp).map_err(|e| {
        MemoryError::Config(format!("Failed to copy {src} → {tmp}: {e}"))
    })?;
    fs::rename(&tmp, dst).map_err(|e| {
        // Clean up tmp on rename failure
        let _ = fs::remove_file(&tmp);
        MemoryError::Config(format!("Failed to rename {tmp} → {dst}: {e}"))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MemoryConfig;

    fn temp_db_path() -> String {
        let id = uuid::Uuid::new_v4();
        format!("/tmp/memme_test_replica_{id}.duckdb")
    }

    fn cleanup(path: &str) {
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(&format!("{path}.replica"));
        let _ = fs::remove_file(&format!("{path}.replica.tmp"));
        let _ = fs::remove_file(&format!("{path}.old"));
        let _ = fs::remove_file(&format!("{path}.wal"));
    }

    #[test]
    fn test_sync_replica_creates_file() {
        let db_path = temp_db_path();
        let config = MemoryConfig::new(&db_path, 384);
        let storage = Storage::open(config).unwrap();

        let result = storage.sync_replica().unwrap().unwrap();
        assert_eq!(result.primary_path, db_path);
        assert!(Path::new(&result.replica_path).exists());
        assert!(result.size_bytes > 0);

        cleanup(&db_path);
    }

    #[test]
    fn test_sync_replica_skips_memory_db() {
        let config = MemoryConfig::new(":memory:", 384);
        let storage = Storage::open(config).unwrap();

        let result = storage.sync_replica().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_replica_status() {
        let db_path = temp_db_path();
        let config = MemoryConfig::new(&db_path, 384);
        let storage = Storage::open(config).unwrap();

        // Before sync
        let status = storage.replica_status().unwrap();
        assert!(status.primary_ok);
        assert!(!status.replica_ok);

        // After sync
        storage.sync_replica().unwrap();
        let status = storage.replica_status().unwrap();
        assert!(status.primary_ok);
        assert!(status.replica_ok);
        assert!(status.last_synced_at.is_some());
        assert_eq!(status.primary_size_bytes, status.replica_size_bytes.unwrap());

        cleanup(&db_path);
    }

    #[test]
    fn test_promote_replica() {
        let db_path = temp_db_path();
        let config = MemoryConfig::new(&db_path, 384);
        let storage = Storage::open(config).unwrap();

        // Write some data and sync
        storage
            .set_config("test_key", "test_value")
            .unwrap();
        storage.sync_replica().unwrap();

        // Promote — should succeed
        drop(storage); // close connections
        promote_replica(&db_path).unwrap();

        // Verify promoted primary is valid
        let config2 = MemoryConfig::new(&db_path, 384);
        let storage2 = Storage::open(config2).unwrap();
        let val = storage2.get_config("test_key").unwrap();
        assert_eq!(val.as_deref(), Some("test_value"));

        cleanup(&db_path);
    }

    #[test]
    fn test_recover_from_replica() {
        let db_path = temp_db_path();
        let config = MemoryConfig::new(&db_path, 384);
        let storage = Storage::open(config).unwrap();

        storage
            .set_config("recover_key", "recover_value")
            .unwrap();
        storage.sync_replica().unwrap();
        drop(storage);

        // Corrupt the primary
        fs::write(&db_path, b"corrupted").unwrap();

        // Recovery should restore from replica
        let recovered = try_recover_from_replica(&db_path);
        assert!(recovered);

        // Verify data is intact
        let config2 = MemoryConfig::new(&db_path, 384);
        let storage2 = Storage::open(config2).unwrap();
        let val = storage2.get_config("recover_key").unwrap();
        assert_eq!(val.as_deref(), Some("recover_value"));

        cleanup(&db_path);
    }

    #[test]
    fn test_restore_no_replica_returns_error() {
        let db_path = "/tmp/memme_no_replica_test.duckdb";
        let _ = fs::remove_file(&format!("{db_path}.replica"));
        let err = restore_primary_from_replica(db_path);
        assert!(err.is_err());
    }
}

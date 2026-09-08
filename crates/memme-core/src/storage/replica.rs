//! Replica management using SQLite's online backup API for data safety.
//!
//! Guarantees that at least one complete copy of the database exists on disk
//! at all times. The replica is a full binary copy of the primary database file.

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

    /// Create a consistent SQLite snapshot and atomically publish it at `dst`.
    fn consistent_snapshot(&self, dst: &str) -> Result<u64> {
        let tmp = format!("{dst}.{}.tmp", uuid::Uuid::new_v4());
        let tmp_path = Path::new(&tmp);
        let result = (|| {
            self.backend.online_backup_to(tmp_path)?;
            super::backend_sqlite::set_private_file_permissions(tmp_path)?;
            validate_db_file(&tmp)?;
            fs::rename(&tmp, dst).map_err(|e| {
                MemoryError::Config(format!("Failed to publish backup {tmp} -> {dst}: {e}"))
            })?;
            super::backend_sqlite::set_private_file_permissions(Path::new(dst))?;
            Ok(fs::metadata(dst).map(|m| m.len()).unwrap_or(0))
        })();
        if result.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        result
    }

    /// Sync primary to replica using a consistent SQLite snapshot.
    pub(crate) fn sync_replica(&self) -> Result<Option<ReplicaSyncResult>> {
        if !self.is_file_backed() {
            return Ok(None);
        }

        let primary = &self.config.db_path;
        let replica = replica_path_for(primary);

        let size_bytes = self.consistent_snapshot(&replica)?;

        let synced_at = chrono::Utc::now().to_rfc3339();
        let _ = self.set_config("replica_last_synced_at", &synced_at);

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

    /// Backup the database to a user-specified path.
    ///
    /// Uses SQLite's online backup API and atomically publishes `backup_path`.
    /// Returns metadata about the backup. For `:memory:` databases, returns an error.
    pub(crate) fn backup_to_path(&self, backup_path: &str) -> Result<crate::types::BackupInfo> {
        if !self.is_file_backed() {
            return Err(MemoryError::Config(
                "Cannot backup an in-memory database".into(),
            ));
        }

        let primary = &self.config.db_path;

        let schema_version = Self::SCHEMA_VERSION.to_string();

        let size_bytes = self.consistent_snapshot(backup_path)?;
        let snapshot = rusqlite::Connection::open(backup_path).map_err(|e| {
            MemoryError::Config(format!("Cannot read completed backup metadata: {e}"))
        })?;
        let memory_count = snapshot
            .query_row("SELECT COUNT(*) FROM memories", [], |row| {
                row.get::<_, u64>(0)
            })
            .map_err(|e| MemoryError::Config(format!("Cannot count backup memories: {e}")))?;
        let created_at = chrono::Utc::now().to_rfc3339();

        info!(
            primary = %primary,
            backup = %backup_path,
            size_bytes,
            memory_count,
            "Database backed up"
        );

        Ok(crate::types::BackupInfo {
            source_path: primary.clone(),
            backup_path: backup_path.to_string(),
            size_bytes,
            created_at,
            memory_count,
            schema_version,
        })
    }
}

/// Validate that a database file is a real SQLite database with tables.
/// SQLite will happily open any file as an empty database, so we also
/// verify that `sqlite_master` contains at least one table.
pub(crate) fn validate_db_file(path: &str) -> Result<()> {
    let conn = rusqlite::Connection::open(path)
        .map_err(|e| MemoryError::Config(format!("Cannot open database: {e}")))?;
    let integrity: String = conn
        .query_row("PRAGMA integrity_check", [], |row| row.get(0))
        .map_err(|e| MemoryError::Config(format!("Cannot verify database integrity: {e}")))?;
    if integrity != "ok" {
        return Err(MemoryError::Config(format!(
            "Backup database failed integrity check: {integrity}"
        )));
    }
    let required_tables: i64 = conn
        .query_row(
            "SELECT count(*) FROM sqlite_master \
             WHERE type='table' AND name IN ('memories', 'sessions', 'events', 'memme_config')",
            [],
            |row| row.get(0),
        )
        .map_err(|e| MemoryError::Config(format!("Not a valid database: {e}")))?;
    if required_tables != 4 {
        return Err(MemoryError::Config(format!(
            "Backup file is not a MemMe database: {path}"
        )));
    }
    Ok(())
}

fn validate_collection_tables(path: &str, collection: &str) -> Result<()> {
    let conn = rusqlite::Connection::open(path)
        .map_err(|e| MemoryError::Config(format!("Cannot open database: {e}")))?;
    let entities = format!("entities_{collection}");
    let relationships = format!("relationships_{collection}");
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN (?1, ?2)",
            rusqlite::params![entities, relationships],
            |row| row.get(0),
        )
        .map_err(|e| MemoryError::Config(format!("Cannot validate collection tables: {e}")))?;
    if count != 2 {
        return Err(MemoryError::Config(format!(
            "Backup does not contain collection '{collection}'"
        )));
    }
    Ok(())
}

/// Check if a database file can be opened and contains tables.
#[allow(dead_code)]
fn is_db_file_valid(path: &str) -> bool {
    let conn = match rusqlite::Connection::open(path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    conn.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type='table'",
        [],
        |row| row.get::<_, i64>(0),
    )
    .map(|count| count > 0)
    .unwrap_or(false)
}

/// Restore the primary database from a backup file.
///
/// The backup is copied to a candidate file and opened with the real MemMe
/// configuration before it can replace the primary. The previous primary is
/// kept until the replacement has also been opened successfully, so a failed
/// restore can be rolled back automatically.
pub(crate) fn restore_from_backup(
    backup_path: &str,
    config: &crate::config::MemoryConfig,
) -> Result<()> {
    let primary_path = &config.db_path;
    if primary_path == ":memory:" {
        return Err(MemoryError::Config(
            "Cannot restore to an in-memory database".into(),
        ));
    }
    if !Path::new(backup_path).exists() {
        return Err(MemoryError::Config(format!(
            "Backup file not found: {backup_path}"
        )));
    }

    validate_db_file(backup_path)?;

    let restore_id = uuid::Uuid::new_v4();
    let candidate = format!("{primary_path}.restore-{restore_id}.tmp");
    let rollback = format!("{primary_path}.restore-{restore_id}.rollback");
    fs::copy(backup_path, &candidate).map_err(|e| {
        MemoryError::Config(format!(
            "Failed to stage backup {backup_path} -> {candidate}: {e}"
        ))
    })?;
    super::backend_sqlite::set_private_file_permissions(Path::new(&candidate))?;

    let candidate_result = (|| {
        validate_db_file(&candidate)?;
        validate_collection_tables(&candidate, &config.collection_name)?;
        let mut candidate_config = config.clone();
        candidate_config.db_path = candidate.clone();
        let candidate_storage = Storage::open(candidate_config)?;
        candidate_storage
            .backend
            .execute_batch(candidate_storage.dialect().checkpoint_sql())?;
        drop(candidate_storage);
        Ok(())
    })();
    if let Err(error) = candidate_result {
        remove_database_files(&candidate);
        return Err(error);
    }

    let had_primary = Path::new(primary_path).exists();
    if had_primary {
        move_database_files(primary_path, &rollback)?;
    }
    if let Err(error) = fs::rename(&candidate, primary_path) {
        if had_primary {
            restore_and_verify_rollback(&rollback, primary_path, config, &error.to_string())?;
        } else {
            return Err(MemoryError::RollbackFailed {
                rollback_path: candidate,
                message: format!(
                    "failed to install restored database and no previous primary exists: {error}"
                ),
            });
        }
        remove_database_files(&candidate);
        return Err(MemoryError::Config(format!(
            "Failed to install restored database: {error}"
        )));
    }

    match Storage::open(config.clone()) {
        Ok(storage) => {
            drop(storage);
            if had_primary {
                remove_database_files(&rollback);
            }
        }
        Err(error) => {
            remove_database_files(primary_path);
            if had_primary {
                restore_and_verify_rollback(&rollback, primary_path, config, &error.to_string())?;
            } else {
                return Err(MemoryError::RollbackFailed {
                    rollback_path: backup_path.to_string(),
                    message: format!(
                        "restored database failed validation and no previous primary exists: {error}"
                    ),
                });
            }
            return Err(MemoryError::Config(format!(
                "Restore validation failed; previous database was restored: {error}"
            )));
        }
    }

    info!(
        backup = %backup_path,
        primary = %primary_path,
        "Database restored from backup"
    );
    Ok(())
}

fn restore_and_verify_rollback(
    rollback: &str,
    primary: &str,
    config: &crate::config::MemoryConfig,
    restore_error: &str,
) -> Result<()> {
    move_database_files(rollback, primary).map_err(|rollback_error| {
        MemoryError::RollbackFailed {
            rollback_path: rollback.to_string(),
            message: format!(
                "restore failed: {restore_error}; could not reinstall previous primary: {rollback_error}"
            ),
        }
    })?;

    match Storage::open(config.clone()) {
        Ok(storage) => {
            drop(storage);
            Ok(())
        }
        Err(verification_error) => {
            // Keep the failed rollback as a separate file for manual repair.
            // If moving it aside also fails, report both possible locations.
            let preserve_error = move_database_files(primary, rollback).err();
            Err(MemoryError::RollbackFailed {
                rollback_path: rollback.to_string(),
                message: format!(
                    "restore failed: {restore_error}; previous primary could not be reopened: {verification_error}; preserve error: {}",
                    preserve_error
                        .map(|error| error.to_string())
                        .unwrap_or_else(|| "none".to_string())
                ),
            })
        }
    }
}

fn sidecar_paths(path: &str) -> [String; 2] {
    [format!("{path}-wal"), format!("{path}-shm")]
}

fn move_database_files(source: &str, destination: &str) -> Result<()> {
    fs::rename(source, destination).map_err(|e| {
        MemoryError::Config(format!("Failed to move {source} -> {destination}: {e}"))
    })?;
    let mut moved_sidecars = Vec::new();
    for (source_sidecar, destination_sidecar) in sidecar_paths(source)
        .into_iter()
        .zip(sidecar_paths(destination))
    {
        if Path::new(&source_sidecar).exists() {
            if let Err(error) = fs::rename(&source_sidecar, &destination_sidecar) {
                for (moved_source, moved_destination) in moved_sidecars.into_iter().rev() {
                    let _ = fs::rename(moved_destination, moved_source);
                }
                let _ = fs::rename(destination, source);
                return Err(MemoryError::Config(format!(
                    "Failed to move SQLite sidecar {source_sidecar}: {error}"
                )));
            }
            moved_sidecars.push((source_sidecar, destination_sidecar));
        }
    }
    Ok(())
}

fn remove_database_files(path: &str) {
    let _ = fs::remove_file(path);
    for sidecar in sidecar_paths(path) {
        let _ = fs::remove_file(sidecar);
    }
}

/// Attempt to recover from a corrupted primary by copying the replica over it.
/// Called during startup before the connection is created.
///
/// Returns `true` if recovery was performed.
#[allow(dead_code)]
pub(crate) fn try_recover_from_replica(db_path: &str) -> bool {
    let replica = replica_path_for(db_path);
    if !Path::new(&replica).exists() {
        return false;
    }

    // Try opening primary to see if it's valid
    if is_db_file_valid(db_path) {
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
        fs::rename(db_path, &old_primary)
            .map_err(|e| MemoryError::Config(format!("Failed to move primary aside: {e}")))?;
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
    fs::copy(src, &tmp)
        .map_err(|e| MemoryError::Config(format!("Failed to copy {src} → {tmp}: {e}")))?;
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
        format!("/tmp/memme_test_replica_{id}.db")
    }

    fn cleanup(path: &str) {
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(format!("{path}.replica"));
        let _ = fs::remove_file(format!("{path}.replica.tmp"));
        let _ = fs::remove_file(format!("{path}.old"));
        let _ = fs::remove_file(format!("{path}.wal"));
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
        assert!(status.primary_size_bytes > 0);
        assert!(status.replica_size_bytes.unwrap() > 0);

        cleanup(&db_path);
    }

    #[test]
    fn test_promote_replica() {
        let db_path = temp_db_path();
        let config = MemoryConfig::new(&db_path, 384);
        let storage = Storage::open(config).unwrap();

        // Write some data and sync
        storage.set_config("test_key", "test_value").unwrap();
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

        storage.set_config("recover_key", "recover_value").unwrap();
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
        let db_path = "/tmp/memme_no_replica_test.db";
        let _ = fs::remove_file(format!("{db_path}.replica"));
        let err = restore_primary_from_replica(db_path);
        assert!(err.is_err());
    }

    // ── Backup / Restore tests ──

    fn cleanup_backup(path: &str) {
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(format!("{path}.tmp"));
    }

    fn temp_backup_path() -> String {
        let id = uuid::Uuid::new_v4();
        format!("/tmp/memme_test_backup_{id}.db")
    }

    #[test]
    fn test_backup_to_path() {
        let db_path = temp_db_path();
        let backup_path = temp_backup_path();
        let config = MemoryConfig::new(&db_path, 384);
        let storage = Storage::open(config).unwrap();

        let info = storage.backup_to_path(&backup_path).unwrap();
        assert_eq!(info.source_path, db_path);
        assert_eq!(info.backup_path, backup_path);
        assert!(info.size_bytes > 0);
        assert_eq!(info.memory_count, 0);
        assert!(!info.schema_version.is_empty());
        assert!(Path::new(&backup_path).exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&db_path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(&backup_path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }

        cleanup(&db_path);
        cleanup_backup(&backup_path);
    }

    #[test]
    fn test_backup_metadata_with_data() {
        let db_path = temp_db_path();
        let backup_path = temp_backup_path();
        let config = MemoryConfig::new(&db_path, 384);
        let storage = Storage::open(config).unwrap();

        // Insert some data
        let emb: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
        storage
            .insert_memory(
                "id1",
                "hello world",
                &emb,
                "user1",
                "h1",
                &crate::storage::InsertMemoryParams::default(),
            )
            .unwrap();

        let info = storage.backup_to_path(&backup_path).unwrap();
        assert_eq!(info.memory_count, 1);

        cleanup(&db_path);
        cleanup_backup(&backup_path);
    }

    #[test]
    fn test_backup_memory_db_returns_error() {
        let config = MemoryConfig::new(":memory:", 384);
        let storage = Storage::open(config).unwrap();

        let result = storage.backup_to_path("/tmp/should_not_exist.db");
        assert!(result.is_err());
    }

    #[test]
    fn test_restore_from_backup() {
        let db_path = temp_db_path();
        let backup_path = temp_backup_path();
        let config = MemoryConfig::new(&db_path, 384);
        let storage = Storage::open(config).unwrap();

        // Write data and backup
        storage.set_config("backup_key", "backup_value").unwrap();
        storage.backup_to_path(&backup_path).unwrap();

        // Write more data after backup
        storage
            .set_config("after_backup", "should_disappear")
            .unwrap();
        drop(storage);

        // Restore from backup
        restore_from_backup(&backup_path, &MemoryConfig::new(&db_path, 384)).unwrap();

        // Verify restored state matches backup (not the later write)
        let config2 = MemoryConfig::new(&db_path, 384);
        let storage2 = Storage::open(config2).unwrap();
        let val = storage2.get_config("backup_key").unwrap();
        assert_eq!(val.as_deref(), Some("backup_value"));
        let gone = storage2.get_config("after_backup").unwrap();
        assert!(gone.is_none());

        cleanup(&db_path);
        cleanup_backup(&backup_path);
    }

    #[test]
    fn test_restore_rejects_incompatible_dimensions_without_replacing_primary() {
        let primary_path = temp_db_path();
        let backup_source_path = temp_db_path();
        let backup_path = temp_backup_path();

        let backup_source = Storage::open(MemoryConfig::new(&backup_source_path, 32)).unwrap();
        backup_source
            .set_config("backup_marker", "wrong dimensions")
            .unwrap();
        backup_source.backup_to_path(&backup_path).unwrap();
        drop(backup_source);

        let primary_config = MemoryConfig::new(&primary_path, 64);
        let primary = Storage::open(primary_config.clone()).unwrap();
        primary
            .set_config("primary_marker", "must survive")
            .unwrap();
        drop(primary);

        let error = restore_from_backup(&backup_path, &primary_config).unwrap_err();
        assert!(error.to_string().contains("dimension mismatch"));

        let reopened = Storage::open(primary_config).unwrap();
        assert_eq!(
            reopened.get_config("primary_marker").unwrap().as_deref(),
            Some("must survive")
        );

        cleanup(&primary_path);
        cleanup(&backup_source_path);
        cleanup_backup(&backup_path);
    }

    #[test]
    fn test_restore_rejects_missing_collection_without_replacing_primary() {
        let primary_path = temp_db_path();
        let backup_source_path = temp_db_path();
        let backup_path = temp_backup_path();

        let mut backup_config = MemoryConfig::new(&backup_source_path, 32);
        backup_config.collection_name = "other_collection".into();
        let backup_source = Storage::open(backup_config).unwrap();
        backup_source.backup_to_path(&backup_path).unwrap();
        drop(backup_source);

        let primary_config = MemoryConfig::new(&primary_path, 32);
        let primary = Storage::open(primary_config.clone()).unwrap();
        primary
            .set_config("primary_marker", "must survive")
            .unwrap();
        drop(primary);

        let error = restore_from_backup(&backup_path, &primary_config).unwrap_err();
        assert!(error.to_string().contains("collection 'default'"));
        let reopened = Storage::open(primary_config).unwrap();
        assert_eq!(
            reopened.get_config("primary_marker").unwrap().as_deref(),
            Some("must survive")
        );

        cleanup(&primary_path);
        cleanup(&backup_source_path);
        cleanup_backup(&backup_path);
    }

    #[test]
    fn test_restore_nonexistent_backup() {
        let id = uuid::Uuid::new_v4();
        let path = format!("/tmp/memme_nonexistent_{id}.db");
        let result = restore_from_backup(&path, &MemoryConfig::new("/tmp/target.db", 384));
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_rollback_is_reported_as_terminal_failure() {
        let id = uuid::Uuid::new_v4();
        let primary = format!("/tmp/memme_missing_primary_{id}.db");
        let rollback = format!("/tmp/memme_missing_rollback_{id}.db");
        let config = MemoryConfig::new(&primary, 384);

        let error = restore_and_verify_rollback(&rollback, &primary, &config, "install failed")
            .unwrap_err();
        assert!(matches!(error, MemoryError::RollbackFailed { .. }));
        assert!(!Path::new(&primary).exists());
    }

    #[test]
    fn test_restore_invalid_backup() {
        let invalid_path = temp_backup_path();
        fs::write(&invalid_path, b"not a database file").unwrap();

        let result = restore_from_backup(&invalid_path, &MemoryConfig::new("/tmp/target.db", 384));
        assert!(result.is_err());

        cleanup_backup(&invalid_path);
    }

    #[test]
    fn test_restore_rejects_non_memme_sqlite_database() {
        let backup_path = temp_backup_path();
        let conn = rusqlite::Connection::open(&backup_path).unwrap();
        conn.execute("CREATE TABLE unrelated (id INTEGER)", [])
            .unwrap();
        drop(conn);

        let result = restore_from_backup(
            &backup_path,
            &MemoryConfig::new("/tmp/memme-restore-target.db", 384),
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not a MemMe database"));

        cleanup_backup(&backup_path);
    }

    #[test]
    fn test_restore_to_memory_db_returns_error() {
        let result =
            restore_from_backup("/tmp/some_backup.db", &MemoryConfig::new(":memory:", 384));
        assert!(result.is_err());
    }
}

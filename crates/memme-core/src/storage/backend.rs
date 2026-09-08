//! Database backend abstraction for connection management and row access.
//!
//! This module provides the [`Backend`] struct and [`RowAccess`] trait.
//! SQLite storage backend.

use crate::error::{MemoryError, Result};
use crate::types::SqlParam;

use super::dialect::SqlDialect;

// ── RowAccess trait ──

/// Database-agnostic row access.
///
/// Wraps database-specific row types so that row mapper functions
/// don't need to know the underlying database engine.
#[allow(dead_code)] // methods called via &dyn RowAccess in mapper closures
pub(crate) trait RowAccess {
    fn get_string(&self, idx: usize) -> Result<String>;
    fn get_opt_string(&self, idx: usize) -> Result<Option<String>>;
    fn get_i64(&self, idx: usize) -> Result<i64>;
    fn get_opt_i64(&self, idx: usize) -> Result<Option<i64>>;
    fn get_f64(&self, idx: usize) -> Result<f64>;
    fn get_opt_f64(&self, idx: usize) -> Result<Option<f64>>;
    fn get_bool(&self, idx: usize) -> Result<bool>;
    fn get_opt_bool(&self, idx: usize) -> Result<Option<bool>>;
}

// ── Backend struct ──

/// Database backend providing connection management and query execution.
pub(crate) struct Backend {
    conn: std::sync::Mutex<rusqlite::Connection>,
    dialect: super::dialect_sqlite::SqliteDialect,
}

/// Operations bound to one SQLite transaction and one connection lock.
pub(crate) struct BackendTransaction<'a> {
    conn: &'a rusqlite::Connection,
}

impl BackendTransaction<'_> {
    pub fn execute(&self, sql: &str, params: &[SqlParam]) -> Result<usize> {
        super::backend_sqlite::sqlite_execute_on(self.conn, sql, params)
    }

    pub fn execute_batch(&self, sql: &str) -> Result<()> {
        super::backend_sqlite::sqlite_execute_batch_on(self.conn, sql)
    }

    pub fn query_read<T>(
        &self,
        sql: &str,
        params: &[SqlParam],
        mut mapper: impl FnMut(&dyn RowAccess) -> Result<T>,
    ) -> Result<Vec<T>> {
        super::backend_sqlite::sqlite_query_collect_on(self.conn, sql, params, &mut mapper)
    }

    pub fn query_one<T>(
        &self,
        sql: &str,
        params: &[SqlParam],
        mapper: impl FnMut(&dyn RowAccess) -> Result<T>,
    ) -> Result<Option<T>> {
        let results = self.query_read(sql, params, mapper)?;
        Ok(results.into_iter().next())
    }

    pub fn query_count(&self, sql: &str, params: &[SqlParam]) -> Result<i64> {
        let results = self.query_read(sql, params, |row| row.get_i64(0))?;
        Ok(results.into_iter().next().unwrap_or(0))
    }

    pub fn table_exists(&self, table_name: &str) -> Result<bool> {
        Ok(self.query_count(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = $1",
            &[SqlParam::Text(table_name.to_string())],
        )? > 0)
    }
}

impl Backend {
    /// Create a SQLite backend.
    pub fn sqlite(conn: rusqlite::Connection) -> Self {
        Self {
            conn: std::sync::Mutex::new(conn),
            dialect: super::dialect_sqlite::SqliteDialect,
        }
    }

    /// Create a transactionally consistent SQLite snapshot while the database
    /// remains available to readers and writers.
    pub fn online_backup_to(&self, path: &std::path::Path) -> Result<()> {
        let source = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let mut destination = rusqlite::Connection::open(path)
            .map_err(|e| MemoryError::Storage(format!("cannot create backup: {e}")))?;
        let backup = rusqlite::backup::Backup::new(&source, &mut destination)
            .map_err(|e| MemoryError::Storage(format!("cannot start SQLite backup: {e}")))?;
        backup
            .run_to_completion(128, std::time::Duration::from_millis(5), None)
            .map_err(|e| MemoryError::Storage(format!("SQLite backup failed: {e}")))?;
        Ok(())
    }

    /// Access the SQL dialect for this backend.
    pub fn dialect(&self) -> &dyn SqlDialect {
        &self.dialect
    }

    /// Run a group of reads and writes atomically while holding the connection lock.
    pub fn transaction<T>(
        &self,
        operation: impl FnOnce(&BackendTransaction<'_>) -> Result<T>,
    ) -> Result<T> {
        let mut conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        let result = {
            let executor = BackendTransaction { conn: &tx };
            operation(&executor)
        };

        match result {
            Ok(value) => {
                tx.commit()
                    .map_err(|e| MemoryError::Storage(e.to_string()))?;
                Ok(value)
            }
            Err(operation_error) => match tx.rollback() {
                Ok(()) => Err(operation_error),
                Err(rollback_error) => Err(MemoryError::Storage(format!(
                    "{operation_error}; transaction rollback failed: {rollback_error}"
                ))),
            },
        }
    }

    /// Run a set of reads against one SQLite snapshot. A deferred transaction
    /// fixes the snapshot at the first read without taking a write lock.
    pub fn read_transaction<T>(
        &self,
        operation: impl FnOnce(&BackendTransaction<'_>) -> Result<T>,
    ) -> Result<T> {
        let mut conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Deferred)
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        let result = {
            let executor = BackendTransaction { conn: &tx };
            operation(&executor)
        };
        match result {
            Ok(value) => {
                tx.commit()
                    .map_err(|e| MemoryError::Storage(e.to_string()))?;
                Ok(value)
            }
            Err(operation_error) => match tx.rollback() {
                Ok(()) => Err(operation_error),
                Err(rollback_error) => Err(MemoryError::Storage(format!(
                    "{operation_error}; read transaction rollback failed: {rollback_error}"
                ))),
            },
        }
    }

    // ── Query execution ──

    /// Execute a write SQL statement with parameters. Returns rows affected.
    pub fn execute(&self, sql: &str, params: &[SqlParam]) -> Result<usize> {
        super::backend_sqlite::sqlite_execute(&self.conn, sql, params)
    }

    /// Execute a batch of SQL statements (no parameters, no return value).
    pub fn execute_batch(&self, sql: &str) -> Result<()> {
        super::backend_sqlite::sqlite_execute_batch(&self.conn, sql)
    }

    /// Execute SQL ignoring errors (for optional features like extensions).
    pub fn execute_batch_ignore(&self, sql: &str) {
        let _ = super::backend_sqlite::sqlite_execute_batch(&self.conn, sql);
    }

    /// Execute a read query and collect results via a mapper function.
    pub fn query_read<T>(
        &self,
        sql: &str,
        params: &[SqlParam],
        mut mapper: impl FnMut(&dyn RowAccess) -> Result<T>,
    ) -> Result<Vec<T>> {
        super::backend_sqlite::sqlite_query_collect(&self.conn, sql, params, &mut mapper)
    }

    /// Execute a read query and return the first result (or None).
    pub fn query_one<T>(
        &self,
        sql: &str,
        params: &[SqlParam],
        mapper: impl FnMut(&dyn RowAccess) -> Result<T>,
    ) -> Result<Option<T>> {
        let results = self.query_read(sql, params, mapper)?;
        Ok(results.into_iter().next())
    }

    /// Execute a read-only statement that returns a scalar count.
    pub fn query_count(&self, sql: &str, params: &[SqlParam]) -> Result<i64> {
        let results = self.query_read(sql, params, |row| row.get_i64(0))?;
        Ok(results.into_iter().next().unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_transaction_keeps_one_snapshot_across_queries() {
        let path =
            std::env::temp_dir().join(format!("memme-read-snapshot-{}.db", uuid::Uuid::new_v4()));
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; CREATE TABLE snapshot_test (id INTEGER PRIMARY KEY); INSERT INTO snapshot_test DEFAULT VALUES;",
        )
        .unwrap();
        let backend = Backend::sqlite(conn);

        backend
            .read_transaction(|tx| {
                assert_eq!(
                    tx.query_count("SELECT COUNT(*) FROM snapshot_test", &[])?,
                    1
                );
                let writer = rusqlite::Connection::open(&path)
                    .map_err(|error| MemoryError::Storage(error.to_string()))?;
                writer
                    .execute("INSERT INTO snapshot_test DEFAULT VALUES", [])
                    .map_err(|error| MemoryError::Storage(error.to_string()))?;
                assert_eq!(
                    tx.query_count("SELECT COUNT(*) FROM snapshot_test", &[])?,
                    1
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(
            backend
                .query_count("SELECT COUNT(*) FROM snapshot_test", &[])
                .unwrap(),
            2
        );

        drop(backend);
        for candidate in [
            path.clone(),
            path.with_extension("db-wal"),
            path.with_extension("db-shm"),
        ] {
            let _ = std::fs::remove_file(candidate);
        }
    }
}

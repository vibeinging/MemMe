//! Database backend abstraction for connection management and row access.
//!
//! This module provides the [`Backend`] struct and [`RowAccess`] trait.
//! SQLite storage backend.

use crate::error::Result;
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

impl Backend {
    /// Create a SQLite backend.
    pub fn sqlite(conn: rusqlite::Connection) -> Self {
        Self {
            conn: std::sync::Mutex::new(conn),
            dialect: super::dialect_sqlite::SqliteDialect,
        }
    }

    /// Access the SQL dialect for this backend.
    pub fn dialect(&self) -> &dyn SqlDialect {
        &self.dialect
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

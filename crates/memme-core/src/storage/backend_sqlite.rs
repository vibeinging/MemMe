//! SQLite backend implementation for the `Backend` enum.
//!
//! Embeddings are stored as BLOBs (raw little-endian f32 bytes via `bytemuck`).
//! Vector search uses `sqlite-vec` extension's `vec_distance_cosine()` function.

use std::sync::Mutex;

use rusqlite::Connection;

use crate::config::MemoryConfig;
use crate::error::{MemoryError, Result};
use crate::types::SqlParam;

use super::backend::RowAccess;

/// Open a SQLite connection, register sqlite-vec extension, and configure WAL mode.
pub(crate) fn open_sqlite(config: &MemoryConfig) -> Result<Connection> {
    // Register sqlite-vec as an auto-extension BEFORE opening connections.
    // This makes vec_distance_cosine() and vec0 virtual tables available.
    unsafe {
        #[allow(clippy::missing_transmute_annotations)]
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    }

    let conn = if config.db_path == ":memory:" {
        Connection::open_in_memory()
    } else {
        Connection::open(&config.db_path)
    }
    .map_err(|e| MemoryError::Storage(e.to_string()))?;

    // Enable WAL mode for better concurrent read performance
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
        .map_err(|e| MemoryError::Storage(e.to_string()))?;

    // Register custom pow() function (SQLite doesn't have built-in math functions
    // unless compiled with SQLITE_ENABLE_MATH_FUNCTIONS).
    conn.create_scalar_function(
        "pow",
        2,
        rusqlite::functions::FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let base: f64 = ctx.get(0)?;
            let exp: f64 = ctx.get(1)?;
            Ok(base.powf(exp))
        },
    )
    .map_err(|e| MemoryError::Storage(format!("Failed to register pow(): {e}")))?;

    // Verify sqlite-vec is loaded
    let vec_version: String = conn
        .query_row("SELECT vec_version()", [], |row| row.get(0))
        .map_err(|e| {
            MemoryError::Storage(format!(
                "sqlite-vec extension failed to load: {e}. \
                 Ensure the 'sqlite' feature is enabled."
            ))
        })?;
    tracing::info!("sqlite-vec {vec_version} loaded");

    Ok(conn)
}

// ── RowAccess for rusqlite ──

pub(crate) struct SqliteRow<'a>(pub &'a rusqlite::Row<'a>);

impl RowAccess for SqliteRow<'_> {
    fn get_string(&self, idx: usize) -> Result<String> {
        self.0
            .get::<_, String>(idx)
            .map_err(|e| MemoryError::Storage(e.to_string()))
    }

    fn get_opt_string(&self, idx: usize) -> Result<Option<String>> {
        self.0
            .get::<_, Option<String>>(idx)
            .map_err(|e| MemoryError::Storage(e.to_string()))
    }

    fn get_i64(&self, idx: usize) -> Result<i64> {
        self.0
            .get::<_, i64>(idx)
            .map_err(|e| MemoryError::Storage(e.to_string()))
    }

    fn get_opt_i64(&self, idx: usize) -> Result<Option<i64>> {
        self.0
            .get::<_, Option<i64>>(idx)
            .map_err(|e| MemoryError::Storage(e.to_string()))
    }

    fn get_f64(&self, idx: usize) -> Result<f64> {
        self.0
            .get::<_, f64>(idx)
            .map_err(|e| MemoryError::Storage(e.to_string()))
    }

    fn get_opt_f64(&self, idx: usize) -> Result<Option<f64>> {
        self.0
            .get::<_, Option<f64>>(idx)
            .map_err(|e| MemoryError::Storage(e.to_string()))
    }

    fn get_bool(&self, idx: usize) -> Result<bool> {
        self.0
            .get::<_, bool>(idx)
            .map_err(|e| MemoryError::Storage(e.to_string()))
    }

    fn get_opt_bool(&self, idx: usize) -> Result<Option<bool>> {
        self.0
            .get::<_, Option<bool>>(idx)
            .map_err(|e| MemoryError::Storage(e.to_string()))
    }
}

// ── SqlParam to rusqlite conversion ──

impl rusqlite::types::ToSql for SqlParam {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        match self {
            SqlParam::Null => Ok(rusqlite::types::ToSqlOutput::Owned(
                rusqlite::types::Value::Null,
            )),
            SqlParam::Bool(b) => b.to_sql(),
            SqlParam::Int(i) => i.to_sql(),
            SqlParam::Float(f) => f.to_sql(),
            SqlParam::Text(s) => Ok(rusqlite::types::ToSqlOutput::Owned(
                rusqlite::types::Value::Text(s.clone()),
            )),
        }
    }
}

// ── Query helpers for Backend::Sqlite ──

/// Execute a write statement on a SQLite connection.
pub(crate) fn sqlite_execute(
    conn: &Mutex<Connection>,
    sql: &str,
    params: &[SqlParam],
) -> Result<usize> {
    let conn = conn.lock().unwrap_or_else(|e| e.into_inner());
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params
        .iter()
        .map(|p| p as &dyn rusqlite::types::ToSql)
        .collect();
    conn.execute(sql, param_refs.as_slice())
        .map_err(|e| MemoryError::Storage(e.to_string()))
}

/// Execute a batch of SQL statements on a SQLite connection.
pub(crate) fn sqlite_execute_batch(conn: &Mutex<Connection>, sql: &str) -> Result<()> {
    let conn = conn.lock().unwrap_or_else(|e| e.into_inner());
    conn.execute_batch(sql)
        .map_err(|e| MemoryError::Storage(e.to_string()))
}

/// Query rows from a SQLite connection with a mapper function.
pub(crate) fn sqlite_query_collect<T>(
    conn: &Mutex<Connection>,
    sql: &str,
    params: &[SqlParam],
    mapper: &mut impl FnMut(&dyn RowAccess) -> Result<T>,
) -> Result<Vec<T>> {
    let conn = conn.lock().unwrap_or_else(|e| e.into_inner());
    let mut stmt = conn
        .prepare(sql)
        .map_err(|e| MemoryError::Storage(e.to_string()))?;
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params
        .iter()
        .map(|p| p as &dyn rusqlite::types::ToSql)
        .collect();
    let mut rows = stmt
        .query(param_refs.as_slice())
        .map_err(|e| MemoryError::Storage(e.to_string()))?;
    let mut results = Vec::new();
    while let Some(row) = rows
        .next()
        .map_err(|e| MemoryError::Storage(e.to_string()))?
    {
        let wrapped = SqliteRow(row);
        results.push(mapper(&wrapped)?);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqlite_vec_loads() {
        // Verify sqlite-vec extension is available
        unsafe {
            #[allow(clippy::missing_transmute_annotations)]
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
        let conn = Connection::open_in_memory().unwrap();
        let version: String = conn
            .query_row("SELECT vec_version()", [], |row| row.get(0))
            .unwrap();
        assert!(!version.is_empty());
    }

    #[test]
    fn test_vec_distance_cosine() {
        unsafe {
            #[allow(clippy::missing_transmute_annotations)]
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
        let conn = Connection::open_in_memory().unwrap();

        // Identical vectors → distance 0
        let a: Vec<f32> = vec![1.0, 0.0, 0.0];
        let b: Vec<f32> = vec![1.0, 0.0, 0.0];
        let a_blob: &[u8] = bytemuck::cast_slice(&a);
        let b_blob: &[u8] = bytemuck::cast_slice(&b);
        let dist: f64 = conn
            .query_row(
                "SELECT vec_distance_cosine(?1, ?2)",
                rusqlite::params![a_blob, b_blob],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            dist.abs() < 1e-6,
            "identical vectors should have distance ~0, got {dist}"
        );

        // Orthogonal vectors → distance 1
        let a: Vec<f32> = vec![1.0, 0.0];
        let b: Vec<f32> = vec![0.0, 1.0];
        let a_blob: &[u8] = bytemuck::cast_slice(&a);
        let b_blob: &[u8] = bytemuck::cast_slice(&b);
        let dist: f64 = conn
            .query_row(
                "SELECT vec_distance_cosine(?1, ?2)",
                rusqlite::params![a_blob, b_blob],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            (dist - 1.0).abs() < 1e-6,
            "orthogonal vectors should have distance ~1, got {dist}"
        );
    }

    #[test]
    fn test_vec0_knn_search() {
        use crate::config::MemoryConfig;
        use crate::storage::Storage;

        let config = MemoryConfig {
            db_path: ":memory:".into(),
            collection_name: "test".into(),
            embedding_dims: 3,
            ..Default::default()
        };
        let storage = Storage::open(config).unwrap();

        // Insert some memories with embeddings
        let params = crate::storage::InsertMemoryParams::default();
        storage
            .insert_memory(
                "m1",
                "coffee lover",
                &[1.0, 0.0, 0.0],
                "alice",
                "h1",
                &params,
            )
            .unwrap();
        storage
            .insert_memory(
                "m2",
                "tea drinker",
                &[0.9, 0.1, 0.0],
                "alice",
                "h2",
                &params,
            )
            .unwrap();
        storage
            .insert_memory("m3", "juice fan", &[0.0, 1.0, 0.0], "alice", "h3", &params)
            .unwrap();
        storage
            .insert_memory("m4", "bob's memory", &[1.0, 0.0, 0.0], "bob", "h4", &params)
            .unwrap();

        // Search for vectors close to [1, 0, 0] — should find m1 first, m2 second, m3 last
        let results = storage
            .vector_search(&[1.0, 0.0, 0.0], "alice", None, None, None, None, 3)
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "m1");
        assert_eq!(results[1].id, "m2");
        assert_eq!(results[2].id, "m3");

        // Score should be cosine distance (0 for identical)
        assert!(
            results[0].score.unwrap() < 0.01,
            "identical vector should have ~0 distance"
        );
        assert!(
            results[2].score.unwrap() > 0.5,
            "orthogonal vector should have high distance"
        );

        // Bob's memory should NOT appear (user isolation via partition key)
        assert!(results.iter().all(|r| r.user_id == "alice"));
    }

    #[test]
    fn test_vec0_delete_sync() {
        use crate::config::MemoryConfig;
        use crate::storage::Storage;

        let config = MemoryConfig {
            db_path: ":memory:".into(),
            collection_name: "test".into(),
            embedding_dims: 3,
            ..Default::default()
        };
        let storage = Storage::open(config).unwrap();
        let params = crate::storage::InsertMemoryParams::default();
        storage
            .insert_memory("m1", "test", &[1.0, 0.0, 0.0], "alice", "h1", &params)
            .unwrap();

        // Verify it's searchable
        let results = storage
            .vector_search(&[1.0, 0.0, 0.0], "alice", None, None, None, None, 5)
            .unwrap();
        assert_eq!(results.len(), 1);

        // Delete it
        storage.delete_memory("m1").unwrap();

        // Should no longer be found
        let results = storage
            .vector_search(&[1.0, 0.0, 0.0], "alice", None, None, None, None, 5)
            .unwrap();
        assert_eq!(results.len(), 0);
    }
}

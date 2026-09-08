//! SQLite backend implementation for the `Backend` enum.
//!
//! Embeddings are stored as BLOBs (raw little-endian f32 bytes via `bytemuck`).
//! Vector search uses the VexDB-Lite SQLite extension.

use std::sync::Mutex;

use rusqlite::Connection;

use crate::config::MemoryConfig;
use crate::error::{MemoryError, Result};
use crate::types::SqlParam;

use super::backend::RowAccess;

#[cfg(unix)]
pub(crate) fn set_private_file_permissions(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    if path.exists() {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).map_err(
            |error| {
                MemoryError::Storage(format!(
                    "cannot set private permissions on {}: {error}",
                    path.display()
                ))
            },
        )?;
    }
    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn set_private_file_permissions(_path: &std::path::Path) -> Result<()> {
    Ok(())
}

fn ensure_database_parent(path: &std::path::Path) -> Result<()> {
    let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    if parent.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(parent).map_err(|error| {
        MemoryError::Storage(format!(
            "cannot create database directory {}: {error}",
            parent.display()
        ))
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700)).map_err(
            |error| {
                MemoryError::Storage(format!(
                    "cannot set private database directory permissions on {}: {error}",
                    parent.display()
                ))
            },
        )?;
    }
    Ok(())
}

/// Open SQLite, load the selected vector extension, and configure WAL mode.
pub(crate) fn open_sqlite(
    config: &MemoryConfig,
    extension_path: &std::path::Path,
) -> Result<Connection> {
    if config.db_path != ":memory:" {
        ensure_database_parent(std::path::Path::new(&config.db_path))?;
    }
    let conn = if config.db_path == ":memory:" {
        Connection::open_in_memory()
    } else {
        Connection::open(&config.db_path)
    }
    .map_err(|e| MemoryError::Storage(e.to_string()))?;

    load_vexdb_lite(&conn, extension_path)?;

    // Enable WAL mode for better concurrent read performance
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
        .map_err(|e| MemoryError::Storage(e.to_string()))?;

    if config.db_path != ":memory:" {
        set_private_file_permissions(std::path::Path::new(&config.db_path))?;
        for suffix in ["-wal", "-shm"] {
            let sidecar = format!("{}{suffix}", config.db_path);
            set_private_file_permissions(std::path::Path::new(&sidecar))?;
        }
    }

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

    let vexdb_version: String = conn
        .query_row("SELECT vexdb_version()", [], |row| row.get(0))
        .map_err(|e| {
            MemoryError::Storage(format!(
                "VexDB-Lite extension loaded without vexdb_version(): {e}"
            ))
        })?;
    tracing::info!("VexDB-Lite {vexdb_version} loaded");

    Ok(conn)
}

fn load_vexdb_lite(conn: &Connection, extension_path: &std::path::Path) -> Result<()> {
    if !extension_path.is_file() {
        return Err(MemoryError::Config(format!(
            "VexDB-Lite SQLite extension does not exist: {}",
            extension_path.display()
        )));
    }

    // The caller opts in with an explicit path. No SQL is executed while
    // extension loading is enabled, and the guard disables it on every exit.
    let guard = unsafe { rusqlite::LoadExtensionGuard::new(conn) }.map_err(|e| {
        MemoryError::Storage(format!("Cannot enable SQLite extension loading: {e}"))
    })?;
    let result = unsafe { conn.load_extension(extension_path, Some("sqlite3_vexdblite_init")) }
        .map_err(|e| {
            MemoryError::Storage(format!(
                "Failed to load VexDB-Lite SQLite extension {}: {e}",
                extension_path.display()
            ))
        });
    drop(guard);
    result
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
    sqlite_execute_on(&conn, sql, params)
}

/// Execute a write statement on an already-locked SQLite connection.
pub(crate) fn sqlite_execute_on(
    conn: &Connection,
    sql: &str,
    params: &[SqlParam],
) -> Result<usize> {
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
    sqlite_execute_batch_on(&conn, sql)
}

/// Execute a batch on an already-locked SQLite connection.
pub(crate) fn sqlite_execute_batch_on(conn: &Connection, sql: &str) -> Result<()> {
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
    sqlite_query_collect_on(&conn, sql, params, mapper)
}

/// Query rows from an already-locked SQLite connection.
pub(crate) fn sqlite_query_collect_on<T>(
    conn: &Connection,
    sql: &str,
    params: &[SqlParam],
    mapper: &mut impl FnMut(&dyn RowAccess) -> Result<T>,
) -> Result<Vec<T>> {
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
    use crate::config::VEXDB_LITE_EXTENSION_ENV;

    fn vexdb_connection() -> Connection {
        let path = std::env::var_os(VEXDB_LITE_EXTENSION_ENV)
            .unwrap_or_else(|| panic!("set {VEXDB_LITE_EXTENSION_ENV} to run memme-core tests"));
        let conn = Connection::open_in_memory().unwrap();
        load_vexdb_lite(&conn, std::path::Path::new(&path)).unwrap();
        conn
    }

    #[test]
    fn test_vexdb_lite_loads() {
        let conn = vexdb_connection();
        let version: String = conn
            .query_row("SELECT vexdb_version()", [], |row| row.get(0))
            .unwrap();
        assert!(!version.is_empty());
    }

    #[test]
    fn test_vexdb_cosine_distance() {
        let conn = vexdb_connection();

        // Identical vectors → distance 0
        let a: Vec<f32> = vec![1.0, 0.0, 0.0];
        let b: Vec<f32> = vec![1.0, 0.0, 0.0];
        let a_blob: &[u8] = bytemuck::cast_slice(&a);
        let b_blob: &[u8] = bytemuck::cast_slice(&b);
        let dist: f64 = conn
            .query_row(
                "SELECT vexdb_cosine_distance(?1, ?2)",
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
                "SELECT vexdb_cosine_distance(?1, ?2)",
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
    fn test_vexdb_knn_search() {
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
            .vector_search(&[1.0, 0.0, 0.0], "alice", None, false, None, None, None, 3)
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
    fn test_vexdb_delete_sync() {
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
            .vector_search(&[1.0, 0.0, 0.0], "alice", None, false, None, None, None, 5)
            .unwrap();
        assert_eq!(results.len(), 1);

        // Delete it
        storage.delete_memory("m1").unwrap();

        // Should no longer be found
        let results = storage
            .vector_search(&[1.0, 0.0, 0.0], "alice", None, false, None, None, None, 5)
            .unwrap();
        assert_eq!(results.len(), 0);
    }
}

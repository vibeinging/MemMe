use std::sync::{Mutex, MutexGuard};

use duckdb::Connection;
use tracing::debug;

use crate::config::MemoryConfig;
use crate::error::{MemoryError, Result};

/// Acquire a mutex lock, recovering from poison if a previous thread panicked.
/// DuckDB connections remain valid after Rust-side panics, so recovery is safe.
fn recover_lock<'a, T>(mutex: &'a Mutex<T>, label: &str) -> MutexGuard<'a, T> {
    mutex.lock().unwrap_or_else(|e| {
        tracing::warn!("{} mutex was poisoned, recovering", label);
        e.into_inner()
    })
}

/// A connection pool that supports concurrent reads with a single writer.
///
/// For file-backed databases: maintains a write connection and N read connections.
/// For in-memory databases: falls back to a single shared connection (no concurrency gain).
pub(crate) struct ConnectionPool {
    inner: PoolInner,
}

#[allow(dead_code)] // Multi variant planned for concurrent read/write
enum PoolInner {
    /// File-backed DB: separate read and write connections.
    Multi {
        write_conn: Mutex<Connection>,
        read_conns: Mutex<Vec<Connection>>,
        db_path: String,
    },
    /// In-memory DB: single connection shared for reads and writes.
    Single { conn: Mutex<Connection> },
}

/// A guard that provides access to a DuckDB connection.
/// Handles both pooled (Multi) and shared (Single) modes transparently.
pub(crate) enum ConnGuard<'a> {
    /// Owns a connection temporarily taken from the read pool.
    Pooled {
        conn: Option<Connection>,
        pool: &'a ConnectionPool,
    },
    /// Holds a MutexGuard on the connection (for write or Single mode).
    Locked { guard: MutexGuard<'a, Connection> },
}

impl<'a> std::ops::Deref for ConnGuard<'a> {
    type Target = Connection;
    fn deref(&self) -> &Connection {
        match self {
            ConnGuard::Pooled { conn, .. } => conn.as_ref().expect("connection already returned"),
            ConnGuard::Locked { guard } => guard,
        }
    }
}

impl<'a> Drop for ConnGuard<'a> {
    fn drop(&mut self) {
        if let ConnGuard::Pooled { conn, pool } = self {
            if let Some(c) = conn.take() {
                pool.return_read(c);
            }
        }
        // Locked variant: MutexGuard releases automatically
    }
}

#[allow(dead_code)] // used by Multi pool mode
const DEFAULT_READ_POOL_SIZE: usize = 4;

impl ConnectionPool {
    /// Open a connection pool. For file-backed DBs, creates 1 write + N read connections.
    /// For in-memory DBs, creates a single shared connection.
    ///
    /// The `init` callback is called with the write connection to initialize the schema.
    pub fn open(
        config: &MemoryConfig,
        init: impl FnOnce(&Connection) -> Result<()>,
    ) -> Result<Self> {
        if config.db_path == ":memory:" {
            let conn = Connection::open_in_memory().map_err(MemoryError::DuckDb)?;
            init(&conn)?;
            debug!("ConnectionPool: opened in-memory (single connection mode)");
            Ok(Self {
                inner: PoolInner::Single {
                    conn: Mutex::new(conn),
                },
            })
        } else {
            // Use single-connection mode for file-backed databases too.
            // DuckDB multi-connection has snapshot isolation which prevents
            // read connections from seeing writes made by the write connection.
            // This causes read-after-write failures in flows like compact.
            let conn = Connection::open(&config.db_path).map_err(MemoryError::DuckDb)?;
            init(&conn)?;

            debug!(
                db_path = %config.db_path,
                "ConnectionPool: opened file-backed (single connection mode)"
            );

            Ok(Self {
                inner: PoolInner::Single {
                    conn: Mutex::new(conn),
                },
            })
        }
    }

    /// Acquire a read connection from the pool.
    ///
    /// - Multi mode: pops from pool (or creates a temporary connection if exhausted).
    /// - Single mode: locks the shared connection.
    pub fn acquire_read(&self) -> ConnGuard<'_> {
        match &self.inner {
            PoolInner::Multi {
                read_conns,
                db_path,
                write_conn,
            } => {
                let mut pool = recover_lock(read_conns, "read pool");
                let conn = if let Some(c) = pool.pop() {
                    c
                } else {
                    debug!("ConnectionPool: read pool exhausted, creating temporary connection");
                    match Connection::open(db_path) {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::error!("Failed to open temporary read connection: {e}, falling back to write connection");
                            drop(pool);
                            return ConnGuard::Locked {
                                guard: recover_lock(write_conn, "write conn fallback"),
                            };
                        }
                    }
                };
                ConnGuard::Pooled {
                    conn: Some(conn),
                    pool: self,
                }
            }
            PoolInner::Single { conn } => ConnGuard::Locked {
                guard: recover_lock(conn, "single conn"),
            },
        }
    }

    /// Acquire the write connection.
    ///
    /// - Multi mode: locks the dedicated write connection.
    /// - Single mode: locks the shared connection (same as read).
    pub fn acquire_write(&self) -> ConnGuard<'_> {
        match &self.inner {
            PoolInner::Multi { write_conn, .. } => ConnGuard::Locked {
                guard: recover_lock(write_conn, "write conn"),
            },
            PoolInner::Single { conn } => ConnGuard::Locked {
                guard: recover_lock(conn, "single conn"),
            },
        }
    }

    /// Return a read connection to the pool. Only meaningful for Multi mode.
    fn return_read(&self, conn: Connection) {
        if let PoolInner::Multi { read_conns, .. } = &self.inner {
            recover_lock(read_conns, "read pool return").push(conn);
        }
    }

    /// Refresh all read connections (e.g., after FTS index rebuild).
    /// Closes existing read connections and opens fresh ones.
    #[allow(dead_code)] // used by Multi pool mode
    pub fn refresh_readers(&self) {
        if let PoolInner::Multi {
            read_conns,
            db_path,
            ..
        } = &self.inner
        {
            let mut pool = recover_lock(read_conns, "read pool refresh");
            pool.clear();
            for _ in 0..DEFAULT_READ_POOL_SIZE {
                if let Ok(c) = Connection::open(db_path) {
                    pool.push(c);
                }
            }
            debug!("ConnectionPool: refreshed read connections");
        }
    }
}

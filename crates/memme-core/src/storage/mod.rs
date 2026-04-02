use std::collections::HashMap;

use duckdb::Connection;
use tracing::{debug, info, warn};

use crate::config::MemoryConfig;
use crate::error::{MemoryError, Result};

mod analytics;
mod config_store;
mod crud;
pub(crate) use crud::InsertMemoryParams;
mod entity_links;
mod episode;
mod export;
mod graph;
mod history;
mod identity_store;
mod meditation_store;
pub(crate) mod pool;
mod procedural;
mod query;
mod recall_store;
mod session;
mod stream;
pub(crate) mod replica;
mod sync;
mod util;

use pool::{ConnGuard, ConnectionPool};

/// DuckDB storage layer handling schema initialization and raw operations.
///
/// Uses a ConnectionPool internally for read-write separation:
/// - File-backed DBs: 1 write connection + N read connections (concurrent reads)
/// - In-memory DBs: single shared connection (no concurrency gain)
pub struct Storage {
    pub(crate) pool: ConnectionPool,
    pub(crate) config: MemoryConfig,
}

impl Storage {
    /// Open a DuckDB connection pool and initialize the schema.
    ///
    /// If the primary database file is corrupted and a replica exists,
    /// automatically recovers from the replica before opening.
    pub fn open(config: MemoryConfig) -> Result<Self> {
        config.validate()?;

        let cfg_ref = config.clone();
        let pool_result = ConnectionPool::open(&config, |conn| {
            Self::run_init_schema(conn, &cfg_ref)
        });

        let pool = match pool_result {
            Ok(p) => p,
            Err(_) if config.db_path != ":memory:" => {
                // Primary failed to open — try recovering from replica
                if replica::try_recover_from_replica(&config.db_path) {
                    let cfg_ref2 = config.clone();
                    ConnectionPool::open(&config, |conn| {
                        Self::run_init_schema(conn, &cfg_ref2)
                    })?
                } else {
                    return pool_result.map(|p| Self { pool: p, config });
                }
            }
            Err(e) => return Err(e),
        };

        Ok(Self { pool, config })
    }

    /// Acquire a read connection from the pool.
    #[inline]
    pub(crate) fn read_conn(&self) -> ConnGuard<'_> {
        self.pool.acquire_read()
    }

    /// Acquire the write connection.
    #[inline]
    pub(crate) fn write_conn(&self) -> ConnGuard<'_> {
        self.pool.acquire_write()
    }

    /// Refresh read connections (call after FTS index rebuild).
    #[allow(dead_code)] // planned: used after FTS index rebuild
    pub(crate) fn refresh_readers(&self) {
        self.pool.refresh_readers();
    }

    /// Helper: execute SQL ignoring errors (for optional features like extensions).
    fn exec_ignore(conn: &Connection, sql: &str) {
        let _ = conn.execute_batch(sql);
    }

    /// Current schema version. Bump this when adding new tables/columns/indexes.
    const SCHEMA_VERSION: &'static str = "5";

    /// Create tables and indexes if they do not exist.
    fn run_init_schema(conn: &Connection, config: &MemoryConfig) -> Result<()> {
        let dims = config.embedding_dims;
        let collection = &config.collection_name;

        // Try to load JSON extension (optional, for json_extract_string in metadata filters)
        Self::exec_ignore(conn, "LOAD json");

        // Fast path: if schema is already at current version, skip all CREATE/ALTER/INDEX.
        if Self::check_schema_version(conn) {
            debug!(collection = %collection, "Schema up-to-date (v{}), skipping init", Self::SCHEMA_VERSION);
            return Ok(());
        }

        info!(collection = %collection, dims, "Initializing memory schema");

        // --- memories table ---
        let create_memories = format!(
            r#"CREATE TABLE IF NOT EXISTS memories (
                id VARCHAR PRIMARY KEY,
                content VARCHAR NOT NULL,
                embedding FLOAT[{dims}],
                user_id VARCHAR NOT NULL,
                agent_id VARCHAR,
                run_id VARCHAR,
                app_id VARCHAR,
                actor_id VARCHAR,
                importance FLOAT DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                hash VARCHAR,
                created_at TIMESTAMP DEFAULT current_timestamp,
                updated_at TIMESTAMP DEFAULT current_timestamp,
                metadata VARCHAR,
                immutable BOOLEAN DEFAULT false,
                expiration_date TIMESTAMP,
                categories VARCHAR[],
                memory_type VARCHAR,
                stability FLOAT DEFAULT 1.0,
                privacy VARCHAR DEFAULT 'syncable',
                event_time TIMESTAMP,
                episode_id VARCHAR,
                session_id VARCHAR,
                resolution VARCHAR DEFAULT 'granular'
            )"#
        );
        conn.execute_batch(&create_memories)?;
        debug!("Created memories table");

        // --- Migrate existing tables: add new columns if they don't exist ---
        Self::exec_ignore(conn, "ALTER TABLE memories ADD COLUMN app_id VARCHAR");
        Self::exec_ignore(
            conn,
            "ALTER TABLE memories ADD COLUMN immutable BOOLEAN DEFAULT false",
        );
        Self::exec_ignore(
            conn,
            "ALTER TABLE memories ADD COLUMN expiration_date TIMESTAMP",
        );
        Self::exec_ignore(conn, "ALTER TABLE memories ADD COLUMN categories VARCHAR[]");
        Self::exec_ignore(conn, "ALTER TABLE memories ADD COLUMN memory_type VARCHAR");
        Self::exec_ignore(
            conn,
            "ALTER TABLE memories ADD COLUMN stability FLOAT DEFAULT 1.0",
        );
        Self::exec_ignore(
            conn,
            "ALTER TABLE memories ADD COLUMN privacy VARCHAR DEFAULT 'syncable'",
        );
        Self::exec_ignore(conn, "ALTER TABLE memories ADD COLUMN event_time TIMESTAMP");
        Self::exec_ignore(conn, "ALTER TABLE memories ADD COLUMN episode_id VARCHAR");
        Self::exec_ignore(conn, "ALTER TABLE memories ADD COLUMN session_id VARCHAR");
        Self::exec_ignore(
            conn,
            "ALTER TABLE memories ADD COLUMN resolution VARCHAR DEFAULT 'granular'",
        );

        // --- Sync columns ---
        Self::exec_ignore(
            conn,
            "ALTER TABLE memories ADD COLUMN sync_version BIGINT DEFAULT 0",
        );
        Self::exec_ignore(conn, "ALTER TABLE memories ADD COLUMN device_id VARCHAR");
        Self::exec_ignore(
            conn,
            "ALTER TABLE memories ADD COLUMN sync_status VARCHAR DEFAULT 'pending'",
        );

        // --- history table ---
        let create_history = r#"
            CREATE TABLE IF NOT EXISTS history (
                id VARCHAR PRIMARY KEY,
                memory_id VARCHAR,
                user_id VARCHAR,
                old_memory VARCHAR,
                new_memory VARCHAR,
                event VARCHAR,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        "#;
        conn.execute_batch(create_history)?;
        debug!("Created history table");

        // --- entities table ---
        let create_entities = format!(
            r#"CREATE TABLE IF NOT EXISTS entities_{collection} (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                entity_type VARCHAR,
                user_id VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT current_timestamp,
                updated_at TIMESTAMP DEFAULT current_timestamp
            )"#
        );
        conn.execute_batch(&create_entities)?;
        debug!("Created entities table");

        // --- relationships table ---
        let create_relationships = format!(
            r#"CREATE TABLE IF NOT EXISTS relationships_{collection} (
                id VARCHAR PRIMARY KEY,
                source_id VARCHAR NOT NULL,
                target_id VARCHAR NOT NULL,
                relation_type VARCHAR NOT NULL,
                user_id VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT current_timestamp,
                FOREIGN KEY (source_id) REFERENCES entities_{collection}(id),
                FOREIGN KEY (target_id) REFERENCES entities_{collection}(id)
            )"#
        );
        conn.execute_batch(&create_relationships)?;
        // Indexes for graph traversal performance
        let _ = conn.execute_batch(&format!(
            "CREATE INDEX IF NOT EXISTS idx_rel_source_{collection} ON relationships_{collection} (source_id, user_id)"
        ));
        let _ = conn.execute_batch(&format!(
            "CREATE INDEX IF NOT EXISTS idx_rel_target_{collection} ON relationships_{collection} (target_id, user_id)"
        ));
        debug!("Created relationships table");

        // --- memory_entities table (links memories to entities for entity-centric retrieval) ---
        let create_memory_entities = r#"CREATE TABLE IF NOT EXISTS memory_entities (
            memory_id VARCHAR NOT NULL,
            entity_id VARCHAR NOT NULL,
            entity_name VARCHAR NOT NULL,
            user_id VARCHAR NOT NULL,
            PRIMARY KEY (memory_id, entity_id)
        )"#;
        conn.execute_batch(create_memory_entities)?;
        debug!("Created memory_entities table");

        // --- procedures table ---
        let create_procedures = r#"CREATE TABLE IF NOT EXISTS procedures (
            id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            description VARCHAR,
            steps VARCHAR,
            user_id VARCHAR NOT NULL,
            trigger_pattern VARCHAR,
            confidence FLOAT DEFAULT 0.5,
            usage_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT current_timestamp,
            updated_at TIMESTAMP DEFAULT current_timestamp
        )"#;
        conn.execute_batch(create_procedures)?;
        debug!("Created procedures table");

        // --- Four-layer architecture: migration columns on existing tables ---
        Self::exec_ignore(
            conn,
            "ALTER TABLE memories ADD COLUMN storage_strength FLOAT DEFAULT 1.0",
        );
        Self::exec_ignore(
            conn,
            "ALTER TABLE memories ADD COLUMN retrieval_strength FLOAT DEFAULT 1.0",
        );
        Self::exec_ignore(
            conn,
            "ALTER TABLE memories ADD COLUMN superseded_by VARCHAR",
        );
        Self::exec_ignore(conn, "ALTER TABLE memories ADD COLUMN valid_from TIMESTAMP");
        Self::exec_ignore(
            conn,
            "ALTER TABLE memories ADD COLUMN valid_until TIMESTAMP",
        );
        Self::exec_ignore(
            conn,
            "ALTER TABLE memories ADD COLUMN confidence FLOAT DEFAULT 0.8",
        );
        Self::exec_ignore(conn, "ALTER TABLE memories ADD COLUMN evidence VARCHAR");
        Self::exec_ignore(conn, "ALTER TABLE memories ADD COLUMN episode_ids VARCHAR");
        Self::exec_ignore(
            conn,
            "ALTER TABLE memories ADD COLUMN ingestion_time TIMESTAMP",
        );

        // Enhance relationships table
        let coll = &config.collection_name;
        Self::exec_ignore(
            conn,
            &format!("ALTER TABLE relationships_{coll} ADD COLUMN strength FLOAT DEFAULT 0.5"),
        );
        Self::exec_ignore(
            conn,
            &format!("ALTER TABLE relationships_{coll} ADD COLUMN context VARCHAR"),
        );
        Self::exec_ignore(
            conn,
            &format!("ALTER TABLE relationships_{coll} ADD COLUMN valid_from TIMESTAMP"),
        );
        Self::exec_ignore(
            conn,
            &format!("ALTER TABLE relationships_{coll} ADD COLUMN valid_until TIMESTAMP"),
        );
        Self::exec_ignore(
            conn,
            &format!("ALTER TABLE relationships_{coll} ADD COLUMN episode_ids VARCHAR"),
        );

        // --- sessions table (immutable conversation containers) ---
        let create_sessions = r#"CREATE TABLE IF NOT EXISTS sessions (
            session_id VARCHAR PRIMARY KEY,
            user_id VARCHAR NOT NULL,
            source_id VARCHAR,
            started_at TIMESTAMP DEFAULT current_timestamp,
            ended_at TIMESTAMP,
            metadata VARCHAR,
            created_at TIMESTAMP DEFAULT current_timestamp
        )"#;
        conn.execute_batch(create_sessions)?;
        Self::exec_ignore(conn, "ALTER TABLE sessions ADD COLUMN structured_notes VARCHAR");
        debug!("Created sessions table");

        // --- sources table ---
        let create_sources = r#"CREATE TABLE IF NOT EXISTS sources (
            source_id VARCHAR PRIMARY KEY,
            source_type VARCHAR NOT NULL,
            name VARCHAR,
            registered_at TIMESTAMP DEFAULT current_timestamp,
            metadata VARCHAR,
            user_id VARCHAR
        )"#;
        conn.execute_batch(create_sources)?;
        debug!("Created sources table");

        // --- events table (Stream layer) ---
        let create_events = format!(
            r#"CREATE TABLE IF NOT EXISTS events (
                event_id VARCHAR PRIMARY KEY,
                source_id VARCHAR,
                session_id VARCHAR,
                timestamp TIMESTAMP NOT NULL DEFAULT current_timestamp,
                event_type VARCHAR NOT NULL DEFAULT 'system',
                content VARCHAR NOT NULL,
                content_vec FLOAT[{dims}],
                parent_id VARCHAR,
                metadata VARCHAR,
                user_id VARCHAR NOT NULL,
                processed BOOLEAN DEFAULT false,
                processed_at TIMESTAMP,
                purified_content VARCHAR,
                purified BOOLEAN DEFAULT false,
                event_time TIMESTAMP,
                location VARCHAR
            )"#
        );
        conn.execute_batch(&create_events)?;
        // Add new columns if they don't exist (for existing databases)
        Self::exec_ignore(
            conn,
            "ALTER TABLE events ADD COLUMN purified_content VARCHAR",
        );
        Self::exec_ignore(
            conn,
            "ALTER TABLE events ADD COLUMN purified BOOLEAN DEFAULT false",
        );
        Self::exec_ignore(conn, "ALTER TABLE events ADD COLUMN event_time TIMESTAMP");
        Self::exec_ignore(conn, "ALTER TABLE events ADD COLUMN location VARCHAR");
        debug!("Created events table");

        // --- episodes table (Episode layer) ---
        let create_episodes = format!(
            r#"CREATE TABLE IF NOT EXISTS episodes (
                episode_id VARCHAR PRIMARY KEY,
                title VARCHAR NOT NULL,
                summary VARCHAR NOT NULL,
                summary_vec FLOAT[{dims}],
                started_at TIMESTAMP NOT NULL,
                ended_at TIMESTAMP,
                significance FLOAT DEFAULT 0.5,
                outcome VARCHAR,
                source_id VARCHAR,
                event_ids VARCHAR,
                user_id VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT current_timestamp,
                last_recalled TIMESTAMP,
                recall_count INTEGER DEFAULT 0,
                storage_strength FLOAT DEFAULT 1.0,
                retrieval_strength FLOAT DEFAULT 1.0
            )"#
        );
        conn.execute_batch(&create_episodes)?;
        Self::exec_ignore(conn, "ALTER TABLE episodes ADD COLUMN session_ids VARCHAR");
        Self::exec_ignore(conn, "ALTER TABLE episodes ADD COLUMN last_meditated_at TIMESTAMP");
        debug!("Created episodes table");

        // --- identity table (Identity layer) ---
        let create_identity = format!(
            r#"CREATE TABLE IF NOT EXISTS identity (
                trait_id VARCHAR PRIMARY KEY,
                trait_type VARCHAR NOT NULL,
                content VARCHAR NOT NULL,
                content_vec FLOAT[{dims}],
                confidence FLOAT DEFAULT 0.5,
                evidence_ids VARCHAR,
                user_id VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT current_timestamp,
                updated_at TIMESTAMP
            )"#
        );
        conn.execute_batch(&create_identity)?;
        debug!("Created identity table");

        // --- associations table (cross-layer links) ---
        let create_associations = r#"CREATE TABLE IF NOT EXISTS associations (
            assoc_id VARCHAR PRIMARY KEY,
            from_id VARCHAR NOT NULL,
            from_layer VARCHAR NOT NULL,
            to_id VARCHAR NOT NULL,
            to_layer VARCHAR NOT NULL,
            assoc_type VARCHAR NOT NULL,
            strength FLOAT DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT current_timestamp
        )"#;
        conn.execute_batch(create_associations)?;
        debug!("Created associations table");

        // --- meditations table (consolidation session log) ---
        let create_meditations = r#"CREATE TABLE IF NOT EXISTS meditations (
            meditation_id VARCHAR PRIMARY KEY,
            triggered_by VARCHAR NOT NULL,
            started_at TIMESTAMP NOT NULL,
            finished_at TIMESTAMP,
            status VARCHAR DEFAULT 'running',
            user_id VARCHAR NOT NULL,
            events_processed INTEGER DEFAULT 0,
            episodes_created INTEGER DEFAULT 0,
            memories_created INTEGER DEFAULT 0,
            memories_updated INTEGER DEFAULT 0,
            memories_decayed INTEGER DEFAULT 0,
            entities_created INTEGER DEFAULT 0,
            relations_created INTEGER DEFAULT 0,
            conflicts_found INTEGER DEFAULT 0,
            journal VARCHAR,
            metadata VARCHAR
        )"#;
        conn.execute_batch(create_meditations)?;
        debug!("Created meditations table");

        // --- recalls table (retrieval log) ---
        let create_recalls = format!(
            r#"CREATE TABLE IF NOT EXISTS recalls (
                recall_id VARCHAR PRIMARY KEY,
                query VARCHAR NOT NULL,
                query_vec FLOAT[{dims}],
                timestamp TIMESTAMP DEFAULT current_timestamp,
                source_id VARCHAR,
                user_id VARCHAR NOT NULL,
                results VARCHAR,
                feedback VARCHAR
            )"#
        );
        conn.execute_batch(&create_recalls)?;
        debug!("Created recalls table");

        // --- memme_config table (key-value config persistence) ---
        let create_config = r#"CREATE TABLE IF NOT EXISTS memme_config (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )"#;
        conn.execute_batch(create_config)?;
        debug!("Created memme_config table");

        // --- B-tree indexes for common queries ---
        Self::exec_ignore(
            conn,
            "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)",
        );
        Self::exec_ignore(
            conn,
            "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)",
        );
        Self::exec_ignore(
            conn,
            "CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)",
        );
        Self::exec_ignore(
            conn,
            "CREATE INDEX IF NOT EXISTS idx_memories_episode ON memories(episode_id)",
        );

        // --- Vector indexes ---
        // HNSW for pure vector search
        let hnsw_idx = format!(
            r#"CREATE INDEX IF NOT EXISTS idx_mem_{collection}
               ON memories USING HNSW (embedding)
               WITH (metric='cosine', m=32, ef_construction=128)"#
        );
        if let Err(e) = conn.execute_batch(&hnsw_idx) {
            warn!("HNSW index creation skipped: {e}");
        } else {
            debug!("Created HNSW index");
        }

        // HNSW with user_id metadata for filtered vector search (ACORN-style)
        let hnsw_filtered_idx = format!(
            r#"CREATE INDEX IF NOT EXISTS idx_mem_user_{collection}
               ON memories USING HNSW (embedding, user_id)
               WITH (metric='cosine')"#
        );
        if let Err(e) = conn.execute_batch(&hnsw_filtered_idx) {
            warn!("HNSW filtered index creation skipped: {e}");
        } else {
            debug!("Created HNSW filtered index");
        }

        // Stamp schema version so subsequent startups take the fast path.
        Self::stamp_schema_version(conn);
        info!("Schema initialized (v{})", Self::SCHEMA_VERSION);

        Ok(())
    }

    /// Check if schema version matches. Returns true if up-to-date.
    fn check_schema_version(conn: &Connection) -> bool {
        let result: std::result::Result<String, _> = conn.query_row(
            "SELECT value FROM memme_config WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        );
        match result {
            Ok(v) => v == Self::SCHEMA_VERSION,
            Err(_) => false, // table doesn't exist yet or no row
        }
    }

    /// Write current schema version to config table.
    fn stamp_schema_version(conn: &Connection) {
        let _ = conn.execute(
            "INSERT INTO memme_config (key, value) VALUES ('schema_version', $1) \
             ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            duckdb::params![Self::SCHEMA_VERSION],
        );
    }

    // ── Reset ──

    /// Delete ALL data from all tables. This is destructive and cannot be undone.
    pub(crate) fn reset(&self) -> Result<()> {
        let conn = self.write_conn();
        conn.execute_batch("DELETE FROM history")?;
        let collection = &self.config.collection_name;
        conn.execute_batch(&format!("DELETE FROM relationships_{collection}"))?;
        conn.execute_batch(&format!("DELETE FROM entities_{collection}"))?;
        conn.execute_batch("DELETE FROM memory_entities")?;
        conn.execute_batch("DELETE FROM memories")?;
        conn.execute_batch("DELETE FROM procedures")?;
        // Four-layer tables
        let _ = conn.execute_batch("DELETE FROM sources");
        let _ = conn.execute_batch("DELETE FROM sessions");
        let _ = conn.execute_batch("DELETE FROM events");
        let _ = conn.execute_batch("DELETE FROM episodes");
        let _ = conn.execute_batch("DELETE FROM identity");
        let _ = conn.execute_batch("DELETE FROM associations");
        let _ = conn.execute_batch("DELETE FROM meditations");
        let _ = conn.execute_batch("DELETE FROM recalls");
        Ok(())
    }
}

// ── Metadata filter helpers (legacy) ──

/// Validate metadata filter keys (alphanumeric + underscore only) and build
/// SQL WHERE clause fragments. Values are serialized as JSON strings and
/// embedded directly (after validation) to avoid dynamic parameter count issues
/// with DuckDB's `params!` macro.
#[allow(dead_code)]
fn build_metadata_filter_conditions(
    filters: &HashMap<String, serde_json::Value>,
) -> Result<Vec<String>> {
    let mut conditions = Vec::new();
    for (key, value) in filters {
        // Validate key: only allow alphanumeric and underscore to prevent SQL injection
        if !key.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(MemoryError::Config(format!(
                "Invalid metadata filter key: {key}"
            )));
        }
        if key.is_empty() {
            return Err(MemoryError::Config(
                "Metadata filter key cannot be empty".into(),
            ));
        }

        // Serialize value to a string for comparison via json_extract_string.
        // For strings, serde_json::Value::String serializes with quotes, so we
        // use the raw string. For numbers/booleans we use the JSON representation.
        let value_str = match value {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };

        // Escape single quotes in the value string to prevent SQL injection
        let escaped = value_str.replace('\'', "''");
        conditions.push(format!(
            "json_extract_string(metadata, '$.{key}') = '{escaped}'"
        ));
    }
    Ok(conditions)
}

// ── Internal row types ──

#[derive(Debug, Clone)]
pub(crate) struct MemoryRow {
    pub id: String,
    pub content: String,
    pub user_id: String,
    pub created_at: String,
    pub updated_at: String,
    pub metadata: Option<String>,
    pub score: Option<f32>,
    pub importance: Option<f32>,
    pub access_count: Option<u32>,
    pub agent_id: Option<String>,
    pub app_id: Option<String>,
    pub run_id: Option<String>,
    pub immutable: bool,
    pub expiration_date: Option<String>,
    pub categories: Option<String>,
    pub memory_type: Option<String>,
    pub stability: Option<f32>,
    pub privacy: Option<String>,
    pub event_time: Option<String>,
    pub episode_id: Option<String>,
    pub session_id: Option<String>,
    pub resolution: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ProcedureRow {
    pub id: String,
    pub name: String,
    pub description: String,
    pub steps: Option<String>,
    pub user_id: String,
    pub trigger_pattern: Option<String>,
    pub confidence: f32,
    pub usage_count: u32,
    pub created_at: String,
    pub updated_at: String,
}

/// Check whether the MemMe-DB extension is loaded in the current connection.
/// Returns `true` if the `vex` extension is available and HNSW can be used.
#[allow(dead_code)]
pub fn is_memme_db_available(conn: &Connection) -> bool {
    // Try a lightweight probe: create and drop a temp table with HNSW
    let result = conn.execute_batch(
        "CREATE TEMP TABLE __vex_probe (id INTEGER, v FLOAT[4]);
         INSERT INTO __vex_probe VALUES (1, [1.0, 0.0, 0.0, 0.0]);
         CREATE INDEX __vex_probe_idx ON __vex_probe USING HNSW (v) WITH (metric='cosine');
         DROP INDEX __vex_probe_idx;
         DROP TABLE __vex_probe;",
    );
    result.is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(dims: usize) -> MemoryConfig {
        MemoryConfig {
            db_path: ":memory:".into(),
            collection_name: "test".into(),
            embedding_dims: dims,
            dedup_threshold: 0.15,
            default_limit: 10,
            ..Default::default()
        }
    }

    fn open_storage(dims: usize) -> Storage {
        Storage::open(test_config(dims)).unwrap()
    }

    fn dummy_embedding(dims: usize, seed: f32) -> Vec<f32> {
        (0..dims).map(|i| (i as f32 * 0.01 + seed).sin()).collect()
    }

    #[test]
    fn test_schema_creation() {
        let storage = open_storage(384);
        // Verify we can insert — tables must exist
        let emb = dummy_embedding(384, 1.0);
        storage
            .insert_memory(
                "id1",
                "content",
                &emb,
                "user1",
                "hash1",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        let row = storage.get_memory("id1").unwrap();
        assert!(row.is_some());
    }

    #[test]
    fn test_format_embedding() {
        let emb = vec![0.1, 0.2, 0.3];
        let formatted = Storage::format_embedding(&emb, 3).unwrap();
        assert!(formatted.starts_with('['));
        assert!(formatted.ends_with("::FLOAT[3]"));
        assert!(formatted.contains("0.1"));
        assert!(formatted.contains("0.2"));
        assert!(formatted.contains("0.3"));
    }

    #[test]
    fn test_format_embedding_rejects_nan() {
        let emb = vec![0.1, f32::NAN, 0.3];
        let result = Storage::format_embedding(&emb, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_embedding_rejects_infinity() {
        let emb = vec![0.1, f32::INFINITY, 0.3];
        let result = Storage::format_embedding(&emb, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_categories() {
        assert_eq!(Storage::parse_categories(None), None);
        assert_eq!(Storage::parse_categories(Some("[]".to_string())), None);
        assert_eq!(Storage::parse_categories(Some("".to_string())), None);

        let cats = Storage::parse_categories(Some("[work, tech]".to_string()));
        assert_eq!(cats, Some(vec!["work".to_string(), "tech".to_string()]));

        let cats = Storage::parse_categories(Some("['work', 'tech']".to_string()));
        assert_eq!(cats, Some(vec!["work".to_string(), "tech".to_string()]));
    }

    /// Test that MemMe-DB extension is loaded and HNSW works.
    /// Only runs when compiled with `--features memme-db` (and without `bundled`).
    #[test]
    #[cfg(all(feature = "memme-db", not(feature = "bundled")))]
    fn test_memme_db_extension_loaded() {
        let conn = Connection::open_in_memory().unwrap();
        assert!(
            is_memme_db_available(&conn),
            "MemMe-DB extension is not loaded. \
             HNSW should be available when built with the memme-db feature."
        );
    }

    /// Test HNSW index end-to-end with MemMe-DB.
    #[test]
    #[cfg(all(feature = "memme-db", not(feature = "bundled")))]
    fn test_memme_db_hnsw_index_e2e() {
        let storage = open_storage(4);

        // Insert test data
        let emb1 = vec![1.0, 0.0, 0.0, 0.0];
        let emb2 = vec![0.0, 1.0, 0.0, 0.0];
        let emb3 = vec![0.9, 0.1, 0.0, 0.0];

        storage
            .insert_memory(
                "id1",
                "hello",
                &emb1,
                "user1",
                "h1",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        storage
            .insert_memory(
                "id2",
                "world",
                &emb2,
                "user1",
                "h2",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        storage
            .insert_memory(
                "id3",
                "close",
                &emb3,
                "user1",
                "h3",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        // Verify HNSW was created (not just silently ignored)
        let mut stmt = storage
            .conn
            .prepare("SELECT index_name FROM duckdb_indexes()")
            .unwrap();
        let indexes: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            !indexes.is_empty(),
            "No HNSW found — MemMe-DB extension may not be loaded"
        );

        // Vector search should work using the index
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = storage
            .vector_search(&query, "user1", None, None, None, None, 3)
            .unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "id1"); // exact match should be first
    }
}

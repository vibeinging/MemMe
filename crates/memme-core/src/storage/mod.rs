use crate::config::{MemoryConfig, VEXDB_LITE_EXTENSION_ENV};
use crate::error::{MemoryError, Result};
use crate::types::SqlParam;

mod analytics;
pub(crate) mod backend;
mod backend_sqlite;
mod config_store;
mod crud;
pub(crate) use crud::InsertMemoryParams;
pub(crate) mod dialect;
mod dialect_sqlite;
mod entity_links;
mod episode;
mod export;
mod graph;
mod history;
mod identity_store;
mod meditation_store;
pub(crate) mod portable_import;
mod procedural;
pub(crate) mod query;
mod recall_store;
pub(crate) mod replica;
mod session;
mod stream;
mod sync;
mod util;

use backend::Backend;
use dialect::SqlDialect;

/// Storage layer handling schema initialization and raw operations.
///
/// Uses a [`Backend`] enum for database-agnostic connection management
/// and a [`SqlDialect`] for database-specific SQL generation.
/// All database operations go through `self.backend.execute()`,
/// `self.backend.query_read()`, etc.
pub struct Storage {
    pub(crate) backend: Backend,
    pub(crate) config: MemoryConfig,
}

impl Storage {
    const VECTOR_INDEX_VERSION: &'static str = "3";

    /// Open a SQLite-backed storage and initialize the schema.
    pub fn open(config: MemoryConfig) -> Result<Self> {
        let extension_path = std::env::var_os(VEXDB_LITE_EXTENSION_ENV).ok_or_else(|| {
            MemoryError::Config(format!(
                "VexDB-Lite SQLite extension is required; set {VEXDB_LITE_EXTENSION_ENV} \
                 or use MemoryStore::new_with_vexdb_lite()"
            ))
        })?;
        Self::open_with_vexdb_lite(config, extension_path)
    }

    /// Open SQLite with an explicit VexDB-Lite extension path.
    pub fn open_with_vexdb_lite(
        config: MemoryConfig,
        extension_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        config.validate()?;
        let conn = backend_sqlite::open_sqlite(&config, extension_path.as_ref())?;
        let backend = Backend::sqlite(conn);
        // Initialize schema using the dialect
        let dialect = backend.dialect();
        let schema = Self::sqlite_init_schema(&config, dialect);
        backend.execute_batch(&schema)?;
        // The source tables are authoritative. Validate them before dropping or
        // recreating any derived vector index so a bad configuration cannot
        // destroy the last usable index before open returns an error.
        Self::validate_source_embedding_dimensions(&backend, config.embedding_dims)?;
        let vector_table_count = backend.query_count(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN ($1, $2)",
            &[
                SqlParam::Text(dialect.memory_vector_table_name().to_string()),
                SqlParam::Text(dialect.event_vector_table_name().to_string()),
            ],
        )?;
        let mut vector_tables_existed = vector_table_count == 2;
        if vector_table_count > 0 {
            if vector_tables_existed {
                Self::validate_vector_table_dimensions(&backend, dialect, config.embedding_dims)?;
            }
            let stored_version = backend.query_one(
                "SELECT value FROM memme_config WHERE key = 'vector_index_version'",
                &[],
                |row| row.get_string(0),
            )?;
            if !vector_tables_existed
                || stored_version.as_deref() != Some(Self::VECTOR_INDEX_VERSION)
            {
                backend.execute_batch(&format!(
                    "DROP TABLE IF EXISTS {}; DROP TABLE IF EXISTS {}",
                    dialect.memory_vector_table_name(),
                    dialect.event_vector_table_name()
                ))?;
                vector_tables_existed = false;
            }
        }
        // Create VexDB-Lite virtual tables for indexed vector search.
        if let Some(vector_sql) = dialect.create_vector_index_sql(config.embedding_dims) {
            backend.execute_batch(&vector_sql)?;
        }
        let storage = Self { backend, config };
        storage.run_migrations()?;
        storage.ensure_vector_index(vector_tables_existed)?;
        storage.ensure_fts_indexes()?;
        storage.retire_legacy_vec0_data()?;
        Ok(storage)
    }

    fn validate_vector_table_dimensions(
        backend: &Backend,
        dialect: &dyn SqlDialect,
        expected_dims: usize,
    ) -> Result<()> {
        for table in [
            dialect.memory_vector_table_name(),
            dialect.event_vector_table_name(),
        ] {
            let sql = format!("SELECT value FROM {table}_config WHERE key = 'dim'");
            let raw = backend
                .query_one(&sql, &[], |row| row.get_string(0))?
                .ok_or_else(|| {
                    MemoryError::Config(format!(
                        "VexDB-Lite index {table} is missing its dimension metadata"
                    ))
                })?;
            let actual_dims = raw.parse::<usize>().map_err(|_| {
                MemoryError::Config(format!(
                    "VexDB-Lite index {table} has invalid dimension metadata: {raw}"
                ))
            })?;
            if actual_dims != expected_dims {
                return Err(MemoryError::Config(format!(
                    "VexDB-Lite index dimension mismatch for {table}: database uses \
                     {actual_dims}, configuration requests {expected_dims}; re-embed into a new \
                     database or reopen with the original dimension"
                )));
            }
        }
        Ok(())
    }

    fn validate_source_embedding_dimensions(backend: &Backend, expected_dims: usize) -> Result<()> {
        let expected_bytes = expected_dims.saturating_mul(4);
        for (table, column) in [("memories", "embedding"), ("events", "content_vec")] {
            let invalid = backend.query_count(
                &format!(
                    "SELECT COUNT(*) FROM {table} WHERE {column} IS NOT NULL \
                     AND length({column}) > 0 AND length({column}) != $1"
                ),
                &[SqlParam::Int(expected_bytes as i64)],
            )?;
            if invalid > 0 {
                return Err(MemoryError::Config(format!(
                    "cannot build a {expected_dims}-dimension VexDB-Lite index: {invalid} rows in \
                     {table}.{column} use a different embedding size"
                )));
            }
        }
        Ok(())
    }

    /// Rebuild the VexDB-Lite index from authoritative embeddings when needed.
    fn ensure_vector_index(&self, vector_tables_existed: bool) -> Result<()> {
        let backend_name = self.dialect().vector_backend_name();
        let previous_backend = self.backend.query_one(
            "SELECT value FROM memme_config WHERE key = 'vector_backend'",
            &[],
            |row| row.get_string(0),
        )?;

        let needs_rebuild =
            !vector_tables_existed || previous_backend.as_deref() != Some(backend_name);
        let rebuild_sql = self.dialect().rebuild_vector_index_sql();
        let index_sql = self.dialect().vector_metadata_indexes_sql();
        self.backend.transaction(|tx| {
            if needs_rebuild {
                if let Some(rebuild_sql) = rebuild_sql.as_deref() {
                    tx.execute_batch(rebuild_sql)?;
                }
            }
            if let Some(index_sql) = index_sql.as_deref() {
                tx.execute_batch(index_sql)?;
            }
            for (key, value) in [
                ("vector_backend", backend_name.to_string()),
                ("vector_dimensions", self.config.embedding_dims.to_string()),
                (
                    "vector_index_version",
                    Self::VECTOR_INDEX_VERSION.to_string(),
                ),
            ] {
                tx.execute(
                    "INSERT INTO memme_config(key, value) VALUES ($1, $2) \
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    &[SqlParam::Text(key.to_string()), SqlParam::Text(value)],
                )?;
            }
            Ok(())
        })
    }

    /// Remove persisted sqlite-vec shadow data after the Vex index is ready.
    ///
    /// The legacy virtual-table schema rows remain because SQLite cannot invoke
    /// sqlite-vec's xDestroy after the module has been removed. Editing
    /// sqlite_schema automatically would risk database corruption, so this
    /// migration only removes the verified ordinary shadow tables that contain
    /// vectors and partition metadata. Rollback to a sqlite-vec build therefore
    /// requires restoring a pre-migration backup.
    fn retire_legacy_vec0_data(&self) -> Result<()> {
        let mut shadow_tables = Vec::new();
        for root in ["vec_memories", "vec_events"] {
            let ddl = self.backend.query_one(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = $1",
                &[SqlParam::Text(root.to_string())],
                |row| row.get_opt_string(0),
            )?;
            let Some(Some(ddl)) = ddl else {
                continue;
            };
            let normalized = ddl.to_ascii_lowercase();
            if !normalized.contains("create virtual table") || !normalized.contains("using vec0") {
                return Err(MemoryError::Config(format!(
                    "refusing to retire unexpected legacy table {root}: schema is not sqlite-vec vec0"
                )));
            }

            let pattern = format!("{root}_*");
            let names = self.backend.query_read(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name GLOB $1",
                &[SqlParam::Text(pattern)],
                |row| row.get_string(0),
            )?;
            for name in names {
                if !Self::is_known_vec0_shadow(root, &name) {
                    return Err(MemoryError::Config(format!(
                        "refusing to remove unknown sqlite-vec shadow table: {name}"
                    )));
                }
                shadow_tables.push(name);
            }
        }

        if shadow_tables.is_empty() {
            return Ok(());
        }

        self.backend.transaction(|tx| {
            for name in &shadow_tables {
                tx.execute_batch(&format!("DROP TABLE IF EXISTS \"{name}\""))?;
            }
            tx.execute(
                "INSERT INTO memme_config(key, value) VALUES ('legacy_vec0_retired', '1') \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                &[],
            )?;
            Ok(())
        })
    }

    fn is_known_vec0_shadow(root: &str, name: &str) -> bool {
        let Some(suffix) = name.strip_prefix(root) else {
            return false;
        };
        if matches!(suffix, "_info" | "_chunks" | "_rowids" | "_auxiliary") {
            return true;
        }
        ["_vector_chunks", "_metadatachunks", "_metadatatext"]
            .iter()
            .any(|prefix| {
                suffix.strip_prefix(prefix).is_some_and(|digits| {
                    digits.len() == 2 && digits.chars().all(|c| c.is_ascii_digit())
                })
            })
    }

    /// Run incremental migrations for columns added after initial schema.
    fn run_migrations(&self) -> Result<()> {
        // Migration: normalize event relationship scope. Older callers may
        // already have stored agent_id inside metadata.
        let has_event_agent =
            self.backend
                .query_read("SELECT agent_id FROM events LIMIT 0", &[], |_| Ok(()));
        if has_event_agent.is_err() {
            self.backend.execute_batch(
                "ALTER TABLE events ADD COLUMN agent_id TEXT; \
                 UPDATE events SET agent_id = json_extract(metadata, '$.agent_id') \
                 WHERE metadata IS NOT NULL AND json_valid(metadata);",
            )?;
        }
        self.backend.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_events_user_agent \
             ON events(user_id, agent_id);",
        )?;

        // Migration: add application and run scopes to raw events. Older rows
        // already carry these values inside metadata when set by callers.
        for (column, json_key) in [("app_id", "app_id"), ("run_id", "run_id")] {
            let has_column = self.backend.query_read(
                &format!("SELECT {column} FROM events LIMIT 0"),
                &[],
                |_| Ok(()),
            );
            if has_column.is_err() {
                self.backend.execute_batch(&format!(
                    "ALTER TABLE events ADD COLUMN {column} TEXT; \
                     UPDATE events SET {column} = json_extract(metadata, '$.{json_key}') \
                     WHERE metadata IS NOT NULL AND json_valid(metadata);"
                ))?;
            }
        }
        self.backend.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_events_user_scope \
             ON events(user_id, agent_id, app_id, run_id);",
        )?;

        // Old versions could persist an LLM credential in the SQLite file.
        // Secrets now come only from the process environment.
        self.backend.execute(
            "DELETE FROM memme_config WHERE key = $1",
            &[SqlParam::Text("llm_api_key".to_string())],
        )?;

        // Migration: add pinned column to memories (added in v0.2)
        let has_pinned =
            self.backend
                .query_read("SELECT pinned FROM memories LIMIT 0", &[], |_| Ok(()));
        if has_pinned.is_err() {
            let _ = self
                .backend
                .execute_batch("ALTER TABLE memories ADD COLUMN pinned INTEGER DEFAULT 0;");
        }
        Ok(())
    }

    /// Generate SQLite schema DDL.
    fn sqlite_init_schema(config: &MemoryConfig, dialect: &dyn SqlDialect) -> String {
        let dims = config.embedding_dims;
        let collection = &config.collection_name;
        let emb_type = dialect.embedding_column_type(dims);
        let arr_type = dialect.string_array_column_type();

        format!(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding {emb_type},
                user_id TEXT NOT NULL,
                agent_id TEXT,
                run_id TEXT,
                app_id TEXT,
                actor_id TEXT,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                hash TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                metadata TEXT,
                immutable INTEGER DEFAULT 0,
                expiration_date TEXT,
                categories {arr_type},
                memory_type TEXT,
                stability REAL DEFAULT 1.0,
                privacy TEXT DEFAULT 'syncable',
                event_time TEXT,
                episode_id TEXT,
                session_id TEXT,
                resolution TEXT DEFAULT 'granular',
                storage_strength REAL DEFAULT 1.0,
                retrieval_strength REAL DEFAULT 1.0,
                superseded_by TEXT,
                valid_from TEXT,
                valid_until TEXT,
                confidence REAL DEFAULT 0.8,
                evidence TEXT,
                episode_ids TEXT,
                ingestion_time TEXT,
                sync_version INTEGER DEFAULT 0,
                device_id TEXT,
                sync_status TEXT DEFAULT 'pending',
                pinned INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                memory_id TEXT,
                user_id TEXT,
                old_memory TEXT,
                new_memory TEXT,
                event TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS entities_{collection} (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT,
                user_id TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS relationships_{collection} (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                strength REAL DEFAULT 0.5,
                context TEXT,
                description TEXT,
                episode_ids TEXT,
                FOREIGN KEY (source_id) REFERENCES entities_{collection}(id),
                FOREIGN KEY (target_id) REFERENCES entities_{collection}(id)
            );

            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                PRIMARY KEY (memory_id, entity_id)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                source_id TEXT,
                started_at TEXT DEFAULT (datetime('now')),
                ended_at TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                structured_notes TEXT,
                queried_count INTEGER DEFAULT 0,
                last_queried_at TEXT
            );

            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                name TEXT,
                registered_at TEXT DEFAULT (datetime('now')),
                metadata TEXT,
                user_id TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                source_id TEXT,
                session_id TEXT,
                agent_id TEXT,
                app_id TEXT,
                run_id TEXT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                event_type TEXT NOT NULL DEFAULT 'system',
                content TEXT NOT NULL,
                content_vec {emb_type},
                parent_id TEXT,
                metadata TEXT,
                user_id TEXT NOT NULL,
                processed INTEGER DEFAULT 0,
                processed_at TEXT,
                purified_content TEXT,
                purified INTEGER DEFAULT 0,
                event_time TEXT,
                location TEXT
            );

            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                summary_vec {emb_type},
                started_at TEXT NOT NULL,
                ended_at TEXT,
                significance REAL DEFAULT 0.5,
                outcome TEXT,
                source_id TEXT,
                event_ids TEXT,
                user_id TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                last_recalled TEXT,
                recall_count INTEGER DEFAULT 0,
                storage_strength REAL DEFAULT 1.0,
                retrieval_strength REAL DEFAULT 1.0,
                session_ids TEXT,
                last_meditated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS identity (
                trait_id TEXT PRIMARY KEY,
                trait_type TEXT NOT NULL,
                content TEXT NOT NULL,
                content_vec {emb_type},
                confidence REAL DEFAULT 0.5,
                evidence_ids TEXT,
                user_id TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS meditations (
                meditation_id TEXT PRIMARY KEY,
                triggered_by TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT DEFAULT 'running',
                user_id TEXT NOT NULL,
                events_processed INTEGER DEFAULT 0,
                episodes_created INTEGER DEFAULT 0,
                memories_created INTEGER DEFAULT 0,
                memories_updated INTEGER DEFAULT 0,
                memories_decayed INTEGER DEFAULT 0,
                entities_created INTEGER DEFAULT 0,
                relations_created INTEGER DEFAULT 0,
                conflicts_found INTEGER DEFAULT 0,
                journal TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                steps TEXT,
                user_id TEXT NOT NULL,
                trigger_pattern TEXT,
                confidence REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS associations (
                assoc_id TEXT PRIMARY KEY,
                from_id TEXT NOT NULL,
                from_layer TEXT NOT NULL,
                to_id TEXT NOT NULL,
                to_layer TEXT NOT NULL,
                assoc_type TEXT NOT NULL,
                strength REAL DEFAULT 0.5,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS recalls (
                recall_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                query_vec {emb_type},
                timestamp TEXT DEFAULT (datetime('now')),
                source_id TEXT,
                user_id TEXT NOT NULL,
                results TEXT,
                feedback TEXT
            );

            CREATE TABLE IF NOT EXISTS memme_config (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
            CREATE INDEX IF NOT EXISTS idx_memories_episode ON memories(episode_id);
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_rel_source_user ON relationships_{collection}(source_id, user_id);
            CREATE INDEX IF NOT EXISTS idx_rel_target_user ON relationships_{collection}(target_id, user_id);
            CREATE INDEX IF NOT EXISTS idx_mem_user_created ON memories(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_me_entity_lookup ON memory_entities(entity_name, memory_id);
            "#
        )
    }

    /// Access the SQL dialect for this backend.
    #[inline]
    pub(crate) fn dialect(&self) -> &dyn SqlDialect {
        self.backend.dialect()
    }

    /// Current schema version. Bump this when adding new tables/columns/indexes.
    pub(crate) const SCHEMA_VERSION: &'static str = "6";

    // ── Reset ──

    /// Delete ALL data from all tables. This is destructive and cannot be undone.
    pub(crate) fn reset(&self) -> Result<()> {
        let collection = &self.config.collection_name;
        let memory_vectors = self.dialect().memory_vector_table_name();
        let event_vectors = self.dialect().event_vector_table_name();
        self.backend.transaction(|tx| {
            tx.execute_batch(&format!(
                "DELETE FROM {memory_vectors}; DELETE FROM {event_vectors}"
            ))?;
            for fts in ["memories_fts", "events_fts", "episodes_fts"] {
                if tx.table_exists(fts)? {
                    tx.execute_batch(&format!("DELETE FROM {fts}"))?;
                }
            }
            tx.execute_batch(&format!(
                "DELETE FROM history;
                 DELETE FROM relationships_{collection};
                 DELETE FROM entities_{collection};
                 DELETE FROM memory_entities;
                 DELETE FROM associations;
                 DELETE FROM memories;
                 DELETE FROM procedures;
                 DELETE FROM sources;
                 DELETE FROM sessions;
                 DELETE FROM events;
                 DELETE FROM episodes;
                 DELETE FROM identity;
                 DELETE FROM meditations;
                 DELETE FROM recalls;"
            ))?;
            Ok(())
        })
    }
}

impl Storage {
    /// Delete ALL data for a single user across every table.
    pub(crate) fn delete_user_data(&self, user_id: &str) -> Result<()> {
        let collection = &self.config.collection_name;
        let memory_vectors = self.dialect().memory_vector_table_name();
        let event_vectors = self.dialect().event_vector_table_name();
        self.backend.transaction(|tx| {
            let p = &[SqlParam::Text(user_id.to_string())];

            tx.execute(
                &format!(
                    "DELETE FROM {memory_vectors} WHERE rowid IN (\
                     SELECT rowid FROM {memory_vectors}_vectors WHERE user_id = $1)"
                ),
                p,
            )?;
            tx.execute(
                &format!(
                    "DELETE FROM {event_vectors} WHERE rowid IN (\
                     SELECT rowid FROM {event_vectors}_vectors WHERE user_id = $1)"
                ),
                p,
            )?;

            if tx.table_exists("memories_fts")? {
                tx.execute(
                    "DELETE FROM memories_fts WHERE id IN (\
                     SELECT id FROM memories WHERE user_id = $1)",
                    p,
                )?;
            }
            if tx.table_exists("events_fts")? {
                tx.execute(
                    "DELETE FROM events_fts WHERE event_id IN (\
                     SELECT event_id FROM events WHERE user_id = $1)",
                    p,
                )?;
            }
            if tx.table_exists("episodes_fts")? {
                tx.execute(
                    "DELETE FROM episodes_fts WHERE episode_id IN (\
                     SELECT episode_id FROM episodes WHERE user_id = $1)",
                    p,
                )?;
            }

            // Associations have no user_id. Remove every edge touching any
            // user-owned layer while those source rows still exist.
            tx.execute(
                &format!(
                    "DELETE FROM associations WHERE
                     (from_layer = 'memory' AND from_id IN (SELECT id FROM memories WHERE user_id = $1)) OR
                     (to_layer = 'memory' AND to_id IN (SELECT id FROM memories WHERE user_id = $1)) OR
                     (from_layer = 'event' AND from_id IN (SELECT event_id FROM events WHERE user_id = $1)) OR
                     (to_layer = 'event' AND to_id IN (SELECT event_id FROM events WHERE user_id = $1)) OR
                     (from_layer = 'episode' AND from_id IN (SELECT episode_id FROM episodes WHERE user_id = $1)) OR
                     (to_layer = 'episode' AND to_id IN (SELECT episode_id FROM episodes WHERE user_id = $1)) OR
                     (from_layer = 'identity' AND from_id IN (SELECT trait_id FROM identity WHERE user_id = $1)) OR
                     (to_layer = 'identity' AND to_id IN (SELECT trait_id FROM identity WHERE user_id = $1)) OR
                     (from_layer = 'entity' AND from_id IN (SELECT id FROM entities_{collection} WHERE user_id = $1)) OR
                     (to_layer = 'entity' AND to_id IN (SELECT id FROM entities_{collection} WHERE user_id = $1))"
                ),
                p,
            )?;
            tx.execute("DELETE FROM memory_entities WHERE user_id = $1", p)?;
            tx.execute(
                &format!("DELETE FROM relationships_{collection} WHERE user_id = $1"),
                p,
            )?;
            tx.execute(
                &format!("DELETE FROM entities_{collection} WHERE user_id = $1"),
                p,
            )?;
            for table in [
                "history",
                "memories",
                "events",
                "sessions",
                "episodes",
                "identity",
                "meditations",
                "recalls",
                "procedures",
                "sources",
            ] {
                tx.execute(&format!("DELETE FROM {table} WHERE user_id = $1"), p)?;
            }
            Ok(())
        })
    }
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

/// A memory that shares entities with another memory (for contradiction detection).
#[derive(Debug, Clone)]
pub(crate) struct EntityNeighborRow {
    pub memory_id: String,
    pub content: String,
    pub shared_entities: usize,
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

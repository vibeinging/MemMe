use crate::config::MemoryConfig;
use crate::error::Result;

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
    /// Open a SQLite-backed storage and initialize the schema.
    pub fn open(config: MemoryConfig) -> Result<Self> {
        config.validate()?;
        let conn = backend_sqlite::open_sqlite(&config)?;
        let backend = Backend::sqlite(conn);
        // Initialize schema using the dialect
        let dialect = backend.dialect();
        let schema = Self::sqlite_init_schema(&config, dialect);
        backend.execute_batch(&schema)?;
        // Create vec0 virtual table for vector search
        if let Some(vec0_sql) = dialect.create_vec0_table_sql(config.embedding_dims) {
            backend.execute_batch(&vec0_sql)?;
        }
        let storage = Self { backend, config };
        storage.run_migrations()?;
        Ok(storage)
    }

    /// Run incremental migrations for columns added after initial schema.
    fn run_migrations(&self) -> Result<()> {
        // Migration: add pinned column to memories (added in v0.2)
        let has_pinned = self.backend.query_read(
            "SELECT pinned FROM memories LIMIT 0", &[], |_| Ok(())
        );
        if has_pinned.is_err() {
            let _ = self.backend.execute_batch(
                "ALTER TABLE memories ADD COLUMN pinned INTEGER DEFAULT 0;"
            );
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
    pub(crate) const SCHEMA_VERSION: &'static str = "5";

    // ── Reset ──

    /// Delete ALL data from all tables. This is destructive and cannot be undone.
    pub(crate) fn reset(&self) -> Result<()> {
        self.backend.execute_batch("DELETE FROM history")?;
        let collection = &self.config.collection_name;
        self.backend
            .execute_batch(&format!("DELETE FROM relationships_{collection}"))?;
        self.backend
            .execute_batch(&format!("DELETE FROM entities_{collection}"))?;
        self.backend.execute_batch("DELETE FROM memory_entities")?;
        self.backend.execute_batch("DELETE FROM memories")?;
        self.backend.execute_batch("DELETE FROM procedures")?;
        // Four-layer tables
        let _ = self.backend.execute_batch("DELETE FROM sources");
        let _ = self.backend.execute_batch("DELETE FROM sessions");
        let _ = self.backend.execute_batch("DELETE FROM events");
        let _ = self.backend.execute_batch("DELETE FROM episodes");
        let _ = self.backend.execute_batch("DELETE FROM identity");
        let _ = self.backend.execute_batch("DELETE FROM associations");
        let _ = self.backend.execute_batch("DELETE FROM meditations");
        let _ = self.backend.execute_batch("DELETE FROM recalls");
        Ok(())
    }
}

impl Storage {
    /// Delete ALL data for a single user across every table.
    pub(crate) fn delete_user_data(&self, user_id: &str) -> Result<()> {
        use crate::types::SqlParam;
        let collection = &self.config.collection_name;
        let p = &[SqlParam::Text(user_id.to_string())];
        let _ = self.backend.execute("DELETE FROM vec_memories WHERE memory_id IN (SELECT id FROM memories WHERE user_id = $1)", p);
        let _ = self.backend.execute("DELETE FROM vec_events WHERE event_id IN (SELECT event_id FROM events WHERE user_id = $1)", p);
        let _ = self.backend.execute(
            "DELETE FROM memories_fts WHERE id IN (SELECT id FROM memories WHERE user_id = $1)",
            p,
        );
        let _ = self.backend.execute("DELETE FROM episodes_fts WHERE episode_id IN (SELECT episode_id FROM episodes WHERE user_id = $1)", p);
        self.backend
            .execute("DELETE FROM memory_entities WHERE user_id = $1", p)?;
        self.backend.execute(
            &format!("DELETE FROM relationships_{collection} WHERE user_id = $1"),
            p,
        )?;
        self.backend.execute(
            &format!("DELETE FROM entities_{collection} WHERE user_id = $1"),
            p,
        )?;
        self.backend
            .execute("DELETE FROM history WHERE user_id = $1", p)?;
        self.backend
            .execute("DELETE FROM memories WHERE user_id = $1", p)?;
        self.backend
            .execute("DELETE FROM events WHERE user_id = $1", p)?;
        self.backend
            .execute("DELETE FROM sessions WHERE user_id = $1", p)?;
        self.backend
            .execute("DELETE FROM episodes WHERE user_id = $1", p)?;
        self.backend
            .execute("DELETE FROM identity WHERE user_id = $1", p)?;
        self.backend
            .execute("DELETE FROM meditations WHERE user_id = $1", p)?;
        self.backend
            .execute("DELETE FROM recalls WHERE user_id = $1", p)?;
        self.backend
            .execute("DELETE FROM procedures WHERE user_id = $1", p)?;
        let _ = self
            .backend
            .execute("DELETE FROM sources WHERE user_id = $1", p);
        let _ = self.backend.execute("DELETE FROM associations WHERE from_id IN (SELECT id FROM memories WHERE user_id = $1) OR to_id IN (SELECT id FROM memories WHERE user_id = $1)", p);
        Ok(())
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

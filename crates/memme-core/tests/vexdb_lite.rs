use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use memme_core::{AddOptions, ChatMessage, MemoryConfig, MemoryStore, SearchOptions};
use memme_embeddings::mock::MockEmbedder;
use rusqlite::{params, Connection};
use uuid::Uuid;

const DIMS: usize = 32;
const EXTENSION_ENV: &str = "MEMME_VEXDB_LITE_EXTENSION";
const MISSING_EXTENSION_SUBPROCESS_ENV: &str = "MEMME_TEST_MISSING_VEXDB_EXTENSION";

struct TempDb(PathBuf);

impl TempDb {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "memme-vexdb-lite-{}-{}.db",
            std::process::id(),
            Uuid::new_v4()
        )))
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        for suffix in ["", "-wal", "-shm"] {
            let candidate = PathBuf::from(format!("{}{}", self.0.display(), suffix));
            let _ = std::fs::remove_file(candidate);
        }
    }
}

fn extension_path() -> PathBuf {
    let path = std::env::var_os(EXTENSION_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {EXTENSION_ENV} to run the VexDB-Lite integration tests"));
    assert!(
        path.is_file(),
        "{EXTENSION_ENV} does not point to a file: {}",
        path.display()
    );
    path
}

fn config(db_path: &Path, dims: usize) -> MemoryConfig {
    MemoryConfig {
        db_path: db_path.to_string_lossy().into_owned(),
        embedding_dims: dims,
        enable_graph: false,
        ..Default::default()
    }
}

fn open_store_with_dims(
    db_path: &Path,
    extension_path: &Path,
    dims: usize,
) -> memme_core::Result<MemoryStore> {
    MemoryStore::new_with_vexdb_lite(
        config(db_path, dims),
        Arc::new(MockEmbedder::new(dims)),
        extension_path,
    )
}

fn open_store(db_path: &Path, extension_path: &Path) -> MemoryStore {
    open_store_with_dims(db_path, extension_path, DIMS).unwrap()
}

fn open_env_store(db_path: &Path) -> MemoryStore {
    MemoryStore::new(config(db_path, DIMS), Arc::new(MockEmbedder::new(DIMS))).unwrap()
}

#[test]
fn vexdb_lite_crud_user_filter_and_reopen() {
    let extension_path = extension_path();
    let db = TempDb::new();

    let first_id = {
        let store = open_store(&db.0, &extension_path);
        let first = store
            .add(
                "Alice keeps the blue notebook beside the window",
                AddOptions::new("user-a"),
            )
            .unwrap();
        let other_user = store
            .add(
                "Alice keeps the blue notebook beside the window",
                AddOptions::new("user-b"),
            )
            .unwrap();

        let results = store
            .search(
                "Alice keeps the blue notebook beside the window",
                SearchOptions::new("user-a").limit(5),
            )
            .unwrap();
        assert_eq!(
            results.first().map(|r| r.id.as_str()),
            Some(first.id.as_str())
        );
        assert!(results.iter().all(|r| r.user_id == "user-a"));
        assert!(results.iter().all(|r| r.id != other_user.id));

        store
            .update_trace(
                &first.id,
                "Alice moved the blue notebook onto the kitchen table",
                None,
            )
            .unwrap();
        let updated = store
            .search(
                "Alice moved the blue notebook onto the kitchen table",
                SearchOptions::new("user-a").limit(1),
            )
            .unwrap();
        assert_eq!(
            updated.first().map(|r| r.id.as_str()),
            Some(first.id.as_str())
        );

        store
            .append_events(
                "session-vexdb",
                &[ChatMessage {
                    role: "user".into(),
                    content: "The emergency access phrase is copper meadow".into(),
                    image_url: None,
                    image_type: None,
                    timestamp: None,
                }],
                "event-user",
                None,
            )
            .unwrap();
        let event_results = store
            .search(
                "The emergency access phrase is copper meadow",
                SearchOptions::new("event-user").limit(1),
            )
            .unwrap();
        assert_eq!(
            event_results.first().map(|r| r.content.as_str()),
            Some("The emergency access phrase is copper meadow")
        );

        first.id
    };

    let store = open_store(&db.0, &extension_path);
    let reopened = store
        .search(
            "Alice moved the blue notebook onto the kitchen table",
            SearchOptions::new("user-a").limit(1),
        )
        .unwrap();
    assert_eq!(
        reopened.first().map(|r| r.id.as_str()),
        Some(first_id.as_str())
    );

    store.delete_trace(&first_id).unwrap();
    let after_delete = store
        .search(
            "Alice moved the blue notebook onto the kitchen table",
            SearchOptions::new("user-a").limit(5),
        )
        .unwrap();
    assert!(after_delete.iter().all(|r| r.id != first_id));
}

#[test]
fn default_constructor_uses_vexdb_lite_environment_path() {
    let extension_path = extension_path();
    let db = TempDb::new();
    let content = "Migration keeps the original memory searchable";

    let memory_id = {
        let store = open_store(&db.0, &extension_path);
        store
            .add(content, AddOptions::new("migration-user"))
            .unwrap()
            .id
    };

    let store = open_env_store(&db.0);
    let results = store
        .search(content, SearchOptions::new("migration-user").limit(1))
        .unwrap();
    assert_eq!(
        results.first().map(|r| r.id.as_str()),
        Some(memory_id.as_str())
    );
}

#[test]
fn default_constructor_without_extension_fails() {
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "subprocess_default_constructor_requires_extension",
            "--nocapture",
        ])
        .env_remove(EXTENSION_ENV)
        .env(MISSING_EXTENSION_SUBPROCESS_ENV, "1")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "missing-extension subprocess failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn subprocess_default_constructor_requires_extension() {
    if std::env::var_os(MISSING_EXTENSION_SUBPROCESS_ENV).is_none() {
        return;
    }

    assert!(std::env::var_os(EXTENSION_ENV).is_none());
    let db = TempDb::new();
    let error = match MemoryStore::new(config(&db.0, DIMS), Arc::new(MockEmbedder::new(DIMS))) {
        Ok(_) => panic!("MemoryStore::new must fail when {EXTENSION_ENV} is missing"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains(EXTENSION_ENV),
        "unexpected error: {error}"
    );
}

#[test]
fn reopening_with_different_embedding_dimensions_fails_during_construction() {
    let extension_path = extension_path();
    let db = TempDb::new();

    let store = open_store(&db.0, &extension_path);
    let memory_id = store
        .add(
            "Dimension checks happen before any data can be changed",
            AddOptions::new("dimension-user"),
        )
        .unwrap()
        .id;
    drop(store);

    let requested_dims = DIMS / 2;
    let error = match open_store_with_dims(&db.0, &extension_path, requested_dims) {
        Ok(_) => {
            panic!("opening a {DIMS}-dimension index as {requested_dims} dimensions must fail")
        }
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("dimension")
            && (message.contains("mismatch") || message.contains("different embedding size")),
        "unexpected error: {message}"
    );
    assert!(
        message.contains(&requested_dims.to_string()),
        "unexpected error: {message}"
    );

    let store = open_store(&db.0, &extension_path);
    assert!(store.get_trace(&memory_id).unwrap().is_some());
}

#[test]
fn zero_search_limit_returns_empty_results() {
    let extension_path = extension_path();
    let db = TempDb::new();
    let store = open_store(&db.0, &extension_path);
    store
        .add(
            "A zero result limit must not reach VexDB-Lite with k equals zero",
            AddOptions::new("limit-user"),
        )
        .unwrap();

    let results = store
        .search(
            "A zero result limit must not reach VexDB-Lite with k equals zero",
            SearchOptions::new("limit-user").limit(0),
        )
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn indexed_search_preserves_agent_run_and_app_scope() {
    let extension_path = extension_path();
    let db = TempDb::new();
    let store = open_store(&db.0, &extension_path);
    let user_id = "scope-user";

    let target = store
        .add(
            "Cobalt telescope target memory",
            AddOptions::new(user_id)
                .agent_id("agent-a")
                .run_id("run-a")
                .app_id("app-a"),
        )
        .unwrap();
    let wrong_agent = store
        .add(
            "Cobalt telescope memory from another agent",
            AddOptions::new(user_id)
                .agent_id("agent-b")
                .run_id("run-a")
                .app_id("app-a"),
        )
        .unwrap();
    let wrong_run = store
        .add(
            "Cobalt telescope memory from another run",
            AddOptions::new(user_id)
                .agent_id("agent-a")
                .run_id("run-b")
                .app_id("app-a"),
        )
        .unwrap();
    let wrong_app = store
        .add(
            "Cobalt telescope memory from another app",
            AddOptions::new(user_id)
                .agent_id("agent-a")
                .run_id("run-a")
                .app_id("app-b"),
        )
        .unwrap();

    let by_agent = store
        .search(
            "Cobalt telescope",
            SearchOptions::new(user_id).agent_id("agent-a").limit(10),
        )
        .unwrap();
    assert!(by_agent
        .iter()
        .all(|result| result.agent_id.as_deref() == Some("agent-a")));
    assert!(by_agent.iter().all(|result| result.id != wrong_agent.id));

    let by_run = store
        .search(
            "Cobalt telescope",
            SearchOptions::new(user_id).run_id("run-a").limit(10),
        )
        .unwrap();
    assert!(by_run
        .iter()
        .all(|result| result.run_id.as_deref() == Some("run-a")));
    assert!(by_run.iter().all(|result| result.id != wrong_run.id));

    let by_app = store
        .search(
            "Cobalt telescope",
            SearchOptions::new(user_id).app_id("app-a").limit(10),
        )
        .unwrap();
    assert!(by_app
        .iter()
        .all(|result| result.app_id.as_deref() == Some("app-a")));
    assert!(by_app.iter().all(|result| result.id != wrong_app.id));

    let fully_scoped = store
        .search(
            "Cobalt telescope",
            SearchOptions::new(user_id)
                .agent_id("agent-a")
                .run_id("run-a")
                .app_id("app-a")
                .limit(10),
        )
        .unwrap();
    assert_eq!(
        fully_scoped
            .iter()
            .map(|result| result.id.as_str())
            .collect::<Vec<_>>(),
        vec![target.id.as_str()]
    );
}

#[test]
fn vex_shadow_metadata_queries_use_indexes() {
    let extension_path = extension_path();
    let db = TempDb::new();
    let store = open_store(&db.0, &extension_path);
    let memory = store
        .add(
            "Indexed metadata lookup",
            AddOptions::new("indexed-user").agent_id("indexed-agent"),
        )
        .unwrap();
    drop(store);

    let conn = Connection::open(&db.0).unwrap();
    let point_plan: String = conn
        .query_row(
            "EXPLAIN QUERY PLAN SELECT rowid FROM vex_memories_vectors WHERE memory_id = ?1",
            params![memory.id],
            |row| row.get(3),
        )
        .unwrap();
    assert!(
        point_plan.contains("idx_vex_memories_memory_id"),
        "unexpected point lookup plan: {point_plan}"
    );

    let scope_plan: String = conn
        .query_row(
            "EXPLAIN QUERY PLAN SELECT rowid FROM vex_memories_vectors \
             WHERE user_id = ?1 AND agent_id = ?2",
            params!["indexed-user", "indexed-agent"],
            |row| row.get(3),
        )
        .unwrap();
    assert!(
        scope_plan.contains("idx_vex_memories_user_agent"),
        "unexpected scope lookup plan: {scope_plan}"
    );
}

#[test]
fn failed_vex_shadow_write_rolls_back_memory_update() {
    let extension_path = extension_path();
    let db = TempDb::new();
    let original_content = "Original content survives a Vex shadow write failure";
    let updated_content = "This update must be rolled back completely";

    let memory_id = {
        let store = open_store(&db.0, &extension_path);
        store
            .add(original_content, AddOptions::new("rollback-user"))
            .unwrap()
            .id
    };

    let conn = Connection::open(&db.0).unwrap();
    conn.execute_batch(
        "CREATE TRIGGER fail_vex_shadow_insert
         BEFORE INSERT ON vex_memories_vectors
         BEGIN
             SELECT RAISE(ABORT, 'injected vex shadow failure');
         END;",
    )
    .unwrap();
    drop(conn);

    let store = open_store(&db.0, &extension_path);
    let error = store
        .update_trace(&memory_id, updated_content, None)
        .expect_err("the injected Vex shadow failure must abort the update");
    assert!(
        error.to_string().contains("injected vex shadow failure"),
        "unexpected error: {error}"
    );

    let memory = store.get_trace(&memory_id).unwrap().unwrap();
    assert_eq!(memory.content, original_content);
    let results = store
        .search(
            original_content,
            SearchOptions::new("rollback-user").limit(5),
        )
        .unwrap();
    assert!(results.iter().any(|result| result.id == memory_id));
    drop(store);

    let conn = Connection::open(&db.0).unwrap();
    let indexed_rows: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM vex_memories_vectors WHERE memory_id = ?1",
            params![memory_id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(indexed_rows, 1);
}

#[test]
fn delete_all_traces_removes_vex_results_before_and_after_reopen() {
    let extension_path = extension_path();
    let db = TempDb::new();
    let user_id = "delete-all-user";

    let memory_ids = {
        let store = open_store(&db.0, &extension_path);
        let first = store
            .add(
                "Delete-all removes the first indexed memory",
                AddOptions::new(user_id),
            )
            .unwrap();
        let second = store
            .add(
                "Delete-all removes the second indexed memory",
                AddOptions::new(user_id),
            )
            .unwrap();

        assert_eq!(
            store.delete_all_traces(user_id, None, None, None).unwrap(),
            2
        );
        let results = store
            .search("Delete-all removes", SearchOptions::new(user_id).limit(10))
            .unwrap();
        assert!(results.is_empty());
        vec![first.id, second.id]
    };

    let store = open_store(&db.0, &extension_path);
    let results = store
        .search("Delete-all removes", SearchOptions::new(user_id).limit(10))
        .unwrap();
    assert!(results.is_empty());
    for memory_id in memory_ids {
        assert!(store.get_trace(&memory_id).unwrap().is_none());
    }
}

#[test]
fn legacy_vec0_shadow_data_is_retired_after_vex_rebuild() {
    let extension_path = extension_path();
    let db = TempDb::new();
    let user_id = "legacy-user";
    let memory_content = "Legacy memory remains searchable after the Vex rebuild";
    let event_content = "Legacy event remains searchable after the Vex rebuild";

    let memory_id = {
        let store = open_store(&db.0, &extension_path);
        let memory = store.add(memory_content, AddOptions::new(user_id)).unwrap();
        store
            .append_events(
                "legacy-session",
                &[ChatMessage {
                    role: "user".into(),
                    content: event_content.into(),
                    image_url: None,
                    image_type: None,
                    timestamp: None,
                }],
                user_id,
                None,
            )
            .unwrap();
        memory.id
    };

    let conn = Connection::open(&db.0).unwrap();
    conn.execute_batch(
        "UPDATE memme_config SET value = 'sqlite-vec' WHERE key = 'vector_backend';
         UPDATE memme_config SET value = '1' WHERE key = 'vector_index_version';
         CREATE TABLE vec_memories_info(key TEXT, value TEXT);
         CREATE TABLE vec_memories_chunks(chunk_id INTEGER PRIMARY KEY, payload BLOB);
         CREATE TABLE vec_events_info(key TEXT, value TEXT);
         CREATE TABLE vec_events_chunks(chunk_id INTEGER PRIMARY KEY, payload BLOB);
         INSERT INTO vec_memories_info VALUES ('version', 'legacy');
         INSERT INTO vec_memories_chunks VALUES (1, X'0102');
         INSERT INTO vec_events_info VALUES ('version', 'legacy');
         INSERT INTO vec_events_chunks VALUES (1, X'0304');
         PRAGMA writable_schema = ON;",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO sqlite_master(type, name, tbl_name, rootpage, sql)
         VALUES ('table', 'vec_memories', 'vec_memories', 0, ?1)",
        params![format!(
            "CREATE VIRTUAL TABLE vec_memories USING vec0(memory_id TEXT PRIMARY KEY, embedding float[{DIMS}], user_id TEXT partition_key)"
        )],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO sqlite_master(type, name, tbl_name, rootpage, sql)
         VALUES ('table', 'vec_events', 'vec_events', 0, ?1)",
        params![format!(
            "CREATE VIRTUAL TABLE vec_events USING vec0(event_id TEXT PRIMARY KEY, content_vec float[{DIMS}], user_id TEXT partition_key)"
        )],
    )
    .unwrap();
    conn.execute_batch("PRAGMA writable_schema = OFF;").unwrap();
    let schema_version: i64 = conn
        .query_row("PRAGMA schema_version", [], |row| row.get(0))
        .unwrap();
    conn.pragma_update(None, "schema_version", schema_version + 1)
        .unwrap();
    drop(conn);

    let store = open_store(&db.0, &extension_path);
    let memory_results = store
        .search(memory_content, SearchOptions::new(user_id).limit(5))
        .unwrap();
    assert!(memory_results.iter().any(|result| result.id == memory_id));

    let event_results = store
        .search(event_content, SearchOptions::new(user_id).limit(5))
        .unwrap();
    assert!(event_results
        .iter()
        .any(|result| result.content == event_content));
    drop(store);

    let conn = Connection::open(&db.0).unwrap();
    let legacy_roots: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table' AND name IN ('vec_memories', 'vec_events')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        legacy_roots, 2,
        "legacy virtual-table schema rows stay untouched"
    );

    let shadow_tables: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table'
               AND (name GLOB 'vec_memories_*' OR name GLOB 'vec_events_*')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        shadow_tables, 0,
        "legacy sqlite-vec shadow data must be removed"
    );

    let retired: String = conn
        .query_row(
            "SELECT value FROM memme_config WHERE key = 'legacy_vec0_retired'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(retired, "1");
}

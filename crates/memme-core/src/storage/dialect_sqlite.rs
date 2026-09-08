//! SQLite-specific SQL dialect implementation.
//!
//! Vectors are stored as BLOBs (raw f32 bytes). Cosine distance is computed
//! via a registered Rust scalar function. FTS uses SQLite's built-in FTS5.

// SQLite dialect — compiled only with the "sqlite" feature flag.

use crate::error::Result;

use super::dialect::SqlDialect;

/// SQLite SQL dialect backed by VexDB-Lite `GRAPH_INDEX` tables.
pub(crate) struct SqliteDialect;

impl SqlDialect for SqliteDialect {
    // ── Schema types ──

    fn embedding_column_type(&self, _dims: usize) -> String {
        "BLOB".to_string()
    }

    fn string_array_column_type(&self) -> &str {
        "TEXT" // JSON array: '["cat1","cat2"]'
    }

    // ── Embedding / vector ──

    fn format_embedding_literal(&self, embedding: &[f32], _dims: usize) -> Result<String> {
        super::dialect::validate_embedding(embedding)?;
        // SQLite: use X'hex' blob literal for embeddings
        let bytes: &[u8] = bytemuck::cast_slice(embedding);
        let mut s = String::with_capacity(2 + bytes.len() * 2 + 1);
        s.push_str("X'");
        for b in bytes {
            use std::fmt::Write;
            write!(s, "{b:02x}").unwrap();
        }
        s.push('\'');
        Ok(s)
    }

    fn cosine_distance_expr(&self, column: &str, embedding_literal: &str) -> String {
        format!("vexdb_cosine_distance({column}, {embedding_literal})")
    }

    // ── HNSW index ──

    fn create_hnsw_index_sql(
        &self,
        _index_name: &str,
        _table: &str,
        _column: &str,
    ) -> Option<String> {
        // SQLite has no built-in HNSW; brute-force scan via cosine_distance_expr
        None
    }

    // ── Full-text search ──

    fn load_fts_extension_sql(&self) -> Vec<&str> {
        // FTS5 is compiled into rusqlite with the "vtab" feature — no loading needed
        vec![]
    }

    fn create_fts_index_sql(&self, table: &str, id_col: &str, content_cols: &[&str]) -> String {
        let all_cols: Vec<&str> = std::iter::once(id_col)
            .chain(content_cols.iter().copied())
            .collect();
        let cols_def = all_cols.join(", ");
        let cols_select = all_cols
            .iter()
            .map(|c| format!("{table}.{c}"))
            .collect::<Vec<_>>()
            .join(", ");
        // Standalone FTS5 table (not external content) — rebuilt from source table
        format!(
            "DROP TABLE IF EXISTS {table}_fts;\n\
             CREATE VIRTUAL TABLE {table}_fts USING fts5({cols_def});\n\
             INSERT INTO {table}_fts({cols_def}) SELECT {cols_select} FROM {table};"
        )
    }

    fn fts_match_score_expr(&self, table: &str, _id_col: &str, _query_param: &str) -> String {
        // FTS5 uses the `rank` hidden column for BM25 scoring
        // The query joins with the FTS table: WHERE {table}_fts MATCH {query_param}
        format!("{table}_fts.rank")
    }

    // ── JSON functions ──

    fn json_extract_string(&self, column: &str, json_path: &str) -> String {
        format!("json_extract({column}, '$.{json_path}')")
    }

    fn load_json_extension_sql(&self) -> Option<&str> {
        // SQLite has built-in JSON1 — no extension loading needed
        None
    }

    // ── Array functions ──

    fn list_contains_expr(&self, column: &str, param: &str) -> String {
        // column stores JSON array like '["cat1","cat2"]'
        format!("EXISTS (SELECT 1 FROM json_each({column}) WHERE value = {param})")
    }

    fn array_icontains_expr(&self, column: &str, param: &str) -> String {
        format!(
            "EXISTS (SELECT 1 FROM json_each({column}) WHERE LOWER(value) LIKE '%' || LOWER({param}) || '%')"
        )
    }

    fn format_categories_literal(&self, categories: &[String]) -> Result<String> {
        super::dialect::validate_categories(categories)?;
        // Store as JSON array string
        let json = serde_json::to_string(categories).map_err(|e| {
            crate::error::MemoryError::Config(format!("Failed to serialize categories: {e}"))
        })?;
        Ok(format!("'{json}'"))
    }

    // ── Date/time functions ──

    fn current_timestamp_expr(&self) -> &str {
        "datetime('now')"
    }

    fn epoch_seconds_expr(&self, column: &str) -> String {
        format!("unixepoch({column})")
    }

    fn date_trunc_expr(&self, granularity: &str, column: &str) -> String {
        match granularity {
            "day" => format!("date({column})"),
            "week" => format!("date({column}, 'weekday 0', '-6 days')"),
            "month" => format!("strftime('%Y-%m-01', {column})"),
            _ => format!("date({column})"), // fallback to day
        }
    }

    // ── Math functions ──

    fn greatest_expr(&self, a: &str, b: &str) -> String {
        format!("MAX({a}, {b})")
    }

    fn least_expr(&self, a: &str, b: &str) -> String {
        format!("MIN({a}, {b})")
    }

    fn power_expr(&self, base: &str, exponent: &str) -> String {
        format!("pow({base}, {exponent})")
    }

    // ── Vector index ──

    fn has_vector_index(&self) -> bool {
        true
    }

    fn vector_backend_name(&self) -> &'static str {
        "vexdb-lite"
    }

    fn memory_vector_table_name(&self) -> &'static str {
        "vex_memories"
    }

    fn event_vector_table_name(&self) -> &'static str {
        "vex_events"
    }

    fn create_vector_index_sql(&self, dims: usize) -> Option<String> {
        Some(format!(
            r#"CREATE VIRTUAL TABLE IF NOT EXISTS vex_memories USING GRAPH_INDEX(
                embedding FLOAT[{dims}],
                memory_id TEXT,
                user_id TEXT,
                agent_id TEXT,
                run_id TEXT,
                app_id TEXT,
                metric=cosine
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS vex_events USING GRAPH_INDEX(
                content_vec FLOAT[{dims}],
                event_id TEXT,
                user_id TEXT,
                agent_id TEXT,
                metric=cosine
            )"#
        ))
    }

    fn rebuild_vector_index_sql(&self) -> Option<String> {
        let memories = self.memory_vector_table_name();
        let events = self.event_vector_table_name();
        Some(format!(
            r#"DELETE FROM {memories};
               DELETE FROM {events};
               INSERT INTO {memories}(embedding, memory_id, user_id, agent_id, run_id, app_id)
                   SELECT embedding, id, user_id, agent_id, run_id, app_id
                   FROM memories
                   WHERE embedding IS NOT NULL AND length(embedding) > 0;
               INSERT INTO {events}(content_vec, event_id, user_id, agent_id)
                   SELECT content_vec, event_id, user_id, agent_id
                   FROM events
                   WHERE content_vec IS NOT NULL AND length(content_vec) > 0;"#
        ))
    }

    fn vector_metadata_indexes_sql(&self) -> Option<String> {
        Some(
            r#"CREATE UNIQUE INDEX IF NOT EXISTS idx_vex_memories_memory_id
                   ON vex_memories_vectors(memory_id);
               CREATE INDEX IF NOT EXISTS idx_vex_memories_user_id
                   ON vex_memories_vectors(user_id);
               CREATE INDEX IF NOT EXISTS idx_vex_memories_user_agent
                   ON vex_memories_vectors(user_id, agent_id);
               CREATE INDEX IF NOT EXISTS idx_vex_memories_user_run
                   ON vex_memories_vectors(user_id, run_id);
               CREATE INDEX IF NOT EXISTS idx_vex_memories_user_app
                   ON vex_memories_vectors(user_id, app_id);
               CREATE UNIQUE INDEX IF NOT EXISTS idx_vex_events_event_id
                   ON vex_events_vectors(event_id);
               CREATE INDEX IF NOT EXISTS idx_vex_events_user_id
                   ON vex_events_vectors(user_id);
               CREATE INDEX IF NOT EXISTS idx_vex_events_user_agent
                   ON vex_events_vectors(user_id, agent_id);"#
                .to_string(),
        )
    }

    fn vector_insert_sql(&self, id_param: &str, embedding_literal: &str) -> Option<String> {
        Some(format!(
            "INSERT INTO vex_memories(embedding, memory_id, user_id, agent_id, run_id, app_id) \
             VALUES ({embedding_literal}, {id_param}, $2, $3, $4, $5)"
        ))
    }

    fn vector_delete_sql(&self) -> Option<&str> {
        Some(
            "DELETE FROM vex_memories WHERE rowid IN (\
             SELECT rowid FROM vex_memories_vectors WHERE memory_id = $1)",
        )
    }

    fn vector_knn_sql(
        &self,
        embedding_param: &str,
        user_id_param: &str,
        limit: usize,
    ) -> Option<String> {
        // VexDB-Lite KNN returns (memory_id, distance), nearest first.
        Some(format!(
            r#"SELECT memory_id, distance
               FROM vex_memories
               WHERE embedding MATCH {embedding_param}
                 AND k = {limit}
                 AND user_id = {user_id_param}"#
        ))
    }

    fn vector_event_insert_sql(&self, id_param: &str, embedding_literal: &str) -> Option<String> {
        Some(format!(
            "INSERT INTO vex_events(content_vec, event_id, user_id, agent_id) \
             VALUES ({embedding_literal}, {id_param}, $2, $3)"
        ))
    }

    fn vector_event_delete_sql(&self) -> Option<&str> {
        Some(
            "DELETE FROM vex_events WHERE rowid IN (\
             SELECT rowid FROM vex_events_vectors WHERE event_id = $1)",
        )
    }

    fn vector_event_knn_sql(
        &self,
        embedding_param: &str,
        user_id_param: &str,
        limit: usize,
    ) -> Option<String> {
        Some(format!(
            r#"SELECT event_id, distance
               FROM vex_events
               WHERE content_vec MATCH {embedding_param}
                 AND k = {limit}
                 AND user_id = {user_id_param}"#
        ))
    }

    // ── String functions ──

    fn left_expr(&self, str_expr: &str, n: &str) -> String {
        format!("SUBSTR({str_expr}, 1, {n})")
    }

    // ── WAL / maintenance ──

    fn checkpoint_sql(&self) -> &str {
        "PRAGMA wal_checkpoint(FULL)"
    }
}

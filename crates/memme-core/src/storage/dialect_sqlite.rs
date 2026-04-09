//! SQLite-specific SQL dialect implementation.
//!
//! Vectors are stored as BLOBs (raw f32 bytes). Cosine distance is computed
//! via a registered Rust scalar function. FTS uses SQLite's built-in FTS5.

// SQLite dialect — compiled only with the "sqlite" feature flag.

use crate::error::Result;

use super::dialect::SqlDialect;

/// SQLite SQL dialect.
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
        // Uses sqlite-vec's built-in vec_distance_cosine function
        format!("vec_distance_cosine({column}, {embedding_literal})")
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
        let all_cols: Vec<&str> = std::iter::once(id_col).chain(content_cols.iter().copied()).collect();
        let cols_def = all_cols.join(", ");
        let cols_select = all_cols.iter().map(|c| format!("{table}.{c}")).collect::<Vec<_>>().join(", ");
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
        format!(
            "EXISTS (SELECT 1 FROM json_each({column}) WHERE value = {param})"
        )
    }

    fn array_icontains_expr(&self, column: &str, param: &str) -> String {
        format!(
            "EXISTS (SELECT 1 FROM json_each({column}) WHERE LOWER(value) LIKE '%' || LOWER({param}) || '%')"
        )
    }

    fn format_categories_literal(&self, categories: &[String]) -> Result<String> {
        super::dialect::validate_categories(categories)?;
        // Store as JSON array string
        let json = serde_json::to_string(categories)
            .map_err(|e| crate::error::MemoryError::Config(format!("Failed to serialize categories: {e}")))?;
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

    // ── Vector index (vec0) ──

    fn has_vec0_table(&self) -> bool {
        true
    }

    fn create_vec0_table_sql(&self, dims: usize) -> Option<String> {
        Some(format!(
            r#"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                memory_id TEXT PRIMARY KEY,
                embedding float[{dims}] distance_metric=cosine,
                user_id TEXT partition_key
            )"#
        ))
    }

    fn vec0_insert_sql(&self, id_param: &str, embedding_literal: &str) -> Option<String> {
        // $2 is user_id, passed directly by the caller
        Some(format!(
            "INSERT OR REPLACE INTO vec_memories(memory_id, embedding, user_id) \
             VALUES ({id_param}, {embedding_literal}, $2)"
        ))
    }

    fn vec0_delete_sql(&self) -> Option<&str> {
        Some("DELETE FROM vec_memories WHERE memory_id = $1")
    }

    fn vec0_knn_sql(
        &self,
        embedding_param: &str,
        user_id_param: &str,
        limit: usize,
    ) -> Option<String> {
        // vec0 KNN query returns (memory_id, distance) ordered by distance
        Some(format!(
            r#"SELECT memory_id, distance
               FROM vec_memories
               WHERE embedding MATCH {embedding_param}
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

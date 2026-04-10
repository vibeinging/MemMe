//! SQL dialect abstraction for database-agnostic query generation.
//!
//! The database backend implements [`SqlDialect`] to provide the correct SQL
//! syntax for its engine. Storage code calls dialect methods instead of
//! hardcoding backend-specific SQL.

use crate::error::{MemoryError, Result};

/// Abstracts database-specific SQL syntax differences.
///
/// Storage code calls these methods to generate SQL fragments that are then
/// interpolated into queries. Each method returns a SQL string fragment —
/// not a full query — that can be embedded in `format!()` expressions.
#[allow(dead_code)] // methods called via &dyn SqlDialect; some reserved for future backends
pub(crate) trait SqlDialect: Send + Sync {
    // ── Schema types ──

    /// Column type for storing embedding vectors of the given dimension.
    /// SQLite: `BLOB`.
    fn embedding_column_type(&self, dims: usize) -> String;

    /// Column type for storing a list of strings.
    /// SQLite: `TEXT` (JSON array).
    fn string_array_column_type(&self) -> &str;

    // ── Embedding / vector ──

    /// Format an embedding vector as a SQL literal.
    /// SQLite: a BLOB hex literal.
    fn format_embedding_literal(&self, embedding: &[f32], dims: usize) -> Result<String>;

    /// SQL expression for cosine distance between a column and an embedding literal.
    /// SQLite: `vec_distance_cosine({col}, {literal})`.
    fn cosine_distance_expr(&self, column: &str, embedding_literal: &str) -> String;

    // ── HNSW index ──

    /// SQL to create an HNSW vector index, or `None` if not supported.
    fn create_hnsw_index_sql(&self, index_name: &str, table: &str, column: &str) -> Option<String>;

    // ── Full-text search ──

    /// SQL statements to install/load the FTS extension (may be empty).
    fn load_fts_extension_sql(&self) -> Vec<&str>;

    /// SQL to create a full-text search index on the given table and columns.
    fn create_fts_index_sql(&self, table: &str, id_col: &str, content_cols: &[&str]) -> String;

    /// SQL expression that produces a BM25 relevance score for FTS matching.
    fn fts_match_score_expr(&self, table: &str, id_col: &str, query_param: &str) -> String;

    // ── JSON functions ──

    /// SQL expression to extract a string from a JSON column.
    fn json_extract_string(&self, column: &str, json_path: &str) -> String;

    /// SQL statements to load JSON extension (may be empty).
    fn load_json_extension_sql(&self) -> Option<&str>;

    // ── Array functions ──

    /// SQL expression to check if an array column contains a value.
    fn list_contains_expr(&self, column: &str, param: &str) -> String;

    /// SQL expression for case-insensitive substring match on array elements.
    fn array_icontains_expr(&self, column: &str, param: &str) -> String;

    /// Format categories as a SQL literal.
    fn format_categories_literal(&self, categories: &[String]) -> Result<String>;

    // ── Date/time functions ──

    /// SQL expression for "current timestamp".
    fn current_timestamp_expr(&self) -> &str;

    /// SQL expression to extract epoch seconds from a timestamp column.
    fn epoch_seconds_expr(&self, column: &str) -> String;

    /// SQL expression for date truncation.
    fn date_trunc_expr(&self, granularity: &str, column: &str) -> String;

    // ── Math functions ──

    /// SQL expression for `MAX(a, b)` (max of two values).
    fn greatest_expr(&self, a: &str, b: &str) -> String;

    /// SQL expression for `MIN(a, b)` (min of two values).
    fn least_expr(&self, a: &str, b: &str) -> String;

    /// SQL expression for `pow(base, exp)`.
    fn power_expr(&self, base: &str, exponent: &str) -> String;

    // ── Vector index (vec0) ──

    /// Whether this backend uses a vec0 virtual table for vector search.
    /// If true, `vector_search_sql()` generates a MATCH-based KNN query
    /// instead of a brute-force ORDER BY distance query.
    fn has_vec0_table(&self) -> bool {
        false
    }

    /// SQL to create a vec0 virtual table for the memories table.
    /// Returns None if vec0 is not supported.
    fn create_vec0_table_sql(&self, _dims: usize) -> Option<String> {
        None
    }

    /// SQL to insert a row into the vec0 table after inserting into memories.
    /// `id_param` is the memory ID placeholder (e.g. `$1`), `$2` is user_id.
    /// `embedding_literal` is the formatted embedding literal.
    fn vec0_insert_sql(&self, _id_param: &str, _embedding_literal: &str) -> Option<String> {
        None
    }

    /// SQL to delete a row from the vec0 table.
    fn vec0_delete_sql(&self) -> Option<&str> {
        None
    }

    /// Generate a vec0 MATCH-based KNN query that returns (id, distance) pairs.
    /// `embedding_param` is the parameter placeholder for the query embedding.
    /// `user_id_param` is the parameter placeholder for user_id (partition key).
    fn vec0_knn_sql(
        &self,
        _embedding_param: &str,
        _user_id_param: &str,
        _limit: usize,
    ) -> Option<String> {
        None
    }

    /// SQL to insert a row into the vec_events table.
    fn vec0_event_insert_sql(&self, _id_param: &str, _embedding_literal: &str) -> Option<String> {
        None
    }

    /// SQL to delete a row from the vec_events table.
    fn vec0_event_delete_sql(&self) -> Option<&str> {
        None
    }

    /// Generate a vec0 MATCH-based KNN query on vec_events.
    fn vec0_event_knn_sql(
        &self,
        _embedding_param: &str,
        _user_id_param: &str,
        _limit: usize,
    ) -> Option<String> {
        None
    }

    // ── String functions ──

    /// SQL expression for LEFT(str, n) — take first n characters.
    fn left_expr(&self, str_expr: &str, n: &str) -> String;

    // ── WAL / maintenance ──

    /// SQL command to checkpoint the WAL.
    fn checkpoint_sql(&self) -> &str;
}

/// Validate that an embedding contains no NaN or Infinity values.
pub(crate) fn validate_embedding(embedding: &[f32]) -> Result<()> {
    if embedding.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(MemoryError::Config(
            "embedding contains NaN or Infinity".into(),
        ));
    }
    Ok(())
}

/// Validate category names (alphanumeric, underscore, hyphen, space only).
pub(crate) fn validate_categories(categories: &[String]) -> Result<()> {
    for cat in categories {
        if !cat
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == ' ')
        {
            return Err(MemoryError::Config(format!(
                "Invalid category name '{}': only alphanumeric, underscore, hyphen, and space allowed", cat
            )));
        }
        if cat.is_empty() {
            return Err(MemoryError::Config("Category name cannot be empty".into()));
        }
    }
    Ok(())
}

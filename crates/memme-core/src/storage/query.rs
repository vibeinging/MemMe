use crate::error::Result;
use crate::types::{FilterExpression, SqlParam};

use super::backend::RowAccess;
use super::{MemoryRow, Storage};

// ── Unified memory column definitions ──

/// Base columns for MemoryRow (without score). Used to build SELECT clauses.
const MEMORY_COLS_BASE: &str =
    "id, content, user_id, created_at, updated_at, metadata";

const MEMORY_COLS_AFTER_SCORE: &str =
    "importance, access_count, agent_id, app_id, run_id, \
     immutable, expiration_date, categories, memory_type, \
     stability, privacy, event_time, episode_id, session_id, resolution";

/// Generate SELECT columns for a memories query.
/// - `score_expr`: SQL expression for score/distance, or `None` for `NULL AS score`
/// - `prefix`: table alias prefix (e.g. `"m."` for JOINs), or `""` for bare columns
pub(crate) fn memory_select_cols(score_expr: Option<&str>, prefix: &str) -> String {
    let score = score_expr.unwrap_or("NULL");
    let base = MEMORY_COLS_BASE
        .split(", ")
        .map(|c| format!("{prefix}{c}"))
        .collect::<Vec<_>>()
        .join(", ");
    let after = MEMORY_COLS_AFTER_SCORE
        .split(", ")
        .map(|c| format!("{prefix}{c}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{base}, {score} AS score, {after}")
}

/// Map a database row to MemoryRow. Unified mapper for all memory queries.
/// Expects 22 columns in the order produced by `memory_select_cols()`:
///   id(0), content(1), user_id(2), created_at(3), updated_at(4), metadata(5),
///   score(6), importance(7), access_count(8), agent_id(9), app_id(10),
///   run_id(11), immutable(12), expiration_date(13), categories(14),
///   memory_type(15), stability(16), privacy(17), event_time(18),
///   episode_id(19), session_id(20), resolution(21)
pub(crate) fn map_memory_row(row: &dyn RowAccess) -> Result<MemoryRow> {
    Ok(MemoryRow {
        id: row.get_string(0)?,
        content: row.get_string(1)?,
        user_id: row.get_string(2)?,
        created_at: row.get_string(3)?,
        updated_at: row.get_string(4)?,
        metadata: row.get_opt_string(5)?,
        score: row.get_opt_f64(6)?.map(|v| v as f32),
        importance: row.get_opt_f64(7)?.map(|v| v as f32),
        access_count: row.get_opt_i64(8)?.map(|v| v as u32),
        agent_id: row.get_opt_string(9)?,
        app_id: row.get_opt_string(10)?,
        run_id: row.get_opt_string(11)?,
        immutable: row.get_opt_bool(12)?.unwrap_or(false),
        expiration_date: row.get_opt_string(13)?,
        categories: row.get_opt_string(14)?,
        memory_type: row.get_opt_string(15)?,
        stability: row.get_opt_f64(16)?.map(|v| v as f32),
        privacy: row.get_opt_string(17)?,
        event_time: row.get_opt_string(18)?,
        episode_id: row.get_opt_string(19)?,
        session_id: row.get_opt_string(20)?,
        resolution: row.get_opt_string(21)?,
    })
}

impl Storage {
    // ── List ──

    pub(crate) fn list_memories(
        &self,
        user_id: &str,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        app_id: Option<&str>,
        filter: Option<&FilterExpression>,
        limit: usize,
    ) -> Result<Vec<MemoryRow>> {
        // We use dynamic params via SqlParam for flexibility
        let mut conditions = vec!["user_id = $1".to_string()];
        let mut dynamic_params: Vec<SqlParam> =
            vec![SqlParam::Text(user_id.to_string())];
        let mut param_idx: usize = 1;

        if let Some(aid) = agent_id {
            param_idx += 1;
            conditions.push(format!("agent_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(aid.to_string()));
        }

        if let Some(rid) = run_id {
            param_idx += 1;
            conditions.push(format!("run_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(rid.to_string()));
        }

        if let Some(appid) = app_id {
            param_idx += 1;
            conditions.push(format!("app_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(appid.to_string()));
        }

        if let Some(f) = filter {
            let mut offset = param_idx;
            let (sql_fragment, filter_params) = f.to_sql(&mut offset);
            conditions.push(format!("({})", sql_fragment));
            dynamic_params.extend(filter_params);
            param_idx = offset;
        }
        let _ = param_idx; // suppress unused warning

        let where_clause = conditions.join(" AND ");
        let cols = memory_select_cols(None, "");
        let sql = format!(
            "SELECT {cols} FROM memories WHERE {where_clause} ORDER BY updated_at DESC LIMIT {limit}"
        );

        let rows = self.backend.query_read(&sql, &dynamic_params, map_memory_row)?;

        Ok(rows)
    }

    // ── Vector search (cosine distance) ──

    /// Search memories by cosine distance to the given embedding.
    /// Uses HNSW to filter by user_id when available,
    /// falls back to a brute-force filter if the index is absent.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn vector_search(
        &self,
        embedding: &[f32],
        user_id: &str,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        app_id: Option<&str>,
        filter: Option<&FilterExpression>,
        limit: usize,
    ) -> Result<Vec<MemoryRow>> {
        let d = self.dialect();
        let emb_literal = d.format_embedding_literal(embedding, self.config.embedding_dims)?;
        let has_extra_filters = agent_id.is_some() || run_id.is_some() || app_id.is_some() || filter.is_some();

        // Use vec0 MATCH for simple user_id-only queries (fast KNN path).
        // Fall back to brute-force scan when extra filters are present,
        // since vec0 doesn't support arbitrary WHERE clauses.
        if d.has_vec0_table() && !has_extra_filters {
            return self.vector_search_vec0(embedding, user_id, limit);
        }

        let mut conditions = vec!["user_id = $1".to_string()];
        let mut dynamic_params: Vec<SqlParam> =
            vec![SqlParam::Text(user_id.to_string())];
        let mut param_idx: usize = 1;

        if let Some(aid) = agent_id {
            param_idx += 1;
            conditions.push(format!("agent_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(aid.to_string()));
        }

        if let Some(rid) = run_id {
            param_idx += 1;
            conditions.push(format!("run_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(rid.to_string()));
        }

        if let Some(appid) = app_id {
            param_idx += 1;
            conditions.push(format!("app_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(appid.to_string()));
        }

        if let Some(f) = filter {
            let mut offset = param_idx;
            let (sql_fragment, filter_params) = f.to_sql(&mut offset);
            conditions.push(format!("({})", sql_fragment));
            dynamic_params.extend(filter_params);
            param_idx = offset;
        }
        let _ = param_idx;

        let where_clause = conditions.join(" AND ");
        let distance_expr = d.cosine_distance_expr("embedding", &emb_literal);
        let cols = memory_select_cols(Some(&distance_expr), "");
        let sql = format!(
            "SELECT {cols} FROM memories WHERE {where_clause} ORDER BY score ASC LIMIT {limit}"
        );

        let rows = self.backend.query_read(&sql, &dynamic_params, map_memory_row)?;

        Ok(rows)
    }

    /// vec0 MATCH-based vector search (SQLite with sqlite-vec).
    /// Uses the vec0 virtual table for KNN, then JOINs back to memories for full data.
    fn vector_search_vec0(
        &self,
        embedding: &[f32],
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<MemoryRow>> {
        let d = self.dialect();
        let emb_literal = d.format_embedding_literal(embedding, self.config.embedding_dims)?;

        let knn_sql = d.vec0_knn_sql(&emb_literal, "$1", limit)
            .expect("vec0_knn_sql should be available when has_vec0_table() is true");

        let cols = memory_select_cols(Some("knn.distance"), "m.");
        let sql = format!(
            "WITH knn AS ({knn_sql}) SELECT {cols} FROM knn JOIN memories m ON m.id = knn.memory_id ORDER BY knn.distance ASC"
        );

        let params = &[SqlParam::Text(user_id.to_string())];
        self.backend.query_read(&sql, params, map_memory_row)
    }

    // ── FTS search ──

    /// Full-text search using BM25 scoring.
    /// Returns results ordered by relevance (highest BM25 score first).
    ///
    /// Note: the FTS index must have been built via `create_fts_index()`
    /// before calling this method. If the index doesn't exist, this will
    /// return an error.
    pub(crate) fn fts_search(
        &self,
        query: &str,
        user_id: &str,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        app_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryRow>> {
        // Build WHERE clause dynamically. $1 = query, $2 = user_id, then agent/run/app
        let mut conditions = vec![
            "memories_fts MATCH $1".to_string(),
            "m.user_id = $2".to_string(),
        ];
        let mut dynamic_params: Vec<SqlParam> = vec![
            SqlParam::Text(query.to_string()),
            SqlParam::Text(user_id.to_string()),
        ];
        let mut param_idx: usize = 2;

        if let Some(aid) = agent_id {
            param_idx += 1;
            conditions.push(format!("m.agent_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(aid.to_string()));
        }

        if let Some(rid) = run_id {
            param_idx += 1;
            conditions.push(format!("m.run_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(rid.to_string()));
        }

        if let Some(appid) = app_id {
            param_idx += 1;
            conditions.push(format!("m.app_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(appid.to_string()));
        }
        let _ = param_idx;

        let where_clause = conditions.join(" AND ");
        // FTS5: JOIN on id column, rank is negative (lower = better match)
        let cols = memory_select_cols(Some("memories_fts.rank"), "m.");
        let sql = format!(
            "SELECT {cols} FROM memories_fts JOIN memories m ON m.id = memories_fts.id \
             WHERE {where_clause} ORDER BY memories_fts.rank LIMIT {limit}"
        );

        self.backend.query_read(&sql, &dynamic_params, map_memory_row)
    }

    // ── Temporal search ──

    /// Search memories by event_time proximity, ordered by closeness to reference time.
    /// Only returns memories that have a non-NULL event_time.
    pub(crate) fn temporal_search(&self, user_id: &str, limit: usize) -> Result<Vec<MemoryRow>> {
        let cols = memory_select_cols(None, "");
        let sql = format!(
            "SELECT {cols} FROM memories WHERE user_id = $1 AND event_time IS NOT NULL \
             ORDER BY event_time DESC LIMIT $2"
        );

        let params = &[
            SqlParam::Text(user_id.to_string()),
            SqlParam::Int(limit as i64),
        ];
        self.backend.query_read(&sql, params, map_memory_row)
    }

    // ── Find by content hash ──

    /// Look up a memory by its content hash for a given user (and optionally agent/app).
    /// Returns `(id, content)` if found.
    pub(crate) fn find_by_hash(
        &self,
        hash: &str,
        user_id: &str,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        app_id: Option<&str>,
    ) -> Result<Option<(String, String)>> {
        let map_row = |row: &dyn RowAccess| -> Result<(String, String)> {
            Ok((row.get_string(0)?, row.get_string(1)?))
        };

        // Build WHERE clause dynamically
        let mut conditions = vec!["hash = $1".to_string(), "user_id = $2".to_string()];
        let mut dynamic_params: Vec<SqlParam> = vec![
            SqlParam::Text(hash.to_string()),
            SqlParam::Text(user_id.to_string()),
        ];
        let mut param_idx: usize = 2;

        if let Some(aid) = agent_id {
            param_idx += 1;
            conditions.push(format!("agent_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(aid.to_string()));
        }

        if let Some(rid) = run_id {
            param_idx += 1;
            conditions.push(format!("run_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(rid.to_string()));
        }

        if let Some(appid) = app_id {
            param_idx += 1;
            conditions.push(format!("app_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(appid.to_string()));
        }
        let _ = param_idx;

        let where_clause = conditions.join(" AND ");
        let sql = format!("SELECT id, content FROM memories WHERE {where_clause} LIMIT 1");

        self.backend.query_one(&sql, &dynamic_params, map_row)
    }
}

#[cfg(test)]
mod tests {
    use crate::config::MemoryConfig;
    use crate::storage::InsertMemoryParams;
    use crate::types::FilterExpression;

    use super::Storage;

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
    fn test_list_memories() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);

        // Insert for user1
        storage
            .insert_memory(
                "id1",
                "a",
                &emb,
                "user1",
                "h1",
                &InsertMemoryParams {
                    agent_id: Some("agent1".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        storage
            .insert_memory(
                "id2",
                "b",
                &emb,
                "user1",
                "h2",
                &InsertMemoryParams {
                    agent_id: Some("agent2".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        storage
            .insert_memory(
                "id3",
                "c",
                &emb,
                "user1",
                "h3",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        // Insert for user2
        storage
            .insert_memory(
                "id4",
                "d",
                &emb,
                "user2",
                "h4",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        // List all for user1
        let rows = storage
            .list_memories("user1", None, None, None, None, 10)
            .unwrap();
        assert_eq!(rows.len(), 3);

        // List for user1 with agent_id filter
        let rows = storage
            .list_memories("user1", Some("agent1"), None, None, None, 10)
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "id1");

        // List for user2
        let rows = storage
            .list_memories("user2", None, None, None, None, 10)
            .unwrap();
        assert_eq!(rows.len(), 1);

        // List with limit
        let rows = storage
            .list_memories("user1", None, None, None, None, 2)
            .unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_list_memories_with_app_id() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);

        storage
            .insert_memory(
                "id1",
                "a",
                &emb,
                "user1",
                "h1",
                &InsertMemoryParams {
                    app_id: Some("app1".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        storage
            .insert_memory(
                "id2",
                "b",
                &emb,
                "user1",
                "h2",
                &InsertMemoryParams {
                    app_id: Some("app2".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        storage
            .insert_memory(
                "id3",
                "c",
                &emb,
                "user1",
                "h3",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        let rows = storage
            .list_memories("user1", None, None, Some("app1"), None, 10)
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "id1");
    }

    #[test]
    fn test_list_memories_with_filter_expression() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);

        storage
            .insert_memory(
                "id1",
                "a",
                &emb,
                "user1",
                "h1",
                &InsertMemoryParams {
                    importance: Some(0.9),
                    ..Default::default()
                },
            )
            .unwrap();
        storage
            .insert_memory(
                "id2",
                "b",
                &emb,
                "user1",
                "h2",
                &InsertMemoryParams {
                    importance: Some(0.3),
                    ..Default::default()
                },
            )
            .unwrap();
        storage
            .insert_memory(
                "id3",
                "c",
                &emb,
                "user1",
                "h3",
                &InsertMemoryParams {
                    importance: Some(0.7),
                    ..Default::default()
                },
            )
            .unwrap();

        let filter = FilterExpression::gte("importance", serde_json::json!(0.5));
        let rows = storage
            .list_memories("user1", None, None, None, Some(&filter), 10)
            .unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_vector_search() {
        let storage = open_storage(4);
        let emb1 = vec![1.0, 0.0, 0.0, 0.0];
        let emb2 = vec![0.0, 1.0, 0.0, 0.0];
        let emb3 = vec![0.9, 0.1, 0.0, 0.0];

        storage
            .insert_memory(
                "id1",
                "a",
                &emb1,
                "user1",
                "h1",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        storage
            .insert_memory(
                "id2",
                "b",
                &emb2,
                "user1",
                "h2",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        storage
            .insert_memory(
                "id3",
                "c",
                &emb3,
                "user1",
                "h3",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        let query_emb = vec![1.0, 0.0, 0.0, 0.0];
        let results = storage
            .vector_search(&query_emb, "user1", None, None, None, None, 3)
            .unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "id1");
        assert!(results[0].score.unwrap() < 0.01);
        assert_eq!(results[1].id, "id3");
        assert_eq!(results[2].id, "id2");
    }

    #[test]
    fn test_vector_search_with_threshold() {
        let storage = open_storage(4);
        let emb1 = vec![1.0, 0.0, 0.0, 0.0];
        let emb2 = vec![0.0, 1.0, 0.0, 0.0];

        storage
            .insert_memory(
                "id1",
                "a",
                &emb1,
                "user1",
                "h1",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        storage
            .insert_memory(
                "id2",
                "b",
                &emb2,
                "user1",
                "h2",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        let query_emb = vec![1.0, 0.0, 0.0, 0.0];
        let results = storage
            .vector_search(&query_emb, "user1", None, None, None, None, 10)
            .unwrap();
        assert_eq!(results.len(), 2);
        let close: Vec<_> = results.iter().filter(|r| r.score.unwrap() < 0.5).collect();
        assert_eq!(close.len(), 1);
        assert_eq!(close[0].id, "id1");
    }

    #[test]
    fn test_vector_search_with_filter() {
        let storage = open_storage(4);
        let emb1 = vec![1.0, 0.0, 0.0, 0.0];
        let emb2 = vec![0.9, 0.1, 0.0, 0.0];

        storage
            .insert_memory(
                "id1",
                "a",
                &emb1,
                "user1",
                "h1",
                &InsertMemoryParams {
                    app_id: Some("app1".into()),
                    importance: Some(0.9),
                    ..Default::default()
                },
            )
            .unwrap();
        storage
            .insert_memory(
                "id2",
                "b",
                &emb2,
                "user1",
                "h2",
                &InsertMemoryParams {
                    app_id: Some("app2".into()),
                    importance: Some(0.3),
                    ..Default::default()
                },
            )
            .unwrap();

        let query_emb = vec![1.0, 0.0, 0.0, 0.0];
        let filter = FilterExpression::gte("importance", serde_json::json!(0.5));
        let results = storage
            .vector_search(&query_emb, "user1", None, None, None, Some(&filter), 10)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "id1");
    }

    #[test]
    fn test_find_by_hash() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);
        storage
            .insert_memory(
                "id1",
                "content",
                &emb,
                "user1",
                "hash123",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        let found = storage
            .find_by_hash("hash123", "user1", None, None, None)
            .unwrap();
        assert!(found.is_some());
        let (id, content) = found.unwrap();
        assert_eq!(id, "id1");
        assert_eq!(content, "content");

        let not_found = storage
            .find_by_hash("wrong_hash", "user1", None, None, None)
            .unwrap();
        assert!(not_found.is_none());

        let not_found = storage
            .find_by_hash("hash123", "user2", None, None, None)
            .unwrap();
        assert!(not_found.is_none());
    }

    #[test]
    fn test_fts_search_basic() {
        let storage = open_storage(4);
        let emb = vec![1.0, 0.0, 0.0, 0.0];

        storage
            .insert_memory(
                "id1",
                "the quick brown fox jumps over the lazy dog",
                &emb,
                "user1",
                "h1",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        storage
            .insert_memory(
                "id2",
                "quantum physics and relativity theory",
                &emb,
                "user1",
                "h2",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        storage
            .insert_memory(
                "id3",
                "the lazy cat sleeps all day long",
                &emb,
                "user1",
                "h3",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        storage.create_fts_index().unwrap();

        let results = storage
            .fts_search("lazy", "user1", None, None, None, 10)
            .unwrap();
        assert!(!results.is_empty());
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"id1"));
        assert!(ids.contains(&"id3"));
        assert!(!ids.contains(&"id2"));
    }

    #[test]
    fn test_fts_search_no_results() {
        let storage = open_storage(4);
        let emb = vec![1.0, 0.0, 0.0, 0.0];

        storage
            .insert_memory(
                "id1",
                "the quick brown fox",
                &emb,
                "user1",
                "h1",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        storage.create_fts_index().unwrap();

        let results = storage
            .fts_search("xylophone", "user1", None, None, None, 10)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_fts_search_user_filter() {
        let storage = open_storage(4);
        let emb = vec![1.0, 0.0, 0.0, 0.0];

        storage
            .insert_memory(
                "id1",
                "the quick brown fox",
                &emb,
                "user1",
                "h1",
                &InsertMemoryParams::default(),
            )
            .unwrap();
        storage
            .insert_memory(
                "id2",
                "the quick brown fox",
                &emb,
                "user2",
                "h2",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        storage.create_fts_index().unwrap();

        let results = storage
            .fts_search("quick", "user1", None, None, None, 10)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "id1");

        let results = storage
            .fts_search("quick", "user2", None, None, None, 10)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "id2");
    }

    #[test]
    fn test_list_with_pagination() {
        let storage = open_storage(384);
        let emb = dummy_embedding(384, 1.0);

        for i in 0..5 {
            storage
                .insert_memory(
                    &format!("id{i}"),
                    &format!("content {i}"),
                    &emb,
                    "user1",
                    &format!("h{i}"),
                    &InsertMemoryParams::default(),
                )
                .unwrap();
        }

        let page1 = storage
            .list_memories("user1", None, None, None, None, 3)
            .unwrap();
        assert_eq!(page1.len(), 3);

        let page2 = storage
            .list_memories("user1", None, None, None, None, 10)
            .unwrap();
        assert_eq!(page2.len(), 5);

        let page3 = storage
            .list_memories("user1", None, None, None, None, 1)
            .unwrap();
        assert_eq!(page3.len(), 1);
    }
}

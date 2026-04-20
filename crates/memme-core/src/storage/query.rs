use crate::error::Result;
use crate::types::{FilterExpression, SqlParam};

use super::backend::RowAccess;
use super::{EntityNeighborRow, MemoryRow, Storage};

// ── FTS query sanitization ──

/// Sanitize a query string for FTS5 MATCH: remove special characters that
/// cause syntax errors (?, *, ^, ", :, etc.) while preserving OR keywords
/// for graph-expanded queries.
fn sanitize_fts_query(query: &str) -> String {
    let mut result = String::with_capacity(query.len());
    for ch in query.chars() {
        match ch {
            // FTS5 special chars that cause syntax errors
            '?' | '*' | '^' | '"' | ':' | '{' | '}' | '(' | ')' | '!' | '~' => {
                result.push(' ');
            }
            _ => result.push(ch),
        }
    }
    // Collapse multiple spaces
    let parts: Vec<&str> = result.split_whitespace().collect();
    parts.join(" ")
}

// ── Unified memory column definitions ──

/// Base columns for MemoryRow (without score). Used to build SELECT clauses.
const MEMORY_COLS_BASE: &str = "id, content, user_id, created_at, updated_at, metadata";

const MEMORY_COLS_AFTER_SCORE: &str = "importance, access_count, agent_id, app_id, run_id, \
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
        let mut dynamic_params: Vec<SqlParam> = vec![SqlParam::Text(user_id.to_string())];
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

        let rows = self
            .backend
            .query_read(&sql, &dynamic_params, map_memory_row)?;

        Ok(rows)
    }

    // ── Entity-based neighbor search (for contradiction detection) ──

    /// Find memories sharing entities with the given memory.
    /// Returns (memory_id, content, shared_entity_count, embedding_blob).
    /// Used by multi-dimensional contradiction detection: entity-first
    /// filtering narrows the search space, then Rust-side vector scoring
    /// ranks candidates by semantic similarity.
    pub(crate) fn entity_neighbor_memories(
        &self,
        memory_id: &str,
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<EntityNeighborRow>> {
        let collection = &self.config.collection_name;
        let sql = format!(
            "SELECT me2.memory_id, m.content, \
                    COUNT(DISTINCT me2.entity_name) as shared_entities \
             FROM memory_entities me1 \
             JOIN memory_entities me2 ON me1.entity_name = me2.entity_name \
                  AND me1.memory_id != me2.memory_id \
             JOIN memories m ON m.id = me2.memory_id \
             WHERE me1.memory_id = $1 \
               AND m.user_id = $2 \
               AND m.superseded_by IS NULL \
             GROUP BY me2.memory_id \
             HAVING shared_entities >= 1 \
             ORDER BY shared_entities DESC \
             LIMIT {limit}"
        );

        self.backend.query_read(
            &sql,
            &[
                SqlParam::Text(memory_id.to_string()),
                SqlParam::Text(user_id.to_string()),
            ],
            |row| {
                Ok(EntityNeighborRow {
                    memory_id: row.get_string(0)?,
                    content: row.get_string(1)?,
                    shared_entities: row.get_i64(2)? as usize,
                })
            },
        )
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
        let has_extra_filters =
            agent_id.is_some() || run_id.is_some() || app_id.is_some() || filter.is_some();

        // Use vec0 MATCH for simple user_id-only queries (fast KNN path).
        // Fall back to brute-force scan when extra filters are present,
        // since vec0 doesn't support arbitrary WHERE clauses.
        if d.has_vec0_table() && !has_extra_filters {
            return self.vector_search_vec0(embedding, user_id, limit);
        }

        let mut conditions = vec!["user_id = $1".to_string()];
        let mut dynamic_params: Vec<SqlParam> = vec![SqlParam::Text(user_id.to_string())];
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

        let rows = self
            .backend
            .query_read(&sql, &dynamic_params, map_memory_row)?;

        Ok(rows)
    }

    /// vec0 MATCH-based vector search (SQLite with sqlite-vec).
    /// Queries both vec_memories and vec_events, merging results by distance.
    fn vector_search_vec0(
        &self,
        embedding: &[f32],
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<MemoryRow>> {
        let d = self.dialect();
        let emb_literal = d.format_embedding_literal(embedding, self.config.embedding_dims)?;

        let knn_sql = d
            .vec0_knn_sql(&emb_literal, "$1", limit)
            .expect("vec0_knn_sql should be available when has_vec0_table() is true");

        let mem_cols = memory_select_cols(Some("knn.distance"), "m.");
        let event_knn_sql = d.vec0_event_knn_sql(&emb_literal, "$1", limit);

        let sql = if let Some(ref evt_knn) = event_knn_sql {
            format!(
                r#"WITH mem_knn AS ({knn_sql}),
                     evt_knn AS ({evt_knn})
                SELECT * FROM (
                    SELECT {mem_cols} FROM mem_knn knn JOIN memories m ON m.id = knn.memory_id
                    UNION ALL
                    SELECT e.event_id AS id,
                           COALESCE(e.purified_content, e.content) AS content,
                           e.user_id,
                           e.timestamp AS created_at,
                           e.timestamp AS updated_at,
                           e.metadata,
                           eknn.distance AS score,
                           0.5 AS importance,
                           0 AS access_count,
                           NULL AS agent_id,
                           NULL AS app_id,
                           NULL AS run_id,
                           0 AS immutable,
                           NULL AS expiration_date,
                           NULL AS categories,
                           NULL AS memory_type,
                           1.0 AS stability,
                           'syncable' AS privacy,
                           e.event_time,
                           NULL AS episode_id,
                           e.session_id,
                           'granular' AS resolution
                    FROM evt_knn eknn JOIN events e ON e.event_id = eknn.event_id
                ) ORDER BY score ASC LIMIT {limit}"#
            )
        } else {
            format!(
                "WITH knn AS ({knn_sql}) SELECT {mem_cols} FROM knn JOIN memories m ON m.id = knn.memory_id ORDER BY knn.distance ASC"
            )
        };

        let params = &[SqlParam::Text(user_id.to_string())];
        self.backend.query_read(&sql, params, map_memory_row)
    }

    /// Find distinct session IDs from events matching a vector query.
    /// Used by lazy compact to scope compaction to relevant sessions only.
    /// Returns session IDs ordered by best match distance.
    pub(crate) fn find_relevant_event_sessions(
        &self,
        embedding: &[f32],
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<String>> {
        let d = self.dialect();
        let emb_literal = d.format_embedding_literal(embedding, self.config.embedding_dims)?;
        let event_knn = d.vec0_event_knn_sql(&emb_literal, "$1", limit * 3);
        match event_knn {
            Some(knn_sql) => self.backend.query_read(
                &format!(
                    r#"WITH eknn AS ({knn_sql})
                       SELECT DISTINCT e.session_id
                       FROM eknn JOIN events e ON e.event_id = eknn.event_id
                       WHERE e.session_id IS NOT NULL
                       ORDER BY MIN(eknn.distance)
                       LIMIT {limit}"#
                ),
                &[SqlParam::Text(user_id.to_string())],
                |row| row.get_string(0),
            ),
            None => Ok(Vec::new()),
        }
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
        // Sanitize query for FTS5: remove characters that are FTS5 operators or
        // cause syntax errors (?, *, ^, ", :, {, }, etc.)
        let sanitized_query = sanitize_fts_query(query);
        if sanitized_query.trim().is_empty() {
            return Ok(Vec::new());
        }

        // Build WHERE clause dynamically. $1 = query, $2 = user_id, then agent/run/app
        let mut conditions = vec![
            "memories_fts MATCH $1".to_string(),
            "m.user_id = $2".to_string(),
        ];
        let mut dynamic_params: Vec<SqlParam> = vec![
            SqlParam::Text(sanitized_query.clone()),
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
        let cols = memory_select_cols(Some("memories_fts.rank"), "m.");
        let has_extra_filters = agent_id.is_some() || run_id.is_some() || app_id.is_some();

        // UNION with events_fts when no agent/run/app filters
        if !has_extra_filters {
            let unified_sql = format!(
                r#"SELECT * FROM (
                    SELECT {cols} FROM memories_fts JOIN memories m ON m.id = memories_fts.id
                        WHERE {where_clause}
                    UNION ALL
                    SELECT e.event_id AS id,
                           COALESCE(e.purified_content, e.content) AS content,
                           e.user_id,
                           e.timestamp AS created_at,
                           e.timestamp AS updated_at,
                           e.metadata,
                           events_fts.rank AS score,
                           0.5 AS importance,
                           0 AS access_count,
                           NULL AS agent_id,
                           NULL AS app_id,
                           NULL AS run_id,
                           0 AS immutable,
                           NULL AS expiration_date,
                           NULL AS categories,
                           NULL AS memory_type,
                           1.0 AS stability,
                           'syncable' AS privacy,
                           e.event_time,
                           NULL AS episode_id,
                           e.session_id,
                           'granular' AS resolution
                    FROM events_fts JOIN events e ON e.event_id = events_fts.event_id
                        WHERE events_fts MATCH $1 AND e.user_id = $2
                ) ORDER BY score LIMIT {limit}"#
            );
            let result = self
                .backend
                .query_read(&unified_sql, &dynamic_params, map_memory_row);
            if result.is_ok() {
                return result;
            }
        }

        let sql = format!(
            "SELECT {cols} FROM memories_fts JOIN memories m ON m.id = memories_fts.id \
             WHERE {where_clause} ORDER BY memories_fts.rank LIMIT {limit}"
        );

        self.backend
            .query_read(&sql, &dynamic_params, map_memory_row)
    }

    // ── Temporal search ──

    /// Search memories by event_time proximity, ordered by closeness to reference time.
    /// Only returns memories that have a non-NULL event_time.
    #[allow(dead_code)] // V4 uses temporal as a filter, not a channel; kept for direct use
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

    /// Search memories + events whose `event_time` falls within any of the given
    /// date ranges. Results are ordered by event_time DESC.
    ///
    /// This is the upgraded temporal channel: instead of "recent N", it filters
    /// by the concrete date ranges extracted from the user's query.
    #[allow(dead_code)] // V4 uses temporal as a post-fusion filter; kept for direct use
    pub(crate) fn temporal_range_search(
        &self,
        user_id: &str,
        ranges: &[crate::time_parser::TimeRange],
        limit: usize,
    ) -> Result<Vec<MemoryRow>> {
        if ranges.is_empty() {
            return self.temporal_search(user_id, limit);
        }

        // Build OR clauses: one `event_time BETWEEN $start AND $end` per range.
        // Append "T23:59:59" to end dates so that timestamps within that day match.
        let mut range_conditions = Vec::new();
        let mut dynamic_params: Vec<SqlParam> = vec![SqlParam::Text(user_id.to_string())];
        let mut param_idx: usize = 1;

        for range in ranges {
            let start_idx = param_idx + 1;
            let end_idx = param_idx + 2;
            range_conditions.push(format!(
                "(event_time >= ${start_idx} AND event_time <= ${end_idx})"
            ));
            dynamic_params.push(SqlParam::Text(range.start.clone()));
            dynamic_params.push(SqlParam::Text(format!("{}T23:59:59", range.end)));
            param_idx += 2;
        }

        let range_where = range_conditions.join(" OR ");

        // Search memories table
        let mem_cols = memory_select_cols(None, "");
        let mem_sql = format!(
            "SELECT {mem_cols} FROM memories \
             WHERE user_id = $1 AND event_time IS NOT NULL AND ({range_where}) \
             ORDER BY event_time DESC LIMIT {limit}"
        );

        let mut results = self
            .backend
            .query_read(&mem_sql, &dynamic_params, map_memory_row)?;

        // Also search events table — map events to MemoryRow for uniform handling.
        // Reuse the same params (user_id + range pairs).
        let evt_sql = format!(
            r#"SELECT
                e.event_id AS id,
                COALESCE(e.purified_content, e.content) AS content,
                e.user_id,
                e.timestamp AS created_at,
                e.timestamp AS updated_at,
                e.metadata,
                NULL AS score,
                0.5 AS importance,
                0 AS access_count,
                NULL AS agent_id,
                NULL AS app_id,
                NULL AS run_id,
                0 AS immutable,
                NULL AS expiration_date,
                NULL AS categories,
                NULL AS memory_type,
                1.0 AS stability,
                'syncable' AS privacy,
                e.event_time,
                NULL AS episode_id,
                e.session_id,
                'granular' AS resolution
            FROM events e
            WHERE e.user_id = $1 AND e.event_time IS NOT NULL AND ({range_where})
            ORDER BY e.event_time DESC LIMIT {limit}"#
        );

        if let Ok(event_rows) = self
            .backend
            .query_read(&evt_sql, &dynamic_params, map_memory_row)
        {
            results.extend(event_rows);
        }

        // Sort combined results by event_time DESC and truncate
        results.sort_by(|a, b| {
            b.event_time
                .as_deref()
                .unwrap_or("")
                .cmp(a.event_time.as_deref().unwrap_or(""))
        });
        results.truncate(limit);

        Ok(results)
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
        MemoryConfig::new(":memory:", dims)
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

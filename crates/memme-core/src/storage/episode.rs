use duckdb::params;

use crate::error::Result;
use crate::types::{CreateEpisodeOptions, Episode, ListEpisodesOptions, SearchEpisodesOptions};

use super::Storage;

const EPISODE_COLS: &str = "episode_id, title, summary, CAST(started_at AS VARCHAR), CAST(ended_at AS VARCHAR), significance, outcome, source_id, event_ids, user_id, CAST(created_at AS VARCHAR), CAST(last_recalled AS VARCHAR), recall_count, storage_strength, retrieval_strength, session_ids, CAST(last_meditated_at AS VARCHAR)";

impl Storage {
    /// Insert a new episode.
    pub(crate) fn insert_episode(
        &self,
        episode_id: &str,
        title: &str,
        summary: &str,
        summary_vec: &[f32],
        options: &CreateEpisodeOptions,
    ) -> Result<()> {
        let emb_literal = Self::format_embedding(summary_vec, self.config.embedding_dims)?;
        let ended_val: duckdb::types::Value = match &options.ended_at {
            Some(t) => duckdb::types::Value::Text(t.clone()),
            None => duckdb::types::Value::Null,
        };
        let outcome_val: duckdb::types::Value = match &options.outcome {
            Some(o) => duckdb::types::Value::Text(o.clone()),
            None => duckdb::types::Value::Null,
        };
        let source_val: duckdb::types::Value = match &options.source_id {
            Some(s) => duckdb::types::Value::Text(s.clone()),
            None => duckdb::types::Value::Null,
        };
        let event_ids_str = serde_json::to_string(&options.event_ids).unwrap_or_default();
        let session_ids_str = serde_json::to_string(&options.session_ids).unwrap_or_default();
        let significance = options.significance.unwrap_or(0.5) as f64;

        let sql = format!(
            r#"INSERT INTO episodes (episode_id, title, summary, summary_vec, started_at, ended_at, significance, outcome, source_id, event_ids, user_id, session_ids)
               VALUES ($1, $2, $3, {emb_literal}, CAST($4 AS TIMESTAMP), CASE WHEN $5 IS NULL THEN NULL ELSE CAST($5 AS TIMESTAMP) END, $6, $7, $8, $9, $10, $11)"#
        );
        let conn = self.write_conn();
        conn.execute(
            &sql,
            params![
                episode_id,
                title,
                summary,
                &options.started_at,
                ended_val,
                significance,
                outcome_val,
                source_val,
                event_ids_str,
                &options.user_id,
                session_ids_str
            ],
        )?;
        Ok(())
    }

    /// Get an episode by ID.
    pub(crate) fn get_episode(&self, episode_id: &str) -> Result<Option<Episode>> {
        let conn = self.read_conn();
        let sql = format!("SELECT {EPISODE_COLS} FROM episodes WHERE episode_id = $1");
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query_map(params![episode_id], map_episode_row)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Search episodes by vector similarity.
    pub(crate) fn search_episodes_by_vector(
        &self,
        query_vec: &[f32],
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<Episode>> {
        let emb_literal = Self::format_embedding(query_vec, self.config.embedding_dims)?;
        let sql = format!(
            r#"SELECT {EPISODE_COLS},
                      array_cosine_distance(summary_vec, {emb_literal}) AS distance
               FROM episodes
               WHERE user_id = $1
               ORDER BY distance ASC
               LIMIT {limit}"#
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![user_id], |row| {
                let mut ep = map_episode_row_inner(row)?;
                ep.score = row.get::<_, Option<f64>>(17)?.map(|d| d as f32);
                Ok(ep)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// List episodes with filters.
    #[allow(dead_code)] // used by list_episodes_for_user and list_episodes_paged
    pub(crate) fn list_episodes(&self, options: &SearchEpisodesOptions) -> Result<Vec<Episode>> {
        let mut conditions = vec!["user_id = $1".to_string()];
        let mut dynamic_params: Vec<duckdb::types::Value> =
            vec![duckdb::types::Value::Text(options.user_id.clone())];
        let mut param_idx: usize = 1;

        if let Some(ref since) = options.since {
            param_idx += 1;
            conditions.push(format!("started_at >= CAST(${param_idx} AS TIMESTAMP)"));
            dynamic_params.push(duckdb::types::Value::Text(since.clone()));
        }
        if let Some(ref until) = options.until {
            param_idx += 1;
            conditions.push(format!("started_at <= CAST(${param_idx} AS TIMESTAMP)"));
            dynamic_params.push(duckdb::types::Value::Text(until.clone()));
        }
        if let Some(min_sig) = options.min_significance {
            if min_sig.is_finite() {
                conditions.push(format!("significance >= {}", min_sig as f64));
            }
        }
        let _ = param_idx;

        let limit = options.limit.unwrap_or(20);
        let where_clause = conditions.join(" AND ");
        // Note: SearchEpisodesOptions doesn't have offset, but ListEpisodesOptions does.
        // We pass offset=0 here since this method is shared.
        let sql = format!(
            "SELECT {EPISODE_COLS} FROM episodes WHERE {where_clause} ORDER BY started_at DESC LIMIT {limit}"
        );

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn duckdb::ToSql> = dynamic_params
            .iter()
            .map(|p| p as &dyn duckdb::ToSql)
            .collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), map_episode_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// List episodes with pagination (limit + offset).
    pub(crate) fn list_episodes_paged(
        &self,
        options: &ListEpisodesOptions,
    ) -> Result<Vec<Episode>> {
        let mut conditions = vec!["user_id = $1".to_string()];
        let mut dynamic_params: Vec<duckdb::types::Value> =
            vec![duckdb::types::Value::Text(options.user_id.clone())];
        let mut param_idx: usize = 1;

        if let Some(ref since) = options.since {
            param_idx += 1;
            conditions.push(format!("started_at >= CAST(${param_idx} AS TIMESTAMP)"));
            dynamic_params.push(duckdb::types::Value::Text(since.clone()));
        }
        if let Some(ref until) = options.until {
            param_idx += 1;
            conditions.push(format!("started_at <= CAST(${param_idx} AS TIMESTAMP)"));
            dynamic_params.push(duckdb::types::Value::Text(until.clone()));
        }
        if let Some(min_sig) = options.min_significance {
            if min_sig.is_finite() {
                conditions.push(format!("significance >= {}", min_sig as f64));
            }
        }
        let _ = param_idx;

        let limit = options.limit.unwrap_or(20);
        let offset = options.offset.unwrap_or(0);
        let where_clause = conditions.join(" AND ");
        let sql = format!(
            "SELECT {EPISODE_COLS} FROM episodes WHERE {where_clause} ORDER BY started_at DESC LIMIT {limit} OFFSET {offset}"
        );

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn duckdb::ToSql> = dynamic_params
            .iter()
            .map(|p| p as &dyn duckdb::ToSql)
            .collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), map_episode_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Reinforce an episode (bump recall_count, update last_recalled, increase storage_strength).
    pub(crate) fn reinforce_episode(&self, episode_id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            r#"UPDATE episodes
               SET recall_count = recall_count + 1,
                   last_recalled = current_timestamp,
                   storage_strength = LEAST(10.0, storage_strength * 1.1)
               WHERE episode_id = $1"#,
            params![episode_id],
        )?;
        Ok(())
    }

    /// Delete an episode.
    pub(crate) fn delete_episode(&self, episode_id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            "DELETE FROM episodes WHERE episode_id = $1",
            params![episode_id],
        )?;
        Ok(())
    }

    /// Mark an episode as meditated (set last_meditated_at to now).
    pub(crate) fn mark_episode_meditated(&self, episode_id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            "UPDATE episodes SET last_meditated_at = current_timestamp WHERE episode_id = $1",
            params![episode_id],
        )?;
        Ok(())
    }

    /// Delete all episodes for a user (used by re_traces to rebuild from scratch).
    pub(crate) fn delete_episodes_for_user(&self, user_id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            "DELETE FROM episodes WHERE user_id = $1",
            params![user_id],
        )?;
        Ok(())
    }

    /// Find an episode by source_id (run_id) and user_id. Returns the most recent one.
    #[allow(dead_code)] // planned API: episode lookup
    pub(crate) fn find_episode_by_source(
        &self,
        source_id: &str,
        user_id: &str,
    ) -> Result<Option<Episode>> {
        let conn = self.read_conn();
        let sql = format!("SELECT {EPISODE_COLS} FROM episodes WHERE source_id = $1 AND user_id = $2 ORDER BY created_at DESC LIMIT 1");
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query_map(params![source_id, user_id], map_episode_row)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Append event IDs to an existing episode.
    #[allow(dead_code)] // planned API: episode event management
    pub(crate) fn append_events_to_episode(
        &self,
        episode_id: &str,
        new_event_ids: &[String],
    ) -> Result<()> {
        // Only fetch event_ids column, not the full episode
        let conn = self.write_conn();
        let mut stmt = conn.prepare("SELECT event_ids FROM episodes WHERE episode_id = $1")?;
        let existing_ids: Vec<String> = stmt
            .query_map(params![episode_id], |row| row.get::<_, Option<String>>(0))?
            .next()
            .transpose()?
            .flatten()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        let mut all_ids = existing_ids;
        all_ids.extend(new_event_ids.iter().cloned());
        let ids_json = serde_json::to_string(&all_ids)
            .map_err(|e| crate::error::MemoryError::Config(e.to_string()))?;
        conn.execute(
            "UPDATE episodes SET event_ids = $1 WHERE episode_id = $2",
            params![ids_json, episode_id],
        )?;
        Ok(())
    }

    /// List all episodes for a user.
    #[allow(dead_code)] // public API: episode listing
    pub(crate) fn list_episodes_for_user(&self, user_id: &str) -> Result<Vec<Episode>> {
        let options = SearchEpisodesOptions {
            user_id: user_id.to_string(),
            since: None,
            until: None,
            min_significance: None,
            limit: Some(100),
        };
        self.list_episodes(&options)
    }

    /// List unmeditated episodes for meditation, filtered by significance and batch size.
    pub(crate) fn list_episodes_for_meditation(
        &self,
        user_id: &str,
        min_significance: f32,
        batch_size: usize,
    ) -> Result<Vec<Episode>> {
        let limit = if batch_size == 0 { 100 } else { batch_size };
        let sql = format!(
            "SELECT {EPISODE_COLS} FROM episodes \
             WHERE user_id = $1 AND last_meditated_at IS NULL AND significance >= $2 \
             ORDER BY significance DESC \
             LIMIT {limit}"
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![user_id, min_significance as f64], map_episode_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// FTS (BM25) search on episodes (title + summary).
    pub(crate) fn fts_search_episodes(
        &self,
        query: &str,
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<Episode>> {
        // For FTS we need table-qualified column names
        let fts_cols = "e.episode_id, e.title, e.summary, CAST(e.started_at AS VARCHAR), CAST(e.ended_at AS VARCHAR), e.significance, e.outcome, e.source_id, e.event_ids, e.user_id, CAST(e.created_at AS VARCHAR), CAST(e.last_recalled AS VARCHAR), e.recall_count, e.storage_strength, e.retrieval_strength, e.session_ids, CAST(e.last_meditated_at AS VARCHAR)";
        let sql = format!(
            r#"SELECT {fts_cols},
                      fts_main_episodes.match_bm25(e.episode_id, $1) AS score
               FROM episodes e
               WHERE score IS NOT NULL AND e.user_id = $2
               ORDER BY score DESC
               LIMIT {limit}"#
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![query, user_id], |row| {
                let mut ep = map_episode_row_inner(row)?;
                ep.score = row.get::<_, Option<f64>>(17)?.map(|d| d as f32);
                Ok(ep)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }
}

fn map_episode_row(row: &duckdb::Row<'_>) -> duckdb::Result<Episode> {
    map_episode_row_inner(row)
}

fn map_episode_row_inner(row: &duckdb::Row<'_>) -> duckdb::Result<Episode> {
    let event_ids_raw: Option<String> = row.get(8)?;
    let event_ids: Vec<String> = event_ids_raw
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    let session_ids_raw: Option<String> = row.get::<_, Option<String>>(15).ok().flatten();
    let session_ids: Vec<String> = session_ids_raw
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    Ok(Episode {
        episode_id: row.get(0)?,
        title: row.get(1)?,
        summary: row.get(2)?,
        started_at: row.get::<_, String>(3)?,
        ended_at: row.get::<_, Option<String>>(4)?,
        significance: row.get::<_, Option<f64>>(5)?.unwrap_or(0.5) as f32,
        outcome: row.get::<_, Option<String>>(6)?,
        source_id: row.get::<_, Option<String>>(7)?,
        event_ids,
        session_ids,
        user_id: row.get(9)?,
        created_at: row.get::<_, String>(10)?,
        last_recalled: row.get::<_, Option<String>>(11)?,
        recall_count: row.get::<_, Option<i32>>(12)?.unwrap_or(0) as u32,
        storage_strength: row.get::<_, Option<f64>>(13)?.unwrap_or(1.0) as f32,
        retrieval_strength: row.get::<_, Option<f64>>(14)?.unwrap_or(1.0) as f32,
        last_meditated_at: row.get::<_, Option<String>>(16)?,
        score: None,
    })
}

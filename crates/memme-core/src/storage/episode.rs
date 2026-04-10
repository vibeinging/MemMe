use crate::error::Result;
use crate::types::{
    CreateEpisodeOptions, Episode, ListEpisodesOptions, SearchEpisodesOptions, SqlParam,
};

use super::backend::RowAccess;
use super::util::opt_text;
use super::Storage;

/// Generate episode SELECT columns.
fn episode_cols() -> &'static str {
    "episode_id, title, summary, started_at, ended_at, significance, outcome, source_id, event_ids, user_id, created_at, last_recalled, recall_count, storage_strength, retrieval_strength, session_ids, last_meditated_at"
}

/// Generate table-qualified episode SELECT columns for FTS queries.
fn episode_cols_qualified(prefix: &str) -> String {
    format!("{prefix}.episode_id, {prefix}.title, {prefix}.summary, {prefix}.started_at, {prefix}.ended_at, {prefix}.significance, {prefix}.outcome, {prefix}.source_id, {prefix}.event_ids, {prefix}.user_id, {prefix}.created_at, {prefix}.last_recalled, {prefix}.recall_count, {prefix}.storage_strength, {prefix}.retrieval_strength, {prefix}.session_ids, {prefix}.last_meditated_at")
}

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
        let emb_literal = self.format_embedding(summary_vec, self.config.embedding_dims)?;
        let ended_val = opt_text(options.ended_at.as_deref());
        let outcome_val = opt_text(options.outcome.as_deref());
        let source_val = opt_text(options.source_id.as_deref());
        let event_ids_str = serde_json::to_string(&options.event_ids).unwrap_or_default();
        let session_ids_str = serde_json::to_string(&options.session_ids).unwrap_or_default();
        let significance = options.significance.unwrap_or(0.5) as f64;

        let sql = format!(
            r#"INSERT INTO episodes (episode_id, title, summary, summary_vec, started_at, ended_at, significance, outcome, source_id, event_ids, user_id, session_ids)
               VALUES ($1, $2, $3, {emb_literal}, $4, $5, $6, $7, $8, $9, $10, $11)"#
        );
        self.backend.execute(
            &sql,
            &[
                SqlParam::Text(episode_id.to_string()),
                SqlParam::Text(title.to_string()),
                SqlParam::Text(summary.to_string()),
                SqlParam::Text(options.started_at.clone()),
                ended_val,
                SqlParam::Float(significance),
                outcome_val,
                source_val,
                SqlParam::Text(event_ids_str),
                SqlParam::Text(options.user_id.clone()),
                SqlParam::Text(session_ids_str),
            ],
        )?;
        Ok(())
    }

    /// Get an episode by ID.
    pub(crate) fn get_episode(&self, episode_id: &str) -> Result<Option<Episode>> {
        let cols = episode_cols();
        let sql = format!("SELECT {cols} FROM episodes WHERE episode_id = $1");
        self.backend
            .query_one(&sql, &[SqlParam::Text(episode_id.to_string())], |row| {
                map_episode_row(row)
            })
    }

    /// Search episodes by vector similarity.
    pub(crate) fn search_episodes_by_vector(
        &self,
        query_vec: &[f32],
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<Episode>> {
        let emb_literal = self.format_embedding(query_vec, self.config.embedding_dims)?;
        let cols = episode_cols();
        let distance_expr = self
            .dialect()
            .cosine_distance_expr("summary_vec", &emb_literal);
        let sql = format!(
            r#"SELECT {cols},
                      {distance_expr} AS distance
               FROM episodes
               WHERE user_id = $1
               ORDER BY distance ASC
               LIMIT {limit}"#
        );
        self.backend
            .query_read(&sql, &[SqlParam::Text(user_id.to_string())], |row| {
                let mut ep = map_episode_row(row)?;
                ep.score = row.get_opt_f64(17)?.map(|d| d as f32);
                Ok(ep)
            })
    }

    /// List episodes with filters.
    #[allow(dead_code)] // used by list_episodes_for_user and list_episodes_paged
    pub(crate) fn list_episodes(&self, options: &SearchEpisodesOptions) -> Result<Vec<Episode>> {
        let mut conditions = vec!["user_id = $1".to_string()];
        let mut dynamic_params: Vec<SqlParam> = vec![SqlParam::Text(options.user_id.clone())];
        let mut param_idx: usize = 1;

        if let Some(ref since) = options.since {
            param_idx += 1;
            conditions.push(format!("started_at >= ${param_idx}"));
            dynamic_params.push(SqlParam::Text(since.clone()));
        }
        if let Some(ref until) = options.until {
            param_idx += 1;
            conditions.push(format!("started_at <= ${param_idx}"));
            dynamic_params.push(SqlParam::Text(until.clone()));
        }
        if let Some(min_sig) = options.min_significance {
            if min_sig.is_finite() {
                conditions.push(format!("significance >= {}", min_sig as f64));
            }
        }
        let _ = param_idx;

        let cols = episode_cols();
        let limit = options.limit.unwrap_or(20);
        let where_clause = conditions.join(" AND ");
        let sql = format!(
            "SELECT {cols} FROM episodes WHERE {where_clause} ORDER BY started_at DESC LIMIT {limit}"
        );

        self.backend
            .query_read(&sql, &dynamic_params, |row| map_episode_row(row))
    }

    /// List episodes with pagination (limit + offset).
    pub(crate) fn list_episodes_paged(
        &self,
        options: &ListEpisodesOptions,
    ) -> Result<Vec<Episode>> {
        let mut conditions = vec!["user_id = $1".to_string()];
        let mut dynamic_params: Vec<SqlParam> = vec![SqlParam::Text(options.user_id.clone())];
        let mut param_idx: usize = 1;

        if let Some(ref since) = options.since {
            param_idx += 1;
            conditions.push(format!("started_at >= ${param_idx}"));
            dynamic_params.push(SqlParam::Text(since.clone()));
        }
        if let Some(ref until) = options.until {
            param_idx += 1;
            conditions.push(format!("started_at <= ${param_idx}"));
            dynamic_params.push(SqlParam::Text(until.clone()));
        }
        if let Some(min_sig) = options.min_significance {
            if min_sig.is_finite() {
                conditions.push(format!("significance >= {}", min_sig as f64));
            }
        }
        let _ = param_idx;

        let cols = episode_cols();
        let limit = options.limit.unwrap_or(20);
        let offset = options.offset.unwrap_or(0);
        let where_clause = conditions.join(" AND ");
        let sql = format!(
            "SELECT {cols} FROM episodes WHERE {where_clause} ORDER BY started_at DESC LIMIT {limit} OFFSET {offset}"
        );

        self.backend
            .query_read(&sql, &dynamic_params, |row| map_episode_row(row))
    }

    /// Reinforce an episode (bump recall_count, update last_recalled, increase storage_strength).
    pub(crate) fn reinforce_episode(&self, episode_id: &str) -> Result<()> {
        let least_expr = self.dialect().least_expr("10.0", "storage_strength * 1.1");
        let now = self.dialect().current_timestamp_expr();
        let sql = format!(
            r#"UPDATE episodes
               SET recall_count = recall_count + 1,
                   last_recalled = {now},
                   storage_strength = {least_expr}
               WHERE episode_id = $1"#
        );
        self.backend
            .execute(&sql, &[SqlParam::Text(episode_id.to_string())])?;
        Ok(())
    }

    /// Delete an episode.
    pub(crate) fn delete_episode(&self, episode_id: &str) -> Result<()> {
        self.backend.execute(
            "DELETE FROM episodes WHERE episode_id = $1",
            &[SqlParam::Text(episode_id.to_string())],
        )?;
        Ok(())
    }

    /// Mark an episode as meditated (set last_meditated_at to now).
    pub(crate) fn mark_episode_meditated(&self, episode_id: &str) -> Result<()> {
        let now = self.dialect().current_timestamp_expr();
        self.backend.execute(
            &format!("UPDATE episodes SET last_meditated_at = {now} WHERE episode_id = $1"),
            &[SqlParam::Text(episode_id.to_string())],
        )?;
        Ok(())
    }

    /// Delete all episodes for a user (used by re_traces to rebuild from scratch).
    pub(crate) fn delete_episodes_for_user(&self, user_id: &str) -> Result<()> {
        self.backend.execute(
            "DELETE FROM episodes WHERE user_id = $1",
            &[SqlParam::Text(user_id.to_string())],
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
        let cols = episode_cols();
        let sql = format!("SELECT {cols} FROM episodes WHERE source_id = $1 AND user_id = $2 ORDER BY created_at DESC LIMIT 1");
        self.backend.query_one(
            &sql,
            &[
                SqlParam::Text(source_id.to_string()),
                SqlParam::Text(user_id.to_string()),
            ],
            |row| map_episode_row(row),
        )
    }

    /// Append event IDs to an existing episode.
    #[allow(dead_code)] // planned API: episode event management
    pub(crate) fn append_events_to_episode(
        &self,
        episode_id: &str,
        new_event_ids: &[String],
    ) -> Result<()> {
        // Only fetch event_ids column, not the full episode
        let existing_ids: Vec<String> = self
            .backend
            .query_one(
                "SELECT event_ids FROM episodes WHERE episode_id = $1",
                &[SqlParam::Text(episode_id.to_string())],
                |row| row.get_opt_string(0),
            )?
            .flatten()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        let mut all_ids = existing_ids;
        all_ids.extend(new_event_ids.iter().cloned());
        let ids_json = serde_json::to_string(&all_ids)
            .map_err(|e| crate::error::MemoryError::Config(e.to_string()))?;
        self.backend.execute(
            "UPDATE episodes SET event_ids = $1 WHERE episode_id = $2",
            &[
                SqlParam::Text(ids_json),
                SqlParam::Text(episode_id.to_string()),
            ],
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
    ///
    /// Ordering prioritizes episodes whose sessions have been hit by search queries
    /// (feedback-driven consolidation), then falls back to significance.
    pub(crate) fn list_episodes_for_meditation(
        &self,
        user_id: &str,
        min_significance: f32,
        batch_size: usize,
    ) -> Result<Vec<Episode>> {
        let cols = episode_cols();
        let limit = if batch_size == 0 { 100 } else { batch_size };
        // Prioritize episodes linked to sessions that users have queried.
        // json_each unnests the session_ids JSON array so we can look up
        // queried_count from the sessions table. Episodes with higher
        // aggregate query counts are meditated first.
        let sql = format!(
            "SELECT {cols} FROM episodes e \
             WHERE e.user_id = $1 AND e.last_meditated_at IS NULL AND e.significance >= $2 \
             ORDER BY \
               COALESCE(( \
                 SELECT MAX(s.queried_count) \
                 FROM sessions s, json_each(e.session_ids) j \
                 WHERE s.session_id = j.value \
               ), 0) DESC, \
               e.significance DESC \
             LIMIT {limit}"
        );
        self.backend.query_read(
            &sql,
            &[
                SqlParam::Text(user_id.to_string()),
                SqlParam::Float(min_significance as f64),
            ],
            |row| map_episode_row(row),
        )
    }

    /// FTS (BM25) search on episodes (title + summary).
    pub(crate) fn fts_search_episodes(
        &self,
        query: &str,
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<Episode>> {
        // For FTS we need table-qualified column names
        let fts_cols = episode_cols_qualified("e");
        let fts_score = self
            .dialect()
            .fts_match_score_expr("episodes", "e.episode_id", "$1");
        let sql = format!(
            r#"SELECT {fts_cols},
                      {fts_score} AS score
               FROM episodes e
               WHERE score IS NOT NULL AND e.user_id = $2
               ORDER BY score DESC
               LIMIT {limit}"#
        );
        self.backend.query_read(
            &sql,
            &[
                SqlParam::Text(query.to_string()),
                SqlParam::Text(user_id.to_string()),
            ],
            |row| {
                let mut ep = map_episode_row(row)?;
                ep.score = row.get_opt_f64(17)?.map(|d| d as f32);
                Ok(ep)
            },
        )
    }
}

fn map_episode_row(row: &dyn RowAccess) -> Result<Episode> {
    let event_ids_raw: Option<String> = row.get_opt_string(8)?;
    let event_ids: Vec<String> = event_ids_raw
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    let session_ids_raw: Option<String> = row.get_opt_string(15)?;
    let session_ids: Vec<String> = session_ids_raw
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    Ok(Episode {
        episode_id: row.get_string(0)?,
        title: row.get_string(1)?,
        summary: row.get_string(2)?,
        started_at: row.get_string(3)?,
        ended_at: row.get_opt_string(4)?,
        significance: row.get_opt_f64(5)?.unwrap_or(0.5) as f32,
        outcome: row.get_opt_string(6)?,
        source_id: row.get_opt_string(7)?,
        event_ids,
        session_ids,
        user_id: row.get_string(9)?,
        created_at: row.get_string(10)?,
        last_recalled: row.get_opt_string(11)?,
        recall_count: row.get_opt_i64(12)?.unwrap_or(0) as u32,
        storage_strength: row.get_opt_f64(13)?.unwrap_or(1.0) as f32,
        retrieval_strength: row.get_opt_f64(14)?.unwrap_or(1.0) as f32,
        last_meditated_at: row.get_opt_string(16)?,
        score: None,
    })
}

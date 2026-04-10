use crate::error::Result;
use crate::types::{Event, EventType, IngestEventOptions, ListEventsOptions, Source, SqlParam};

use super::backend::RowAccess;
use super::util::opt_text;
use super::Storage;

/// Generate event SELECT columns.
fn event_cols() -> &'static str {
    "event_id, source_id, session_id, timestamp, event_type, content, parent_id, metadata, user_id, processed, processed_at, purified_content, purified, event_time, location"
}

/// Map a row to an Event struct. Expects columns in standard order:
/// event_id(0), source_id(1), session_id(2), timestamp(3), event_type(4),
/// content(5), parent_id(6), metadata(7), user_id(8), processed(9), processed_at(10),
/// purified_content(11), purified(12), event_time(13), location(14)
fn map_event_row(row: &dyn RowAccess) -> Result<Event> {
    Ok(Event {
        event_id: row.get_string(0)?,
        source_id: row.get_opt_string(1)?,
        session_id: row.get_opt_string(2)?,
        timestamp: row.get_string(3)?,
        event_type: EventType::parse(&row.get_opt_string(4)?.unwrap_or_default()),
        content: row.get_string(5)?,
        parent_id: row.get_opt_string(6)?,
        metadata: row
            .get_opt_string(7)?
            .and_then(|s| serde_json::from_str(&s).ok()),
        user_id: row.get_string(8)?,
        processed: row.get_opt_bool(9)?.unwrap_or(false),
        processed_at: row.get_opt_string(10)?,
        purified_content: row.get_opt_string(11)?,
        purified: row.get_opt_bool(12)?.unwrap_or(false),
        event_time: row.get_opt_string(13)?,
        location: row.get_opt_string(14)?,
    })
}

fn map_source_row(row: &dyn RowAccess) -> Result<Source> {
    Ok(Source {
        source_id: row.get_string(0)?,
        source_type: row.get_string(1)?,
        name: row.get_opt_string(2)?,
        registered_at: row.get_string(3)?,
        metadata: row
            .get_opt_string(4)?
            .and_then(|s| serde_json::from_str(&s).ok()),
    })
}

impl Storage {
    /// Register a data source.
    #[allow(dead_code)] // planned API: stream source management
    pub(crate) fn register_source(
        &self,
        source_id: &str,
        source_type: &str,
        name: Option<&str>,
        metadata: Option<&serde_json::Value>,
        user_id: &str,
    ) -> Result<()> {
        let name_val = opt_text(name);
        let meta_val = opt_text(metadata.map(|m| serde_json::to_string(m).unwrap_or_default()));
        self.backend.execute(
            "INSERT INTO sources (source_id, source_type, name, metadata, user_id) VALUES ($1, $2, $3, $4, $5)",
            &[
                SqlParam::Text(source_id.to_string()),
                SqlParam::Text(source_type.to_string()),
                name_val,
                meta_val,
                SqlParam::Text(user_id.to_string()),
            ],
        )?;
        Ok(())
    }

    /// Get a source by ID.
    #[allow(dead_code)] // planned API: stream source management
    pub(crate) fn get_source(&self, source_id: &str) -> Result<Option<Source>> {
        self.backend.query_one(
            "SELECT source_id, source_type, name, registered_at, metadata FROM sources WHERE source_id = $1",
            &[SqlParam::Text(source_id.to_string())],
            |row| map_source_row(row),
        )
    }

    /// List all sources for a user.
    #[allow(dead_code)] // planned API: stream source management
    pub(crate) fn list_sources(&self, user_id: &str) -> Result<Vec<Source>> {
        self.backend.query_read(
            "SELECT source_id, source_type, name, registered_at, metadata FROM sources WHERE user_id = $1 ORDER BY registered_at DESC",
            &[SqlParam::Text(user_id.to_string())],
            |row| map_source_row(row),
        )
    }

    /// Insert a new event into the stream.
    /// content_vec can be empty (will be filled during compact after purification).
    pub(crate) fn insert_event(
        &self,
        event_id: &str,
        content: &str,
        content_vec: &[f32],
        options: &IngestEventOptions,
    ) -> Result<()> {
        let emb_literal = if content_vec.is_empty() {
            "NULL".to_string()
        } else {
            self.format_embedding(content_vec, self.config.embedding_dims)?
        };
        let event_type = options.event_type.as_deref().unwrap_or("system");
        let timestamp_val = opt_text(options.timestamp.as_deref());
        let source_val = opt_text(options.source_id.as_deref());
        let session_val = opt_text(options.session_id.as_deref());
        let parent_val = opt_text(options.parent_id.as_deref());
        let meta_val = opt_text(
            options
                .metadata
                .as_ref()
                .map(|m| serde_json::to_string(m).unwrap_or_default()),
        );

        let now_expr = self.dialect().current_timestamp_expr();
        let sql = format!(
            r#"INSERT INTO events (event_id, source_id, session_id, timestamp, event_type, content, content_vec, parent_id, metadata, user_id)
               VALUES ($1, $2, $3, CASE WHEN $4 IS NULL THEN {now_expr} ELSE $4 END, $5, $6, {emb_literal}, $7, $8, $9)"#
        );
        self.backend.execute(
            &sql,
            &[
                SqlParam::Text(event_id.to_string()),
                source_val,
                session_val,
                timestamp_val,
                SqlParam::Text(event_type.to_string()),
                SqlParam::Text(content.to_string()),
                parent_val,
                meta_val,
                SqlParam::Text(options.user_id.clone()),
            ],
        )?;
        // Sync vec_events virtual table for unified vector search
        if !content_vec.is_empty() {
            if let Some(vec0_sql) = self.dialect().vec0_event_insert_sql("$1", &emb_literal) {
                self.backend.execute(
                    &vec0_sql,
                    &[
                        SqlParam::Text(event_id.to_string()),
                        SqlParam::Text(options.user_id.clone()),
                    ],
                )?;
            }
        }
        Ok(())
    }

    /// Update event with purified content and its embedding.
    pub(crate) fn update_event_embedding(
        &self,
        event_id: &str,
        purified_content: &str,
        content_vec: &[f32],
        event_time: Option<&str>,
        location: Option<&str>,
    ) -> Result<()> {
        // Normalize incomplete date formats for TIMESTAMP compatibility
        let normalized_time = event_time.map(|t| {
            let t = t.trim();
            if t.len() == 4 && t.chars().all(|c| c.is_ascii_digit()) {
                format!("{t}-01-01")
            } else if t.len() == 7 && t.as_bytes().get(4) == Some(&b'-') {
                format!("{t}-01")
            } else {
                t.to_string()
            }
        });
        let emb_literal = self.format_embedding(content_vec, self.config.embedding_dims)?;
        let sql = format!(
            r#"UPDATE events SET
                purified_content = $1,
                content_vec = {emb_literal},
                purified = true,
                event_time = $2,
                location = $3
               WHERE event_id = $4"#
        );
        self.backend.execute(
            &sql,
            &[
                SqlParam::Text(purified_content.to_string()),
                opt_text(normalized_time.as_deref()),
                opt_text(location),
                SqlParam::Text(event_id.to_string()),
            ],
        )?;
        // Sync vec_events virtual table (delete + insert since vec0 doesn't support REPLACE)
        if !content_vec.is_empty() {
            if let Some(del_sql) = self.dialect().vec0_event_delete_sql() {
                let _ = self.backend.execute(
                    del_sql,
                    &[SqlParam::Text(event_id.to_string())],
                );
            }
            if let Some(vec0_sql) = self.dialect().vec0_event_insert_sql("$1", &emb_literal) {
                let user_id = self.backend.query_one(
                    "SELECT user_id FROM events WHERE event_id = $1",
                    &[SqlParam::Text(event_id.to_string())],
                    |row| row.get_string(0),
                )?;
                if let Some(uid) = user_id {
                    self.backend.execute(
                        &vec0_sql,
                        &[SqlParam::Text(event_id.to_string()), SqlParam::Text(uid)],
                    )?;
                }
            }
        }
        // Incrementally update events FTS index
        self.fts_insert_event(event_id, purified_content);
        Ok(())
    }

    /// List events with filters.
    pub(crate) fn list_events(&self, options: &ListEventsOptions) -> Result<Vec<Event>> {
        let mut conditions = vec!["user_id = $1".to_string()];
        let mut dynamic_params: Vec<SqlParam> = vec![SqlParam::Text(options.user_id.clone())];
        let mut param_idx: usize = 1;

        if let Some(ref sid) = options.source_id {
            param_idx += 1;
            conditions.push(format!("source_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(sid.clone()));
        }
        if let Some(ref sess) = options.session_id {
            param_idx += 1;
            conditions.push(format!("session_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(sess.clone()));
        }
        if let Some(ref since) = options.since {
            param_idx += 1;
            conditions.push(format!("timestamp >= ${param_idx}"));
            dynamic_params.push(SqlParam::Text(since.clone()));
        }
        if let Some(ref until) = options.until {
            param_idx += 1;
            conditions.push(format!("timestamp <= ${param_idx}"));
            dynamic_params.push(SqlParam::Text(until.clone()));
        }
        if let Some(processed) = options.processed {
            conditions.push(format!("processed = {processed}"));
        }
        let _ = param_idx;

        let limit = options.limit.unwrap_or(100);
        let where_clause = conditions.join(" AND ");
        let cols = event_cols();
        let sql = format!(
            "SELECT {cols} FROM events WHERE {where_clause} ORDER BY timestamp DESC LIMIT {limit}"
        );

        self.backend
            .query_read(&sql, &dynamic_params, |row| map_event_row(row))
    }

    /// Get a single event by ID.
    pub(crate) fn get_event(&self, event_id: &str) -> Result<Option<Event>> {
        self.backend.query_one(
            &format!("SELECT {} FROM events WHERE event_id = $1", event_cols()),
            &[SqlParam::Text(event_id.to_string())],
            |row| map_event_row(row),
        )
    }

    /// Mark events as processed (batch UPDATE with WHERE IN).
    pub(crate) fn mark_events_processed(&self, event_ids: &[&str]) -> Result<()> {
        if event_ids.is_empty() {
            return Ok(());
        }
        let placeholders: Vec<String> = (1..=event_ids.len()).map(|i| format!("${i}")).collect();
        let sql = format!(
            "UPDATE events SET processed = true, processed_at = current_timestamp WHERE event_id IN ({})",
            placeholders.join(", ")
        );
        let param_vals: Vec<SqlParam> = event_ids
            .iter()
            .map(|id| SqlParam::Text(id.to_string()))
            .collect();
        self.backend.execute(&sql, &param_vals)?;
        Ok(())
    }

    /// Get events by a list of IDs, preserving insertion order.
    pub(crate) fn get_events_by_ids(&self, event_ids: &[String]) -> Result<Vec<Event>> {
        if event_ids.is_empty() {
            return Ok(Vec::new());
        }
        // Build IN clause with positional params
        let placeholders: Vec<String> = (1..=event_ids.len()).map(|i| format!("${i}")).collect();
        let sql = format!(
            "SELECT {} FROM events WHERE event_id IN ({}) ORDER BY timestamp ASC",
            event_cols(),
            placeholders.join(", ")
        );
        let param_vals: Vec<SqlParam> = event_ids
            .iter()
            .map(|id| SqlParam::Text(id.clone()))
            .collect();
        self.backend
            .query_read(&sql, &param_vals, |row| map_event_row(row))
    }

    /// Search events by vector similarity within a specific session/episode.
    /// `event_ids` constrains the search to those events only.
    #[allow(dead_code)] // planned API: event vector search
    pub(crate) fn search_events_by_vector(
        &self,
        query_vec: &[f32],
        event_ids: &[String],
        limit: usize,
    ) -> Result<Vec<Event>> {
        if event_ids.is_empty() {
            return Ok(Vec::new());
        }
        let emb_literal = self.format_embedding(query_vec, self.config.embedding_dims)?;
        let distance_expr = self
            .dialect()
            .cosine_distance_expr("content_vec", &emb_literal);
        let placeholders: Vec<String> = (1..=event_ids.len()).map(|i| format!("${i}")).collect();
        let cols = event_cols();
        let sql = format!(
            r#"SELECT {cols},
                      {distance_expr} AS distance
               FROM events
               WHERE event_id IN ({})
               ORDER BY distance ASC
               LIMIT {limit}"#,
            placeholders.join(", ")
        );
        let param_vals: Vec<SqlParam> = event_ids
            .iter()
            .map(|id| SqlParam::Text(id.clone()))
            .collect();
        self.backend
            .query_read(&sql, &param_vals, |row| map_event_row(row))
    }

    /// Search events by vector similarity for a user (no event_id constraint).
    /// Returns (Event, distance) tuples sorted by ascending cosine distance.
    #[allow(dead_code)] // kept for direct event search use cases
    pub(crate) fn search_events_by_vector_for_user(
        &self,
        query_vec: &[f32],
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<(Event, f32)>> {
        let emb_literal = self.format_embedding(query_vec, self.config.embedding_dims)?;
        let distance_expr = self
            .dialect()
            .cosine_distance_expr("content_vec", &emb_literal);
        let cols = event_cols();
        let sql = format!(
            r#"SELECT {cols},
                      {distance_expr} AS distance
               FROM events
               WHERE user_id = $1
                 AND content_vec IS NOT NULL
               ORDER BY distance ASC
               LIMIT {limit}"#,
        );
        self.backend
            .query_read(&sql, &[SqlParam::Text(user_id.to_string())], |row| {
                let event = map_event_row(row)?;
                let distance: f32 = row.get_f64(15).unwrap_or(2.0) as f32;
                Ok((event, distance))
            })
    }

    /// FTS (BM25) search on events, constrained to specific event IDs.
    #[allow(dead_code)] // planned API: event FTS search
    pub(crate) fn fts_search_events(
        &self,
        query: &str,
        event_ids: &[String],
        limit: usize,
    ) -> Result<Vec<Event>> {
        if event_ids.is_empty() {
            return Ok(Vec::new());
        }
        let fts_score = self
            .dialect()
            .fts_match_score_expr("events", "e.event_id", "$1");
        let placeholders: Vec<String> =
            (2..=event_ids.len() + 1).map(|i| format!("${i}")).collect();
        let cols = event_cols();
        let sql = format!(
            r#"SELECT {cols}
               FROM events e
               WHERE {fts_score} IS NOT NULL
                 AND e.event_id IN ({})
               ORDER BY {fts_score} DESC
               LIMIT {limit}"#,
            placeholders.join(", ")
        );
        let mut param_vals: Vec<SqlParam> = vec![SqlParam::Text(query.to_string())];
        for id in event_ids {
            param_vals.push(SqlParam::Text(id.clone()));
        }
        self.backend
            .query_read(&sql, &param_vals, |row| map_event_row(row))
    }

    /// Count unprocessed events for a user.
    #[allow(dead_code)] // planned API: event processing status
    pub(crate) fn count_unprocessed_events(&self, user_id: &str) -> Result<u64> {
        let count = self.backend.query_count(
            "SELECT COUNT(*) FROM events WHERE user_id = $1 AND processed = false",
            &[SqlParam::Text(user_id.to_string())],
        )?;
        Ok(count as u64)
    }

    /// Count unprocessed events in a specific session.
    pub(crate) fn count_unprocessed_events_in_session(&self, session_id: &str) -> Result<u64> {
        let count = self.backend.query_count(
            "SELECT COUNT(*) FROM events WHERE session_id = $1 AND processed = false",
            &[SqlParam::Text(session_id.to_string())],
        )?;
        Ok(count as u64)
    }

    /// Get all unprocessed events in a session, ordered by timestamp.
    pub(crate) fn get_unprocessed_events_in_session(&self, session_id: &str) -> Result<Vec<Event>> {
        let cols = event_cols();
        let sql = format!(
            "SELECT {cols} FROM events WHERE session_id = $1 AND processed = false ORDER BY timestamp ASC"
        );
        self.backend
            .query_read(&sql, &[SqlParam::Text(session_id.to_string())], |row| {
                map_event_row(row)
            })
    }

    /// Reset all events in a session to unprocessed (for re-extraction).
    pub(crate) fn reset_events_processed(&self, session_id: &str) -> Result<()> {
        self.backend.execute(
            "UPDATE events SET processed = false, processed_at = NULL WHERE session_id = $1",
            &[SqlParam::Text(session_id.to_string())],
        )?;
        Ok(())
    }

    /// Delete all events for a user.
    #[allow(dead_code)] // planned API: user data cleanup
    pub(crate) fn delete_user_events(&self, user_id: &str) -> Result<()> {
        self.backend.execute(
            "DELETE FROM events WHERE user_id = $1",
            &[SqlParam::Text(user_id.to_string())],
        )?;
        Ok(())
    }

    /// Get all events in a session ordered by timestamp (oldest first).
    /// Used by get_session_context for token budget allocation.
    pub(crate) fn get_session_events_ordered(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<Event>> {
        let cols = event_cols();
        let sql = format!(
            "SELECT {cols} FROM events WHERE session_id = $1 ORDER BY timestamp ASC LIMIT $2"
        );
        self.backend.query_read(
            &sql,
            &[
                SqlParam::Text(session_id.to_string()),
                SqlParam::Int(limit as i64),
            ],
            |row| map_event_row(row),
        )
    }

    /// Get the episode summary for a session (from memories table, resolution=Narrative).
    pub(crate) fn get_session_episode_summary(&self, session_id: &str) -> Result<Option<String>> {
        self.backend.query_one(
            r#"SELECT metadata->>'$.summary' as summary
               FROM memories
               WHERE session_id = $1 AND resolution = 'Narrative'
               ORDER BY created_at DESC
               LIMIT 1"#,
            &[SqlParam::Text(session_id.to_string())],
            |row| row.get_string(0),
        )
    }
}

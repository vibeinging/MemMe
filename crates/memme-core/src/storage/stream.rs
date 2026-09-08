use crate::error::{MemoryError, Result};
use crate::types::{Event, EventType, IngestEventOptions, ListEventsOptions, Source, SqlParam};

use super::backend::RowAccess;
use super::util::opt_text;
use super::Storage;

/// Generate event SELECT columns.
fn event_cols() -> &'static str {
    "event_id, source_id, session_id, timestamp, event_type, content, parent_id, metadata, user_id, processed, processed_at, purified_content, purified, event_time, location, agent_id, app_id, run_id"
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
        agent_id: row.get_opt_string(15)?,
        app_id: row.get_opt_string(16)?,
        run_id: row.get_opt_string(17)?,
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
        user_id: row.get_opt_string(5)?,
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
            "SELECT source_id, source_type, name, registered_at, metadata, user_id FROM sources WHERE source_id = $1",
            &[SqlParam::Text(source_id.to_string())],
            |row| map_source_row(row),
        )
    }

    /// List all sources for a user.
    #[allow(dead_code)] // planned API: stream source management
    pub(crate) fn list_sources(&self, user_id: &str) -> Result<Vec<Source>> {
        self.backend.query_read(
            "SELECT source_id, source_type, name, registered_at, metadata, user_id FROM sources WHERE user_id = $1 ORDER BY registered_at DESC",
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
        self.insert_event_idempotent(event_id, content, content_vec, options)
            .map(|_| ())
    }

    /// Insert an event, or accept an exact replay of an existing event.
    ///
    /// Returns `true` when a row was inserted and `false` for an exact replay.
    /// Reusing an event ID with a different payload is rejected.
    pub(crate) fn insert_event_idempotent(
        &self,
        event_id: &str,
        content: &str,
        content_vec: &[f32],
        options: &IngestEventOptions,
    ) -> Result<bool> {
        if !content_vec.is_empty() && content_vec.len() != self.config.embedding_dims {
            return Err(MemoryError::Config(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dims,
                content_vec.len()
            )));
        }
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
        let agent_id = options
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("agent_id"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let agent_val = opt_text(agent_id.as_deref());
        let app_id = options
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("app_id"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let app_val = opt_text(app_id.as_deref());
        let run_id = options
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("run_id"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let run_val = opt_text(run_id.as_deref());
        let meta_val = opt_text(
            options
                .metadata
                .as_ref()
                .map(|m| serde_json::to_string(m).unwrap_or_default()),
        );

        let now_expr = self.dialect().current_timestamp_expr();
        let sql = format!(
            r#"INSERT INTO events (event_id, source_id, session_id, timestamp, event_type, content, content_vec, parent_id, metadata, user_id, agent_id, app_id, run_id)
               VALUES ($1, $2, $3, CASE WHEN $4 IS NULL THEN {now_expr} ELSE $4 END, $5, $6, {emb_literal}, $7, $8, $9, $10, $11, $12)"#
        );
        let event_params = vec![
            SqlParam::Text(event_id.to_string()),
            source_val,
            session_val,
            timestamp_val,
            SqlParam::Text(event_type.to_string()),
            SqlParam::Text(content.to_string()),
            parent_val,
            meta_val,
            SqlParam::Text(options.user_id.clone()),
            agent_val,
            app_val,
            run_val,
        ];
        let vector_sql = if content_vec.is_empty() {
            None
        } else {
            self.dialect().vector_event_insert_sql("$1", &emb_literal)
        };
        self.backend.transaction(|tx| {
            let existing = tx.query_one(
                "SELECT source_id, session_id, timestamp, event_type, content, parent_id, metadata, user_id FROM events WHERE event_id = $1",
                &[SqlParam::Text(event_id.to_string())],
                |row| {
                    Ok((
                        row.get_opt_string(0)?,
                        row.get_opt_string(1)?,
                        row.get_string(2)?,
                        row.get_string(3)?,
                        row.get_string(4)?,
                        row.get_opt_string(5)?,
                        row.get_opt_string(6)?,
                        row.get_string(7)?,
                    ))
                },
            )?;
            if let Some((
                source_id,
                session_id,
                timestamp,
                stored_event_type,
                stored_content,
                parent_id,
                stored_metadata,
                stored_user_id,
            )) = existing
            {
                let stored_metadata = stored_metadata
                    .as_deref()
                    .map(serde_json::from_str::<serde_json::Value>)
                    .transpose()?;
                let timestamp_matches = match options.timestamp.as_deref() {
                    Some(expected) => expected == timestamp,
                    None => true,
                };
                let exact_replay = source_id == options.source_id
                    && session_id == options.session_id
                    && timestamp_matches
                    && stored_event_type == event_type
                    && stored_content == content
                    && parent_id == options.parent_id
                    && stored_metadata == options.metadata
                    && stored_user_id == options.user_id;
                if exact_replay {
                    return Ok(false);
                }
                return Err(MemoryError::Config(format!(
                    "event_id '{event_id}' already exists with a different payload"
                )));
            }

            tx.execute(&sql, &event_params)?;
            if let Some(vector_sql) = vector_sql.as_deref() {
                tx.execute(
                    vector_sql,
                    &[
                        SqlParam::Text(event_id.to_string()),
                        SqlParam::Text(options.user_id.clone()),
                        opt_text(agent_id.as_deref()),
                    ],
                )?;
            }
            if tx.table_exists("events_fts")? {
                tx.execute(
                    "INSERT OR REPLACE INTO events_fts(event_id, content) VALUES ($1, $2)",
                    &[
                        SqlParam::Text(event_id.to_string()),
                        SqlParam::Text(content.to_string()),
                    ],
                )?;
            }
            Ok(true)
        })
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
        if content_vec.len() != self.config.embedding_dims {
            return Err(MemoryError::Config(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dims,
                content_vec.len()
            )));
        }
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
        let update_params = vec![
            SqlParam::Text(purified_content.to_string()),
            opt_text(normalized_time.as_deref()),
            opt_text(location),
            SqlParam::Text(event_id.to_string()),
        ];
        let delete_sql = self.dialect().vector_event_delete_sql();
        let insert_sql = self.dialect().vector_event_insert_sql("$1", &emb_literal);
        self.backend.transaction(|tx| {
            let (user_id, agent_id) = tx
                .query_one(
                    "SELECT user_id, agent_id FROM events WHERE event_id = $1",
                    &[SqlParam::Text(event_id.to_string())],
                    |row| Ok((row.get_string(0)?, row.get_opt_string(1)?)),
                )?
                .ok_or_else(|| MemoryError::NotFound(event_id.to_string()))?;

            tx.execute(&sql, &update_params)?;
            if let Some(delete_sql) = delete_sql {
                tx.execute(delete_sql, &[SqlParam::Text(event_id.to_string())])?;
            }
            if let Some(insert_sql) = insert_sql.as_deref() {
                tx.execute(
                    insert_sql,
                    &[
                        SqlParam::Text(event_id.to_string()),
                        SqlParam::Text(user_id),
                        opt_text(agent_id.as_deref()),
                    ],
                )?;
            }
            if tx.table_exists("events_fts")? {
                tx.execute(
                    "INSERT OR REPLACE INTO events_fts(event_id, content) VALUES ($1, $2)",
                    &[
                        SqlParam::Text(event_id.to_string()),
                        SqlParam::Text(purified_content.to_string()),
                    ],
                )?;
            }
            Ok(())
        })
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

    pub(crate) fn event_has_embedding(&self, event_id: &str) -> Result<bool> {
        self.backend
            .query_one(
                "SELECT content_vec IS NOT NULL FROM events WHERE event_id = $1",
                &[SqlParam::Text(event_id.to_string())],
                |row| row.get_bool(0),
            )?
            .ok_or_else(|| MemoryError::NotFound(event_id.to_string()))
    }

    /// Backfill only the raw event embedding. This leaves purification and
    /// lifecycle fields unchanged, so an exact event replay remains harmless.
    pub(crate) fn backfill_event_embedding(
        &self,
        event_id: &str,
        content_vec: &[f32],
    ) -> Result<()> {
        if content_vec.len() != self.config.embedding_dims {
            return Err(MemoryError::Config(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dims,
                content_vec.len()
            )));
        }
        let emb_literal = self.format_embedding(content_vec, self.config.embedding_dims)?;
        let delete_sql = self.dialect().vector_event_delete_sql();
        let insert_sql = self.dialect().vector_event_insert_sql("$1", &emb_literal);
        self.backend.transaction(|tx| {
            let (user_id, agent_id) = tx
                .query_one(
                    "SELECT user_id, agent_id FROM events WHERE event_id = $1",
                    &[SqlParam::Text(event_id.to_string())],
                    |row| Ok((row.get_string(0)?, row.get_opt_string(1)?)),
                )?
                .ok_or_else(|| MemoryError::NotFound(event_id.to_string()))?;
            tx.execute(
                &format!("UPDATE events SET content_vec = {emb_literal} WHERE event_id = $1"),
                &[SqlParam::Text(event_id.to_string())],
            )?;
            if let Some(delete_sql) = delete_sql {
                tx.execute(delete_sql, &[SqlParam::Text(event_id.to_string())])?;
            }
            if let Some(insert_sql) = insert_sql.as_deref() {
                tx.execute(
                    insert_sql,
                    &[
                        SqlParam::Text(event_id.to_string()),
                        SqlParam::Text(user_id),
                        opt_text(agent_id.as_deref()),
                    ],
                )?;
            }
            Ok(())
        })
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

    /// Get one user's events by a list of IDs, preserving timestamp order.
    pub(crate) fn get_events_by_ids_for_user(
        &self,
        event_ids: &[String],
        user_id: &str,
    ) -> Result<Vec<Event>> {
        if event_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders: Vec<String> =
            (2..=event_ids.len() + 1).map(|i| format!("${i}")).collect();
        let sql = format!(
            "SELECT {} FROM events WHERE user_id = $1 AND event_id IN ({}) ORDER BY timestamp ASC",
            event_cols(),
            placeholders.join(", ")
        );
        let mut param_vals = Vec::with_capacity(event_ids.len() + 1);
        param_vals.push(SqlParam::Text(user_id.to_string()));
        param_vals.extend(event_ids.iter().cloned().map(SqlParam::Text));
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
                let distance: f32 = row.get_f64(18).unwrap_or(2.0) as f32;
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
        let vector_table = self.dialect().event_vector_table_name();
        self.backend.transaction(|tx| {
            let params = &[SqlParam::Text(user_id.to_string())];
            tx.execute(
                &format!(
                    "DELETE FROM {vector_table} WHERE rowid IN (\
                     SELECT rowid FROM {vector_table}_vectors WHERE user_id = $1)"
                ),
                params,
            )?;
            if tx.table_exists("events_fts")? {
                tx.execute(
                    "DELETE FROM events_fts WHERE event_id IN (\
                     SELECT event_id FROM events WHERE user_id = $1)",
                    params,
                )?;
            }
            tx.execute("DELETE FROM events WHERE user_id = $1", params)?;
            Ok(())
        })
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

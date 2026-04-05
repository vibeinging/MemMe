use duckdb::params;

use crate::error::Result;
use crate::types::{Event, EventType, IngestEventOptions, ListEventsOptions, Source};

use super::util::opt_text;
use super::Storage;

/// Standard columns for Event SELECT queries.
const EVENT_COLS: &str = "event_id, source_id, session_id, CAST(timestamp AS VARCHAR), \
    event_type, content, parent_id, metadata, user_id, \
    processed, CAST(processed_at AS VARCHAR), \
    purified_content, purified, CAST(event_time AS VARCHAR), location";

/// Map a DuckDB row to an Event struct. Expects columns in standard order:
/// event_id(0), source_id(1), session_id(2), timestamp(3), event_type(4),
/// content(5), parent_id(6), metadata(7), user_id(8), processed(9), processed_at(10),
/// purified_content(11), purified(12), event_time(13), location(14)
fn map_event_row(row: &duckdb::Row<'_>) -> duckdb::Result<Event> {
    Ok(Event {
        event_id: row.get(0)?,
        source_id: row.get::<_, Option<String>>(1)?,
        session_id: row.get::<_, Option<String>>(2)?,
        timestamp: row.get::<_, String>(3)?,
        event_type: EventType::parse(&row.get::<_, Option<String>>(4)?.unwrap_or_default()),
        content: row.get(5)?,
        parent_id: row.get::<_, Option<String>>(6)?,
        metadata: row
            .get::<_, Option<String>>(7)?
            .and_then(|s| serde_json::from_str(&s).ok()),
        user_id: row.get(8)?,
        processed: row.get::<_, Option<bool>>(9)?.unwrap_or(false),
        processed_at: row.get::<_, Option<String>>(10)?,
        purified_content: row.get::<_, Option<String>>(11)?,
        purified: row.get::<_, Option<bool>>(12)?.unwrap_or(false),
        event_time: row.get::<_, Option<String>>(13)?,
        location: row.get::<_, Option<String>>(14)?,
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
        let conn = self.write_conn();
        conn.execute(
            "INSERT INTO sources (source_id, source_type, name, metadata, user_id) VALUES ($1, $2, $3, $4, $5)",
            params![source_id, source_type, name_val, meta_val, user_id],
        )?;
        Ok(())
    }

    /// Get a source by ID.
    #[allow(dead_code)] // planned API: stream source management
    pub(crate) fn get_source(&self, source_id: &str) -> Result<Option<Source>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT source_id, source_type, name, CAST(registered_at AS VARCHAR), metadata FROM sources WHERE source_id = $1",
        )?;
        let mut rows = stmt.query_map(params![source_id], |row| {
            Ok(Source {
                source_id: row.get(0)?,
                source_type: row.get(1)?,
                name: row.get::<_, Option<String>>(2)?,
                registered_at: row.get::<_, String>(3)?,
                metadata: row
                    .get::<_, Option<String>>(4)?
                    .and_then(|s| serde_json::from_str(&s).ok()),
            })
        })?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// List all sources for a user.
    #[allow(dead_code)] // planned API: stream source management
    pub(crate) fn list_sources(&self, user_id: &str) -> Result<Vec<Source>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT source_id, source_type, name, CAST(registered_at AS VARCHAR), metadata FROM sources WHERE user_id = $1 ORDER BY registered_at DESC",
        )?;
        let rows = stmt
            .query_map(params![user_id], |row| {
                Ok(Source {
                    source_id: row.get(0)?,
                    source_type: row.get(1)?,
                    name: row.get::<_, Option<String>>(2)?,
                    registered_at: row.get::<_, String>(3)?,
                    metadata: row
                        .get::<_, Option<String>>(4)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
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
            Self::format_embedding(content_vec, self.config.embedding_dims)?
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

        let sql = format!(
            r#"INSERT INTO events (event_id, source_id, session_id, timestamp, event_type, content, content_vec, parent_id, metadata, user_id)
               VALUES ($1, $2, $3, CASE WHEN $4 IS NULL THEN current_timestamp ELSE CAST($4 AS TIMESTAMP) END, $5, $6, {emb_literal}, $7, $8, $9)"#
        );
        let conn = self.write_conn();
        conn.execute(
            &sql,
            params![
                event_id,
                source_val,
                session_val,
                timestamp_val,
                event_type,
                content,
                parent_val,
                meta_val,
                &options.user_id
            ],
        )?;
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
        // Normalize incomplete date formats that DuckDB can't parse as TIMESTAMP
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
        let emb_literal = Self::format_embedding(content_vec, self.config.embedding_dims)?;
        let sql = format!(
            r#"UPDATE events SET
                purified_content = $1,
                content_vec = {emb_literal},
                purified = true,
                event_time = $2,
                location = $3
               WHERE event_id = $4"#
        );
        let conn = self.write_conn();
        conn.execute(
            &sql,
            params![
                purified_content,
                normalized_time.as_deref(),
                location,
                event_id
            ],
        )?;
        Ok(())
    }

    /// List events with filters.
    pub(crate) fn list_events(&self, options: &ListEventsOptions) -> Result<Vec<Event>> {
        let mut conditions = vec!["user_id = $1".to_string()];
        let mut dynamic_params: Vec<duckdb::types::Value> =
            vec![duckdb::types::Value::Text(options.user_id.clone())];
        let mut param_idx: usize = 1;

        if let Some(ref sid) = options.source_id {
            param_idx += 1;
            conditions.push(format!("source_id = ${param_idx}"));
            dynamic_params.push(duckdb::types::Value::Text(sid.clone()));
        }
        if let Some(ref sess) = options.session_id {
            param_idx += 1;
            conditions.push(format!("session_id = ${param_idx}"));
            dynamic_params.push(duckdb::types::Value::Text(sess.clone()));
        }
        if let Some(ref since) = options.since {
            param_idx += 1;
            conditions.push(format!("timestamp >= CAST(${param_idx} AS TIMESTAMP)"));
            dynamic_params.push(duckdb::types::Value::Text(since.clone()));
        }
        if let Some(ref until) = options.until {
            param_idx += 1;
            conditions.push(format!("timestamp <= CAST(${param_idx} AS TIMESTAMP)"));
            dynamic_params.push(duckdb::types::Value::Text(until.clone()));
        }
        if let Some(processed) = options.processed {
            conditions.push(format!("processed = {processed}"));
        }
        let _ = param_idx;

        let limit = options.limit.unwrap_or(100);
        let where_clause = conditions.join(" AND ");
        let sql = format!(
            "SELECT {EVENT_COLS} FROM events WHERE {where_clause} ORDER BY timestamp DESC LIMIT {limit}"
        );

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn duckdb::ToSql> = dynamic_params
            .iter()
            .map(|p| p as &dyn duckdb::ToSql)
            .collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), map_event_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get a single event by ID.
    pub(crate) fn get_event(&self, event_id: &str) -> Result<Option<Event>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&format!(
            "SELECT {EVENT_COLS} FROM events WHERE event_id = $1"
        ))?;
        let mut rows = stmt.query_map(params![event_id], map_event_row)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
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
        let conn = self.write_conn();
        let param_vals: Vec<duckdb::types::Value> = event_ids
            .iter()
            .map(|id| duckdb::types::Value::Text(id.to_string()))
            .collect();
        let param_refs: Vec<&dyn duckdb::ToSql> =
            param_vals.iter().map(|p| p as &dyn duckdb::ToSql).collect();
        conn.execute(&sql, param_refs.as_slice())?;
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
            "SELECT {EVENT_COLS} FROM events WHERE event_id IN ({}) ORDER BY timestamp ASC",
            placeholders.join(", ")
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let param_vals: Vec<duckdb::types::Value> = event_ids
            .iter()
            .map(|id| duckdb::types::Value::Text(id.clone()))
            .collect();
        let param_refs: Vec<&dyn duckdb::ToSql> =
            param_vals.iter().map(|p| p as &dyn duckdb::ToSql).collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), map_event_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
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
        let emb_literal = Self::format_embedding(query_vec, self.config.embedding_dims)?;
        let placeholders: Vec<String> = (1..=event_ids.len()).map(|i| format!("${i}")).collect();
        let sql = format!(
            r#"SELECT {EVENT_COLS},
                      array_cosine_distance(content_vec, {emb_literal}) AS distance
               FROM events
               WHERE event_id IN ({})
               ORDER BY distance ASC
               LIMIT {limit}"#,
            placeholders.join(", ")
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let param_vals: Vec<duckdb::types::Value> = event_ids
            .iter()
            .map(|id| duckdb::types::Value::Text(id.clone()))
            .collect();
        let param_refs: Vec<&dyn duckdb::ToSql> =
            param_vals.iter().map(|p| p as &dyn duckdb::ToSql).collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), map_event_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
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
        let placeholders: Vec<String> =
            (2..=event_ids.len() + 1).map(|i| format!("${i}")).collect();
        let sql = format!(
            r#"SELECT {EVENT_COLS}
               FROM events e
               WHERE fts_main_events.match_bm25(e.event_id, $1) IS NOT NULL
                 AND e.event_id IN ({})
               ORDER BY fts_main_events.match_bm25(e.event_id, $1) DESC
               LIMIT {limit}"#,
            placeholders.join(", ")
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let mut param_vals: Vec<duckdb::types::Value> =
            vec![duckdb::types::Value::Text(query.to_string())];
        for id in event_ids {
            param_vals.push(duckdb::types::Value::Text(id.clone()));
        }
        let param_refs: Vec<&dyn duckdb::ToSql> =
            param_vals.iter().map(|p| p as &dyn duckdb::ToSql).collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), map_event_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Count unprocessed events for a user.
    #[allow(dead_code)] // planned API: event processing status
    pub(crate) fn count_unprocessed_events(&self, user_id: &str) -> Result<u64> {
        let conn = self.read_conn();
        let mut stmt =
            conn.prepare("SELECT COUNT(*) FROM events WHERE user_id = $1 AND processed = false")?;
        let count: i64 = stmt
            .query_map(params![user_id], |row| row.get(0))?
            .next()
            .expect("COUNT always returns a row")?;
        Ok(count as u64)
    }

    /// Count unprocessed events in a specific session.
    pub(crate) fn count_unprocessed_events_in_session(&self, session_id: &str) -> Result<u64> {
        let conn = self.read_conn();
        let mut stmt = conn
            .prepare("SELECT COUNT(*) FROM events WHERE session_id = $1 AND processed = false")?;
        let count: i64 = stmt
            .query_map(params![session_id], |row| row.get(0))?
            .next()
            .expect("COUNT always returns a row")?;
        Ok(count as u64)
    }

    /// Get all unprocessed events in a session, ordered by timestamp.
    pub(crate) fn get_unprocessed_events_in_session(&self, session_id: &str) -> Result<Vec<Event>> {
        let sql = format!(
            "SELECT {EVENT_COLS} FROM events WHERE session_id = $1 AND processed = false ORDER BY timestamp ASC"
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![session_id], map_event_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Reset all events in a session to unprocessed (for re-extraction).
    pub(crate) fn reset_events_processed(&self, session_id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            "UPDATE events SET processed = false, processed_at = NULL WHERE session_id = $1",
            params![session_id],
        )?;
        Ok(())
    }

    /// Delete all events for a user.
    #[allow(dead_code)] // planned API: user data cleanup
    pub(crate) fn delete_user_events(&self, user_id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute("DELETE FROM events WHERE user_id = $1", params![user_id])?;
        Ok(())
    }

    /// Get all events in a session ordered by timestamp (oldest first).
    /// Used by get_session_context for token budget allocation.
    pub(crate) fn get_session_events_ordered(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<Event>> {
        let sql = format!(
            "SELECT {EVENT_COLS} FROM events WHERE session_id = $1 ORDER BY timestamp ASC LIMIT $2"
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![session_id, limit as i64], map_event_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get the episode summary for a session (from memories table, resolution=Narrative).
    pub(crate) fn get_session_episode_summary(&self, session_id: &str) -> Result<Option<String>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            r#"SELECT metadata->>'$.summary' as summary
               FROM memories
               WHERE session_id = $1 AND resolution = 'Narrative'
               ORDER BY created_at DESC
               LIMIT 1"#,
        )?;
        let mut rows = stmt.query_map(params![session_id], |row| row.get(0))?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }
}

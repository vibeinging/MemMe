use crate::error::Result;
use crate::types::{ListSessionsOptions, Session, SqlParam};

use super::backend::RowAccess;
use super::util::opt_text;
use super::Storage;

impl Storage {
    /// Insert a new session.
    pub(crate) fn insert_session(
        &self,
        session_id: &str,
        user_id: &str,
        source_id: Option<&str>,
        started_at: &str,
        metadata: Option<&str>,
    ) -> Result<()> {
        let source_val = opt_text(source_id);
        let meta_val = opt_text(metadata);
        self.backend.execute(
            r#"INSERT INTO sessions (session_id, user_id, source_id, started_at, metadata)
               VALUES ($1, $2, $3, $4, $5)
               ON CONFLICT DO NOTHING"#,
            &[
                SqlParam::Text(session_id.to_string()),
                SqlParam::Text(user_id.to_string()),
                source_val,
                SqlParam::Text(started_at.to_string()),
                meta_val,
            ],
        )?;
        Ok(())
    }

    /// Get a session by ID, with event_count computed from the events table.
    pub(crate) fn get_session(&self, session_id: &str) -> Result<Option<Session>> {
        self.backend.query_one(
            r#"SELECT s.session_id, s.user_id, s.source_id,
                      s.started_at, s.ended_at,
                      s.metadata, s.created_at,
                      (SELECT COUNT(*) FROM events WHERE session_id = s.session_id) AS event_count,
                      s.structured_notes
               FROM sessions s
               WHERE s.session_id = $1"#,
            &[SqlParam::Text(session_id.to_string())],
            |row| map_session_row(row),
        )
    }

    /// List sessions with filters and pagination.
    pub(crate) fn list_sessions(&self, options: &ListSessionsOptions) -> Result<Vec<Session>> {
        let mut conditions = vec!["s.user_id = $1".to_string()];
        let mut dynamic_params: Vec<SqlParam> = vec![SqlParam::Text(options.user_id.clone())];
        let mut param_idx: usize = 1;

        if let Some(ref source_id) = options.source_id {
            param_idx += 1;
            conditions.push(format!("s.source_id = ${param_idx}"));
            dynamic_params.push(SqlParam::Text(source_id.clone()));
        }
        if let Some(ref since) = options.since {
            param_idx += 1;
            conditions.push(format!("s.started_at >= ${param_idx}"));
            dynamic_params.push(SqlParam::Text(since.clone()));
        }
        if let Some(ref until) = options.until {
            param_idx += 1;
            conditions.push(format!("s.started_at <= ${param_idx}"));
            dynamic_params.push(SqlParam::Text(until.clone()));
        }
        let _ = param_idx;

        let limit = options.limit.unwrap_or(20);
        let offset = options.offset.unwrap_or(0);
        let where_clause = conditions.join(" AND ");
        let sql = format!(
            r#"SELECT s.session_id, s.user_id, s.source_id,
                      s.started_at, s.ended_at,
                      s.metadata, s.created_at,
                      (SELECT COUNT(*) FROM events WHERE session_id = s.session_id) AS event_count,
                      s.structured_notes
               FROM sessions s
               WHERE {where_clause}
               ORDER BY s.started_at DESC
               LIMIT {limit} OFFSET {offset}"#
        );

        self.backend
            .query_read(&sql, &dynamic_params, |row| map_session_row(row))
    }

    /// Set ended_at on a session (close it).
    #[allow(dead_code)] // planned API: session lifecycle
    pub(crate) fn close_session(&self, session_id: &str, ended_at: &str) -> Result<()> {
        self.backend.execute(
            "UPDATE sessions SET ended_at = $1 WHERE session_id = $2",
            &[
                SqlParam::Text(ended_at.to_string()),
                SqlParam::Text(session_id.to_string()),
            ],
        )?;
        Ok(())
    }

    /// Delete a session.
    pub(crate) fn delete_session(&self, session_id: &str) -> Result<()> {
        self.backend.execute(
            "DELETE FROM sessions WHERE session_id = $1",
            &[SqlParam::Text(session_id.to_string())],
        )?;
        Ok(())
    }

    /// Find a session by its ID, creating it if it doesn't exist.
    #[allow(dead_code)] // planned API: session management
    pub(crate) fn get_or_create_session(
        &self,
        session_id: &str,
        user_id: &str,
        source_id: Option<&str>,
        started_at: &str,
        metadata: Option<&str>,
    ) -> Result<Session> {
        if let Some(session) = self.get_session(session_id)? {
            return Ok(session);
        }
        self.insert_session(session_id, user_id, source_id, started_at, metadata)?;
        self.get_session(session_id)?
            .ok_or_else(|| crate::error::MemoryError::Config("Failed to create session".into()))
    }
    /// Append a line to a session's structured notes, capped at 2000 chars.
    pub(crate) fn append_structured_note(&self, session_id: &str, note: &str) -> Result<()> {
        let left = self
            .dialect()
            .left_expr("COALESCE(structured_notes, '') || $1", "2000");
        self.backend.execute(
            &format!(
                r#"UPDATE sessions
               SET structured_notes = {left}
               WHERE session_id = $2"#
            ),
            &[
                SqlParam::Text(note.to_string()),
                SqlParam::Text(session_id.to_string()),
            ],
        )?;
        Ok(())
    }

    /// Clear structured notes for a session (after compact).
    pub(crate) fn clear_structured_notes(&self, session_id: &str) -> Result<()> {
        self.backend.execute(
            "UPDATE sessions SET structured_notes = NULL WHERE session_id = $1",
            &[SqlParam::Text(session_id.to_string())],
        )?;
        Ok(())
    }

    /// Increment the queried_count for a session and update last_queried_at.
    /// Used by feedback-driven consolidation to prioritize sessions that
    /// have been hit by search queries during meditation.
    pub(crate) fn mark_session_queried(&self, session_id: &str) -> Result<()> {
        let now = self.dialect().current_timestamp_expr();
        self.backend.execute(
            &format!(
                "UPDATE sessions SET queried_count = COALESCE(queried_count, 0) + 1, \
                 last_queried_at = {now} WHERE session_id = $1"
            ),
            &[SqlParam::Text(session_id.to_string())],
        )?;
        Ok(())
    }

    /// Get the queried_count for a session. Returns 0 if session not found.
    pub(crate) fn get_session_queried_count(&self, session_id: &str) -> Result<u32> {
        let count = self.backend.query_one(
            "SELECT COALESCE(queried_count, 0) FROM sessions WHERE session_id = $1",
            &[SqlParam::Text(session_id.to_string())],
            |row| row.get_opt_i64(0),
        )?;
        Ok(count.flatten().unwrap_or(0) as u32)
    }

    /// Check if a session has already been compacted (has a corresponding episode).
    pub(crate) fn session_has_episode(&self, session_id: &str) -> Result<bool> {
        // Episodes store session_ids as a JSON array. Check if any episode
        // contains this session_id in its session_ids column.
        let count = self.backend.query_one(
            r#"SELECT COUNT(*) FROM episodes e, json_each(e.session_ids) j
               WHERE j.value = $1"#,
            &[SqlParam::Text(session_id.to_string())],
            |row| row.get_opt_i64(0),
        )?;
        Ok(count.flatten().unwrap_or(0) > 0)
    }

    /// Get session IDs for a user that have unprocessed events and no episode yet.
    /// These are sessions that need compact before search can find their content effectively.
    pub(crate) fn get_uncompacted_session_ids(&self, user_id: &str) -> Result<Vec<String>> {
        // A session is "uncompacted" if:
        // 1. It belongs to this user
        // 2. It has at least one unprocessed event
        // 3. No episode references it yet
        self.backend.query_read(
            r#"SELECT DISTINCT s.session_id
               FROM sessions s
               JOIN events e ON e.session_id = s.session_id AND e.processed = 0
               WHERE s.user_id = $1
                 AND NOT EXISTS (
                     SELECT 1 FROM episodes ep, json_each(ep.session_ids) j
                     WHERE j.value = s.session_id
                 )
               ORDER BY s.started_at DESC"#,
            &[SqlParam::Text(user_id.to_string())],
            |row| row.get_string(0),
        )
    }

    /// Delete all sessions for a user.
    #[allow(dead_code)] // planned API: user data cleanup
    pub(crate) fn delete_user_sessions(&self, user_id: &str) -> Result<()> {
        self.backend.execute(
            "DELETE FROM sessions WHERE user_id = $1",
            &[SqlParam::Text(user_id.to_string())],
        )?;
        Ok(())
    }
}

fn map_session_row(row: &dyn RowAccess) -> Result<Session> {
    Ok(Session {
        session_id: row.get_string(0)?,
        user_id: row.get_string(1)?,
        source_id: row.get_opt_string(2)?,
        started_at: row.get_string(3)?,
        ended_at: row.get_opt_string(4)?,
        metadata: row
            .get_opt_string(5)?
            .and_then(|s| serde_json::from_str(&s).ok()),
        created_at: row.get_string(6)?,
        event_count: row.get_opt_i64(7)?.unwrap_or(0) as u32,
        structured_notes: row.get_opt_string(8)?,
    })
}

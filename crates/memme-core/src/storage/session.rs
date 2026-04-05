use duckdb::params;

use crate::error::Result;
use crate::types::{ListSessionsOptions, Session};

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
        let conn = self.write_conn();
        conn.execute(
            r#"INSERT INTO sessions (session_id, user_id, source_id, started_at, metadata)
               VALUES ($1, $2, $3, CAST($4 AS TIMESTAMP), $5)
               ON CONFLICT DO NOTHING"#,
            params![session_id, user_id, source_val, started_at, meta_val],
        )?;
        Ok(())
    }

    /// Get a session by ID, with event_count computed from the events table.
    pub(crate) fn get_session(&self, session_id: &str) -> Result<Option<Session>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            r#"SELECT s.session_id, s.user_id, s.source_id,
                      CAST(s.started_at AS VARCHAR), CAST(s.ended_at AS VARCHAR),
                      s.metadata, CAST(s.created_at AS VARCHAR),
                      (SELECT COUNT(*) FROM events WHERE session_id = s.session_id) AS event_count,
                      s.structured_notes
               FROM sessions s
               WHERE s.session_id = $1"#,
        )?;
        let mut rows = stmt.query_map(params![session_id], map_session_row)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// List sessions with filters and pagination.
    pub(crate) fn list_sessions(&self, options: &ListSessionsOptions) -> Result<Vec<Session>> {
        let mut conditions = vec!["s.user_id = $1".to_string()];
        let mut dynamic_params: Vec<duckdb::types::Value> =
            vec![duckdb::types::Value::Text(options.user_id.clone())];
        let mut param_idx: usize = 1;

        if let Some(ref source_id) = options.source_id {
            param_idx += 1;
            conditions.push(format!("s.source_id = ${param_idx}"));
            dynamic_params.push(duckdb::types::Value::Text(source_id.clone()));
        }
        if let Some(ref since) = options.since {
            param_idx += 1;
            conditions.push(format!("s.started_at >= CAST(${param_idx} AS TIMESTAMP)"));
            dynamic_params.push(duckdb::types::Value::Text(since.clone()));
        }
        if let Some(ref until) = options.until {
            param_idx += 1;
            conditions.push(format!("s.started_at <= CAST(${param_idx} AS TIMESTAMP)"));
            dynamic_params.push(duckdb::types::Value::Text(until.clone()));
        }
        let _ = param_idx;

        let limit = options.limit.unwrap_or(20);
        let offset = options.offset.unwrap_or(0);
        let where_clause = conditions.join(" AND ");
        let sql = format!(
            r#"SELECT s.session_id, s.user_id, s.source_id,
                      CAST(s.started_at AS VARCHAR), CAST(s.ended_at AS VARCHAR),
                      s.metadata, CAST(s.created_at AS VARCHAR),
                      (SELECT COUNT(*) FROM events WHERE session_id = s.session_id) AS event_count,
                      s.structured_notes
               FROM sessions s
               WHERE {where_clause}
               ORDER BY s.started_at DESC
               LIMIT {limit} OFFSET {offset}"#
        );

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn duckdb::ToSql> = dynamic_params
            .iter()
            .map(|p| p as &dyn duckdb::ToSql)
            .collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), map_session_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Set ended_at on a session (close it).
    #[allow(dead_code)] // planned API: session lifecycle
    pub(crate) fn close_session(&self, session_id: &str, ended_at: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            "UPDATE sessions SET ended_at = CAST($1 AS TIMESTAMP) WHERE session_id = $2",
            params![ended_at, session_id],
        )?;
        Ok(())
    }

    /// Delete a session.
    pub(crate) fn delete_session(&self, session_id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            "DELETE FROM sessions WHERE session_id = $1",
            params![session_id],
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
        let conn = self.write_conn();
        conn.execute(
            r#"UPDATE sessions
               SET structured_notes = LEFT(COALESCE(structured_notes, '') || $1, 2000)
               WHERE session_id = $2"#,
            params![note, session_id],
        )?;
        Ok(())
    }

    /// Clear structured notes for a session (after compact).
    pub(crate) fn clear_structured_notes(&self, session_id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute(
            "UPDATE sessions SET structured_notes = NULL WHERE session_id = $1",
            params![session_id],
        )?;
        Ok(())
    }

    /// Delete all sessions for a user.
    #[allow(dead_code)] // planned API: user data cleanup
    pub(crate) fn delete_user_sessions(&self, user_id: &str) -> Result<()> {
        let conn = self.write_conn();
        conn.execute("DELETE FROM sessions WHERE user_id = $1", params![user_id])?;
        Ok(())
    }
}

fn map_session_row(row: &duckdb::Row<'_>) -> duckdb::Result<Session> {
    Ok(Session {
        session_id: row.get(0)?,
        user_id: row.get(1)?,
        source_id: row.get::<_, Option<String>>(2)?,
        started_at: row.get::<_, String>(3)?,
        ended_at: row.get::<_, Option<String>>(4)?,
        metadata: row
            .get::<_, Option<String>>(5)?
            .and_then(|s| serde_json::from_str(&s).ok()),
        created_at: row.get::<_, String>(6)?,
        event_count: row.get::<_, Option<i64>>(7)?.unwrap_or(0) as u32,
        structured_notes: row.get::<_, Option<String>>(8)?,
    })
}

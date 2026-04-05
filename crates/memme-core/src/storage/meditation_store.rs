use duckdb::params;

use crate::error::Result;
use crate::types::{MeditationRecord, MeditationStatus};

use super::util::opt_text;
use super::Storage;

impl Storage {
    /// Insert a new meditation record.
    pub(crate) fn insert_meditation(&self, record: &MeditationRecord) -> Result<()> {
        let meta_val = opt_text(
            record
                .metadata
                .as_ref()
                .map(|m| serde_json::to_string(m).unwrap_or_default()),
        );
        let journal_val = opt_text(record.journal.as_deref());
        let finished_val = opt_text(record.finished_at.as_deref());

        let conn = self.write_conn();
        conn.execute(
            r#"INSERT INTO meditations (meditation_id, triggered_by, started_at, finished_at, status, user_id, journal, metadata)
               VALUES ($1, $2, CAST($3 AS TIMESTAMP), CASE WHEN $4 IS NULL THEN NULL ELSE CAST($4 AS TIMESTAMP) END, $5, $6, $7, $8)"#,
            params![
                &record.meditation_id,
                &record.triggered_by,
                &record.started_at,
                finished_val,
                record.status.as_str(),
                &record.user_id,
                journal_val,
                meta_val
            ],
        )?;
        Ok(())
    }

    /// Update a meditation record with stats, status and journal.
    pub(crate) fn update_meditation(
        &self,
        meditation_id: &str,
        status: &str,
        stats: &MeditationRecord,
        journal: Option<&str>,
    ) -> Result<()> {
        let finished_val = opt_text(stats.finished_at.as_deref());
        let journal_val = opt_text(journal);

        let conn = self.write_conn();
        conn.execute(
            r#"UPDATE meditations
               SET status = $1,
                   finished_at = CASE WHEN $2 IS NULL THEN NULL ELSE CAST($2 AS TIMESTAMP) END,
                   events_processed = $3,
                   episodes_created = $4,
                   memories_created = $5,
                   memories_updated = $6,
                   memories_decayed = $7,
                   entities_created = $8,
                   relations_created = $9,
                   conflicts_found = $10,
                   journal = $11
               WHERE meditation_id = $12"#,
            params![
                status,
                finished_val,
                stats.events_processed as i32,
                stats.episodes_created as i32,
                stats.memories_created as i32,
                stats.memories_updated as i32,
                stats.memories_decayed as i32,
                stats.entities_created as i32,
                stats.relations_created as i32,
                stats.conflicts_found as i32,
                journal_val,
                meditation_id
            ],
        )?;
        Ok(())
    }

    /// Get a meditation record by ID.
    #[allow(dead_code)] // planned API: meditation inspection
    pub(crate) fn get_meditation(&self, meditation_id: &str) -> Result<Option<MeditationRecord>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            r#"SELECT meditation_id, triggered_by, CAST(started_at AS VARCHAR),
                      CAST(finished_at AS VARCHAR), status, user_id,
                      events_processed, episodes_created, memories_created,
                      memories_updated, memories_decayed, entities_created,
                      relations_created, conflicts_found, journal, metadata
               FROM meditations WHERE meditation_id = $1"#,
        )?;
        let mut rows = stmt.query_map(params![meditation_id], map_meditation_row)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// List meditation records for a user.
    #[allow(dead_code)] // planned API: meditation history
    pub(crate) fn list_meditations(
        &self,
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<MeditationRecord>> {
        let sql = format!(
            r#"SELECT meditation_id, triggered_by, CAST(started_at AS VARCHAR),
                      CAST(finished_at AS VARCHAR), status, user_id,
                      events_processed, episodes_created, memories_created,
                      memories_updated, memories_decayed, entities_created,
                      relations_created, conflicts_found, journal, metadata
               FROM meditations WHERE user_id = $1
               ORDER BY started_at DESC
               LIMIT {limit}"#
        );
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![user_id], map_meditation_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get the last meditation record for a user.
    #[allow(dead_code)] // planned API: meditation history
    pub(crate) fn last_meditation(&self, user_id: &str) -> Result<Option<MeditationRecord>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            r#"SELECT meditation_id, triggered_by, CAST(started_at AS VARCHAR),
                      CAST(finished_at AS VARCHAR), status, user_id,
                      events_processed, episodes_created, memories_created,
                      memories_updated, memories_decayed, entities_created,
                      relations_created, conflicts_found, journal, metadata
               FROM meditations WHERE user_id = $1
               ORDER BY started_at DESC
               LIMIT 1"#,
        )?;
        let mut rows = stmt.query_map(params![user_id], map_meditation_row)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }
}

#[allow(dead_code)] // used by meditation query methods above
fn map_meditation_row(row: &duckdb::Row<'_>) -> duckdb::Result<MeditationRecord> {
    Ok(MeditationRecord {
        meditation_id: row.get(0)?,
        triggered_by: row.get(1)?,
        started_at: row.get::<_, String>(2)?,
        finished_at: row.get::<_, Option<String>>(3)?,
        status: MeditationStatus::parse(&row.get::<_, String>(4).unwrap_or_default()),
        user_id: row.get(5)?,
        events_processed: row.get::<_, Option<i32>>(6)?.unwrap_or(0) as u32,
        episodes_created: row.get::<_, Option<i32>>(7)?.unwrap_or(0) as u32,
        memories_created: row.get::<_, Option<i32>>(8)?.unwrap_or(0) as u32,
        memories_updated: row.get::<_, Option<i32>>(9)?.unwrap_or(0) as u32,
        memories_decayed: row.get::<_, Option<i32>>(10)?.unwrap_or(0) as u32,
        entities_created: row.get::<_, Option<i32>>(11)?.unwrap_or(0) as u32,
        relations_created: row.get::<_, Option<i32>>(12)?.unwrap_or(0) as u32,
        conflicts_found: row.get::<_, Option<i32>>(13)?.unwrap_or(0) as u32,
        journal: row.get::<_, Option<String>>(14)?,
        metadata: row
            .get::<_, Option<String>>(15)?
            .and_then(|s| serde_json::from_str(&s).ok()),
    })
}

use crate::error::Result;
use crate::types::{MeditationRecord, MeditationStatus, SqlParam};

use super::backend::RowAccess;
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

        self.backend.execute(
            r#"INSERT INTO meditations (meditation_id, triggered_by, started_at, finished_at, status, user_id, journal, metadata)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)"#,
            &[
                SqlParam::Text(record.meditation_id.clone()),
                SqlParam::Text(record.triggered_by.clone()),
                SqlParam::Text(record.started_at.clone()),
                finished_val,
                SqlParam::Text(record.status.as_str().to_string()),
                SqlParam::Text(record.user_id.clone()),
                journal_val,
                meta_val,
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

        self.backend.execute(
            r#"UPDATE meditations
               SET status = $1,
                   finished_at = $2,
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
            &[
                SqlParam::Text(status.to_string()),
                finished_val,
                SqlParam::Int(stats.events_processed as i64),
                SqlParam::Int(stats.episodes_created as i64),
                SqlParam::Int(stats.memories_created as i64),
                SqlParam::Int(stats.memories_updated as i64),
                SqlParam::Int(stats.memories_decayed as i64),
                SqlParam::Int(stats.entities_created as i64),
                SqlParam::Int(stats.relations_created as i64),
                SqlParam::Int(stats.conflicts_found as i64),
                journal_val,
                SqlParam::Text(meditation_id.to_string()),
            ],
        )?;
        Ok(())
    }

    /// Get a meditation record by ID.
    #[allow(dead_code)] // planned API: meditation inspection
    pub(crate) fn get_meditation(&self, meditation_id: &str) -> Result<Option<MeditationRecord>> {
        self.backend.query_one(
            r#"SELECT meditation_id, triggered_by, started_at,
                      finished_at, status, user_id,
                      events_processed, episodes_created, memories_created,
                      memories_updated, memories_decayed, entities_created,
                      relations_created, conflicts_found, journal, metadata
               FROM meditations WHERE meditation_id = $1"#,
            &[SqlParam::Text(meditation_id.to_string())],
            |row| map_meditation_row(row),
        )
    }

    /// List meditation records for a user.
    #[allow(dead_code)] // planned API: meditation history
    pub(crate) fn list_meditations(
        &self,
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<MeditationRecord>> {
        let sql = format!(
            r#"SELECT meditation_id, triggered_by, started_at,
                      finished_at, status, user_id,
                      events_processed, episodes_created, memories_created,
                      memories_updated, memories_decayed, entities_created,
                      relations_created, conflicts_found, journal, metadata
               FROM meditations WHERE user_id = $1
               ORDER BY started_at DESC
               LIMIT {limit}"#
        );
        self.backend.query_read(
            &sql,
            &[SqlParam::Text(user_id.to_string())],
            |row| map_meditation_row(row),
        )
    }

    /// Get the last meditation record for a user.
    #[allow(dead_code)] // planned API: meditation history
    pub(crate) fn last_meditation(&self, user_id: &str) -> Result<Option<MeditationRecord>> {
        self.backend.query_one(
            r#"SELECT meditation_id, triggered_by, started_at,
                      finished_at, status, user_id,
                      events_processed, episodes_created, memories_created,
                      memories_updated, memories_decayed, entities_created,
                      relations_created, conflicts_found, journal, metadata
               FROM meditations WHERE user_id = $1
               ORDER BY started_at DESC
               LIMIT 1"#,
            &[SqlParam::Text(user_id.to_string())],
            |row| map_meditation_row(row),
        )
    }
}

#[allow(dead_code)] // used by meditation query methods above
fn map_meditation_row(row: &dyn RowAccess) -> Result<MeditationRecord> {
    Ok(MeditationRecord {
        meditation_id: row.get_string(0)?,
        triggered_by: row.get_string(1)?,
        started_at: row.get_string(2)?,
        finished_at: row.get_opt_string(3)?,
        status: MeditationStatus::parse(&row.get_string(4).unwrap_or_default()),
        user_id: row.get_string(5)?,
        events_processed: row.get_opt_i64(6)?.map(|v| v as u32).unwrap_or(0),
        episodes_created: row.get_opt_i64(7)?.map(|v| v as u32).unwrap_or(0),
        memories_created: row.get_opt_i64(8)?.map(|v| v as u32).unwrap_or(0),
        memories_updated: row.get_opt_i64(9)?.map(|v| v as u32).unwrap_or(0),
        memories_decayed: row.get_opt_i64(10)?.map(|v| v as u32).unwrap_or(0),
        entities_created: row.get_opt_i64(11)?.map(|v| v as u32).unwrap_or(0),
        relations_created: row.get_opt_i64(12)?.map(|v| v as u32).unwrap_or(0),
        conflicts_found: row.get_opt_i64(13)?.map(|v| v as u32).unwrap_or(0),
        journal: row.get_opt_string(14)?,
        metadata: row
            .get_opt_string(15)?
            .and_then(|s| serde_json::from_str(&s).ok()),
    })
}

use duckdb::params;

use crate::analytics::{EntityStat, EventCount, TimeBucket, UserStats};
use crate::error::{MemoryError, Result};

use super::Storage;

impl Storage {
    /// Get summary statistics for a user.
    pub(crate) fn user_stats(&self, user_id: &str) -> Result<UserStats> {
        let collection = &self.config.collection_name;

        let sql = format!(
            r#"SELECT
                (SELECT COUNT(*) FROM memories WHERE user_id = $1) AS total_memories,
                (SELECT COUNT(*) FROM entities_{collection} WHERE user_id = $1) AS total_entities,
                (SELECT COUNT(*) FROM relationships_{collection} WHERE user_id = $1) AS total_relationships,
                (SELECT CAST(MIN(created_at) AS VARCHAR) FROM memories WHERE user_id = $1) AS earliest_memory,
                (SELECT CAST(MAX(created_at) AS VARCHAR) FROM memories WHERE user_id = $1) AS latest_memory,
                (SELECT COUNT(DISTINCT agent_id) FROM memories WHERE user_id = $1 AND agent_id IS NOT NULL) AS unique_agents"#
        );

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query_map(params![user_id], |row| {
            Ok(UserStats {
                user_id: user_id.to_string(),
                total_memories: row.get::<_, i64>(0)? as u64,
                total_entities: row.get::<_, i64>(1)? as u64,
                total_relationships: row.get::<_, i64>(2)? as u64,
                earliest_memory: row.get::<_, Option<String>>(3)?,
                latest_memory: row.get::<_, Option<String>>(4)?,
                unique_agents: row.get::<_, i64>(5)? as u64,
            })
        })?;

        Ok(rows.next().ok_or_else(|| {
            crate::error::MemoryError::Config("user_stats returned no rows".into())
        })??)
    }

    /// Get memory creation frequency by time period.
    ///
    /// `granularity` must be one of `"day"`, `"week"`, or `"month"`.
    pub(crate) fn memory_frequency(
        &self,
        user_id: &str,
        granularity: &str,
        limit: usize,
    ) -> Result<Vec<TimeBucket>> {
        match granularity {
            "day" | "week" | "month" => {}
            other => {
                return Err(MemoryError::Config(format!(
                    "Invalid granularity '{other}': must be \"day\", \"week\", or \"month\""
                )));
            }
        }

        let sql = format!(
            r#"SELECT CAST(date_trunc('{granularity}', created_at) AS VARCHAR) AS period,
                      COUNT(*) AS cnt
               FROM memories
               WHERE user_id = $1
               GROUP BY period
               ORDER BY period DESC
               LIMIT {limit}"#
        );

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![user_id], |row| {
                Ok(TimeBucket {
                    period: row.get::<_, String>(0)?,
                    count: row.get::<_, i64>(1)? as u64,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(rows)
    }

    /// Get history event distribution for a user.
    #[allow(dead_code)] // planned API: analytics
    pub(crate) fn event_distribution(&self, user_id: &str) -> Result<Vec<EventCount>> {
        let sql = r#"SELECT event, COUNT(*) AS cnt
                     FROM history
                     WHERE user_id = $1
                     GROUP BY event"#;

        let conn = self.read_conn();
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt
            .query_map(params![user_id], |row| {
                Ok(EventCount {
                    event: row.get::<_, String>(0)?,
                    count: row.get::<_, i64>(1)? as u64,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(rows)
    }

    /// Get top entities by relationship count.
    pub(crate) fn top_entities(&self, user_id: &str, limit: usize) -> Result<Vec<EntityStat>> {
        let collection = &self.config.collection_name;

        let sql = format!(
            r#"SELECT e.name, e.entity_type, COUNT(r.id) AS rel_count
               FROM entities_{collection} e
               LEFT JOIN relationships_{collection} r
                   ON (r.source_id = e.id OR r.target_id = e.id)
               WHERE e.user_id = $1
               GROUP BY e.id, e.name, e.entity_type
               ORDER BY rel_count DESC
               LIMIT {limit}"#
        );

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![user_id], |row| {
                Ok(EntityStat {
                    name: row.get::<_, String>(0)?,
                    entity_type: row.get::<_, Option<String>>(1)?,
                    relationship_count: row.get::<_, i64>(2)? as u64,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(rows)
    }
}

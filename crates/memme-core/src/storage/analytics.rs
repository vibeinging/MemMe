use crate::analytics::{EntityStat, EventCount, TimeBucket, UserStats};
use crate::error::{MemoryError, Result};
use crate::types::SqlParam;

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
                (SELECT MIN(created_at) FROM memories WHERE user_id = $1) AS earliest_memory,
                (SELECT MAX(created_at) FROM memories WHERE user_id = $1) AS latest_memory,
                (SELECT COUNT(DISTINCT agent_id) FROM memories WHERE user_id = $1 AND agent_id IS NOT NULL) AS unique_agents"#
        );

        let params = &[SqlParam::Text(user_id.to_string())];
        let rows = self.backend.query_read(&sql, params, |row| {
            Ok(UserStats {
                user_id: user_id.to_string(),
                total_memories: row.get_i64(0)? as u64,
                total_entities: row.get_i64(1)? as u64,
                total_relationships: row.get_i64(2)? as u64,
                earliest_memory: row.get_opt_string(3)?,
                latest_memory: row.get_opt_string(4)?,
                unique_agents: row.get_i64(5)? as u64,
            })
        })?;

        rows.into_iter()
            .next()
            .ok_or_else(|| crate::error::MemoryError::Config("user_stats returned no rows".into()))
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

        let trunc_expr = self.dialect().date_trunc_expr(granularity, "created_at");
        let sql = format!(
            r#"SELECT {trunc_expr} AS period,
                      COUNT(*) AS cnt
               FROM memories
               WHERE user_id = $1
               GROUP BY period
               ORDER BY period DESC
               LIMIT {limit}"#
        );

        self.backend
            .query_read(&sql, &[SqlParam::Text(user_id.to_string())], |row| {
                Ok(TimeBucket {
                    period: row.get_string(0)?,
                    count: row.get_i64(1)? as u64,
                })
            })
    }

    /// Get history event distribution for a user.
    #[allow(dead_code)] // planned API: analytics
    pub(crate) fn event_distribution(&self, user_id: &str) -> Result<Vec<EventCount>> {
        let sql = r#"SELECT event, COUNT(*) AS cnt
                     FROM history
                     WHERE user_id = $1
                     GROUP BY event"#;

        self.backend
            .query_read(sql, &[SqlParam::Text(user_id.to_string())], |row| {
                Ok(EventCount {
                    event: row.get_string(0)?,
                    count: row.get_i64(1)? as u64,
                })
            })
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

        self.backend
            .query_read(&sql, &[SqlParam::Text(user_id.to_string())], |row| {
                Ok(EntityStat {
                    name: row.get_string(0)?,
                    entity_type: row.get_opt_string(1)?,
                    relationship_count: row.get_i64(2)? as u64,
                })
            })
    }
}

use crate::error::Result;

impl super::MemoryStore {
    /// Get summary statistics for a user.
    pub fn user_stats(&self, user_id: &str) -> Result<crate::analytics::UserStats> {
        self.storage.user_stats(user_id)
    }

    /// Get memory creation frequency by time period.
    ///
    /// `granularity` must be one of `"day"`, `"week"`, or `"month"`.
    pub fn memory_frequency(
        &self,
        user_id: &str,
        granularity: &str,
        limit: usize,
    ) -> Result<Vec<crate::analytics::TimeBucket>> {
        self.storage.memory_frequency(user_id, granularity, limit)
    }

    /// Get history event distribution for a user.
    #[allow(dead_code)] // planned API: analytics dashboard
    pub(crate) fn event_distribution(
        &self,
        user_id: &str,
    ) -> Result<Vec<crate::analytics::EventCount>> {
        self.storage.event_distribution(user_id)
    }

    /// Get top entities by relationship count.
    pub fn top_entities(
        &self,
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<crate::analytics::EntityStat>> {
        self.storage.top_entities(user_id, limit)
    }
}

use crate::error::{MemoryError, Result};
use crate::types::*;

impl super::MemoryStore {
    /// Get a session by ID.
    pub fn get_session(&self, session_id: &str) -> Result<Option<Session>> {
        self.storage.get_session(session_id)
    }

    /// List sessions for a user.
    pub fn list_sessions(&self, options: ListSessionsOptions) -> Result<Vec<Session>> {
        self.storage.list_sessions(&options)
    }

    /// Delete a session.
    pub fn delete_session(&self, session_id: &str) -> Result<()> {
        self.storage.delete_session(session_id)
    }

    /// **Internal** — Used by `append_events()`. Most users don't need to call this directly.
    ///
    /// Get or create a session by ID.
    #[allow(dead_code)] // planned API: session management
    pub(crate) fn get_or_create_session(
        &self,
        session_id: &str,
        user_id: &str,
        source_id: Option<&str>,
    ) -> Result<Session> {
        let now = chrono::Utc::now().to_rfc3339();
        self.storage
            .get_or_create_session(session_id, user_id, source_id, &now, None)
    }

    /// Get events for a session with pagination.
    pub fn get_session_events(
        &self,
        session_id: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<Event>> {
        // Resolve user_id from the session itself so the caller doesn't need to pass it.
        let session = self
            .storage
            .get_session(session_id)?
            .ok_or_else(|| MemoryError::NotFound(format!("Session {session_id}")))?;
        let offset = offset.unwrap_or(0);
        let limit = limit.unwrap_or(50);
        let opts = ListEventsOptions::new(&session.user_id)
            .session_id(session_id)
            .limit(limit + offset);
        let events = self.list_events(opts)?;
        Ok(events.into_iter().skip(offset).take(limit).collect())
    }

    /// # Core API — Get session context for retrieval
    ///
    /// Returns purified events from a session within a token budget.
    /// This is used during retrieval to provide current conversation context.
    ///
    /// The function:
    /// 1. Gets the session's events (preferring purified content when available)
    /// 2. Estimates token count for each event
    /// 3. Returns events within the token budget (most recent first)
    /// 4. Includes episode summary if available
    ///
    /// # Arguments
    /// * `session_id` - The session to get context from
    /// * `options` - Options controlling token budget and what to include
    ///
    /// # Concurrency Note
    /// If compact is in progress, events may be in one of three states:
    /// - **Purified**: Already processed by compact, has `purified_content`
    /// - **Unprocessed**: Not yet compacted, only has raw `content`
    /// - **Processing**: Currently being compacted (MVCC ensures atomicity)
    ///
    /// By default, all events are included. Use `.purified_only()` to exclude
    /// unprocessed events if you need guaranteed high-quality context.
    pub fn get_session_context(
        &self,
        session_id: &str,
        options: GetSessionContextOptions,
    ) -> Result<SessionContext> {
        // Verify session exists
        let _session = self
            .storage
            .get_session(session_id)?
            .ok_or_else(|| MemoryError::NotFound(format!("Session {session_id}")))?;

        let events = self
            .storage
            .get_session_events_ordered(session_id, options.max_events)?;

        let mut selected_events = Vec::new();
        let mut tokens_used = 0;
        let token_budget = options.token_budget;

        for event in events.into_iter().rev() {
            if !options.include_unprocessed && event.purified_content.is_none() {
                continue;
            }

            let content = event.purified_content.as_ref().unwrap_or(&event.content);

            let event_tokens = super::helpers::estimate_tokens(content);

            if tokens_used + event_tokens <= token_budget {
                tokens_used += event_tokens;
                selected_events.push(event);
            } else {
                break;
            }
        }

        selected_events.reverse();

        let purified_count = selected_events
            .iter()
            .filter(|e| e.purified_content.is_some())
            .count();

        let episode_summary = if options.include_summary {
            self.storage.get_session_episode_summary(session_id)?
        } else {
            None
        };

        if let Some(ref summary) = episode_summary {
            tokens_used += summary.len() / 4;
        }

        Ok(SessionContext {
            session_id: session_id.to_string(),
            raw_count: selected_events.len().saturating_sub(purified_count),
            purified_count,
            events: selected_events,
            tokens_used,
            token_budget,
            episode_summary,
        })
    }
}

//! Background task queue for deferred LLM operations.
//!
//! Tasks (compact, meditate) are enqueued during normal operations and
//! processed one-at-a-time when the caller invokes `process_background()`.
//! This keeps LLM calls off the critical path of `append_events()` and
//! `search()`.

use std::collections::VecDeque;

/// A background task that requires LLM processing.
#[derive(Debug, Clone)]
pub(crate) enum BackgroundTask {
    /// Compact the specified session into an episode.
    #[allow(dead_code)] // planned: queued by automatic session compaction
    CompactSession(String),
    /// Run meditation on the specified episode.
    #[allow(dead_code)] // planned: triggered by low-confidence search results
    MeditateEpisode(String),
}

impl super::MemoryStore {
    /// Enqueue a background task. Does not execute it immediately.
    pub(crate) fn enqueue_background(&self, task: BackgroundTask) {
        let mut queue = super::helpers::recover_lock(&self.background_queue, "background_queue");
        // Deduplicate: don't enqueue the same task twice.
        let dominated = queue.iter().any(|existing| match (existing, &task) {
            (BackgroundTask::CompactSession(a), BackgroundTask::CompactSession(b)) => a == b,
            (BackgroundTask::MeditateEpisode(a), BackgroundTask::MeditateEpisode(b)) => a == b,
            _ => false,
        });
        if !dominated {
            tracing::debug!(?task, "Enqueued background task");
            queue.push_back(task);
        }
    }

    /// Process one task from the background queue.
    ///
    /// Returns `true` if a task was processed, `false` if the queue was empty.
    /// Each call processes at most one task to avoid blocking.
    ///
    /// Safe to call without an LLM configured — tasks that need LLM are
    /// skipped and re-enqueued at the back.
    pub fn process_background(&self) -> bool {
        let task = {
            let mut queue =
                super::helpers::recover_lock(&self.background_queue, "background_queue");
            queue.pop_front()
        };

        let task = match task {
            Some(t) => t,
            None => return false,
        };

        match task {
            BackgroundTask::CompactSession(ref session_id) => {
                if !self.has_llm() {
                    tracing::debug!(session_id, "Skipping CompactSession — no LLM configured");
                    self.enqueue_background(task);
                    return false;
                }
                match self.compact(session_id) {
                    Ok(result) => {
                        tracing::info!(
                            session_id,
                            episode_id = %result.episode_id,
                            events = result.events_processed,
                            "Background compact completed"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(session_id, error = %e, "Background compact failed");
                    }
                }
            }
            BackgroundTask::MeditateEpisode(ref episode_id) => {
                if !self.has_llm() {
                    tracing::debug!(episode_id, "Skipping MeditateEpisode — no LLM configured");
                    self.enqueue_background(task);
                    return false;
                }
                // Find the episode to get the user_id for meditation
                match self.storage.get_episode(episode_id) {
                    Ok(Some(episode)) => {
                        let opts = crate::types::MeditateOptions {
                            user_id: episode.user_id.clone(),
                            triggered_by: "background".to_string(),
                            since: None,
                        };
                        match self.meditate(opts) {
                            Ok(record) => {
                                tracing::info!(
                                    episode_id,
                                    meditation_id = %record.meditation_id,
                                    "Background meditate completed"
                                );
                            }
                            Err(e) => {
                                tracing::warn!(episode_id, error = %e, "Background meditate failed");
                            }
                        }
                    }
                    Ok(None) => {
                        tracing::warn!(episode_id, "Background meditate: episode not found");
                    }
                    Err(e) => {
                        tracing::warn!(
                            episode_id,
                            error = %e,
                            "Background meditate: failed to fetch episode"
                        );
                    }
                }
            }
        }

        true
    }

    /// Process all pending background tasks. Returns the number of tasks processed.
    pub fn drain_background(&self) -> usize {
        let mut count = 0;
        while self.process_background() {
            count += 1;
        }
        count
    }

    /// Number of tasks currently in the background queue.
    pub fn background_queue_len(&self) -> usize {
        super::helpers::recover_lock(&self.background_queue, "background_queue").len()
    }

    /// Peek at the background queue without modifying it (for diagnostics).
    #[allow(dead_code)]
    pub(crate) fn peek_background_queue(&self) -> VecDeque<BackgroundTask> {
        super::helpers::recover_lock(&self.background_queue, "background_queue").clone()
    }
}

use std::sync::Arc;
use uuid::Uuid;

use memme_llm::prompts::{
    get_fact_retrieval_messages_with_prompt, get_fact_retrieval_messages_with_time,
    parse_fact_retrieval_response,
};
use memme_llm::{generate_structured, LlmProvider, ResponseFormat, StructuredGenConfig};

use crate::error::{MemoryError, Result};
use crate::types::*;

impl super::MemoryStore {
    /// Start a meditation session. This is the orchestrator that:
    /// 1. Collects processed Episodes (narrative traces from compact)
    /// 2. Extracts semantic memories from purified Episodes (requires LLM)
    /// 3. Builds entity graph from extracted memories (requires LLM)
    /// 4. Updates identity traits (requires LLM)
    /// 5. Applies decay to old memories
    /// 6. Generates a meditation journal
    ///
    /// Without LLM, only performs decay and basic statistics.
    pub fn meditate(&self, options: MeditateOptions) -> Result<MeditationRecord> {
        let meditation_id = Uuid::new_v4().to_string();
        let started_at = chrono::Utc::now().to_rfc3339();

        // Create initial meditation record
        let mut record = MeditationRecord {
            meditation_id: meditation_id.clone(),
            triggered_by: options.triggered_by.clone(),
            started_at,
            finished_at: None,
            status: MeditationStatus::Running,
            user_id: options.user_id.clone(),
            events_processed: 0,
            episodes_created: 0,
            memories_created: 0,
            memories_updated: 0,
            memories_decayed: 0,
            entities_created: 0,
            relations_created: 0,
            conflicts_found: 0,
            journal: None,
            metadata: None,
        };
        self.storage.insert_meditation(&record)?;

        // Phase 1: Apply decay to existing memories (always works, no LLM needed)
        let consolidate_result = self.consolidate(&options.user_id, 0.01, 0.0, false)?;
        record.memories_decayed = consolidate_result.decayed_count as u32;

        // Phase 2-5: LLM-dependent operations
        let mut llm_failed = false;
        if let Some(llm) = self.llm() {
            match self.meditate_with_llm(&options, &llm, &mut record) {
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!("Meditation LLM operations failed: {e}");
                    llm_failed = true;
                }
            }
        }

        // Mark completion
        let finished_at = chrono::Utc::now().to_rfc3339();
        record.finished_at = Some(finished_at);
        if llm_failed {
            record.status = MeditationStatus::Failed;
            record.journal = Some(format!(
                "Meditation partially failed (LLM error). Decayed {} memories, but extraction skipped.",
                record.memories_decayed
            ));
        } else {
            record.status = MeditationStatus::Completed;
            record.journal = Some(format!(
                "Meditation completed. Created {} memories, updated {}, decayed {}.",
                record.memories_created, record.memories_updated, record.memories_decayed
            ));
        }

        self.storage.update_meditation(
            &meditation_id,
            record.status.as_str(),
            &record,
            record.journal.as_deref(),
        )?;

        Ok(record)
    }

    /// LLM-dependent meditation operations.
    fn meditate_with_llm(
        &self,
        options: &MeditateOptions,
        llm: &Arc<dyn LlmProvider>,
        record: &mut MeditationRecord,
    ) -> Result<()> {
        // Phase 2: Get Episodes (narrative traces) for this user
        let episodes = self.storage.list_episodes_for_user(&options.user_id)?;
        if episodes.is_empty() {
            tracing::info!("No episodes to meditate on for user {}", options.user_id);
            return Ok(());
        }

        // Phase 3: Extract memories from each Episode
        let mut total_memories = 0;

        for episode in &episodes {
            // Build text from episode summary (the narrative trace)
            let text = format!("{}: {}", episode.title, episode.summary);

            // Extract facts from the episode using LLM directly
            let facts = self
                .extract_facts_from_episode(
                    llm,
                    &text,
                    self.config.custom_fact_extraction_prompt.as_deref(),
                    Some(&episode.started_at),
                )
                .unwrap_or_default();

            // Add each fact as a memory
            for fact in &facts {
                if fact.trim().is_empty() {
                    continue;
                }
                // Get first session_id if available
                let session_id = episode.session_ids.first();
                let mut add_opts = AddOptions::new(&options.user_id)
                    .importance(episode.significance)
                    .event_time(&episode.started_at);
                if let Some(sid) = session_id {
                    add_opts = add_opts.session_id(sid);
                }

                match self.add(fact, add_opts) {
                    Ok(_) => total_memories += 1,
                    Err(e) => tracing::warn!("Failed to add memory during meditation: {e}"),
                }
            }
        }

        record.memories_created = total_memories;

        // Phase 4: Build entity graph if enabled
        if self.config.enable_graph {
            let graph_processor = crate::graph::GraphProcessor::new(llm.clone());

            // Process recent episodes for entities
            for episode in episodes.iter().take(5) {
                let text = format!("{}: {}", episode.title, episode.summary);
                match graph_processor.process(&self.storage, &text, &options.user_id) {
                    Ok(graph_result) => {
                        record.entities_created += graph_result.entities.len() as u32;
                        record.relations_created += graph_result.relations.len() as u32;
                    }
                    Err(e) => tracing::warn!("Graph extraction failed for episode: {e}"),
                }
            }
        }

        Ok(())
    }

    /// Extract facts from episode text using LLM.
    fn extract_facts_from_episode(
        &self,
        llm: &Arc<dyn LlmProvider>,
        text: &str,
        custom_prompt: Option<&str>,
        conversation_time: Option<&str>,
    ) -> Result<Vec<String>> {
        let messages = if let Some(prompt) = custom_prompt {
            get_fact_retrieval_messages_with_prompt(text, prompt)
        } else {
            get_fact_retrieval_messages_with_time(text, conversation_time)
        };

        let config = StructuredGenConfig {
            base_temperature: Some(0.1),
            max_tokens: Some(2048),
            response_format: Some(ResponseFormat::Json),
            ..Default::default()
        };

        let facts = generate_structured(llm.as_ref(), &messages, &config, |raw| {
            parse_fact_retrieval_response(raw).map(|r| r.facts)
        })
        .map_err(|e| MemoryError::Llm(e.to_string()))?;
        Ok(facts)
    }

    /// Get the last meditation record for a user.
    #[allow(dead_code)] // planned API: meditation history
    pub(crate) fn last_meditation(&self, user_id: &str) -> Result<Option<MeditationRecord>> {
        self.storage.last_meditation(user_id)
    }

    /// List meditation history.
    #[allow(dead_code)] // planned API: meditation history
    pub(crate) fn list_meditations(
        &self,
        user_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<MeditationRecord>> {
        self.storage.list_meditations(user_id, limit.unwrap_or(10))
    }
}

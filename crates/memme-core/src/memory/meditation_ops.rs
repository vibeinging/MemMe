use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use memme_llm::prompts::{
    get_fact_retrieval_messages_with_prompt, get_fact_retrieval_messages_with_time,
    parse_fact_retrieval_response, ExtractedFact,
};
use memme_llm::{generate_structured, LlmProvider, ResponseFormat, StructuredGenConfig};

use crate::error::{MemoryError, Result};
use crate::text_utils;
use crate::types::*;

/// Metadata carried per-fact for operation execution.
#[derive(Clone)]
struct FactMeta {
    event_time: Option<String>,
    session_id: Option<String>,
    significance: f32,
}

impl super::MemoryStore {
    /// Start a meditation session. This is the orchestrator that:
    /// 1. Applies decay to old memories
    /// 2. Extracts facts from purified episodes (requires LLM)
    /// 3. Reconciles facts against existing memories (ADD/UPDATE/DELETE)
    /// 4. Builds entity graph and links entities to memories
    ///
    /// Without LLM, only performs decay and basic statistics.
    pub fn meditate(&self, options: MeditateOptions) -> Result<MeditationRecord> {
        if self.is_within_cooldown(&options.user_id) {
            let now = chrono::Utc::now().to_rfc3339();
            return Ok(MeditationRecord {
                meditation_id: Uuid::new_v4().to_string(),
                triggered_by: options.triggered_by.clone(),
                started_at: now.clone(),
                finished_at: Some(now),
                status: MeditationStatus::Completed,
                user_id: options.user_id.clone(),
                journal: Some("Skipped: cooldown period not elapsed".to_string()),
                ..Default::default()
            });
        }

        let meditation_id = Uuid::new_v4().to_string();
        let started_at = chrono::Utc::now().to_rfc3339();

        // Sync replica before meditation — preserve the last known good state
        if let Err(e) = self.sync_replica() {
            tracing::warn!("Pre-meditation replica sync failed: {e}");
        }

        let mut record = MeditationRecord {
            meditation_id: meditation_id.clone(),
            triggered_by: options.triggered_by.clone(),
            started_at,
            finished_at: None,
            status: MeditationStatus::Running,
            user_id: options.user_id.clone(),
            ..Default::default()
        };
        self.storage.insert_meditation(&record)?;

        // Phase 1: Apply decay to existing memories
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
                "Meditation completed. Created {} memories, updated {}, deleted {}, decayed {}.",
                record.memories_created,
                record.memories_updated,
                record.conflicts_found,
                record.memories_decayed
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

    /// LLM-dependent meditation operations — small-batch processing.
    ///
    /// Mimics v10 add_smart behavior: extract facts from episodes, then
    /// reconcile in small batches (~10 facts each) against the growing
    /// memory base. Small batches produce better ADD/UPDATE decisions
    /// because the LLM can focus on fewer facts at a time.
    ///
    /// Flow per episode:
    /// 1. Extract facts (1 LLM call)
    /// 2. Split facts into batches of ~10
    /// 3. For each batch: reconcile against existing memories (1 LLM call)
    ///    → later batches see memories created by earlier ones
    /// 4. Build entity graph (1 LLM call, if enabled)
    fn meditate_with_llm(
        &self,
        options: &MeditateOptions,
        llm: &Arc<dyn LlmProvider>,
        record: &mut MeditationRecord,
    ) -> Result<()> {
        let graph_processor = if self.config.enable_graph {
            Some(crate::graph::GraphProcessor::new(llm.clone()))
        } else {
            None
        };

        let mut total_processed: usize = 0;

        // Loop until all unmeditated episodes are processed.
        // Each iteration fetches up to meditation_batch_size episodes.
        loop {
            let episodes = self.storage.list_episodes_for_meditation(
                &options.user_id,
                self.config.meditation_min_significance,
                self.config.meditation_batch_size,
            )?;
            if episodes.is_empty() {
                break;
            }
            tracing::info!(
                "Meditation batch: {} episodes (processed {} so far) for {}",
                episodes.len(),
                total_processed,
                options.user_id
            );

            let mut episode_ids_to_mark: Vec<String> = Vec::new();

            for (ep_idx, episode) in episodes.iter().enumerate() {
                // Per-episode timeout: 3 minutes max to prevent a single stalled
                // LLM call from blocking the entire meditation run.
                let ep_start = std::time::Instant::now();
                let ep_timeout = std::time::Duration::from_secs(180);

                let events = self.storage.get_events_by_ids(&episode.event_ids)?;
                if events.is_empty() {
                    let _ = self.storage.mark_episode_meditated(&episode.episode_id);
                    continue;
                }

                let text: String = events
                    .iter()
                    .map(|e| e.purified_content.as_deref().unwrap_or(&e.content))
                    .collect::<Vec<_>>()
                    .join("\n");

                let conversation_time = events
                    .iter()
                    .find_map(|e| e.event_time.as_deref())
                    .unwrap_or(&episode.started_at);

                let session_id = episode.session_ids.first().cloned();

                // ── Step 1: Extract all facts from this episode ──
                let extracted = match self.extract_facts_from_episode(
                    llm,
                    &text,
                    self.config.custom_fact_extraction_prompt.as_deref(),
                    Some(conversation_time),
                ) {
                    Ok(facts) => facts,
                    Err(e) => {
                        tracing::warn!(
                            "Episode {}/{}: fact extraction failed: {e}, skipping",
                            ep_idx + 1,
                            episodes.len()
                        );
                        episode_ids_to_mark.push(episode.episode_id.clone());
                        continue;
                    }
                };

                // Check per-episode timeout after the LLM call
                if ep_start.elapsed() > ep_timeout {
                    tracing::warn!(
                        "Episode {}/{}: exceeded {}s timeout after fact extraction, skipping remaining steps",
                        ep_idx + 1, episodes.len(), ep_timeout.as_secs()
                    );
                    episode_ids_to_mark.push(episode.episode_id.clone());
                    continue;
                }

                let mut facts: Vec<String> = Vec::new();
                let mut fact_meta: HashMap<String, FactMeta> = HashMap::new();

                for fact in extracted {
                    if fact.text.trim().is_empty() {
                        continue;
                    }
                    let event_time = fact
                        .happened_at
                        .or_else(|| text_utils::extract_iso_date_from_text(&fact.text))
                        .or_else(|| Some(conversation_time.to_string()));

                    if !fact_meta.contains_key(&fact.text) {
                        fact_meta.insert(
                            fact.text.clone(),
                            FactMeta {
                                event_time,
                                session_id: session_id.clone(),
                                significance: episode.significance,
                            },
                        );
                        facts.push(fact.text);
                    }
                }

                tracing::info!(
                    "Episode {}/{}: extracted {} facts",
                    ep_idx + 1,
                    episodes.len(),
                    facts.len()
                );

                if facts.is_empty() {
                    episode_ids_to_mark.push(episode.episode_id.clone());
                    continue;
                }

                // ── Step 2: Store facts via batch add ──
                // Batch embed all facts in one API call, then dedup + store each.
                let mut all_new_memories: Vec<MemoryResult> = Vec::new();

                let batch_items: Vec<(String, AddOptions)> = facts
                    .iter()
                    .map(|fact| {
                        let mut add_opts = AddOptions::new(&options.user_id);
                        if let Some(meta) = fact_meta.get(fact) {
                            if let Some(ref et) = meta.event_time {
                                add_opts = add_opts.event_time(et);
                            }
                            if let Some(ref sid) = meta.session_id {
                                add_opts = add_opts.session_id(sid);
                            }
                            add_opts = add_opts.importance(meta.significance);
                        }
                        (fact.clone(), add_opts)
                    })
                    .collect();

                match self.add_batch(&batch_items) {
                    Ok(results) => {
                        record.memories_created += results.len() as u32;
                        all_new_memories.extend(results);
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Batch add failed, falling back to individual adds");
                        // Fallback: try one by one
                        for (fact, opts) in &batch_items {
                            match self.add(fact, opts.clone()) {
                                Ok(result) => {
                                    all_new_memories.push(result);
                                    record.memories_created += 1;
                                }
                                Err(e) => {
                                    tracing::warn!(text = %fact, error = %e, "Failed to add memory");
                                }
                            }
                        }
                    }
                }

                // ── Step 3: Graph extraction + entity linking ──
                if let Some(ref gp) = graph_processor {
                    // Skip graph if we're already past the timeout budget
                    if ep_start.elapsed() <= ep_timeout {
                        self.process_graph_batch(
                            gp,
                            &text,
                            &options.user_id,
                            &all_new_memories,
                            record,
                        );
                    }
                }

                episode_ids_to_mark.push(episode.episode_id.clone());
            }

            // Mark episodes as meditated after this batch completes
            for episode_id in &episode_ids_to_mark {
                if let Err(e) = self.storage.mark_episode_meditated(episode_id) {
                    tracing::warn!("Failed to mark episode {episode_id} as meditated: {e}");
                }
            }

            total_processed += episodes.len();
        }

        Ok(())
    }

    /// Process graph extraction for an episode and link entities to memories.
    fn process_graph_batch(
        &self,
        gp: &crate::graph::GraphProcessor,
        text: &str,
        user_id: &str,
        new_memories: &[MemoryResult],
        record: &mut MeditationRecord,
    ) {
        match gp.process(&self.storage, text, user_id) {
            Ok(graph_result) => {
                record.entities_created += graph_result.entities.len() as u32;
                record.relations_created += graph_result.relations.len() as u32;

                if !graph_result.entities.is_empty() && !new_memories.is_empty() {
                    let entity_names: Vec<String> = graph_result
                        .entities
                        .iter()
                        .map(|e| e.name.clone())
                        .collect();
                    let entity_index = crate::entity_index::EntityIndex::build(&entity_names);

                    for memory in new_memories {
                        let matched = entity_index.extract(&memory.content);
                        for matched_name in &matched {
                            if let Some(entity) = graph_result
                                .entities
                                .iter()
                                .find(|e| e.name.to_lowercase() == *matched_name)
                            {
                                let _ = self.storage.link_memory_entity(
                                    &memory.id,
                                    &entity.id,
                                    &entity.name,
                                    user_id,
                                );
                            }
                        }
                    }
                }
            }
            Err(e) => tracing::warn!("Graph extraction failed: {e}"),
        }
    }

    /// Extract facts from episode text using LLM.
    fn extract_facts_from_episode(
        &self,
        llm: &Arc<dyn LlmProvider>,
        text: &str,
        custom_prompt: Option<&str>,
        conversation_time: Option<&str>,
    ) -> Result<Vec<ExtractedFact>> {
        let messages = if let Some(prompt) = custom_prompt {
            get_fact_retrieval_messages_with_prompt(text, prompt)
        } else {
            get_fact_retrieval_messages_with_time(text, conversation_time)
        };

        // Scale max_tokens based on input length: more input → more facts → more output needed
        let input_tokens = text.len() / 4;
        let scaled_max = (input_tokens * 2).clamp(1024, 8192);

        let config = StructuredGenConfig {
            base_temperature: Some(0.1),
            max_tokens: Some(scaled_max),
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
    #[allow(dead_code)]
    pub(crate) fn last_meditation(&self, user_id: &str) -> Result<Option<MeditationRecord>> {
        self.storage.last_meditation(user_id)
    }

    /// List meditation history.
    #[allow(dead_code)]
    pub(crate) fn list_meditations(
        &self,
        user_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<MeditationRecord>> {
        self.storage.list_meditations(user_id, limit.unwrap_or(10))
    }

    /// Check if the user is within the meditation cooldown period.
    fn is_within_cooldown(&self, user_id: &str) -> bool {
        if self.config.meditation_cooldown_hours == 0 {
            return false;
        }
        let last = match self.last_meditation(user_id) {
            Ok(Some(m))
                if m.status == MeditationStatus::Completed
                    || m.status == MeditationStatus::Running =>
            {
                m
            }
            Ok(_) => return false,
            Err(e) => {
                tracing::warn!("Failed to check meditation cooldown for {user_id}: {e}");
                return false;
            }
        };
        let last_time = match chrono::DateTime::parse_from_rfc3339(&last.started_at) {
            Ok(t) => t.with_timezone(&chrono::Utc),
            Err(_) => return false,
        };
        let cooldown = chrono::Duration::hours(self.config.meditation_cooldown_hours as i64);
        chrono::Utc::now() - last_time < cooldown
    }
}

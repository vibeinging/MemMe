use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use memme_llm::prompts::{
    get_fact_retrieval_messages_with_prompt, get_fact_retrieval_messages_with_time,
    get_update_memory_messages, get_update_memory_messages_with_prompt,
    parse_fact_retrieval_response, parse_update_memory_response, ExtractedFact, MemoryEvent,
    OldMemory,
};
use memme_llm::{generate_structured, LlmProvider, ResponseFormat, StructuredGenConfig};

use crate::error::{MemoryError, Result};
use crate::text_utils;
use crate::types::*;

/// Metadata carried per-fact for operation execution.
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

    /// LLM-dependent meditation operations.
    ///
    /// Three phases:
    /// A. Extract facts from purified events (per-episode)
    /// B. Reconcile all facts against existing memories (batched, LLM-driven ADD/UPDATE/DELETE)
    /// C. Build entity graph and link entities to memories
    fn meditate_with_llm(
        &self,
        options: &MeditateOptions,
        llm: &Arc<dyn LlmProvider>,
        record: &mut MeditationRecord,
    ) -> Result<()> {
        let episodes = self.storage.list_episodes_for_meditation(
            &options.user_id,
            self.config.meditation_min_significance,
            self.config.meditation_batch_size,
        )?;
        if episodes.is_empty() {
            tracing::info!("No unmeditated episodes for user {}", options.user_id);
            return Ok(());
        }

        let graph_processor = if self.config.enable_graph {
            Some(crate::graph::GraphProcessor::new(llm.clone()))
        } else {
            None
        };

        // ── Phase A: Extract facts from all episodes ──
        let mut all_facts: Vec<String> = Vec::new();
        let mut fact_meta: HashMap<String, FactMeta> = HashMap::new();
        // Collect episode texts for graph extraction in Phase C
        let mut episode_texts: Vec<String> = Vec::new();

        for episode in &episodes {
            let events = self.storage.get_events_by_ids(&episode.event_ids)?;
            if events.is_empty() {
                // Mark empty episodes as meditated so they don't block future runs
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

            let extracted = match self.extract_facts_from_episode(
                llm,
                &text,
                self.config.custom_fact_extraction_prompt.as_deref(),
                Some(conversation_time),
            ) {
                Ok(facts) => facts,
                Err(e) => {
                    tracing::warn!(episode = %episode.episode_id, error = %e, "Fact extraction failed, skipping episode");
                    continue;
                }
            };

            let session_id = episode.session_ids.first().cloned();

            for fact in extracted {
                if fact.text.trim().is_empty() {
                    continue;
                }
                let event_time = fact
                    .happened_at
                    .or_else(|| text_utils::extract_iso_date_from_text(&fact.text))
                    .or_else(|| Some(conversation_time.to_string()));

                // Deduplicate: keep first occurrence (with its metadata)
                if !fact_meta.contains_key(&fact.text) {
                    fact_meta.insert(
                        fact.text.clone(),
                        FactMeta {
                            event_time,
                            session_id: session_id.clone(),
                            significance: episode.significance,
                        },
                    );
                    all_facts.push(fact.text);
                }
            }

            if graph_processor.is_some() {
                episode_texts.push(text);
            }

            if let Err(e) = self.storage.mark_episode_meditated(&episode.episode_id) {
                tracing::warn!(
                    "Failed to mark episode {} as meditated: {e}",
                    episode.episode_id
                );
            }
        }

        if all_facts.is_empty() {
            tracing::warn!("No facts extracted from episodes — skipping reconciliation");
            return Ok(());
        }

        tracing::info!("Extracted {} facts for reconciliation", all_facts.len());

        // ── Phase B: Reconcile facts against existing memories ──
        let new_memories =
            self.reconcile_facts(llm, &all_facts, &fact_meta, &options.user_id, record)?;

        // ── Phase C: Graph extraction + entity-memory linking ──
        if let Some(ref gp) = graph_processor {
            for text in &episode_texts {
                match gp.process(&self.storage, text, &options.user_id) {
                    Ok(graph_result) => {
                        record.entities_created += graph_result.entities.len() as u32;
                        record.relations_created += graph_result.relations.len() as u32;

                        // Entity-memory linking via Aho-Corasick
                        if !graph_result.entities.is_empty() && !new_memories.is_empty() {
                            let entity_names: Vec<String> = graph_result
                                .entities
                                .iter()
                                .map(|e| e.name.clone())
                                .collect();
                            let entity_index =
                                crate::entity_index::EntityIndex::build(&entity_names);

                            for memory in &new_memories {
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
                                            &options.user_id,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => tracing::warn!("Graph extraction failed for episode: {e}"),
                }
            }
        }

        Ok(())
    }

    /// Reconcile extracted facts against existing memories using LLM-driven
    /// ADD/UPDATE/DELETE decisions. Returns the list of memories created/updated.
    fn reconcile_facts(
        &self,
        llm: &Arc<dyn LlmProvider>,
        facts: &[String],
        fact_meta: &HashMap<String, FactMeta>,
        user_id: &str,
        record: &mut MeditationRecord,
    ) -> Result<Vec<MemoryResult>> {
        // Search existing memories for each fact (5 nearest neighbors)
        let mut all_old_memories: Vec<OldMemory> = Vec::new();
        let search_results = self.batch_search_facts(facts, user_id, 5)?;
        for results in &search_results {
            for r in results {
                if !all_old_memories.iter().any(|om| om.id == r.id) {
                    all_old_memories.push(OldMemory {
                        id: r.id.clone(),
                        text: r.content.clone(),
                    });
                }
            }
        }

        // Map UUIDs to integer indices to prevent LLM hallucination
        let mut idx_to_uuid: HashMap<usize, String> = HashMap::new();
        let indexed_old_memories: Vec<OldMemory> = all_old_memories
            .iter()
            .enumerate()
            .map(|(idx, om)| {
                idx_to_uuid.insert(idx, om.id.clone());
                OldMemory {
                    id: idx.to_string(),
                    text: om.text.clone(),
                }
            })
            .collect();

        // LLM decides ADD/UPDATE/DELETE for each fact
        let update_messages =
            if let Some(ref prompt) = self.config.custom_update_memory_prompt {
                get_update_memory_messages_with_prompt(facts, &indexed_old_memories, prompt)
            } else {
                get_update_memory_messages(facts, &indexed_old_memories)
            };
        let config = StructuredGenConfig {
            base_temperature: Some(0.1),
            max_tokens: Some(2048),
            response_format: Some(ResponseFormat::Json),
            ..Default::default()
        };
        let update_response =
            generate_structured(llm.as_ref(), &update_messages, &config, |raw| {
                parse_update_memory_response(raw)
            })
            .map_err(|e| MemoryError::Llm(e.to_string()))?;

        tracing::info!("Reconciliation returned {} operations", update_response.memory.len());

        // Execute operations
        let mut results = Vec::new();

        for op in update_response.memory {
            match op.event {
                MemoryEvent::Add => {
                    let mut add_opts = AddOptions::new(user_id);
                    if let Some(meta) = fact_meta.get(&op.text) {
                        if let Some(ref et) = meta.event_time {
                            add_opts = add_opts.event_time(et);
                        }
                        if let Some(ref sid) = meta.session_id {
                            add_opts = add_opts.session_id(sid);
                        }
                        add_opts = add_opts.importance(meta.significance);
                    }
                    match self.add(&op.text, add_opts) {
                        Ok(result) => {
                            results.push(result);
                            record.memories_created += 1;
                        }
                        Err(e) => {
                            tracing::warn!(text = %op.text, error = %e, "Failed to add memory, skipping");
                        }
                    }
                }
                MemoryEvent::Update => {
                    let real_id =
                        resolve_memory_id(&op.id, &idx_to_uuid, &all_old_memories);
                    if let Some(id) = real_id {
                        match self.update_trace(&id, &op.text, None) {
                            Ok(result) => {
                                results.push(result);
                                record.memories_updated += 1;
                            }
                            Err(e) => {
                                tracing::warn!(id = %id, error = %e, "Failed to update memory, skipping");
                            }
                        }
                    } else {
                        tracing::warn!(raw_id = %op.id, "LLM returned unresolvable memory ID for UPDATE, skipping");
                    }
                }
                MemoryEvent::Delete => {
                    let real_id =
                        resolve_memory_id(&op.id, &idx_to_uuid, &all_old_memories);
                    if let Some(id) = real_id {
                        if let Err(e) = self.delete_trace(&id) {
                            tracing::warn!(id = %id, error = %e, "Failed to delete memory, skipping");
                        } else {
                            record.conflicts_found += 1;
                        }
                    } else {
                        tracing::warn!(raw_id = %op.id, "LLM returned unresolvable memory ID for DELETE, skipping");
                    }
                }
                MemoryEvent::None => {}
            }
        }

        Ok(results)
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

/// Resolve an LLM-returned ID (integer index) back to a real UUID.
fn resolve_memory_id(
    op_id: &str,
    idx_to_uuid: &HashMap<usize, String>,
    all_old_memories: &[OldMemory],
) -> Option<String> {
    if let Ok(idx) = op_id.parse::<usize>() {
        if let Some(real_id) = idx_to_uuid.get(&idx) {
            return Some(real_id.clone());
        }
    }
    // Fallback: LLM returned the real UUID directly
    if all_old_memories.iter().any(|om| om.id == op_id) {
        return Some(op_id.to_string());
    }
    None
}

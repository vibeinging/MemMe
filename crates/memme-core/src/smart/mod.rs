#[cfg(test)]
mod tests;
pub(crate) mod text_utils;

use std::collections::HashMap;
use std::sync::Arc;

use memme_llm::prompts::{
    get_agent_memory_messages, get_detail_extraction_messages,
    get_fact_retrieval_messages_with_prompt, get_fact_retrieval_messages_with_time,
    get_temporal_fact_extraction_messages, get_update_memory_messages,
    get_update_memory_messages_with_prompt, parse_fact_retrieval_response, parse_temporal_facts,
    parse_update_memory_response, ExtractedFact, MemoryEvent, OldMemory,
};
use memme_llm::{generate_structured, LlmProvider, ResponseFormat, StructuredGenConfig};

use crate::config::ExtractionDepth;
use crate::error::MemoryError;
use crate::memory::MemoryStore;
use crate::types::{AddOptions, MemoryResult};

/// Processor that uses an LLM to extract facts from text and intelligently
/// add, update, or delete memories based on existing content.
pub struct SmartProcessor {
    llm: Arc<dyn LlmProvider>,
    extraction_depth: ExtractionDepth,
}

impl SmartProcessor {
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        Self {
            llm,
            extraction_depth: ExtractionDepth::Standard,
        }
    }

    /// Create a new SmartProcessor with the given extraction depth.
    pub fn with_extraction_depth(llm: Arc<dyn LlmProvider>, depth: ExtractionDepth) -> Self {
        Self {
            llm,
            extraction_depth: depth,
        }
    }

    /// Resolve an ID from the LLM response back to a real UUID.
    ///
    /// The LLM receives integer indices (e.g., "0", "1") instead of real UUIDs,
    /// but it may occasionally return the real UUID anyway (fallback).
    fn resolve_memory_id(
        op_id: &str,
        idx_to_uuid: &HashMap<usize, String>,
        all_old_memories: &[OldMemory],
    ) -> Option<String> {
        // Primary path: parse as integer index and look up the real UUID
        if let Ok(idx) = op_id.parse::<usize>() {
            if let Some(real_id) = idx_to_uuid.get(&idx) {
                return Some(real_id.clone());
            }
        }

        // Fallback: the LLM may have returned the real UUID directly
        if all_old_memories.iter().any(|om| om.id == op_id) {
            return Some(op_id.to_string());
        }

        None
    }

    /// Extract facts from a single chunk of text using the LLM.
    ///
    /// When `conversation_time` is provided, the prompt includes temporal resolution rules
    /// so the LLM resolves "yesterday", "last week" etc. to absolute dates.
    fn extract_facts_single(
        &self,
        text: &str,
        custom_fact_prompt: Option<&str>,
        extract_agent_memory: bool,
        conversation_time: Option<&str>,
    ) -> Result<Vec<String>, MemoryError> {
        let messages = if let Some(prompt) = custom_fact_prompt {
            get_fact_retrieval_messages_with_prompt(text, prompt)
        } else if extract_agent_memory {
            get_agent_memory_messages(text)
        } else {
            get_fact_retrieval_messages_with_time(text, conversation_time)
        };
        let config = StructuredGenConfig {
            base_temperature: Some(0.1),
            max_tokens: Some(2048),
            response_format: Some(ResponseFormat::Json),
            ..Default::default()
        };
        let facts_response = generate_structured(self.llm.as_ref(), &messages, &config, |raw| {
            parse_fact_retrieval_response(raw).map(|r| r.facts)
        })
        .map_err(|e| MemoryError::Llm(e.to_string()))?;
        Ok(facts_response)
    }

    /// Extract facts using sliding window chunking for long texts.
    /// Results are deduplicated by exact string match.
    fn extract_facts_chunked(
        &self,
        text: &str,
        custom_prompt: Option<&str>,
        extract_agent: bool,
        conversation_time: Option<&str>,
    ) -> Result<Vec<String>, MemoryError> {
        let chunks = text_utils::split_into_chunks(text, 2500, 300);
        let mut all_facts = Vec::new();
        for chunk in &chunks {
            let facts =
                self.extract_facts_single(chunk, custom_prompt, extract_agent, conversation_time)?;
            all_facts.extend(facts);
        }
        // Deduplicate by exact match
        all_facts.sort();
        all_facts.dedup();
        Ok(all_facts)
    }

    /// Extract temporal facts from a single chunk of text.
    fn extract_temporal_facts_single(
        &self,
        text: &str,
        conversation_time: Option<&str>,
    ) -> Result<Vec<ExtractedFact>, MemoryError> {
        let messages = get_temporal_fact_extraction_messages(text, conversation_time);
        let config = StructuredGenConfig {
            base_temperature: Some(0.1),
            max_tokens: Some(2048),
            response_format: Some(ResponseFormat::Json),
            ..Default::default()
        };
        generate_structured(self.llm.as_ref(), &messages, &config, |raw| {
            parse_temporal_facts(raw)
        })
        .map_err(|e| MemoryError::Llm(e.to_string()))
    }

    /// Second-pass detail extraction focused on specific numbers, names, dates, titles.
    fn extract_details(&self, text: &str) -> Result<Vec<String>, MemoryError> {
        let messages = get_detail_extraction_messages(text);
        let config = StructuredGenConfig {
            base_temperature: Some(0.1),
            max_tokens: Some(2048),
            response_format: Some(ResponseFormat::Json),
            ..Default::default()
        };
        let facts = generate_structured(self.llm.as_ref(), &messages, &config, |raw| {
            parse_fact_retrieval_response(raw).map(|r| r.facts)
        })
        .map_err(|e| MemoryError::Llm(e.to_string()))?;
        Ok(facts)
    }

    /// Process text through the LLM to extract facts and reconcile them
    /// with existing memories in the store.
    ///
    /// `run_id` — optional run scope (currently unused, reserved for future use).
    /// `custom_fact_prompt` — optional custom system prompt for fact extraction.
    /// `custom_update_prompt` — optional custom system prompt for update-memory.
    /// `extract_agent_memory` — if true, use agent memory extraction prompt
    ///   (extracts facts about the AI assistant from assistant messages).
    #[allow(clippy::too_many_arguments)]
    pub fn process(
        &self,
        store: &MemoryStore,
        text: &str,
        user_id: &str,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        metadata: Option<serde_json::Value>,
        custom_fact_prompt: Option<&str>,
        custom_update_prompt: Option<&str>,
        extract_agent_memory: bool,
    ) -> Result<Vec<MemoryResult>, MemoryError> {
        // 1. Extract facts from text using LLM
        let use_thorough = self.extraction_depth == ExtractionDepth::Thorough;

        // Detect conversation timestamp from [Date: ...] prefix.
        // This is passed to the standard extraction prompt so the LLM can resolve
        // relative time references ("yesterday" → absolute date).
        let conversation_time = text_utils::extract_date_prefix(text);
        let conv_time_ref = conversation_time.as_deref();

        // Track facts with optional event_time for temporal-aware storage.
        // In Thorough mode, use the dedicated temporal extraction for structured event_time.
        // In Standard mode, use the enhanced standard prompt (with temporal rules when date prefix exists).
        let mut temporal_facts: Vec<ExtractedFact> = Vec::new();

        if use_thorough && text_utils::looks_like_dialogue(text) {
            let dialogue_chunks = text_utils::split_dialogue_turns(text);
            if !dialogue_chunks.is_empty() {
                for chunk in &dialogue_chunks {
                    let full_text = if chunk.context.is_empty() {
                        chunk.content.clone()
                    } else {
                        format!("{}\n{}", chunk.context, chunk.content)
                    };
                    let ts = chunk.timestamp.as_deref().or(conv_time_ref);
                    let extracted = self.extract_temporal_facts_single(&full_text, ts)?;
                    temporal_facts.extend(extracted);
                }
            } else {
                let extracted = self.extract_temporal_facts_single(text, conv_time_ref)?;
                temporal_facts.extend(extracted);
            }
        } else if use_thorough && text.len() > 500 {
            let chunks = text_utils::split_into_chunks(text, 2500, 300);
            for chunk in &chunks {
                let extracted = self.extract_temporal_facts_single(chunk, conv_time_ref)?;
                temporal_facts.extend(extracted);
            }
        } else if use_thorough {
            let extracted = self.extract_temporal_facts_single(text, conv_time_ref)?;
            temporal_facts.extend(extracted);
        }

        // Standard mode: use the enhanced standard prompt (one prompt, one LLM call).
        // When conversation_time is provided, the prompt includes temporal resolution rules
        // so dates are naturally embedded in the fact text by the LLM.
        let mut facts: Vec<String>;
        if !temporal_facts.is_empty() {
            facts = temporal_facts.iter().map(|f| f.text.clone()).collect();
        } else {
            facts = if text.len() > 3000 {
                self.extract_facts_chunked(
                    text,
                    custom_fact_prompt,
                    extract_agent_memory,
                    conv_time_ref,
                )?
            } else {
                self.extract_facts_single(
                    text,
                    custom_fact_prompt,
                    extract_agent_memory,
                    conv_time_ref,
                )?
            };
        }

        // Detail extraction pass (thorough mode only)
        if use_thorough {
            let details = self.extract_details(text)?;
            for detail in details {
                if !facts.contains(&detail) {
                    facts.push(detail);
                }
            }
        }

        // Deduplicate facts
        facts.sort();
        facts.dedup();

        if facts.is_empty() {
            return Ok(vec![]);
        }

        // Build a lookup from fact text to event_time.
        // Store both the original text AND the date-appended version as keys,
        // so we can match regardless of how the LLM rephrases.
        let mut fact_to_time: HashMap<String, Option<String>> = HashMap::new();
        for f in &temporal_facts {
            fact_to_time.insert(f.text.clone(), f.event_time.clone());
            // Also insert the date-appended version
            if let Some(ref time) = f.event_time {
                if !f.text.contains(time) {
                    fact_to_time.insert(format!("{} on {}", f.text, time), f.event_time.clone());
                }
            }
        }

        // For standard extraction: extract dates from fact text via pattern matching.
        // The enhanced standard prompt embeds dates like "on 2023-05-07" in fact text.
        if temporal_facts.is_empty() && conversation_time.is_some() {
            for fact in &facts {
                let extracted_date = text_utils::extract_iso_date_from_text(fact);
                if let Some(date) = extracted_date {
                    fact_to_time.insert(fact.clone(), Some(date));
                }
            }
        }

        // 2. Search existing memories for each fact to find potential duplicates/updates
        //    Uses batch_search to minimize embedding API calls
        let mut all_old_memories: Vec<OldMemory> = Vec::new();
        let search_results = store.batch_search_facts(&facts, user_id, 5)?;
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

        // 3. Map UUIDs to integer indices to prevent LLM hallucination.
        // The LLM sees "0", "1", "2" instead of real UUIDs, which prevents
        // it from hallucinating wrong UUIDs in its response.
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

        // 4. Ask LLM to decide what to do with each fact vs existing memories
        let update_messages = if let Some(prompt) = custom_update_prompt {
            get_update_memory_messages_with_prompt(&facts, &indexed_old_memories, prompt)
        } else {
            get_update_memory_messages(&facts, &indexed_old_memories)
        };
        let config = StructuredGenConfig {
            base_temperature: Some(0.1),
            max_tokens: Some(2048),
            response_format: Some(ResponseFormat::Json),
            ..Default::default()
        };
        let update_response =
            generate_structured(self.llm.as_ref(), &update_messages, &config, |raw| {
                parse_update_memory_response(raw)
            })
            .map_err(|e| MemoryError::Llm(e.to_string()))?;

        // 5. Execute operations, mapping integer indices back to real UUIDs.
        //    Individual operation failures are logged and skipped (best-effort),
        //    so a single bad fact doesn't block the entire batch.
        let mut results = Vec::new();

        for op in update_response.memory {
            match op.event {
                MemoryEvent::Add => {
                    let mut add_opts = AddOptions::new(user_id);
                    if let Some(aid) = agent_id {
                        add_opts = add_opts.agent_id(aid);
                    }
                    if let Some(rid) = run_id {
                        add_opts = add_opts.run_id(rid);
                    }
                    if let Some(ref meta) = metadata {
                        add_opts = add_opts.metadata(meta.clone());
                    }
                    // Attach event_time from temporal extraction if available.
                    // Try exact match first, then fuzzy substring match as fallback
                    // (the LLM may rephrase the fact text slightly).
                    let event_time =
                        fact_to_time
                            .get(&op.text)
                            .and_then(|t| t.clone())
                            .or_else(|| {
                                temporal_facts
                                    .iter()
                                    .filter(|f| f.event_time.is_some())
                                    .find(|f| {
                                        op.text.contains(&f.text) || f.text.contains(&op.text)
                                    })
                                    .and_then(|f| f.event_time.clone())
                            });
                    if let Some(et) = event_time {
                        add_opts = add_opts.event_time(et);
                    }
                    match store.add(&op.text, add_opts) {
                        Ok(result) => results.push(result),
                        Err(e) => {
                            tracing::warn!(text = %op.text, error = %e, "Failed to add memory, skipping");
                        }
                    }
                }
                MemoryEvent::Update => {
                    let real_id = Self::resolve_memory_id(&op.id, &idx_to_uuid, &all_old_memories);
                    if let Some(id) = real_id {
                        match store.update_trace(&id, &op.text, None) {
                            Ok(result) => results.push(result),
                            Err(e) => {
                                tracing::warn!(id = %id, error = %e, "Failed to update memory, skipping");
                            }
                        }
                    } else {
                        tracing::warn!(raw_id = %op.id, "LLM returned unresolvable memory ID for UPDATE, skipping");
                    }
                }
                MemoryEvent::Delete => {
                    let real_id = Self::resolve_memory_id(&op.id, &idx_to_uuid, &all_old_memories);
                    if let Some(id) = real_id {
                        if let Err(e) = store.delete_trace(&id) {
                            tracing::warn!(id = %id, error = %e, "Failed to delete memory, skipping");
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
}

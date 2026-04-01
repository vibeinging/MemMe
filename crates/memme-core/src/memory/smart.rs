use std::sync::Arc;

use crate::config::ExtractionDepth;
use crate::error::Result;
use crate::types::*;

use super::helpers::recover_lock;

/// Identify "information-rich" sentences from input text.
///
/// A sentence is considered information-rich if it contains:
/// - Quoted text (single or double quotes)
/// - Numbers (digits)
/// - Proper nouns (words starting with uppercase, not at sentence start)
fn extract_information_rich_sentences(text: &str) -> Vec<String> {
    let mut results = Vec::new();

    for line in text.lines() {
        for sentence in line.split(['.', '!', '?']) {
            let trimmed = sentence.trim();
            if trimmed.is_empty() || trimmed.len() < 10 {
                continue;
            }

            let has_quotes = trimmed.contains('"') || trimmed.contains('\'');
            let has_numbers = trimmed.chars().any(|c| c.is_ascii_digit());

            // Check for proper nouns: uppercase words not at the start of the sentence
            let words: Vec<&str> = trimmed.split_whitespace().collect();
            let has_proper_noun = words.iter().skip(1).any(|w| {
                w.chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
                    // Filter out common sentence-start words after colons, etc.
                    && w.len() > 1
            });

            if has_quotes || has_numbers || has_proper_noun {
                results.push(trimmed.to_string());
            }
        }
    }

    results
}

impl super::MemoryStore {
    /// **Advanced** — Most users should use `append_events()` + `compact()` instead.
    ///
    /// Smart add: use an LLM to extract facts from the text and
    /// intelligently add, update, or delete memories.
    ///
    /// Accepts optional `run_id` for scoping and optional custom prompts
    /// from the config.
    ///
    /// When `config.enable_graph` is true, also extracts entities and
    /// relationships into the knowledge graph using the provided LLM.
    #[allow(clippy::too_many_arguments)]
    pub fn add_smart(
        &self,
        text: &str,
        user_id: &str,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        metadata: Option<serde_json::Value>,
        llm: Arc<dyn memme_llm::LlmProvider>,
        extract_agent_memory: bool,
    ) -> Result<SmartAddResult> {
        // Begin transaction — all writes rollback on error.
        let conn = self.storage.write_conn();
        conn.execute_batch("BEGIN TRANSACTION")
            .map_err(crate::error::MemoryError::DuckDb)?;
        drop(conn);

        let result = self.add_smart_inner(
            text,
            user_id,
            agent_id,
            run_id,
            metadata,
            llm,
            extract_agent_memory,
        );

        let conn = self.storage.write_conn();
        match &result {
            Ok(_) => {
                conn.execute_batch("COMMIT")
                    .map_err(crate::error::MemoryError::DuckDb)?;
            }
            Err(_) => {
                let _ = conn.execute_batch("ROLLBACK");
            }
        }

        result
    }

    /// Inner implementation of add_smart, called within a transaction.
    #[allow(clippy::too_many_arguments)]
    fn add_smart_inner(
        &self,
        text: &str,
        user_id: &str,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        metadata: Option<serde_json::Value>,
        llm: Arc<dyn memme_llm::LlmProvider>,
        extract_agent_memory: bool,
    ) -> Result<SmartAddResult> {
        // 1. Extract facts and manage memories (existing logic)
        let processor = crate::smart::SmartProcessor::with_extraction_depth(
            llm.clone(),
            self.config.extraction_depth.clone(),
        );
        let memories = processor.process(
            self,
            text,
            user_id,
            agent_id,
            run_id,
            metadata.clone(),
            self.config.custom_fact_extraction_prompt.as_deref(),
            self.config.custom_update_memory_prompt.as_deref(),
            extract_agent_memory,
        )?;

        // 2. In Thorough mode, also store information-rich raw dialogue phrases
        if self.config.extraction_depth == ExtractionDepth::Thorough {
            let rich_sentences = extract_information_rich_sentences(text);
            for sentence in &rich_sentences {
                let mut raw_meta = metadata
                    .as_ref()
                    .and_then(|m| m.as_object().cloned())
                    .unwrap_or_default();
                raw_meta.insert(
                    "type".to_string(),
                    serde_json::Value::String("raw_dialogue".to_string()),
                );
                let mut add_opts =
                    AddOptions::new(user_id).metadata(serde_json::Value::Object(raw_meta));
                if let Some(aid) = agent_id {
                    add_opts = add_opts.agent_id(aid);
                }
                if let Some(rid) = run_id {
                    add_opts = add_opts.run_id(rid);
                }
                // Best-effort: ignore errors for raw dialogue storage
                let _ = self.add(sentence, add_opts);
            }
        }

        // 3. If enable_graph, also extract entities/relationships
        let graph = if self.config.enable_graph {
            let graph_processor = crate::graph::GraphProcessor::new(llm);
            let graph_result = graph_processor.process(&self.storage, text, user_id)?;

            // 4. Build entity-memory links using Aho-Corasick matching
            if !graph_result.entities.is_empty() && !memories.is_empty() {
                let entity_names: Vec<String> = graph_result
                    .entities
                    .iter()
                    .map(|e| e.name.clone())
                    .collect();
                let entity_index = crate::entity_index::EntityIndex::build(&entity_names);

                for memory in &memories {
                    let matched = entity_index.extract(&memory.content);
                    for matched_name in &matched {
                        // Find the entity ID for this name
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

            Some(graph_result)
        } else {
            None
        };

        Ok(SmartAddResult {
            memories,
            graph,
            episode_id: None,
            session_id: None,
        })
    }

    /// Get the internally configured LLM provider, or return an error.
    pub(crate) fn require_llm(&self) -> Result<std::sync::Arc<dyn memme_llm::LlmProvider>> {
        recover_lock(&self.llm, "llm")
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                crate::error::MemoryError::Config(
                    "No LLM configured. Call set_llm() first or set MEMME_LLM_API_KEY env var."
                        .into(),
                )
            })
    }

    /// **Advanced** — Most users should use `append_events()` + `compact()` instead.
    ///
    /// Add graph entries using the internally configured LLM.
    pub fn add_graph_auto(&self, text: &str, user_id: &str) -> Result<GraphSearchResult> {
        self.add_graph(text, user_id, self.require_llm()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_information_rich_with_numbers() {
        let text = "I have 3 children and we live in a house.";
        let results = extract_information_rich_sentences(text);
        assert!(!results.is_empty(), "Should detect sentence with numbers");
    }

    #[test]
    fn test_extract_information_rich_with_proper_nouns() {
        let text = "I went to visit Paris last summer.";
        let results = extract_information_rich_sentences(text);
        assert!(
            !results.is_empty(),
            "Should detect sentence with proper noun Paris"
        );
    }

    #[test]
    fn test_extract_information_rich_with_quotes() {
        let text = "She said \"I love this book\" yesterday.";
        let results = extract_information_rich_sentences(text);
        assert!(
            !results.is_empty(),
            "Should detect sentence with quoted text"
        );
    }

    #[test]
    fn test_extract_information_rich_empty() {
        let text = "hello world and nothing special here.";
        let results = extract_information_rich_sentences(text);
        assert!(
            results.is_empty(),
            "Should not detect anything in plain lowercase text"
        );
    }

    #[test]
    fn test_extract_information_rich_short_sentences_skipped() {
        let text = "Hi. Ok. Yes.";
        let results = extract_information_rich_sentences(text);
        assert!(results.is_empty(), "Short sentences should be skipped");
    }

    #[test]
    fn test_extract_information_rich_multiline() {
        let text = "User: I read 'Becoming Nicole' last week.\nAssistant: That sounds interesting.";
        let results = extract_information_rich_sentences(text);
        // Should detect the line with quotes and proper noun
        assert!(!results.is_empty());
    }
}

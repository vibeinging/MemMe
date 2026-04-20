use std::sync::Arc;

use crate::error::MemoryError;
use crate::error::Result;
use crate::storage::InsertMemoryParams;
#[allow(unused_imports)]
use crate::types::*;

impl super::MemoryStore {
    /// # Core API — Primary ingestion method
    ///
    /// Send events (messages, actions, locations, etc.) to MemMe.
    /// Events are stored in a Session. The returned `compact_needed` flag
    /// indicates when unprocessed events exceed `compact_threshold`, but
    /// compact is **never** triggered automatically — call `compact()` or
    /// `process_background()` explicitly when ready.
    ///
    /// This is the recommended way to add data to MemMe. For most use cases,
    /// you only need three methods: `append_events()`, `search()`, and `compact()`.
    ///
    /// # Arguments
    /// * `session_id` - Session to append to (created if not exists)
    /// * `messages` - Chat messages to store as events
    /// * `user_id` - User who owns this session
    /// * `metadata` - Optional metadata for the events
    pub fn append_events(
        &self,
        session_id: &str,
        messages: &[ChatMessage],
        user_id: &str,
        metadata: Option<serde_json::Value>,
    ) -> Result<AppendEventsResult> {
        let non_system: Vec<&ChatMessage> =
            messages.iter().filter(|m| m.role != "system").collect();

        if non_system.is_empty() {
            return Ok(AppendEventsResult {
                session_id: session_id.to_string(),
                events_appended: 0,
                total_unprocessed: 0,
                compact_needed: false,
            });
        }

        // Ensure session exists
        let now = chrono::Utc::now().to_rfc3339();
        let meta_str = metadata
            .as_ref()
            .map(|m| serde_json::to_string(m).unwrap_or_default());
        self.storage
            .get_or_create_session(session_id, user_id, None, &now, meta_str.as_deref())?;

        // Batch embed all messages in one API call, then insert events with embeddings.
        // This makes events immediately searchable (V3 store-first principle).
        let texts: Vec<&str> = non_system.iter().map(|m| m.content.as_str()).collect();
        let embeddings = self
            .embedder
            .embed_batch(&texts)
            .unwrap_or_else(|_| vec![vec![]; texts.len()]);

        let mut notes_batch = String::new();
        for (i, msg) in non_system.iter().enumerate() {
            let event_type = match msg.role.as_str() {
                "user" => "user_message",
                "assistant" => "ai_response",
                "tool" => "tool_result",
                _ => "system",
            };
            let mut opts = IngestEventOptions::new(user_id)
                .session_id(session_id)
                .event_type(event_type);
            if let Some(ts) = &msg.timestamp {
                opts = opts.timestamp(ts);
            }
            if let Some(ref m) = metadata {
                opts = opts.metadata(m.clone());
            }

            let event_id = uuid::Uuid::new_v4().to_string();
            let emb = if i < embeddings.len() { &embeddings[i] } else { &[] as &[f32] };
            self.storage.insert_event(&event_id, &msg.content, emb, &opts)?;

            // Accumulate structured note
            let preview: String = msg.content.chars().take(120).collect();
            let ts = msg.timestamp.as_deref().unwrap_or(&now);
            use std::fmt::Write;
            let _ = writeln!(notes_batch, "[{}] {}: {}", ts, msg.role, preview);
        }

        // Batch-append all notes in a single SQL UPDATE
        if !notes_batch.is_empty() {
            let _ = self
                .storage
                .append_structured_note(session_id, &notes_batch);
        }

        let total_unprocessed = self
            .storage
            .count_unprocessed_events_in_session(session_id)?;

        // Check whether compact is advisable (informational only — never auto-triggered).
        let threshold = self.config.tuning.compact_threshold;
        let compact_needed = threshold > 0 && total_unprocessed >= threshold as u64;

        // Build co-occurrence graph from entities detected in the new messages.
        // Zero-LLM: uses Aho-Corasick on known entities + simple heuristics
        // for candidate entities (quoted strings, capitalized sequences).
        if self.config.enable_graph {
            let contents: Vec<&str> = non_system.iter().map(|m| m.content.as_str()).collect();
            if let Err(e) = self.build_cooccurrence_graph(&contents, user_id) {
                tracing::warn!("Co-occurrence graph build failed: {e}");
            }
        }

        let result = AppendEventsResult {
            session_id: session_id.to_string(),
            events_appended: non_system.len(),
            total_unprocessed,
            compact_needed,
        };

        // Process one background task opportunistically (non-blocking).
        if self.has_llm() {
            let _ = self.process_background();
        }

        Ok(result)
    }

    /// # Core API — Compact a session
    ///
    /// Process unprocessed events in a session: purify events via LLM,
    /// create an Episode summary, and mark events as processed.
    ///
    /// This is **never** auto-triggered. Call it explicitly when a conversation
    /// ends, or rely on `process_background()` which schedules compaction for
    /// sessions that have been hit by search queries.
    ///
    /// This is the recommended way to process data in MemMe. For most use cases,
    /// you only need three methods: `append_events()`, `search()`, and `compact()`.
    ///
    /// Uses the internally configured LLM. Call `set_llm()` first.
    pub fn compact(&self, session_id: &str) -> Result<CompactResult> {
        self.compact_inner(session_id, self.require_llm()?, true)
    }

    /// Compact without FTS rebuild (used by re_traces to batch rebuild at end).
    #[allow(dead_code)]
    fn compact_no_fts(&self, session_id: &str) -> Result<CompactResult> {
        self.compact_inner(session_id, self.require_llm()?, false)
    }

    /// Internal compact implementation — purifies events, generates embeddings, does NOT extract memories.
    fn compact_inner(
        &self,
        session_id: &str,
        llm: Arc<dyn memme_llm::LlmProvider>,
        rebuild_fts: bool,
    ) -> Result<CompactResult> {
        let session = self
            .storage
            .get_session(session_id)?
            .ok_or_else(|| MemoryError::NotFound(format!("Session {session_id}")))?;

        let events = self.storage.get_unprocessed_events_in_session(session_id)?;
        if events.is_empty() {
            return Ok(CompactResult {
                session_id: session_id.to_string(),
                episode_id: String::new(),
                memories: vec![],
                graph: None,
                events_processed: 0,
            });
        }

        let event_ids: Vec<String> = events.iter().map(|e| e.event_id.clone()).collect();

        // Short-session optimization: skip LLM for low-token sessions
        let estimated_tokens: usize = events
            .iter()
            .map(|e| super::helpers::estimate_tokens(&e.content))
            .sum();
        let use_fallback = self.config.tuning.compact_fallback_token_threshold > 0
            && estimated_tokens < self.config.tuning.compact_fallback_token_threshold;

        // Purify events + generate episode summary in a single LLM call
        let (purified, title, summary, significance, prospective_queries) = if use_fallback {
            let purified = fallback_purify_events(&events);
            let (t, s, sig) = match session.structured_notes {
                Some(ref notes) if !notes.trim().is_empty() => {
                    let title_text = extract_title_from_events(&events);
                    let summary_text: String = notes.chars().take(500).collect();
                    (title_text, summary_text, 0.5)
                }
                _ => fallback_episode_summary(&events),
            };
            (purified, t, s, sig, vec![])
        } else {
            compact_with_llm(&events, &llm)
        };

        // Fall back to original content if purification returned empty strings
        let purified: Vec<_> = purified
            .into_iter()
            .zip(events.iter())
            .map(|(mut p, ev)| {
                if p.purified_content.trim().is_empty() {
                    p.purified_content = ev.content.clone();
                }
                p
            })
            .collect();

        // Embed purified content (batch for efficiency)
        let purified_texts: Vec<&str> = purified
            .iter()
            .map(|p| p.purified_content.as_str())
            .collect();
        let embeddings = self
            .embedder
            .embed_batch(&purified_texts)
            .map_err(MemoryError::Embedding)?;

        // Persist purified content + embeddings back to events
        for ((event, p), emb) in events.iter().zip(purified.iter()).zip(embeddings.iter()) {
            self.storage.update_event_embedding(
                &event.event_id,
                &p.purified_content,
                emb,
                p.event_time.as_deref(),
                p.location.as_deref(),
            )?;
        }

        // Insert narrative trace into memories table
        // Append prospective queries to narrative content for BM25 + vector search matching
        let narrative_content = if prospective_queries.is_empty() {
            format!("{}: {}", title, summary)
        } else {
            format!(
                "{}: {} [Prospective: {}]",
                title,
                summary,
                prospective_queries.join(" ")
            )
        };
        let narrative_embedding = self
            .embedder
            .embed(&narrative_content)
            .map_err(MemoryError::Embedding)?;

        // Create Episode record first (less critical — if narrative insert fails,
        // the episode is still usable by meditation; reverse order would leave orphan narratives)
        let episode_id = uuid::Uuid::new_v4().to_string();
        let episode_opts = crate::types::CreateEpisodeOptions::new(
            &title,
            &summary,
            &session.user_id,
            &events[0].timestamp,
        )
        .ended_at(&events[events.len() - 1].timestamp)
        .significance(significance)
        .outcome("completed")
        .event_ids(event_ids.clone())
        .session_ids(vec![session_id.to_string()]);

        self.storage.insert_episode(
            &episode_id,
            &title,
            &summary,
            &narrative_embedding,
            &episode_opts,
        )?;

        // Insert narrative trace into memories table
        let narrative_id = uuid::Uuid::new_v4().to_string();
        let narrative_hash = crate::memory::helpers::content_hash(&narrative_content);
        let meta_json = serde_json::json!({
            "title": title,
            "summary": summary,
            "event_ids": event_ids,
            "outcome": "completed",
            "started_at": events[0].timestamp,
            "ended_at": events[events.len() - 1].timestamp,
        })
        .to_string();
        self.storage.insert_memory(
            &narrative_id,
            &narrative_content,
            &narrative_embedding,
            &session.user_id,
            &narrative_hash,
            &InsertMemoryParams {
                metadata: Some(meta_json),
                importance: Some(significance),
                immutable: true,
                event_time: Some(events[0].timestamp.clone()),
                session_id: Some(session_id.to_string()),
                resolution: Resolution::Narrative,
                ..Default::default()
            },
        )?;

        let event_id_refs: Vec<&str> = event_ids.iter().map(|s| s.as_str()).collect();
        self.storage.mark_events_processed(&event_id_refs)?;

        // Clear structured notes after compact to avoid stale content in future compacts
        let _ = self.storage.clear_structured_notes(session_id);

        if rebuild_fts {
            let _ = self.storage.create_fts_index();
            let _ = self.storage.create_fts_index_events();
        }

        // Note: No memories extracted here — use meditate() for that
        Ok(CompactResult {
            session_id: session_id.to_string(),
            episode_id,
            memories: vec![], // Empty — memories are extracted by meditate()
            graph: None,      // Graph extraction moved to meditate()
            events_processed: event_ids.len(),
        })
    }

    /// # Core API — Re-extract all traces from raw sessions
    ///
    /// Deletes all existing traces and re-runs compact on every session
    /// using the current LLM. Use this after upgrading to a better model
    /// to get higher-quality memory extraction from the same raw data.
    ///
    /// Sessions and events are preserved (they are immutable recordings).
    /// Only the derived traces (facts, summaries, identity) are regenerated.
    #[allow(dead_code)]
    pub(crate) fn re_traces(&self, user_id: &str) -> Result<Vec<CompactResult>> {
        // 1. Delete all existing traces and episodes for this user
        self.delete_all_traces(user_id, None, None, None)?;
        self.storage.delete_episodes_for_user(user_id)?;

        // 2. Get all sessions for this user
        let sessions = self.list_sessions(ListSessionsOptions::new(user_id))?;

        // 3. Re-compact each session (skip per-session FTS rebuild)
        let mut results = Vec::new();
        for session in &sessions {
            self.storage.reset_events_processed(&session.session_id)?;
            let _ = self.storage.clear_structured_notes(&session.session_id);

            match self.compact_no_fts(&session.session_id) {
                Ok(result) => results.push(result),
                Err(e) => {
                    tracing::warn!(
                        "re_traces: failed to compact session {}: {e}",
                        session.session_id
                    );
                }
            }
        }

        // 4. Rebuild FTS index once at the end
        let _ = self.storage.create_fts_index();

        Ok(results)
    }
}

/// Generate episode title, summary, and significance.
/// Tries LLM first, falls back to truncation-based approach.
/// Combined purification + summarization in a single LLM call.
/// Returns (purified_events, title, summary, significance, prospective_queries).
fn compact_with_llm(
    events: &[Event],
    llm: &Arc<dyn memme_llm::LlmProvider>,
) -> (Vec<PurifiedEvent>, String, String, f32, Vec<String>) {
    let context = events
        .iter()
        .enumerate()
        .map(|(i, e)| format!("[{}] {}: {}", i + 1, e.event_type.as_str(), e.content))
        .collect::<Vec<_>>()
        .join("\n");

    let conversation_time = events.first().map(|e| e.timestamp.as_str()).unwrap_or("");

    let prompt = format!(
        r#"Process this conversation and return a single JSON object with TWO sections:

## Section 1: Purified Messages
For each message, resolve coreferences (pronouns → names), ground temporal references (relative → absolute dates using {conversation_time} as anchor), and extract locations.

## Section 2: Episode Summary
Summarize the entire conversation as a title, summary, and significance score.

## Section 3: Implication Queries (Prospective Indexing)
Generate 5 SHORT search queries (3-8 words each) that would help find this memory in the FUTURE when someone is in a related situation. Think BEYOND the literal content:
- What implicit constraints or decisions were established?
- What life situations or problems would this memory be relevant to?
- What emotional states or challenges connect to this topic?
- What would someone search for if they face a similar situation but use completely different words?

**Input** ({count} messages):
{context}

**Output format**:
```json
{{
  "purified": [
    {{"content": "Purified text with pronouns resolved", "event_time": "YYYY-MM-DD" or null, "location": "place" or null}},
    ...
  ],
  "title": "Brief title (max 60 chars)",
  "summary": "2-3 sentence summary of what was discussed",
  "significance": 0.7,
  "prospective_queries": ["nostalgic about a special dinner", "celebrating anniversary restaurant ideas", "meaningful evening with close friend", "favorite dining experiences", "romantic dinner spot recommendation"]
}}
```

**Rules**:
- "purified" must have exactly {count} items, one per input message
- Resolve pronouns to actual names, "there" to actual place
- CRITICAL: Replace ALL relative time expressions in the purified text with absolute dates based on the conversation date above. E.g. "yesterday" → "on YYYY-MM-DD", "last week" → "on YYYY-MM-DD". The purified text must be self-contained — readable without knowing the conversation date.
- If no purification needed, use original content
- "significance": rate the **personal memory value** of this conversation:
  - 0.0-0.2: Generic Q&A, coding help, informational queries with no personal context
  - 0.3-0.5: Mild personal context, routine activities, general preferences mentioned in passing
  - 0.6-0.8: Significant personal events, strong preferences, relationships, plans, goals
  - 0.9-1.0: Life-changing events, core identity revelations, deeply emotional moments
- "prospective_queries": exactly 5 short phrases (3-8 words each, NOT full questions). Each should describe a situation or scenario where this memory would be relevant. Use everyday language, not formal queries. Examples: "feeling overwhelmed with commitments", "struggling to say no to requests", "work-life balance tips". The goal: if someone searches with COMPLETELY DIFFERENT words but a RELATED situation, these phrases should bridge the gap.
- Respond ONLY with JSON, no other text."#,
        conversation_time = conversation_time,
        context = context,
        count = events.len()
    );

    let messages = vec![memme_llm::Message {
        role: memme_llm::MessageRole::User,
        content: prompt,
    }];

    // Scale tokens: ~150 per purified event + 700 base for summary/structure
    let uncapped = events.len() * 150 + 700;
    let scaled_tokens = uncapped.min(16384);
    if uncapped > 16384 {
        tracing::warn!(
            events = events.len(),
            uncapped_tokens = uncapped,
            capped_tokens = scaled_tokens,
            "Compact output budget capped at 16384 tokens — consider chunking for very long sessions"
        );
    }
    let config = memme_llm::StructuredGenConfig {
        base_temperature: Some(0.1),
        max_tokens: Some(scaled_tokens),
        response_format: Some(memme_llm::ResponseFormat::Json),
        ..Default::default()
    };

    let expected_len = events.len();
    match memme_llm::generate_structured(llm.as_ref(), &messages, &config, |raw| {
        let repaired = memme_llm::try_repair_json(raw);
        let parsed: serde_json::Value =
            serde_json::from_str(&repaired).map_err(|e| format!("JSON parse failed: {e}"))?;

        // Parse purified events
        let purified_arr = parsed["purified"]
            .as_array()
            .ok_or("Missing 'purified' array")?;
        let purified: Vec<PurifiedEvent> = purified_arr
            .iter()
            .map(|p| PurifiedEvent {
                purified_content: p["content"].as_str().unwrap_or("").to_string(),
                event_time: p["event_time"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string()),
                location: p["location"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string()),
            })
            .collect();

        // Parse summary
        let title = parsed["title"]
            .as_str()
            .unwrap_or("Conversation")
            .to_string();
        let summary = parsed["summary"].as_str().unwrap_or("").to_string();
        let significance = parsed["significance"].as_f64().unwrap_or(0.5) as f32;

        // Parse prospective queries
        let prospective_queries = parsed["prospective_queries"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok((
            purified,
            title,
            summary,
            significance.clamp(0.0, 1.0),
            prospective_queries,
        ))
    }) {
        Ok((mut purified, title, summary, significance, prospective_queries)) => {
            // Pad or truncate purified to match event count
            purified.resize_with(expected_len, || PurifiedEvent {
                purified_content: String::new(),
                event_time: None,
                location: None,
            });
            (purified, title, summary, significance, prospective_queries)
        }
        Err(e) => {
            tracing::warn!("Combined compact LLM call failed: {e}, using fallback");
            let purified = fallback_purify_events(events);
            let (t, s, sig) = fallback_episode_summary(events);
            (purified, t, s, sig, vec![])
        }
    }
}

/// Extract a title from events: first user message truncated to 60 chars.
fn extract_title_from_events(events: &[Event]) -> String {
    events
        .iter()
        .find(|e| e.event_type == EventType::UserMessage)
        .map(|e| {
            let t: String = e.content.chars().take(60).collect();
            if e.content.chars().count() > 60 {
                format!("{t}...")
            } else {
                t
            }
        })
        .unwrap_or_else(|| "Conversation".to_string())
}

/// Truncation-based fallback when LLM is unavailable.
fn fallback_episode_summary(events: &[Event]) -> (String, String, f32) {
    let title = extract_title_from_events(events);

    let summary = events
        .iter()
        .take(5)
        .map(|e| {
            let content: String = e.content.chars().take(100).collect();
            format!("{}: {}", e.event_type.as_str(), content)
        })
        .collect::<Vec<_>>()
        .join("\n");

    (title, summary, 0.5)
}

// ---------------------------------------------------------------------------
// Event Purification (Coreference Resolution + Temporal/Spatial Grounding)
// ---------------------------------------------------------------------------

/// Result of purifying a single event.
#[derive(Debug, Clone)]
struct PurifiedEvent {
    /// Content with pronouns resolved and references clarified.
    purified_content: String,
    /// Extracted event time (ISO 8601) if mentioned.
    event_time: Option<String>,
    /// Extracted location if mentioned.
    location: Option<String>,
}

/// Fallback purification when LLM is unavailable.
/// Returns original content with no metadata.
fn fallback_purify_events(events: &[Event]) -> Vec<PurifiedEvent> {
    events
        .iter()
        .map(|e| PurifiedEvent {
            purified_content: e.content.clone(),
            event_time: None,
            location: None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Co-occurrence Graph (Zero LLM)
// ---------------------------------------------------------------------------

impl super::MemoryStore {
    /// Build co-occurrence edges from entities found in message contents.
    ///
    /// 1. Detects known entities via the existing Aho-Corasick EntityIndex.
    /// 2. Detects candidate entities via simple heuristics (quoted strings,
    ///    capitalized word sequences) and auto-creates entity nodes for them.
    /// 3. Entities co-occurring in the same message get a `co_occurs` edge.
    /// 4. All entities across the batch (same session) get a weaker
    ///    `session_context` edge.
    pub(crate) fn build_cooccurrence_graph(&self, contents: &[&str], user_id: &str) -> Result<()> {
        use std::collections::HashSet;

        // Build Aho-Corasick index from known entities
        let entity_index = self.get_entity_index(user_id);

        // Per-message entity extraction + co-occurrence edges
        let mut all_session_entities: HashSet<String> = HashSet::new();

        for content in contents {
            let mut entities_in_msg: Vec<String> = Vec::new();

            // 1a. Known entities via Aho-Corasick
            let known = entity_index.extract(content);
            for name in &known {
                entities_in_msg.push(name.to_lowercase());
            }

            // 1b. Candidate entities via heuristics
            let candidates = extract_candidate_entities(content);
            for name in &candidates {
                let lower = name.to_lowercase();
                if entities_in_msg.contains(&lower) {
                    continue;
                }
                // Auto-create entity node (idempotent — skipped if already exists)
                let existing = self.storage.find_entity_by_name(name, user_id)?;
                if existing.is_none() {
                    let id = uuid::Uuid::new_v4().to_string();
                    self.storage
                        .upsert_entity(&id, name, Some("candidate"), user_id)?;
                }
                entities_in_msg.push(lower);
            }

            // De-duplicate and cap at 5 entities per message to avoid O(n²) explosion
            let unique: Vec<String> = {
                let mut seen = HashSet::new();
                entities_in_msg
                    .into_iter()
                    .filter(|e| seen.insert(e.clone()))
                    .take(5)
                    .collect()
            };

            // 2. Co-occurrence edges: every pair in the same message (max C(5,2)=10 edges)
            for i in 0..unique.len() {
                for j in (i + 1)..unique.len() {
                    let _ = self.ensure_relationship(&unique[i], &unique[j], "co_occurs", user_id);
                }
            }

            for e in &unique {
                all_session_entities.insert(e.clone());
            }
        }

        // Session-context edges removed: too many combinations (O(n²) on all session
        // entities) and low signal-to-noise ratio. Co-occurrence within individual
        // messages provides sufficient relationship signal.
        let _ = &all_session_entities; // suppress unused warning

        Ok(())
    }

    /// Ensure a relationship exists between two entities (by name).
    fn ensure_relationship(
        &self,
        name_a: &str,
        name_b: &str,
        relation_type: &str,
        user_id: &str,
    ) -> Result<()> {
        let id_a = self.resolve_entity_id(name_a, user_id)?;
        let id_b = self.resolve_entity_id(name_b, user_id)?;

        let rel_id = uuid::Uuid::new_v4().to_string();
        self.storage
            .insert_relationship(&rel_id, &id_a, &id_b, relation_type, user_id, None)?;
        Ok(())
    }

    /// Resolve an entity name to its ID, creating the entity if it doesn't exist.
    fn resolve_entity_id(&self, name: &str, user_id: &str) -> Result<String> {
        if let Some((id, _, _)) = self.storage.find_entity_by_name(name, user_id)? {
            Ok(id)
        } else {
            let id = uuid::Uuid::new_v4().to_string();
            self.storage
                .upsert_entity(&id, name, Some("candidate"), user_id)?;
            Ok(id)
        }
    }
}

/// Extract candidate entity names from text using simple heuristics (no LLM).
///
/// Detects:
/// - Quoted strings (double quotes and Chinese quotes)
/// - Capitalized word sequences (e.g. "San Francisco", "Project Alpha")
fn extract_candidate_entities(text: &str) -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // 1. Quoted strings
    let quote_pairs: &[(&str, &str)] = &[
        ("\"", "\""),
        ("\u{201c}", "\u{201d}"),
        ("\u{300c}", "\u{300d}"),
    ];
    for &(open, close) in quote_pairs {
        let mut search_from = 0;
        while let Some(start) = text[search_from..].find(open) {
            let abs_start = search_from + start + open.len();
            if abs_start >= text.len() {
                break;
            }
            if let Some(end) = text[abs_start..].find(close) {
                let inner = text[abs_start..abs_start + end].trim();
                if inner.len() >= 2 && inner.len() <= 50 && !inner.contains('\n') {
                    let key = inner.to_lowercase();
                    if seen.insert(key) {
                        candidates.push(inner.to_string());
                    }
                }
                search_from = abs_start + end + close.len();
            } else {
                break;
            }
        }
    }

    // 2. Capitalized word sequences (English), not at sentence start
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut i = 0;
    while i < words.len() {
        let word = words[i];
        let first_char = word.chars().next();
        let is_upper = first_char.is_some_and(|c| c.is_uppercase());

        if is_upper && i > 0 {
            let stripped = word
                .trim_start_matches(|c: char| !c.is_alphanumeric())
                .trim_end_matches(|c: char| !c.is_alphanumeric());
            if is_common_word(stripped) {
                i += 1;
                continue;
            }

            let start = i;
            let mut end = i + 1;
            while end < words.len() && end - start < 4 {
                let w = words[end];
                let fc = w.chars().next();
                if fc.is_some_and(|c| c.is_uppercase()) {
                    end += 1;
                } else {
                    break;
                }
            }

            let phrase: String = words[start..end]
                .iter()
                .map(|w| {
                    w.trim_start_matches(|c: char| !c.is_alphanumeric())
                        .trim_end_matches(|c: char| !c.is_alphanumeric())
                })
                .collect::<Vec<_>>()
                .join(" ");

            if phrase.len() >= 2 && !is_common_word(&phrase) {
                let key = phrase.to_lowercase();
                if seen.insert(key) {
                    candidates.push(phrase);
                }
            }
            i = end;
        } else {
            i += 1;
        }
    }

    candidates
}

/// Common English words that should not be treated as entity names.
fn is_common_word(word: &str) -> bool {
    let lower = word.to_lowercase();
    matches!(
        lower.as_str(),
        "the"
            | "a"
            | "an"
            | "is"
            | "are"
            | "was"
            | "were"
            | "it"
            | "this"
            | "that"
            | "i"
            | "my"
            | "me"
            | "we"
            | "you"
            | "he"
            | "she"
            | "they"
            | "yes"
            | "no"
            | "ok"
            | "hi"
            | "hello"
            | "hey"
            | "sure"
            | "what"
            | "when"
            | "where"
            | "how"
            | "why"
            | "who"
            | "which"
            | "do"
            | "does"
            | "did"
            | "can"
            | "could"
            | "would"
            | "should"
            | "will"
            | "have"
            | "has"
            | "had"
            | "been"
            | "be"
            | "not"
            | "but"
            | "and"
            | "or"
            | "if"
            | "so"
            | "then"
            | "also"
            | "just"
            | "very"
    )
}

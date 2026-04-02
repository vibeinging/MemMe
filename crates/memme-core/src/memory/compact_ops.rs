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
    /// Events are stored in a Session and automatically compacted when
    /// the threshold is exceeded (`compact_threshold` in config).
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

        // Ingest each message as an Event
        for msg in &non_system {
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
            self.ingest_event(&msg.content, opts)?;
        }

        let total_unprocessed = self
            .storage
            .count_unprocessed_events_in_session(session_id)?;

        // Check if compact is needed (never auto-compact — let the caller decide)
        let threshold = self.config.compact_threshold;
        let compact_needed = threshold > 0 && total_unprocessed >= threshold as u64;

        Ok(AppendEventsResult {
            session_id: session_id.to_string(),
            events_appended: non_system.len(),
            total_unprocessed,
            compact_needed,
        })
    }

    /// # Core API — Compact a session
    ///
    /// Process unprocessed events in a session: extract memories via LLM,
    /// create an Episode summary, and mark events as processed.
    ///
    /// Usually triggered automatically by `append_events()` when the event count
    /// exceeds `compact_threshold`. Call manually when a conversation ends.
    ///
    /// This is the recommended way to process data in MemMe. For most use cases,
    /// you only need three methods: `append_events()`, `search()`, and `compact()`.
    ///
    /// Uses the internally configured LLM. Call `set_llm()` first.
    pub fn compact(&self, session_id: &str) -> Result<CompactResult> {
        self.compact_inner(session_id, self.require_llm()?, true)
    }

    /// Compact without FTS rebuild (used by re_traces to batch rebuild at end).
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

        // Purify events via LLM (coreference resolution, temporal/spatial grounding)
        let purified = purify_events(&events, &llm);

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

        let (title, summary, significance) = generate_episode_summary(&events, &llm);

        // Insert narrative trace into memories table
        let narrative_content = format!("{}: {}", title, summary);
        let narrative_embedding = self
            .embedder
            .embed(&narrative_content)
            .map_err(MemoryError::Embedding)?;

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

        if rebuild_fts {
            let _ = self.storage.create_fts_index();
            let _ = self.storage.create_fts_index_events();
        }

        // Note: No memories extracted here — use meditate() for that
        Ok(CompactResult {
            session_id: session_id.to_string(),
            episode_id: narrative_id,
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
    pub fn re_traces(&self, user_id: &str) -> Result<Vec<CompactResult>> {
        // 1. Delete all existing traces for this user
        self.delete_all_traces(user_id, None, None, None)?;

        // 2. Get all sessions for this user
        let sessions = self.list_sessions(ListSessionsOptions::new(user_id))?;

        // 3. Re-compact each session (skip per-session FTS rebuild)
        let mut results = Vec::new();
        for session in &sessions {
            self.storage.reset_events_processed(&session.session_id)?;

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
fn generate_episode_summary(
    events: &[Event],
    llm: &Arc<dyn memme_llm::LlmProvider>,
) -> (String, String, f32) {
    let text = events
        .iter()
        .map(|e| e.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        r#"Summarize this conversation into a JSON object with three fields:
- "title": a brief title (max 60 chars)
- "summary": a 2-3 sentence summary of what was discussed
- "significance": a float 0.0-1.0 indicating how important/memorable this conversation is

Conversation:
{text}

Respond ONLY with the JSON object, no other text."#
    );

    let messages = vec![memme_llm::Message {
        role: memme_llm::MessageRole::User,
        content: prompt,
    }];
    let config = memme_llm::StructuredGenConfig {
        base_temperature: Some(0.3),
        max_tokens: Some(500),
        response_format: Some(memme_llm::ResponseFormat::Json),
        ..Default::default()
    };
    let text_for_fallback = text.clone();
    match memme_llm::generate_structured(llm.as_ref(), &messages, &config, |raw| {
        let repaired = memme_llm::try_repair_json(raw);
        let parsed: serde_json::Value =
            serde_json::from_str(&repaired).map_err(|e| format!("JSON parse failed: {e}"))?;
        let title = parsed["title"]
            .as_str()
            .unwrap_or("Conversation")
            .to_string();
        let summary = parsed["summary"]
            .as_str()
            .unwrap_or(&text_for_fallback[..text_for_fallback.len().min(200)])
            .to_string();
        let significance = parsed["significance"].as_f64().unwrap_or(0.5) as f32;
        Ok((title, summary, significance.clamp(0.0, 1.0)))
    }) {
        Ok(result) => result,
        Err(_) => fallback_episode_summary(events),
    }
}

/// Truncation-based fallback when LLM is unavailable.
fn fallback_episode_summary(events: &[Event]) -> (String, String, f32) {
    let title = events
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
        .unwrap_or_else(|| "Conversation".to_string());

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

/// Purify events via LLM: resolve coreferences, ground temporal/spatial references.
///
/// This is the core of the Compact purification process. Each event is processed
/// to resolve ambiguous references and extract structured metadata.
fn purify_events(events: &[Event], llm: &Arc<dyn memme_llm::LlmProvider>) -> Vec<PurifiedEvent> {
    // Build context: all events in the session for coreference resolution
    let context = events
        .iter()
        .enumerate()
        .map(|(i, e)| format!("[{}] {}: {}", i + 1, e.event_type.as_str(), e.content))
        .collect::<Vec<_>>()
        .join("\n");

    // Get conversation timestamp for temporal resolution
    let conversation_time = events.first().map(|e| e.timestamp.as_str()).unwrap_or("");

    let prompt = format!(
        r#"You are an Event Purification Engine. Your job is to process each message in a conversation and output a "purified" version that:

1. **Resolves Coreferences**: Replace pronouns (he/she/it/they/that/there) with the actual names/entities they refer to.
   - "She said she'd come" → "Alice said Alice would come"
   - "I went there yesterday" → "I went to the coffee shop yesterday"

2. **Grounds Temporal References**: Resolve relative time expressions to absolute dates.
   - Use the conversation timestamp ({conversation_time}) as the reference point.
   - "yesterday" → compute the day before {conversation_time}
   - "last week" → approximately 7 days before {conversation_time}
   - "next Friday" → the first Friday after {conversation_time}
   - If a time is mentioned, extract it to the "event_time" field

3. **Grounds Spatial References**: Resolve location references.
   - "there" → "at the office" (if previously mentioned)
   - Extract explicit locations to the "location" field

4. **Preserves Meaning**: Keep the core message intact. Don't add information that isn't implied.

**Input format**: Numbered messages [1], [2], [3]...
**Output format**: JSON array where each element corresponds to the input message by index.

```json
{{
  "purified": [
    {{
      "content": "Purified message text with pronouns resolved",
      "event_time": "2026-03-27" or null,
      "location": "coffee shop" or null
    }},
    ...
  ]
}}
```

**Conversation**:
{context}

**Rules**:
- Output exactly {count} objects in the "purified" array, one per input message.
- If no purification is needed, use the original content.
- Use ISO 8601 date format (YYYY-MM-DD) for event_time.
- If location or time cannot be determined, set to null.
- Respond ONLY with the JSON object, no other text."#,
        conversation_time = conversation_time,
        context = context,
        count = events.len()
    );

    let messages = vec![memme_llm::Message {
        role: memme_llm::MessageRole::User,
        content: prompt,
    }];

    // Scale max_tokens based on event count: each purified event needs ~150 output tokens
    let scaled_tokens = (events.len() * 150 + 500).min(4096);
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
        let purified_arr = parsed
            .get("purified")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "Missing 'purified' array".to_string())?;
        let result: Vec<PurifiedEvent> = purified_arr
            .iter()
            .map(|item| {
                let content = item
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let event_time = item
                    .get("event_time")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty() && s != &"null")
                    .map(|s| s.to_string());
                let location = item
                    .get("location")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty() && s != &"null")
                    .map(|s| s.to_string());
                PurifiedEvent {
                    purified_content: content,
                    event_time,
                    location,
                }
            })
            .collect();
        if result.len() != expected_len {
            return Err(format!(
                "Purification returned {} results for {} events",
                result.len(),
                expected_len,
            ));
        }
        Ok(result)
    }) {
        Ok(result) => result,
        Err(e) => {
            tracing::warn!("LLM purification failed: {e}, using fallback");
            fallback_purify_events(events)
        }
    }
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

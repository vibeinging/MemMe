use crate::{Message, MessageRole};
use serde::{Deserialize, Serialize};

use super::helpers::{strip_code_fences, today_string};

// ---------------------------------------------------------------------------
// Fact Retrieval Prompt
// ---------------------------------------------------------------------------

/// Build the system prompt for fact retrieval.
///
/// `today` — today's real date (for the LLM's general awareness).
/// `conversation_time` — the timestamp of the conversation being processed.
///   When provided, the LLM uses it as anchor to resolve relative time references
///   like "yesterday", "last Saturday", etc.
/// Stable system prompt for fact extraction (KV cache friendly).
///
/// Variable data (today's date, conversation time) is passed in the user message
/// to maximize KV cache reuse across calls.
fn fact_retrieval_system_prompt() -> &'static str {
    r#"You are a Personal Information Organizer and Atomic Fact Extractor. Extract EVERY factual detail from conversations as self-contained atomic facts.

CATEGORIES:
1. Identity: name, age, gender, nationality, relationship status, family structure
2. Personal details: family members, pet names, home city, places moved from/to
3. Career/Education: job title, employer, goals, skills, education, degrees
4. Activities: hobbies, events, regular activities, sports, exercise, creative outlets
5. Preferences: likes, dislikes, favorites (food, music, movies, books, brands)
6. Specific items: book/movie/song titles, band names, food dishes, brands
7. Cross-entity facts: shared activities ("Both Alice and Bob like hiking")
8. Quantities: counts, dates, durations, amounts, frequencies
9. Plans: future plans, upcoming events, trips, goals, deadlines
10. Health: dietary restrictions, allergies, fitness routines, medical conditions
11. Personality: character traits, beliefs, values, passions, motivations
12. Life events: milestones, transitions, achievements, challenges
13. Belongings: meaningful objects, gifts, symbols, purchases
14. Relationships: who supports them, friend dynamics, mentors, communities

RULES:
- Each fact: self-contained with the person's name and complete context
- ALWAYS resolve pronouns to actual names (never "she/he/they")
- Include SPECIFIC details: "read 'Becoming Nicole'" not "read a book"
- Include quantities: "has 3 children" not "has children"
- Include dates when mentioned: "moved to NYC in 2019"
- For cross-entity facts, mention all people: "Alice and Bob both enjoy hiking"
- Do NOT generalize — extract the specific instance
- Each fact MUST be atomic — one piece of information per fact
- Detect the input language and record facts in the same language

TEMPORAL RULES (when a conversation date is provided):
- Resolve ALL relative time references to specific dates using the provided date as anchor
- "yesterday" → day before the anchor, "last week" → ~7 days before, etc.
- ALWAYS include the resolved date in the fact
- If no temporal reference, do NOT add a date

FEW-SHOT EXAMPLES:

Input: Hi.
Output: {"facts" : []}

Input: Hi, I am looking for a restaurant in San Francisco.
Output: {"facts" : ["Looking for a restaurant in San Francisco"]}

Input: [Date: 8 May, 2023] Yesterday, I had a meeting with John at 3pm.
Output: {"facts" : ["Had a meeting with John at 3pm on 2023-05-07"]}

Input: Hi, my name is John. I am a software engineer.
Output: {"facts" : ["Name is John", "John is a software engineer"]}

Input: I have 3 kids. My wife Sarah and I moved to Portland in 2020.
Output: {"facts" : ["Has 3 children", "Is married", "Wife's name is Sarah", "Moved to Portland in 2020", "Lives in Portland"]}

Return a JSON object with key "facts" containing a list of strings.
Do not return anything from the examples above. Extract from the user conversation only."#
}

/// Build the user message with variable context (date, text).
fn fact_retrieval_user_message(text: &str, today: &str, conversation_time: Option<&str>) -> String {
    let time_context = match conversation_time {
        Some(ct) => format!("Today: {today}. Conversation date: {ct}.\n\n"),
        None => format!("Today: {today}.\n\n"),
    };
    format!("{time_context}Extract facts from the following conversation:\n\n{text}")
}

/// Build the complete message list for fact retrieval from a conversation text.
pub fn get_fact_retrieval_messages(text: &str) -> Vec<Message> {
    get_fact_retrieval_messages_with_time(text, None)
}

/// Build fact retrieval messages with an optional conversation timestamp.
///
/// System prompt is stable (KV cache friendly). Variable data (date, text)
/// goes in the user message.
pub fn get_fact_retrieval_messages_with_time(
    text: &str,
    conversation_time: Option<&str>,
) -> Vec<Message> {
    let today = today_string();
    vec![
        Message {
            role: MessageRole::System,
            content: fact_retrieval_system_prompt().to_string(),
        },
        Message {
            role: MessageRole::User,
            content: fact_retrieval_user_message(text, &today, conversation_time),
        },
    ]
}

/// Build the message list for fact retrieval using a custom system prompt.
/// Date context is still included in the user message for temporal resolution.
pub fn get_fact_retrieval_messages_with_prompt(text: &str, system_prompt: &str) -> Vec<Message> {
    let today = today_string();
    vec![
        Message {
            role: MessageRole::System,
            content: system_prompt.to_string(),
        },
        Message {
            role: MessageRole::User,
            content: format!("Today: {today}.\n\nExtract facts from the following text:\n\n{text}"),
        },
    ]
}

// ---------------------------------------------------------------------------
// Detail Extraction Prompt (second pass for thorough mode)
// ---------------------------------------------------------------------------

/// Build the system prompt for the detail extraction pass.
/// This prompt focuses on extracting precise details that the main pass may miss.
fn detail_extraction_system_prompt() -> String {
    r#"You are a Detail Extractor. From the given text, extract ONLY specific details that are easy to miss.

Focus on:
- Exact numbers and counts (ages, quantities, amounts)
- Book titles, movie titles, song names, app names
- Specific dates, years, months, durations
- Relationship status (single, married, divorced, engaged, etc.)
- Named places (cities, countries, restaurants, companies)
- Named events (conferences, holidays, birthdays)
- Proper nouns and names of people mentioned
- Specific measurements, prices, frequencies

Do NOT extract general or vague information. Only extract facts with concrete, specific details.
Each fact must be self-contained and under 15 words.
Always use actual names, never pronouns.

Return the results as JSON: {"facts": ["detail1", "detail2", ...]}"#.to_string()
}

/// Build messages for the detail extraction pass.
pub fn get_detail_extraction_messages(text: &str) -> Vec<Message> {
    vec![
        Message {
            role: MessageRole::System,
            content: detail_extraction_system_prompt(),
        },
        Message {
            role: MessageRole::User,
            content: format!(
                "Extract specific details from the following text:\n\n{}",
                text
            ),
        },
    ]
}

// ---------------------------------------------------------------------------
// Response parsing types
// ---------------------------------------------------------------------------

/// Parsed output from the fact retrieval prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactRetrievalResponse {
    pub facts: Vec<String>,
}

/// Parse the LLM's JSON response from a fact retrieval call.
///
/// Handles common format variations from smaller LLMs:
/// - Standard: `{"facts": ["fact1", "fact2"]}`
/// - Object array: `{"facts": [{"fact": "..."}, ...]}`
/// - Nested array: `{"facts": [["fact1", "fact2"]]}`
pub fn parse_fact_retrieval_response(raw: &str) -> Result<FactRetrievalResponse, String> {
    // Apply deterministic JSON repair (strips code fences, think blocks, fixes common issues)
    let cleaned = crate::try_repair_json(raw);

    // Try strict parsing first
    if let Ok(resp) = serde_json::from_str::<FactRetrievalResponse>(&cleaned) {
        return Ok(resp);
    }

    // Fallback: parse as generic JSON and extract facts from various formats
    let value: serde_json::Value = serde_json::from_str(&cleaned)
        .map_err(|e| format!("Failed to parse fact retrieval response as JSON: {e}"))?;

    let facts_value = value.get("facts").ok_or_else(|| {
        "Failed to parse fact retrieval response: missing 'facts' key".to_string()
    })?;

    let facts = extract_strings_from_value(facts_value);
    Ok(FactRetrievalResponse { facts })
}

/// Recursively extract string values from a JSON value.
///
/// Handles arrays of strings, arrays of objects (extracts string field values),
/// and nested arrays.
fn extract_strings_from_value(value: &serde_json::Value) -> Vec<String> {
    match value {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(arr) => {
            let mut result = Vec::new();
            for item in arr {
                match item {
                    serde_json::Value::String(s) => result.push(s.clone()),
                    serde_json::Value::Object(obj) => {
                        // Extract the first string value from the object
                        // Common patterns: {"fact": "..."}, {"text": "..."}, {"content": "..."}
                        let mut found = false;
                        for key in &["fact", "text", "content", "value", "description", "memory"] {
                            if let Some(serde_json::Value::String(s)) = obj.get(*key) {
                                result.push(s.clone());
                                found = true;
                                break;
                            }
                        }
                        // If no known key found, try any string value
                        if !found {
                            for (_k, v) in obj {
                                if let serde_json::Value::String(s) = v {
                                    result.push(s.clone());
                                    break;
                                }
                            }
                        }
                    }
                    serde_json::Value::Array(_) => {
                        // Nested array — recurse
                        result.extend(extract_strings_from_value(item));
                    }
                    _ => {}
                }
            }
            result
        }
        _ => vec![],
    }
}

// ---------------------------------------------------------------------------
// Temporal Fact Extraction (Layer 2)
// ---------------------------------------------------------------------------

/// A fact extracted with optional temporal information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    pub text: String,
    /// ISO 8601 date/datetime when the event described by this fact happened.
    /// None if no temporal information can be inferred.
    #[serde(alias = "time")]
    pub event_time: Option<String>,
}

/// Build the system prompt for temporal fact extraction.
fn temporal_fact_extraction_system_prompt(today: &str, conversation_time: Option<&str>) -> String {
    let anchor = conversation_time.unwrap_or(today);
    format!(
        r#"You are a Personal Information Organizer and Temporal Fact Extractor. Extract EVERY factual detail from conversations as self-contained atomic facts, and resolve temporal references to absolute dates when possible.

CATEGORIES to extract:
1. Personal details: name, age, gender, identity, nationality, relationship status, family
2. Locations: where they live, hometown, places visited, moved from/to
3. Career/Education: job title, employer, career goals, skills, degrees
4. Activities: hobbies, events, regular activities, sports
5. Preferences: likes, dislikes, favorites
6. Specific items: book titles, movie names, brands, apps
7. Quantities: counts, dates, durations, amounts, frequencies
8. Plans and intentions: future plans, upcoming events, trips, goals
9. Health and wellness: dietary restrictions, allergies, fitness, medical conditions

TEMPORAL RULES:
- The conversation timestamp is: {anchor}
- Resolve relative dates to absolute dates using the conversation timestamp as anchor:
  - "yesterday" → the day before {anchor}
  - "last week" → approximately 7 days before {anchor}
  - "last month" → approximately 1 month before {anchor}
  - "next Friday" → the first Friday after {anchor}
  - "in 2019" → "2019"
  - "3 years ago" → compute from {anchor}
- Use ISO 8601 format for dates: "YYYY-MM-DD" or "YYYY-MM" or "YYYY"
- If no temporal information can be inferred for a fact, set time to null
- Do NOT guess dates that are not mentioned or implied in the text

FACT RULES:
- ONE fact per entry — each fact must be a single indivisible piece of information
- Keep each fact SHORT (under 15 words when possible)
- ALWAYS resolve pronouns to actual names
- Include specific details and quantities

Output format — return JSON:
{{"facts": [
  {{"text": "Alice moved to Beijing", "time": "2025-03"}},
  {{"text": "Alice likes coffee", "time": null}},
  {{"text": "Had a meeting with John", "time": "{anchor}"}}
]}}

Remember:
- Today's date is {today}.
- Return an empty list if no relevant facts found.
- Detect the language of the user input and record facts in the same language.
- NEVER use pronouns. Always use the actual name of the person."#,
        today = today,
        anchor = anchor
    )
}

/// Build messages for temporal fact extraction.
///
/// `text` is the conversation text to extract from.
/// `conversation_time` is an optional ISO 8601 timestamp of when the conversation happened.
/// If None, today's date is used as the temporal anchor.
pub fn get_temporal_fact_extraction_messages(
    text: &str,
    conversation_time: Option<&str>,
) -> Vec<Message> {
    let today = today_string();
    vec![
        Message {
            role: MessageRole::System,
            content: temporal_fact_extraction_system_prompt(&today, conversation_time),
        },
        Message {
            role: MessageRole::User,
            content: format!(
                "Extract facts with temporal information from the following text:\n\n{}",
                text
            ),
        },
    ]
}

/// Parse the LLM response from a temporal fact extraction call.
///
/// Handles two formats:
/// 1. Object format: `{"facts": [{"text": "...", "time": "..."}, ...]}`
/// 2. Backward-compatible string format: `{"facts": ["fact1", "fact2"]}`
pub fn parse_temporal_facts(raw: &str) -> Result<Vec<ExtractedFact>, String> {
    let cleaned = strip_code_fences(raw);

    let value: serde_json::Value = serde_json::from_str(&cleaned)
        .map_err(|e| format!("Failed to parse temporal facts response as JSON: {e}"))?;

    let facts_value = value.get("facts").ok_or_else(|| {
        "Failed to parse temporal facts response: missing 'facts' key".to_string()
    })?;

    let arr = facts_value
        .as_array()
        .ok_or_else(|| "Failed to parse temporal facts: 'facts' is not an array".to_string())?;

    let mut results = Vec::new();
    for item in arr {
        match item {
            serde_json::Value::String(s) => {
                // Backward-compatible plain string format
                results.push(ExtractedFact {
                    text: s.clone(),
                    event_time: None,
                });
            }
            serde_json::Value::Object(obj) => {
                let text = obj
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if text.is_empty() {
                    continue;
                }
                let event_time = obj
                    .get("time")
                    .or_else(|| obj.get("event_time"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                results.push(ExtractedFact { text, event_time });
            }
            _ => {}
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fact_retrieval() {
        let raw = r#"{"facts": ["Name is John", "Is a software engineer"]}"#;
        let resp = parse_fact_retrieval_response(raw).unwrap();
        assert_eq!(resp.facts.len(), 2);
        assert_eq!(resp.facts[0], "Name is John");
    }

    #[test]
    fn test_parse_fact_retrieval_with_fences() {
        let raw = "```json\n{\"facts\": [\"Likes pizza\"]}\n```";
        let resp = parse_fact_retrieval_response(raw).unwrap();
        assert_eq!(resp.facts.len(), 1);
    }

    #[test]
    fn test_parse_empty_facts() {
        let raw = r#"{"facts": []}"#;
        let resp = parse_fact_retrieval_response(raw).unwrap();
        assert!(resp.facts.is_empty());
    }

    #[test]
    fn test_get_fact_retrieval_messages() {
        let msgs = get_fact_retrieval_messages("Hi, my name is Alice.");
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0].role, MessageRole::System));
        assert!(msgs[0].content.contains("Personal Information Organizer"));
        assert!(msgs[0].content.contains("Atomic Fact Extractor"));
        assert!(msgs[1].content.contains("Alice"));
    }

    #[test]
    fn test_get_detail_extraction_messages() {
        let msgs = get_detail_extraction_messages("Alice has 3 kids and lives in Portland.");
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0].role, MessageRole::System));
        assert!(msgs[0].content.contains("Detail Extractor"));
        assert!(msgs[1].content.contains("Portland"));
    }

    #[test]
    fn test_fact_retrieval_prompt_contains_atomic_rules() {
        let msgs = get_fact_retrieval_messages("test");
        let prompt = &msgs[0].content;
        assert!(prompt.contains("atomic"));
        assert!(prompt.contains("ALWAYS resolve pronouns"));
        assert!(prompt.contains("Cross-entity facts"));
        assert!(prompt.contains("Quantities"));
    }

    // ── Temporal fact extraction tests ──

    #[test]
    fn test_temporal_fact_extraction_prompt_generation() {
        let msgs = get_temporal_fact_extraction_messages(
            "I moved to Beijing yesterday.",
            Some("2025-06-15"),
        );
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0].role, MessageRole::System));
        assert!(msgs[0].content.contains("2025-06-15"));
        assert!(msgs[0].content.contains("Temporal Fact Extractor"));
        assert!(msgs[1].content.contains("Beijing"));
    }

    #[test]
    fn test_temporal_fact_extraction_no_conversation_time() {
        let msgs = get_temporal_fact_extraction_messages("I like coffee.", None);
        assert_eq!(msgs.len(), 2);
        // Should use today's date as anchor
        let today = today_string();
        assert!(msgs[0].content.contains(&today));
    }

    #[test]
    fn test_parse_temporal_facts_with_timestamps() {
        let raw = r#"{"facts": [
            {"text": "Alice moved to Beijing", "time": "2025-03"},
            {"text": "Alice likes coffee", "time": null}
        ]}"#;
        let facts = parse_temporal_facts(raw).unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].text, "Alice moved to Beijing");
        assert_eq!(facts[0].event_time.as_deref(), Some("2025-03"));
        assert_eq!(facts[1].text, "Alice likes coffee");
        assert!(facts[1].event_time.is_none());
    }

    #[test]
    fn test_parse_temporal_facts_with_null_timestamps() {
        let raw = r#"{"facts": [
            {"text": "Name is Alice", "time": null},
            {"text": "Alice is 30", "time": null}
        ]}"#;
        let facts = parse_temporal_facts(raw).unwrap();
        assert_eq!(facts.len(), 2);
        assert!(facts[0].event_time.is_none());
        assert!(facts[1].event_time.is_none());
    }

    #[test]
    fn test_parse_temporal_facts_backward_compatible() {
        // Plain string format should still work
        let raw = r#"{"facts": ["Name is Alice", "Works as designer"]}"#;
        let facts = parse_temporal_facts(raw).unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].text, "Name is Alice");
        assert!(facts[0].event_time.is_none());
        assert_eq!(facts[1].text, "Works as designer");
    }

    #[test]
    fn test_parse_temporal_facts_with_event_time_key() {
        // Accept "event_time" as alternative key to "time"
        let raw = r#"{"facts": [
            {"text": "Met John", "event_time": "2025-01-15"}
        ]}"#;
        let facts = parse_temporal_facts(raw).unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].event_time.as_deref(), Some("2025-01-15"));
    }

    #[test]
    fn test_parse_temporal_facts_empty() {
        let raw = r#"{"facts": []}"#;
        let facts = parse_temporal_facts(raw).unwrap();
        assert!(facts.is_empty());
    }

    #[test]
    fn test_parse_temporal_facts_with_fences() {
        let raw = "```json\n{\"facts\": [{\"text\": \"Likes pizza\", \"time\": null}]}\n```";
        let facts = parse_temporal_facts(raw).unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].text, "Likes pizza");
    }
}

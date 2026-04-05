use crate::{Message, MessageRole};

use super::helpers::today_string;

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
6. Specific items: book/movie/song titles, band names, food dishes, brands, platforms
7. Cross-entity facts: shared activities ("Both Alice and Bob like hiking")
8. Quantities: counts, dates, durations, amounts, frequencies
9. Plans: future plans, upcoming events, trips, goals, deadlines
10. Health: dietary restrictions, allergies, fitness routines, medical conditions
11. Personality: character traits, beliefs, values, passions, motivations
12. Life events: milestones, transitions, achievements, challenges
13. Belongings: meaningful objects, gifts, symbols, purchases
14. Relationships: who supports them, friend dynamics, mentors, communities, nicknames

RULES:
- Each fact: self-contained with the person's name and complete context
- ALWAYS resolve pronouns to actual names (never "she/he/they")
- Include SPECIFIC details: "read 'Becoming Nicole'" not "read a book"
- Include quantities: "has 3 children" not "has children"
- Preserve exact names: colors, brands, platforms, titles — never generalize
- Extract nicknames and address forms: if someone calls another by a short name, pet name, or nickname, record it (e.g., "Nate calls Joanna 'Jo'")
- For cross-entity facts, mention all people: "Alice and Bob both enjoy hiking"
- Do NOT generalize — extract the specific instance
- Each fact MUST be atomic — one piece of information per fact
- Detect the input language and record facts in the same language
- INFER well-known associations: when a place, product, or institution is mentioned, also extract its commonly-known parent category as a separate fact. Examples: city → state/country ("Tampa is in Florida"), game → platform ("Xenoblade Chronicles is a Nintendo Switch game"), university → location. Only infer what is encyclopedic certainty — never speculate.
- DEDUPLICATE: if the same fact appears in slightly different wording within the text, extract only the most specific version once

OUTPUT FORMAT:
Return a JSON object: {"facts": [{"text": "...", "happened_at": "YYYY-MM-DD"}, ...]}
- "text": the atomic fact as a self-contained sentence
- "happened_at": ISO 8601 date when this fact/event occurred, or null if no temporal info

TEMPORAL RULES (when a conversation date is provided):
- Use the conversation date as anchor to resolve relative time references
- "yesterday" → day before anchor; "last week" → ~7 days before; "上周" → ~7 days before
- Set happened_at to the resolved date; set null if no temporal reference exists
- Do NOT guess dates that are not mentioned or implied

FEW-SHOT EXAMPLES:

Input: Hi.
Output: {"facts": []}

Input: Hi, I am looking for a restaurant in San Francisco.
Output: {"facts": [{"text": "Looking for a restaurant in San Francisco", "happened_at": null}]}

Input: (Conversation date: 2023-05-08) Yesterday, I had a meeting with John at 3pm.
Output: {"facts": [{"text": "Had a meeting with John at 3pm on 2023-05-07", "happened_at": "2023-05-07"}]}

Input: (Conversation date: 2022-04-15) Alice: Hey Jo, guess what? I dyed my hair last week! Bob: What color? Alice: Purple! Bright and bold.
Output: {"facts": [{"text": "Alice dyed her hair purple", "happened_at": "2022-04-08"}, {"text": "Alice uses the nickname 'Jo' for Bob", "happened_at": null}, {"text": "Alice chose purple because it is bright and bold", "happened_at": null}]}

Input: I have 3 kids. My wife Sarah and I moved to Portland in 2020.
Output: {"facts": [{"text": "Has 3 children", "happened_at": null}, {"text": "Is married", "happened_at": null}, {"text": "Wife's name is Sarah", "happened_at": null}, {"text": "Moved to Portland in 2020", "happened_at": "2020"}, {"text": "Lives in Portland", "happened_at": null}]}

Input: (Conversation date: 2022-11-07) 7 people came to my gaming party last weekend. We played Catan on my Nintendo Switch.
Output: {"facts": [{"text": "7 people attended the gaming party", "happened_at": "2022-11-05"}, {"text": "Played Catan at the gaming party", "happened_at": "2022-11-05"}, {"text": "Owns a Nintendo Switch", "happened_at": null}]}

Input: (Conversation date: 2022-11-10) Nate: I took my turtles to the beach in Tampa yesterday! Jo: That's awesome! I'm filming my own movie from the road-trip script here in Fort Wayne.
Output: {"facts": [{"text": "Nate took his turtles to the beach in Tampa", "happened_at": "2022-11-09"}, {"text": "Tampa is a city in Florida", "happened_at": null}, {"text": "Nate calls Joanna 'Jo'", "happened_at": null}, {"text": "Joanna is filming her own movie from a road-trip script", "happened_at": "2022-11-10"}, {"text": "Joanna is in Fort Wayne for filming", "happened_at": "2022-11-10"}, {"text": "Fort Wayne is a city in Indiana", "happened_at": null}]}

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
// Response parsing types
// ---------------------------------------------------------------------------

/// A single extracted fact with optional temporal information.
#[derive(Debug, Clone)]
pub struct ExtractedFact {
    pub text: String,
    /// When the event described by this fact happened (ISO 8601).
    /// None if no temporal information can be inferred.
    pub happened_at: Option<String>,
}

/// Parsed output from the fact retrieval prompt.
#[derive(Debug, Clone)]
pub struct FactRetrievalResponse {
    pub facts: Vec<ExtractedFact>,
}

/// Parse the LLM's JSON response from a fact retrieval call.
///
/// Handles multiple output formats:
/// - Structured: `{"facts": [{"text": "...", "happened_at": "2022-04-08"}, ...]}`
/// - Plain strings: `{"facts": ["fact1", "fact2"]}` (backward compat, happened_at = None)
/// - Object array: `{"facts": [{"fact": "..."}, ...]}`
/// - Nested array: `{"facts": [["fact1", "fact2"]]}`
pub fn parse_fact_retrieval_response(raw: &str) -> Result<FactRetrievalResponse, String> {
    let cleaned = crate::try_repair_json(raw);

    let value: serde_json::Value = serde_json::from_str(&cleaned)
        .map_err(|e| format!("Failed to parse fact retrieval response as JSON: {e}"))?;

    let facts_value = value.get("facts").ok_or_else(|| {
        "Failed to parse fact retrieval response: missing 'facts' key".to_string()
    })?;

    let arr = facts_value.as_array().ok_or_else(|| {
        "Failed to parse fact retrieval response: 'facts' is not an array".to_string()
    })?;

    let mut facts = Vec::new();
    for item in arr {
        match item {
            serde_json::Value::String(s) => {
                facts.push(ExtractedFact {
                    text: s.clone(),
                    happened_at: None,
                });
            }
            serde_json::Value::Object(obj) => {
                // Try structured format: {"text": "...", "happened_at": "..."}
                let text = obj
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                if let Some(text) = text {
                    if !text.is_empty() {
                        let happened_at = obj
                            .get("happened_at")
                            .or_else(|| obj.get("time"))
                            .or_else(|| obj.get("event_time"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        facts.push(ExtractedFact { text, happened_at });
                        continue;
                    }
                }

                // Fallback: try common keys {"fact": "..."}, {"content": "..."}, etc.
                for key in &["fact", "content", "value", "description", "memory"] {
                    if let Some(serde_json::Value::String(s)) = obj.get(*key) {
                        facts.push(ExtractedFact {
                            text: s.clone(),
                            happened_at: None,
                        });
                        break;
                    }
                }
            }
            serde_json::Value::Array(_) => {
                // Nested array — extract strings
                for nested in extract_strings_from_value(item) {
                    facts.push(ExtractedFact {
                        text: nested,
                        happened_at: None,
                    });
                }
            }
            _ => {}
        }
    }
    Ok(FactRetrievalResponse { facts })
}

/// Recursively extract string values from a JSON value.
fn extract_strings_from_value(value: &serde_json::Value) -> Vec<String> {
    match value {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(arr) => arr.iter().flat_map(extract_strings_from_value).collect(),
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_plain_strings() {
        let raw = r#"{"facts": ["Name is John", "Is a software engineer"]}"#;
        let resp = parse_fact_retrieval_response(raw).unwrap();
        assert_eq!(resp.facts.len(), 2);
        assert_eq!(resp.facts[0].text, "Name is John");
        assert!(resp.facts[0].happened_at.is_none());
    }

    #[test]
    fn test_parse_structured_with_time() {
        let raw = r#"{"facts": [
            {"text": "Moved to Beijing", "happened_at": "2025-03"},
            {"text": "Likes coffee", "happened_at": null}
        ]}"#;
        let resp = parse_fact_retrieval_response(raw).unwrap();
        assert_eq!(resp.facts.len(), 2);
        assert_eq!(resp.facts[0].text, "Moved to Beijing");
        assert_eq!(resp.facts[0].happened_at.as_deref(), Some("2025-03"));
        assert!(resp.facts[1].happened_at.is_none());
    }

    #[test]
    fn test_parse_structured_with_time_alias() {
        let raw = r#"{"facts": [{"text": "Met John", "time": "2025-01-15"}]}"#;
        let resp = parse_fact_retrieval_response(raw).unwrap();
        assert_eq!(resp.facts[0].happened_at.as_deref(), Some("2025-01-15"));
    }

    #[test]
    fn test_parse_with_fences() {
        let raw = "```json\n{\"facts\": [\"Likes pizza\"]}\n```";
        let resp = parse_fact_retrieval_response(raw).unwrap();
        assert_eq!(resp.facts.len(), 1);
        assert_eq!(resp.facts[0].text, "Likes pizza");
    }

    #[test]
    fn test_parse_empty_facts() {
        let raw = r#"{"facts": []}"#;
        let resp = parse_fact_retrieval_response(raw).unwrap();
        assert!(resp.facts.is_empty());
    }

    #[test]
    fn test_parse_object_fallback() {
        let raw = r#"{"facts": [{"fact": "Has a dog"}, {"content": "Lives in NYC"}]}"#;
        let resp = parse_fact_retrieval_response(raw).unwrap();
        assert_eq!(resp.facts.len(), 2);
        assert_eq!(resp.facts[0].text, "Has a dog");
        assert_eq!(resp.facts[1].text, "Lives in NYC");
    }

    #[test]
    fn test_get_fact_retrieval_messages() {
        let msgs = get_fact_retrieval_messages("Hi, my name is Alice.");
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0].role, MessageRole::System));
        assert!(msgs[0].content.contains("Personal Information Organizer"));
        assert!(msgs[1].content.contains("Alice"));
    }

    #[test]
    fn test_prompt_contains_structured_output() {
        let msgs = get_fact_retrieval_messages("test");
        let prompt = &msgs[0].content;
        assert!(prompt.contains("happened_at"));
        assert!(prompt.contains("nickname"));
        assert!(prompt.contains("Preserve exact names"));
    }
}

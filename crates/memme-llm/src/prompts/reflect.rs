use crate::{Message, MessageRole};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Reflect Prompt — generate a human-readable reflection from recent memories
// ---------------------------------------------------------------------------

fn reflect_system_prompt() -> &'static str {
    r#"You are a Reflection Analyst. Given a list of recent memories about a person, generate a concise reflection summary.

Your reflection should:
1. Identify key themes and patterns across the memories
2. Note any changes or developments in the person's life/work
3. Highlight important decisions, achievements, or challenges
4. Suggest areas of focus going forward

RULES:
- Write in the same language as the memories (Chinese if memories are in Chinese)
- Be concise: 100-200 words
- Be insightful, not just summarizing — find patterns and connections
- Use a warm, thoughtful tone
- Structure as a brief narrative, not bullet points

Output JSON:
{"reflection": "your reflection text here", "themes": ["theme1", "theme2"], "focus_suggestions": ["suggestion1", "suggestion2"]}"#
}

fn reflect_user_message(
    memories: &[String],
    identity_traits: &[String],
    context: Option<&str>,
) -> String {
    let mut msg = String::new();

    if !identity_traits.is_empty() {
        msg.push_str("KNOWN IDENTITY TRAITS:\n");
        for t in identity_traits {
            msg.push_str("- ");
            msg.push_str(t);
            msg.push('\n');
        }
        msg.push('\n');
    }

    msg.push_str("RECENT MEMORIES:\n");
    for (i, m) in memories.iter().enumerate() {
        msg.push_str(&format!("[{}] {}\n", i, m));
    }

    if let Some(ctx) = context {
        msg.push_str(&format!("\nADDITIONAL CONTEXT:\n{}\n", ctx));
    }

    msg.push_str("\nGenerate a reflection based on the memories above.");
    msg
}

/// Build the complete message list for reflection.
pub fn get_reflect_messages(
    memories: &[String],
    identity_traits: &[String],
    context: Option<&str>,
) -> Vec<Message> {
    vec![
        Message {
            role: MessageRole::System,
            content: reflect_system_prompt().to_string(),
        },
        Message {
            role: MessageRole::User,
            content: reflect_user_message(memories, identity_traits, context),
        },
    ]
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectResponse {
    pub reflection: String,
    #[serde(default)]
    pub themes: Vec<String>,
    #[serde(default)]
    pub focus_suggestions: Vec<String>,
}

pub fn parse_reflect_response(raw: &str) -> Result<ReflectResponse, String> {
    let cleaned = crate::try_repair_json(raw);

    if let Ok(resp) = serde_json::from_str::<ReflectResponse>(&cleaned) {
        return Ok(resp);
    }

    let value: serde_json::Value = serde_json::from_str(&cleaned)
        .map_err(|e| format!("Failed to parse reflect response: {e}"))?;

    let reflection = value
        .get("reflection")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let themes = value
        .get("themes")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let focus_suggestions = value
        .get("focus_suggestions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    if reflection.is_empty() {
        return Err("Empty reflection in response".to_string());
    }

    Ok(ReflectResponse {
        reflection,
        themes,
        focus_suggestions,
    })
}

// ---------------------------------------------------------------------------
// Feedback Learning Prompt
// ---------------------------------------------------------------------------

fn feedback_system_prompt() -> &'static str {
    r#"You are a Feedback Analyst. Given a list of user corrections/feedback, extract behavioral principles that should be remembered.

Each correction contains:
- trigger: what situation caused the wrong behavior
- wrong: what was done incorrectly (optional)
- correct: what should have been done instead

Your job is to synthesize these into clear, actionable behavioral principles.

RULES:
- Each principle should be a clear, concise rule (under 30 words)
- Merge similar corrections into a single principle
- Assign confidence based on how many corrections support it
- Write in the same language as the corrections
- Maximum 10 principles per batch

Output JSON:
{"principles": [
  {"content": "the principle text", "confidence": 0.8, "evidence_indices": [0, 2]},
  {"content": "another principle", "confidence": 0.5, "evidence_indices": [1]}
]}"#
}

fn feedback_user_message(corrections: &[FeedbackItem]) -> String {
    let mut msg = String::from("CORRECTIONS:\n");
    for (i, c) in corrections.iter().enumerate() {
        msg.push_str(&format!("[{}] Trigger: \"{}\"\n", i, c.trigger));
        if let Some(wrong) = &c.wrong_behavior {
            msg.push_str(&format!("    Wrong: {}\n", wrong));
        }
        msg.push_str(&format!("    Correct: {}\n\n", c.correct_behavior));
    }
    msg.push_str("Synthesize behavioral principles from the corrections above.");
    msg
}

/// A single correction/feedback item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackItem {
    pub trigger: String,
    pub wrong_behavior: Option<String>,
    pub correct_behavior: String,
}

/// Build the complete message list for feedback learning.
pub fn get_feedback_messages(corrections: &[FeedbackItem]) -> Vec<Message> {
    vec![
        Message {
            role: MessageRole::System,
            content: feedback_system_prompt().to_string(),
        },
        Message {
            role: MessageRole::User,
            content: feedback_user_message(corrections),
        },
    ]
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedPrinciple {
    pub content: String,
    pub confidence: f32,
    #[serde(default)]
    pub evidence_indices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackResponse {
    pub principles: Vec<LearnedPrinciple>,
}

pub fn parse_feedback_response(raw: &str) -> Result<FeedbackResponse, String> {
    let cleaned = crate::try_repair_json(raw);

    if let Ok(resp) = serde_json::from_str::<FeedbackResponse>(&cleaned) {
        return Ok(resp);
    }

    let value: serde_json::Value = serde_json::from_str(&cleaned)
        .map_err(|e| format!("Failed to parse feedback response: {e}"))?;

    let principles_value = value
        .get("principles")
        .ok_or("Missing 'principles' key")?;

    let arr = principles_value
        .as_array()
        .ok_or("'principles' is not an array")?;

    let mut principles = Vec::new();
    for item in arr {
        if let serde_json::Value::Object(obj) = item {
            let content = obj
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if content.is_empty() {
                continue;
            }
            let confidence = obj
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5) as f32;
            let evidence_indices = obj
                .get("evidence_indices")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();

            principles.push(LearnedPrinciple {
                content,
                confidence,
                evidence_indices,
            });
        }
    }

    Ok(FeedbackResponse { principles })
}

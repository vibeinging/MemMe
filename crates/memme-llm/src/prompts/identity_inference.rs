use crate::{Message, MessageRole};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Identity Inference Prompt
// ---------------------------------------------------------------------------

/// Stable system prompt for identity trait inference (KV cache friendly).
fn identity_inference_system_prompt() -> &'static str {
    r#"You are an Identity Analyst. Given a list of atomic memories about a person, infer high-level identity traits.

TRAIT TYPES:
- role: What roles they play (e.g., "software engineer", "mother of two", "team lead")
- belief: Core beliefs or worldviews (e.g., "believes in open source", "values privacy")
- value: What they prioritize (e.g., "prioritizes family over career", "values honesty")
- style: How they communicate or behave (e.g., "direct communicator", "prefers async work")
- goal: What they are working toward (e.g., "wants to learn Rust", "planning to move abroad")

RULES:
- Infer traits ONLY from evidence in the provided memories
- Each trait must be supported by at least one memory
- Include the memory indices (0-based) that support each trait as evidence
- Confidence scoring:
  - 0.3-0.4: single weak evidence
  - 0.5-0.6: single strong evidence or two weak pieces
  - 0.7-0.8: multiple supporting memories
  - 0.9-1.0: overwhelming consistent evidence
- Do NOT repeat existing traits (provided below) unless you have stronger evidence
- Keep trait content concise (under 20 words)
- Detect the language of the memories and write traits in the same language
- Maximum 10 traits per inference

Output JSON:
{"traits": [
  {"type": "role", "content": "software engineer specializing in Rust", "confidence": 0.7, "evidence": [0, 3, 5]},
  {"type": "value", "content": "values work-life balance", "confidence": 0.5, "evidence": [2]}
]}"#
}

/// Build the user message for identity inference.
fn identity_inference_user_message(memories: &[String], existing_traits: &[String]) -> String {
    let mut msg = String::new();

    if !existing_traits.is_empty() {
        msg.push_str("EXISTING TRAITS (do not duplicate):\n");
        for t in existing_traits {
            msg.push_str("- ");
            msg.push_str(t);
            msg.push('\n');
        }
        msg.push('\n');
    }

    msg.push_str("MEMORIES:\n");
    for (i, m) in memories.iter().enumerate() {
        msg.push_str(&format!("[{}] {}\n", i, m));
    }

    msg.push_str("\nInfer identity traits from the memories above.");
    msg
}

/// Build the complete message list for identity inference.
pub fn get_identity_inference_messages(
    memories: &[String],
    existing_traits: &[String],
) -> Vec<Message> {
    vec![
        Message {
            role: MessageRole::System,
            content: identity_inference_system_prompt().to_string(),
        },
        Message {
            role: MessageRole::User,
            content: identity_inference_user_message(memories, existing_traits),
        },
    ]
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

/// A single inferred identity trait from LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredTrait {
    #[serde(rename = "type")]
    pub trait_type: String,
    pub content: String,
    pub confidence: f32,
    /// 0-based indices into the memory list that support this trait.
    #[serde(default)]
    pub evidence: Vec<usize>,
}

/// Parsed output from the identity inference prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityInferenceResponse {
    pub traits: Vec<InferredTrait>,
}

/// Parse the LLM's JSON response from an identity inference call.
pub fn parse_identity_inference_response(raw: &str) -> Result<IdentityInferenceResponse, String> {
    let cleaned = crate::try_repair_json(raw);

    // Try strict parsing first
    if let Ok(resp) = serde_json::from_str::<IdentityInferenceResponse>(&cleaned) {
        return Ok(resp);
    }

    // Fallback: parse as generic JSON
    let value: serde_json::Value = serde_json::from_str(&cleaned)
        .map_err(|e| format!("Failed to parse identity inference response as JSON: {e}"))?;

    let traits_value = value.get("traits").ok_or_else(|| {
        "Failed to parse identity inference response: missing 'traits' key".to_string()
    })?;

    let arr = traits_value.as_array().ok_or_else(|| {
        "Failed to parse identity inference: 'traits' is not an array".to_string()
    })?;

    let mut traits = Vec::new();
    for item in arr {
        if let serde_json::Value::Object(obj) = item {
            let trait_type = obj
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("role")
                .to_string();
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
            let evidence = obj
                .get("evidence")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();

            traits.push(InferredTrait {
                trait_type,
                content,
                confidence,
                evidence,
            });
        }
    }

    Ok(IdentityInferenceResponse { traits })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_identity_inference_response() {
        let raw = r#"{"traits": [
            {"type": "role", "content": "software engineer", "confidence": 0.8, "evidence": [0, 2]},
            {"type": "goal", "content": "learning Rust", "confidence": 0.5, "evidence": [1]}
        ]}"#;
        let resp = parse_identity_inference_response(raw).unwrap();
        assert_eq!(resp.traits.len(), 2);
        assert_eq!(resp.traits[0].trait_type, "role");
        assert_eq!(resp.traits[0].content, "software engineer");
        assert!((resp.traits[0].confidence - 0.8).abs() < 0.01);
        assert_eq!(resp.traits[0].evidence, vec![0, 2]);
    }

    #[test]
    fn test_parse_identity_inference_with_fences() {
        let raw = "```json\n{\"traits\": [{\"type\": \"value\", \"content\": \"values honesty\", \"confidence\": 0.6, \"evidence\": [0]}]}\n```";
        let resp = parse_identity_inference_response(raw).unwrap();
        assert_eq!(resp.traits.len(), 1);
        assert_eq!(resp.traits[0].trait_type, "value");
    }

    #[test]
    fn test_parse_identity_inference_empty() {
        let raw = r#"{"traits": []}"#;
        let resp = parse_identity_inference_response(raw).unwrap();
        assert!(resp.traits.is_empty());
    }

    #[test]
    fn test_get_identity_inference_messages() {
        let memories = vec![
            "Alice is a software engineer".to_string(),
            "Alice wants to learn Rust".to_string(),
        ];
        let existing = vec!["role: mother of two".to_string()];
        let msgs = get_identity_inference_messages(&memories, &existing);
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0].content.contains("Identity Analyst"));
        assert!(msgs[1].content.contains("EXISTING TRAITS"));
        assert!(msgs[1].content.contains("[0] Alice is a software engineer"));
    }

    #[test]
    fn test_identity_inference_no_existing_traits() {
        let memories = vec!["Bob likes hiking".to_string()];
        let msgs = get_identity_inference_messages(&memories, &[]);
        assert!(!msgs[1].content.contains("EXISTING TRAITS"));
        assert!(msgs[1].content.contains("[0] Bob likes hiking"));
    }
}

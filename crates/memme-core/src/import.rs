//! External chat format import adapters.
//!
//! Converts exports from ChatGPT, Claude, and Gemini into MemMe's internal
//! [`ImportedConversation`] format, ready for ingestion via
//! [`MemoryStore::import_conversations`](crate::MemoryStore::import_conversations).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public output type
// ---------------------------------------------------------------------------

/// A parsed conversation ready for ingestion into MemMe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportedConversation {
    /// Conversation title (from the external platform).
    pub title: Option<String>,
    /// Source platform: "chatgpt", "claude", "gemini".
    pub source: String,
    /// When the conversation started (ISO 8601).
    pub created_at: Option<String>,
    /// Model used (if available).
    pub model: Option<String>,
    /// Messages in chronological order as (role, content) pairs.
    /// Role is normalised to: "user", "assistant", "system".
    pub messages: Vec<(String, String)>,
}

/// Result of importing external conversations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportConversationsResult {
    /// Number of sessions created (one per conversation).
    pub sessions_created: u64,
    /// Total number of events ingested across all sessions.
    pub events_created: u64,
}

// ---------------------------------------------------------------------------
// ChatGPT (conversations.json)
// ---------------------------------------------------------------------------

/// Intermediate serde types for the ChatGPT export format.
mod chatgpt_schema {
    use super::*;

    #[derive(Deserialize)]
    pub struct Conversation {
        pub title: Option<String>,
        pub create_time: Option<f64>,
        pub mapping: HashMap<String, Node>,
    }

    #[derive(Deserialize)]
    pub struct Node {
        pub id: String,
        pub parent: Option<String>,
        pub children: Option<Vec<String>>,
        pub message: Option<Message>,
    }

    #[derive(Deserialize)]
    pub struct Message {
        pub author: Author,
        pub create_time: Option<f64>,
        pub content: Content,
    }

    #[derive(Deserialize)]
    pub struct Author {
        pub role: String,
    }

    #[derive(Deserialize)]
    #[allow(dead_code)]
    pub struct Content {
        pub content_type: Option<String>,
        pub parts: Option<Vec<serde_json::Value>>,
    }
}

/// Parse OpenAI ChatGPT `conversations.json` content.
///
/// The file is a JSON array of conversations. Each conversation stores its
/// messages in a tree structure (`mapping`) linked by `parent` / `children`
/// pointers. This function walks from root nodes to leaves, producing a flat
/// chronological message list per conversation.
pub fn parse_chatgpt(json: &str) -> Result<Vec<ImportedConversation>, String> {
    let conversations: Vec<chatgpt_schema::Conversation> =
        serde_json::from_str(json).map_err(|e| format!("ChatGPT JSON parse error: {e}"))?;

    let mut result = Vec::with_capacity(conversations.len());

    for conv in &conversations {
        let mut messages: Vec<(f64, String, String)> = Vec::new();

        // Find root node(s): those with parent == null
        let roots: Vec<&str> = conv
            .mapping
            .values()
            .filter(|n| n.parent.is_none())
            .map(|n| n.id.as_str())
            .collect();

        // BFS / DFS from roots following children links
        let mut stack: Vec<&str> = roots;
        // We'll collect in traversal order; ChatGPT trees are typically
        // linear chains so BFS == DFS in practice.
        let mut visited = std::collections::HashSet::new();
        let mut ordered: Vec<&str> = Vec::new();

        while let Some(node_id) = stack.pop() {
            if !visited.insert(node_id) {
                continue;
            }
            ordered.push(node_id);
            if let Some(node) = conv.mapping.get(node_id) {
                if let Some(children) = &node.children {
                    // Push children in order (stack reverses, so push reversed)
                    for child in children.iter().rev() {
                        stack.push(child);
                    }
                }
            }
        }

        // Now process nodes in traversal order
        for node_id in &ordered {
            if let Some(node) = conv.mapping.get(*node_id) {
                if let Some(ref msg) = node.message {
                    let role = normalise_role(&msg.author.role);
                    let text = extract_chatgpt_text(&msg.content);
                    if text.is_empty() {
                        continue;
                    }
                    let ts = msg.create_time.unwrap_or(0.0);
                    messages.push((ts, role, text));
                }
            }
        }

        // Sort by timestamp to guarantee chronological order
        messages.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        if messages.is_empty() {
            continue;
        }

        let created_at = conv.create_time.map(unix_float_to_iso8601);

        result.push(ImportedConversation {
            title: conv.title.clone(),
            source: "chatgpt".to_string(),
            created_at,
            model: None,
            messages: messages.into_iter().map(|(_, r, c)| (r, c)).collect(),
        });
    }

    Ok(result)
}

/// Join ChatGPT `content.parts[]` into a single string, skipping non-string parts.
fn extract_chatgpt_text(content: &chatgpt_schema::Content) -> String {
    match &content.parts {
        Some(parts) => parts
            .iter()
            .filter_map(|p| p.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        None => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Claude (conversations.jsonl)
// ---------------------------------------------------------------------------

/// Intermediate serde types for the Claude export format.
mod claude_schema {
    use serde::Deserialize;

    #[derive(Deserialize)]
    #[allow(dead_code)]
    pub struct Conversation {
        pub uuid: Option<String>,
        pub name: Option<String>,
        pub created_at: Option<String>,
        pub model: Option<String>,
        pub chat_messages: Option<Vec<ChatMessage>>,
    }

    #[derive(Deserialize)]
    #[allow(dead_code)]
    pub struct ChatMessage {
        pub sender: String,
        pub text: Option<String>,
        pub created_at: Option<String>,
        /// Some exports store structured content instead of plain text.
        pub content: Option<Vec<ContentBlock>>,
    }

    #[derive(Deserialize)]
    #[allow(dead_code)]
    pub struct ContentBlock {
        #[serde(rename = "type")]
        pub block_type: Option<String>,
        pub text: Option<String>,
    }
}

/// Parse Claude `conversations.jsonl` content (one JSON object per line).
///
/// Each line is a complete conversation with a flat `chat_messages` array.
/// Roles are normalised: `human` -> `user`, `assistant` -> `assistant`.
pub fn parse_claude_export(jsonl: &str) -> Result<Vec<ImportedConversation>, String> {
    let mut result = Vec::new();

    for (line_no, line) in jsonl.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let conv: claude_schema::Conversation = serde_json::from_str(line)
            .map_err(|e| format!("Claude JSONL parse error on line {}: {e}", line_no + 1))?;

        let messages: Vec<(String, String)> = conv
            .chat_messages
            .unwrap_or_default()
            .into_iter()
            .filter_map(|m| {
                let role = normalise_role(&m.sender);
                // Try plain text first, fall back to content blocks
                let text = m
                    .text
                    .filter(|t| !t.is_empty())
                    .or_else(|| extract_claude_content_blocks(&m.content));
                text.map(|t| (role, t))
            })
            .collect();

        if messages.is_empty() {
            continue;
        }

        result.push(ImportedConversation {
            title: conv.name.clone(),
            source: "claude".to_string(),
            created_at: conv.created_at.clone(),
            model: conv.model.clone(),
            messages,
        });
    }

    Ok(result)
}

/// Extract text from Claude's structured content blocks.
fn extract_claude_content_blocks(
    blocks: &Option<Vec<claude_schema::ContentBlock>>,
) -> Option<String> {
    let blocks = blocks.as_ref()?;
    let texts: Vec<&str> = blocks.iter().filter_map(|b| b.text.as_deref()).collect();
    if texts.is_empty() {
        None
    } else {
        Some(texts.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Gemini
// ---------------------------------------------------------------------------

/// Intermediate serde types for the Gemini export format.
mod gemini_schema {
    use serde::Deserialize;

    #[derive(Deserialize)]
    pub struct Export {
        pub conversations: Vec<Conversation>,
    }

    #[derive(Deserialize)]
    #[allow(dead_code)]
    pub struct Conversation {
        pub id: Option<String>,
        pub title: Option<String>,
        pub entries: Option<Vec<Entry>>,
    }

    #[derive(Deserialize)]
    pub struct Entry {
        pub role: String,
        pub text: Option<String>,
        pub create_time: Option<String>,
    }
}

/// Parse Google Gemini export JSON content.
///
/// The file is a JSON object with a top-level `conversations` array.
/// Roles are normalised: `user` -> `user`, `model` -> `assistant`.
pub fn parse_gemini(json: &str) -> Result<Vec<ImportedConversation>, String> {
    let export: gemini_schema::Export =
        serde_json::from_str(json).map_err(|e| format!("Gemini JSON parse error: {e}"))?;

    let mut result = Vec::with_capacity(export.conversations.len());

    for conv in &export.conversations {
        let messages: Vec<(String, String)> = conv
            .entries
            .as_ref()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|entry| {
                let text = entry.text.as_ref()?.clone();
                if text.is_empty() {
                    return None;
                }
                let role = normalise_role(&entry.role);
                Some((role, text))
            })
            .collect();

        if messages.is_empty() {
            continue;
        }

        // Use the first entry's create_time as the conversation's created_at
        let created_at = conv
            .entries
            .as_ref()
            .and_then(|e| e.first())
            .and_then(|e| e.create_time.clone());

        result.push(ImportedConversation {
            title: conv.title.clone(),
            source: "gemini".to_string(),
            created_at,
            model: None,
            messages,
        });
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Normalise external role names to MemMe's standard: "user", "assistant", "system".
fn normalise_role(role: &str) -> String {
    match role.to_lowercase().as_str() {
        "user" | "human" => "user".to_string(),
        "assistant" | "model" | "bot" => "assistant".to_string(),
        "system" => "system".to_string(),
        // Tool messages are kept as-is (MemMe's ChatMessage supports "tool")
        "tool" => "tool".to_string(),
        other => other.to_string(),
    }
}

/// Convert a Unix float timestamp (seconds since epoch) to an ISO 8601 string.
fn unix_float_to_iso8601(ts: f64) -> String {
    use chrono::DateTime;
    let secs = ts as i64;
    let nanos = ((ts - secs as f64) * 1_000_000_000.0) as u32;
    match DateTime::from_timestamp(secs, nanos) {
        Some(dt) => dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        None => format!("{ts}"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_chatgpt_basic() {
        let json = r#"[{
            "id": "conv-1",
            "title": "Test Chat",
            "create_time": 1700000000.0,
            "mapping": {
                "root": {
                    "id": "root",
                    "parent": null,
                    "children": ["msg1"],
                    "message": null
                },
                "msg1": {
                    "id": "msg1",
                    "parent": "root",
                    "children": ["msg2"],
                    "message": {
                        "author": {"role": "user"},
                        "create_time": 1700000001.0,
                        "content": {"content_type": "text", "parts": ["Hello!"]}
                    }
                },
                "msg2": {
                    "id": "msg2",
                    "parent": "msg1",
                    "children": [],
                    "message": {
                        "author": {"role": "assistant"},
                        "create_time": 1700000002.0,
                        "content": {"content_type": "text", "parts": ["Hi there!"]}
                    }
                }
            }
        }]"#;

        let convs = parse_chatgpt(json).unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].title.as_deref(), Some("Test Chat"));
        assert_eq!(convs[0].source, "chatgpt");
        assert_eq!(convs[0].messages.len(), 2);
        assert_eq!(convs[0].messages[0], ("user".to_string(), "Hello!".to_string()));
        assert_eq!(
            convs[0].messages[1],
            ("assistant".to_string(), "Hi there!".to_string())
        );
        assert!(convs[0].created_at.is_some());
    }

    #[test]
    fn test_parse_chatgpt_multipart() {
        let json = r#"[{
            "id": "conv-2",
            "title": null,
            "create_time": null,
            "mapping": {
                "root": {
                    "id": "root",
                    "parent": null,
                    "children": ["msg1"],
                    "message": null
                },
                "msg1": {
                    "id": "msg1",
                    "parent": "root",
                    "children": [],
                    "message": {
                        "author": {"role": "user"},
                        "create_time": 1700000000.0,
                        "content": {"content_type": "text", "parts": ["Part 1", "Part 2"]}
                    }
                }
            }
        }]"#;

        let convs = parse_chatgpt(json).unwrap();
        assert_eq!(convs[0].messages[0].1, "Part 1\nPart 2");
    }

    #[test]
    fn test_parse_claude_basic() {
        let jsonl = r#"{"uuid":"a1b2","name":"Test","created_at":"2025-11-04T09:22:11.000Z","model":"claude-3-5-sonnet","chat_messages":[{"sender":"human","text":"Help me","created_at":"2025-11-04T09:22:11.000Z"},{"sender":"assistant","text":"Sure!","created_at":"2025-11-04T09:22:14.000Z"}]}
{"uuid":"c3d4","name":"Second","created_at":"2025-11-05T10:00:00.000Z","model":"claude-3-opus","chat_messages":[{"sender":"human","text":"Question","created_at":"2025-11-05T10:00:00.000Z"}]}"#;

        let convs = parse_claude_export(jsonl).unwrap();
        assert_eq!(convs.len(), 2);
        assert_eq!(convs[0].title.as_deref(), Some("Test"));
        assert_eq!(convs[0].source, "claude");
        assert_eq!(convs[0].model.as_deref(), Some("claude-3-5-sonnet"));
        assert_eq!(convs[0].messages.len(), 2);
        assert_eq!(convs[0].messages[0].0, "user");
        assert_eq!(convs[0].messages[1].0, "assistant");
    }

    #[test]
    fn test_parse_gemini_basic() {
        let json = r#"{
            "conversations": [{
                "id": "gemini-1",
                "title": "Async Rust",
                "entries": [
                    {"role": "user", "text": "How do I use tokio?", "create_time": "2025-06-15T14:30:05Z"},
                    {"role": "model", "text": "tokio::spawn is...", "create_time": "2025-06-15T14:30:12Z"}
                ]
            }]
        }"#;

        let convs = parse_gemini(json).unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].title.as_deref(), Some("Async Rust"));
        assert_eq!(convs[0].source, "gemini");
        assert_eq!(convs[0].messages.len(), 2);
        assert_eq!(convs[0].messages[0].0, "user");
        assert_eq!(convs[0].messages[1].0, "assistant"); // model -> assistant
    }

    #[test]
    fn test_normalise_role() {
        assert_eq!(normalise_role("human"), "user");
        assert_eq!(normalise_role("Human"), "user");
        assert_eq!(normalise_role("user"), "user");
        assert_eq!(normalise_role("model"), "assistant");
        assert_eq!(normalise_role("assistant"), "assistant");
        assert_eq!(normalise_role("system"), "system");
        assert_eq!(normalise_role("tool"), "tool");
        assert_eq!(normalise_role("unknown_role"), "unknown_role");
    }

    #[test]
    fn test_unix_float_to_iso8601() {
        let iso = unix_float_to_iso8601(1700000000.0);
        assert!(iso.starts_with("2023-11-14"));
    }

    #[test]
    fn test_empty_conversations_skipped() {
        // ChatGPT conversation with no real messages
        let json = r#"[{
            "id": "conv-empty",
            "title": "Empty",
            "create_time": null,
            "mapping": {
                "root": {
                    "id": "root",
                    "parent": null,
                    "children": [],
                    "message": null
                }
            }
        }]"#;
        let convs = parse_chatgpt(json).unwrap();
        assert!(convs.is_empty());

        // Claude with empty messages
        let jsonl = r#"{"uuid":"x","name":"Empty","chat_messages":[]}"#;
        let convs = parse_claude_export(jsonl).unwrap();
        assert!(convs.is_empty());

        // Gemini with no entries
        let json = r#"{"conversations":[{"id":"g1","title":"Empty","entries":[]}]}"#;
        let convs = parse_gemini(json).unwrap();
        assert!(convs.is_empty());
    }

    #[test]
    fn test_parse_error_returns_err() {
        assert!(parse_chatgpt("not json").is_err());
        assert!(parse_claude_export("{invalid json").is_err());
        assert!(parse_gemini("[]").is_err());
    }
}

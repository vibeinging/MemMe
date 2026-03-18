use crate::{Message, MessageRole};
use serde::{Deserialize, Serialize};

use super::helpers::strip_code_fences;

// ---------------------------------------------------------------------------
// Update Memory Prompt
// ---------------------------------------------------------------------------

/// An existing memory entry, used when building the update-memory prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OldMemory {
    /// Unique identifier of this memory (opaque string, e.g. UUID).
    pub id: String,
    /// The textual content of the memory.
    pub text: String,
}

/// The system prompt that instructs the LLM on how to compare new facts
/// against existing memories and produce ADD / UPDATE / DELETE / NONE operations.
const UPDATE_MEMORY_SYSTEM_PROMPT: &str = r#"You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

Based on the above four operations, the memory will change.

Compare newly retrieved facts with the existing memory. For each new fact, decide whether to:
- ADD: Add it to the memory as a new element
- UPDATE: Update an existing memory element
- DELETE: Delete an existing memory element
- NONE: Make no change (if the fact is already present or irrelevant)

There are specific guidelines to select which operation to perform:

1. **Add**: If the retrieved facts contain new information not present in the memory, then you have to add it by generating a new ID in the id field.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "User is a software engineer"
            }
        ]
    - Retrieved facts: ["Name is John"]
    - New Memory:
        {
            "memory" : [
                {
                    "id" : "0",
                    "text" : "User is a software engineer",
                    "event" : "NONE"
                },
                {
                    "id" : "1",
                    "text" : "Name is John",
                    "event" : "ADD"
                }
            ]
        }

2. **Update**: If the retrieved facts contain information that is already present in the memory but the information is totally different, then you have to update it.
If the retrieved fact contains information that conveys the same thing as the elements present in the memory, then you have to keep the fact which has the most information.
Example (a) -- if the memory contains "User likes to play cricket" and the retrieved fact is "Loves to play cricket with friends", then update the memory with the retrieved facts.
Example (b) -- if the memory contains "Likes cheese pizza" and the retrieved fact is "Loves cheese pizza", then you do not need to update it because they convey the same information.
Please keep in mind while updating you have to keep the same ID.
Please note to return the IDs in the output from the input IDs only and do not generate any new ID.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "I really like cheese pizza"
            },
            {
                "id" : "1",
                "text" : "User is a software engineer"
            },
            {
                "id" : "2",
                "text" : "User likes to play cricket"
            }
        ]
    - Retrieved facts: ["Loves chicken pizza", "Loves to play cricket with friends"]
    - New Memory:
        {
            "memory" : [
                {
                    "id" : "0",
                    "text" : "Loves cheese and chicken pizza",
                    "event" : "UPDATE",
                    "old_memory" : "I really like cheese pizza"
                },
                {
                    "id" : "1",
                    "text" : "User is a software engineer",
                    "event" : "NONE"
                },
                {
                    "id" : "2",
                    "text" : "Loves to play cricket with friends",
                    "event" : "UPDATE",
                    "old_memory" : "User likes to play cricket"
                }
            ]
        }

3. **Delete**: If the retrieved facts contain information that contradicts the information present in the memory, then you have to delete it. Or if the direction is to delete the memory, then you have to delete it.
Please note to return the IDs in the output from the input IDs only and do not generate any new ID.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "Name is John"
            },
            {
                "id" : "1",
                "text" : "Loves cheese pizza"
            }
        ]
    - Retrieved facts: ["Dislikes cheese pizza"]
    - New Memory:
        {
            "memory" : [
                {
                    "id" : "0",
                    "text" : "Name is John",
                    "event" : "NONE"
                },
                {
                    "id" : "1",
                    "text" : "Loves cheese pizza",
                    "event" : "DELETE"
                }
            ]
        }

4. **No Change**: If the retrieved facts contain information that is already present in the memory, then you do not need to make any changes.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "Name is John"
            },
            {
                "id" : "1",
                "text" : "Loves cheese pizza"
            }
        ]
    - Retrieved facts: ["Name is John"]
    - New Memory:
        {
            "memory" : [
                {
                    "id" : "0",
                    "text" : "Name is John",
                    "event" : "NONE"
                },
                {
                    "id" : "1",
                    "text" : "Loves cheese pizza",
                    "event" : "NONE"
                }
            ]
        }"#;

/// Build the complete message list for the update-memory operation.
///
/// `new_facts` — facts just extracted from the latest conversation.
/// `old_memories` — the current memory entries retrieved from storage.
pub fn get_update_memory_messages(
    new_facts: &[String],
    old_memories: &[OldMemory],
) -> Vec<Message> {
    // Format old memories as JSON array
    let old_memory_json = if old_memories.is_empty() {
        "Current memory is empty.".to_string()
    } else {
        let entries: Vec<serde_json::Value> = old_memories
            .iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "text": m.text,
                })
            })
            .collect();
        format!(
            "Below is the current content of my memory which I have collected till now. You have to update it in the following format only:\n\n```\n{}\n```",
            serde_json::to_string_pretty(&entries).unwrap_or_default()
        )
    };

    // Format new facts as JSON array
    let new_facts_json =
        serde_json::to_string_pretty(&new_facts).unwrap_or_else(|_| "[]".to_string());

    let user_content = format!(
        r#"{old_memory}

The new retrieved facts are mentioned in the triple backticks. You have to analyze the new retrieved facts and determine whether these facts should be added, updated, or deleted in the memory.

```
{new_facts}
```

You must return your response in the following JSON structure only:

{{
    "memory" : [
        {{
            "id" : "<ID of the memory>",
            "text" : "<Content of the memory>",
            "event" : "<Operation to be performed>",
            "old_memory" : "<Old memory content>"
        }},
        ...
    ]
}}

Follow the instructions mentioned below:
- Do not return anything from the custom few shot prompts provided above.
- If the current memory is empty, then you have to add the new retrieved facts to the memory.
- You should return the updated memory in only JSON format as shown above.
- If there is an addition, generate a new key and add the new memory corresponding to it.
- If there is a deletion, the memory key-value pair should be removed from the memory.
- If there is an update, the ID key should remain the same and only the value needs to be updated.
- The "old_memory" field is required only when event is "UPDATE".
- Do not return anything except the JSON format."#,
        old_memory = old_memory_json,
        new_facts = new_facts_json,
    );

    vec![
        Message {
            role: MessageRole::System,
            content: UPDATE_MEMORY_SYSTEM_PROMPT.to_string(),
        },
        Message {
            role: MessageRole::User,
            content: user_content,
        },
    ]
}

/// Build the update-memory messages using a custom system prompt.
pub fn get_update_memory_messages_with_prompt(
    new_facts: &[String],
    old_memories: &[OldMemory],
    system_prompt: &str,
) -> Vec<Message> {
    // Reuse the same user-content logic
    let msgs = get_update_memory_messages(new_facts, old_memories);
    vec![
        Message {
            role: MessageRole::System,
            content: system_prompt.to_string(),
        },
        msgs.into_iter()
            .nth(1)
            .expect("get_update_memory_messages always returns 2 messages"),
    ]
}

// ---------------------------------------------------------------------------
// Response parsing types
// ---------------------------------------------------------------------------

/// A single memory operation returned by the update-memory prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOperation {
    pub id: String,
    pub text: String,
    pub event: MemoryEvent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_memory: Option<String>,
}

/// The kind of operation the LLM decided on for a memory entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum MemoryEvent {
    Add,
    Update,
    Delete,
    None,
}

/// Parsed output from the update-memory prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateMemoryResponse {
    pub memory: Vec<MemoryOperation>,
}

/// Parse the LLM's JSON response from an update-memory call.
///
/// Two-layer fallback: strict deserialize, then manual extraction from Value.
/// JSON is parsed only once; repair is applied before the single parse.
pub fn parse_update_memory_response(raw: &str) -> Result<UpdateMemoryResponse, String> {
    let cleaned = strip_code_fences(raw);
    let repaired = crate::try_repair_json(&cleaned);

    // Try strict typed parse first
    if let Ok(resp) = serde_json::from_str::<UpdateMemoryResponse>(&repaired) {
        return Ok(resp);
    }

    // Fallback: parse as generic Value once, then extract manually
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&repaired) {
        if let Some(arr) = val.get("memory").and_then(|v| v.as_array()) {
            let ops: Vec<MemoryOperation> = arr
                .iter()
                .filter_map(|item| {
                    let id = item.get("id")?.as_str()?.to_string();
                    let text = item.get("text")?.as_str()?.to_string();
                    let event_str = item.get("event")?.as_str()?;
                    let event = match event_str.to_uppercase().as_str() {
                        "ADD" => MemoryEvent::Add,
                        "UPDATE" => MemoryEvent::Update,
                        "DELETE" => MemoryEvent::Delete,
                        _ => MemoryEvent::None,
                    };
                    let old_memory = item
                        .get("old_memory")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    Some(MemoryOperation {
                        id,
                        text,
                        event,
                        old_memory,
                    })
                })
                .collect();
            if !ops.is_empty() {
                return Ok(UpdateMemoryResponse { memory: ops });
            }
        }
    }

    Err(format!(
        "Failed to parse update memory response: {}",
        &repaired[..repaired.len().min(200)]
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_update_memory() {
        let raw = r#"{
            "memory": [
                {"id": "0", "text": "Name is John", "event": "NONE"},
                {"id": "1", "text": "Loves pizza", "event": "ADD"}
            ]
        }"#;
        let resp = parse_update_memory_response(raw).unwrap();
        assert_eq!(resp.memory.len(), 2);
        assert_eq!(resp.memory[0].event, MemoryEvent::None);
        assert_eq!(resp.memory[1].event, MemoryEvent::Add);
    }

    #[test]
    fn test_parse_update_memory_with_old() {
        let raw = r#"{
            "memory": [
                {
                    "id": "0",
                    "text": "Loves chicken pizza",
                    "event": "UPDATE",
                    "old_memory": "Loves cheese pizza"
                }
            ]
        }"#;
        let resp = parse_update_memory_response(raw).unwrap();
        assert_eq!(
            resp.memory[0].old_memory.as_deref(),
            Some("Loves cheese pizza")
        );
    }

    #[test]
    fn test_get_update_memory_messages_empty() {
        let msgs = get_update_memory_messages(&["Name is John".to_string()], &[]);
        assert_eq!(msgs.len(), 2);
        assert!(msgs[1].content.contains("Current memory is empty"));
    }

    #[test]
    fn test_get_update_memory_messages_with_existing() {
        let old = vec![OldMemory {
            id: "0".to_string(),
            text: "Likes pizza".to_string(),
        }];
        let msgs = get_update_memory_messages(&["Name is John".to_string()], &old);
        assert!(msgs[1].content.contains("Likes pizza"));
        assert!(msgs[1].content.contains("Name is John"));
    }
}

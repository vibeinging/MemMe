use crate::{Message, MessageRole};
use serde::{Deserialize, Serialize};

use super::helpers::strip_code_fences;

// ---------------------------------------------------------------------------
// Entity Extraction Prompt (for Knowledge Graph)
// ---------------------------------------------------------------------------

/// Build the system prompt for entity extraction from text.
fn entity_extraction_system_prompt() -> String {
    r#"You are an Entity Extractor specialized in identifying named entities from text. Your task is to extract all meaningful entities mentioned in the input text.

For each entity, determine its type from the following categories:
- person: A human being (e.g., "Alice", "Dr. Smith")
- organization: A company, institution, or group (e.g., "Google", "MIT")
- location: A place or geographical entity (e.g., "San Francisco", "Japan")
- product: A product, service, or tool (e.g., "iPhone", "Python")
- concept: An abstract idea or topic (e.g., "machine learning", "democracy")
- event: A named event (e.g., "World Cup 2024", "Christmas")
- other: Anything that doesn't fit the above categories

Here are some few-shot examples:

Input: Alice works at Google in San Francisco.
Output: {"entities": [{"name": "Alice", "type": "person"}, {"name": "Google", "type": "organization"}, {"name": "San Francisco", "type": "location"}]}

Input: I love using Python for machine learning projects.
Output: {"entities": [{"name": "Python", "type": "product"}, {"name": "machine learning", "type": "concept"}]}

Input: Hi there, how are you?
Output: {"entities": []}

Rules:
- Return entities as a JSON object with key "entities" containing an array of objects with "name" and "type" fields.
- Entity names should be normalized: use proper capitalization and the most common form of the name.
- Do not extract generic pronouns (I, you, he, she) or common nouns (thing, stuff).
- Each entity should appear only once in the output.
- Detect the language of the input and use entity names in their original language."#.to_string()
}

/// Build the complete message list for entity extraction from text.
pub fn get_entity_extraction_messages(text: &str) -> Vec<Message> {
    vec![
        Message {
            role: MessageRole::System,
            content: entity_extraction_system_prompt(),
        },
        Message {
            role: MessageRole::User,
            content: format!("Extract entities from the following text:\n\n{}", text),
        },
    ]
}

// ---------------------------------------------------------------------------
// Relationship Extraction Prompt (for Knowledge Graph)
// ---------------------------------------------------------------------------

/// Build the system prompt for relationship extraction between entities.
fn relationship_extraction_system_prompt() -> String {
    r#"You are a Relationship Extractor specialized in identifying relationships between entities. Given a text and a list of entities that have been extracted from it, identify all meaningful relationships between those entities.

For each relationship, provide:
- "source": The name of the source entity (must be one of the provided entities)
- "relation": A concise description of the relationship (e.g., "works_at", "located_in", "is_friend_of", "uses", "created_by")
- "target": The name of the target entity (must be one of the provided entities)

Here are some few-shot examples:

Text: Alice works at Google in San Francisco.
Entities: ["Alice", "Google", "San Francisco"]
Output: {"relationships": [{"source": "Alice", "relation": "works_at", "target": "Google"}, {"source": "Google", "relation": "located_in", "target": "San Francisco"}]}

Text: John and Mary are married. They live in Tokyo.
Entities: ["John", "Mary", "Tokyo"]
Output: {"relationships": [{"source": "John", "relation": "married_to", "target": "Mary"}, {"source": "John", "relation": "lives_in", "target": "Tokyo"}, {"source": "Mary", "relation": "lives_in", "target": "Tokyo"}]}

Text: Python is used for machine learning.
Entities: ["Python", "machine learning"]
Output: {"relationships": [{"source": "Python", "relation": "used_for", "target": "machine learning"}]}

Rules:
- Return relationships as a JSON object with key "relationships" containing an array of relationship objects.
- Only use entity names from the provided list — do not introduce new entities.
- Use snake_case for relation types (e.g., "works_at", "lives_in", "is_part_of").
- Each relationship should be directional — choose the most natural direction.
- If no relationships can be identified, return an empty array.
- Keep relation types concise and descriptive."#.to_string()
}

/// Build the complete message list for relationship extraction between entities.
pub fn get_relationship_extraction_messages(text: &str, entities: &[&str]) -> Vec<Message> {
    let entities_json = serde_json::to_string(entities).unwrap_or_else(|_| "[]".to_string());
    vec![
        Message {
            role: MessageRole::System,
            content: relationship_extraction_system_prompt(),
        },
        Message {
            role: MessageRole::User,
            content: format!(
                "Text: {}\nEntities: {}\n\nExtract the relationships between these entities.",
                text, entities_json
            ),
        },
    ]
}

// ---------------------------------------------------------------------------
// Combined Entity + Relationship Extraction (single LLM call)
// ---------------------------------------------------------------------------

/// System prompt that extracts both entities and relationships in one pass.
fn graph_extraction_system_prompt() -> String {
    r#"You are a Knowledge Graph Extractor. Given text, extract all entities and relationships in a single pass.

**Entity types**: person, organization, location, product, concept, event, other

**Output format** (JSON):
```json
{
  "entities": [{"name": "Alice", "type": "person"}, ...],
  "relationships": [{"source": "Alice", "relation": "works_at", "target": "Google"}, ...]
}
```

**Examples**:

Input: Alice works at Google in San Francisco.
Output: {"entities": [{"name": "Alice", "type": "person"}, {"name": "Google", "type": "organization"}, {"name": "San Francisco", "type": "location"}], "relationships": [{"source": "Alice", "relation": "works_at", "target": "Google"}, {"source": "Google", "relation": "located_in", "target": "San Francisco"}]}

Input: Python is used for machine learning.
Output: {"entities": [{"name": "Python", "type": "product"}, {"name": "machine learning", "type": "concept"}], "relationships": [{"source": "Python", "relation": "used_for", "target": "machine learning"}]}

Input: Hi there, how are you?
Output: {"entities": [], "relationships": []}

**Rules**:
- Normalize entity names (proper capitalization, most common form).
- Do NOT extract pronouns (I, you, he, she) or common nouns.
- Use snake_case for relation types (works_at, lives_in, is_part_of).
- Relationships must only reference entities in your "entities" array.
- Choose the most natural direction for each relationship.
- Each entity appears only once. Detect input language and preserve entity names."#.to_string()
}

/// Build messages for combined entity + relationship extraction.
pub fn get_graph_extraction_messages(text: &str) -> Vec<Message> {
    vec![
        Message {
            role: MessageRole::System,
            content: graph_extraction_system_prompt(),
        },
        Message {
            role: MessageRole::User,
            content: format!("Extract entities and relationships from:\n\n{}", text),
        },
    ]
}

/// Combined response from a single graph extraction call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphExtractionResponse {
    #[serde(default)]
    pub entities: Vec<ExtractedEntity>,
    #[serde(default)]
    pub relationships: Vec<ExtractedRelationship>,
}

/// Parse the combined entity + relationship extraction response.
pub fn parse_graph_extraction_response(raw: &str) -> Result<GraphExtractionResponse, String> {
    let cleaned = strip_code_fences(raw);
    let repaired = crate::try_repair_json(&cleaned);
    serde_json::from_str::<GraphExtractionResponse>(&repaired)
        .map_err(|e| format!("Failed to parse graph extraction response: {e}"))
}

// ---------------------------------------------------------------------------
// Entity & Relationship Extraction Response Types (legacy, kept for compat)
// ---------------------------------------------------------------------------

/// Parsed output from the entity extraction prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtractionResponse {
    pub entities: Vec<ExtractedEntity>,
}

/// A single extracted entity from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    #[serde(rename = "type")]
    pub entity_type: String,
}

/// Parsed output from the relationship extraction prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipExtractionResponse {
    pub relationships: Vec<ExtractedRelationship>,
}

/// A single extracted relationship from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelationship {
    pub source: String,
    pub relation: String,
    pub target: String,
}

/// Parse the LLM's JSON response from an entity extraction call.
pub fn parse_entity_extraction_response(raw: &str) -> Result<EntityExtractionResponse, String> {
    let cleaned = strip_code_fences(raw);
    serde_json::from_str::<EntityExtractionResponse>(&cleaned)
        .map_err(|e| format!("Failed to parse entity extraction response: {e}"))
}

/// Parse the LLM's JSON response from a relationship extraction call.
pub fn parse_relationship_extraction_response(
    raw: &str,
) -> Result<RelationshipExtractionResponse, String> {
    let cleaned = strip_code_fences(raw);
    serde_json::from_str::<RelationshipExtractionResponse>(&cleaned)
        .map_err(|e| format!("Failed to parse relationship extraction response: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_entity_extraction() {
        let raw = r#"{"entities": [{"name": "Alice", "type": "person"}, {"name": "Google", "type": "organization"}]}"#;
        let resp = parse_entity_extraction_response(raw).unwrap();
        assert_eq!(resp.entities.len(), 2);
        assert_eq!(resp.entities[0].name, "Alice");
        assert_eq!(resp.entities[0].entity_type, "person");
        assert_eq!(resp.entities[1].name, "Google");
    }

    #[test]
    fn test_parse_entity_extraction_empty() {
        let raw = r#"{"entities": []}"#;
        let resp = parse_entity_extraction_response(raw).unwrap();
        assert!(resp.entities.is_empty());
    }

    #[test]
    fn test_parse_entity_extraction_with_fences() {
        let raw = "```json\n{\"entities\": [{\"name\": \"Alice\", \"type\": \"person\"}]}\n```";
        let resp = parse_entity_extraction_response(raw).unwrap();
        assert_eq!(resp.entities.len(), 1);
    }

    #[test]
    fn test_parse_relationship_extraction() {
        let raw = r#"{"relationships": [{"source": "Alice", "relation": "works_at", "target": "Google"}]}"#;
        let resp = parse_relationship_extraction_response(raw).unwrap();
        assert_eq!(resp.relationships.len(), 1);
        assert_eq!(resp.relationships[0].source, "Alice");
        assert_eq!(resp.relationships[0].relation, "works_at");
        assert_eq!(resp.relationships[0].target, "Google");
    }

    #[test]
    fn test_parse_relationship_extraction_empty() {
        let raw = r#"{"relationships": []}"#;
        let resp = parse_relationship_extraction_response(raw).unwrap();
        assert!(resp.relationships.is_empty());
    }

    #[test]
    fn test_get_entity_extraction_messages() {
        let msgs = get_entity_extraction_messages("Alice works at Google.");
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0].role, MessageRole::System));
        assert!(msgs[0].content.contains("Entity Extractor"));
        assert!(msgs[1].content.contains("Alice"));
    }

    #[test]
    fn test_get_relationship_extraction_messages() {
        let msgs =
            get_relationship_extraction_messages("Alice works at Google.", &["Alice", "Google"]);
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0].content.contains("Relationship Extractor"));
        assert!(msgs[1].content.contains("Alice"));
        assert!(msgs[1].content.contains("Google"));
    }
}

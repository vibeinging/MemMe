use crate::error::MemoryError;
use crate::storage::Storage;
use crate::types::*;

/// Search the knowledge graph using pure SQL (no LLM required).
///
/// Searches entities by name, then returns their neighborhood up to `depth` hops.
pub fn search_graph(
    storage: &Storage,
    query: &str,
    user_id: &str,
    depth: usize,
) -> Result<GraphSearchResult, MemoryError> {
    let depth = depth.min(10);
    let matched = storage.search_entities_by_name(query, user_id, 10)?;

    let mut all_entities: Vec<Entity> = Vec::new();
    let mut all_relations: Vec<GraphRelation> = Vec::new();
    let mut seen_entity_ids = std::collections::HashSet::new();
    let mut seen_relation_ids = std::collections::HashSet::new();

    for (id, name, entity_type) in &matched {
        if seen_entity_ids.insert(id.clone()) {
            all_entities.push(Entity {
                id: id.clone(),
                name: name.clone(),
                entity_type: entity_type.clone(),
                user_id: user_id.to_string(),
            });
        }

        let neighborhood = storage.get_entity_neighborhood(id, user_id, depth)?;
        for rel in neighborhood {
            // Collect entities from relationships using IDs (not name lookup,
            // which could return wrong entity if names collide)
            if seen_entity_ids.insert(rel.source_id.clone()) {
                all_entities.push(Entity {
                    id: rel.source_id.clone(),
                    name: rel.source.clone(),
                    entity_type: None,
                    user_id: user_id.to_string(),
                });
            }
            if seen_entity_ids.insert(rel.target_id.clone()) {
                all_entities.push(Entity {
                    id: rel.target_id.clone(),
                    name: rel.target.clone(),
                    entity_type: None,
                    user_id: user_id.to_string(),
                });
            }

            if seen_relation_ids.insert(rel.id.clone()) {
                all_relations.push(rel);
            }
        }
    }

    Ok(GraphSearchResult {
        entities: all_entities,
        relations: all_relations,
    })
}

// ---------------------------------------------------------------------------
// LLM-powered graph extraction (requires "smart" feature)
// ---------------------------------------------------------------------------

pub use smart_graph::GraphProcessor;

mod smart_graph {
    use std::sync::Arc;

    use memme_llm::prompts::{get_graph_extraction_messages, parse_graph_extraction_response};
    use memme_llm::{generate_structured, LlmProvider, ResponseFormat, StructuredGenConfig};
    use uuid::Uuid;

    use crate::error::MemoryError;
    use crate::storage::Storage;
    use crate::types::*;

    /// Processor that uses an LLM to extract entities and relationships
    /// from text and store them in the knowledge graph.
    pub struct GraphProcessor {
        llm: Arc<dyn LlmProvider>,
    }

    impl GraphProcessor {
        pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
            Self { llm }
        }

        /// Extract entities and relationships from text, store in graph.
        pub fn process(
            &self,
            storage: &Storage,
            text: &str,
            user_id: &str,
        ) -> Result<GraphSearchResult, MemoryError> {
            let config = StructuredGenConfig {
                base_temperature: Some(0.1),
                max_tokens: Some(2000),
                response_format: Some(ResponseFormat::Json),
                ..Default::default()
            };

            // Single LLM call: extract entities + relationships together
            let messages = get_graph_extraction_messages(text);
            let graph_response =
                generate_structured(self.llm.as_ref(), &messages, &config, |raw| {
                    parse_graph_extraction_response(raw)
                })
                .map_err(|e| MemoryError::Llm(e.to_string()))?;

            if graph_response.entities.is_empty() {
                return Ok(GraphSearchResult {
                    entities: vec![],
                    relations: vec![],
                });
            }

            // Upsert entities — reuse existing IDs for matching names
            let mut created_entities: Vec<Entity> = Vec::new();
            let mut name_to_id = std::collections::HashMap::new();

            for extracted in &graph_response.entities {
                let (entity_id, _is_new) =
                    match storage.find_entity_by_name(&extracted.name, user_id)? {
                        Some((existing_id, _, _)) => {
                            // Update the existing entity's type if provided
                            storage.upsert_entity(
                                &existing_id,
                                &extracted.name,
                                Some(&extracted.entity_type),
                                user_id,
                            )?;
                            (existing_id, false)
                        }
                        None => {
                            let new_id = Uuid::new_v4().to_string();
                            storage.upsert_entity(
                                &new_id,
                                &extracted.name,
                                Some(&extracted.entity_type),
                                user_id,
                            )?;
                            (new_id, true)
                        }
                    };

                name_to_id.insert(extracted.name.to_lowercase(), entity_id.clone());
                created_entities.push(Entity {
                    id: entity_id,
                    name: extracted.name.clone(),
                    entity_type: Some(extracted.entity_type.clone()),
                    user_id: user_id.to_string(),
                });
            }

            // 4. Insert relationships
            let mut created_relations: Vec<GraphRelation> = Vec::new();

            for rel in &graph_response.relationships {
                let source_id = name_to_id.get(&rel.source.to_lowercase());
                let target_id = name_to_id.get(&rel.target.to_lowercase());

                if let (Some(src_id), Some(tgt_id)) = (source_id, target_id) {
                    let rel_id = Uuid::new_v4().to_string();
                    storage.insert_relationship(
                        &rel_id,
                        src_id,
                        tgt_id,
                        &rel.relation,
                        user_id,
                        rel.description.as_deref(),
                    )?;

                    created_relations.push(GraphRelation {
                        id: rel_id,
                        source: rel.source.clone(),
                        source_id: src_id.clone(),
                        target: rel.target.clone(),
                        target_id: tgt_id.clone(),
                        relation_type: rel.relation.clone(),
                        user_id: user_id.to_string(),
                        description: rel.description.clone(),
                    });
                }
            }

            Ok(GraphSearchResult {
                entities: created_entities,
                relations: created_relations,
            })
        }

        /// Search the graph for entities matching query, return neighborhood.
        #[allow(dead_code)]
        pub fn search(
            &self,
            storage: &Storage,
            query: &str,
            user_id: &str,
            depth: usize,
        ) -> Result<GraphSearchResult, MemoryError> {
            // Delegate to the non-LLM search function
            super::search_graph(storage, query, user_id, depth)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MemoryConfig;
    use crate::storage::Storage;

    fn test_config() -> MemoryConfig {
        MemoryConfig {
            db_path: ":memory:".into(),
            collection_name: "test".into(),
            embedding_dims: 4,
            dedup_threshold: 0.15,
            default_limit: 10,
            ..Default::default()
        }
    }

    fn open_storage() -> Storage {
        Storage::open(test_config()).unwrap()
    }

    // ── Storage-level tests ──

    #[test]
    fn test_entity_upsert_and_find() {
        let storage = open_storage();
        storage
            .upsert_entity("e1", "Alice", Some("person"), "user1")
            .unwrap();

        let found = storage.find_entity_by_name("Alice", "user1").unwrap();
        assert!(found.is_some());
        let (id, name, etype) = found.unwrap();
        assert_eq!(id, "e1");
        assert_eq!(name, "Alice");
        assert_eq!(etype.as_deref(), Some("person"));

        // Case-insensitive find
        let found_lower = storage.find_entity_by_name("alice", "user1").unwrap();
        assert!(found_lower.is_some());
        assert_eq!(found_lower.unwrap().0, "e1");
    }

    #[test]
    fn test_entity_not_found() {
        let storage = open_storage();
        let found = storage.find_entity_by_name("NonExistent", "user1").unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn test_relationship_insert_and_find() {
        let storage = open_storage();
        storage
            .upsert_entity("e1", "Alice", Some("person"), "user1")
            .unwrap();
        storage
            .upsert_entity("e2", "Google", Some("organization"), "user1")
            .unwrap();
        storage
            .insert_relationship("r1", "e1", "e2", "works_at", "user1", None)
            .unwrap();

        // Find by source entity
        let rels = storage.find_relationships("e1", "user1").unwrap();
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].source, "Alice");
        assert_eq!(rels[0].target, "Google");
        assert_eq!(rels[0].relation_type, "works_at");

        // Find by target entity
        let rels = storage.find_relationships("e2", "user1").unwrap();
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].source, "Alice");
    }

    #[test]
    fn test_delete_entity_cascades() {
        let storage = open_storage();
        storage
            .upsert_entity("e1", "Alice", Some("person"), "user1")
            .unwrap();
        storage
            .upsert_entity("e2", "Google", Some("organization"), "user1")
            .unwrap();
        storage
            .insert_relationship("r1", "e1", "e2", "works_at", "user1", None)
            .unwrap();

        // Delete Alice — should also remove the relationship
        storage.delete_entity("e1").unwrap();

        let found = storage.find_entity_by_name("Alice", "user1").unwrap();
        assert!(found.is_none());

        let rels = storage.find_relationships("e2", "user1").unwrap();
        assert!(rels.is_empty());
    }

    #[test]
    fn test_search_entities_by_name() {
        let storage = open_storage();
        storage
            .upsert_entity("e1", "Alice Smith", Some("person"), "user1")
            .unwrap();
        storage
            .upsert_entity("e2", "Bob", Some("person"), "user1")
            .unwrap();
        storage
            .upsert_entity("e3", "Alice Johnson", Some("person"), "user1")
            .unwrap();

        let results = storage
            .search_entities_by_name("Alice", "user1", 10)
            .unwrap();
        assert_eq!(results.len(), 2);

        // Verify case-insensitive
        let results = storage
            .search_entities_by_name("alice", "user1", 10)
            .unwrap();
        assert_eq!(results.len(), 2);

        // No match
        let results = storage
            .search_entities_by_name("Charlie", "user1", 10)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_entity_neighborhood() {
        let storage = open_storage();
        // Build a chain: Alice -> Google -> San Francisco
        storage
            .upsert_entity("e1", "Alice", Some("person"), "user1")
            .unwrap();
        storage
            .upsert_entity("e2", "Google", Some("organization"), "user1")
            .unwrap();
        storage
            .upsert_entity("e3", "San Francisco", Some("location"), "user1")
            .unwrap();
        storage
            .insert_relationship("r1", "e1", "e2", "works_at", "user1", None)
            .unwrap();
        storage
            .insert_relationship("r2", "e2", "e3", "located_in", "user1", None)
            .unwrap();

        // Depth 1: from Alice, should get only the direct relationship
        let depth1 = storage.get_entity_neighborhood("e1", "user1", 1).unwrap();
        assert_eq!(depth1.len(), 1);
        assert_eq!(depth1[0].relation_type, "works_at");

        // Depth 2: from Alice, should get both relationships
        let depth2 = storage.get_entity_neighborhood("e1", "user1", 2).unwrap();
        assert_eq!(depth2.len(), 2);
    }

    // ── Graph search tests ──

    #[test]
    fn test_graph_search_returns_neighborhood() {
        let storage = open_storage();
        storage
            .upsert_entity("e1", "Alice", Some("person"), "user1")
            .unwrap();
        storage
            .upsert_entity("e2", "Google", Some("organization"), "user1")
            .unwrap();
        storage
            .insert_relationship("r1", "e1", "e2", "works_at", "user1", None)
            .unwrap();

        let result = search_graph(&storage, "Alice", "user1", 1).unwrap();
        assert!(!result.entities.is_empty());
        assert!(!result.relations.is_empty());
        assert_eq!(result.relations[0].relation_type, "works_at");
    }

    #[test]
    fn test_graph_search_empty() {
        let storage = open_storage();
        let result = search_graph(&storage, "NonExistent", "user1", 1).unwrap();
        assert!(result.entities.is_empty());
        assert!(result.relations.is_empty());
    }
}

#[cfg(test)]
mod smart_tests {
    use super::*;
    use crate::config::MemoryConfig;
    use crate::storage::Storage;
    use memme_llm::{GenerateOptions, LlmError, LlmProvider, Message};
    use std::sync::{Arc, Mutex};

    struct MockLlm {
        responses: Mutex<Vec<String>>,
    }

    impl MockLlm {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    impl LlmProvider for MockLlm {
        fn generate(
            &self,
            _messages: &[Message],
            _options: &GenerateOptions,
        ) -> Result<String, LlmError> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                Err(LlmError::NotAvailable("no more mock responses".into()))
            } else {
                Ok(responses.remove(0))
            }
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    fn test_config() -> MemoryConfig {
        MemoryConfig {
            db_path: ":memory:".into(),
            collection_name: "test".into(),
            embedding_dims: 4,
            dedup_threshold: 0.15,
            default_limit: 10,
            ..Default::default()
        }
    }

    fn open_storage() -> Storage {
        Storage::open(test_config()).unwrap()
    }

    #[test]
    fn test_graph_extract_and_store() {
        let storage = open_storage();

        // Single LLM call: combined entity + relationship extraction
        let combined_response =
            r#"{"entities": [{"name": "Alice", "type": "person"}, {"name": "Google", "type": "organization"}], "relationships": [{"source": "Alice", "relation": "works_at", "target": "Google"}]}"#
                .to_string();

        let llm = Arc::new(MockLlm::new(vec![combined_response]));
        let processor = GraphProcessor::new(llm);
        let result = processor
            .process(&storage, "Alice works at Google.", "user1")
            .unwrap();

        assert_eq!(result.entities.len(), 2);
        assert_eq!(result.relations.len(), 1);
        assert_eq!(result.relations[0].relation_type, "works_at");

        // Verify entities are in storage
        let found = storage.find_entity_by_name("Alice", "user1").unwrap();
        assert!(found.is_some());
    }

    #[test]
    fn test_graph_entity_dedup() {
        let storage = open_storage();

        // Pre-populate Alice
        storage
            .upsert_entity("existing-alice", "Alice", Some("person"), "user1")
            .unwrap();

        // LLM returns Alice again (combined response)
        let combined_response =
            r#"{"entities": [{"name": "Alice", "type": "person"}, {"name": "Bob", "type": "person"}], "relationships": [{"source": "Alice", "relation": "knows", "target": "Bob"}]}"#
                .to_string();

        let llm = Arc::new(MockLlm::new(vec![combined_response]));
        let processor = GraphProcessor::new(llm);
        let result = processor
            .process(&storage, "Alice knows Bob.", "user1")
            .unwrap();

        // Alice should reuse existing ID
        let alice_entity = result.entities.iter().find(|e| e.name == "Alice").unwrap();
        assert_eq!(alice_entity.id, "existing-alice");

        // Bob should be new
        let bob_entity = result.entities.iter().find(|e| e.name == "Bob").unwrap();
        assert_ne!(bob_entity.id, "existing-alice");

        // Verify only 2 entities total in storage
        let all_entities = storage.list_entities("user1").unwrap();
        assert_eq!(all_entities.len(), 2);
    }

    #[test]
    fn test_graph_search_via_processor() {
        let storage = open_storage();
        storage
            .upsert_entity("e1", "Alice", Some("person"), "user1")
            .unwrap();
        storage
            .upsert_entity("e2", "Google", Some("organization"), "user1")
            .unwrap();
        storage
            .insert_relationship("r1", "e1", "e2", "works_at", "user1", None)
            .unwrap();

        let llm = Arc::new(MockLlm::new(vec![]));
        let processor = GraphProcessor::new(llm);
        let result = processor.search(&storage, "Alice", "user1", 1).unwrap();

        assert!(!result.entities.is_empty());
        assert!(!result.relations.is_empty());
    }

    #[test]
    fn test_graph_empty_extraction() {
        let storage = open_storage();

        // LLM returns no entities
        let graph_response = r#"{"entities": []}"#.to_string();

        let llm = Arc::new(MockLlm::new(vec![graph_response]));
        let processor = GraphProcessor::new(llm);
        let result = processor
            .process(&storage, "Hello there!", "user1")
            .unwrap();

        assert!(result.entities.is_empty());
        assert!(result.relations.is_empty());
    }
}

//! Knowledge graph example: demonstrates entity/relationship extraction and graph search.
//!
//! This example uses a mock LLM provider to extract entities and relationships
//! from text, then queries the knowledge graph to explore connections.
//! In production, you would use OllamaProvider or OpenAIProvider instead.
//!
//! Run with:
//!   cargo run -p memme-core --example graph

use std::sync::Arc;
use std::sync::Mutex;

use memme_core::config::MemoryConfig;
use memme_core::memory::MemoryStore;
use memme_embeddings::mock::MockEmbedder;
use memme_llm::{GenerateOptions, LlmError, LlmProvider, Message};

// ── Mock LLM provider ───────────────────────────────────────────────
// Returns pre-scripted responses in order. Each call to `generate`
// pops the next response from the queue.
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

fn main() {
    println!("=== MemMe Knowledge Graph Example ===\n");

    // ── 1. Create an in-memory store ────────────────────────────────
    let config = MemoryConfig {
        db_path: ":memory:".into(),
        collection_name: "graph_demo".into(),
        embedding_dims: 384,
        dedup_threshold: 0.15,
        default_limit: 10,
        ..Default::default()
    };
    let embedder = Arc::new(MockEmbedder::new(384));
    let store = MemoryStore::new(config, embedder).expect("Failed to create MemoryStore");
    println!("[1] MemoryStore created (in-memory).\n");

    // ── 2. Add graph entries from first text snippet ────────────────
    // The mock LLM returns entities and relationships extracted from text.
    println!("[2] Extracting entities and relationships from text...\n");
    println!("  Text: \"Alice works at Google as a senior engineer.");
    println!("         She collaborates with Bob on the Search team.\"\n");

    let graph_response_1 = r#"{
        "entities": [
            {"name": "Alice", "type": "person"},
            {"name": "Google", "type": "organization"},
            {"name": "Bob", "type": "person"},
            {"name": "Search team", "type": "team"}
        ],
        "relationships": [
            {"source": "Alice", "relation": "works_at", "target": "Google"},
            {"source": "Alice", "relation": "collaborates_with", "target": "Bob"},
            {"source": "Alice", "relation": "member_of", "target": "Search team"},
            {"source": "Bob", "relation": "member_of", "target": "Search team"}
        ]
    }"#
    .to_string();

    let llm1 = Arc::new(MockLlm::new(vec![graph_response_1]));
    let result1 = store
        .add_graph(
            "Alice works at Google as a senior engineer. She collaborates with Bob on the Search team.",
            "user1",
            llm1,
        )
        .expect("add_graph failed");

    println!("  Extracted {} entities:", result1.entities.len());
    for e in &result1.entities {
        println!(
            "    - {} (type: {})",
            e.name,
            e.entity_type.as_deref().unwrap_or("unknown")
        );
    }
    println!("\n  Extracted {} relationships:", result1.relations.len());
    for r in &result1.relations {
        println!("    - {} --[{}]--> {}", r.source, r.relation_type, r.target);
    }
    println!();

    // ── 3. Add more graph entries from a second snippet ─────────────
    println!("[3] Adding more knowledge from a second text...\n");
    println!("  Text: \"Google is headquartered in Mountain View, California.");
    println!("         Alice recently moved to San Francisco.\"\n");

    let graph_response_2 = r#"{
        "entities": [
            {"name": "Google", "type": "organization"},
            {"name": "Mountain View", "type": "location"},
            {"name": "California", "type": "location"},
            {"name": "Alice", "type": "person"},
            {"name": "San Francisco", "type": "location"}
        ],
        "relationships": [
            {"source": "Google", "relation": "headquartered_in", "target": "Mountain View"},
            {"source": "Mountain View", "relation": "located_in", "target": "California"},
            {"source": "Alice", "relation": "lives_in", "target": "San Francisco"},
            {"source": "San Francisco", "relation": "located_in", "target": "California"}
        ]
    }"#
    .to_string();

    let llm2 = Arc::new(MockLlm::new(vec![graph_response_2]));
    let result2 = store
        .add_graph(
            "Google is headquartered in Mountain View, California. Alice recently moved to San Francisco.",
            "user1",
            llm2,
        )
        .expect("add_graph failed");

    println!(
        "  Added {} entities, {} relationships.\n",
        result2.entities.len(),
        result2.relations.len()
    );

    // ── 4. Search the graph for "Alice" ─────────────────────────────
    // Graph search is pure SQL — no LLM required.
    println!("[4] Searching knowledge graph for \"Alice\" (depth=1)...\n");

    let search1 = store
        .search_graph("Alice", "user1", 1)
        .expect("search_graph failed");

    println!("  Found {} entities:", search1.entities.len());
    for e in &search1.entities {
        println!(
            "    - {} (type: {})",
            e.name,
            e.entity_type.as_deref().unwrap_or("unknown")
        );
    }
    println!("\n  Found {} relationships:", search1.relations.len());
    for r in &search1.relations {
        println!("    - {} --[{}]--> {}", r.source, r.relation_type, r.target);
    }
    println!();

    // ── 5. Deeper search: 2 hops from "Alice" ──────────────────────
    println!("[5] Searching for \"Alice\" with depth=2 (2 hops)...\n");

    let search2 = store
        .search_graph("Alice", "user1", 2)
        .expect("search_graph failed");

    println!("  Found {} entities:", search2.entities.len());
    for e in &search2.entities {
        println!(
            "    - {} (type: {})",
            e.name,
            e.entity_type.as_deref().unwrap_or("unknown")
        );
    }
    println!("\n  Found {} relationships:", search2.relations.len());
    for r in &search2.relations {
        println!("    - {} --[{}]--> {}", r.source, r.relation_type, r.target);
    }
    println!();

    // ── 6. Search for a different entity ────────────────────────────
    println!("[6] Searching for \"Google\" (depth=1)...\n");

    let search3 = store
        .search_graph("Google", "user1", 1)
        .expect("search_graph failed");

    println!("  Found {} entities:", search3.entities.len());
    for e in &search3.entities {
        println!(
            "    - {} (type: {})",
            e.name,
            e.entity_type.as_deref().unwrap_or("unknown")
        );
    }
    println!("\n  Found {} relationships:", search3.relations.len());
    for r in &search3.relations {
        println!("    - {} --[{}]--> {}", r.source, r.relation_type, r.target);
    }

    println!("\n=== Done! ===");
}

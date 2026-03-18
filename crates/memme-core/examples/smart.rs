//! Smart mode example: demonstrates LLM-powered memory extraction and updates.
//!
//! This example uses a mock LLM provider that returns predefined responses.
//! In production, you would use OllamaProvider or OpenAIProvider instead.
//!
//! Run with:
//!   cargo run -p memme-core --example smart

fn main() {
    use std::sync::Arc;
    use std::sync::Mutex;

    use memme_core::config::MemoryConfig;
    use memme_core::memory::MemoryStore;
    use memme_core::types::*;
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

    println!("=== MemMe Smart Mode Example ===\n");

    // ── 1. Create store ─────────────────────────────────────────────────
    let config = MemoryConfig {
        db_path: ":memory:".into(),
        collection_name: "smart_demo".into(),
        embedding_dims: 384,
        dedup_threshold: 0.15,
        default_limit: 10,
        ..Default::default()
    };
    let embedder = Arc::new(MockEmbedder::new(384));
    let store = MemoryStore::new(config, embedder).expect("Failed to create MemoryStore");
    println!("[1] MemoryStore created.\n");

    // ── 2. First conversation: extract and add memories ─────────────────
    println!("[2] Processing conversation snippet...\n");
    println!("  User says: \"Hi! I'm Bob, I work at Google as a software engineer.");
    println!("              I love drinking coffee and running every morning.\"\n");

    // The mock LLM will return these two responses in sequence:
    //   Call 1 (fact extraction): returns extracted facts
    //   Call 2 (update decision): returns ADD operations for each fact
    let fact_response_1 =
        r#"{"facts": ["User's name is Bob", "User works at Google as a software engineer", "User loves drinking coffee", "User runs every morning"]}"#
            .to_string();

    let update_response_1 = r#"{
        "memory": [
            {"id": "new1", "text": "User's name is Bob", "event": "ADD"},
            {"id": "new2", "text": "User works at Google as a software engineer", "event": "ADD"},
            {"id": "new3", "text": "User loves drinking coffee", "event": "ADD"},
            {"id": "new4", "text": "User runs every morning", "event": "ADD"}
        ]
    }"#
    .to_string();

    let llm1 = Arc::new(MockLlm::new(vec![fact_response_1, update_response_1]));
    let result = store
        .add_smart(
            "Hi! I'm Bob, I work at Google as a software engineer. I love drinking coffee and running every morning.",
            "user_bob",
            None,
            None,
            None,
            llm1,
            false,
        )
        .expect("add_smart failed");

    println!("  Extracted and stored {} memories:", result.memories.len());
    for r in &result.memories {
        println!("    - \"{}\" (id: {})", r.content, &r.id[..8]);
    }
    println!();

    // ── 3. Show stored memories ─────────────────────────────────────────
    println!("[3] Current memories in store:\n");
    let all = store
        .list_traces(ListOptions::new("user_bob"))
        .expect("List failed");
    for (i, m) in all.iter().enumerate() {
        println!("  #{}: \"{}\"", i + 1, m.content);
    }
    println!();

    // ── 4. Second conversation with contradicting info ──────────────────
    println!("[4] Processing new conversation with contradicting info...\n");
    println!("  User says: \"Actually I just moved to Apple. And I switched to tea.\"\n");

    // Grab the IDs of the memories we want the LLM to update
    let work_memory_id = all
        .iter()
        .find(|m| m.content.contains("Google"))
        .map(|m| m.id.clone())
        .expect("should find Google memory");
    let coffee_memory_id = all
        .iter()
        .find(|m| m.content.contains("coffee"))
        .map(|m| m.id.clone())
        .expect("should find coffee memory");

    let fact_response_2 =
        r#"{"facts": ["User now works at Apple", "User switched from coffee to tea"]}"#.to_string();

    let update_response_2 = format!(
        r#"{{
        "memory": [
            {{"id": "{work_memory_id}", "text": "User works at Apple as a software engineer", "event": "UPDATE", "old_memory": "User works at Google as a software engineer"}},
            {{"id": "{coffee_memory_id}", "text": "User prefers drinking tea", "event": "UPDATE", "old_memory": "User loves drinking coffee"}}
        ]
    }}"#
    );

    let llm2 = Arc::new(MockLlm::new(vec![fact_response_2, update_response_2]));
    let updated_result = store
        .add_smart(
            "Actually I just moved to Apple. And I switched to tea.",
            "user_bob",
            None,
            None,
            None,
            llm2,
            false,
        )
        .expect("add_smart failed");

    println!("  Updated {} memories:", updated_result.memories.len());
    for r in &updated_result.memories {
        println!("    - \"{}\"", r.content);
    }
    println!();

    // ── 5. Show final state ─────────────────────────────────────────────
    println!("[5] Final memories in store:\n");
    let final_list = store
        .list_traces(ListOptions::new("user_bob"))
        .expect("List failed");
    for (i, m) in final_list.iter().enumerate() {
        println!("  #{}: \"{}\"", i + 1, m.content);
    }

    println!("\n=== Done! ===");
}

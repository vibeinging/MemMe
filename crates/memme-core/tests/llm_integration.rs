//! Integration tests using OpenAI-format API for LLM and embedding.
//!
//! These tests use the OpenAI-compatible API format, which works with:
//! - OpenAI (api.openai.com)
//! - Ollama (localhost:11434/v1)
//! - vLLM, LM Studio, DeepSeek, etc.
//!
//! Default: Ollama's OpenAI-compatible endpoint (no API key needed).
//!
//! Configure via env vars:
//!   MEMME_LLM_BASE_URL    (default: http://localhost:11434  — SDK appends /v1/chat/completions)
//!   MEMME_LLM_API_KEY     (default: "ollama")
//!   MEMME_LLM_MODEL       (default: "llama3.2")
//!   MEMME_EMBED_BASE_URL  (default: http://localhost:11434/v1  — SDK appends /embeddings)
//!   MEMME_EMBED_API_KEY   (default: "ollama")
//!   MEMME_EMBED_MODEL     (default: "mxbai-embed-large")
//!   MEMME_EMBED_DIMS      (default: 1024)
//!
//! Run with: cargo test -p memme-core --features "smart openai" --test llm_integration -- --ignored --nocapture

use std::sync::Arc;
use std::time::Instant;

use memme_core::config::MemoryConfig;
use memme_core::memory::MemoryStore;
use memme_core::types::*;
use memme_embeddings::openai::{OpenAiEmbedder, OpenAiModel};
use memme_llm::openai::{OpenAIConfig, OpenAIProvider};
use memme_llm::LlmProvider;

const MAX_RETRIES: usize = 3;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Create a MemoryStore using OpenAI-format embedding API.
///
/// MEMME_EMBED_BASE_URL: the embedder calls `{base_url}/embeddings`,
///   so for Ollama use "http://localhost:11434/v1".
/// MEMME_LLM_BASE_URL: the LLM calls `{base_url}/v1/chat/completions`,
///   so for Ollama use "http://localhost:11434" (no /v1 suffix).
fn create_store() -> MemoryStore {
    let base_url = env_or("MEMME_EMBED_BASE_URL", "http://localhost:11434/v1");
    let api_key = env_or("MEMME_EMBED_API_KEY", "ollama");
    let model_name = env_or("MEMME_EMBED_MODEL", "mxbai-embed-large");
    let dims: usize = env_or("MEMME_EMBED_DIMS", "1024").parse().unwrap();

    let embedder = OpenAiEmbedder::new(api_key)
        .with_base_url(base_url)
        .with_model(OpenAiModel::Custom {
            name: model_name,
            dims,
        });
    let config = MemoryConfig::new(":memory:", dims);
    MemoryStore::new(config, Arc::new(embedder)).expect("Failed to create MemoryStore")
}

/// Create an LLM provider using OpenAI-format API.
fn create_llm() -> Arc<dyn LlmProvider> {
    let config = OpenAIConfig {
        base_url: env_or("MEMME_LLM_BASE_URL", "http://localhost:11434"),
        api_key: env_or("MEMME_LLM_API_KEY", "ollama"),
        model: env_or("MEMME_LLM_MODEL", "llama3.2"),
    };
    Arc::new(OpenAIProvider::new(config))
}

/// Helper: run a closure on a blocking thread to avoid the
/// "cannot block_on inside a runtime" panic from OllamaProvider.
async fn run_blocking<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    tokio::task::spawn_blocking(f).await.unwrap()
}

/// Retry add_smart up to MAX_RETRIES times, since llama3.2 (3B) sometimes
/// returns malformed JSON that fails to parse.
fn add_smart_with_retry(
    store: &MemoryStore,
    text: &str,
    user_id: &str,
    llm: &Arc<dyn LlmProvider>,
) -> Result<Vec<MemoryResult>, memme_core::error::MemoryError> {
    for attempt in 1..=MAX_RETRIES {
        let start = Instant::now();
        match store.add_smart(text, user_id, None, None, None, llm.clone(), false) {
            Ok(result) => {
                println!(
                    "  [attempt {attempt}] ok, {} memories in {:?}",
                    result.memories.len(),
                    start.elapsed()
                );
                return Ok(result.memories);
            }
            Err(e) => {
                println!("  [attempt {attempt}] failed in {:?}: {e}", start.elapsed());
                if attempt == MAX_RETRIES {
                    return Err(e);
                }
                println!("  retrying...");
            }
        }
    }
    unreachable!()
}

// ---------------------------------------------------------------------------
// Test 1: Smart add — extract facts from a conversation
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_llm_smart_add() {
    println!("\n=== test_llm_smart_add ===\n");

    let results = run_blocking(move || {
        let store = create_store();
        let llm = create_llm();

        let input = "Hi, I'm Alice. I work at Google as a software engineer. I love drinking coffee every morning.";
        println!("Input: {input}");
        add_smart_with_retry(&store, input, "user1", &llm)
    })
    .await;

    match results {
        Ok(memories) => {
            println!("\nExtracted {} memories:", memories.len());
            for (i, m) in memories.iter().enumerate() {
                println!("  [{i}] id={} content={:?}", m.id, m.content);
            }
            assert!(
                !memories.is_empty(),
                "Expected at least one memory to be extracted"
            );
        }
        Err(e) => {
            println!("add_smart failed after {MAX_RETRIES} retries: {e}");
            println!(
                "NOTE: llama3.2 (3B) may produce malformed JSON. Consider using a larger model."
            );
            panic!("add_smart returned error: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: Smart update — update contradicting information
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_llm_smart_update() {
    println!("\n=== test_llm_smart_update ===\n");

    let all = run_blocking(move || {
        let store = create_store();
        let llm = create_llm();

        // First: add initial fact
        println!("Step 1: Adding 'I work at Google'");
        match add_smart_with_retry(&store, "I work at Google", "user1", &llm) {
            Ok(mems) => {
                for m in &mems {
                    println!("  -> id={} content={:?}", m.id, m.content);
                }
            }
            Err(e) => panic!("First add_smart failed after retries: {e}"),
        }

        // Second: update with contradicting fact
        println!("\nStep 2: Adding 'Actually I just moved to Apple'");
        match add_smart_with_retry(&store, "Actually I just moved to Apple", "user1", &llm) {
            Ok(mems) => {
                for m in &mems {
                    println!("  -> id={} content={:?}", m.id, m.content);
                }
            }
            Err(e) => println!("  Second add_smart failed (may be parsing): {e}"),
        }

        // List all memories to see the final state
        let all = store.list_traces(ListOptions::new("user1")).unwrap();
        println!("\nFinal state - {} memories:", all.len());
        for m in &all {
            println!("  id={} content={:?}", m.id, m.content);
        }
        all
    })
    .await;

    // Flexible assertion: we should have at least one memory mentioning Apple
    let mentions_apple = all
        .iter()
        .any(|m| m.content.to_lowercase().contains("apple"));
    assert!(
        mentions_apple,
        "Expected at least one memory mentioning 'Apple' after the update"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Smart search — semantic search over extracted memories
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_llm_smart_search() {
    println!("\n=== test_llm_smart_search ===\n");

    let (music_results, food_results) = run_blocking(move || {
        let store = create_store();
        let llm = create_llm();

        // Add several memories
        let inputs = [
            "My name is Bob and I love playing guitar",
            "I'm a big fan of Japanese cuisine, especially sushi and ramen",
            "I work as a data scientist at Microsoft",
        ];

        for input in &inputs {
            println!("Adding: {input}");
            match add_smart_with_retry(&store, input, "user1", &llm) {
                Ok(mems) => {
                    for m in &mems {
                        println!("  -> {:?}", m.content);
                    }
                }
                Err(e) => println!("  -> Error after retries: {e} (continuing)"),
            }
        }

        // Search for music-related content
        println!("\nSearching for 'music instruments'...");
        let start = Instant::now();
        let music_results = store
            .search("music instruments", SearchOptions::new("user1").limit(5))
            .unwrap();
        println!("  Search took: {:?}", start.elapsed());
        println!("  Found {} results:", music_results.len());
        for r in &music_results {
            println!(
                "    score={:.4} content={:?}",
                r.score.unwrap_or(0.0),
                r.content
            );
        }

        // Search for food-related content
        println!("\nSearching for 'food and restaurants'...");
        let start = Instant::now();
        let food_results = store
            .search("food and restaurants", SearchOptions::new("user1").limit(5))
            .unwrap();
        println!("  Search took: {:?}", start.elapsed());
        println!("  Found {} results:", food_results.len());
        for r in &food_results {
            println!(
                "    score={:.4} content={:?}",
                r.score.unwrap_or(0.0),
                r.content
            );
        }

        (music_results, food_results)
    })
    .await;

    assert!(
        !music_results.is_empty(),
        "Expected search to return results for 'music instruments'"
    );
    assert!(
        !food_results.is_empty(),
        "Expected search to return results for 'food and restaurants'"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Graph extraction — entities and relationships
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_llm_graph_extraction() {
    println!("\n=== test_llm_graph_extraction ===\n");

    let (graph, search_result) = run_blocking(move || {
        let store = create_store();
        let llm = create_llm();

        let text =
            "Alice works at Google. Bob is Alice's manager. Google is headquartered in Mountain View.";
        println!("Input: {text}");

        let start = Instant::now();
        let result = store.add_graph(text, "user1", llm);
        let elapsed = start.elapsed();
        println!("add_graph took: {elapsed:?}");

        let graph = match result {
            Ok(g) => {
                println!("\nExtracted {} entities:", g.entities.len());
                for e in &g.entities {
                    println!("  [{:?}] {} (type={:?})", e.id, e.name, e.entity_type);
                }

                println!("\nExtracted {} relations:", g.relations.len());
                for r in &g.relations {
                    println!("  {} --[{}]--> {}", r.source, r.relation_type, r.target);
                }
                g
            }
            Err(e) => {
                println!("add_graph failed: {e}");
                panic!("add_graph returned error: {e}");
            }
        };

        // Also test graph search
        println!("\nSearching graph for 'Alice'...");
        let search_result = store.search_graph("Alice", "user1", 2);
        match &search_result {
            Ok(g) => {
                println!("  Found {} entities, {} relations", g.entities.len(), g.relations.len());
                for e in &g.entities {
                    println!("    Entity: {} ({:?})", e.name, e.entity_type);
                }
                for r in &g.relations {
                    println!("    Relation: {} --[{}]--> {}", r.source, r.relation_type, r.target);
                }
            }
            Err(e) => println!("  Graph search error: {e}"),
        }

        (graph, search_result)
    })
    .await;

    // Assertions on graph extraction
    assert!(
        !graph.entities.is_empty(),
        "Expected at least one entity to be extracted"
    );
    assert!(
        !graph.relations.is_empty(),
        "Expected at least one relationship to be extracted"
    );

    let entity_names: Vec<String> = graph
        .entities
        .iter()
        .map(|e| e.name.to_lowercase())
        .collect();
    println!("\nEntity names (lowercase): {entity_names:?}");

    let has_alice = entity_names.iter().any(|n| n.contains("alice"));
    let has_google = entity_names.iter().any(|n| n.contains("google"));
    assert!(has_alice, "Expected entity 'Alice' to be extracted");
    assert!(has_google, "Expected entity 'Google' to be extracted");

    // Graph search should also work
    assert!(search_result.is_ok(), "Graph search should succeed");
}

// ---------------------------------------------------------------------------
// Test 5: End-to-end flow
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_llm_end_to_end() {
    println!("\n=== test_llm_end_to_end ===\n");

    let all = run_blocking(move || {
        let store = create_store();
        let llm = create_llm();

        // Step 1: Add initial facts
        println!("Step 1: add_smart(\"I'm Bob, I love coffee and work at Apple\")");
        match add_smart_with_retry(
            &store,
            "I'm Bob, I love coffee and work at Apple",
            "user1",
            &llm,
        ) {
            Ok(mems) => {
                for m in &mems {
                    println!("  -> {:?}", m.content);
                }
            }
            Err(e) => panic!("Step 1 failed after retries: {e}"),
        }

        // Step 2: Add more facts
        println!("\nStep 2: add_smart(\"My favorite programming language is Rust\")");
        match add_smart_with_retry(
            &store,
            "My favorite programming language is Rust",
            "user1",
            &llm,
        ) {
            Ok(mems) => {
                for m in &mems {
                    println!("  -> {:?}", m.content);
                }
            }
            Err(e) => panic!("Step 2 failed after retries: {e}"),
        }

        // Step 3: Search for beverages
        println!("\nStep 3: search('beverages')");
        let start = Instant::now();
        let results = store
            .search("beverages", SearchOptions::new("user1").limit(5))
            .unwrap();
        println!("  Search took: {:?}", start.elapsed());
        println!("  Found {} results:", results.len());
        for r in &results {
            println!(
                "    score={:.4} content={:?}",
                r.score.unwrap_or(0.0),
                r.content
            );
        }

        let has_coffee = results
            .iter()
            .any(|r| r.content.to_lowercase().contains("coffee"));
        println!("  Contains 'coffee': {has_coffee}");

        // Step 4: Update preference
        println!("\nStep 4: add_smart(\"I switched from coffee to tea\")");
        match add_smart_with_retry(&store, "I switched from coffee to tea", "user1", &llm) {
            Ok(mems) => {
                for m in &mems {
                    println!("  -> {:?}", m.content);
                }
            }
            Err(e) => println!("  Step 4 failed (may be parsing): {e}"),
        }

        // Step 5: Search again
        println!("\nStep 5: search('beverages') after update");
        let start = Instant::now();
        let results = store
            .search("beverages", SearchOptions::new("user1").limit(5))
            .unwrap();
        println!("  Search took: {:?}", start.elapsed());
        println!("  Found {} results:", results.len());
        for r in &results {
            println!(
                "    score={:.4} content={:?}",
                r.score.unwrap_or(0.0),
                r.content
            );
        }

        // Step 6: List all memories
        println!("\nStep 6: list all memories");
        let all = store
            .list_traces(ListOptions::new("user1").limit(100))
            .unwrap();
        println!("  Total memories: {}", all.len());
        for (i, m) in all.iter().enumerate() {
            println!("  [{i}] id={} content={:?}", m.id, m.content);
        }
        all
    })
    .await;

    assert!(
        !all.is_empty(),
        "Expected at least one memory in the final state"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Hybrid search — vector + FTS with RRF fusion
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_llm_hybrid_search() {
    println!("\n=== test_llm_hybrid_search ===\n");

    let (vector_results, hybrid_results) = run_blocking(move || {
        let store = create_store();
        let llm = create_llm();

        // Add diverse memories via add_smart
        let inputs = [
            "I enjoy hiking in the mountains during summer",
            "My cat Luna loves sleeping on the keyboard",
            "I recently started learning Japanese language",
            "Python and Rust are my favorite programming languages",
            "I brew espresso every morning with my Breville machine",
        ];

        for input in &inputs {
            println!("Adding: {input}");
            match add_smart_with_retry(&store, input, "user1", &llm) {
                Ok(mems) => {
                    for m in &mems {
                        println!("  -> {:?}", m.content);
                    }
                }
                Err(e) => println!("  -> Error after retries: {e} (continuing)"),
            }
        }

        // Rebuild FTS index for hybrid search
        println!("\nRebuilding FTS index...");
        let start = Instant::now();
        store.rebuild_fts_index().unwrap();
        println!("  took: {:?}", start.elapsed());

        let query = "coffee espresso morning";

        // Pure vector search
        println!("\nVector search for '{query}':");
        let start = Instant::now();
        let vector_results = store
            .search(query, SearchOptions::new("user1").limit(5))
            .unwrap();
        println!("  took: {:?}", start.elapsed());
        println!("  Found {} results:", vector_results.len());
        for r in &vector_results {
            println!(
                "    score={:.4} content={:?}",
                r.score.unwrap_or(0.0),
                r.content
            );
        }

        // Hybrid search (using search with BM25 enabled)
        println!("\nHybrid search for '{query}':");
        let start = Instant::now();
        let hybrid_results = store
            .search(
                query,
                SearchOptions::new("user1").limit(5).keyword_search(true),
            )
            .unwrap();
        println!("  took: {:?}", start.elapsed());
        println!("  Found {} results:", hybrid_results.len());
        for r in &hybrid_results {
            println!(
                "    score={:.4} content={:?}",
                r.score.unwrap_or(0.0),
                r.content
            );
        }

        // Compare orderings
        println!("\n--- Comparison ---");
        println!(
            "Vector top: {:?}",
            vector_results.first().map(|r| &r.content)
        );
        println!(
            "Hybrid top: {:?}",
            hybrid_results.first().map(|r| &r.content)
        );

        (vector_results, hybrid_results)
    })
    .await;

    // Both should return results
    assert!(
        !vector_results.is_empty(),
        "Expected vector search to return results"
    );
    assert!(
        !hybrid_results.is_empty(),
        "Expected hybrid search to return results"
    );
}

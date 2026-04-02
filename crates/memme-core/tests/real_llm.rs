//! Real-world integration tests using DashScope (Qwen) API.
//!
//! These tests hit a live API and are gated with `#[ignore]`.
//! Run with: `cargo test -p memme-core --features smart --test real_llm -- --ignored --test-threads=1`

use std::sync::Arc;

use memme_core::config::MemoryConfig;
use memme_core::memory::MemoryStore;
use memme_core::types::*;
use memme_embeddings::openai::{OpenAiEmbedder, OpenAiModel};
use memme_embeddings::Embedder;
use memme_llm::openai::{OpenAIConfig, OpenAIProvider};

fn api_key() -> String {
    std::env::var("DASHSCOPE_API_KEY").expect("DASHSCOPE_API_KEY env var must be set")
}

/// Brief pause between LLM calls to avoid API rate limiting.
fn api_pause() {
    std::thread::sleep(std::time::Duration::from_secs(2));
}
fn embedding_base_url() -> String {
    std::env::var("EMBEDDING_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1/embeddings".to_string())
}

fn llm_base_url() -> String {
    std::env::var("LLM_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1/chat/completions".to_string())
}

fn embedding_model() -> String {
    std::env::var("EMBEDDING_MODEL").unwrap_or_else(|_| "text-embedding-3-small".to_string())
}

fn llm_model() -> String {
    std::env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string())
}

fn embedding_dims() -> usize {
    std::env::var("EMBEDDING_DIMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1536)
}

fn make_embedder() -> Arc<OpenAiEmbedder> {
    Arc::new(
        OpenAiEmbedder::new(&api_key(), &embedding_base_url())
            .with_model(OpenAiModel::Custom {
                name: embedding_model(),
                dims: embedding_dims(),
                send_dims: true,
            }),
    )
}

fn make_llm() -> Arc<OpenAIProvider> {
    let config = OpenAIConfig {
        api_key: api_key(),
        base_url: llm_base_url(),
        model: llm_model(),
    };
    Arc::new(OpenAIProvider::new(config))
}

fn make_config(collection: &str, enable_graph: bool) -> MemoryConfig {
    MemoryConfig {
        db_path: ":memory:".into(),
        collection_name: collection.into(),
        embedding_dims: embedding_dims(),
        enable_graph,
        ..Default::default()
    }
}

fn make_real_store() -> MemoryStore {
    let embedder = make_embedder();
    let llm = make_llm();
    let config = make_config("real_test", true);
    MemoryStore::new(config, embedder).unwrap().with_llm(llm)
}

fn make_store_no_graph() -> MemoryStore {
    let embedder = make_embedder();
    let llm = make_llm();
    let config = make_config("real_nograph", false);
    MemoryStore::new(config, embedder).unwrap().with_llm(llm)
}

// ============================================================
// Embedding Quality Tests
// ============================================================

#[test]
#[ignore]
fn test_real_embedding_basic() {
    let embedder = make_embedder();
    let result = embedder.embed("hello world").unwrap();
    eprintln!("Embedding dims: {}", result.len());
    eprintln!("First 5 values: {:?}", &result[..5]);
    assert_eq!(
        result.len(),
        embedding_dims(),
        "Expected {} dimensions",
        embedding_dims()
    );
    // Verify it's not all zeros
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!("L2 norm: {}", norm);
    assert!(norm > 0.1, "Embedding should not be a zero vector");
}

#[test]
#[ignore]
fn test_real_embedding_chinese() {
    let embedder = make_embedder();
    let result = embedder.embed("我喜欢喝咖啡").unwrap();
    eprintln!("Chinese embedding dims: {}", result.len());
    assert_eq!(
        result.len(),
        embedding_dims(),
        "Expected {} dimensions for Chinese text",
        embedding_dims()
    );
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!("Chinese embedding L2 norm: {}", norm);
    assert!(norm > 0.1, "Chinese embedding should not be a zero vector");
}

#[test]
#[ignore]
fn test_real_semantic_similarity() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_store_no_graph();

    store
        .add(
            "I love coffee and drink it every morning",
            AddOptions::new("user1"),
        )
        .unwrap();
    store
        .add(
            "I enjoy drinking tea in the afternoon",
            AddOptions::new("user1"),
        )
        .unwrap();
    store
        .add(
            "The stock market crashed yesterday",
            AddOptions::new("user1"),
        )
        .unwrap();

    let results = store
        .search(
            "hot beverages I like to drink",
            SearchOptions::new("user1").limit(10),
        )
        .unwrap();

    eprintln!("Search results for 'hot beverages':");
    for r in &results {
        eprintln!("  score={:?} content={}", r.score, r.content);
    }

    assert!(results.len() >= 2, "Should find at least 2 results");
    // The top 2 results should be about coffee/tea, not stock market
    let top2_contents: Vec<&str> = results.iter().take(2).map(|r| r.content.as_str()).collect();
    let has_beverage = top2_contents
        .iter()
        .any(|c| c.contains("coffee") || c.contains("tea"));
    assert!(
        has_beverage,
        "Top results should include beverage-related memories, got: {:?}",
        top2_contents
    );
}

#[test]
#[ignore]
fn test_real_dissimilar_content() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_store_no_graph();

    store
        .add(
            "Quantum entanglement allows particles to be correlated over vast distances",
            AddOptions::new("user1"),
        )
        .unwrap();
    store
        .add(
            "To make chocolate cake, mix flour, sugar, cocoa powder, and eggs",
            AddOptions::new("user1"),
        )
        .unwrap();

    let results = store
        .search(
            "physics research papers",
            SearchOptions::new("user1").limit(5),
        )
        .unwrap();

    eprintln!("Search results for 'physics research papers':");
    for r in &results {
        eprintln!("  score={:?} content={}", r.score, r.content);
    }

    assert!(!results.is_empty(), "Should find results");
    assert!(
        results[0].content.contains("Quantum") || results[0].content.contains("entanglement"),
        "Top result should be about physics, got: {}",
        results[0].content
    );
}

#[test]
#[ignore]
fn test_real_multilingual_search() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_store_no_graph();

    store
        .add(
            "I work as a software engineer at a tech company",
            AddOptions::new("user1"),
        )
        .unwrap();
    store
        .add(
            "我是一名软件工程师，在科技公司工作",
            AddOptions::new("user1"),
        )
        .unwrap();
    store
        .add(
            "I love hiking in the mountains on weekends",
            AddOptions::new("user1"),
        )
        .unwrap();

    // Search in Chinese for English content
    let results_cn = store
        .search("软件开发工作", SearchOptions::new("user1").limit(5))
        .unwrap();
    eprintln!("Chinese query results:");
    for r in &results_cn {
        eprintln!("  score={:?} content={}", r.score, r.content);
    }
    assert!(
        !results_cn.is_empty(),
        "Chinese query should return results"
    );
    // At least one of the top 2 should be about software engineering
    let top2_about_work = results_cn
        .iter()
        .take(2)
        .any(|r| r.content.contains("software") || r.content.contains("软件"));
    assert!(
        top2_about_work,
        "Chinese query should find work-related content"
    );

    // Search in English for Chinese content
    let results_en = store
        .search(
            "software engineer job",
            SearchOptions::new("user1").limit(5),
        )
        .unwrap();
    eprintln!("English query results:");
    for r in &results_en {
        eprintln!("  score={:?} content={}", r.score, r.content);
    }
    assert!(
        !results_en.is_empty(),
        "English query should return results"
    );
    let top2_about_work_en = results_en
        .iter()
        .take(2)
        .any(|r| r.content.contains("software") || r.content.contains("软件"));
    assert!(
        top2_about_work_en,
        "English query should find work-related content"
    );
}

// ============================================================
// Smart Mode Tests (LLM fact extraction)
// ============================================================

#[test]
#[ignore]
fn test_real_smart_add_messages() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_store_no_graph();
    let llm = make_llm();

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "My favorite programming language is Rust and I use it for all my projects."
                .to_string(),
            image_url: None,
            image_type: None,
            timestamp: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "Rust is great for performance and safety!".to_string(),
            image_url: None,
            image_type: None,
            timestamp: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Yes, I also use Python for data analysis at my job.".to_string(),
            image_url: None,
            image_type: None,
            timestamp: None,
        },
    ];

    let store = store.with_llm(llm);
    let session_id = "test-session";

    // Phase 1: append events
    store
        .append_events(session_id, &messages, "user1", None)
        .unwrap();

    // Phase 2: compact
    let result = store.compact(session_id).unwrap();

    eprintln!(
        "Smart add messages result: {} memories extracted",
        result.memories.len()
    );
    for m in &result.memories {
        eprintln!("  - {}", m.content);
    }
    assert!(
        !result.memories.is_empty(),
        "Should extract facts from chat messages"
    );
}

// ============================================================
// Graph Memory Tests
// ============================================================

#[test]
#[ignore]
fn test_real_graph_entity_extraction() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_real_store();
    let llm = make_llm();

    let result = store
        .add_graph("Alice works at Google as a senior engineer.", "user1", llm)
        .unwrap();

    eprintln!("Graph entities extracted: {}", result.entities.len());
    for e in &result.entities {
        eprintln!("  Entity: {} (type={:?})", e.name, e.entity_type);
    }
    eprintln!("Graph relations extracted: {}", result.relations.len());
    for r in &result.relations {
        eprintln!(
            "  Relation: {} --[{}]--> {}",
            r.source, r.relation_type, r.target
        );
    }

    assert!(
        !result.entities.is_empty(),
        "Should extract entities from text"
    );
    let entity_names: Vec<String> = result
        .entities
        .iter()
        .map(|e| e.name.to_lowercase())
        .collect();
    assert!(
        entity_names.iter().any(|n| n.contains("alice")),
        "Should extract 'Alice' as entity, got: {:?}",
        entity_names
    );
    assert!(
        entity_names.iter().any(|n| n.contains("google")),
        "Should extract 'Google' as entity, got: {:?}",
        entity_names
    );
}

#[test]
#[ignore]
fn test_real_graph_relationship() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_real_store();
    let llm = make_llm();

    let result = store
        .add_graph(
            "Bob is Alice's manager at the engineering department.",
            "user1",
            llm,
        )
        .unwrap();

    eprintln!("Graph relationship result:");
    for r in &result.relations {
        eprintln!("  {} --[{}]--> {}", r.source, r.relation_type, r.target);
    }

    assert!(
        !result.relations.is_empty(),
        "Should extract at least one relationship"
    );
    // Verify a relationship involving Bob and Alice exists
    let has_bob_alice = result.relations.iter().any(|r| {
        (r.source.to_lowercase().contains("bob") && r.target.to_lowercase().contains("alice"))
            || (r.source.to_lowercase().contains("alice")
                && r.target.to_lowercase().contains("bob"))
    });
    assert!(
        has_bob_alice,
        "Should extract a relationship between Bob and Alice"
    );
}

#[test]
#[ignore]
fn test_real_graph_search() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_real_store();
    let llm = make_llm();

    store
        .add_graph("Alice works at Google.", "user1", llm.clone())
        .unwrap();
    api_pause();
    store
        .add_graph(
            "Bob is Alice's friend and works at Meta.",
            "user1",
            llm.clone(),
        )
        .unwrap();
    api_pause();
    store
        .add_graph("Charlie is Bob's colleague at Meta.", "user1", llm)
        .unwrap();

    let result = store.search_graph("Alice", "user1", 2).unwrap();

    eprintln!("Graph search for 'Alice':");
    eprintln!("  Entities: {}", result.entities.len());
    for e in &result.entities {
        eprintln!("    {} (type={:?})", e.name, e.entity_type);
    }
    eprintln!("  Relations: {}", result.relations.len());
    for r in &result.relations {
        eprintln!("    {} --[{}]--> {}", r.source, r.relation_type, r.target);
    }

    assert!(
        !result.entities.is_empty(),
        "Graph search should find entities related to Alice"
    );
}

// ============================================================
// Hybrid Search Tests
// ============================================================

#[test]
#[ignore]
fn test_real_hybrid_search() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_store_no_graph();

    let topics = [
        "Machine learning models need large datasets for training",
        "The best restaurants in Tokyo serve fresh sushi",
        "Python is widely used for data science applications",
        "Classical music concerts are held at the symphony hall",
        "Electric vehicles are becoming more affordable each year",
        "Mediterranean diet includes olive oil and fresh vegetables",
        "Space exploration has advanced significantly since the 1960s",
        "Remote work became popular during the pandemic",
        "Blockchain technology enables decentralized transactions",
        "Yoga and meditation help reduce stress and anxiety",
    ];

    for topic in &topics {
        store.add(*topic, AddOptions::new("user1")).unwrap();
    }

    // Rebuild FTS index
    store.rebuild_fts_index().unwrap();

    let results = store
        .search(
            "data science and machine learning",
            SearchOptions::new("user1").keyword_search(true).limit(5),
        )
        .unwrap();

    eprintln!("Hybrid search results for 'data science and machine learning':");
    for r in &results {
        eprintln!("  score={:?} content={}", r.score, r.content);
    }

    assert!(!results.is_empty(), "Hybrid search should return results");
    // Top results should be about ML or data science
    let top_content = &results[0].content;
    assert!(
        top_content.contains("Machine learning")
            || top_content.contains("Python")
            || top_content.contains("data"),
        "Top hybrid result should be about ML/data science, got: {}",
        top_content
    );
}

#[test]
#[ignore]
fn test_real_keyword_search_flag() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_store_no_graph();

    store
        .add(
            "The Rust programming language is memory safe",
            AddOptions::new("user1"),
        )
        .unwrap();
    store
        .add(
            "Iron oxide causes rust on metal surfaces",
            AddOptions::new("user1"),
        )
        .unwrap();
    store
        .add(
            "Go is another systems programming language",
            AddOptions::new("user1"),
        )
        .unwrap();

    // Rebuild FTS index
    store.rebuild_fts_index().unwrap();

    // Search with keyword_search enabled
    let results = store
        .search(
            "Rust programming",
            SearchOptions::new("user1").keyword_search(true).limit(5),
        )
        .unwrap();

    eprintln!("Keyword search results for 'Rust programming':");
    for r in &results {
        eprintln!("  score={:?} content={}", r.score, r.content);
    }

    assert!(!results.is_empty(), "Keyword search should return results");
}

// ============================================================
// Full Workflow Tests
// ============================================================

#[test]
#[ignore]
fn test_real_full_lifecycle() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_store_no_graph();

    // Add
    let added = store
        .add(
            "I have a golden retriever named Max who loves to play fetch",
            AddOptions::new("user1"),
        )
        .unwrap();
    eprintln!("Added memory: id={}, content={}", added.id, added.content);
    assert_eq!(added.user_id, "user1");
    let id = added.id.clone();

    // Search — should find it
    let results = store
        .search("my pet dog", SearchOptions::new("user1"))
        .unwrap();
    eprintln!("Search for 'my pet dog': {} results", results.len());
    assert!(
        results.iter().any(|r| r.id == id),
        "Should find the added memory when searching for 'my pet dog'"
    );

    // Update
    let updated = store
        .update_trace(
            &id,
            "I have a golden retriever named Max and a cat named Luna",
            None,
        )
        .unwrap();
    assert_eq!(updated.id, id);
    eprintln!("Updated content: {}", updated.content);

    // Search again — should find updated content
    let results2 = store
        .search("cat Luna", SearchOptions::new("user1"))
        .unwrap();
    eprintln!("Search for 'cat Luna': {} results", results2.len());
    assert!(
        results2.iter().any(|r| r.content.contains("Luna")),
        "Should find updated memory containing Luna"
    );

    // Delete
    store.delete_trace(&id).unwrap();

    // Verify deleted
    let fetched = store.get_trace(&id).unwrap();
    assert!(fetched.is_none(), "Memory should be deleted");
    eprintln!("Memory successfully deleted");
}

#[test]
#[ignore]
fn test_real_multi_user_isolation() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_store_no_graph();

    store
        .add(
            "User A likes classical music and plays piano",
            AddOptions::new("user_a"),
        )
        .unwrap();
    store
        .add(
            "User B likes rock music and plays guitar",
            AddOptions::new("user_b"),
        )
        .unwrap();

    // Search as user_a
    let results_a = store
        .search("music preferences", SearchOptions::new("user_a"))
        .unwrap();
    eprintln!("user_a search results: {} results", results_a.len());
    for r in &results_a {
        eprintln!("  user_id={} content={}", r.user_id, r.content);
        assert_eq!(r.user_id, "user_a", "user_a should only see own memories");
    }

    // Search as user_b
    let results_b = store
        .search("music preferences", SearchOptions::new("user_b"))
        .unwrap();
    eprintln!("user_b search results: {} results", results_b.len());
    for r in &results_b {
        eprintln!("  user_id={} content={}", r.user_id, r.content);
        assert_eq!(r.user_id, "user_b", "user_b should only see own memories");
    }

    assert!(
        !results_a.is_empty(),
        "user_a should find their music memory"
    );
    assert!(
        !results_b.is_empty(),
        "user_b should find their music memory"
    );
}

#[test]
#[ignore]
fn test_real_export_import_with_embeddings() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();

    // Create store 1 and add memories
    let store1 = make_store_no_graph();
    store1
        .add(
            "Rust is a systems programming language",
            AddOptions::new("user1"),
        )
        .unwrap();
    store1
        .add(
            "Python is great for machine learning",
            AddOptions::new("user1"),
        )
        .unwrap();
    store1
        .add("JavaScript runs in the browser", AddOptions::new("user1"))
        .unwrap();

    // Export
    let exported = store1.export(Some("user1")).unwrap();
    eprintln!("Exported {} memories", exported.len());
    assert_eq!(exported.len(), 3, "Should export 3 memories");

    // Create store 2 and import
    let embedder2 = make_embedder();
    let config2 = make_config("import_test", false);
    let store2 = MemoryStore::new(config2, embedder2).unwrap();

    let imported_count = store2.import_memories(&exported).unwrap();
    eprintln!("Imported {} memories", imported_count);
    assert_eq!(imported_count, 3, "Should import 3 memories");

    // Search in store 2 — should find relevant results
    let results = store2
        .search("systems programming", SearchOptions::new("user1").limit(5))
        .unwrap();
    eprintln!("Search in imported store:");
    for r in &results {
        eprintln!("  score={:?} content={}", r.score, r.content);
    }
    assert!(!results.is_empty(), "Should find results in imported store");
    assert!(
        results[0].content.contains("Rust") || results[0].content.contains("systems"),
        "Top result should be about Rust/systems programming"
    );
}

#[test]
#[ignore]
fn test_real_consolidate_with_decay() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_store_no_graph();

    store
        .add(
            "Old memory that should decay over time",
            AddOptions::new("user1").importance(0.5),
        )
        .unwrap();
    store
        .add(
            "Another old memory for decay testing",
            AddOptions::new("user1").importance(0.3),
        )
        .unwrap();

    // Run consolidation with decay
    let result = store.consolidate("user1", 0.01, 0.05, false).unwrap();
    eprintln!(
        "Consolidation: decayed={}, deleted={}",
        result.decayed_count, result.deleted_count
    );

    // The memories should still exist (decay is time-based, just created)
    let list = store.list_traces(ListOptions::new("user1")).unwrap();
    eprintln!("Memories after consolidation: {}", list.len());
    assert_eq!(
        list.len(),
        2,
        "Memories should still exist after consolidation (just created)"
    );
}

#[test]
#[ignore]
fn test_real_categories_and_filter() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_store_no_graph();

    store
        .add(
            "I have a meeting with the engineering team at 3pm",
            AddOptions::new("user1").categories(vec!["work".into(), "meetings".into()]),
        )
        .unwrap();
    store
        .add(
            "Need to buy groceries: milk, eggs, bread",
            AddOptions::new("user1").categories(vec!["personal".into(), "shopping".into()]),
        )
        .unwrap();
    store
        .add(
            "Code review for the authentication module is pending",
            AddOptions::new("user1").categories(vec!["work".into(), "code_review".into()]),
        )
        .unwrap();

    // Search with category filter
    let results = store
        .search(
            "tasks",
            SearchOptions::new("user1")
                .filter(FilterExpression::contains("categories", "work"))
                .limit(10),
        )
        .unwrap();

    eprintln!("Filtered search results (categories contains 'work'):");
    for r in &results {
        eprintln!("  content={} categories={:?}", r.content, r.categories);
    }

    assert!(!results.is_empty(), "Should find work-related memories");
    for r in &results {
        let cats = r.categories.as_ref().expect("Should have categories");
        assert!(
            cats.contains(&"work".to_string()),
            "Filtered results should have 'work' category, got: {:?}",
            cats
        );
    }
}

// ============================================================
// Edge Cases with Real API
// ============================================================

#[test]
#[ignore]
fn test_real_empty_search_query() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_store_no_graph();

    store
        .add(
            "Some random memory content for testing",
            AddOptions::new("user1"),
        )
        .unwrap();
    store
        .add(
            "Another piece of information stored here",
            AddOptions::new("user1"),
        )
        .unwrap();

    // Search with a very generic query
    let results = store
        .search("things", SearchOptions::new("user1").limit(10))
        .unwrap();

    eprintln!("Generic search results: {} results", results.len());
    for r in &results {
        eprintln!("  score={:?} content={}", r.score, r.content);
    }
    // Should not panic and should return results
    assert!(
        !results.is_empty(),
        "Generic search should still return results"
    );
}

#[test]
#[ignore]
fn test_real_dedup_semantic() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_store_no_graph();

    let first = store
        .add(
            "I love drinking coffee every morning",
            AddOptions::new("user1"),
        )
        .unwrap();
    let second = store
        .add(
            "I love drinking coffee every morning",
            AddOptions::new("user1"),
        )
        .unwrap();

    eprintln!("First add id: {}", first.id);
    eprintln!("Second add id: {}", second.id);
    eprintln!("Same id (dedup): {}", first.id == second.id);

    // Exact duplicate should be deduped
    assert_eq!(
        first.id, second.id,
        "Identical content should be deduped to same memory"
    );

    // Verify only one memory exists
    let list = store.list_traces(ListOptions::new("user1")).unwrap();
    eprintln!("Total memories after dedup: {}", list.len());
    assert_eq!(list.len(), 1, "Should have exactly 1 memory after dedup");
}

// ============================================================
// Additional Tests
// ============================================================

#[test]
#[ignore]
fn test_real_batch_embed() {
    let embedder = make_embedder();
    let texts = &["hello world", "machine learning", "我喜欢编程"];
    let results = embedder.embed_batch(texts).unwrap();

    eprintln!(
        "Batch embedding: {} texts -> {} embeddings",
        texts.len(),
        results.len()
    );
    assert_eq!(results.len(), 3, "Should return 3 embeddings for 3 texts");
    for (i, emb) in results.iter().enumerate() {
        assert_eq!(
            emb.len(),
            embedding_dims(),
            "Embedding {} should have {} dims",
            i,
            embedding_dims()
        );
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("  Embedding {} norm: {}", i, norm);
        assert!(norm > 0.1, "Embedding {} should not be zero", i);
    }
}

#[test]
#[ignore]
fn test_real_search_all_with_graph() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_real_store();
    let llm = make_llm();

    // Add vector memories
    store
        .add(
            "Alice is a software engineer at Google",
            AddOptions::new("user1"),
        )
        .unwrap();
    store
        .add(
            "Bob manages the Cloud team at Google",
            AddOptions::new("user1"),
        )
        .unwrap();

    // Add graph data
    api_pause();
    store
        .add_graph(
            "Alice works at Google. Bob is Alice's manager.",
            "user1",
            llm,
        )
        .unwrap();

    // Search (with graph enabled, entity-centric retrieval is included in multi-channel fusion)
    let results = store
        .search("who works at Google", SearchOptions::new("user1"))
        .unwrap();

    eprintln!("search results:");
    eprintln!("  memories: {}", results.len());
    for m in &results {
        eprintln!("    score={:?} content={}", m.score, m.content);
    }

    assert!(!results.is_empty(), "Should find memories");
}

#[test]
#[ignore]
fn test_real_importance_weighted_search() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_store_no_graph();

    store
        .add(
            "Regular daily standup meeting notes",
            AddOptions::new("user1").importance(0.2),
        )
        .unwrap();
    store
        .add(
            "Critical production incident: database outage affecting all users",
            AddOptions::new("user1").importance(0.95),
        )
        .unwrap();
    store
        .add(
            "Team lunch scheduled for Friday",
            AddOptions::new("user1").importance(0.1),
        )
        .unwrap();

    // Search normally (importance is factored in when forgetting curve is enabled)
    let results = store
        .search("work events", SearchOptions::new("user1").limit(3))
        .unwrap();

    eprintln!("Importance-weighted search results:");
    for r in &results {
        eprintln!(
            "  score={:?} importance={:?} content={}",
            r.score, r.importance, r.content
        );
    }

    assert!(!results.is_empty(), "Should return results");
    // The critical incident should rank higher due to high importance
    assert!(
        results[0].content.contains("Critical") || results[0].content.contains("incident"),
        "Highest importance memory should rank first with high importance_weight, got: {}",
        results[0].content
    );
}

#[test]
#[ignore]
fn test_real_history_with_real_embeddings() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    let store = make_store_no_graph();

    // Add a memory
    let added = store
        .add("My favorite color is blue", AddOptions::new("user1"))
        .unwrap();
    let id = added.id.clone();

    // Update it
    store
        .update_trace(&id, "My favorite color is green now", None)
        .unwrap();

    // Check history
    let history = store.trace_history(&id).unwrap();
    eprintln!("History for memory {}:", id);
    for h in &history {
        eprintln!(
            "  event={} old={:?} new={}",
            h.event,
            h.old_memory.as_deref().unwrap_or("(none)"),
            h.new_memory
        );
    }

    assert!(
        history.len() >= 2,
        "Should have at least ADD + UPDATE events"
    );
    assert_eq!(history[0].event, "ADD", "First event should be ADD");
    assert_eq!(history[1].event, "UPDATE", "Second event should be UPDATE");
    assert!(
        history[1].old_memory.as_deref() == Some("My favorite color is blue"),
        "UPDATE should preserve old content"
    );
}

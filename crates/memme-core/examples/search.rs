//! Hybrid search example: demonstrates vector search, keyword search, filters,
//! and search configuration options.
//!
//! Run with:
//!   cargo run -p memme-core --example search

use std::sync::Arc;

use memme_core::config::MemoryConfig;
use memme_core::memory::MemoryStore;
use memme_core::types::*;
use memme_embeddings::mock::MockEmbedder;
use serde_json::json;

fn main() {
    println!("=== MemMe Hybrid Search Example ===\n");

    // ── 1. Create store and populate with diverse memories ──────────
    let config = MemoryConfig {
        db_path: ":memory:".into(),
        collection_name: "search_demo".into(),
        embedding_dims: 384,
        dedup_threshold: 0.15,
        default_limit: 10,
        ..Default::default()
    };
    let embedder = Arc::new(MockEmbedder::new(384));
    let store = MemoryStore::new(config, embedder).expect("Failed to create MemoryStore");
    println!("[1] MemoryStore created.\n");

    // Add memories for two different users with varied metadata
    let memories = vec![
        // user_alice's memories
        (
            "User prefers Rust for systems programming",
            "user_alice",
            json!({"category": "tech"}),
            Some(0.9_f32),
            vec!["tech", "preferences"],
        ),
        (
            "User drinks matcha every morning before work",
            "user_alice",
            json!({"category": "habits"}),
            Some(0.5),
            vec!["habits", "food"],
        ),
        (
            "User works remotely from Tokyo, Japan",
            "user_alice",
            json!({"category": "location"}),
            Some(0.7),
            vec!["location", "work"],
        ),
        (
            "User is learning Japanese through weekly tutoring sessions",
            "user_alice",
            json!({"category": "learning"}),
            Some(0.6),
            vec!["learning", "language"],
        ),
        (
            "User built a personal note-taking app using Rust and SQLite",
            "user_alice",
            json!({"category": "projects"}),
            Some(0.8),
            vec!["tech", "projects"],
        ),
        (
            "User enjoys hiking in the mountains on weekends",
            "user_alice",
            json!({"category": "hobbies"}),
            Some(0.4),
            vec!["hobbies", "outdoor"],
        ),
        // user_bob's memories
        (
            "User is a Python developer specializing in machine learning",
            "user_bob",
            json!({"category": "tech"}),
            Some(0.8),
            vec!["tech", "career"],
        ),
        (
            "User runs 5km every morning before breakfast",
            "user_bob",
            json!({"category": "fitness"}),
            Some(0.6),
            vec!["fitness", "habits"],
        ),
        (
            "User lives in San Francisco and commutes to Palo Alto",
            "user_bob",
            json!({"category": "location"}),
            Some(0.7),
            vec!["location", "work"],
        ),
    ];

    println!("[2] Adding {} memories for two users...\n", memories.len());
    for (content, user_id, metadata, importance, categories) in &memories {
        let mut opts = AddOptions::new(*user_id)
            .metadata(metadata.clone())
            .categories(categories.iter().map(|s| s.to_string()).collect());
        if let Some(imp) = importance {
            opts = opts.importance(*imp);
        }
        store.add(content, opts).expect("Failed to add memory");
        println!("  Added: \"{}\" (user: {})", content, user_id);
    }
    println!();

    // ── 3. Basic vector search ──────────────────────────────────────
    println!("[3] Vector search: \"programming languages\" (user_alice, limit=3)...\n");

    let search_opts = SearchOptions::new("user_alice").limit(3);
    let results = store
        .search("programming languages", search_opts)
        .expect("Search failed");

    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: \"{}\" (score: {:.4})",
            i + 1,
            r.content,
            r.score.unwrap_or(0.0)
        );
    }
    println!();

    // ── 4. Search with keyword (FTS) enabled ────────────────────────
    // When keyword_search is enabled, both vector and BM25 full-text search
    // run in parallel and results are fused via Reciprocal Rank Fusion.
    println!("[4] Hybrid search (vector + keyword): \"morning routine\" (user_alice)...\n");

    let search_opts = SearchOptions::new("user_alice")
        .limit(3)
        .keyword_search(true);
    let results = store
        .search("morning routine", search_opts)
        .expect("Search failed");

    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: \"{}\" (score: {:.4})",
            i + 1,
            r.content,
            r.score.unwrap_or(0.0)
        );
    }
    println!();

    // ── 5. Search with metadata filter ──────────────────────────────
    // Filter results by metadata fields using FilterExpression.
    println!("[5] Search with category filter: \"tech\" category only...\n");

    let search_opts = SearchOptions::new("user_alice")
        .limit(5)
        .filter(FilterExpression::contains("categories", "tech"));
    let results = store
        .search("building software", search_opts)
        .expect("Search failed");

    println!("  Results filtered to 'tech' category:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: \"{}\" (categories: {:?})",
            i + 1,
            r.content,
            r.categories
        );
    }
    println!();

    // ── 6. Search with compound filter ──────────────────────────────
    // Combine multiple filter conditions with AND/OR logic.
    println!("[6] Search with compound filter: importance >= 0.7 AND category 'tech'...\n");

    let search_opts = SearchOptions::new("user_alice")
        .limit(5)
        .filter(FilterExpression::and(vec![
            FilterExpression::gte("importance", json!(0.7)),
            FilterExpression::contains("categories", "tech"),
        ]));
    let results = store
        .search("software development", search_opts)
        .expect("Search failed");

    println!("  High-importance tech memories:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: \"{}\" (importance: {:.1})",
            i + 1,
            r.content,
            r.importance.unwrap_or(0.0)
        );
    }
    println!();

    // ── 7. Search with score threshold ──────────────────────────────
    println!("[7] Search with score threshold: only results with score < 0.5...\n");

    let search_opts = SearchOptions::new("user_alice").limit(5).threshold(0.5);
    let results = store
        .search("hiking outdoors", search_opts)
        .expect("Search failed");

    println!("  Results passing threshold:");
    if results.is_empty() {
        println!("  (no results passed the threshold)");
    }
    for (i, r) in results.iter().enumerate() {
        println!(
            "  #{}: \"{}\" (score: {:.4})",
            i + 1,
            r.content,
            r.score.unwrap_or(0.0)
        );
    }
    println!();

    // ── 8. Search across different users ────────────────────────────
    // Memories are scoped by user_id — each user only sees their own memories.
    println!("[8] User isolation: searching \"morning\" for each user...\n");

    let results_alice = store
        .search("morning", SearchOptions::new("user_alice").limit(3))
        .expect("Search failed");
    let results_bob = store
        .search("morning", SearchOptions::new("user_bob").limit(3))
        .expect("Search failed");

    println!("  user_alice results ({}):", results_alice.len());
    for r in &results_alice {
        println!("    - \"{}\"", r.content);
    }
    println!("  user_bob results ({}):", results_bob.len());
    for r in &results_bob {
        println!("    - \"{}\"", r.content);
    }
    println!();

    // ── 9. List with filter (no vector search) ──────────────────────
    println!("[9] Listing memories with filter (no vector search)...\n");

    let list_opts =
        ListOptions::new("user_alice").filter(FilterExpression::contains("categories", "learning"));
    let results = store.list_traces(list_opts).expect("List failed");

    println!("  Memories in 'learning' category:");
    for (i, r) in results.iter().enumerate() {
        println!("  #{}: \"{}\"", i + 1, r.content);
    }

    println!("\n=== Done! ===");
}

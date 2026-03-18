//! Basic example: demonstrates core memory operations without LLM.
//!
//! Run with:
//!   cargo run -p memme-core --example basic

use std::sync::Arc;

use memme_core::config::MemoryConfig;
use memme_core::memory::MemoryStore;
use memme_core::types::*;
use memme_embeddings::mock::MockEmbedder;
use serde_json::json;

fn main() {
    println!("=== MemMe Basic Example ===\n");

    // ── 1. Create an in-memory store with MockEmbedder ──────────────────
    let config = MemoryConfig {
        db_path: ":memory:".into(),
        collection_name: "demo".into(),
        embedding_dims: 384,
        dedup_threshold: 0.15,
        default_limit: 10,
        ..Default::default()
    };
    let embedder = Arc::new(MockEmbedder::new(384));
    let store = MemoryStore::new(config, embedder).expect("Failed to create MemoryStore");
    println!("[1] MemoryStore created (in-memory, 384-dim embeddings)\n");

    // ── 2. Add 5 memories about a user's preferences ────────────────────
    println!("[2] Adding 5 memories...\n");

    let memories = vec![
        (
            "User prefers dark mode in all applications",
            json!({"category": "preferences"}),
        ),
        (
            "User's favorite programming language is Rust",
            json!({"category": "tech"}),
        ),
        (
            "User drinks 3 cups of coffee every morning",
            json!({"category": "habits"}),
        ),
        (
            "User works remotely from Tokyo",
            json!({"category": "location"}),
        ),
        (
            "User is learning Japanese on weekends",
            json!({"category": "learning"}),
        ),
    ];

    let mut ids = Vec::new();
    for (content, metadata) in &memories {
        let opts = AddOptions::new("user_alice").metadata(metadata.clone());
        let result = store.add(content, opts).expect("Failed to add memory");
        println!("  Added: \"{}\"", result.content);
        println!("    id: {}", result.id);
        println!("    metadata: {}", metadata);
        ids.push(result.id);
    }
    println!();

    // ── 3. Search for related memories ──────────────────────────────────
    println!("[3] Searching for \"coffee and morning routine\"...\n");

    let search_opts = SearchOptions::new("user_alice").limit(3);
    let results = store
        .search("coffee and morning routine", search_opts)
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

    // ── 4. Update a memory ──────────────────────────────────────────────
    let update_id = &ids[2]; // the coffee memory
    println!("[4] Updating memory (coffee habit)...\n");
    println!("  Before: \"{}\"", memories[2].0);

    let updated = store
        .update_trace(
            update_id,
            "User switched from coffee to matcha in the morning",
            None,
        )
        .expect("Update failed");
    println!("  After:  \"{}\"", updated.content);
    println!();

    // ── 5. Delete a memory ──────────────────────────────────────────────
    let delete_id = &ids[3]; // the Tokyo memory
    println!("[5] Deleting memory: \"{}\"", memories[3].0);

    store.delete_trace(delete_id).expect("Delete failed");
    println!("  Deleted successfully.\n");

    // ── 6. List remaining memories ──────────────────────────────────────
    println!("[6] Listing all remaining memories for user_alice:\n");

    let list_opts = ListOptions::new("user_alice");
    let remaining = store.list_traces(list_opts).expect("List failed");

    for (i, r) in remaining.iter().enumerate() {
        println!("  #{}: \"{}\"", i + 1, r.content);
    }
    println!("\n  Total: {} memories", remaining.len());

    println!("\n=== Done! ===");
}

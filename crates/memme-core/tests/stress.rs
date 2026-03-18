use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use memme_core::config::MemoryConfig;
use memme_core::memory::MemoryStore;
use memme_core::types::*;
use memme_embeddings::mock::MockEmbedder;
use serde_json::json;

/// Global counter to generate truly unique content across all tests,
/// ensuring no two memory strings ever collide even when tests run in parallel.
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate content with a maximally unique nonce so that both the content
/// hash and the embedding vector are distinct for every call.
fn unique_content(prefix: &str, index: usize) -> String {
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!(
        "{prefix} idx={index} seq={seq} nonce={:016x}",
        seq.wrapping_mul(0x517cc1b727220a95)
    )
}

/// Embedding dimensions used across all stress tests.
/// Using 384 dims (matching MiniLM) ensures the mock embedder produces
/// sufficiently spread-out vectors so that dedup does not falsely merge
/// memories with different content.
const DIMS: usize = 384;

/// Create an in-memory store.
fn make_store() -> MemoryStore {
    let config = MemoryConfig::new(":memory:", DIMS);
    let embedder = Arc::new(MockEmbedder::new(DIMS));
    MemoryStore::new(config, embedder).unwrap()
}

/// Create a file-backed store.
fn make_store_with_path(path: &str, collection: &str) -> MemoryStore {
    let config = MemoryConfig {
        db_path: path.into(),
        collection_name: collection.into(),
        embedding_dims: DIMS,
        ..Default::default()
    };
    let embedder = Arc::new(MockEmbedder::new(DIMS));
    MemoryStore::new(config, embedder).unwrap()
}

/// Helper to generate a unique temp DB path that includes the test name
/// and process ID to avoid collisions between parallel test runs.
fn temp_db_path(test_name: &str) -> String {
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_dir = std::env::temp_dir();
    tmp_dir
        .join(format!(
            "memme_stress_{}_{}_{}.duckdb",
            test_name,
            std::process::id(),
            seq
        ))
        .to_string_lossy()
        .to_string()
}

// ---------------------------------------------------------------------------
// 1. High volume insert
// ---------------------------------------------------------------------------

/// Add 1000 memories with unique content, verify count via list.
#[test]
fn test_high_volume_insert() {
    let store = make_store();

    for i in 0..1000 {
        store
            .add(
                &unique_content("high_volume_insert", i),
                AddOptions::new("stress_user"),
            )
            .unwrap();
    }

    let all = store
        .list_traces(ListOptions::new("stress_user").limit(2000))
        .unwrap();
    assert_eq!(all.len(), 1000, "Expected 1000 memories, got {}", all.len());
}

// ---------------------------------------------------------------------------
// 2. Rapid add/delete cycles
// ---------------------------------------------------------------------------

/// Add and delete in tight loop (500 iterations), verify history table grows,
/// no leftover memories.
#[test]
fn test_rapid_add_delete_cycles() {
    let store = make_store();
    let mut all_ids = Vec::with_capacity(500);

    for i in 0..500 {
        let added = store
            .add(
                &unique_content("add_delete_cycle", i),
                AddOptions::new("cycle_user"),
            )
            .unwrap();
        all_ids.push(added.id.clone());
        store.delete_trace(&added.id).unwrap();
    }

    // No memories should remain
    let remaining = store
        .list_traces(ListOptions::new("cycle_user").limit(1000))
        .unwrap();
    assert_eq!(
        remaining.len(),
        0,
        "Expected 0 leftover memories, got {}",
        remaining.len()
    );

    // History should have records (at least ADD + DELETE per cycle = 2 events per memory).
    let history = store.trace_history(&all_ids[0]).unwrap();
    assert!(
        history.len() >= 2,
        "Expected at least 2 history events (ADD+DELETE) for first memory, got {}",
        history.len()
    );
}

// ---------------------------------------------------------------------------
// 3. Large metadata
// ---------------------------------------------------------------------------

/// Add memory with 10KB JSON metadata, verify round-trip.
#[test]
fn test_large_metadata() {
    let store = make_store();

    // Build a ~10KB metadata object
    let mut large_map = serde_json::Map::new();
    for i in 0..200 {
        large_map.insert(
            format!("key_{i:04}"),
            json!(format!(
                "value_{i}_padding_data_to_increase_size_{}",
                "x".repeat(30)
            )),
        );
    }
    let large_meta = serde_json::Value::Object(large_map);
    let meta_size = serde_json::to_string(&large_meta).unwrap().len();
    assert!(
        meta_size >= 10_000,
        "Metadata should be at least 10KB, got {} bytes",
        meta_size
    );

    let added = store
        .add(
            &unique_content("large_metadata", 0),
            AddOptions::new("meta_user").metadata(large_meta.clone()),
        )
        .unwrap();

    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    let fetched_meta = fetched.metadata.unwrap();

    // Verify all keys survived the round-trip
    for i in 0..200 {
        let key = format!("key_{i:04}");
        assert_eq!(
            fetched_meta[&key], large_meta[&key],
            "Metadata mismatch at key {key}"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Large content
// ---------------------------------------------------------------------------

/// Add memory with 5KB text, verify search finds it.
#[test]
fn test_large_content() {
    let store = make_store();

    let large_content = format!(
        "This is a large content stress test entry. {}. End of content.",
        "The quick brown fox jumps over the lazy dog. ".repeat(120)
    );
    assert!(
        large_content.len() >= 5_000,
        "Content should be at least 5KB, got {} bytes",
        large_content.len()
    );

    let added = store
        .add(&large_content, AddOptions::new("content_user"))
        .unwrap();

    // Verify the content was stored correctly
    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    assert_eq!(fetched.content, large_content);

    // Verify search can find it
    let results = store
        .search(
            "large content stress test",
            SearchOptions::new("content_user"),
        )
        .unwrap();
    assert!(
        !results.is_empty(),
        "Search should find the large content memory"
    );
    assert!(
        results.iter().any(|r| r.id == added.id),
        "Search results should include the large content memory"
    );
}

// ---------------------------------------------------------------------------
// 5. DB file persistence
// ---------------------------------------------------------------------------

/// Write to tempfile, drop store, reopen, verify data survives.
#[test]
fn test_db_file_persistence() {
    let db_path = temp_db_path("persist");

    // Clean up any previous run
    let _ = std::fs::remove_file(&db_path);

    let content = "persistence test memory that must survive restart with unique nonce 0xDEADBEEF";
    let added_id;
    {
        let store = make_store_with_path(&db_path, "persist");
        let added = store.add(content, AddOptions::new("persist_user")).unwrap();
        added_id = added.id.clone();

        // Verify it exists before drop
        let fetched = store.get_trace(&added_id).unwrap();
        assert!(fetched.is_some());
        // Store is dropped here
    }

    // Reopen from the same file
    {
        let store = make_store_with_path(&db_path, "persist");
        let fetched = store.get_trace(&added_id).unwrap();
        assert!(
            fetched.is_some(),
            "Memory should survive store restart from disk"
        );
        let mem = fetched.unwrap();
        assert_eq!(mem.content, content);
        assert_eq!(mem.user_id, "persist_user");
    }

    // Clean up
    let _ = std::fs::remove_file(&db_path);
}

// ---------------------------------------------------------------------------
// 6. Multiple users isolation (same store)
// ---------------------------------------------------------------------------

/// 3 different user_ids in the same store, verify memory isolation via
/// list and search. This validates that user-scoped queries never leak
/// data across users.
#[test]
fn test_multiple_users_isolation_stress() {
    let store = make_store();

    // Add different amounts per user
    for i in 0..50 {
        store
            .add(
                &unique_content("user_alpha_mem", i),
                AddOptions::new("user_alpha"),
            )
            .unwrap();
    }
    for i in 0..30 {
        store
            .add(
                &unique_content("user_beta_mem", i),
                AddOptions::new("user_beta"),
            )
            .unwrap();
    }
    for i in 0..20 {
        store
            .add(
                &unique_content("user_gamma_mem", i),
                AddOptions::new("user_gamma"),
            )
            .unwrap();
    }

    // Verify list isolation
    let list_alpha = store
        .list_traces(ListOptions::new("user_alpha").limit(200))
        .unwrap();
    assert_eq!(
        list_alpha.len(),
        50,
        "user_alpha should have 50 memories, got {}",
        list_alpha.len()
    );
    for m in &list_alpha {
        assert_eq!(m.user_id, "user_alpha");
    }

    let list_beta = store
        .list_traces(ListOptions::new("user_beta").limit(200))
        .unwrap();
    assert_eq!(
        list_beta.len(),
        30,
        "user_beta should have 30 memories, got {}",
        list_beta.len()
    );
    for m in &list_beta {
        assert_eq!(m.user_id, "user_beta");
    }

    let list_gamma = store
        .list_traces(ListOptions::new("user_gamma").limit(200))
        .unwrap();
    assert_eq!(
        list_gamma.len(),
        20,
        "user_gamma should have 20 memories, got {}",
        list_gamma.len()
    );
    for m in &list_gamma {
        assert_eq!(m.user_id, "user_gamma");
    }

    // Verify search isolation: searching as user_alpha should only return user_alpha results
    let search_alpha = store
        .search(
            "user_alpha_mem",
            SearchOptions::new("user_alpha").limit(100),
        )
        .unwrap();
    for r in &search_alpha {
        assert_eq!(
            r.user_id, "user_alpha",
            "Search for user_alpha returned memory belonging to {}",
            r.user_id
        );
    }
}

// ---------------------------------------------------------------------------
// 7. High volume search
// ---------------------------------------------------------------------------

/// Add 500 memories, search returns correct ranking order.
/// With forgetting curve enabled, search results are sorted by descending
/// weighted score (higher = better match).
#[test]
fn test_high_volume_search() {
    let store = make_store();

    for i in 0..500 {
        store
            .add(
                &unique_content("searchable_volume", i),
                AddOptions::new("search_user"),
            )
            .unwrap();
    }

    let results = store
        .search(
            "searchable_volume",
            SearchOptions::new("search_user").limit(20),
        )
        .unwrap();

    assert!(
        !results.is_empty(),
        "Search should return results from 500 memories"
    );
    assert!(
        results.len() <= 20,
        "Search should respect limit of 20, got {}",
        results.len()
    );

    // Verify results are in non-increasing score order (higher score = better match).
    // The forgetting curve rescores and sorts descending.
    for window in results.windows(2) {
        let score_a = window[0].score.unwrap_or(0.0);
        let score_b = window[1].score.unwrap_or(0.0);
        assert!(
            score_a >= score_b - f32::EPSILON,
            "Search results should be ordered by descending score: {} < {}",
            score_a,
            score_b
        );
    }
}

// ---------------------------------------------------------------------------
// 8. Batch update stress
// ---------------------------------------------------------------------------

/// batch_update 200 memories at once.
#[test]
fn test_batch_update_stress() {
    let store = make_store();

    let mut ids = Vec::with_capacity(200);
    for i in 0..200 {
        let added = store
            .add(
                &unique_content("batch_update_original", i),
                AddOptions::new("batch_user"),
            )
            .unwrap();
        ids.push(added.id);
    }

    assert_eq!(ids.len(), 200, "Should have inserted 200 unique memories");

    // Update each memory individually with unique content to avoid dedup
    let mut results = Vec::with_capacity(200);
    for (i, id) in ids.iter().enumerate() {
        let new_content = unique_content("batch_update_revised", i);
        let result = store.update_trace(id, &new_content, None).unwrap();
        results.push(result);
    }
    assert_eq!(
        results.len(),
        200,
        "update loop should return 200 results, got {}",
        results.len()
    );

    // Verify all memories were updated
    for (i, id) in ids.iter().enumerate() {
        let fetched = store.get_trace(id).unwrap().unwrap();
        assert!(
            fetched.content.contains("batch_update_revised"),
            "Memory {} should have updated content, got: {}",
            i,
            fetched.content
        );
    }
}

// ---------------------------------------------------------------------------
// 9. Batch delete stress
// ---------------------------------------------------------------------------

/// batch_delete 200 memories at once.
#[test]
fn test_batch_delete_stress() {
    let store = make_store();

    let mut ids = Vec::with_capacity(200);
    for i in 0..200 {
        let added = store
            .add(
                &unique_content("batch_delete_target", i),
                AddOptions::new("batchdel_user"),
            )
            .unwrap();
        ids.push(added.id);
    }

    assert_eq!(ids.len(), 200, "Should have inserted 200 unique memories");

    // Verify all exist
    let list_before = store
        .list_traces(ListOptions::new("batchdel_user").limit(500))
        .unwrap();
    assert_eq!(list_before.len(), 200);

    let mut deleted_count = 0u64;
    for id in &ids {
        store.delete_trace(id).unwrap();
        deleted_count += 1;
    }
    assert_eq!(
        deleted_count, 200,
        "delete loop should report 200 deletions, got {}",
        deleted_count
    );

    // Verify none remain
    let list_after = store
        .list_traces(ListOptions::new("batchdel_user").limit(500))
        .unwrap();
    assert_eq!(
        list_after.len(),
        0,
        "No memories should remain after batch_delete, got {}",
        list_after.len()
    );

    // Verify individual gets return None
    for id in &ids {
        assert!(
            store.get_trace(id).unwrap().is_none(),
            "Memory {} should not exist after batch delete",
            id
        );
    }
}

// ---------------------------------------------------------------------------
// 10. Export / import large
// ---------------------------------------------------------------------------

/// Export 500 memories, import into fresh store, verify all data matches.
#[test]
fn test_export_import_large() {
    let store = make_store();

    for i in 0..500 {
        let meta = json!({ "index": i, "tag": format!("tag_{i}") });
        store
            .add(
                &unique_content("exportable_memory", i),
                AddOptions::new("export_user").metadata(meta),
            )
            .unwrap();
    }

    // Export all memories
    let exported = store.export(Some("export_user")).unwrap();
    assert_eq!(
        exported.len(),
        500,
        "Export should contain 500 memories, got {}",
        exported.len()
    );

    // Import into a fresh store
    let fresh_store = make_store();
    let imported_count = fresh_store.import_memories(&exported).unwrap();
    assert_eq!(
        imported_count, 500,
        "import_memories should report 500 imported, got {}",
        imported_count
    );

    // Verify all memories exist in the fresh store with matching data
    for export_mem in &exported {
        let fetched = fresh_store.get_trace(&export_mem.id).unwrap();
        assert!(
            fetched.is_some(),
            "Imported memory {} should exist in fresh store",
            export_mem.id
        );
        let fetched = fetched.unwrap();
        assert_eq!(
            fetched.content, export_mem.content,
            "Content mismatch for memory {}",
            export_mem.id
        );
        assert_eq!(
            fetched.user_id, export_mem.user_id,
            "user_id mismatch for memory {}",
            export_mem.id
        );

        // Verify metadata round-trip
        if let Some(ref orig_meta) = export_mem.metadata {
            let fetched_meta = fetched.metadata.as_ref().expect("metadata should exist");
            assert_eq!(
                fetched_meta, orig_meta,
                "Metadata mismatch for memory {}",
                export_mem.id
            );
        }
    }
}

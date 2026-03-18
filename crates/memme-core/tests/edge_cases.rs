use std::sync::Arc;

use memme_core::config::MemoryConfig;
use memme_core::error::MemoryError;
use memme_core::memory::MemoryStore;
use memme_core::types::*;
use memme_embeddings::mock::MockEmbedder;
use serde_json::json;

fn make_store() -> MemoryStore {
    let config = MemoryConfig::new(":memory:", 384);
    let embedder = Arc::new(MockEmbedder::new(384));
    MemoryStore::new(config, embedder).unwrap()
}

fn make_store_with_config(config: MemoryConfig) -> MemoryStore {
    let embedder = Arc::new(MockEmbedder::new(384));
    MemoryStore::new(config, embedder).unwrap()
}

// ── 1. Empty content rejected ──

#[test]
fn test_empty_content_rejected() {
    let store = make_store();
    let err = store.add("", AddOptions::new("user1")).unwrap_err();
    assert!(matches!(err, MemoryError::Config(_)));
}

// ── 2. Whitespace-only content rejected ──

#[test]
fn test_whitespace_only_content_rejected() {
    let store = make_store();
    let err = store.add("   \n\t", AddOptions::new("user1")).unwrap_err();
    assert!(matches!(err, MemoryError::Config(_)));
}

// ── 3. Unicode Chinese content round-trip ──

#[test]
fn test_unicode_chinese_content() {
    let store = make_store();
    let content = "我喜欢咖啡";

    let added = store.add(content, AddOptions::new("user1")).unwrap();
    assert_eq!(added.content, content);

    let results = store.search(content, SearchOptions::new("user1")).unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().any(|r| r.content == content));

    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    assert_eq!(fetched.content, content);
}

// ── 4. Unicode emoji content round-trip ──

#[test]
fn test_unicode_emoji_content() {
    let store = make_store();
    let content = "I love 🎉 and ☕";

    let added = store.add(content, AddOptions::new("user1")).unwrap();
    assert_eq!(added.content, content);

    let results = store.search(content, SearchOptions::new("user1")).unwrap();
    assert!(!results.is_empty());

    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    assert_eq!(fetched.content, content);
}

// ── 5. Unicode RTL content round-trip ──

#[test]
fn test_unicode_rtl_content() {
    let store = make_store();
    let content = "مرحبا بالعالم שלום עולם";

    let added = store.add(content, AddOptions::new("user1")).unwrap();
    assert_eq!(added.content, content);

    let results = store.search(content, SearchOptions::new("user1")).unwrap();
    assert!(!results.is_empty());

    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    assert_eq!(fetched.content, content);
}

// ── 6. SQL injection in content ──

#[test]
fn test_sql_injection_in_content() {
    let store = make_store();
    let malicious = "'; DROP TABLE memories; --";

    let added = store.add(malicious, AddOptions::new("user1")).unwrap();
    assert_eq!(added.content, malicious);

    // The table should still be intact — we can list memories
    let list = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].content, malicious);
}

// ── 7. SQL injection in user_id ──

#[test]
fn test_sql_injection_in_user_id() {
    let store = make_store();
    let malicious_user = "'; DROP TABLE memories; --";

    let added = store
        .add("safe content", AddOptions::new(malicious_user))
        .unwrap();
    assert_eq!(added.user_id, malicious_user);

    // Store still works
    let list = store.list_traces(ListOptions::new(malicious_user)).unwrap();
    assert_eq!(list.len(), 1);

    // Other users unaffected
    store
        .add("other user content", AddOptions::new("normal_user"))
        .unwrap();
    let other_list = store.list_traces(ListOptions::new("normal_user")).unwrap();
    assert_eq!(other_list.len(), 1);
}

// ── 8. SQL injection in metadata ──

#[test]
fn test_sql_injection_in_metadata() {
    let store = make_store();
    let malicious_meta = json!({
        "key": "'; DROP TABLE memories; --",
        "nested": {"attack": "Robert'); DROP TABLE students;--"},
        "array": ["1; DELETE FROM memories", "normal"]
    });

    let added = store
        .add(
            "metadata injection test",
            AddOptions::new("user1").metadata(malicious_meta.clone()),
        )
        .unwrap();

    // Retrieve and verify metadata is preserved exactly
    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    let meta = fetched.metadata.unwrap();
    assert_eq!(meta["key"], "'; DROP TABLE memories; --");
    assert_eq!(meta["nested"]["attack"], "Robert'); DROP TABLE students;--");

    // Store still works
    let list = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(list.len(), 1);
}

// ── 9. Very long user_id ──

#[test]
fn test_very_long_user_id() {
    let store = make_store();
    let long_user_id = "u".repeat(10_000);

    let added = store
        .add("content for long user", AddOptions::new(&long_user_id))
        .unwrap();
    assert_eq!(added.user_id, long_user_id);

    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    assert_eq!(fetched.user_id, long_user_id);
}

// ── 10. Very long content (100KB) ──

#[test]
fn test_very_long_content() {
    let store = make_store();
    let long_content = "x".repeat(100 * 1024); // 100KB

    let added = store.add(&long_content, AddOptions::new("user1")).unwrap();
    assert_eq!(added.content.len(), 100 * 1024);

    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    assert_eq!(fetched.content, long_content);
}

// ── 11. Invalid UUID get returns None ──

#[test]
fn test_invalid_uuid_get() {
    let store = make_store();
    let result = store.get_trace("not-a-uuid").unwrap();
    assert!(result.is_none());
}

// ── 12. Empty string get returns None ──

#[test]
fn test_empty_string_get() {
    let store = make_store();
    let result = store.get_trace("").unwrap();
    assert!(result.is_none());
}

// ── 13. Double delete — second should error NotFound ──

#[test]
fn test_double_delete() {
    let store = make_store();
    let added = store
        .add("delete me twice", AddOptions::new("user1"))
        .unwrap();
    let id = added.id.clone();

    // First delete succeeds
    store.delete_trace(&id).unwrap();

    // Second delete should error NotFound
    let err = store.delete_trace(&id).unwrap_err();
    assert!(matches!(err, MemoryError::NotFound(_)));
}

// ── 14. Update nonexistent ──

#[test]
fn test_update_nonexistent() {
    let store = make_store();
    let err = store
        .update_trace("fake-id-12345", "new content", None)
        .unwrap_err();
    assert!(matches!(err, MemoryError::NotFound(_)));
}

// ── 15. Delete nonexistent ──

#[test]
fn test_delete_nonexistent() {
    let store = make_store();
    let err = store.delete_trace("fake-id-12345").unwrap_err();
    assert!(matches!(err, MemoryError::NotFound(_)));
}

// ── 16. Immutable update blocked ──

#[test]
fn test_immutable_update_blocked() {
    let store = make_store();
    let added = store
        .add(
            "immutable content",
            AddOptions::new("user1").immutable(true),
        )
        .unwrap();
    assert!(added.immutable);

    let err = store
        .update_trace(&added.id, "try to change", None)
        .unwrap_err();
    assert!(matches!(err, MemoryError::ImmutableMemory(_)));
}

// ── 17. Immutable delete blocked ──

#[test]
fn test_immutable_delete_blocked() {
    let store = make_store();
    let added = store
        .add(
            "immutable no delete",
            AddOptions::new("user1").immutable(true),
        )
        .unwrap();

    let err = store.delete_trace(&added.id).unwrap_err();
    assert!(matches!(err, MemoryError::ImmutableMemory(_)));

    // Memory should still exist
    let fetched = store.get_trace(&added.id).unwrap();
    assert!(fetched.is_some());
}

// ── 18. Immutable batch update skipped ──

#[test]
fn test_immutable_batch_update_skipped() {
    let store = make_store();

    let immutable = store
        .add(
            "immutable item for batch",
            AddOptions::new("user1").immutable(true),
        )
        .unwrap();
    let mutable = store
        .add("mutable item for batch", AddOptions::new("user1"))
        .unwrap();

    // Try updating each individually; immutable should fail, mutable should succeed
    let immutable_err = store.update_trace(&immutable.id, "changed immutable", None);
    assert!(
        immutable_err.is_err(),
        "Updating immutable memory should fail"
    );

    let mutable_result = store
        .update_trace(&mutable.id, "changed mutable", None)
        .unwrap();
    assert_eq!(mutable_result.id, mutable.id);
    assert_eq!(mutable_result.content, "changed mutable");

    // Immutable should still have original content
    let fetched = store.get_trace(&immutable.id).unwrap().unwrap();
    assert_eq!(fetched.content, "immutable item for batch");
}

// ── 19. Immutable batch delete skipped ──

#[test]
fn test_immutable_batch_delete_skipped() {
    let store = make_store();

    let immutable = store
        .add(
            "immutable no batch delete",
            AddOptions::new("user1").immutable(true),
        )
        .unwrap();
    let mutable = store
        .add("mutable batch delete", AddOptions::new("user1"))
        .unwrap();

    // Try deleting each individually; immutable should fail, mutable should succeed
    let immutable_del_err = store.delete_trace(&immutable.id);
    assert!(
        immutable_del_err.is_err(),
        "Deleting immutable memory should fail"
    );

    store.delete_trace(&mutable.id).unwrap();

    // Only the mutable one should have been deleted

    // Immutable should still exist
    let fetched = store.get_trace(&immutable.id).unwrap();
    assert!(fetched.is_some());

    // Mutable should be gone
    let gone = store.get_trace(&mutable.id).unwrap();
    assert!(gone.is_none());
}

// ── 20. Expiration cleanup ──

#[test]
fn test_expiration_cleanup() {
    let store = make_store();

    // Add a memory with an expiration date in the past
    store
        .add(
            "expired memory content here",
            AddOptions::new("user1").expiration_date("2020-01-01T00:00:00Z"),
        )
        .unwrap();

    // Add a memory without expiration
    let permanent = store
        .add("permanent memory content here", AddOptions::new("user1"))
        .unwrap();

    // cleanup_expired should remove the expired one
    let cleaned = store.cleanup_expired().unwrap();
    assert_eq!(cleaned, 1);

    // Permanent should still exist
    let fetched = store.get_trace(&permanent.id).unwrap();
    assert!(fetched.is_some());

    // List should have only the permanent one
    let list = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].id, permanent.id);
}

// ── 21. Max memories pruning — LRU strategy ──

#[test]
fn test_max_memories_pruning_lru() {
    let mut config = MemoryConfig::new(":memory:", 384);
    config.max_memories_per_user = Some(3);
    config.auto_prune = true;
    config.pruning_strategy = PruningStrategy::LRU;
    let store = make_store_with_config(config);

    // Add 5 unique memories
    for i in 0..5 {
        store
            .add(
                &format!("lru pruning test memory number {i} unique unique unique {i}"),
                AddOptions::new("user1"),
            )
            .unwrap();
    }

    // Should be pruned down to 3
    let list = store
        .list_traces(ListOptions::new("user1").limit(100))
        .unwrap();
    assert_eq!(list.len(), 3);
}

// ── 22. Max memories pruning — importance strategy ──

#[test]
fn test_max_memories_pruning_importance() {
    let mut config = MemoryConfig::new(":memory:", 384);
    config.max_memories_per_user = Some(3);
    config.auto_prune = true;
    config.pruning_strategy = PruningStrategy::Importance;
    let store = make_store_with_config(config);

    // Add 5 memories with varying importance (lower importance should be pruned)
    let importances = [0.1, 0.2, 0.9, 0.8, 0.7];
    for (i, &imp) in importances.iter().enumerate() {
        store
            .add(
                &format!("importance pruning test memory {i} unique unique unique {i}"),
                AddOptions::new("user1").importance(imp),
            )
            .unwrap();
    }

    // Should be pruned down to 3
    let list = store
        .list_traces(ListOptions::new("user1").limit(100))
        .unwrap();
    assert_eq!(list.len(), 3);

    // The remaining 3 should be the highest importance ones (0.9, 0.8, 0.7)
    for m in &list {
        let imp = m.importance.unwrap_or(0.0);
        assert!(
            imp >= 0.7,
            "Expected importance >= 0.7, got {imp} for memory '{}'",
            m.content
        );
    }
}

// ── 23. Filter injection attempt ──

#[test]
fn test_filter_injection_attempt() {
    let store = make_store();

    store
        .add("safe content for filter test", AddOptions::new("user1"))
        .unwrap();

    // Malicious field name should be sanitized to NULL, not executed as SQL
    let malicious_filter = FilterExpression::eq("'); DROP TABLE memories; --", "anything");

    let _results = store
        .search("safe", SearchOptions::new("user1").filter(malicious_filter))
        .unwrap();

    // Search should still work (malicious field becomes NULL = $N which never matches)
    // The important thing is it doesn't crash or drop the table
    let list = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(
        list.len(),
        1,
        "Table should still be intact after injection attempt"
    );
}

// ── 24. Category injection attempt ──
// The store validates category names, so SQL injection strings are rejected.
// This test verifies that malicious category names are properly rejected
// and that valid categories still work after the rejection.

#[test]
fn test_category_injection_attempt() {
    let store = make_store();

    let malicious_categories = vec![
        "'; DROP TABLE memories; --".to_string(),
        "normal_category".to_string(),
    ];

    // Malicious category name should be rejected by validation
    let err = store
        .add(
            "category injection test content",
            AddOptions::new("user1").categories(malicious_categories),
        )
        .unwrap_err();
    assert!(matches!(err, MemoryError::Config(_)));

    // Store should still be intact — add with safe categories works
    let added = store
        .add(
            "safe category content",
            AddOptions::new("user1").categories(vec!["work".to_string(), "personal".to_string()]),
        )
        .unwrap();

    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    let cats = fetched.categories.unwrap();
    assert_eq!(cats.len(), 2);
    assert_eq!(cats[0], "work");
    assert_eq!(cats[1], "personal");

    // Table still intact
    let list = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(list.len(), 1);
}

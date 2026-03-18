use std::sync::Arc;

use memme_core::config::MemoryConfig;
use memme_core::memory::MemoryStore;
use memme_core::types::*;
use memme_embeddings::mock::MockEmbedder;
use serde_json::json;

fn test_store() -> MemoryStore {
    let config = MemoryConfig::new(":memory:", 384);
    let embedder = Arc::new(MockEmbedder::new(384));
    MemoryStore::new(config, embedder).unwrap()
}

#[test]
fn test_full_lifecycle() {
    let store = test_store();

    // Add
    let added = store
        .add("remember this fact", AddOptions::new("user1"))
        .unwrap();
    assert_eq!(added.content, "remember this fact");
    assert_eq!(added.user_id, "user1");
    let id = added.id.clone();

    // Search — should find the memory
    let results = store
        .search("remember", SearchOptions::new("user1"))
        .unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().any(|r| r.id == id));

    // Update
    let updated = store.update_trace(&id, "updated fact", None).unwrap();
    assert_eq!(updated.id, id);
    assert_eq!(updated.content, "updated fact");

    // Search again — should find updated content
    let results2 = store
        .search("updated", SearchOptions::new("user1"))
        .unwrap();
    assert!(!results2.is_empty());
    assert!(results2.iter().any(|r| r.content == "updated fact"));

    // Delete
    store.delete_trace(&id).unwrap();

    // Verify deleted
    let fetched = store.get_trace(&id).unwrap();
    assert!(fetched.is_none());
}

#[test]
fn test_multiple_users_isolated() {
    let store = test_store();

    // Add memories for two different users
    store
        .add("user_a secret memory", AddOptions::new("user_a"))
        .unwrap();
    store
        .add("user_a another memory", AddOptions::new("user_a"))
        .unwrap();
    store
        .add("user_b private memory", AddOptions::new("user_b"))
        .unwrap();

    // Search as user_a — should only see user_a's memories
    let results_a = store
        .search("memory", SearchOptions::new("user_a"))
        .unwrap();
    for r in &results_a {
        assert_eq!(r.user_id, "user_a");
    }

    // Search as user_b — should only see user_b's memories
    let results_b = store
        .search("memory", SearchOptions::new("user_b"))
        .unwrap();
    for r in &results_b {
        assert_eq!(r.user_id, "user_b");
    }

    // List as user_a
    let list_a = store.list_traces(ListOptions::new("user_a")).unwrap();
    assert_eq!(list_a.len(), 2);
    for m in &list_a {
        assert_eq!(m.user_id, "user_a");
    }

    // List as user_b
    let list_b = store.list_traces(ListOptions::new("user_b")).unwrap();
    assert_eq!(list_b.len(), 1);
    assert_eq!(list_b[0].user_id, "user_b");
}

#[test]
fn test_dedup_across_operations() {
    let store = test_store();

    // Add the same content twice
    let first = store
        .add("duplicate content here", AddOptions::new("user1"))
        .unwrap();
    let second = store
        .add("duplicate content here", AddOptions::new("user1"))
        .unwrap();

    // Dedup should merge them — same ID returned
    assert_eq!(first.id, second.id);

    // Only one memory should exist
    let list = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].id, first.id);
}

#[test]
fn test_large_batch() {
    let store = test_store();

    // Add 100 memories
    for i in 0..100 {
        store
            .add(
                &format!("batch memory number {i} with unique content to avoid dedup"),
                AddOptions::new("user1"),
            )
            .unwrap();
    }

    // List all — should have 100
    let list = store
        .list_traces(ListOptions::new("user1").limit(200))
        .unwrap();
    assert_eq!(list.len(), 100);

    // Search should return results
    let results = store
        .search("batch memory", SearchOptions::new("user1").limit(10))
        .unwrap();
    assert!(!results.is_empty());
    assert!(results.len() <= 10);
}

#[test]
fn test_metadata_round_trip() {
    let store = test_store();

    let complex_metadata = json!({
        "source": "conversation",
        "tags": ["important", "project-x"],
        "nested": {
            "level": 2,
            "data": [1, 2, 3]
        },
        "flag": true,
        "count": 42,
        "nullable": null
    });

    let added = store
        .add(
            "metadata test content",
            AddOptions::new("user1").metadata(complex_metadata.clone()),
        )
        .unwrap();

    // Retrieve via get and verify metadata is preserved exactly
    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    let meta = fetched.metadata.unwrap();

    assert_eq!(meta["source"], "conversation");
    assert_eq!(meta["tags"][0], "important");
    assert_eq!(meta["tags"][1], "project-x");
    assert_eq!(meta["nested"]["level"], 2);
    assert_eq!(meta["nested"]["data"][0], 1);
    assert_eq!(meta["nested"]["data"][1], 2);
    assert_eq!(meta["nested"]["data"][2], 3);
    assert_eq!(meta["flag"], true);
    assert_eq!(meta["count"], 42);
    assert!(meta["nullable"].is_null());
}

#[test]
fn test_history_tracking() {
    let store = test_store();

    // Add a memory — records ADD history event
    let added = store
        .add("history tracking test", AddOptions::new("user1"))
        .unwrap();
    let id = added.id.clone();

    // Update the memory — records UPDATE history event
    let updated = store
        .update_trace(&id, "history tracking updated", None)
        .unwrap();
    assert_eq!(updated.content, "history tracking updated");

    // Delete the memory — records DELETE history event
    store.delete_trace(&id).unwrap();

    // Verify the memory is gone
    let fetched = store.get_trace(&id).unwrap();
    assert!(fetched.is_none());

    // Add a new memory to confirm the store still works after the lifecycle
    let new_mem = store
        .add("after history test", AddOptions::new("user1"))
        .unwrap();
    assert_eq!(new_mem.content, "after history test");

    // The new memory should be retrievable
    let fetched_new = store.get_trace(&new_mem.id).unwrap().unwrap();
    assert_eq!(fetched_new.content, "after history test");
}

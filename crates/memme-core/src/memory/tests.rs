use super::helpers::content_hash;
use super::*;
use memme_embeddings::mock::MockEmbedder;

fn make_store(dims: usize) -> MemoryStore {
    let config = MemoryConfig {
        db_path: ":memory:".into(),
        collection_name: "test".into(),
        embedding_dims: dims,
        dedup_threshold: 0.15,
        default_limit: 10,
        ..Default::default()
    };
    let embedder = Arc::new(MockEmbedder::new(dims));
    MemoryStore::new(config, embedder).unwrap()
}

#[test]
fn test_add_new_memory() {
    let store = make_store(384);
    let opts = AddOptions::new("user1");
    let result = store.add("hello world", opts).unwrap();
    assert!(!result.id.is_empty());
    assert_eq!(result.content, "hello world");
    assert_eq!(result.user_id, "user1");
}

#[test]
fn test_add_dedup_updates() {
    let store = make_store(384);
    let r1 = store.add("hello world", AddOptions::new("user1")).unwrap();
    let r2 = store.add("hello world", AddOptions::new("user1")).unwrap();
    // Second add should return the same id (dedup hit)
    assert_eq!(r1.id, r2.id);
}

#[test]
fn test_add_dedup_preserves_metadata() {
    let store = make_store(384);
    let opts1 = AddOptions::new("user1").metadata(serde_json::json!({"v": 1}));
    let r1 = store.add("hello world", opts1).unwrap();

    // Add same content with new metadata
    let opts2 = AddOptions::new("user1").metadata(serde_json::json!({"v": 2}));
    let r2 = store.add("hello world", opts2).unwrap();
    assert_eq!(r1.id, r2.id);
    // Metadata should be updated
    assert_eq!(r2.metadata.unwrap()["v"], 2);
}

#[test]
fn test_search_returns_relevant() {
    let store = make_store(384);
    store
        .add("the cat sat on the mat", AddOptions::new("user1"))
        .unwrap();
    store
        .add("quantum physics lecture notes", AddOptions::new("user1"))
        .unwrap();
    store
        .add("dogs playing in the park", AddOptions::new("user1"))
        .unwrap();

    let results = store
        .search("cat sitting", SearchOptions::new("user1"))
        .unwrap();
    assert!(!results.is_empty());
    // Results ordered by weighted score (descending) — higher is more relevant
    if results.len() >= 2 {
        assert!(results[0].score.unwrap() >= results[1].score.unwrap());
    }
}

#[test]
fn test_search_with_threshold() {
    let store = make_store(384);
    store
        .add("the cat sat on the mat", AddOptions::new("user1"))
        .unwrap();
    store
        .add("quantum physics lecture notes", AddOptions::new("user1"))
        .unwrap();

    // With forgetting curve enabled (default), scores are weighted (higher = better).
    // A higher threshold is stricter, so should return fewer results.
    let strict = store
        .search("test query", SearchOptions::new("user1").threshold(0.9))
        .unwrap();
    let relaxed = store
        .search("test query", SearchOptions::new("user1").threshold(0.01))
        .unwrap();
    assert!(
        strict.len() <= relaxed.len(),
        "strict(0.9)={} should be <= relaxed(0.01)={}",
        strict.len(),
        relaxed.len()
    );
}

#[test]
fn test_search_with_limit() {
    let store = make_store(384);
    for i in 0..5 {
        store
            .add(&format!("memory number {i}"), AddOptions::new("user1"))
            .unwrap();
    }
    let opts = SearchOptions::new("user1").limit(2);
    let results = store.search("memory", opts).unwrap();
    assert!(results.len() <= 2);
}

#[test]
fn test_get_existing() {
    let store = make_store(384);
    let added = store.add("test content", AddOptions::new("user1")).unwrap();
    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    assert_eq!(fetched.id, added.id);
    assert_eq!(fetched.content, "test content");
}

#[test]
fn test_get_nonexistent() {
    let store = make_store(384);
    let result = store.get_trace("nonexistent-id").unwrap();
    assert!(result.is_none());
}

#[test]
fn test_update_success() {
    let store = make_store(384);
    let added = store
        .add("original content", AddOptions::new("user1"))
        .unwrap();
    let updated = store
        .update_trace(&added.id, "updated content", None)
        .unwrap();
    assert_eq!(updated.id, added.id);
    assert_eq!(updated.content, "updated content");
}

#[test]
fn test_update_nonexistent() {
    let store = make_store(384);
    let result = store.update_trace("nonexistent-id", "new content", None);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), MemoryError::NotFound(_)));
}

#[test]
fn test_delete_success() {
    let store = make_store(384);
    let added = store
        .add("to be deleted", AddOptions::new("user1"))
        .unwrap();
    store.delete_trace(&added.id).unwrap();
    let fetched = store.get_trace(&added.id).unwrap();
    assert!(fetched.is_none());
}

#[test]
fn test_delete_nonexistent() {
    let store = make_store(384);
    let result = store.delete_trace("nonexistent-id");
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), MemoryError::NotFound(_)));
}

#[test]
fn test_list_by_user() {
    let store = make_store(384);
    store
        .add("user1 memory 1", AddOptions::new("user1"))
        .unwrap();
    store
        .add("user1 memory 2", AddOptions::new("user1"))
        .unwrap();
    store
        .add("user2 memory 1", AddOptions::new("user2"))
        .unwrap();

    let user1_list = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(user1_list.len(), 2);
    for m in &user1_list {
        assert_eq!(m.user_id, "user1");
    }

    let user2_list = store.list_traces(ListOptions::new("user2")).unwrap();
    assert_eq!(user2_list.len(), 1);
}

#[test]
fn test_list_by_agent() {
    let store = make_store(384);
    store
        .add("mem1", AddOptions::new("user1").agent_id("agent_a"))
        .unwrap();
    store
        .add("mem2", AddOptions::new("user1").agent_id("agent_b"))
        .unwrap();
    store
        .add("mem3", AddOptions::new("user1").agent_id("agent_a"))
        .unwrap();

    let opts = ListOptions::new("user1").agent_id("agent_a");
    let results = store.list_traces(opts).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_dimension_mismatch() {
    let config = MemoryConfig {
        db_path: ":memory:".into(),
        collection_name: "test".into(),
        embedding_dims: 384,
        dedup_threshold: 0.15,
        default_limit: 10,
        ..Default::default()
    };
    // MockEmbedder with 128 dims, config expects 384
    let embedder = Arc::new(MockEmbedder::new(128));
    let result = MemoryStore::new(config, embedder);
    assert!(matches!(result, Err(MemoryError::Config(_))));
}

#[test]
fn test_hash_deterministic() {
    let h1 = content_hash("hello world");
    let h2 = content_hash("hello world");
    assert_eq!(h1, h2);

    let h3 = content_hash("different content");
    assert_ne!(h1, h3);
}

#[test]
fn test_add_records_history() {
    let store = make_store(384);
    let added = store.add("history test", AddOptions::new("user1")).unwrap();

    // Check that a history event was recorded
    let conn = store.storage.read_conn();
    let mut stmt = conn
        .prepare("SELECT event, new_memory FROM history WHERE memory_id = $1")
        .unwrap();
    let rows: Vec<(String, String)> = stmt
        .query_map(duckdb::params![&added.id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0, "ADD");
    assert_eq!(rows[0].1, "history test");
}

#[test]
fn test_hybrid_search_basic() {
    let store = make_store(384);
    store
        .add(
            "the quick brown fox jumps over the lazy dog",
            AddOptions::new("user1"),
        )
        .unwrap();
    store
        .add(
            "quantum physics and relativity theory",
            AddOptions::new("user1"),
        )
        .unwrap();
    store
        .add("the lazy cat sleeps all day long", AddOptions::new("user1"))
        .unwrap();

    // Build FTS index
    store.rebuild_fts_index().unwrap();

    // Hybrid search via search() with keyword_search(true)
    let opts = SearchOptions::new("user1").keyword_search(true).limit(10);
    let results = store.search("lazy", opts).unwrap();
    assert!(!results.is_empty());
    // All results should have a score
    for r in &results {
        assert!(r.score.is_some());
    }
}

#[test]
fn test_keyword_search() {
    let store = make_store(384);
    store
        .add("the quick brown fox jumps", AddOptions::new("user1"))
        .unwrap();
    store
        .add("quantum physics lecture notes", AddOptions::new("user1"))
        .unwrap();
    store.rebuild_fts_index().unwrap();

    // Search with keyword_search enabled
    let opts = SearchOptions::new("user1").keyword_search(true).limit(10);
    let results = store.search("fox", opts).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_search_reranked() {
    let store = make_store(384);
    store
        .add("the cat sat on the mat", AddOptions::new("user1"))
        .unwrap();
    store
        .add("quantum physics lecture notes", AddOptions::new("user1"))
        .unwrap();
    store
        .add("dogs playing in the park", AddOptions::new("user1"))
        .unwrap();

    let reranker = crate::rerank::NoOpReranker;
    let opts = SearchOptions::new("user1").limit(2);
    let results = store
        .search_reranked("cat sitting", opts, &reranker, 3)
        .unwrap();
    // Should return at most 2 results (the limit)
    assert!(results.len() <= 2);
}

#[test]
fn test_rebuild_fts_index() {
    let store = make_store(384);
    // Rebuild on empty table should not error
    store.rebuild_fts_index().unwrap();

    // Add data and rebuild again
    store
        .add("some test content", AddOptions::new("user1"))
        .unwrap();
    store.rebuild_fts_index().unwrap();

    // Rebuild a second time (overwrite) should not error
    store.rebuild_fts_index().unwrap();
}

// ── Agent isolation tests ──

#[test]
fn test_search_by_agent() {
    let store = make_store(384);
    store
        .add(
            "agent_a memory about cats",
            AddOptions::new("user1").agent_id("agent_a"),
        )
        .unwrap();
    store
        .add(
            "agent_b memory about cats",
            AddOptions::new("user1").agent_id("agent_b"),
        )
        .unwrap();
    store
        .add(
            "agent_a memory about dogs",
            AddOptions::new("user1").agent_id("agent_a"),
        )
        .unwrap();

    // Search scoped to agent_a
    let opts = SearchOptions::new("user1").agent_id("agent_a").limit(10);
    let results = store.search("cats", opts).unwrap();
    assert_eq!(
        results.len(),
        2,
        "agent_a should see only its own 2 memories"
    );

    // Search scoped to agent_b
    let opts = SearchOptions::new("user1").agent_id("agent_b").limit(10);
    let results = store.search("cats", opts).unwrap();
    assert_eq!(results.len(), 1, "agent_b should see only its own 1 memory");
}

#[test]
fn test_agent_scoped_dedup() {
    let store = make_store(384);
    // Same content for two different agents should NOT be deduped
    let r1 = store
        .add(
            "shared knowledge",
            AddOptions::new("user1").agent_id("agent_a"),
        )
        .unwrap();
    let r2 = store
        .add(
            "shared knowledge",
            AddOptions::new("user1").agent_id("agent_b"),
        )
        .unwrap();

    // They should have different IDs (not deduped across agents)
    assert_ne!(
        r1.id, r2.id,
        "same content for different agents should create separate memories"
    );

    // Same content for the same agent SHOULD be deduped
    let r3 = store
        .add(
            "shared knowledge",
            AddOptions::new("user1").agent_id("agent_a"),
        )
        .unwrap();
    assert_eq!(
        r1.id, r3.id,
        "same content for same agent should be deduped"
    );
}

#[test]
fn test_keyword_search_by_agent() {
    let store = make_store(384);
    store
        .add(
            "the quick brown fox jumps",
            AddOptions::new("user1").agent_id("agent_a"),
        )
        .unwrap();
    store
        .add(
            "quantum physics and relativity",
            AddOptions::new("user1").agent_id("agent_b"),
        )
        .unwrap();
    store
        .add(
            "the lazy cat sleeps all day",
            AddOptions::new("user1").agent_id("agent_a"),
        )
        .unwrap();

    store.rebuild_fts_index().unwrap();

    // Keyword search scoped to agent_a
    let opts = SearchOptions::new("user1")
        .agent_id("agent_a")
        .keyword_search(true)
        .limit(10);
    let results = store.search("fox", opts).unwrap();
    // Should only return agent_a's memories
    for r in &results {
        // Verify none of agent_b's memories leaked through
        assert_ne!(r.content, "quantum physics and relativity");
    }
}

#[test]
fn test_agent_none_sees_all() {
    let store = make_store(384);
    store
        .add(
            "agent_a memory",
            AddOptions::new("user1").agent_id("agent_a"),
        )
        .unwrap();
    store
        .add(
            "agent_b memory",
            AddOptions::new("user1").agent_id("agent_b"),
        )
        .unwrap();
    store
        .add("no agent memory", AddOptions::new("user1"))
        .unwrap();

    // Search WITHOUT agent_id should see all memories for the user
    let opts = SearchOptions::new("user1").limit(10);
    let results = store.search("memory", opts).unwrap();
    assert_eq!(
        results.len(),
        3,
        "search without agent_id should return all user memories"
    );
}

#[test]
fn test_list_and_search_agent_consistency() {
    let store = make_store(384);
    store
        .add("alpha fact", AddOptions::new("user1").agent_id("agent_x"))
        .unwrap();
    store
        .add("beta fact", AddOptions::new("user1").agent_id("agent_x"))
        .unwrap();
    store
        .add("gamma fact", AddOptions::new("user1").agent_id("agent_y"))
        .unwrap();

    // list with agent_id
    let list_results = store
        .list_traces(ListOptions::new("user1").agent_id("agent_x"))
        .unwrap();
    assert_eq!(list_results.len(), 2);

    // search with same agent_id
    let search_results = store
        .search(
            "fact",
            SearchOptions::new("user1").agent_id("agent_x").limit(10),
        )
        .unwrap();
    assert_eq!(search_results.len(), 2);

    // Both should return the same set of memory IDs
    let mut list_ids: Vec<String> = list_results.iter().map(|r| r.id.clone()).collect();
    let mut search_ids: Vec<String> = search_results.iter().map(|r| r.id.clone()).collect();
    list_ids.sort();
    search_ids.sort();
    assert_eq!(
        list_ids, search_ids,
        "list and search with same agent_id should return same memories"
    );
}

#[test]
fn test_delete_all() {
    let store = make_store(384);
    store.add("user1 mem 1", AddOptions::new("user1")).unwrap();
    store.add("user1 mem 2", AddOptions::new("user1")).unwrap();
    store.add("user1 mem 3", AddOptions::new("user1")).unwrap();
    store.add("user2 mem 1", AddOptions::new("user2")).unwrap();
    store.add("user2 mem 2", AddOptions::new("user2")).unwrap();

    let deleted = store.delete_all_traces("user1", None, None, None).unwrap();
    assert_eq!(deleted, 3);

    let user1_list = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(user1_list.len(), 0);

    let user2_list = store.list_traces(ListOptions::new("user2")).unwrap();
    assert_eq!(user2_list.len(), 2);
}

#[test]
fn test_delete_all_with_agent() {
    let store = make_store(384);
    store
        .add("mem1", AddOptions::new("user1").agent_id("agent_a"))
        .unwrap();
    store
        .add("mem2", AddOptions::new("user1").agent_id("agent_a"))
        .unwrap();
    store
        .add("mem3", AddOptions::new("user1").agent_id("agent_b"))
        .unwrap();
    store.add("mem4", AddOptions::new("user1")).unwrap();

    let deleted = store
        .delete_all_traces("user1", Some("agent_a"), None, None)
        .unwrap();
    assert_eq!(deleted, 2);

    // agent_b and no-agent memories should remain
    let remaining = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(remaining.len(), 2);
}

#[test]
fn test_history() {
    let store = make_store(384);
    let added = store.add("original", AddOptions::new("user1")).unwrap();
    store
        .update_trace(&added.id, "updated content", None)
        .unwrap();
    store.delete_trace(&added.id).unwrap();

    let history = store.trace_history(&added.id).unwrap();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].event, "ADD");
    assert_eq!(history[1].event, "UPDATE");
    assert_eq!(history[2].event, "DELETE");
    assert!(history[0].old_memory.is_none());
    assert_eq!(history[1].old_memory.as_deref(), Some("original"));
    assert_eq!(history[1].new_memory, "updated content");
}

#[test]
fn test_reset() {
    let store = make_store(384);
    store.add("mem1", AddOptions::new("user1")).unwrap();
    store.add("mem2", AddOptions::new("user1")).unwrap();
    store.add("mem3", AddOptions::new("user2")).unwrap();

    store.reset().unwrap();

    let user1 = store.list_traces(ListOptions::new("user1")).unwrap();
    let user2 = store.list_traces(ListOptions::new("user2")).unwrap();
    assert_eq!(user1.len(), 0);
    assert_eq!(user2.len(), 0);
}

// ── Metadata filter tests ──

#[test]
fn test_search_with_metadata_filter() {
    let store = make_store(384);
    store
        .add(
            "memory about cats",
            AddOptions::new("user1")
                .metadata(serde_json::json!({"category": "animals", "priority": "high"})),
        )
        .unwrap();
    store
        .add(
            "memory about dogs",
            AddOptions::new("user1")
                .metadata(serde_json::json!({"category": "animals", "priority": "low"})),
        )
        .unwrap();
    store
        .add(
            "memory about physics",
            AddOptions::new("user1").metadata(serde_json::json!({"category": "science"})),
        )
        .unwrap();

    // Filter by category=animals
    let mut filter = std::collections::HashMap::new();
    filter.insert("category".to_string(), serde_json::json!("animals"));
    let opts = SearchOptions::new("user1")
        .metadata_filter(filter)
        .limit(10);
    let results = store.search("memory", opts).unwrap();
    assert_eq!(results.len(), 2, "should find 2 animal memories");
    for r in &results {
        let meta = r.metadata.as_ref().unwrap();
        assert_eq!(meta["category"], "animals");
    }

    // Filter by category=science
    let mut filter = std::collections::HashMap::new();
    filter.insert("category".to_string(), serde_json::json!("science"));
    let opts = SearchOptions::new("user1")
        .metadata_filter(filter)
        .limit(10);
    let results = store.search("memory", opts).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "memory about physics");

    // Filter by category=animals AND priority=high
    let mut filter = std::collections::HashMap::new();
    filter.insert("category".to_string(), serde_json::json!("animals"));
    filter.insert("priority".to_string(), serde_json::json!("high"));
    let opts = SearchOptions::new("user1")
        .metadata_filter(filter)
        .limit(10);
    let results = store.search("memory", opts).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "memory about cats");

    // Filter by nonexistent category
    let mut filter = std::collections::HashMap::new();
    filter.insert("category".to_string(), serde_json::json!("music"));
    let opts = SearchOptions::new("user1")
        .metadata_filter(filter)
        .limit(10);
    let results = store.search("memory", opts).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_list_with_metadata_filter() {
    let store = make_store(384);
    store
        .add(
            "mem1",
            AddOptions::new("user1").metadata(serde_json::json!({"env": "prod"})),
        )
        .unwrap();
    store
        .add(
            "mem2",
            AddOptions::new("user1").metadata(serde_json::json!({"env": "staging"})),
        )
        .unwrap();
    store
        .add(
            "mem3",
            AddOptions::new("user1").metadata(serde_json::json!({"env": "prod"})),
        )
        .unwrap();

    // List with filter env=prod
    let mut filter = std::collections::HashMap::new();
    filter.insert("env".to_string(), serde_json::json!("prod"));
    let opts = ListOptions::new("user1").metadata_filter(filter);
    let results = store.list_traces(opts).unwrap();
    assert_eq!(results.len(), 2);
    for r in &results {
        let meta = r.metadata.as_ref().unwrap();
        assert_eq!(meta["env"], "prod");
    }

    // List with filter env=staging
    let mut filter = std::collections::HashMap::new();
    filter.insert("env".to_string(), serde_json::json!("staging"));
    let opts = ListOptions::new("user1").metadata_filter(filter);
    let results = store.list_traces(opts).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "mem2");
}

#[test]
fn test_metadata_filter_invalid_key() {
    let store = make_store(384);
    store
        .add(
            "mem1",
            AddOptions::new("user1").metadata(serde_json::json!({"key": "value"})),
        )
        .unwrap();

    // Key with SQL injection attempt — FilterExpression treats unknown fields as
    // metadata JSON paths via json_extract_string, which is parameterized and safe.
    // The query should succeed but return no results (no metadata field matches).
    let mut filter = std::collections::HashMap::new();
    filter.insert(
        "key'; DROP TABLE memories; --".to_string(),
        serde_json::json!("value"),
    );
    let opts = SearchOptions::new("user1")
        .metadata_filter(filter)
        .limit(10);
    let result = store.search("mem1", opts);
    // Should not error — the FilterExpression approach handles this safely
    assert!(
        result.is_ok(),
        "FilterExpression should handle special chars safely"
    );
    let results = result.unwrap();
    // No results expected since the metadata field path won't match
    assert!(
        results.is_empty(),
        "should find no results for injection key"
    );

    // Verify the table still exists and data is intact
    let all = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(all.len(), 1, "memories table should still have data");
}

// ── run_id tests ──

#[test]
fn test_add_with_run_id() {
    let store = make_store(384);

    // Add memories with different run_ids
    store
        .add("run1 mem1", AddOptions::new("user1").run_id("run_a"))
        .unwrap();
    store
        .add("run1 mem2", AddOptions::new("user1").run_id("run_a"))
        .unwrap();
    store
        .add("run2 mem1", AddOptions::new("user1").run_id("run_b"))
        .unwrap();
    store.add("no run mem", AddOptions::new("user1")).unwrap();

    // List filtered by run_id
    let results = store
        .list_traces(ListOptions::new("user1").run_id("run_a"))
        .unwrap();
    assert_eq!(results.len(), 2, "run_a should have 2 memories");

    let results = store
        .list_traces(ListOptions::new("user1").run_id("run_b"))
        .unwrap();
    assert_eq!(results.len(), 1, "run_b should have 1 memory");

    // List without run_id should see all
    let results = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(results.len(), 4, "all user1 memories");

    // Search filtered by run_id
    let results = store
        .search("mem", SearchOptions::new("user1").run_id("run_a").limit(10))
        .unwrap();
    assert_eq!(results.len(), 2, "search with run_a");

    let results = store
        .search("mem", SearchOptions::new("user1").run_id("run_b").limit(10))
        .unwrap();
    assert_eq!(results.len(), 1, "search with run_b");
}

// ── ChatMessage / append_events + compact tests ──

#[test]
fn test_append_events_and_compact() {
    use memme_llm::{LlmError, Message as LlmMessage};
    use std::sync::Mutex;

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
    impl memme_llm::LlmProvider for MockLlm {
        fn generate(
            &self,
            _messages: &[LlmMessage],
            _options: &memme_llm::GenerateOptions,
        ) -> std::result::Result<String, LlmError> {
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

    // Use compact_fallback_token_threshold: 0 to always use LLM for compact in this test
    let config = MemoryConfig {
        db_path: ":memory:".into(),
        collection_name: "test".into(),
        embedding_dims: 384,
        dedup_threshold: 0.15,
        default_limit: 10,
        compact_fallback_token_threshold: 0,
        ..Default::default()
    };
    let embedder = Arc::new(MockEmbedder::new(384));
    let store = MemoryStore::new(config, embedder).unwrap();
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
            image_url: None,
            image_type: None,
            timestamp: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "My name is Alice and I love painting.".to_string(),
            image_url: None,
            image_type: None,
            timestamp: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "Nice to meet you, Alice! Painting is wonderful.".to_string(),
            image_url: None,
            image_type: None,
            timestamp: None,
        },
    ];

    // compact() does purification + summarization in a single LLM call
    let compact_response = r#"{"purified": [
        {"content": "Alice's name is Alice and Alice loves painting.", "event_time": null, "location": null},
        {"content": "Nice to meet you, Alice! Painting is wonderful.", "event_time": null, "location": null}
    ], "title": "Meeting Alice", "summary": "Alice introduced herself and shared her love of painting.", "significance": 0.6}"#.to_string();

    let llm = Arc::new(MockLlm::new(vec![compact_response]));
    let store = store.with_llm(llm);

    // Phase 1: append events
    let session_id = "test-session-1";
    let append_result = store
        .append_events(session_id, &messages, "user1", None)
        .unwrap();
    assert!(append_result.events_appended > 0);

    // Phase 2: compact — creates an episode narrative, not individual memories
    let compact_result = store.compact(session_id).unwrap();

    assert!(
        compact_result.events_processed >= 2,
        "Should process at least 2 events"
    );
    assert!(
        !compact_result.episode_id.is_empty(),
        "Should create an episode"
    );
    // compact no longer returns memories (moved to meditate)
    assert!(
        compact_result.memories.is_empty(),
        "compact returns no memories — use meditate() for that"
    );

    // Verify the narrative trace was inserted into the store
    let listed = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(listed.len(), 1, "Should have 1 narrative trace");
    assert!(
        listed[0].content.contains("Meeting Alice"),
        "Narrative should contain episode title"
    );
}

// ── Graph search tests ──

mod smart_graph_tests {
    use super::*;

    fn make_store_with_graph(enable_graph: bool) -> MemoryStore {
        let config = MemoryConfig {
            db_path: ":memory:".into(),
            collection_name: "test".into(),
            embedding_dims: 384,
            dedup_threshold: 0.15,
            default_limit: 10,
            enable_graph,
            ..Default::default()
        };
        let embedder = Arc::new(memme_embeddings::mock::MockEmbedder::new(384));
        MemoryStore::new(config, embedder).unwrap()
    }

    #[test]
    fn test_search_with_graph_enabled() {
        let store = make_store_with_graph(true);

        store
            .add("Alice works at Google", AddOptions::new("user1"))
            .unwrap();
        store
            .storage()
            .upsert_entity("e1", "Alice", Some("person"), "user1")
            .unwrap();
        store
            .storage()
            .upsert_entity("e2", "Google", Some("organization"), "user1")
            .unwrap();
        store
            .storage()
            .insert_relationship("r1", "e1", "e2", "works_at", "user1")
            .unwrap();

        // search() with graph enabled includes entity-centric retrieval
        let results = store.search("Alice", SearchOptions::new("user1")).unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_search_without_graph() {
        let store = make_store_with_graph(false);

        store
            .add("Alice works at Google", AddOptions::new("user1"))
            .unwrap();

        let results = store.search("Alice", SearchOptions::new("user1")).unwrap();

        assert!(!results.is_empty());
    }
}

// ── Edge-first: access tracking, importance, consolidation, weighted search ──

#[test]
fn test_access_count_incremented() {
    let store = make_store(384);
    let added = store
        .add("test access tracking", AddOptions::new("user1"))
        .unwrap();

    // get_trace() queues a deferred access increment; flush it before checking
    let _fetched = store.get_trace(&added.id).unwrap().unwrap();
    store.flush_deferred_writes();
    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    assert!(
        fetched.access_count.unwrap_or(0) > 0,
        "access_count should be > 0 after get + flush"
    );

    // search() triggers flush internally via add path; check increment
    store.flush_deferred_writes();
    let before = store
        .get_trace(&added.id)
        .unwrap()
        .unwrap()
        .access_count
        .unwrap_or(0);
    let _results = store
        .search("test access", SearchOptions::new("user1"))
        .unwrap();
    store.flush_deferred_writes();
    let after = store
        .get_trace(&added.id)
        .unwrap()
        .unwrap()
        .access_count
        .unwrap_or(0);
    assert!(
        after >= before,
        "access_count should not decrease after search, before={before} after={after}"
    );
}

#[test]
fn test_importance_on_add() {
    let store = make_store(384);

    // Default importance (0.5)
    let r1 = store
        .add("default importance", AddOptions::new("user1"))
        .unwrap();
    assert!(
        (r1.importance.unwrap_or(0.0) - 0.5).abs() < 0.01,
        "default importance should be 0.5"
    );

    // Custom importance
    let r2 = store
        .add("high importance", AddOptions::new("user1").importance(0.9))
        .unwrap();
    assert!(
        (r2.importance.unwrap_or(0.0) - 0.9).abs() < 0.01,
        "custom importance should be 0.9"
    );

    let r3 = store
        .add("low importance", AddOptions::new("user1").importance(0.1))
        .unwrap();
    assert!(
        (r3.importance.unwrap_or(0.0) - 0.1).abs() < 0.01,
        "custom importance should be 0.1"
    );
}

#[test]
fn test_consolidate_decay() {
    let store = make_store(384);

    // Add memories with known importance
    store
        .add("memory one", AddOptions::new("user1").importance(0.8))
        .unwrap();
    store
        .add("memory two", AddOptions::new("user1").importance(0.6))
        .unwrap();

    // Consolidate with a decay rate -- since memories were just created,
    // days_since_last_access is 0, so importance should not change significantly
    let result = store.consolidate("user1", 0.01, 0.0, false).unwrap();
    assert!(result.decayed_count > 0, "should report decayed memories");
    assert_eq!(
        result.deleted_count, 0,
        "should not delete when delete_below=false"
    );

    // Verify importance is still approximately the same (0 days elapsed)
    let memories = store.list_traces(ListOptions::new("user1")).unwrap();
    for m in &memories {
        assert!(
            m.importance.unwrap_or(0.0) >= 0.0,
            "importance should not go negative"
        );
    }
}

#[test]
fn test_consolidate_delete() {
    let store = make_store(384);

    // Add memories with very low importance
    store
        .add("will be deleted", AddOptions::new("user1").importance(0.01))
        .unwrap();
    store
        .add("will survive", AddOptions::new("user1").importance(0.9))
        .unwrap();

    // First decay to push low-importance below threshold
    // Then delete below threshold
    let result = store.consolidate("user1", 0.0, 0.05, true).unwrap();

    // The memory with importance=0.01 should be deleted (< 0.05 threshold)
    assert_eq!(
        result.deleted_count, 1,
        "should delete 1 memory below threshold"
    );

    let remaining = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(remaining.len(), 1, "only 1 memory should remain");
    assert_eq!(remaining[0].content, "will survive");
}

#[test]
fn test_search_with_importance() {
    let store = make_store(384);

    // Add memories with different importance levels
    store
        .add(
            "low importance content",
            AddOptions::new("user1").importance(0.1),
        )
        .unwrap();
    store
        .add(
            "high importance content",
            AddOptions::new("user1").importance(0.9),
        )
        .unwrap();

    // Search -- importance is used in scoring when forgetting curve is enabled
    let results = store
        .search("content", SearchOptions::new("user1").limit(10))
        .unwrap();

    assert!(!results.is_empty());
    // All results should have a score
    for r in &results {
        assert!(r.score.is_some(), "search should set score");
    }
}

// ══════════════════════════════════════════════════════════════
// P1 Feature Tests
// ══════════════════════════════════════════════════════════════

// ── 1. app_id 4-level scoping ──

#[test]
fn test_app_id_appears_in_result() {
    let store = make_store(384);
    let r = store
        .add("hello", AddOptions::new("user1").app_id("myapp"))
        .unwrap();
    assert_eq!(r.app_id.as_deref(), Some("myapp"));

    let fetched = store.get_trace(&r.id).unwrap().unwrap();
    assert_eq!(fetched.app_id.as_deref(), Some("myapp"));
}

#[test]
fn test_list_filter_by_app_id() {
    let store = make_store(384);
    store
        .add("app1 mem1", AddOptions::new("user1").app_id("app1"))
        .unwrap();
    store
        .add("app1 mem2", AddOptions::new("user1").app_id("app1"))
        .unwrap();
    store
        .add("app2 mem1", AddOptions::new("user1").app_id("app2"))
        .unwrap();
    store.add("no app mem", AddOptions::new("user1")).unwrap();

    let results = store
        .list_traces(ListOptions::new("user1").app_id("app1"))
        .unwrap();
    assert_eq!(results.len(), 2);
    for r in &results {
        assert_eq!(r.app_id.as_deref(), Some("app1"));
    }

    let results = store
        .list_traces(ListOptions::new("user1").app_id("app2"))
        .unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_search_isolates_by_app_id() {
    let store = make_store(384);
    store
        .add("cats in app1", AddOptions::new("user1").app_id("app1"))
        .unwrap();
    store
        .add("cats in app2", AddOptions::new("user1").app_id("app2"))
        .unwrap();
    store.add("cats no app", AddOptions::new("user1")).unwrap();

    let results = store
        .search("cats", SearchOptions::new("user1").app_id("app1").limit(10))
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].app_id.as_deref(), Some("app1"));
}

#[test]
fn test_cross_app_isolation() {
    let store = make_store(384);
    store
        .add("secret from app1", AddOptions::new("user1").app_id("app1"))
        .unwrap();
    store
        .add("secret from app2", AddOptions::new("user1").app_id("app2"))
        .unwrap();

    // app1 should not see app2's memories
    let results = store
        .list_traces(ListOptions::new("user1").app_id("app1"))
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "secret from app1");

    let results = store
        .search(
            "secret",
            SearchOptions::new("user1").app_id("app2").limit(10),
        )
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "secret from app2");
}

#[test]
fn test_keyword_search_with_app_id() {
    let store = make_store(384);
    store
        .add("quick fox in app1", AddOptions::new("user1").app_id("app1"))
        .unwrap();
    store
        .add("quick fox in app2", AddOptions::new("user1").app_id("app2"))
        .unwrap();
    store.rebuild_fts_index().unwrap();

    let opts = SearchOptions::new("user1")
        .app_id("app1")
        .keyword_search(true)
        .limit(10);
    let results = store.search("fox", opts).unwrap();
    for r in &results {
        assert_ne!(
            r.content, "quick fox in app2",
            "app2 memories should not appear in app1 keyword search"
        );
    }
}

#[test]
fn test_combined_user_agent_app_run_isolation() {
    let store = make_store(384);
    store
        .add(
            "full scope mem",
            AddOptions::new("user1")
                .agent_id("agent1")
                .app_id("app1")
                .run_id("run1"),
        )
        .unwrap();
    store
        .add(
            "different run",
            AddOptions::new("user1")
                .agent_id("agent1")
                .app_id("app1")
                .run_id("run2"),
        )
        .unwrap();
    store
        .add(
            "different app",
            AddOptions::new("user1")
                .agent_id("agent1")
                .app_id("app2")
                .run_id("run1"),
        )
        .unwrap();
    store
        .add(
            "different agent",
            AddOptions::new("user1")
                .agent_id("agent2")
                .app_id("app1")
                .run_id("run1"),
        )
        .unwrap();

    // Exact 4-level scope should find exactly 1
    let results = store
        .list_traces(
            ListOptions::new("user1")
                .agent_id("agent1")
                .app_id("app1")
                .run_id("run1"),
        )
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "full scope mem");
}

// ── 2. Immutable memories ──

#[test]
fn test_immutable_memory_cannot_be_updated() {
    let store = make_store(384);
    let r = store
        .add("immutable fact", AddOptions::new("user1").immutable(true))
        .unwrap();
    assert!(r.immutable);

    let result = store.update_trace(&r.id, "new content", None);
    assert!(result.is_err());
    match result.unwrap_err() {
        MemoryError::ImmutableMemory(_) => {}
        other => panic!("Expected ImmutableMemory error, got: {other}"),
    }

    // Content should be unchanged
    let fetched = store.get_trace(&r.id).unwrap().unwrap();
    assert_eq!(fetched.content, "immutable fact");
}

#[test]
fn test_immutable_memory_cannot_be_deleted() {
    let store = make_store(384);
    let r = store
        .add("immutable fact", AddOptions::new("user1").immutable(true))
        .unwrap();

    let result = store.delete_trace(&r.id);
    assert!(result.is_err());
    match result.unwrap_err() {
        MemoryError::ImmutableMemory(_) => {}
        other => panic!("Expected ImmutableMemory error, got: {other}"),
    }

    // Memory should still exist
    let fetched = store.get_trace(&r.id).unwrap();
    assert!(fetched.is_some());
}

#[test]
fn test_immutable_flag_in_search_results() {
    let store = make_store(384);
    store
        .add("immutable one", AddOptions::new("user1").immutable(true))
        .unwrap();
    store
        .add("mutable one", AddOptions::new("user1").immutable(false))
        .unwrap();

    let results = store
        .search("one", SearchOptions::new("user1").limit(10))
        .unwrap();
    let immutable_count = results.iter().filter(|r| r.immutable).count();
    let mutable_count = results.iter().filter(|r| !r.immutable).count();
    assert_eq!(immutable_count, 1);
    assert_eq!(mutable_count, 1);
}

#[test]
fn test_non_immutable_can_be_updated_and_deleted() {
    let store = make_store(384);
    let r = store
        .add("mutable fact", AddOptions::new("user1").immutable(false))
        .unwrap();
    assert!(!r.immutable);

    // Update should succeed
    let updated = store
        .update_trace(&r.id, "updated mutable fact", None)
        .unwrap();
    assert_eq!(updated.content, "updated mutable fact");

    // Delete should succeed
    store.delete_trace(&r.id).unwrap();
    let fetched = store.get_trace(&r.id).unwrap();
    assert!(fetched.is_none());
}

#[test]
fn test_mix_immutable_and_mutable_same_user() {
    let store = make_store(384);
    let imm = store
        .add("immutable mem", AddOptions::new("user1").immutable(true))
        .unwrap();
    let mut_mem = store
        .add("mutable mem", AddOptions::new("user1").immutable(false))
        .unwrap();

    // Can update the mutable one
    store
        .update_trace(&mut_mem.id, "changed mutable", None)
        .unwrap();
    // Cannot update the immutable one
    assert!(store
        .update_trace(&imm.id, "changed immutable", None)
        .is_err());

    // Can delete the mutable one
    store.delete_trace(&mut_mem.id).unwrap();
    // Cannot delete the immutable one
    assert!(store.delete_trace(&imm.id).is_err());

    // Only the immutable one should remain
    let list = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].content, "immutable mem");
}

// ── 3. TTL / expiration ──

#[test]
fn test_expired_memory_cleaned_by_consolidate() {
    let store = make_store(384);
    // Past expiration
    store
        .add(
            "expired mem",
            AddOptions::new("user1").expiration_date("2020-01-01T00:00:00"),
        )
        .unwrap();
    // Future expiration
    store
        .add(
            "future mem",
            AddOptions::new("user1").expiration_date("2099-12-31T23:59:59"),
        )
        .unwrap();
    // No expiration
    store.add("forever mem", AddOptions::new("user1")).unwrap();

    let result = store.consolidate("user1", 0.0, 0.0, false).unwrap();
    assert_eq!(result.expired_count, 1);

    let remaining = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(remaining.len(), 2);
    let contents: Vec<&str> = remaining.iter().map(|r| r.content.as_str()).collect();
    assert!(contents.contains(&"future mem"));
    assert!(contents.contains(&"forever mem"));
    assert!(!contents.contains(&"expired mem"));
}

#[test]
fn test_future_expiration_not_cleaned() {
    let store = make_store(384);
    store
        .add(
            "future mem",
            AddOptions::new("user1").expiration_date("2099-12-31T23:59:59"),
        )
        .unwrap();

    let result = store.consolidate("user1", 0.0, 0.0, false).unwrap();
    assert_eq!(result.expired_count, 0);

    let remaining = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(remaining.len(), 1);
}

#[test]
fn test_expiration_date_in_results() {
    let store = make_store(384);
    let r = store
        .add(
            "expiring",
            AddOptions::new("user1").expiration_date("2099-06-15T10:00:00"),
        )
        .unwrap();
    assert!(r.expiration_date.is_some());

    let fetched = store.get_trace(&r.id).unwrap().unwrap();
    assert!(fetched.expiration_date.is_some());
}

#[test]
fn test_no_expiration_never_expires() {
    let store = make_store(384);
    store.add("no exp mem", AddOptions::new("user1")).unwrap();

    // Consolidate many times, should never be expired
    for _ in 0..3 {
        let result = store.consolidate("user1", 0.0, 0.0, false).unwrap();
        assert_eq!(result.expired_count, 0);
    }
    let remaining = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(remaining.len(), 1);
}

// ── 4. Custom categories ──

#[test]
fn test_add_memory_with_categories() {
    let store = make_store(384);
    let r = store
        .add(
            "work meeting notes",
            AddOptions::new("user1").categories(vec!["work".into(), "meetings".into()]),
        )
        .unwrap();
    assert!(r.categories.is_some());
    let cats = r.categories.unwrap();
    assert!(cats.contains(&"work".to_string()));
    assert!(cats.contains(&"meetings".to_string()));
}

#[test]
fn test_categories_in_get_list_search() {
    let store = make_store(384);
    let added = store
        .add(
            "categorized memory",
            AddOptions::new("user1").categories(vec!["tech".into(), "rust".into()]),
        )
        .unwrap();

    // get
    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    assert!(fetched
        .categories
        .as_ref()
        .unwrap()
        .contains(&"tech".to_string()));

    // list
    let listed = store.list_traces(ListOptions::new("user1")).unwrap();
    assert!(listed[0]
        .categories
        .as_ref()
        .unwrap()
        .contains(&"rust".to_string()));

    // search
    let searched = store
        .search("categorized", SearchOptions::new("user1").limit(10))
        .unwrap();
    assert!(searched[0]
        .categories
        .as_ref()
        .unwrap()
        .contains(&"tech".to_string()));
}

#[test]
fn test_search_with_categories_filter() {
    let store = make_store(384);
    store
        .add(
            "work item",
            AddOptions::new("user1").categories(vec!["work".into()]),
        )
        .unwrap();
    store
        .add(
            "personal item",
            AddOptions::new("user1").categories(vec!["personal".into()]),
        )
        .unwrap();
    store
        .add(
            "work and personal",
            AddOptions::new("user1").categories(vec!["work".into(), "personal".into()]),
        )
        .unwrap();

    let opts = SearchOptions::new("user1")
        .filter(FilterExpression::contains("categories", "work"))
        .limit(10);
    let results = store.search("item", opts).unwrap();
    assert_eq!(results.len(), 2);
    for r in &results {
        assert!(r.categories.as_ref().unwrap().contains(&"work".to_string()));
    }
}

#[test]
fn test_empty_categories() {
    let store = make_store(384);
    let r = store.add("no cats", AddOptions::new("user1")).unwrap();
    // No categories set -- should be None or empty
    assert!(r.categories.is_none() || r.categories.as_ref().unwrap().is_empty());
}

// ── 5. Advanced filter expressions ──

#[test]
fn test_filter_eq_on_user_id() {
    let store = make_store(384);
    store.add("alice mem", AddOptions::new("alice")).unwrap();
    store.add("bob mem", AddOptions::new("bob")).unwrap();

    let opts = ListOptions::new("alice").filter(FilterExpression::eq("user_id", "alice"));
    let results = store.list_traces(opts).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].user_id, "alice");
}

#[test]
fn test_filter_ne() {
    let store = make_store(384);
    store
        .add("agent_a mem", AddOptions::new("user1").agent_id("agent_a"))
        .unwrap();
    store
        .add("agent_b mem", AddOptions::new("user1").agent_id("agent_b"))
        .unwrap();
    store.add("no agent mem", AddOptions::new("user1")).unwrap();

    let opts = SearchOptions::new("user1")
        .filter(FilterExpression::ne("agent_id", "agent_a"))
        .limit(10);
    let results = store.search("mem", opts).unwrap();
    // Should return agent_b and possibly no-agent (agent_id != 'agent_a')
    for r in &results {
        assert_ne!(r.agent_id.as_deref(), Some("agent_a"));
    }
}

#[test]
fn test_filter_gte_lte_on_importance() {
    let store = make_store(384);
    store
        .add("low imp", AddOptions::new("user1").importance(0.2))
        .unwrap();
    store
        .add("mid imp", AddOptions::new("user1").importance(0.5))
        .unwrap();
    store
        .add("high imp", AddOptions::new("user1").importance(0.9))
        .unwrap();

    // gte 0.5
    let opts = ListOptions::new("user1")
        .filter(FilterExpression::gte("importance", serde_json::json!(0.5)));
    let results = store.list_traces(opts).unwrap();
    assert_eq!(results.len(), 2);
    for r in &results {
        assert!(r.importance.unwrap() >= 0.49, "importance should be >= 0.5");
    }

    // lte 0.5
    let opts = ListOptions::new("user1")
        .filter(FilterExpression::lte("importance", serde_json::json!(0.5)));
    let results = store.list_traces(opts).unwrap();
    assert_eq!(results.len(), 2);
    for r in &results {
        assert!(r.importance.unwrap() <= 0.51, "importance should be <= 0.5");
    }
}

#[test]
fn test_filter_is_in_on_agent_id() {
    let store = make_store(384);
    store
        .add("a1 mem", AddOptions::new("user1").agent_id("a1"))
        .unwrap();
    store
        .add("a2 mem", AddOptions::new("user1").agent_id("a2"))
        .unwrap();
    store
        .add("a3 mem", AddOptions::new("user1").agent_id("a3"))
        .unwrap();

    let opts = ListOptions::new("user1").filter(FilterExpression::is_in(
        "agent_id",
        vec!["a1".into(), "a3".into()],
    ));
    let results = store.list_traces(opts).unwrap();
    assert_eq!(results.len(), 2);
    let agent_ids: Vec<Option<&str>> = results.iter().map(|r| r.agent_id.as_deref()).collect();
    assert!(agent_ids.contains(&Some("a1")));
    assert!(agent_ids.contains(&Some("a3")));
    assert!(!agent_ids.contains(&Some("a2")));
}

#[test]
fn test_filter_contains_on_categories() {
    let store = make_store(384);
    store
        .add(
            "work mem",
            AddOptions::new("user1").categories(vec!["work".into(), "urgent".into()]),
        )
        .unwrap();
    store
        .add(
            "personal mem",
            AddOptions::new("user1").categories(vec!["personal".into()]),
        )
        .unwrap();

    let opts = ListOptions::new("user1").filter(FilterExpression::contains("categories", "work"));
    let results = store.list_traces(opts).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "work mem");
}

#[test]
fn test_filter_icontains_on_content() {
    let store = make_store(384);
    store
        .add("I love COFFEE every morning", AddOptions::new("user1"))
        .unwrap();
    store.add("Tea is great", AddOptions::new("user1")).unwrap();

    let opts = ListOptions::new("user1").filter(FilterExpression::icontains("content", "coffee"));
    let results = store.list_traces(opts).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].content.to_lowercase().contains("coffee"));
}

#[test]
fn test_filter_and_multiple_conditions() {
    let store = make_store(384);
    store
        .add(
            "high imp work",
            AddOptions::new("user1")
                .importance(0.9)
                .categories(vec!["work".into()]),
        )
        .unwrap();
    store
        .add(
            "low imp work",
            AddOptions::new("user1")
                .importance(0.2)
                .categories(vec!["work".into()]),
        )
        .unwrap();
    store
        .add(
            "high imp personal",
            AddOptions::new("user1")
                .importance(0.9)
                .categories(vec!["personal".into()]),
        )
        .unwrap();

    let opts = ListOptions::new("user1").filter(FilterExpression::and(vec![
        FilterExpression::gte("importance", serde_json::json!(0.5)),
        FilterExpression::contains("categories", "work"),
    ]));
    let results = store.list_traces(opts).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "high imp work");
}

#[test]
fn test_filter_or_multiple_conditions() {
    let store = make_store(384);
    store
        .add("agent_a mem", AddOptions::new("user1").agent_id("agent_a"))
        .unwrap();
    store
        .add("agent_b mem", AddOptions::new("user1").agent_id("agent_b"))
        .unwrap();
    store
        .add("agent_c mem", AddOptions::new("user1").agent_id("agent_c"))
        .unwrap();

    let opts = ListOptions::new("user1").filter(FilterExpression::or(vec![
        FilterExpression::eq("agent_id", "agent_a"),
        FilterExpression::eq("agent_id", "agent_c"),
    ]));
    let results = store.list_traces(opts).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_filter_nested_and_or() {
    let store = make_store(384);
    store
        .add(
            "a high",
            AddOptions::new("user1").agent_id("a").importance(0.9),
        )
        .unwrap();
    store
        .add(
            "a low",
            AddOptions::new("user1").agent_id("a").importance(0.1),
        )
        .unwrap();
    store
        .add(
            "b high",
            AddOptions::new("user1").agent_id("b").importance(0.9),
        )
        .unwrap();
    store
        .add(
            "b low",
            AddOptions::new("user1").agent_id("b").importance(0.1),
        )
        .unwrap();

    // (agent_id = 'a' AND importance >= 0.5) OR (agent_id = 'b' AND importance < 0.5)
    let opts = ListOptions::new("user1").filter(FilterExpression::or(vec![
        FilterExpression::and(vec![
            FilterExpression::eq("agent_id", "a"),
            FilterExpression::gte("importance", serde_json::json!(0.5)),
        ]),
        FilterExpression::and(vec![
            FilterExpression::eq("agent_id", "b"),
            FilterExpression::lt("importance", serde_json::json!(0.5)),
        ]),
    ]));
    let results = store.list_traces(opts).unwrap();
    assert_eq!(results.len(), 2);
    let contents: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();
    assert!(contents.contains(&"a high"));
    assert!(contents.contains(&"b low"));
}

#[test]
fn test_filter_no_results() {
    let store = make_store(384);
    store.add("some memory", AddOptions::new("user1")).unwrap();

    let opts =
        ListOptions::new("user1").filter(FilterExpression::eq("agent_id", "nonexistent_agent"));
    let results = store.list_traces(opts).unwrap();
    assert!(results.is_empty());
}

// ── 6. Custom timestamp on update (via storage layer) ──

#[test]
fn test_update_with_custom_timestamp() {
    let store = make_store(384);
    let added = store.add("original", AddOptions::new("user1")).unwrap();

    // Update via storage with custom timestamp
    let embedding = store.embedder.embed("updated content").unwrap();
    let hash = content_hash("updated content");
    let opts = UpdateOptions::new().timestamp("2025-01-01T00:00:00");
    store
        .storage()
        .update_memory(
            &added.id,
            "updated content",
            &embedding,
            &hash,
            None,
            Some(&opts),
        )
        .unwrap();

    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    assert_eq!(fetched.content, "updated content");
    assert!(
        fetched.updated_at.contains("2025-01-01"),
        "updated_at should reflect custom timestamp, got: {}",
        fetched.updated_at
    );
}

#[test]
fn test_update_without_custom_timestamp_uses_current() {
    let store = make_store(384);
    let added = store.add("original", AddOptions::new("user1")).unwrap();

    // Update normally (no custom timestamp)
    let updated = store
        .update_trace(&added.id, "updated content", None)
        .unwrap();
    assert_eq!(updated.content, "updated content");
    // updated_at should be recent (current year)
    assert!(
        updated.updated_at.starts_with("202"),
        "updated_at should be recent, got: {}",
        updated.updated_at
    );
}

// ── 7. Export / Import ──

#[test]
fn test_export_and_import() {
    let store = make_store(384);
    store
        .add(
            "mem1",
            AddOptions::new("user1").metadata(serde_json::json!({"key": "val"})),
        )
        .unwrap();
    store
        .add(
            "mem2",
            AddOptions::new("user1").categories(vec!["work".into()]),
        )
        .unwrap();
    store.add("mem3", AddOptions::new("user2")).unwrap();

    // Export all
    let exported = store.storage().export_memories(None).unwrap();
    assert_eq!(exported.len(), 3);

    // Verify structure
    let m1 = exported.iter().find(|e| e.content == "mem1").unwrap();
    assert_eq!(m1.user_id, "user1");
    assert!(m1.metadata.is_some());
    assert_eq!(m1.metadata.as_ref().unwrap()["key"], "val");

    let m2 = exported.iter().find(|e| e.content == "mem2").unwrap();
    assert!(m2
        .categories
        .as_ref()
        .unwrap()
        .contains(&"work".to_string()));

    // Import into a new store
    let store2 = make_store(384);
    let imported_count = store2.storage().import_memories(&exported).unwrap();
    assert_eq!(imported_count, 3);

    // Verify imported data
    let listed = store2.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(listed.len(), 2);
    let listed_all = store2.list_traces(ListOptions::new("user2")).unwrap();
    assert_eq!(listed_all.len(), 1);
}

#[test]
fn test_export_with_user_filter() {
    let store = make_store(384);
    store.add("user1 mem", AddOptions::new("user1")).unwrap();
    store.add("user2 mem", AddOptions::new("user2")).unwrap();

    let exported = store.storage().export_memories(Some("user1")).unwrap();
    assert_eq!(exported.len(), 1);
    assert_eq!(exported[0].user_id, "user1");
}

#[test]
fn test_import_preserves_categories_and_metadata() {
    let store = make_store(384);
    store
        .add(
            "rich mem",
            AddOptions::new("user1")
                .metadata(serde_json::json!({"source": "web"}))
                .categories(vec!["tech".into(), "news".into()])
                .app_id("myapp")
                .importance(0.8)
                .immutable(true)
                .expiration_date("2099-12-31T00:00:00"),
        )
        .unwrap();

    let exported = store.storage().export_memories(None).unwrap();
    assert_eq!(exported.len(), 1);
    let e = &exported[0];
    assert_eq!(e.app_id.as_deref(), Some("myapp"));
    assert!((e.importance - 0.8).abs() < 0.01);
    assert!(e.immutable);
    assert!(e.expiration_date.is_some());

    // Import and verify
    let store2 = make_store(384);
    store2.storage().import_memories(&exported).unwrap();
    let listed = store2.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(listed.len(), 1);
    let imported = &listed[0];
    assert_eq!(imported.metadata.as_ref().unwrap()["source"], "web");
    assert!(imported
        .categories
        .as_ref()
        .unwrap()
        .contains(&"tech".to_string()));
    assert!(imported
        .categories
        .as_ref()
        .unwrap()
        .contains(&"news".to_string()));
}

#[test]
fn test_export_json_serialization() {
    let store = make_store(384);
    store.add("json test", AddOptions::new("user1")).unwrap();
    let exported = store.storage().export_memories(None).unwrap();

    // Should serialize to valid JSON
    let json_str = serde_json::to_string(&exported).unwrap();
    assert!(json_str.contains("json test"));

    // Should deserialize back
    let deserialized: Vec<MemoryExport> = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.len(), 1);
    assert_eq!(deserialized[0].content, "json test");
}

// ── 8. Inclusion / Exclusion prompts (config only) ──

#[test]
fn test_config_with_inclusion_exclusion_prompts() {
    let config = MemoryConfig {
        db_path: ":memory:".into(),
        collection_name: "test".into(),
        embedding_dims: 384,
        inclusion_prompt: Some("Extract work-related tasks and deadlines".into()),
        exclusion_prompt: Some("Do not extract passwords or financial data".into()),
        ..Default::default()
    };
    assert!(config.validate().is_ok());
    assert_eq!(
        config.inclusion_prompt.as_deref(),
        Some("Extract work-related tasks and deadlines")
    );
    assert_eq!(
        config.exclusion_prompt.as_deref(),
        Some("Do not extract passwords or financial data")
    );
}

#[test]
fn test_config_without_prompts_validates() {
    let config = MemoryConfig {
        db_path: ":memory:".into(),
        collection_name: "test".into(),
        embedding_dims: 384,
        inclusion_prompt: None,
        exclusion_prompt: None,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
    assert!(config.inclusion_prompt.is_none());
    assert!(config.exclusion_prompt.is_none());
}

#[test]
fn test_store_creation_with_prompts() {
    let config = MemoryConfig {
        db_path: ":memory:".into(),
        collection_name: "test".into(),
        embedding_dims: 384,
        inclusion_prompt: Some("Only extract cooking recipes".into()),
        exclusion_prompt: Some("Ignore small talk".into()),
        ..Default::default()
    };
    let embedder = Arc::new(MockEmbedder::new(384));
    let store = MemoryStore::new(config, embedder);
    assert!(
        store.is_ok(),
        "Store should create successfully with inclusion/exclusion prompts"
    );
}

// ══════════════════════════════════════════════════════════════
// P3 Feature Tests
// ══════════════════════════════════════════════════════════════

// ── 1. Batch operations ──

#[test]
fn test_batch_update_happy_path() {
    let store = make_store(384);
    let r1 = store.add("mem one", AddOptions::new("user1")).unwrap();
    let r2 = store.add("mem two", AddOptions::new("user1")).unwrap();

    let updates = vec![
        (r1.id.clone(), "updated one".to_string()),
        (r2.id.clone(), "updated two".to_string()),
    ];
    let results = store.batch_update_traces(&updates).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].content, "updated one");
    assert_eq!(results[1].content, "updated two");
}

#[test]
fn test_batch_update_skips_immutable() {
    let store = make_store(384);
    let r1 = store.add("mutable mem", AddOptions::new("user1")).unwrap();
    let r2 = store
        .add("immutable mem", AddOptions::new("user1").immutable(true))
        .unwrap();

    let updates = vec![
        (r1.id.clone(), "changed mutable".to_string()),
        (r2.id.clone(), "should be skipped".to_string()),
    ];
    let results = store.batch_update_traces(&updates).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "changed mutable");

    // Immutable memory should remain unchanged
    let fetched = store.get_trace(&r2.id).unwrap().unwrap();
    assert_eq!(fetched.content, "immutable mem");
}

#[test]
fn test_batch_update_empty_input() {
    let store = make_store(384);
    let results = store.batch_update_traces(&[]).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_batch_update_nonexistent_ids() {
    let store = make_store(384);
    let updates = vec![
        ("nonexistent-1".to_string(), "content".to_string()),
        ("nonexistent-2".to_string(), "content".to_string()),
    ];
    let results = store.batch_update_traces(&updates).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_batch_delete_happy_path() {
    let store = make_store(384);
    let r1 = store.add("del one", AddOptions::new("user1")).unwrap();
    let r2 = store.add("del two", AddOptions::new("user1")).unwrap();
    let r3 = store.add("keep this", AddOptions::new("user1")).unwrap();

    let ids = vec![r1.id.clone(), r2.id.clone()];
    let count = store.batch_delete_traces(&ids).unwrap();
    assert_eq!(count, 2);

    // Verify deleted
    assert!(store.get_trace(&r1.id).unwrap().is_none());
    assert!(store.get_trace(&r2.id).unwrap().is_none());
    // Verify kept
    assert!(store.get_trace(&r3.id).unwrap().is_some());
}

#[test]
fn test_batch_delete_skips_immutable() {
    let store = make_store(384);
    let r1 = store.add("mutable del", AddOptions::new("user1")).unwrap();
    let r2 = store
        .add("immutable del", AddOptions::new("user1").immutable(true))
        .unwrap();

    let ids = vec![r1.id.clone(), r2.id.clone()];
    let count = store.batch_delete_traces(&ids).unwrap();
    assert_eq!(count, 1);

    // Mutable deleted
    assert!(store.get_trace(&r1.id).unwrap().is_none());
    // Immutable still exists
    assert!(store.get_trace(&r2.id).unwrap().is_some());
}

#[test]
fn test_batch_delete_nonexistent_ids() {
    let store = make_store(384);
    let ids = vec!["nonexistent-1".to_string(), "nonexistent-2".to_string()];
    let count = store.batch_delete_traces(&ids).unwrap();
    assert_eq!(count, 0);
}

#[test]
fn test_batch_delete_empty_input() {
    let store = make_store(384);
    let count = store.batch_delete_traces(&[]).unwrap();
    assert_eq!(count, 0);
}

#[test]
fn test_batch_delete_mixed_existing_nonexistent_immutable() {
    let store = make_store(384);
    let r1 = store.add("deletable", AddOptions::new("user1")).unwrap();
    let r2 = store
        .add("immutable", AddOptions::new("user1").immutable(true))
        .unwrap();

    let ids = vec![r1.id.clone(), "nonexistent-id".to_string(), r2.id.clone()];
    let count = store.batch_delete_traces(&ids).unwrap();
    assert_eq!(count, 1); // Only the deletable one
}

// ── 2. Memory type (shared memory) ──

#[test]
fn test_add_memory_with_type() {
    let store = make_store(384);
    let r = store
        .add(
            "session memory",
            AddOptions::new("user1").memory_type("session"),
        )
        .unwrap();
    assert_eq!(r.memory_type.as_deref(), Some("session"));
}

#[test]
fn test_memory_type_in_get() {
    let store = make_store(384);
    let added = store
        .add(
            "shared memory",
            AddOptions::new("user1").memory_type("shared"),
        )
        .unwrap();
    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    assert_eq!(fetched.memory_type.as_deref(), Some("shared"));
}

#[test]
fn test_memory_type_default_none() {
    let store = make_store(384);
    let r = store
        .add("default memory", AddOptions::new("user1"))
        .unwrap();
    assert!(r.memory_type.is_none());
}

#[test]
fn test_filter_by_memory_type() {
    let store = make_store(384);
    store
        .add(
            "session one",
            AddOptions::new("user1").memory_type("session"),
        )
        .unwrap();
    store
        .add(
            "long term one",
            AddOptions::new("user1").memory_type("long_term"),
        )
        .unwrap();
    store
        .add("shared one", AddOptions::new("user1").memory_type("shared"))
        .unwrap();
    store.add("default one", AddOptions::new("user1")).unwrap();

    let opts = ListOptions::new("user1").filter(FilterExpression::eq("memory_type", "session"));
    let results = store.list_traces(opts).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory_type.as_deref(), Some("session"));

    let opts = ListOptions::new("user1").filter(FilterExpression::eq("memory_type", "shared"));
    let results = store.list_traces(opts).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory_type.as_deref(), Some("shared"));
}

#[test]
fn test_memory_type_in_search_results() {
    let store = make_store(384);
    store
        .add(
            "searchable session",
            AddOptions::new("user1").memory_type("session"),
        )
        .unwrap();
    store
        .add(
            "searchable shared",
            AddOptions::new("user1").memory_type("shared"),
        )
        .unwrap();

    let results = store
        .search("searchable", SearchOptions::new("user1").limit(10))
        .unwrap();
    assert_eq!(results.len(), 2);
    let types: Vec<Option<&str>> = results.iter().map(|r| r.memory_type.as_deref()).collect();
    assert!(types.contains(&Some("session")));
    assert!(types.contains(&Some("shared")));
}

#[test]
fn test_memory_type_in_list_results() {
    let store = make_store(384);
    store
        .add("lt mem", AddOptions::new("user1").memory_type("long_term"))
        .unwrap();

    let results = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory_type.as_deref(), Some("long_term"));
}

// ── 3. Webhook system ──

#[cfg(feature = "webhooks")]
mod webhook_tests {
    use super::*;

    #[test]
    fn test_webhook_config_in_memory_config() {
        use crate::webhook::{WebhookConfig as WC, WebhookEvent as WE};
        let config = MemoryConfig {
            db_path: ":memory:".into(),
            collection_name: "test".into(),
            embedding_dims: 384,
            webhooks: Some(vec![WC {
                url: "https://example.com/hook".to_string(),
                events: vec![WE::MemoryAdd],
                active: true,
            }]),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
        assert!(config.webhooks.is_some());
    }

    #[test]
    fn test_store_with_webhook_config() {
        use crate::webhook::{WebhookConfig as WC, WebhookEvent as WE};
        let config = MemoryConfig {
            db_path: ":memory:".into(),
            collection_name: "test".into(),
            embedding_dims: 384,
            webhooks: Some(vec![WC {
                url: "https://example.com/hook".to_string(),
                events: vec![WE::MemoryAdd, WE::MemoryUpdate, WE::MemoryDelete],
                active: true,
            }]),
            ..Default::default()
        };
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = MemoryStore::new(config, embedder);
        assert!(store.is_ok());
        assert!(store.unwrap().webhook_manager.is_some());
    }

    #[test]
    fn test_store_without_webhook_config() {
        let store = make_store(384);
        assert!(store.webhook_manager.is_none());
    }
}

// ════════════════════════════════════════════════
// M0-1: Memory Size Limits + Auto-Pruning
// ════════════════════════════════════════════════

fn make_store_with_limits(max_memories: usize) -> MemoryStore {
    let config = MemoryConfig {
        db_path: ":memory:".into(),
        collection_name: "test".into(),
        embedding_dims: 384,
        dedup_threshold: 0.15,
        default_limit: 10,
        max_memories_per_user: Some(max_memories),
        auto_prune: true,
        pruning_strategy: PruningStrategy::LRU,
        ..Default::default()
    };
    let embedder = Arc::new(MockEmbedder::new(384));
    MemoryStore::new(config, embedder).unwrap()
}

#[test]
fn test_count_user_memories() {
    let store = make_store(384);
    store.add("mem1", AddOptions::new("user1")).unwrap();
    store.add("mem2", AddOptions::new("user1")).unwrap();
    store.add("mem3", AddOptions::new("user2")).unwrap();
    assert_eq!(store.count_traces("user1").unwrap(), 2);
    assert_eq!(store.count_traces("user2").unwrap(), 1);
    assert_eq!(store.count_traces("user3").unwrap(), 0);
}

#[test]
fn test_db_size_bytes() {
    let store = make_store(384);
    let size = store.db_size_bytes().unwrap();
    // In-memory DB returns estimated size
    assert!(size >= 0);
}

#[test]
fn test_auto_prune_lru() {
    let store = make_store_with_limits(3);
    store.add("first memory", AddOptions::new("user1")).unwrap();
    store
        .add("second memory", AddOptions::new("user1"))
        .unwrap();
    store.add("third memory", AddOptions::new("user1")).unwrap();
    // Adding a 4th should trigger auto-prune to keep only 3
    store
        .add("fourth memory", AddOptions::new("user1"))
        .unwrap();
    let count = store.count_traces("user1").unwrap();
    assert!(
        count <= 3,
        "auto-prune should keep at most 3 memories, got {}",
        count
    );
}

#[test]
fn test_prune_importance_strategy() {
    let config = MemoryConfig {
        db_path: ":memory:".into(),
        collection_name: "test".into(),
        embedding_dims: 384,
        dedup_threshold: 0.15,
        default_limit: 10,
        pruning_strategy: PruningStrategy::Importance,
        ..Default::default()
    };
    let embedder = Arc::new(MockEmbedder::new(384));
    let store = MemoryStore::new(config, embedder).unwrap();

    store
        .add("low importance", AddOptions::new("user1").importance(0.1))
        .unwrap();
    store
        .add("high importance", AddOptions::new("user1").importance(0.9))
        .unwrap();
    store
        .add(
            "medium importance",
            AddOptions::new("user1").importance(0.5),
        )
        .unwrap();

    // Prune 1 (should remove lowest importance)
    let pruned = store.prune("user1", 1).unwrap();
    assert_eq!(pruned, 1);

    let remaining = store.list_traces(ListOptions::new("user1")).unwrap();
    assert_eq!(remaining.len(), 2);
    // The low importance one should be gone
    assert!(!remaining.iter().any(|r| r.content == "low importance"));
}

#[test]
fn test_prune_specific_count() {
    let store = make_store(384);
    for i in 0..5 {
        store
            .add(&format!("memory {i}"), AddOptions::new("user1"))
            .unwrap();
    }
    assert_eq!(store.count_traces("user1").unwrap(), 5);

    let pruned = store.prune("user1", 2).unwrap();
    assert_eq!(pruned, 2);
    assert_eq!(store.count_traces("user1").unwrap(), 3);
}

#[test]
fn test_prune_zero_count() {
    let store = make_store(384);
    store.add("mem1", AddOptions::new("user1")).unwrap();
    let pruned = store.prune("user1", 0).unwrap();
    assert_eq!(pruned, 0);
    assert_eq!(store.count_traces("user1").unwrap(), 1);
}

// ════════════════════════════════════════════════
// M0-2: Privacy Controls
// ════════════════════════════════════════════════

#[test]
fn test_add_with_privacy_local_only() {
    let store = make_store(384);
    let opts = AddOptions::new("user1").privacy(Privacy::LocalOnly);
    let result = store.add("secret local note", opts).unwrap();
    assert_eq!(result.privacy, "local_only");
}

#[test]
fn test_add_with_privacy_syncable() {
    let store = make_store(384);
    let opts = AddOptions::new("user1").privacy(Privacy::Syncable);
    let result = store.add("syncable note", opts).unwrap();
    assert_eq!(result.privacy, "syncable");
}

#[test]
fn test_add_with_privacy_encrypted_sync() {
    let store = make_store(384);
    let opts = AddOptions::new("user1").privacy(Privacy::EncryptedSync);
    let result = store.add("encrypted note", opts).unwrap();
    assert_eq!(result.privacy, "encrypted_sync");
}

#[test]
fn test_default_privacy_is_syncable() {
    let store = make_store(384);
    let result = store
        .add("default privacy", AddOptions::new("user1"))
        .unwrap();
    assert_eq!(result.privacy, "syncable");
}

#[test]
fn test_privacy_in_get_result() {
    let store = make_store(384);
    let opts = AddOptions::new("user1").privacy(Privacy::LocalOnly);
    let added = store.add("private note", opts).unwrap();
    let fetched = store.get_trace(&added.id).unwrap().unwrap();
    assert_eq!(fetched.privacy, "local_only");
}

#[test]
fn test_export_skips_local_only() {
    let store = make_store(384);
    store
        .add(
            "public note",
            AddOptions::new("user1").privacy(Privacy::Syncable),
        )
        .unwrap();
    store
        .add(
            "private note",
            AddOptions::new("user1").privacy(Privacy::LocalOnly),
        )
        .unwrap();
    store
        .add(
            "encrypted note",
            AddOptions::new("user1").privacy(Privacy::EncryptedSync),
        )
        .unwrap();

    // Default export should skip local_only
    let exported = store.export(Some("user1")).unwrap();
    assert_eq!(exported.len(), 2, "export should skip local_only memories");

    // Export with include_local should include all
    let exported_all = store.export_with_privacy(Some("user1"), true).unwrap();
    assert_eq!(
        exported_all.len(),
        3,
        "export with include_local should include all"
    );
}

#[test]
fn test_privacy_string_conversion() {
    assert_eq!(Privacy::LocalOnly.as_str(), "local_only");
    assert_eq!(Privacy::Syncable.as_str(), "syncable");
    assert_eq!(Privacy::EncryptedSync.as_str(), "encrypted_sync");

    assert_eq!(Privacy::parse("local_only"), Privacy::LocalOnly);
    assert_eq!(Privacy::parse("syncable"), Privacy::Syncable);
    assert_eq!(Privacy::parse("encrypted_sync"), Privacy::EncryptedSync);
    assert_eq!(Privacy::parse("unknown"), Privacy::Syncable);
}

// ════════════════════════════════════════════════
// M0-3: Battery-Aware Processing
// ════════════════════════════════════════════════

fn make_store_with_power() -> MemoryStore {
    let config = MemoryConfig {
        db_path: ":memory:".into(),
        collection_name: "test".into(),
        embedding_dims: 384,
        dedup_threshold: 0.15,
        default_limit: 10,
        power_config: Some(crate::config::PowerConfig {
            full_power_threshold: 0.5,
            power_save_threshold: 0.2,
            defer_when_critical: true,
        }),
        ..Default::default()
    };
    let embedder = Arc::new(MockEmbedder::new(384));
    MemoryStore::new(config, embedder).unwrap()
}

#[test]
fn test_set_battery_level() {
    let store = make_store_with_power();
    store.set_battery_level(0.8, false);
    assert!(!store.is_power_save());
    assert!(!store.is_critical_power());
}

#[test]
fn test_power_save_mode() {
    let store = make_store_with_power();
    store.set_battery_level(0.3, false);
    assert!(store.is_power_save()); // 0.3 < 0.5 threshold
    assert!(!store.is_critical_power()); // 0.3 >= 0.2 threshold
}

#[test]
fn test_critical_power_mode() {
    let store = make_store_with_power();
    store.set_battery_level(0.1, false);
    assert!(store.is_power_save());
    assert!(store.is_critical_power()); // 0.1 < 0.2 threshold
}

#[test]
fn test_charging_overrides_power_save() {
    let store = make_store_with_power();
    store.set_battery_level(0.1, true); // low battery but charging
    assert!(!store.is_power_save());
    assert!(!store.is_critical_power());
}

#[test]
fn test_no_power_config_means_no_power_save() {
    let store = make_store(384); // no power_config
    store.set_battery_level(0.01, false);
    assert!(!store.is_power_save());
    assert!(!store.is_critical_power());
}

// ════════════════════════════════════════════════
// M0-4: Deferred Operation Queue
// ════════════════════════════════════════════════

#[test]
fn test_defer_on_critical_battery() {
    let store = make_store_with_power();
    store.set_battery_level(0.1, false); // critical

    let result = store
        .add("deferred memory", AddOptions::new("user1"))
        .unwrap();
    assert_eq!(result.id, "deferred");
    assert_eq!(store.deferred_count(), 1);

    // Memory was NOT actually stored
    assert_eq!(store.count_traces("user1").unwrap(), 0);
}

#[test]
fn test_process_deferred_ops() {
    let store = make_store_with_power();
    store.set_battery_level(0.1, false); // critical

    store.add("deferred1", AddOptions::new("user1")).unwrap();
    store.add("deferred2", AddOptions::new("user1")).unwrap();
    assert_eq!(store.deferred_count(), 2);
    assert_eq!(store.count_traces("user1").unwrap(), 0);

    // "Plug in" the charger
    store.set_battery_level(0.8, true);
    let processed = store.process_deferred().unwrap();
    assert_eq!(processed, 2);
    assert_eq!(store.deferred_count(), 0);
    assert_eq!(store.count_traces("user1").unwrap(), 2);
}

#[test]
fn test_deferred_count_empty() {
    let store = make_store(384);
    assert_eq!(store.deferred_count(), 0);
}

// ════════════════════════════════════════════════
// M0-5: Procedural Memory
// ════════════════════════════════════════════════

#[test]
fn test_add_procedure() {
    let store = make_store(384);
    let steps = vec![
        crate::procedural::ProcedureStep {
            order: 1,
            action: "open_editor".into(),
            parameters: None,
        },
        crate::procedural::ProcedureStep {
            order: 2,
            action: "write_code".into(),
            parameters: Some(serde_json::json!({"lang": "rust"})),
        },
    ];
    let proc = store
        .add_procedure("Code Review", "Review PR code", steps, "user1")
        .unwrap();
    assert!(!proc.id.is_empty());
    assert_eq!(proc.name, "Code Review");
    assert_eq!(proc.description, "Review PR code");
    assert_eq!(proc.steps.len(), 2);
    assert_eq!(proc.user_id, "user1");
    assert_eq!(proc.confidence, 0.5);
    assert_eq!(proc.usage_count, 0);
}

#[test]
fn test_get_procedure() {
    let store = make_store(384);
    let steps = vec![crate::procedural::ProcedureStep {
        order: 1,
        action: "test".into(),
        parameters: None,
    }];
    let added = store
        .add_procedure("Test Proc", "A test", steps, "user1")
        .unwrap();

    let fetched = store.get_procedure(&added.id).unwrap();
    assert!(fetched.is_some());
    let p = fetched.unwrap();
    assert_eq!(p.name, "Test Proc");
}

#[test]
fn test_get_nonexistent_procedure() {
    let store = make_store(384);
    let result = store.get_procedure("nonexistent").unwrap();
    assert!(result.is_none());
}

#[test]
fn test_list_procedures_by_user() {
    let store = make_store(384);
    let steps = vec![];
    store
        .add_procedure("Proc1", "First", steps.clone(), "user1")
        .unwrap();
    store
        .add_procedure("Proc2", "Second", steps.clone(), "user1")
        .unwrap();
    store
        .add_procedure("Proc3", "Third", steps, "user2")
        .unwrap();

    let user1_procs = store.list_procedures("user1").unwrap();
    assert_eq!(user1_procs.len(), 2);

    let user2_procs = store.list_procedures("user2").unwrap();
    assert_eq!(user2_procs.len(), 1);
    assert_eq!(user2_procs[0].name, "Proc3");
}

#[test]
fn test_delete_procedure() {
    let store = make_store(384);
    let added = store
        .add_procedure("ToDelete", "Will be deleted", vec![], "user1")
        .unwrap();
    store.delete_procedure(&added.id).unwrap();
    assert!(store.get_procedure(&added.id).unwrap().is_none());
}

// ════════════════════════════════════════════════
// M0-6: Vision Message Support
// ════════════════════════════════════════════════

#[test]
fn test_chat_message_with_image_url() {
    let msg = ChatMessage {
        role: "user".into(),
        content: "What is this?".into(),
        image_url: Some("https://example.com/image.png".into()),
        image_type: Some("url".into()),
        timestamp: None,
    };
    assert_eq!(
        msg.image_url.as_deref(),
        Some("https://example.com/image.png")
    );
    assert_eq!(msg.image_type.as_deref(), Some("url"));
}

#[test]
fn test_chat_message_with_base64_image() {
    let msg = ChatMessage {
        role: "user".into(),
        content: "Analyze this image".into(),
        image_url: Some("data:image/png;base64,iVBOR...".into()),
        image_type: Some("base64".into()),
        timestamp: None,
    };
    assert_eq!(msg.image_type.as_deref(), Some("base64"));
}

#[test]
fn test_chat_message_without_image() {
    let msg = ChatMessage {
        role: "assistant".into(),
        content: "Hello!".into(),
        image_url: None,
        image_type: None,
        timestamp: None,
    };
    assert!(msg.image_url.is_none());
    assert!(msg.image_type.is_none());
}

#[test]
fn test_chat_message_serialization() {
    let msg = ChatMessage {
        role: "user".into(),
        content: "Hello".into(),
        image_url: Some("http://img.png".into()),
        image_type: Some("url".into()),
        timestamp: None,
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("image_url"));
    let deserialized: ChatMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.image_url.as_deref(), Some("http://img.png"));
}

// ════════════════════════════════════════════════
// Additional edge cases
// ════════════════════════════════════════════════

#[test]
fn test_pruning_strategy_default() {
    let strategy = PruningStrategy::default();
    assert_eq!(strategy, PruningStrategy::LRU);
}

#[test]
fn test_privacy_default() {
    let privacy = Privacy::default();
    assert_eq!(privacy, Privacy::Syncable);
}

#[test]
fn test_add_options_privacy_builder() {
    let opts = AddOptions::new("user1")
        .privacy(Privacy::LocalOnly)
        .importance(0.9);
    assert_eq!(opts.privacy, Privacy::LocalOnly);
    assert_eq!(opts.importance, Some(0.9));
}

// ═══════════════════════════════════════════════
//  Sync primitives tests
// ═══════════════════════════════════════════════

#[test]
fn test_current_sync_version_initial() {
    let store = make_store(384);
    let v = store.current_sync_version().unwrap();
    assert_eq!(v, 0);
}

#[test]
fn test_sync_version_increments_on_add() {
    let store = make_store(384);
    store.add("first memory", AddOptions::new("user1")).unwrap();
    let v1 = store.current_sync_version().unwrap();
    assert!(v1 >= 1, "sync version should be >= 1 after add, got {}", v1);

    store
        .add("second memory", AddOptions::new("user1"))
        .unwrap();
    let v2 = store.current_sync_version().unwrap();
    assert!(v2 > v1, "sync version should increase after second add");
}

#[test]
fn test_export_changes_since_empty() {
    let store = make_store(384);
    let delta = store.export_changes_since(0, "device-a").unwrap();
    assert!(delta.changes.is_empty());
    assert_eq!(delta.from_version, 0);
    assert_eq!(delta.to_version, 0);
    assert_eq!(delta.device_id, "device-a");
}

#[test]
fn test_export_changes_since_with_data() {
    let store = make_store(384);
    store.add("hello world", AddOptions::new("user1")).unwrap();
    store
        .add("goodbye world", AddOptions::new("user1"))
        .unwrap();

    let delta = store.export_changes_since(0, "device-a").unwrap();
    assert_eq!(delta.changes.len(), 2);
    assert!(delta.to_version >= 2);
    assert!(!delta.exported_at.is_empty());
}

#[test]
fn test_export_changes_since_partial() {
    let store = make_store(384);
    store.add("first", AddOptions::new("user1")).unwrap();
    let v1 = store.current_sync_version().unwrap();

    store.add("second", AddOptions::new("user1")).unwrap();
    let delta = store.export_changes_since(v1, "device-a").unwrap();
    assert_eq!(delta.changes.len(), 1);
    assert_eq!(delta.changes[0].content.as_deref(), Some("second"));
}

#[test]
fn test_storage_stats() {
    let store = make_store(384);
    store.add("hello", AddOptions::new("user1")).unwrap();
    store.add("world", AddOptions::new("user1")).unwrap();

    let stats = store.storage_stats().unwrap();
    assert_eq!(stats.total_memories, 2);
    assert_eq!(stats.total_entities, 0);
    assert_eq!(stats.total_relationships, 0);
    assert_eq!(stats.embedding_dims, 384);
}

#[test]
fn test_storage_stats_empty() {
    let store = make_store(384);
    let stats = store.storage_stats().unwrap();
    assert_eq!(stats.total_memories, 0);
    assert_eq!(stats.embedding_dims, 384);
}

#[test]
fn test_sync_dedup_update_bumps_version() {
    let store = make_store(384);
    store.add("hello world", AddOptions::new("user1")).unwrap();
    let v1 = store.current_sync_version().unwrap();

    // Add same content again (dedup will update, not insert)
    store.add("hello world", AddOptions::new("user1")).unwrap();
    let v2 = store.current_sync_version().unwrap();
    assert!(v2 > v1, "sync version should increase on dedup update");
}

//! JSON → MemMe API helpers for binding crates (dora, mcp, server).
//!
//! Provides common JSON-to-options conversion used by multiple integration layers.

use crate::memory::MemoryStore;
use crate::types::*;

/// Add a memory from a JSON object. Expects `content` and `user_id` fields.
pub fn add_from_json(
    json: &serde_json::Value,
    store: &MemoryStore,
) -> Result<MemoryResult, String> {
    let content = json["content"].as_str().ok_or("'content' field required")?;
    let user_id = json["user_id"].as_str().ok_or("'user_id' field required")?;

    let mut opts = AddOptions::new(user_id);
    if let Some(aid) = json.get("agent_id").and_then(|v| v.as_str()) {
        opts = opts.agent_id(aid);
    }
    if let Some(m) = json.get("metadata") {
        if !m.is_null() {
            opts = opts.metadata(m.clone());
        }
    }

    store.add(content, opts).map_err(|e| e.to_string())
}

/// Search memories from a JSON object. Expects `query` and `user_id` fields.
pub fn search_from_json(
    json: &serde_json::Value,
    store: &MemoryStore,
) -> Result<Vec<MemoryResult>, String> {
    let query = json["query"].as_str().ok_or("'query' field required")?;
    let user_id = json["user_id"].as_str().ok_or("'user_id' field required")?;
    let limit = json.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

    let opts = SearchOptions::new(user_id).limit(limit);
    store.search(query, opts).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MemoryConfig;
    use std::sync::Arc;

    fn test_store() -> MemoryStore {
        let config = MemoryConfig {
            db_path: ":memory:".into(),
            embedding_dims: 384,
            ..Default::default()
        };
        let embedder: Arc<dyn memme_embeddings::Embedder> =
            Arc::new(memme_embeddings::mock::MockEmbedder::new(384));
        MemoryStore::new(config, embedder).unwrap()
    }

    #[test]
    fn test_add_from_json() {
        let store = test_store();
        let json = serde_json::json!({
            "content": "test memory",
            "user_id": "u1",
            "agent_id": "a1",
        });
        let result = add_from_json(&json, &store).unwrap();
        assert_eq!(result.content, "test memory");
    }

    #[test]
    fn test_add_from_json_missing_content() {
        let store = test_store();
        let json = serde_json::json!({"user_id": "u1"});
        let err = add_from_json(&json, &store).unwrap_err();
        assert!(err.contains("content"));
    }

    #[test]
    fn test_search_from_json() {
        let store = test_store();
        add_from_json(
            &serde_json::json!({"content": "hello world", "user_id": "u1"}),
            &store,
        )
        .unwrap();

        let json = serde_json::json!({
            "query": "hello",
            "user_id": "u1",
            "limit": 3
        });
        let results = search_from_json(&json, &store).unwrap();
        assert!(!results.is_empty());
    }
}

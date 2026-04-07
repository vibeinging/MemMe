//! MemMe Dora Node — handler logic (testable without dora runtime).

use eyre::{Context, Result};

use memme_core::json_ops;
use memme_core::memory::MemoryStore;
use memme_core::types::*;

/// Dispatch an input to the appropriate handler. Returns (output_id, json_payload).
pub fn dispatch(input_id: &str, json: &str, store: &MemoryStore) -> Result<(&'static str, String)> {
    match input_id {
        "store" => handle_store(json, store),
        "search" => handle_search(json, store),
        "session_event" => handle_session_event(json, store),
        "compact" => handle_compact(json, store),
        _ => Err(eyre::eyre!("unknown input: {input_id}")),
    }
}

pub fn handle_store(json: &str, store: &MemoryStore) -> Result<(&'static str, String)> {
    let v: serde_json::Value = serde_json::from_str(json).wrap_err("invalid JSON for store")?;
    let result = json_ops::add_from_json(&v, store).map_err(|e| eyre::eyre!(e))?;
    let payload = serde_json::to_string(&result)?;
    Ok(("stored", payload))
}

pub fn handle_search(json: &str, store: &MemoryStore) -> Result<(&'static str, String)> {
    let v: serde_json::Value = serde_json::from_str(json).wrap_err("invalid JSON for search")?;
    let results = json_ops::search_from_json(&v, store).map_err(|e| eyre::eyre!(e))?;
    let payload = serde_json::to_string(&results)?;
    Ok(("results", payload))
}

pub fn handle_session_event(json: &str, store: &MemoryStore) -> Result<(&'static str, String)> {
    let v: serde_json::Value =
        serde_json::from_str(json).wrap_err("invalid JSON for session_event")?;

    let content = v["content"]
        .as_str()
        .ok_or_else(|| eyre::eyre!("'content' field required"))?;
    let user_id = v["user_id"]
        .as_str()
        .ok_or_else(|| eyre::eyre!("'user_id' field required"))?;

    let mut opts = IngestEventOptions::new(user_id);
    if let Some(sid) = v.get("session_id").and_then(|v| v.as_str()) {
        opts = opts.session_id(sid);
    }
    if let Some(et) = v.get("event_type").and_then(|v| v.as_str()) {
        match et {
            "user_message" | "ai_response" | "tool_call" | "tool_result" | "error" | "system" => {
                opts = opts.event_type(et);
            }
            _ => {
                return Err(eyre::eyre!(
                    "invalid event_type '{et}'. Use: user_message, ai_response, tool_call, tool_result, error, system"
                ));
            }
        }
    }
    if let Some(m) = v.get("metadata") {
        if !m.is_null() {
            opts = opts.metadata(m.clone());
        }
    }

    let event = store.ingest_event(content, opts)?;
    let payload = serde_json::to_string(&event)?;
    Ok(("event_ack", payload))
}

pub fn handle_compact(json: &str, store: &MemoryStore) -> Result<(&'static str, String)> {
    let v: serde_json::Value = serde_json::from_str(json).wrap_err("invalid JSON for compact")?;

    let session_id = v["session_id"]
        .as_str()
        .ok_or_else(|| eyre::eyre!("'session_id' field required"))?;

    let result = store.compact(session_id)?;
    let payload = serde_json::to_string(&result)?;
    Ok(("compacted", payload))
}

/// Create a MemoryStore with MockEmbedder for testing (no API key needed).
#[cfg(test)]
pub fn test_store() -> MemoryStore {
    use std::sync::Arc;
    let config = memme_core::config::MemoryConfig {
        db_path: ":memory:".into(),
        embedding_dims: 384,
        ..Default::default()
    };
    let embedder: Arc<dyn memme_embeddings::Embedder> =
        Arc::new(memme_embeddings::mock::MockEmbedder::new(384));
    MemoryStore::new(config, embedder).expect("failed to create test store")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> MemoryStore {
        test_store()
    }

    #[test]
    fn test_store_and_search() {
        let s = store();

        // Store a memory
        let (out, payload) = handle_store(
            r#"{"content": "The red cup is on the kitchen table", "user_id": "robot-1"}"#,
            &s,
        )
        .unwrap();
        assert_eq!(out, "stored");
        let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert!(v["id"].is_string());
        assert_eq!(v["content"], "The red cup is on the kitchen table");

        // Search
        let (out, payload) = handle_search(
            r#"{"query": "where is the cup", "user_id": "robot-1", "top_k": 3}"#,
            &s,
        )
        .unwrap();
        assert_eq!(out, "results");
        let results: Vec<serde_json::Value> = serde_json::from_str(&payload).unwrap();
        assert!(!results.is_empty());
        assert!(results[0]["content"].as_str().unwrap().contains("red cup"));
    }

    #[test]
    fn test_store_with_metadata() {
        let s = store();
        let (out, payload) = handle_store(
            r#"{
                "content": "Picked up blue box",
                "user_id": "robot-1",
                "agent_id": "arm-1",
                "metadata": {"task": "pick-and-place", "success": true}
            }"#,
            &s,
        )
        .unwrap();
        assert_eq!(out, "stored");
        let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(v["user_id"], "robot-1");
    }

    #[test]
    fn test_session_event_and_compact() {
        let s = store();

        // Ingest events
        let (out, payload) = handle_session_event(
            r#"{
                "content": "User said: please get me the red cup",
                "user_id": "robot-1",
                "session_id": "session-001",
                "event_type": "user_message"
            }"#,
            &s,
        )
        .unwrap();
        assert_eq!(out, "event_ack");
        let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert!(v["event_id"].is_string());

        // Ingest more events
        handle_session_event(
            r#"{
                "content": "Moving to kitchen table to find red cup",
                "user_id": "robot-1",
                "session_id": "session-001",
                "event_type": "ai_response"
            }"#,
            &s,
        )
        .unwrap();

        // Compact — will fail without LLM but should not panic
        let result = handle_compact(r#"{"session_id": "session-001"}"#, &s);
        // Without LLM this may error, but it should be a clean error
        match result {
            Ok((out, _)) => assert_eq!(out, "compacted"),
            Err(e) => {
                // Expected: LLM required for compact
                let msg = e.to_string();
                assert!(
                    msg.contains("LLM") || msg.contains("llm") || msg.contains("provider"),
                    "unexpected error: {msg}"
                );
            }
        }
    }

    #[test]
    fn test_dispatch_routing() {
        let s = store();

        // Valid routes
        let (out, _) = dispatch("store", r#"{"content": "test", "user_id": "u1"}"#, &s).unwrap();
        assert_eq!(out, "stored");

        // Unknown route
        let err = dispatch("unknown", "{}", &s);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("unknown input"));
    }

    #[test]
    fn test_invalid_json() {
        let s = store();
        let err = handle_store("not json", &s);
        assert!(err.is_err());
    }

    #[test]
    fn test_missing_required_fields() {
        let s = store();

        // Missing content
        let err = handle_store(r#"{"user_id": "u1"}"#, &s);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("content"));

        // Missing user_id
        let err = handle_store(r#"{"content": "test"}"#, &s);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("user_id"));

        // Missing query in search
        let err = handle_search(r#"{"user_id": "u1"}"#, &s);
        assert!(err.is_err());
    }

    #[test]
    fn test_multiple_store_and_search() {
        let s = store();

        handle_store(
            r#"{"content": "Robot arm calibrated at position zero", "user_id": "robot-1"}"#,
            &s,
        )
        .unwrap();
        handle_store(
            r#"{"content": "Battery level is at 85%", "user_id": "robot-1"}"#,
            &s,
        )
        .unwrap();
        handle_store(
            r#"{"content": "Obstacle detected at coordinates (2,3)", "user_id": "robot-1"}"#,
            &s,
        )
        .unwrap();

        let (_, payload) = handle_search(
            r#"{"query": "battery", "user_id": "robot-1", "top_k": 2}"#,
            &s,
        )
        .unwrap();
        let results: Vec<serde_json::Value> = serde_json::from_str(&payload).unwrap();
        assert!(results.len() <= 2);
    }
}

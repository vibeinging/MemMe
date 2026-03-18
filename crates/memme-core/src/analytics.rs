//! OLAP analytics powered by DuckDB's analytical engine.
//! Provides insights into memory usage patterns.

use serde::{Deserialize, Serialize};

/// Summary statistics for a user's memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserStats {
    pub user_id: String,
    pub total_memories: u64,
    pub total_entities: u64,
    pub total_relationships: u64,
    pub earliest_memory: Option<String>,
    pub latest_memory: Option<String>,
    pub unique_agents: u64,
}

/// Memory count per time period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeBucket {
    /// e.g., "2026-03", "2026-03-17"
    pub period: String,
    pub count: u64,
}

/// History event count by type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventCount {
    /// "ADD", "UPDATE", "DELETE"
    pub event: String,
    pub count: u64,
}

/// Top entities by relationship count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityStat {
    pub name: String,
    pub entity_type: Option<String>,
    pub relationship_count: u64,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use memme_embeddings::mock::MockEmbedder;

    use crate::config::MemoryConfig;
    use crate::memory::MemoryStore;
    use crate::types::AddOptions;

    fn make_store() -> MemoryStore {
        let config = MemoryConfig {
            db_path: ":memory:".into(),
            collection_name: "test".into(),
            embedding_dims: 384,
            dedup_threshold: 0.15,
            default_limit: 10,
            ..Default::default()
        };
        let embedder = Arc::new(MockEmbedder::new(384));
        MemoryStore::new(config, embedder).unwrap()
    }

    #[test]
    fn test_user_stats_empty() {
        let store = make_store();
        let stats = store.user_stats("nonexistent_user").unwrap();
        assert_eq!(stats.total_memories, 0);
        assert_eq!(stats.total_entities, 0);
        assert_eq!(stats.total_relationships, 0);
        assert!(stats.earliest_memory.is_none());
        assert!(stats.latest_memory.is_none());
        assert_eq!(stats.unique_agents, 0);
    }

    #[test]
    fn test_user_stats_with_data() {
        let store = make_store();

        // Add some memories
        store
            .add("memory one", AddOptions::new("user1").agent_id("agent_a"))
            .unwrap();
        store
            .add("memory two", AddOptions::new("user1").agent_id("agent_b"))
            .unwrap();
        store
            .add("memory three", AddOptions::new("user1").agent_id("agent_a"))
            .unwrap();

        let stats = store.user_stats("user1").unwrap();
        assert_eq!(stats.total_memories, 3);
        assert_eq!(stats.unique_agents, 2);
        assert!(stats.earliest_memory.is_some());
        assert!(stats.latest_memory.is_some());
    }

    #[test]
    fn test_memory_frequency_by_day() {
        let store = make_store();

        store.add("memory one", AddOptions::new("user1")).unwrap();
        store.add("memory two", AddOptions::new("user1")).unwrap();
        store.add("memory three", AddOptions::new("user1")).unwrap();

        let buckets = store.memory_frequency("user1", "day", 10).unwrap();
        // All memories were added on the same day, so there should be one bucket
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].count, 3);

        // Test invalid granularity
        let result = store.memory_frequency("user1", "hour", 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_event_distribution() {
        let store = make_store();

        // Add two memories (generates ADD events)
        let m1 = store.add("content one", AddOptions::new("user1")).unwrap();
        let _m2 = store.add("content two", AddOptions::new("user1")).unwrap();

        // Update m1 (generates UPDATE event)
        store
            .update_trace(&m1.id, "content one updated", None)
            .unwrap();

        // Note: delete removes the memory row, so history events for
        // deleted memories won't appear in the distribution (because
        // the JOIN with memories has no match). We test with existing
        // memories only.
        let events = store.event_distribution("user1").unwrap();

        let add_count: u64 = events
            .iter()
            .filter(|e| e.event == "ADD")
            .map(|e| e.count)
            .sum();
        let update_count: u64 = events
            .iter()
            .filter(|e| e.event == "UPDATE")
            .map(|e| e.count)
            .sum();

        assert_eq!(add_count, 2);
        assert_eq!(update_count, 1);
    }

    #[test]
    fn test_top_entities() {
        let store = make_store();

        // Manually insert entities and relationships to test ordering
        let storage = &store.storage();

        // Insert entities
        storage
            .upsert_entity("e1", "Alice", Some("person"), "user1")
            .unwrap();
        storage
            .upsert_entity("e2", "Bob", Some("person"), "user1")
            .unwrap();
        storage
            .upsert_entity("e3", "Rust", Some("language"), "user1")
            .unwrap();

        // Alice has 3 relationships, Bob has 1, Rust has 2
        storage
            .insert_relationship("r1", "e1", "e2", "knows", "user1")
            .unwrap();
        storage
            .insert_relationship("r2", "e1", "e3", "uses", "user1")
            .unwrap();
        storage
            .insert_relationship("r3", "e3", "e1", "used_by", "user1")
            .unwrap();

        let top = store.top_entities("user1", 10).unwrap();
        assert!(!top.is_empty());
        // Alice should be first (3 relationships: r1, r2, r3)
        assert_eq!(top[0].name, "Alice");
        assert_eq!(top[0].relationship_count, 3);
    }

    #[test]
    fn test_user_stats_isolation() {
        let store = make_store();

        store
            .add("user1 memory", AddOptions::new("user1").agent_id("agent_x"))
            .unwrap();
        store
            .add("user2 memory a", AddOptions::new("user2"))
            .unwrap();
        store
            .add("user2 memory b", AddOptions::new("user2"))
            .unwrap();

        let stats1 = store.user_stats("user1").unwrap();
        let stats2 = store.user_stats("user2").unwrap();

        assert_eq!(stats1.total_memories, 1);
        assert_eq!(stats1.unique_agents, 1);
        assert_eq!(stats2.total_memories, 2);
        assert_eq!(stats2.unique_agents, 0); // no agent_id set for user2
    }
}

//! Sync primitives for cross-device memory synchronization.
//!
//! This module defines data types for change tracking, delta export,
//! and storage statistics. It does NOT implement a full sync protocol;
//! instead it provides the building blocks that higher layers can use
//! to build CRDTs, operational transforms, or simple last-writer-wins
//! strategies.

use serde::{Deserialize, Serialize};

/// A single change to a memory, used for change-log based sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncChange {
    /// ID of the affected memory.
    pub memory_id: String,
    /// What happened: Create, Update, or Delete.
    pub operation: SyncOperation,
    /// Memory content (present for Create/Update, absent for Delete).
    pub content: Option<String>,
    /// Associated metadata JSON.
    pub metadata: Option<serde_json::Value>,
    /// ISO 8601 timestamp of the change.
    pub timestamp: String,
    /// Identifier of the device that originated this change.
    pub device_id: String,
    /// Monotonically increasing version number for ordering.
    pub sync_version: u64,
}

/// The type of sync operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SyncOperation {
    Create,
    Update,
    Delete,
}

/// A batch of changes between two version numbers, suitable for
/// shipping to another device or cloud endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncDelta {
    /// Device ID of the exporter.
    pub device_id: String,
    /// Start version (exclusive).
    pub from_version: u64,
    /// End version (inclusive).
    pub to_version: u64,
    /// Changes in version order.
    pub changes: Vec<SyncChange>,
    /// ISO 8601 timestamp when this delta was exported.
    pub exported_at: String,
}

/// High-level statistics about the storage layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Total number of memories.
    pub total_memories: u64,
    /// Total number of entities in the knowledge graph.
    pub total_entities: u64,
    /// Total number of relationships in the knowledge graph.
    pub total_relationships: u64,
    /// Estimated storage size in bytes.
    pub estimated_size_bytes: u64,
    /// Embedding dimensionality configured.
    pub embedding_dims: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_change_serialization() {
        let change = SyncChange {
            memory_id: "mem-1".into(),
            operation: SyncOperation::Create,
            content: Some("hello world".into()),
            metadata: Some(serde_json::json!({"source": "test"})),
            timestamp: "2026-03-18T00:00:00Z".into(),
            device_id: "device-a".into(),
            sync_version: 1,
        };
        let json = serde_json::to_string(&change).unwrap();
        let parsed: SyncChange = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.memory_id, "mem-1");
        assert_eq!(parsed.operation, SyncOperation::Create);
        assert_eq!(parsed.sync_version, 1);
    }

    #[test]
    fn test_sync_delta_serialization() {
        let delta = SyncDelta {
            device_id: "device-a".into(),
            from_version: 0,
            to_version: 5,
            changes: vec![
                SyncChange {
                    memory_id: "mem-1".into(),
                    operation: SyncOperation::Create,
                    content: Some("hello".into()),
                    metadata: None,
                    timestamp: "2026-03-18T00:00:00Z".into(),
                    device_id: "device-a".into(),
                    sync_version: 1,
                },
                SyncChange {
                    memory_id: "mem-1".into(),
                    operation: SyncOperation::Update,
                    content: Some("hello updated".into()),
                    metadata: None,
                    timestamp: "2026-03-18T01:00:00Z".into(),
                    device_id: "device-a".into(),
                    sync_version: 2,
                },
            ],
            exported_at: "2026-03-18T02:00:00Z".into(),
        };
        let json = serde_json::to_string(&delta).unwrap();
        let parsed: SyncDelta = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.changes.len(), 2);
        assert_eq!(parsed.from_version, 0);
        assert_eq!(parsed.to_version, 5);
    }

    #[test]
    fn test_sync_operation_equality() {
        assert_eq!(SyncOperation::Create, SyncOperation::Create);
        assert_ne!(SyncOperation::Create, SyncOperation::Update);
        assert_ne!(SyncOperation::Update, SyncOperation::Delete);
    }

    #[test]
    fn test_storage_stats_serialization() {
        let stats = StorageStats {
            total_memories: 100,
            total_entities: 50,
            total_relationships: 30,
            estimated_size_bytes: 1024000,
            embedding_dims: 384,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let parsed: StorageStats = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_memories, 100);
        assert_eq!(parsed.total_entities, 50);
        assert_eq!(parsed.total_relationships, 30);
        assert_eq!(parsed.estimated_size_bytes, 1024000);
        assert_eq!(parsed.embedding_dims, 384);
    }

    #[test]
    fn test_sync_change_delete_no_content() {
        let change = SyncChange {
            memory_id: "mem-2".into(),
            operation: SyncOperation::Delete,
            content: None,
            metadata: None,
            timestamp: "2026-03-18T03:00:00Z".into(),
            device_id: "device-b".into(),
            sync_version: 10,
        };
        assert!(change.content.is_none());
        assert_eq!(change.operation, SyncOperation::Delete);
    }

    #[test]
    fn test_sync_delta_empty_changes() {
        let delta = SyncDelta {
            device_id: "device-a".into(),
            from_version: 5,
            to_version: 5,
            changes: vec![],
            exported_at: "2026-03-18T04:00:00Z".into(),
        };
        assert!(delta.changes.is_empty());
        assert_eq!(delta.from_version, delta.to_version);
    }
}

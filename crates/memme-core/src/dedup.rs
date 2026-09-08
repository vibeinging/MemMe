use tracing::debug;

use crate::error::Result;
use crate::storage::Storage;

/// Result of a dedup check.
pub(crate) enum DedupResult {
    /// No duplicate found — should insert as new memory.
    New,
    /// Found an existing memory within the threshold — should update it.
    Duplicate {
        existing_id: String,
        existing_content: String,
    },
}

/// Search for a duplicate memory, first by exact content hash, then by vector
/// similarity within the given threshold.
///
/// Returns `DedupResult::Duplicate` if any existing memory for the same user
/// (and agent, when provided) matches by hash or has cosine distance < `threshold`
/// to the new embedding.
#[allow(clippy::collapsible_if)]
pub(crate) fn check_dedup(
    storage: &Storage,
    embedding: &[f32],
    hash: &str,
    content: &str,
    user_id: &str,
    agent_id: Option<&str>,
    threshold: f32,
) -> Result<DedupResult> {
    // Fast path: exact content hash match
    if let Some((id, existing_content)) =
        storage.find_by_hash(hash, user_id, agent_id, None, None)?
    {
        // Verify content actually matches (guard against hash collisions)
        if existing_content == content {
            debug!(existing_id = %id, "Dedup check: exact hash match");
            return Ok(DedupResult::Duplicate {
                existing_id: id,
                existing_content,
            });
        }
        // Hash collision — fall through to vector search
        debug!("Hash collision detected, falling through to vector search");
    }

    // Slow path: vector similarity search for top-1 closest memory
    let results =
        storage.vector_search(embedding, user_id, agent_id, false, None, None, None, 1)?;

    if let Some(closest) = results.first() {
        if let Some(distance) = closest.score {
            debug!(
                distance,
                threshold,
                existing_id = %closest.id,
                "Dedup check: vector similarity"
            );
            if distance < threshold {
                return Ok(DedupResult::Duplicate {
                    existing_id: closest.id.clone(),
                    existing_content: closest.content.clone(),
                });
            }
        }
    }

    Ok(DedupResult::New)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MemoryConfig;
    use crate::storage::InsertMemoryParams;

    fn test_config(dims: usize) -> MemoryConfig {
        MemoryConfig::new(":memory:", dims)
    }

    #[test]
    fn test_empty_store_returns_new() {
        let storage = Storage::open(test_config(4)).unwrap();
        let emb = vec![1.0, 0.0, 0.0, 0.0];
        let result = check_dedup(
            &storage,
            &emb,
            "somehash",
            "some content",
            "user1",
            None,
            0.15,
        )
        .unwrap();
        assert!(matches!(result, DedupResult::New));
    }

    #[test]
    fn test_hash_exact_match_returns_duplicate() {
        let storage = Storage::open(test_config(4)).unwrap();
        let emb = vec![1.0, 0.0, 0.0, 0.0];
        storage
            .insert_memory(
                "existing1",
                "hello world",
                &emb,
                "user1",
                "hash_abc",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        // Same hash and same content should return Duplicate
        let emb2 = vec![0.0, 1.0, 0.0, 0.0]; // different embedding doesn't matter
        let result = check_dedup(
            &storage,
            &emb2,
            "hash_abc",
            "hello world",
            "user1",
            None,
            0.15,
        )
        .unwrap();
        match result {
            DedupResult::Duplicate {
                existing_id,
                existing_content,
            } => {
                assert_eq!(existing_id, "existing1");
                assert_eq!(existing_content, "hello world");
            }
            DedupResult::New => panic!("Expected Duplicate, got New"),
        }
    }

    #[test]
    fn test_vector_similar_returns_duplicate() {
        let storage = Storage::open(test_config(4)).unwrap();
        let emb = vec![1.0, 0.0, 0.0, 0.0];
        storage
            .insert_memory(
                "existing1",
                "hello",
                &emb,
                "user1",
                "hash1",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        // Very similar embedding, different hash
        let emb2 = vec![0.99, 0.01, 0.0, 0.0];
        let result = check_dedup(
            &storage,
            &emb2,
            "different_hash",
            "hello similar",
            "user1",
            None,
            0.15,
        )
        .unwrap();
        assert!(matches!(result, DedupResult::Duplicate { .. }));
    }

    #[test]
    fn test_vector_dissimilar_returns_new() {
        let storage = Storage::open(test_config(4)).unwrap();
        let emb = vec![1.0, 0.0, 0.0, 0.0];
        storage
            .insert_memory(
                "existing1",
                "hello",
                &emb,
                "user1",
                "hash1",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        // Orthogonal embedding (distance ~1.0), different hash
        let emb2 = vec![0.0, 1.0, 0.0, 0.0];
        let result = check_dedup(
            &storage,
            &emb2,
            "different_hash",
            "something different",
            "user1",
            None,
            0.15,
        )
        .unwrap();
        assert!(matches!(result, DedupResult::New));
    }

    #[test]
    fn test_different_user_no_dedup() {
        let storage = Storage::open(test_config(4)).unwrap();
        let emb = vec![1.0, 0.0, 0.0, 0.0];
        storage
            .insert_memory(
                "existing1",
                "hello",
                &emb,
                "user1",
                "hash1",
                &InsertMemoryParams::default(),
            )
            .unwrap();

        // Same hash and embedding but different user — should be New
        let result = check_dedup(&storage, &emb, "hash1", "hello", "user2", None, 0.15).unwrap();
        assert!(matches!(result, DedupResult::New));
    }
}

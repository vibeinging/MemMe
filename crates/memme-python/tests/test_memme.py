"""Tests for the memme Python bindings.

Run with: pytest crates/memme-python/tests/test_memme.py -v
"""

import json
import pytest
import memme


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    """Create an in-memory store with mock embedder."""
    return memme.MemoryStore(":memory:", embedder="mock")


@pytest.fixture
def populated_store(store):
    """Store pre-loaded with 5 memories."""
    store.add("I love drinking coffee every morning", user_id="alice")
    store.add("My favorite programming language is Rust", user_id="alice")
    store.add("I work at Google as a software engineer", user_id="alice")
    store.add("I enjoy hiking in the mountains", user_id="alice", agent_id="travel")
    store.add("My cat Luna sleeps on the keyboard", user_id="bob")
    return store


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_default_memory(self):
        s = memme.MemoryStore(":memory:", embedder="mock")
        assert s is not None

    def test_mock_with_dims(self):
        s = memme.MemoryStore(":memory:", embedder="mock", dims=128)
        assert s is not None

    def test_unknown_embedder_raises(self):
        with pytest.raises(RuntimeError, match="Unknown embedder"):
            memme.MemoryStore(":memory:", embedder="invalid")

    def test_openai_without_key_raises(self):
        with pytest.raises(RuntimeError, match="api_key required"):
            memme.MemoryStore(":memory:", embedder="openai")


# ---------------------------------------------------------------------------
# Add tests
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_returns_dict(self, store):
        result = store.add("hello world", user_id="user1")
        assert isinstance(result, dict)
        assert "id" in result
        assert result["content"] == "hello world"
        assert result["user_id"] == "user1"
        assert result["created_at"] is not None
        assert result["updated_at"] is not None

    def test_add_with_metadata(self, store):
        meta = json.dumps({"source": "chat", "importance": 0.9})
        result = store.add("test", user_id="user1", metadata=meta)
        assert result["metadata"] is not None
        parsed = json.loads(result["metadata"])
        assert parsed["source"] == "chat"
        assert parsed["importance"] == 0.9

    def test_add_with_agent_id(self, store):
        result = store.add("test", user_id="user1", agent_id="agent-1")
        assert result["id"] is not None

    def test_add_invalid_metadata_raises(self, store):
        with pytest.raises(RuntimeError, match="Invalid metadata JSON"):
            store.add("test", user_id="user1", metadata="not json")

    def test_add_dedup_same_content(self, store):
        r1 = store.add("exact same content", user_id="user1")
        r2 = store.add("exact same content", user_id="user1")
        assert r1["id"] == r2["id"]  # dedup returns same id


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_list(self, populated_store):
        results = populated_store.search("coffee", user_id="alice")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_has_scores(self, populated_store):
        results = populated_store.search("coffee", user_id="alice")
        for r in results:
            assert r["score"] is not None

    def test_search_limit(self, populated_store):
        results = populated_store.search("coffee", user_id="alice", limit=2)
        assert len(results) <= 2

    def test_search_user_isolation(self, populated_store):
        results = populated_store.search("cat", user_id="alice")
        for r in results:
            assert r["user_id"] == "alice"

    def test_search_empty_query_raises(self, populated_store):
        with pytest.raises(RuntimeError, match="empty"):
            populated_store.search("", user_id="alice")


# ---------------------------------------------------------------------------
# Get tests
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_existing(self, store):
        added = store.add("find me", user_id="user1")
        found = store.get(added["id"])
        assert found is not None
        assert found["content"] == "find me"
        assert found["id"] == added["id"]

    def test_get_nonexistent(self, store):
        result = store.get("nonexistent-id")
        assert result is None


# ---------------------------------------------------------------------------
# Update tests
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_content(self, store):
        added = store.add("old content", user_id="user1")
        updated = store.update(added["id"], "new content")
        assert updated["content"] == "new content"
        assert updated["id"] == added["id"]

    def test_update_nonexistent_raises(self, store):
        with pytest.raises(RuntimeError):
            store.update("nonexistent-id", "new content")


# ---------------------------------------------------------------------------
# Delete tests
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_existing(self, store):
        added = store.add("delete me", user_id="user1")
        store.delete(added["id"])
        result = store.get(added["id"])
        assert result is None

    def test_delete_nonexistent_raises(self, store):
        with pytest.raises(RuntimeError):
            store.delete("nonexistent-id")


# ---------------------------------------------------------------------------
# List tests
# ---------------------------------------------------------------------------

class TestList:
    def test_list_by_user(self, populated_store):
        alice_mems = populated_store.list(user_id="alice")
        for m in alice_mems:
            assert m["user_id"] == "alice"

        bob_mems = populated_store.list(user_id="bob")
        for m in bob_mems:
            assert m["user_id"] == "bob"

    def test_list_by_agent(self, populated_store):
        results = populated_store.list(user_id="alice", agent_id="travel")
        assert len(results) == 1
        assert "hiking" in results[0]["content"]

    def test_list_limit(self, populated_store):
        results = populated_store.list(user_id="alice", limit=2)
        assert len(results) <= 2

    def test_list_empty_user(self, store):
        results = store.list(user_id="nobody")
        assert results == []


# ---------------------------------------------------------------------------
# FTS & Hybrid search tests
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def test_rebuild_fts_index(self, populated_store):
        populated_store.rebuild_fts_index()

    def test_hybrid_search(self, populated_store):
        populated_store.rebuild_fts_index()
        results = populated_store.hybrid_search("coffee morning", user_id="alice")
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert r["score"] is not None

    def test_hybrid_search_custom_weights(self):
        weighted_store = memme.MemoryStore(
            ":memory:",
            embedder="mock",
            rrf_vector_weight=0.3,
            rrf_fts_weight=0.7,
        )
        weighted_store.add(
            "I love drinking coffee every morning",
            user_id="alice",
        )
        weighted_store.rebuild_fts_index()
        results = weighted_store.hybrid_search("coffee", user_id="alice")
        assert len(results) > 0

    def test_hybrid_search_without_fts_index(self, populated_store):
        # Should still work (FTS portion fails gracefully, returns vector-only)
        results = populated_store.hybrid_search("coffee", user_id="alice")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Graph tests (search_graph doesn't need LLM)
# ---------------------------------------------------------------------------

class TestGraph:
    def test_search_graph_empty(self, store):
        result = store.search_graph("Alice", user_id="user1")
        assert isinstance(result, dict)
        assert "entities" in result
        assert "relations" in result
        assert result["entities"] == []
        assert result["relations"] == []


# ---------------------------------------------------------------------------
# Full lifecycle test
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_full_lifecycle(self):
        store = memme.MemoryStore(":memory:", embedder="mock")

        # Add
        r1 = store.add("I love coffee", user_id="alice")
        r2 = store.add("I work at Google", user_id="alice")
        assert r1["id"] != r2["id"]

        # Search
        results = store.search("beverages", user_id="alice")
        assert len(results) == 2

        # Update
        store.update(r1["id"], "I love tea now")
        updated = store.get(r1["id"])
        assert updated["content"] == "I love tea now"

        # Delete
        store.delete(r2["id"])
        remaining = store.list(user_id="alice")
        assert len(remaining) == 1
        assert remaining[0]["id"] == r1["id"]

        # Hybrid search
        store.rebuild_fts_index()
        hybrid = store.hybrid_search("tea", user_id="alice")
        assert len(hybrid) > 0

    def test_multi_user_isolation(self):
        store = memme.MemoryStore(":memory:", embedder="mock")

        store.add("Alice's secret", user_id="alice")
        store.add("Bob's secret", user_id="bob")

        alice_results = store.list(user_id="alice")
        bob_results = store.list(user_id="bob")

        assert len(alice_results) == 1
        assert len(bob_results) == 1
        assert alice_results[0]["content"] == "Alice's secret"
        assert bob_results[0]["content"] == "Bob's secret"


# ---------------------------------------------------------------------------
# DeleteAll tests
# ---------------------------------------------------------------------------

class TestDeleteAll:
    def test_delete_all_by_user(self, populated_store):
        count = populated_store.delete_all(user_id="alice")
        assert count >= 1
        remaining = populated_store.list(user_id="alice")
        assert len(remaining) == 0
        # bob's memories should be untouched
        bob = populated_store.list(user_id="bob")
        assert len(bob) == 1

    def test_delete_all_by_agent(self, store):
        store.add("mem1", user_id="u1", agent_id="a1")
        store.add("mem2", user_id="u1", agent_id="a2")
        count = store.delete_all(user_id="u1", agent_id="a1")
        assert count == 1
        remaining = store.list(user_id="u1")
        assert len(remaining) == 1


# ---------------------------------------------------------------------------
# History tests
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_lifecycle(self, store):
        added = store.add("original", user_id="u1")
        store.update(added["id"], "updated")
        store.delete(added["id"])
        records = store.history(added["id"])
        assert len(records) == 3
        events = [r["event"] for r in records]
        assert "ADD" in events
        assert "UPDATE" in events
        assert "DELETE" in events

    def test_history_nonexistent(self, store):
        records = store.history("nonexistent")
        assert records == []


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset(self, populated_store):
        populated_store.reset()
        alice = populated_store.list(user_id="alice")
        bob = populated_store.list(user_id="bob")
        assert len(alice) == 0
        assert len(bob) == 0


# ---------------------------------------------------------------------------
# RunId tests
# ---------------------------------------------------------------------------

class TestRunId:
    def test_add_with_run_id(self, store):
        store.add("run mem", user_id="u1", run_id="run-1")
        results = store.list(user_id="u1")
        assert len(results) == 1

    def test_search_with_run_id(self, store):
        store.add("r1 content", user_id="u1", run_id="run-1")
        store.add("r2 content", user_id="u1", run_id="run-2")
        # search should work (run_id filter may or may not be on search)
        results = store.search("content", user_id="u1")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Analytics tests
# ---------------------------------------------------------------------------

class TestAnalytics:
    def test_user_stats(self, populated_store):
        stats = populated_store.user_stats("alice")
        assert isinstance(stats, dict)
        assert stats["total_memories"] >= 1
        assert "earliest_memory" in stats

    def test_memory_frequency(self, populated_store):
        freq = populated_store.memory_frequency("alice", "day", limit=10)
        assert isinstance(freq, list)
        assert len(freq) >= 1
        assert "period" in freq[0]
        assert "count" in freq[0]

    def test_top_entities(self, store):
        # Empty store should return empty list
        entities = store.top_entities("u1", limit=5)
        assert isinstance(entities, list)

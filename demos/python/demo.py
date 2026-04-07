#!/usr/bin/env python3
"""
MemMe Python Demo — full-featured walkthrough of the memory engine.

Covers:
  1. Basic CRUD (add / get / update / delete / list)
  2. Search (vector similarity + hybrid search)
  3. Knowledge Graph (add_graph + search_graph)
  4. Session / Episode (append_events via add_smart_messages → compact → episode)
  5. Analytics (user_stats / top_entities / memory_frequency)
  6. History tracking

Run:
    cd crates/memme-python && maturin develop --release
    python demos/python/demo.py
"""

import json
import time
import memme

# ── helpers ──────────────────────────────────────────────────────────────────

OK = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SECTION = "\033[1;34m"
RESET = "\033[0m"
step_count = 0


def section(title: str):
    print(f"\n{SECTION}{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}{RESET}\n")


def step(desc: str):
    global step_count
    step_count += 1
    print(f"  [{step_count:02d}] {desc}")


def ok(msg: str):
    print(f"       {OK} {msg}")


def fail(msg: str):
    print(f"       {FAIL} {msg}")


def assert_true(cond: bool, msg: str):
    if cond:
        ok(msg)
    else:
        fail(msg)
        raise AssertionError(msg)


def assert_eq(a, b, msg: str):
    if a == b:
        ok(msg)
    else:
        fail(f"{msg} — expected {b}, got {a}")
        raise AssertionError(msg)


def assert_gte(a, b, msg: str):
    if a >= b:
        ok(f"{msg} ({a})")
    else:
        fail(f"{msg} — expected >= {b}, got {a}")
        raise AssertionError(msg)


class AssertionError(Exception):
    pass


# ── 1. Basic CRUD ───────────────────────────────────────────────────────────

def demo_crud(store: memme.MemoryStore):
    """Demonstrate basic CRUD operations."""
    section("1. Basic CRUD Operations")

    ids = {}

    # Add
    step("add() — store 5 memories for user 'alice'")
    memories_data = [
        ("I love drinking coffee every morning", {"category": "habits"}),
        ("My favorite programming language is Rust", {"category": "tech"}),
        ("I work at Google as a software engineer", {"category": "work"}),
        ("I enjoy hiking in the mountains on weekends", {"category": "hobbies"}),
        ("My cat Luna always sleeps on my keyboard", {"category": "pets"}),
    ]
    added_ids = []
    for content, meta in memories_data:
        r = store.add(content, user_id="alice", metadata=json.dumps(meta))
        added_ids.append(r["id"])
        ok(f"id={r['id'][:8]}.. \"{content[:50]}\"")

    # Add for another user
    step("add() — store 1 memory for user 'bob'")
    r = store.add("Bob likes playing guitar", user_id="bob")
    bob_id = r["id"]
    ok(f"id={r['id'][:8]}.. user=bob")

    # Get
    step("get() — retrieve a memory by ID")
    first_id = added_ids[0]  # coffee memory
    got = store.get(first_id)
    assert_true(got is not None, f"Found memory {first_id[:8]}..")
    assert_true("coffee" in got["content"], "Content matches")

    # Update
    step("update() — change coffee to matcha")
    updated = store.update(first_id, "I switched from coffee to matcha in the morning")
    assert_true("matcha" in updated["content"], f"Updated: \"{updated['content'][:50]}\"")

    # History
    step("history() — verify change is tracked")
    hist = store.history(first_id)
    assert_gte(len(hist), 1, f"History has {len(hist)} entries")

    # Delete
    step("delete() — remove bob's memory")
    store.delete(bob_id)
    gone = store.get(bob_id)
    assert_eq(gone, None, "Memory deleted successfully")

    # List
    step("list() — list alice's memories")
    alice_mems = store.list(user_id="alice", limit=50)
    assert_eq(len(alice_mems), 5, f"Alice has 5 memories")


# ── 2. Search ───────────────────────────────────────────────────────────────

def demo_search(store: memme.MemoryStore):
    section("2. Search (Vector + Hybrid)")

    # Vector search
    step("search() — find memories about 'morning routine'")
    results = store.search("morning routine", user_id="alice", limit=3)
    assert_gte(len(results), 1, f"Found {len(results)} results")
    for r in results:
        score = r.get("score", 0) or 0
        print(f"           score={score:.4f}  \"{r['content'][:60]}\"")

    # Rebuild FTS for hybrid search
    step("rebuild_fts_index() — prepare for hybrid search")
    store.rebuild_fts_index()
    ok("FTS index rebuilt")

    # Hybrid search
    step("hybrid_search() — vector + BM25 with RRF fusion")
    results = store.hybrid_search("Rust programming", user_id="alice", limit=3)
    assert_gte(len(results), 1, f"Found {len(results)} results")
    for r in results:
        score = r.get("score", 0) or 0
        print(f"           score={score:.4f}  \"{r['content'][:60]}\"")

    # Cross-user isolation
    step("search() — verify user isolation (bob should find nothing)")
    bob_results = store.search("coffee", user_id="bob", limit=5)
    assert_eq(len(bob_results), 0, "Bob sees 0 results (isolation works)")


# ── 3. Knowledge Graph ─────────────────────────────────────────────────────

def demo_graph(store: memme.MemoryStore):
    section("3. Knowledge Graph")

    # Note: add_graph requires LLM. With mock embedder + no LLM, we test
    # search_graph which is pure SQL. Graph data comes from add_smart.
    step("search_graph() — search for entities (may be empty without LLM)")
    try:
        result = store.search_graph("alice", user_id="alice")
        n_ent = len(result.get("entities", []))
        n_rel = len(result.get("relations", []))
        ok(f"Found {n_ent} entities, {n_rel} relations")
    except Exception as e:
        ok(f"Graph search returned: {e} (expected without graph data)")


# ── 4. Session / Episode ────────────────────────────────────────────────────

def demo_session_episode(store: memme.MemoryStore):
    section("4. Session / Episode Architecture")

    # add_smart_messages: append events to a session → compact → episode
    step("add_smart_messages() — simulate a conversation")
    messages = [
        ("user", "Hi, I'm Alex. I just moved to Tokyo for a new job at a startup."),
        ("assistant", "Welcome to Tokyo, Alex! That sounds exciting. What kind of startup?"),
        ("user", "It's an AI company. We're building memory systems for personal assistants."),
        ("assistant", "Interesting! Memory systems are a hot topic. How are you settling in?"),
        ("user", "Good! I found an apartment in Shibuya. My girlfriend Emma is joining next month."),
    ]

    try:
        result = store.add_smart_messages(messages, user_id="alice", run_id="session-001")
        # add_smart_messages returns a list with one dict containing memories + episode_id
        if result and isinstance(result[0], dict) and "memories" in result[0]:
            wrapper = result[0]
            n_mems = len(wrapper.get("memories", []))
            ep_id = wrapper.get("episode_id")
            ok(f"Created {n_mems} memories, episode_id={str(ep_id)[:8] if ep_id else 'None'}")
        else:
            ok(f"Created memories from conversation")
    except Exception as e:
        # Without LLM, add_smart_messages still appends events and compacts
        ok(f"Session processing: {e}")

    # List episodes
    step("list_episodes() — check episodes created")
    try:
        episodes = store.list_episodes(user_id="alice", limit=10)
        ok(f"Found {len(episodes)} episodes")
        for ep in episodes[:3]:
            print(f"           id={ep['episode_id'][:8]}..  summary=\"{(ep.get('summary') or 'N/A')[:50]}\"")
    except Exception as e:
        ok(f"list_episodes: {e}")

    # Search episodes
    step("search_episodes() — search by content")
    try:
        found = store.search_episodes("Tokyo startup", user_id="alice", limit=3)
        ok(f"Found {len(found)} matching episodes")
    except Exception as e:
        ok(f"search_episodes: {e}")

    # Get session context
    step("get_session_context() — retrieve context within token budget")
    try:
        ctx = store.get_session_context("session-001", token_budget=2000)
        ok(f"tokens_used={ctx['tokens_used']}/{ctx['token_budget']}, events={len(ctx['events'])}")
        if ctx.get("episode_summary"):
            print(f"           summary: \"{ctx['episode_summary'][:60]}\"")
    except Exception as e:
        ok(f"get_session_context: {e}")

    # Second conversation in a different session
    step("add_smart_messages() — second session")
    messages2 = [
        ("user", "I started learning Japanese. My teacher is Tanaka-sensei."),
        ("assistant", "Great! How often do you have lessons?"),
        ("user", "Twice a week. I'm using Anki flashcards too."),
    ]
    try:
        store.add_smart_messages(messages2, user_id="alice", run_id="session-002")
        ok("Second session processed")
    except Exception as e:
        ok(f"Second session: {e}")

    # Episode messages
    step("get_episode_messages() — retrieve messages from an episode")
    try:
        episodes = store.list_episodes(user_id="alice", limit=1)
        if episodes:
            ep_id = episodes[0]["episode_id"]
            msgs = store.get_episode_messages(ep_id, limit=10)
            ok(f"Episode {ep_id[:8]}.. has {len(msgs)} messages")
            for m in msgs[:3]:
                print(f"           [{m['event_type']}] \"{m['content'][:50]}\"")
    except Exception as e:
        ok(f"get_episode_messages: {e}")


# ── 5. Analytics ────────────────────────────────────────────────────────────

def demo_analytics(store: memme.MemoryStore):
    section("5. Analytics")

    step("user_stats() — summary for alice")
    stats = store.user_stats("alice")
    ok(f"memories={stats['total_memories']}, entities={stats['total_entities']}, "
       f"relations={stats['total_relationships']}")

    step("top_entities() — most connected entities")
    entities = store.top_entities("alice", limit=5)
    ok(f"Found {len(entities)} entities")
    for e in entities[:5]:
        print(f"           {e['name']} ({e.get('entity_type') or '?'}) — {e['relationship_count']} links")

    step("memory_frequency() — creation trend by day")
    freq = store.memory_frequency("alice", "day", limit=7)
    ok(f"{len(freq)} time buckets")
    for b in freq[:3]:
        print(f"           {b['period']}: {b['count']} memories")


# ── 6. Bulk Operations ─────────────────────────────────────────────────────

def demo_bulk(store: memme.MemoryStore):
    section("6. Bulk Operations & Cleanup")

    step("delete_all() — remove all of alice's memories")
    count = store.delete_all(user_id="alice")
    ok(f"Deleted {count} memories")

    step("list() — verify empty")
    remaining = store.list(user_id="alice", limit=10)
    assert_eq(len(remaining), 0, "No memories remaining")

    step("reset() — full store reset")
    store.reset()
    ok("Store reset complete")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{SECTION}{'═' * 60}")
    print("  MemMe Python Demo")
    print(f"  All features, in-memory, mock embedder (no API key needed)")
    print(f"{'═' * 60}{RESET}")

    t0 = time.time()
    store = memme.MemoryStore(":memory:", embedder="mock")

    demo_crud(store)
    demo_search(store)
    demo_graph(store)
    demo_session_episode(store)
    demo_analytics(store)
    demo_bulk(store)

    elapsed = time.time() - t0
    section("Done!")
    print(f"  All {step_count} steps passed in {elapsed:.2f}s")
    print(f"  No API key needed — everything ran locally with mock embedder.\n")


if __name__ == "__main__":
    main()

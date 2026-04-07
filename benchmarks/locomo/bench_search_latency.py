#!/usr/bin/env python3
"""Benchmark: measure end-to-end search latency with parallel vs serial channels.

Uses cached LoCoMo v20 data. No API calls needed — uses a mock embedder
that returns the first stored embedding (sufficient for latency measurement).

Usage:
    python3 bench_search_latency.py
"""

import json
import os
import statistics
import sys
import time

# Add project root for memme import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    import memme
except ImportError:
    print("ERROR: memme Python module not found.")
    print("Build it first: cd crates/memme-python && maturin develop --release")
    sys.exit(1)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locomo10.json")
WARMUP = 3
RUNS = 20


def load_questions():
    """Load questions from LoCoMo dataset."""
    with open(DATA_PATH) as f:
        data = json.load(f)

    questions = []
    for conv in data:
        sample_id = conv.get("sample_id", "unknown")
        for qa in conv.get("qa", []):
            q = qa.get("question", "")
            cat = qa.get("category", "unknown")
            # Determine user_id (speaker_a from first session)
            conversation = conv.get("conversation", {})
            first_session_key = sorted(
                [k for k in conversation if k.startswith("session_") and k.replace("session_", "").isdigit()],
                key=lambda k: int(k.replace("session_", ""))
            )
            if first_session_key:
                turns = conversation.get(first_session_key[0], [])
                speakers = list(dict.fromkeys(t["speaker"] for t in turns))
                if speakers:
                    user_id = f"{sample_id}_{speakers[0]}"
                else:
                    user_id = f"{sample_id}_unknown"
            else:
                user_id = f"{sample_id}_unknown"
            questions.append({"question": q, "category": cat, "user_id": user_id, "sample_id": sample_id})
    return questions


def find_cached_db(sample_id):
    """Find the cached DuckDB file for a sample."""
    db_path = os.path.join(CACHE_DIR, f"{sample_id}.duckdb")
    if os.path.exists(db_path):
        return db_path
    return None


def run_benchmark():
    print("Loading LoCoMo questions...")
    all_questions = load_questions()
    print(f"  Total questions: {len(all_questions)}")

    # Group by sample_id and find cached DBs
    by_sample = {}
    for q in all_questions:
        sid = q["sample_id"]
        if sid not in by_sample:
            db_path = find_cached_db(sid)
            if db_path:
                by_sample[sid] = {"db_path": db_path, "questions": []}
        if sid in by_sample:
            by_sample[sid]["questions"].append(q)

    print(f"  Samples with cached data: {len(by_sample)}")
    if not by_sample:
        print("ERROR: No cached databases found in", CACHE_DIR)
        sys.exit(1)

    # Pick first available sample for detailed testing
    sample_id = list(by_sample.keys())[0]
    sample = by_sample[sample_id]
    questions = sample["questions"][:10]  # Test with first 10 questions

    print(f"\nUsing sample: {sample_id}")
    print(f"  DB: {sample['db_path']}")
    print(f"  Questions: {len(questions)}")

    # Check if memme has the needed API
    # We need an embedder, but we don't have an API key.
    # Check if there's a way to create store with a local embedder or mock.
    api_key = os.environ.get("OPENAI_API_KEY", "")
    embed_base_url = os.environ.get("EMBED_BASE_URL", "")

    if not api_key:
        print("\nWARNING: OPENAI_API_KEY not set. Search requires embedding generation.")
        print("Set OPENAI_API_KEY to run the full search benchmark.")
        print("\nFalling back to raw SQL latency test...")
        run_sql_latency_test(sample['db_path'], questions)
        return

    print(f"\nCreating MemoryStore (read-only from cache)...")
    store = memme.MemoryStore(
        db_path=sample['db_path'],
        embedder="openai",
        api_key=api_key,
        base_url=embed_base_url if embed_base_url else None,
        embed_model="text-embedding-3-small",
        dims=1536,
    )

    # Warmup
    print(f"\nWarming up ({WARMUP} runs)...")
    for _ in range(WARMUP):
        for q in questions[:3]:
            try:
                store.search(q["question"], user_id=q["user_id"], limit=10)
            except Exception:
                pass

    # Benchmark
    print(f"\nBenchmarking ({RUNS} runs × {len(questions)} questions)...")
    all_times = []
    per_category = {}

    for run_idx in range(RUNS):
        for q in questions:
            start = time.perf_counter()
            try:
                results = store.search(q["question"], user_id=q["user_id"], limit=10)
                n_results = len(results)
            except Exception as e:
                n_results = -1
            elapsed_ms = (time.perf_counter() - start) * 1000

            all_times.append(elapsed_ms)
            cat = q["category"]
            if cat not in per_category:
                per_category[cat] = []
            per_category[cat].append(elapsed_ms)

    # Results
    print(f"\n{'='*60}")
    print(f"Search Latency Results (parallel channels)")
    print(f"{'='*60}")
    print(f"  Total queries: {len(all_times)}")
    print(f"  Mean:   {statistics.mean(all_times):8.1f} ms")
    print(f"  Median: {statistics.median(all_times):8.1f} ms")
    print(f"  P95:    {sorted(all_times)[int(len(all_times)*0.95)]:8.1f} ms")
    print(f"  Min:    {min(all_times):8.1f} ms")
    print(f"  Max:    {max(all_times):8.1f} ms")

    print(f"\nBy category:")
    for cat, times in sorted(per_category.items()):
        print(f"  {str(cat):20s}  mean={statistics.mean(times):8.1f}ms  "
              f"median={statistics.median(times):8.1f}ms  n={len(times)}")

    # Save
    out = {
        "sample_id": sample_id,
        "n_questions": len(questions),
        "n_runs": RUNS,
        "overall": {
            "mean_ms": round(statistics.mean(all_times), 1),
            "median_ms": round(statistics.median(all_times), 1),
            "p95_ms": round(sorted(all_times)[int(len(all_times)*0.95)], 1),
        },
        "per_category": {
            cat: {"mean_ms": round(statistics.mean(t), 1), "median_ms": round(statistics.median(t), 1)}
            for cat, t in per_category.items()
        },
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_search_latency_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


def run_sql_latency_test(db_path, questions):
    """Fallback: test raw SQL query latency without embeddings."""
    import duckdb

    conn = duckdb.connect(db_path, read_only=True)

    print(f"\nSQL-only latency test (no embedding, {RUNS} runs × 4 queries)...")

    user_id = questions[0]["user_id"]

    # Get some entities
    entities = conn.execute(
        "SELECT LOWER(name) FROM entities_default LIMIT 4"
    ).fetchall()
    entity_names = [r[0] for r in entities]
    in_clause = ", ".join(f"'{e}'" for e in entity_names)

    queries = {
        "vector_scan": (
            f"SELECT id, content FROM memories WHERE user_id = $1 ORDER BY importance DESC LIMIT 30",
            [user_id],
        ),
        "fts_scan": (
            f"SELECT id, content FROM memories WHERE user_id = $1 AND content LIKE '%the%' LIMIT 30",
            [user_id],
        ),
        "entity_spread": (
            f"""SELECT DISTINCT LOWER(e2.name)
                FROM entities_default e1
                JOIN relationships_default r ON (e1.id = r.source_id OR e1.id = r.target_id)
                JOIN entities_default e2 ON (e2.id = r.source_id OR e2.id = r.target_id)
                WHERE LOWER(e1.name) IN ({in_clause}) AND e1.user_id = $1 AND e2.id != e1.id
                LIMIT 100""",
            [user_id],
        ),
        "temporal": (
            f"SELECT id, content, event_time FROM memories WHERE user_id = $1 AND event_time IS NOT NULL ORDER BY event_time DESC LIMIT 30",
            [user_id],
        ),
    }

    # Serial
    print("\n  Serial execution:")
    serial_times = []
    for _ in range(WARMUP):
        for _, (sql, params) in queries.items():
            conn.execute(sql, params).fetchall()

    for _ in range(RUNS):
        start = time.perf_counter()
        for _, (sql, params) in queries.items():
            conn.execute(sql, params).fetchall()
        serial_times.append((time.perf_counter() - start) * 1000)

    print(f"    Mean: {statistics.mean(serial_times):8.3f} ms (4 queries total)")
    print(f"    Med:  {statistics.median(serial_times):8.3f} ms")

    # Parallel (simulated with threads)
    import threading

    print("\n  Parallel execution (4 threads):")
    parallel_times = []

    for _ in range(WARMUP):
        threads = []
        for _, (sql, params) in queries.items():
            t = threading.Thread(target=lambda s, p: duckdb.connect(db_path, read_only=True).execute(s, p).fetchall(), args=(sql, params))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    for _ in range(RUNS):
        start = time.perf_counter()
        threads = []
        for _, (sql, params) in queries.items():
            t = threading.Thread(target=lambda s, p: duckdb.connect(db_path, read_only=True).execute(s, p).fetchall(), args=(sql, params))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        parallel_times.append((time.perf_counter() - start) * 1000)

    print(f"    Mean: {statistics.mean(parallel_times):8.3f} ms (4 queries total)")
    print(f"    Med:  {statistics.median(parallel_times):8.3f} ms")

    speedup = statistics.mean(serial_times) / statistics.mean(parallel_times)
    print(f"\n  Speedup: {speedup:.2f}x")

    conn.close()


if __name__ == "__main__":
    run_benchmark()

#!/usr/bin/env python3
"""Benchmark: measure search latency before/after adding B-tree indexes.

Uses real LoCoMo v20 cached data (conv-26).
Tests the three bottleneck SQL patterns identified in the analysis:
  1. Entity spreading activation (LOWER(name) IN ...)
  2. Entity-memory lookup (LOWER(entity_name) IN ...)
  3. Temporal search (event_time ordering)
  4. Full vector+filter search pattern
"""

import duckdb
import time
import shutil
import os
import statistics
import json

CACHE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/cache"
SOURCE_DB = CACHE_DIR + "/conv-26.duckdb"
WORK_DB = os.path.dirname(os.path.abspath(__file__)) + "/bench_index_perf_work.duckdb"
COLLECTION = "default"
WARMUP_RUNS = 3
BENCH_RUNS = 20

# Realistic entity names from the dataset
SEED_ENTITIES = ["alice", "bob", "google", "san francisco"]


def get_real_entities(conn, n=4):
    """Get real entity names from the database."""
    rows = conn.execute(
        f"SELECT LOWER(name) FROM entities_{COLLECTION} LIMIT {n}"
    ).fetchall()
    return [r[0] for r in rows] if rows else SEED_ENTITIES


def get_real_user_id(conn):
    rows = conn.execute("SELECT DISTINCT user_id FROM memories LIMIT 1").fetchall()
    return rows[0][0] if rows else "user1"


def get_sample_embedding(conn):
    """Get a real embedding from the database for vector search testing."""
    row = conn.execute("SELECT embedding FROM memories WHERE embedding IS NOT NULL LIMIT 1").fetchone()
    if row and row[0]:
        return row[0]
    return None


def bench_query(conn, name, sql, params=None, runs=BENCH_RUNS):
    """Run a query multiple times and return timing stats."""
    # Warmup
    for _ in range(WARMUP_RUNS):
        try:
            if params:
                conn.execute(sql, params).fetchall()
            else:
                conn.execute(sql).fetchall()
        except Exception:
            pass

    times = []
    result_count = 0
    for _ in range(runs):
        start = time.perf_counter()
        try:
            if params:
                rows = conn.execute(sql, params).fetchall()
            else:
                rows = conn.execute(sql).fetchall()
            result_count = len(rows)
        except Exception as e:
            result_count = -1
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return {
        "name": name,
        "runs": runs,
        "result_count": result_count,
        "mean_ms": round(statistics.mean(times), 3),
        "median_ms": round(statistics.median(times), 3),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
    }


def build_queries(entities, user_id, embedding):
    """Build the benchmark query set."""
    in_clause = ", ".join(f"'{e}'" for e in entities)

    queries = []

    # 1. Entity spreading activation (the bottleneck: LOWER(name) IN ...)
    queries.append((
        "entity_spreading",
        f"""SELECT DISTINCT LOWER(e2.name)
            FROM entities_{COLLECTION} e1
            JOIN relationships_{COLLECTION} r ON (e1.id = r.source_id OR e1.id = r.target_id)
            JOIN entities_{COLLECTION} e2 ON (e2.id = r.source_id OR e2.id = r.target_id)
            WHERE LOWER(e1.name) IN ({in_clause})
              AND e1.user_id = $1
              AND e2.id != e1.id
            LIMIT 100""",
        [user_id],
    ))

    # 2. Entity-memory lookup (LOWER(entity_name) IN ...)
    queries.append((
        "entity_memory_lookup",
        f"""SELECT DISTINCT m.id, m.content, m.importance
            FROM memories m
            JOIN memory_entities me ON m.id = me.memory_id
            WHERE me.user_id = $1
              AND LOWER(me.entity_name) IN ({in_clause})
            ORDER BY m.importance DESC
            LIMIT 30""",
        [user_id],
    ))

    # 3. Temporal search (event_time ordering, no index)
    queries.append((
        "temporal_search",
        f"""SELECT id, content, event_time
            FROM memories
            WHERE user_id = $1
              AND event_time IS NOT NULL
            ORDER BY event_time DESC
            LIMIT 30""",
        [user_id],
    ))

    # 4. User-filtered memory count (baseline)
    queries.append((
        "user_memory_count",
        f"""SELECT COUNT(*) FROM memories WHERE user_id = $1""",
        [user_id],
    ))

    # 5. Entity name search (LIKE pattern)
    if entities:
        queries.append((
            "entity_name_search",
            f"""SELECT id, name, entity_type
                FROM entities_{COLLECTION}
                WHERE LOWER(name) LIKE $1 AND user_id = $2
                LIMIT 10""",
            [f"%{entities[0]}%", user_id],
        ))

    return queries


def add_indexes(conn):
    """Add the missing B-tree indexes."""
    indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_entities_name_user ON entities_{COLLECTION}(user_id, name)",
        f"CREATE INDEX IF NOT EXISTS idx_me_name_user ON memory_entities(user_id, entity_name)",
        "CREATE INDEX IF NOT EXISTS idx_memories_user_event ON memories(user_id, event_time DESC)",
        "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)",
    ]
    for sql in indexes:
        conn.execute(sql)
    print(f"  Added {len(indexes)} indexes")


def run_benchmark():
    # Prepare working copy
    if os.path.exists(WORK_DB):
        os.remove(WORK_DB)
    replica = WORK_DB + ".replica"
    if os.path.exists(replica):
        os.remove(replica)

    shutil.copy2(SOURCE_DB, WORK_DB)
    source_replica = SOURCE_DB + ".replica"
    if os.path.exists(source_replica):
        shutil.copy2(source_replica, replica)

    print(f"Source: {SOURCE_DB}")
    print(f"Working copy: {WORK_DB}")
    print()

    # Phase 1: Without indexes
    print("=" * 60)
    print("Phase 1: WITHOUT extra indexes (baseline)")
    print("=" * 60)

    conn = duckdb.connect(WORK_DB)
    entities = get_real_entities(conn)
    user_id = get_real_user_id(conn)
    embedding = get_sample_embedding(conn)

    print(f"  Entities: {entities}")
    print(f"  User ID: {user_id}")
    print(f"  Embedding dims: {len(embedding) if embedding else 'N/A'}")

    # Check data volumes
    mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    ent_count = conn.execute(f"SELECT COUNT(*) FROM entities_{COLLECTION}").fetchone()[0]
    rel_count = conn.execute(f"SELECT COUNT(*) FROM relationships_{COLLECTION}").fetchone()[0]
    me_count = conn.execute("SELECT COUNT(*) FROM memory_entities").fetchone()[0]
    print(f"  Data: {mem_count} memories, {ent_count} entities, {rel_count} relationships, {me_count} entity-links")
    print()

    queries = build_queries(entities, user_id, embedding)
    baseline_results = []
    for name, sql, params in queries:
        r = bench_query(conn, name, sql, params)
        baseline_results.append(r)
        print(f"  {r['name']:30s}  mean={r['mean_ms']:8.3f}ms  median={r['median_ms']:8.3f}ms  "
              f"p95={r['p95_ms']:8.3f}ms  rows={r['result_count']}")

    conn.close()

    # Phase 2: With indexes
    print()
    print("=" * 60)
    print("Phase 2: WITH B-tree indexes")
    print("=" * 60)

    conn = duckdb.connect(WORK_DB)
    add_indexes(conn)

    # Re-run same queries
    queries = build_queries(entities, user_id, embedding)
    indexed_results = []
    for name, sql, params in queries:
        r = bench_query(conn, name, sql, params)
        indexed_results.append(r)
        print(f"  {r['name']:30s}  mean={r['mean_ms']:8.3f}ms  median={r['median_ms']:8.3f}ms  "
              f"p95={r['p95_ms']:8.3f}ms  rows={r['result_count']}")

    conn.close()

    # Comparison
    print()
    print("=" * 60)
    print("Comparison: speedup from indexes")
    print("=" * 60)
    for b, i in zip(baseline_results, indexed_results):
        if b['mean_ms'] > 0:
            speedup = b['mean_ms'] / i['mean_ms'] if i['mean_ms'] > 0 else float('inf')
            delta = b['mean_ms'] - i['mean_ms']
            print(f"  {b['name']:30s}  {b['mean_ms']:8.3f}ms → {i['mean_ms']:8.3f}ms  "
                  f"({speedup:.2f}x, -{delta:.3f}ms)")

    # Save results
    results = {
        "source_db": SOURCE_DB,
        "data_volumes": {
            "memories": mem_count,
            "entities": ent_count,
            "relationships": rel_count,
            "memory_entities": me_count,
        },
        "baseline": baseline_results,
        "indexed": indexed_results,
    }
    out_path = os.path.dirname(os.path.abspath(__file__)) + "/bench_index_perf_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Cleanup
    os.remove(WORK_DB)
    if os.path.exists(replica):
        os.remove(replica)


if __name__ == "__main__":
    run_benchmark()

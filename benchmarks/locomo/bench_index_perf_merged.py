#!/usr/bin/env python3
"""Benchmark: measure search latency with merged data from ALL conversations.

Merges all 10 LoCoMo v20 conversations into a single database to simulate
a realistic multi-user scenario with larger data volumes.
"""

import duckdb
import time
import os
import statistics
import json
import glob

CACHE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/cache"
WORK_DB = os.path.dirname(os.path.abspath(__file__)) + "/bench_merged_work.duckdb"
COLLECTION = "default"
WARMUP_RUNS = 5
BENCH_RUNS = 50


def merge_databases():
    """Merge all conversation databases into one."""
    db_files = sorted(glob.glob(CACHE_DIR + "/conv-*.duckdb"))
    if not db_files:
        raise RuntimeError(f"No .duckdb files found in {CACHE_DIR}")

    if os.path.exists(WORK_DB):
        os.remove(WORK_DB)
    replica = WORK_DB + ".replica"
    if os.path.exists(replica):
        os.remove(replica)

    print(f"Merging {len(db_files)} databases...")

    # Use first DB as base
    import shutil
    shutil.copy2(db_files[0], WORK_DB)
    src_replica = db_files[0] + ".replica"
    if os.path.exists(src_replica):
        shutil.copy2(src_replica, replica)

    conn = duckdb.connect(WORK_DB)

    # Merge remaining databases
    tables_to_merge = [
        "memories", "memory_entities",
        f"entities_{COLLECTION}", f"relationships_{COLLECTION}",
        "events", "episodes", "sessions", "history",
    ]

    for db_path in db_files[1:]:
        name = os.path.basename(db_path)
        conn.execute(f"ATTACH '{db_path}' AS src (READ_ONLY)")
        for table in tables_to_merge:
            try:
                conn.execute(f"INSERT INTO main.{table} SELECT * FROM src.{table}")
            except Exception:
                pass  # table might not exist or have conflicts
        conn.execute("DETACH src")
        print(f"  Merged {name}")

    conn.close()
    return len(db_files)


def get_real_entities(conn, n=6):
    rows = conn.execute(
        f"SELECT LOWER(name), COUNT(*) as cnt FROM entities_{COLLECTION} "
        f"GROUP BY LOWER(name) ORDER BY cnt DESC LIMIT {n}"
    ).fetchall()
    return [r[0] for r in rows]


def get_real_user_ids(conn, n=3):
    rows = conn.execute("SELECT DISTINCT user_id FROM memories LIMIT $1", [n]).fetchall()
    return [r[0] for r in rows]


def bench_query(conn, name, sql, params=None, runs=BENCH_RUNS):
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
        except Exception:
            result_count = -1
        elapsed = (time.perf_counter() - start) * 1000
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


def build_queries(entities, user_id):
    in_clause = ", ".join(f"'{e}'" for e in entities)
    queries = []

    # 1. Entity spreading activation
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

    # 2. Entity-memory lookup
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

    # 3. Temporal search
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

    # 4. Cross-user entity search (worst case: no user filter on entities)
    queries.append((
        "global_entity_search",
        f"""SELECT id, name, entity_type, user_id
            FROM entities_{COLLECTION}
            WHERE LOWER(name) LIKE $1
            LIMIT 20""",
        [f"%{entities[0]}%"],
    ))

    # 5. Memory-entity JOIN with vector distance (simulated with importance ranking)
    queries.append((
        "entity_memory_ranked",
        f"""SELECT m.id, m.content, m.importance, me.entity_name
            FROM memories m
            JOIN memory_entities me ON m.id = me.memory_id
            WHERE me.user_id = $1
              AND LOWER(me.entity_name) IN ({in_clause})
            ORDER BY m.importance DESC
            LIMIT 50""",
        [user_id],
    ))

    # 6. Relationship traversal (2-hop in single query via recursive CTE)
    queries.append((
        "graph_2hop_cte",
        f"""WITH RECURSIVE neighborhood AS (
              SELECT r.id, r.source_id, r.target_id, r.relation_type, 1 AS depth
              FROM relationships_{COLLECTION} r
              JOIN entities_{COLLECTION} e ON e.id = r.source_id
              WHERE LOWER(e.name) IN ({in_clause}) AND e.user_id = $1
              UNION
              SELECT r.id, r.source_id, r.target_id, r.relation_type, n.depth + 1
              FROM relationships_{COLLECTION} r
              JOIN neighborhood n ON (r.source_id = n.target_id OR r.target_id = n.source_id)
              WHERE n.depth < 2 AND r.id != n.id
            )
            SELECT DISTINCT nb.id, nb.source_id, nb.target_id, nb.relation_type
            FROM neighborhood nb
            LIMIT 200""",
        [user_id],
    ))

    return queries


def add_indexes(conn):
    indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_entities_name_user ON entities_{COLLECTION}(user_id, name)",
        f"CREATE INDEX IF NOT EXISTS idx_entities_lower_name ON entities_{COLLECTION}(user_id, LOWER(name))",
        f"CREATE INDEX IF NOT EXISTS idx_me_name_user ON memory_entities(user_id, entity_name)",
        f"CREATE INDEX IF NOT EXISTS idx_me_lower_name ON memory_entities(user_id, LOWER(entity_name))",
        "CREATE INDEX IF NOT EXISTS idx_memories_user_event ON memories(user_id, event_time DESC)",
        "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)",
        f"CREATE INDEX IF NOT EXISTS idx_rel_src ON relationships_{COLLECTION}(source_id)",
        f"CREATE INDEX IF NOT EXISTS idx_rel_tgt ON relationships_{COLLECTION}(target_id)",
    ]
    for sql in indexes:
        try:
            conn.execute(sql)
        except Exception as e:
            print(f"  WARN: {sql[:60]}... → {e}")
    print(f"  Added {len(indexes)} indexes")


def run_benchmark():
    n_dbs = merge_databases()

    # Phase 1: Without indexes
    print()
    print("=" * 70)
    print("Phase 1: WITHOUT extra indexes (baseline)")
    print("=" * 70)

    conn = duckdb.connect(WORK_DB)
    entities = get_real_entities(conn)
    user_ids = get_real_user_ids(conn)
    user_id = user_ids[0]

    mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    ent_count = conn.execute(f"SELECT COUNT(*) FROM entities_{COLLECTION}").fetchone()[0]
    rel_count = conn.execute(f"SELECT COUNT(*) FROM relationships_{COLLECTION}").fetchone()[0]
    me_count = conn.execute("SELECT COUNT(*) FROM memory_entities").fetchone()[0]
    print(f"  {n_dbs} conversations merged")
    print(f"  Data: {mem_count} memories, {ent_count} entities, {rel_count} rels, {me_count} entity-links")
    print(f"  Test entities: {entities[:4]}...")
    print(f"  Test user: {user_id}")
    print()

    queries = build_queries(entities, user_id)
    baseline_results = []
    for name, sql, params in queries:
        r = bench_query(conn, name, sql, params)
        baseline_results.append(r)
        print(f"  {r['name']:30s}  mean={r['mean_ms']:8.3f}ms  med={r['median_ms']:8.3f}ms  "
              f"p95={r['p95_ms']:8.3f}ms  rows={r['result_count']}")

    conn.close()

    # Phase 2: With indexes
    print()
    print("=" * 70)
    print("Phase 2: WITH B-tree indexes")
    print("=" * 70)

    conn = duckdb.connect(WORK_DB)
    add_indexes(conn)
    print()

    queries = build_queries(entities, user_id)
    indexed_results = []
    for name, sql, params in queries:
        r = bench_query(conn, name, sql, params)
        indexed_results.append(r)
        print(f"  {r['name']:30s}  mean={r['mean_ms']:8.3f}ms  med={r['median_ms']:8.3f}ms  "
              f"p95={r['p95_ms']:8.3f}ms  rows={r['result_count']}")

    conn.close()

    # Comparison
    print()
    print("=" * 70)
    print("Comparison: speedup from indexes")
    print("=" * 70)
    total_before = 0
    total_after = 0
    for b, i in zip(baseline_results, indexed_results):
        speedup = b['mean_ms'] / i['mean_ms'] if i['mean_ms'] > 0 else float('inf')
        delta = b['mean_ms'] - i['mean_ms']
        marker = "✓" if speedup > 1.1 else "─"
        print(f"  {marker} {b['name']:28s}  {b['mean_ms']:8.3f}ms → {i['mean_ms']:8.3f}ms  "
              f"({speedup:.2f}x, {'-' if delta > 0 else '+'}{abs(delta):.3f}ms)")
        total_before += b['mean_ms']
        total_after += i['mean_ms']

    print(f"\n  Total query time: {total_before:.1f}ms → {total_after:.1f}ms "
          f"({total_before/total_after:.2f}x overall)")

    # Save
    results = {
        "n_conversations": n_dbs,
        "data_volumes": {
            "memories": mem_count, "entities": ent_count,
            "relationships": rel_count, "memory_entities": me_count,
        },
        "baseline": baseline_results,
        "indexed": indexed_results,
    }
    out_path = os.path.dirname(os.path.abspath(__file__)) + "/bench_index_perf_merged_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Cleanup
    os.remove(WORK_DB)
    replica = WORK_DB + ".replica"
    if os.path.exists(replica):
        os.remove(replica)


if __name__ == "__main__":
    run_benchmark()

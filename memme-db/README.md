# MemMe-DB

Custom DuckDB build with the **vex** vector search extension, optimized for MemMe's memory engine.

## What is this?

MemMe has two storage modes:

| Mode | Feature Flag | DuckDB Source | HNSW Index | Use Case |
|------|-------------|---------------|------------|----------|
| **Bundled** (default) | `bundled` | duckdb-rs compiles standard DuckDB | No | Quick start, development, CI |
| **MemMe-DB** | `memme-db` | This directory builds DuckDB + vex extension | Yes | Production, mobile, performance |

The `bundled` mode works out of the box — `cargo build` and done. MemMe-DB is for when you need HNSW vector indexing with in-graph filtering.

## vex Extension

The `vex/` directory contains a DuckDB extension that adds:

- **HNSW index** — `CREATE INDEX ... USING HNSW (embedding)` for approximate nearest neighbor search
- **Filtered HNSW** — `CREATE INDEX ... USING HNSW (embedding, user_id)` with ACORN-style in-graph metadata filtering
- **Distance functions** — `l2_distance()`, `cosine_distance()`, `inner_product()`
- **Product quantization** — Optional PQ compression for large-scale deployments

### Filtered Search Strategies

When a query includes a WHERE clause on metadata columns (e.g. `WHERE user_id = 'alice'`), the optimizer pushes the filter into the HNSW index. The strategy is chosen automatically based on selectivity:

| Selectivity | Strategy | Description |
|-------------|----------|-------------|
| < 1% | **Pre-filter** | Brute-force scan of matching nodes only |
| 1% - 90% | **In-graph (ACORN)** | Traverse graph for connectivity, only collect matching nodes |
| > 90% | **Post-filter** | Standard ANN search, over-fetch, then filter results |

### SQL Examples

```sql
-- Create table with vector column
CREATE TABLE memories (
    id UUID,
    user_id VARCHAR,
    content VARCHAR,
    embedding FLOAT[384]
);

-- HNSW index (pure vector search)
CREATE INDEX idx_mem ON memories USING HNSW (embedding)
    WITH (metric='cosine', m=32, ef_construction=128);

-- HNSW index with metadata (filtered search)
CREATE INDEX idx_mem_user ON memories USING HNSW (embedding, user_id)
    WITH (metric='cosine');

-- Vector search — optimizer automatically uses HNSW
SELECT id, content, cosine_distance(embedding, [0.1, 0.2, ...]::FLOAT[384]) AS dist
FROM memories
WHERE user_id = 'alice'
ORDER BY dist
LIMIT 10;
```

## How MemMe Uses It

When `memme-core` initializes with the `memme-db` feature, it:

1. Creates two HNSW indexes on the `memories` table:
   - `idx_mem_{collection}` — pure vector search (cosine, m=32, ef_construction=128)
   - `idx_mem_user_{collection}` — filtered search with `user_id` metadata
2. Vector search queries (`vector_search()`) automatically use the HNSW index when available, falling back to brute-force cosine distance if not

The index creation is graceful — if vex is not loaded (bundled mode), it logs a warning and continues without HNSW.

## Building

### Prerequisites

- CMake 3.14+
- C++ compiler (Clang or GCC)
- Git (for auto-downloading DuckDB)

### Build

```bash
./memme-db/build.sh release
```

This will:
1. Auto-download DuckDB v1.5.0 source (first run only, ~290MB)
2. Link the `vex` extension into DuckDB's extension directory
3. Compile DuckDB + vex + FTS + JSON into a single static library
4. Output: `memme-db/build/release/libduckdb_static.a`

### Use with MemMe

```bash
export DUCKDB_LIB_DIR=memme-db/build/release
export DUCKDB_INCLUDE_DIR=memme-db/duckdb/src/include
export DUCKDB_STATIC=1
cargo build -p memme-core --no-default-features --features memme-db
```

### What Gets Built

The static library includes:

| Component | Description |
|-----------|-------------|
| DuckDB core | SQL engine, storage, optimizer |
| vex extension | HNSW index, distance functions, filtered search |
| FTS extension | BM25 full-text search |
| JSON extension | JSON type and functions |
| core_functions | Basic SQL functions |

Excluded for size: parquet, icu, tpch, tpcds, autocomplete.

## Directory Structure

```
memme-db/
├── vex/                    # Vector search extension source
│   ├── index/              #   HNSW index implementation
│   ├── distance/           #   Distance functions (L2, cosine, IP)
│   ├── functions/          #   SQL function registration
│   ├── optimizer/          #   Query optimizer (filter push-down)
│   ├── quantizer/          #   Product quantization
│   └── include/            #   Header files
├── CMakeLists.txt          # Build configuration
├── build.sh                # Build script (auto-downloads DuckDB)
├── duckdb/                 # DuckDB source (gitignored, auto-downloaded)
└── build/                  # Build output (gitignored)
```

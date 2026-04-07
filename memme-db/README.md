# MemMe-DB

Custom DuckDB build with the **VSS** (Vector Similarity Search) extension, optimized for MemMe's memory engine.

## What is this?

MemMe has two storage modes:

| Mode | Feature Flag | DuckDB Source | HNSW Index | Use Case |
|------|-------------|---------------|------------|----------|
| **Bundled** (default) | `bundled` | duckdb-rs compiles standard DuckDB | No | Quick start, development, CI |
| **MemMe-DB** | `memme-db` | This directory builds DuckDB + VSS extension | Yes | Production, mobile, performance |

The `bundled` mode works out of the box — `cargo build` and done. MemMe-DB is for when you need HNSW vector indexing with filtered search.

## VSS Extension

The `vss/` directory is based on the [official DuckDB VSS extension](https://github.com/duckdb/duckdb_vss) (v1.5 branch), powered by the [usearch](https://github.com/unum-cloud/usearch) library. MemMe adds filtered search support on top:

- **HNSW index** — `CREATE INDEX ... USING HNSW (embedding)` for approximate nearest neighbor search
- **Filtered search** — `WHERE user_id = 'alice' ORDER BY distance LIMIT k` automatically uses usearch's `filtered_search` with in-graph predicate callback
- **Distance functions** — `array_cosine_distance()`, `array_distance()`, `array_negative_inner_product()`

### Filtered Search

When a query includes `WHERE col = constant ORDER BY distance LIMIT k`, the optimizer:
1. Extracts equality filters from the WHERE clause
2. Pre-scans the table to collect matching row IDs
3. Passes them as a predicate callback to `usearch::filtered_search`
4. usearch evaluates the predicate inline during HNSW graph traversal

This avoids post-filtering which can miss relevant results at low selectivity.

### SQL Examples

```sql
-- Create table with vector column
CREATE TABLE memories (
    id UUID,
    user_id VARCHAR,
    content VARCHAR,
    embedding FLOAT[384]
);

-- HNSW index
CREATE INDEX idx_mem ON memories USING HNSW (embedding)
    WITH (metric='cosine', m=32, ef_construction=128);

-- Vector search with filter — optimizer automatically uses HNSW + filtered_search
SELECT id, content, array_cosine_distance(embedding, [0.1, 0.2, ...]::FLOAT[384]) AS dist
FROM memories
WHERE user_id = 'alice'
ORDER BY dist
LIMIT 10;
```

## How MemMe Uses It

When `memme-core` initializes with the `memme-db` feature, it:

1. Creates an HNSW index on the `memories` table:
   - `idx_mem_{collection}` — cosine metric, m=32, ef_construction=128
2. Vector search queries (`vector_search()`) automatically use the HNSW index when available, falling back to brute-force cosine distance if not

The index creation is graceful — if VSS is not loaded (bundled mode), it logs a warning and continues without HNSW.

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
2. Link the VSS extension into DuckDB's extension directory
3. Compile DuckDB + VSS + FTS + JSON into a single static library
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
| VSS extension | HNSW index (usearch), filtered search, distance functions |
| FTS extension | BM25 full-text search |
| JSON extension | JSON type and functions |
| core_functions | Basic SQL functions |

Excluded for size: parquet, icu, tpch, tpcds, autocomplete.

## Directory Structure

```
memme-db/
├── vss/                    # VSS extension source (based on duckdb-vss)
│   ├── hnsw/               #   HNSW index, optimizer, scan, pragmas
│   ├── include/             #   Headers (usearch, hnsw, simsimd, fp16)
│   ├── CMakeLists.txt       #   Extension build config
│   └── vss_extension.cpp    #   Entry point
├── CMakeLists.txt          # Build configuration
├── build.sh                # Build script (auto-downloads DuckDB)
├── duckdb/                 # DuckDB source (gitignored, auto-downloaded)
└── build/                  # Build output (gitignored)
```

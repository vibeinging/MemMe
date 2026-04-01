English | [中文](README_CN.md)

# MemMe

**The first embeddable AI memory engine for mobile and edge devices.**

> An open-source alternative to [mem0](https://github.com/mem0ai/mem0) — offline-first, single-file, sub-10ms latency. Built in Rust with Python, Node.js, and Swift bindings.

[![CI](https://github.com/vibeinging/MemMe/actions/workflows/ci.yml/badge.svg)](https://github.com/vibeinging/MemMe/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/memme-core.svg)](https://crates.io/crates/memme-core)

## What is MemMe?

MemMe is an AI memory layer that gives your agents and apps long-term memory for AI agents, personal AI assistants, and conversational memory systems. Built in Rust on top of DuckDB, it combines a vector database, knowledge graph, and full-text search into a single embeddable file — no external services required.

Unlike cloud-based solutions, MemMe is designed for on-device AI and edge AI scenarios where offline AI capability matters. It handles embedding, RAG-style hybrid retrieval, and memory consolidation locally, making it ideal for mobile apps, IoT devices, and privacy-sensitive deployments.

- **All-in-one storage** — vectors, knowledge graph, full-text index, and change history in a single file
- **Works offline** — LLM is optional and pluggable via the `LlmProvider` trait (Ollama, OpenAI, Anthropic, or none)
- **Sub-10ms without LLM** — pure vector dedup path for latency-critical scenarios
- **Multi-platform bindings** — Python (PyO3), Node.js (NAPI-RS), Swift (UniFFI), WASM

## Key Features

- **Single-file deployment** -- one `.duckdb` file holds everything: vectors, graph, FTS index, history
- **Pluggable LLM** -- works without LLM (<10ms vector dedup) or with any LLM (fact extraction + smart dedup)
- **Knowledge graph** -- entity/relationship extraction with DuckPGQ graph traversal
- **Hybrid search** -- vector similarity + BM25 full-text search fused via Reciprocal Rank Fusion
- **Forgetting curve** -- Ebbinghaus-inspired memory decay with stability reinforcement on access
- **Four-level scoping** -- `user_id` / `agent_id` / `app_id` / `run_id` isolation
- **Privacy controls** -- `LocalOnly`, `Syncable`, `EncryptedSync` per-memory privacy levels
- **Battery-aware processing** -- defers heavy operations when device battery is critical
- **Immutable memories + TTL** -- lock memories from modification or set auto-expiration dates
- **Advanced filters** -- Eq/Ne/Gt/Gte/Lt/Lte/In/Contains/IContains with AND/OR combinators
- **Export/Import** -- full JSON export including memories, entities, and relationships
- **Multi-language** -- Rust core + Python (PyO3) + Node.js (NAPI-RS) + Swift (UniFFI) + WASM

## Benchmark — AI Memory Accuracy (LoCoMo, 1540 questions, gpt-4o-mini judge)

| Category | **MemMe** | mem0 | mem0-graph | Zep |
|---|---|---|---|---|
| **Single-hop** | **80.50** | 67.13 | 65.71 | 61.70 |
| **Multi-hop** | **55.76** | 51.15 | 47.19 | 41.35 |
| **Temporal** | **59.38** | 55.51 | 58.13 | 49.31 |
| **Open-domain** | **74.55** | 72.93 | 75.71 | 76.60 |

MemMe outperforms mem0 across **all four categories** on the [LoCoMo benchmark](https://github.com/snap-stanford/locomo). Retrieval pipeline: 4-channel search (vector + BM25 + entity spreading + temporal) with RRF fusion and cross-encoder reranking.

## Why MemMe vs mem0 / Zep — Feature Comparison

| | **MemMe** | **mem0** | **Zep** |
|---|---|---|---|
| **Deployment** | Single `.duckdb` file | Server + Qdrant/Pinecone + Neo4j | Managed cloud |
| **Mobile / iOS** | Native (UniFFI) | No | No |
| **Offline** | Full offline support | Requires cloud APIs | Cloud only |
| **Without-LLM latency** | <10ms | Always needs LLM | Always needs LLM |
| **Language** | Rust core | Python only | Go (server) |
| **Storage** | Embedded DuckDB | External vector DB + graph DB | Managed |
| **Knowledge graph** | Built-in (DuckPGQ) | External Neo4j | No |
| **FTS + hybrid search** | Built-in (RRF) | No | Partial |
| **Reranking** | Built-in (API / ONNX) | Optional (Cohere) | No |
| **Forgetting curve** | Built-in | No | No |
| **Privacy scoping** | Per-memory privacy levels | No | SOC2/HIPAA (cloud) |

## Try It — Interactive Playground

No code needed. Just run:

```bash
pip install memme
python demos/playground/server.py
```

A local web app opens in your browser. Add memories, search by meaning, chat with your memory store. All data stays in a local `.duckdb` file on your machine.

See [demos/playground/README.md](demos/playground/README.md) for details.

## Quick Start

### Rust

```toml
[dependencies]
memme-core = "0.1"
memme-embeddings = { version = "0.1", features = ["onnx"] }
```

```rust
use std::sync::Arc;
use memme_core::{MemoryConfig, MemoryStore, AddOptions, SearchOptions};
use memme_embeddings::onnx::OnnxEmbedder;

fn main() -> memme_core::Result<()> {
    let config = MemoryConfig::new("memory.duckdb", 384);
    let embedder = Arc::new(OnnxEmbedder::new()?);
    let store = MemoryStore::new(config, embedder)?;

    store.add("User prefers dark mode", AddOptions::new("alice"))?;
    store.add("User drinks coffee every morning", AddOptions::new("alice"))?;

    let results = store.search("morning routine", SearchOptions::new("alice").limit(5))?;
    for r in &results {
        println!("{} (score: {:.4})", r.content, r.score.unwrap_or(0.0));
    }
    Ok(())
}
```

### Python

```bash
pip install memme
```

```python
from memme import MemoryStore

store = MemoryStore("memory.duckdb")  # uses local ONNX embeddings by default
store.add("User prefers dark mode", user_id="alice")
results = store.search("preferences", user_id="alice")
for r in results:
    print(r["content"], r["score"])
```

### Node.js

```bash
npm install memme
```

```javascript
const { MemoryStore } = require("memme");

const store = new MemoryStore("memory.duckdb");
await store.add("User prefers dark mode", { userId: "alice" });
const results = await store.search("preferences", { userId: "alice" });
console.log(results);
```

### Swift (UniFFI)

```swift
import MemMe

let store = try MemoryStore(dbPath: "memory.duckdb", embedder: "onnx")
try store.add("User prefers dark mode", userId: "alice")
let results = try store.search("preferences", userId: "alice")
```

## Architecture — How MemMe Works

```
┌──────────────────────────────────────────────────────────────┐
│                      Language Bindings                        │
│  Python (PyO3)  │  Swift (UniFFI)  │  Node.js (NAPI-RS)     │
│  memme-python      memme-ffi          memme-node             │
├──────────────────────────────────────────────────────────────┤
│                      memme-server (axum REST API)            │
│                      memme-mcp (MCP stdio server)            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                    memme-core (Rust)                          │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────────┐    │
│  │ MemoryStore │  │ GraphStore │  │ SearchEngine         │    │
│  │ add()       │  │ add_graph()│  │ vector search        │    │
│  │ search()    │  │ entities   │  │ FTS (BM25)           │    │
│  │ update_trace│  │ relations  │  │ hybrid RRF           │    │
│  │ delete_trace│  │ traverse   │  │ rerank (API / ONNX)   │    │
│  └─────┬──────┘  └─────┬──────┘  └──────────┬──────────┘    │
│        │               │                     │               │
│  ┌─────┴───────────────┴─────────────────────┴─────────┐     │
│  │          DuckDB Storage Layer (.duckdb file)         │     │
│  │  memories │ entities │ relationships │ history        │     │
│  │  FTS index │ vector index (MemMe-DB optional)      │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌──────────────────┐   ┌──────────────────────────────┐     │
│  │ memme-embeddings  │   │ memme-llm                    │     │
│  │ ├ OnnxEmbedder    │   │ ├ OllamaProvider             │     │
│  │ ├ OpenAIEmbedder  │   │ ├ OpenAIProvider              │     │
│  │ └ OllamaEmbedder  │   │ ├ AnthropicProvider           │     │
│  └──────────────────┘   │ ├ GeminiProvider              │     │
│                          │ └ NoopProvider (fallback)     │     │
│                          └──────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

**Workspace crates:**

| Crate | Purpose |
|---|---|
| `memme-core` | Core memory engine: CRUD, search, graph, dedup, analytics |
| `memme-embeddings` | Embedding trait + ONNX/OpenAI/Ollama backends |
| `memme-llm` | LLM trait + Ollama/OpenAI/Anthropic/Gemini backends for fact extraction |
| `memme-python` | Python bindings via PyO3 + maturin |
| `memme-ffi` | Swift/C bindings via UniFFI |
| `memme-node` | Node.js bindings via NAPI-RS |
| `memme-wasm` | WASM bindings via wasm-bindgen |
| `memme-server` | REST API server (axum) |
| `memme-mcp` | MCP stdio server for Claude Desktop / Cursor |

## API Overview

| Method | Description |
|---|---|
| `add(content, AddOptions)` | Add a memory with automatic dedup |
| `add_smart(messages, AddOptions)` | LLM-driven fact extraction and memory management |
| `search(query, SearchOptions)` | Vector similarity search |
| `hybrid_search(query, HybridSearchOptions)` | Vector + FTS with RRF fusion |
| `get(id)` | Get a single memory by ID |
| `update_trace(id, content, UpdateOptions)` | Update memory content |
| `delete_trace(id)` | Delete a memory |
| `list_traces(ListOptions)` | List memories filtered by user/agent/app/run |
| `history(memory_id)` | Get change history for a memory |
| `batch_update(ids, contents)` | Batch update (skips immutable) |
| `batch_delete(ids)` | Batch delete (skips immutable) |
| `add_graph(text, user_id, llm)` | Extract entities/relationships into knowledge graph |
| `search_graph(query, user_id)` | Search the knowledge graph |
| `consolidate()` | Decay retention, prune expired/low-retention memories |
| `export(user_id) / import(data)` | Full JSON export/import |
| `user_stats(user_id)` | OLAP analytics per user |
| `memory_frequency(user_id)` | Memory creation trends |
| `top_entities(user_id)` | Most mentioned knowledge graph entities |
| `add_procedure / get_procedure` | Procedural memory (skills, workflows) |
| `export_changes_since(version)` | Incremental sync deltas |
| `storage_stats()` | Memory count, entity count, estimated DB size |

## Configuration

```rust
let config = MemoryConfig {
    db_path: "memory.duckdb".into(),       // or ":memory:"
    collection_name: "default".into(),      // table prefix
    embedding_dims: 384,                    // must match embedder
    dedup_threshold: 0.15,                  // cosine distance for dedup
    default_limit: 10,                      // default search/list limit
    enable_graph: true,                     // auto graph extraction in smart mode
    enable_forgetting_curve: true,          // Ebbinghaus decay
    retention_weight: 0.7,                  // weight of retention in scoring
    max_memories_per_user: Some(1000),      // auto-prune when exceeded
    pruning_strategy: PruningStrategy::LRU, // or Importance, Decay
    auto_prune: true,                       // prune on add() when over limit
    inclusion_prompt: Some("Extract work-related tasks".into()),
    exclusion_prompt: Some("Ignore passwords and secrets".into()),
    power_config: Some(PowerConfig {        // mobile battery awareness
        full_power_threshold: 0.5,
        power_save_threshold: 0.2,
        defer_when_critical: true,
    }),
    ..Default::default()
};
```

## REST API

Start the server:

```bash
cargo run -p memme-server -- --db memory.duckdb --port 8080
```

Example requests:

```bash
# Add a memory
curl -X POST http://localhost:8080/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "User likes dark mode", "user_id": "alice"}'

# Search
curl -X POST http://localhost:8080/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "UI preferences", "user_id": "alice", "limit": 5}'

# Get by ID
curl http://localhost:8080/v1/memories/{id}

# Update
curl -X PUT http://localhost:8080/v1/memories/{id} \
  -H "Content-Type: application/json" \
  -d '{"content": "User switched to light mode"}'

# Delete
curl -X DELETE http://localhost:8080/v1/memories/{id}

# List
curl -X POST http://localhost:8080/v1/memories/list \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice"}'

# History
curl http://localhost:8080/v1/memories/{id}/history

# Export / Import
curl -X POST http://localhost:8080/v1/memories/export \
  -d '{"user_id": "alice"}'
curl -X POST http://localhost:8080/v1/memories/import \
  -H "Content-Type: application/json" \
  -d @exported.json

# Health check
curl http://localhost:8080/health
```

## MCP Server

MemMe includes an MCP (Model Context Protocol) stdio server for integration with Claude Desktop, Cursor, and other MCP clients.

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "memme": {
      "command": "/path/to/memme-mcp",
      "args": ["--db", "memory.duckdb"]
    }
  }
}
```

Available MCP tools: `add_memory`, `search_memory`, `get_memory`, `update_memory`, `delete_memory`, `list_memories`, `delete_all_memories`.

Build the MCP server:

```bash
cargo build -p memme-mcp --release
```

## Building from Source

### Default (bundled DuckDB)

```bash
git clone --recurse-submodules https://github.com/vibeinging/MemMe.git
cd MemMe
cargo build --release
```

### With MemMe-DB (HNSW)

For advanced vector indexing with partition-level filtering:

```bash
# 1. Build DuckDB + MemMe-DB static library
./scripts/build_memme_db.sh

# 2. Build MemMe with memme-db feature
export DUCKDB_LIB_DIR=<path>/build/release/src
export DUCKDB_INCLUDE_DIR=<path>/src/include
cargo build -p memme-core --no-default-features --features memme-db --release
```

### Feature flags

| Feature | Description |
|---|---|
| `bundled` (default) | Compile DuckDB from source |
| `memme-db` | Link precompiled DuckDB + MemMe-DB |
| `api-rerank` | API-based cross-encoder reranker (Jina/Cohere/DashScope) |
| `onnx-rerank` | Local ONNX cross-encoder reranker via fastembed |

### Run tests

```bash
cargo test                                    # all unit tests (244)
```

### Build Python package

```bash
cd crates/memme-python
maturin develop --release
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- [Report a bug](https://github.com/vibeinging/MemMe/issues/new?template=bug_report.md)
- [Request a feature](https://github.com/vibeinging/MemMe/issues/new?template=feature_request.md)
- [Roadmap](docs/ROADMAP.md)

## Use Cases

- **AI Agents** — persistent memory across conversations for chatbots and autonomous agents
- **Personal AI Assistants** — remember user preferences, habits, and context on-device
- **Mobile Apps** — offline-first memory that works without internet, syncs when connected
- **RAG Pipelines** — hybrid retrieval (vector + graph + FTS) as a local knowledge base
- **Digital Twins** — build a memory-powered digital representation of a person

## Community

- [GitHub Discussions](https://github.com/vibeinging/MemMe/discussions) — questions & ideas
- [Issue Tracker](https://github.com/vibeinging/MemMe/issues) — bug reports & feature requests

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.

English | [中文](README_CN.md)

<div align="center">

# MemMe

**Memories that are actually yours.**

An embeddable AI memory engine. One file. Your device. Your rules.

[![CI](https://github.com/vibeinging/MemMe/actions/workflows/ci.yml/badge.svg)](https://github.com/vibeinging/MemMe/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/memme-core.svg)](https://crates.io/crates/memme-core)

</div>

---

MemMe gives AI agents and apps long-term memory — stored in a single `.duckdb` file on your device, not in the cloud. Built in Rust with native bindings for Python, Node.js, and Swift/Kotlin (via UniFFI).

Vectors, knowledge graph, full-text search, and change history — all in one file. No Qdrant, no Neo4j, no infrastructure. Plug in any LLM for smart extraction, or run pure vector mode at sub-10ms latency without one.

## Try It Now

```bash
pip install memme
python demos/playground/server.py
```

A local web app opens in your browser. Store memories, search by meaning, chat with your memory. All data stays on your machine. See [playground docs](demos/playground/README.md).

## Benchmark

MemMe vs mem0 on the [LoCoMo benchmark](https://github.com/snap-stanford/locomo) (1540 questions, 10 conversations, GPT-4o-mini judge):

| Category | **MemMe** | mem0 | mem0-graph | Zep |
|---|---|---|---|---|
| Single-hop | **85.46** | 67.13 | 65.71 | 61.70 |
| Multi-hop | **57.32** | 51.15 | 47.19 | 41.35 |
| Temporal | **66.67** | 55.51 | 58.13 | 49.31 |
| Open-domain | **85.26** | 72.93 | 75.71 | 76.60 |
| **Overall** | **78.31** | — | — | — |

Pipeline: append_events → compact → meditate (per-episode fact extraction + vector dedup). 4-channel retrieval (vector + BM25 + entity spreading + temporal) with RRF fusion and cross-encoder reranking.

## Why MemMe

| | **MemMe** | **mem0** | **Zep** |
|---|---|---|---|
| Deployment | Single `.duckdb` file | Server + Qdrant + Neo4j | Managed cloud |
| Mobile / iOS | Native (UniFFI) | No | No |
| Offline | Full support | Requires cloud APIs | Cloud only |
| Latency (no LLM) | <10ms | Always needs LLM | Always needs LLM |
| Language | Rust core | Python only | Go (server) |
| Knowledge graph | Built-in (DuckDB) | External Neo4j | No |
| Hybrid search | Vector + BM25 + RRF | No | Partial |
| Reranking | Built-in (API / ONNX) | Optional | No |
| Forgetting curve | Built-in | No | No |
| Data protection | Dual-replica + full export | No | SOC2/HIPAA (cloud) |
| Chat import | ChatGPT / Claude / Gemini | No | No |

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

store = MemoryStore("memory.duckdb")
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

// Use OpenAI embeddings (or newMock() for testing without API)
const store = MemoryStore.newOpenai(process.env.OPENAI_API_KEY, "memory.duckdb");
await store.add("User prefers dark mode", "alice");
const results = await store.search("preferences", "alice");
console.log(results);
```

### Swift (UniFFI)

```swift
import MemMe

// Host app provides HTTP transport (URLSession, OkHttp, etc.)
let store = try MemoryStore.newWithHttpClient(
    dbPath: "memory.duckdb",
    httpClient: myHttpClient,  // implements HttpClient protocol
    apiKey: "sk-...",
    model: "text-embedding-3-small",
    dims: 1536
)
try store.add("User prefers dark mode", userId: "alice")
let results = try store.search("preferences", userId: "alice")
```

## Features

### Memory Pipeline

- **Session/Episode architecture** — Stream -> Session -> Episode -> Memory four-layer data model
- **Meditation** — memory consolidation: decay + fact extraction + LLM reconciliation (ADD/UPDATE/DELETE) + graph building + entity-memory linking
- **Forgetting curve** — FSRS-based memory decay with stability reinforcement on access
- **Knowledge graph** — entity/relationship extraction with spreading activation traversal
- **Reflection** — LLM-powered analysis of recent memories, identifying themes, patterns, and focus areas
- **Feedback learning** — extract behavioral principles from user corrections, stored as high-importance memories and identity traits

### Search

- **Four-channel hybrid search** — vector + BM25 + entity graph + temporal, fused via RRF
- **Reranking** — API reranker (Jina/Cohere) or local ONNX cross-encoder

### Data Safety

- **Single-file deployment** — one `.duckdb` file holds vectors, graph, FTS index, and history
- **Dual-replica protection** — CHECKPOINT + atomic file copy before meditation; auto-recovery from replica on corruption
- **Full export/import** — export all 8 data layers (memories, sessions, events, episodes, entities, relations, identity, sources) as JSON; import back losslessly
- **External chat import** — import conversations from ChatGPT, Claude, and Gemini exports into the memory pipeline
- **Privacy controls** — `LocalOnly`, `Syncable`, `EncryptedSync` per-memory

### Platform

- **Pluggable LLM** — OpenAI, Anthropic, Gemini, Ollama, or none
- **Diagnostics** — built-in health checks for storage, embedder, and LLM with per-check latency
- **Four-level scoping** — `user_id` / `agent_id` / `app_id` / `run_id` isolation
- **Battery-aware** — defers heavy operations on low battery
- **Analytics** — user stats, memory frequency, top entities (powered by DuckDB OLAP)

## Import Your Chat History

Bring your existing conversations from other AI platforms:

```rust
use memme_core::import::{parse_chatgpt, parse_claude_export, parse_gemini};

// Parse ChatGPT export (conversations.json from Settings > Export)
let convs = parse_chatgpt(&std::fs::read_to_string("conversations.json")?)?;

// Parse Claude export (conversations.jsonl from Settings > Export Data)
let convs = parse_claude_export(&std::fs::read_to_string("conversations.jsonl")?)?;

// Parse Gemini export (from Google Takeout)
let convs = parse_gemini(&std::fs::read_to_string("gemini_export.json")?)?;

// Ingest into MemMe → then compact & meditate to extract memories
store.import_conversations(&convs, "alice")?;
```

## Data Protection

MemMe maintains a dual-replica system to ensure your memories are never lost:

```rust
// Sync primary → replica (CHECKPOINT + atomic file copy)
store.sync_replica()?;

// Check status of both copies
let status = store.replica_status()?;
println!("Primary: {} bytes, Replica: {:?} bytes", status.primary_size_bytes, status.replica_size_bytes);

// Full export: all 8 data layers as a single JSON
let export = store.full_export(Some("alice"))?;
std::fs::write("backup.json", serde_json::to_string_pretty(&export)?)?;

// Full import: restore everything
let data: FullExport = serde_json::from_str(&std::fs::read_to_string("backup.json")?)?;
store.full_import(&data)?;
```

Meditation automatically syncs the replica before starting. If the primary file corrupts, MemMe auto-recovers from the replica on next startup.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Language Bindings                      │
│  Python (PyO3)  │  Node.js (NAPI-RS)  │  Swift (UniFFI) │
├──────────────────────────────────────────────────────────┤
│  REST API (axum)          │  MCP Server (stdio)          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│                   memme-core (Rust)                       │
│                                                          │
│  Stream ──► Session ──► Episode ──► Memory                │
│                    compact    meditate  │                 │
│                            ┌───────────┤                 │
│                            ▼           ▼                 │
│                        Identity      Graph               │
│                                                          │
│  Search: Vector + BM25 + Graph + Temporal                │
│          ──► RRF Fusion ──► Rerank                       │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  DuckDB (.duckdb single file + .replica backup)    │  │
│  │  memories │ entities │ relationships │ episodes     │  │
│  │  sessions │ events │ identity │ history │ meditations│  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  memme-embeddings          memme-llm                     │
│  (ONNX / OpenAI / Ollama)  (OpenAI / Anthropic / Gemini  │
│                             / Ollama / Noop)             │
└──────────────────────────────────────────────────────────┘
```

| Crate | Purpose |
|---|---|
| `memme-core` | Core engine: CRUD, search, graph, meditation, replica, export/import |
| `memme-embeddings` | Embedding trait + ONNX/OpenAI/Ollama backends |
| `memme-llm` | LLM trait + OpenAI/Anthropic/Gemini/Ollama backends |
| `memme-python` | Python bindings (PyO3 + maturin) |
| `memme-ffi` | Swift/C bindings (UniFFI) |
| `memme-node` | Node.js bindings (NAPI-RS) |
| `memme-wasm` | WASM bindings (wasm-bindgen) |
| `memme-server` | REST API server (axum) |
| `memme-mcp` | MCP server for Claude Desktop / Cursor |

## REST API

```bash
cargo run -p memme-server -- --db-path memory.duckdb --port 8080
```

```bash
# Add a memory
curl -X POST http://localhost:8080/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "User likes dark mode", "user_id": "alice"}'

# Search
curl -X POST http://localhost:8080/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "UI preferences", "user_id": "alice", "limit": 5}'

# Hybrid search (vector + keyword)
curl -X POST http://localhost:8080/v1/memories/hybrid-search \
  -H "Content-Type: application/json" \
  -d '{"query": "coffee", "user_id": "alice"}'
```

Full API reference: [docs/openapi.yaml](docs/openapi.yaml) — paste into [Swagger Editor](https://editor.swagger.io) to browse all 23 endpoints.

## MCP Server

For Claude Desktop, Cursor, and other MCP clients:

```json
{
  "mcpServers": {
    "memme": {
      "command": "/path/to/memme-mcp",
      "args": ["--db-path", "memory.duckdb"]
    }
  }
}
```

```bash
cargo build -p memme-mcp --release
```

## Building from Source

```bash
git clone --recurse-submodules https://github.com/vibeinging/MemMe.git
cd MemMe
cargo build --release
cargo test   # 340+ tests
```

### Feature Flags

| Feature | Description |
|---|---|
| `bundled` (default) | Compile DuckDB from source |
| `memme-db` | Link precompiled DuckDB + MemMe-DB (HNSW) |
| `api-rerank` | API-based reranker (Jina/Cohere) |
| `onnx-rerank` | Local ONNX cross-encoder reranker |

### Build Python Package

```bash
cd crates/memme-python && maturin develop --release
```

## Use Cases

- **AI Agents** — persistent memory across conversations
- **Personal AI** — remember preferences, habits, and context on-device
- **Mobile Apps** — offline-first memory that syncs when connected
- **RAG Pipelines** — local hybrid retrieval as a knowledge base
- **Digital Twins** — memory-powered digital representation of a person
- **Embodied AI** — sub-10ms on-device memory for robots, drones, and IoT; single-file deployment with no network dependency; native Rust integrations with Dora-rs, LeRobot, and Copper-rs
- **Chat Migration** — import your ChatGPT/Claude/Gemini history, own your data

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. [Roadmap](docs/ROADMAP.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).

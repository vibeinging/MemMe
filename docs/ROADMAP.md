# MemMe Roadmap

> Your memories, truly yours.
>
> Edge-first AI memory engine — Rust core, single DuckDB file, multi-language native bindings.
>
> Last updated: 2026-04-01

---

## What's Shipped

MemMe 0.1.0 is a fully functional AI memory engine. Here's what's already working:

### Core Engine

| Feature | Description |
|---------|-------------|
| **Stream → Session → Episode → Memory** | Four-layer data model. Raw events compact into episodes, episodes extract into semantic memories |
| **Meditation** | Memory consolidation: forgetting curve decay + episode→memory extraction + graph building + identity distillation |
| **Knowledge Graph** | Entity/relationship extraction (LLM), DuckDB storage, spreading activation search |
| **Four-Channel Hybrid Retrieval** | Vector + BM25 FTS + Entity Graph + Temporal, fused via Reciprocal Rank Fusion |
| **Reranker** | LLM reranker, ONNX cross-encoder (fastembed), API reranker (Jina/Cohere compatible) |
| **FSRS Forgetting Curve** | `R(t,S) = (1 + t/(c*S))^(-p)`, stability reinforcement on access |
| **Identity Traits** | High-level personality distillation from memory corpus |
| **Procedural Memory** | Skill/habit/workflow storage and retrieval |
| **Smart Mode** | LLM fact extraction → vector dedup → ADD/UPDATE/DELETE decisions |
| **Analytics (OLAP)** | user_stats, memory_frequency, top_entities — powered by DuckDB's columnar engine |
| **Privacy Controls** | Per-memory privacy levels: LocalOnly / Syncable / EncryptedSync |
| **Four-Level Scoping** | user_id + agent_id + app_id + run_id isolation |
| **Lifecycle Management** | TTL expiration, LRU/importance/decay pruning, storage limits, deferred writes |
| **Incremental Sync** | export_changes_since + sync_version tracking |
| **Webhook Events** | memory_add/update/delete HTTP POST callbacks |

### Multi-Language Bindings

| Binding | Package | Status |
|---------|---------|--------|
| **Rust** | `memme-core` on crates.io | Shipping |
| **Python** | `memme` on PyPI (PyO3 + maturin) | Shipping |
| **Node.js / TypeScript** | `memme` on npm (NAPI-RS) | Shipping |
| **Swift / Kotlin** | UniFFI .xcframework | Shipping |
| **WebAssembly** | wasm-bindgen | Experimental |

### Infrastructure

| Component | Description |
|-----------|-------------|
| **REST API Server** | axum, 23 endpoints, bearer token auth |
| **MCP Server** | stdio JSON-RPC for Claude Desktop / Cursor |
| **CI/CD** | GitHub Actions: fmt + clippy + tests on Ubuntu/macOS |
| **Publish Pipelines** | crates.io, PyPI, npm, iOS/Android binary workflows |

### Benchmark (LoCoMo, 1540 questions, GPT-4o-mini judge)

| Category | MemMe | mem0 | mem0-graph | Zep |
|----------|-------|------|------------|-----|
| Single-hop | **80.50** | 67.13 | 65.71 | 61.70 |
| Multi-hop | **55.76** | 51.15 | 47.19 | 41.35 |
| Temporal | **59.38** | 55.51 | 58.13 | 49.31 |
| Open-domain | **74.55** | 72.93 | 75.71 | 76.60 |

---

## Roadmap

Organized by priority. Each milestone is designed to unblock the next wave of adoption.

### M0: Production Hardening (Before Public Launch)

> Make the existing engine bulletproof. No new features — just fix what's fragile.

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| LLM transaction safety | P0 | ✅ Done | `add_smart()` wrapped in BEGIN/COMMIT with ROLLBACK on failure |
| LLM/Embedding retry | P0 | ✅ Done | Exponential backoff (3 retries) on all providers (OpenAI + Ollama) |
| RRF candidate limit | P1 | ✅ Done | Bounded by `rrf_candidate_multiplier` config |
| LLM call timeout | P1 | ✅ Done | OpenAI 120s, Ollama 600s at HTTP client level |
| Deferred writes flush | P1 | ✅ Done | Queue cap 500 + time-based flush (30s interval) |
| API naming consistency | P1 | ✅ Done | Unified `top_k` → `limit` across server/MCP/cu-memme/json_ops |

### M1: Developer Experience (Week 1-2 after launch)

> Make it trivially easy to try and adopt.

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| **Rustdoc on public APIs** | P0 | ✅ Done | `///` doc comments on all public types and methods in lib.rs |
| **Error type docs** | P2 | ✅ Done | All MemoryError variants documented in error.rs |
| **OpenAPI spec** | P0 | ✅ Done | docs/openapi.yaml — OpenAPI 3.1 covering all 23 endpoints |
| **Local Playground** | P1 | ✅ Done | `python demos/playground/server.py` — Remember/Recall/Chat modes |
| **WASM Playground** | P2 | ❌ Open | Browser-only demo via DuckDB-WASM (future) |
| **Quickstart guides** | P1 | ⚠️ Partial | README has code examples per language, but no dedicated one-page guides |

### M2: Data Import (Week 3-4)

> Bring your own history. The killer adoption driver.

| Task | Priority | Description |
|------|----------|-------------|
| **ChatGPT history import** | P0 | Parse `conversations.json` from ChatGPT export → Session/Episode model |
| **Claude history import** | P0 | Parse Claude export format → Session/Episode model |
| **Generic conversation import** | P1 | Standard JSON/JSONL schema for any chat history |
| **Import CLI tool** | P1 | `memme import --format chatgpt --file conversations.json --db memory.duckdb` |
| **Import progress & stats** | P2 | Show progress bar, report: conversations imported, memories extracted, entities found |

### M3: Ecosystem Integration (Week 5-8)

> Meet developers where they already are.

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| **OpenClaw plugin** | P1 | ⚠️ Partial | Tools scaffolded (store/recall/forget), hooks not yet wired |
| **LangChain Memory** | P0 | ❌ Open | `MemMeMemory` class implementing LangChain's BaseMemory |
| **LlamaIndex integration** | P1 | ❌ Open | MemMe as retriever/storage backend |
| **CrewAI / AutoGen** | P1 | ❌ Open | Shared memory for multi-agent frameworks |
| **Obsidian plugin** | P2 | ❌ Open | Two-way sync: Obsidian vault ↔ MemMe memories |

### M4: Edge & Embodied AI (Week 9-12)

> The moat: MemMe runs where others can't.

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| **LeRobot integration** | P0 | ⚠️ Partial | MemoryRobot wrapper + episode logging scaffolded |
| **Copper-rs CuTask** | P1 | ⚠️ Partial | cu-memme crate with message types, needs full CuTask impl |
| **iOS SDK (CocoaPods/SPM)** | P1 | ❌ Open | Pre-built .xcframework + package manager distribution |
| **Android SDK (Maven/Gradle)** | P1 | ❌ Open | AAR package with Kotlin DSL |
| **On-device LLM** | P2 | ❌ Open | Integrate llama.cpp / MLX for fully offline smart mode |
| **Raspberry Pi benchmark** | P2 | ❌ Open | Prove edge deployment with real performance numbers |

### M5: Advanced Memory Science (Week 13-20)

> Push the frontier of what AI memory can do.

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| **Sleep-time compute** | P1 | ⚠️ Partial | Meditation system covers decay + extraction; missing background scheduling |
| **Vision memory** | P1 | ❌ Open | Multimodal: extract and store memories from images |
| **Causal reasoning** | P2 | ❌ Open | Track cause-effect chains across memories |
| **Emotion tagging** | P2 | ❌ Open | Sentiment/emotion extraction and retrieval |
| **Memory clustering** | P2 | ❌ Open | Auto-discover topic clusters, generate memory maps |
| **Multi-agent isolation** | P2 | ❌ Open | Full per-agent memory spaces with optional sharing policies |

### M6: Scale & Performance (Ongoing)

| Task | Priority | Description |
|------|----------|-------------|
| **Connection pool optimization** | P1 | WAL mode for read/write parallelism |
| **Async Rust API** | P1 | Native async on core methods |
| **Streaming search** | P2 | Return results as they arrive from each channel |
| **Distributed mode** | P3 | Multi-node MemMe for cloud deployment (long-term) |

---

## Non-Goals (Explicitly Out of Scope)

- **Managed cloud service** — MemMe is an engine, not a platform. Others can build SaaS on top.
- **Replacing vector databases** — MemMe is a memory engine that happens to use vectors, not a general-purpose vector DB.
- **Python-first** — Rust is the source of truth. All bindings are generated, never hand-written.

---

## Architecture

```
        ┌─────────────────────────────────────────────┐
        │             Language Bindings                │
        │  Python │ Node.js │ Swift │ Kotlin │ WASM   │
        ├─────────────────────────────────────────────┤
        │  REST API (axum)  │  MCP Server (stdio)     │
        ├─────────────────────────────────────────────┤
        │                                             │
        │              memme-core (Rust)              │
        │                                             │
        │  Stream ──► Session ──► Episode ──► Memory  │
        │                                     │       │
        │                           ┌─────────┤       │
        │                           ▼         ▼       │
        │                       Identity    Graph     │
        │                                             │
        │  Search: Vector + BM25 + Graph + Temporal   │
        │          ──► RRF Fusion ──► Rerank          │
        │                                             │
        │  ┌───────────────────────────────────────┐  │
        │  │   DuckDB  (.duckdb single file)       │  │
        │  │   memories │ entities │ relationships  │  │
        │  │   sessions │ episodes │ events         │  │
        │  │   identity │ procedures │ history      │  │
        │  └───────────────────────────────────────┘  │
        │                                             │
        │  memme-embeddings    memme-llm              │
        │  (ONNX/OpenAI/       (Ollama/OpenAI/        │
        │   Ollama)             Anthropic/Gemini)      │
        └─────────────────────────────────────────────┘
```

---

## How to Contribute

See [CONTRIBUTING.md](../CONTRIBUTING.md). We especially welcome:

- **Good first issues**: Labeled in GitHub Issues
- **Language bindings**: Exposing new core APIs to Python/Node/Swift
- **Import formats**: Adding support for new chat history formats
- **Integrations**: LangChain, LlamaIndex, and other framework adapters
- **Benchmarks**: Running MemMe on new datasets or hardware

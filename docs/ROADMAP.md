# MemMe Roadmap

> Memories that are actually yours.
>
> Last updated: 2026-04-01

This roadmap outlines what's been built, what's coming next, and where we'd love community help.

---

## Shipped (v0.1)

MemMe 0.1 is a fully functional AI memory engine:

- **Four-layer data model** — Stream → Session → Episode → Memory
- **Knowledge graph** — LLM entity/relationship extraction, stored in SQLite
- **Four-channel hybrid search** — Vector + BM25 + Entity + Temporal, RRF fusion, optional reranking
- **Forgetting curve** — FSRS-based memory decay with stability reinforcement
- **Meditation** — Memory consolidation: decay, extraction, graph building, identity distillation
- **Smart mode** — LLM fact extraction with dedup (or pure vector mode at sub-10ms)
- **Privacy controls** — Per-memory levels: LocalOnly / Syncable / EncryptedSync
- **Multi-language bindings** — Rust, Python (PyO3), Node.js (NAPI-RS), Swift/Kotlin (UniFFI)
- **REST API** — axum server, 23 endpoints, OpenAPI spec
- **MCP server** — Claude Desktop / Cursor integration
- **Interactive playground** — Local web demo with Remember / Recall / Chat modes

Full feature list in the [README](../README.md).

---

## What's Next

### Data Import

Make it easy to bring your existing AI conversations into MemMe.

- [ ] ChatGPT history import (`conversations.json` → Session/Episode)
- [ ] Claude history import
- [ ] Generic conversation import (JSON/JSONL schema)
- [ ] CLI tool: `memme import --format chatgpt --file conversations.json`

### Ecosystem Integration

Meet developers where they already are.

- [ ] **LangChain** — `MemMeMemory` implementing BaseMemory
- [ ] **LlamaIndex** — MemMe as retriever/storage backend
- [ ] **CrewAI / AutoGen** — Shared memory for multi-agent frameworks
- [ ] **Obsidian** — Two-way sync between vault and MemMe

### Mobile SDK Packaging

Native SDKs beyond raw FFI bindings.

- [ ] iOS SDK via CocoaPods / Swift Package Manager
- [ ] Android SDK via Maven / Gradle
- [ ] On-device LLM integration (llama.cpp / MLX)

### Advanced Memory

Push the frontier of what AI memory can do.

- [ ] Background consolidation (scheduled meditation)
- [ ] Vision memory — extract and store memories from images
- [ ] Memory clustering — auto-discover topic groups
- [ ] Emotion tagging
- [ ] Causal reasoning across memories

### Performance

- [ ] WAL mode for concurrent reads/writes
- [ ] Native async Rust API
- [ ] Streaming search results

---

## Integrations In Progress

These have initial code but are not yet complete:

| Integration | Description | Status |
|-------------|-------------|--------|
| [YiYi](https://github.com/vibeinging/YiYi) | Desktop AI assistant — MemMe powers its memory | Integrated |
| OpenClaw | Agent framework plugin | Tools scaffolded |
| Dora-rs | Rust robotics memory node | Initial crate |
| LeRobot | Hugging Face robotics memory wrapper | Initial package |
| Copper-rs | Real-time robotics CuTask | Initial crate |

---

## Non-Goals

Things we've explicitly decided not to build:

- **Managed cloud service** — MemMe is an engine, not a platform
- **General-purpose vector database** — MemMe uses vectors but isn't competing with Qdrant/Milvus
- **Python-first design** — Rust is the source of truth; all bindings are generated

---

## Contributing

We especially welcome contributions in these areas:

- **Import formats** — Adding support for new chat history formats (Gemini, Copilot, etc.)
- **Framework adapters** — LangChain, LlamaIndex, CrewAI integrations
- **Language bindings** — Exposing new core APIs to Python/Node/Swift
- **Benchmarks** — Running MemMe on new datasets or hardware

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

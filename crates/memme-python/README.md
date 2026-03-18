# MemMe — Python SDK

**Edge-first AI memory engine for Python.** Store, search, and manage long-term memories for AI agents and applications. Powered by an embedded DuckDB database with built-in vector search, BM25 full-text search, knowledge graph, and forgetting curve — all in a single file, no external services required.

[![PyPI](https://img.shields.io/pypi/v/memme.svg)](https://pypi.org/project/memme/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/vibeinging/MemMe/blob/main/LICENSE)

## Installation

```bash
pip install memme
```

## Quick Start

### Basic Usage

```python
from memme import MemoryStore

# Local ONNX embeddings (no API key needed)
store = MemoryStore("memory.duckdb")

# Add memories
store.add("User prefers dark mode", user_id="alice")
store.add("User drinks coffee every morning", user_id="alice")

# Search memories
results = store.search("morning routine", user_id="alice", limit=5)
for r in results:
    print(r["content"], r["score"])
```

### With OpenAI-compatible Embeddings

```python
store = MemoryStore(
    "memory.duckdb",
    embedder="openai",
    api_key="sk-xxx",
    base_url="https://api.openai.com/v1",
    embed_model="text-embedding-3-small",
    dims=1536,
)
```

### LLM-powered Smart Extraction

Automatically extract structured facts from natural language conversations:

```python
store = MemoryStore(
    "memory.duckdb",
    embedder="openai",
    api_key="sk-xxx",
    llm_api_key="sk-xxx",
    llm_model="gpt-4o-mini",
)

# Extracts facts: location=Tokyo, activity=learning Japanese
store.add_smart(
    "I moved to Tokyo last year and started learning Japanese",
    user_id="alice",
)

results = store.search("where does alice live", user_id="alice")
# -> "User moved to Tokyo"
```

### Session and Episode

Organize memories into sessions (conversations) and episodes (meaningful segments):

```python
# Create a session
session = store.create_session(user_id="alice", metadata={"topic": "onboarding"})

# Add messages to the session
store.add_message(session.id, role="user", content="Hi, I'm new here!")
store.add_message(session.id, role="assistant", content="Welcome! Let me help you get started.")
store.add_message(session.id, role="user", content="I prefer dark mode and minimal notifications.")

# Compact session into long-term memories
store.compact_session(session.id)

# Later, search across all memories
results = store.search("user preferences", user_id="alice")
```

## Features

- **Edge-first** — Runs locally with embedded DuckDB, no external services required
- **Single-file storage** — Vectors, knowledge graph, FTS index, and history in one `.duckdb` file
- **Multi-channel retrieval** — Vector + BM25 + Entity Graph + Temporal (RRF fusion)
- **LLM-powered extraction** — Automatic fact extraction with temporal date resolution
- **Forgetting curve** — Ebbinghaus-inspired memory decay with stability reinforcement
- **Privacy controls** — `LocalOnly`, `Syncable`, `EncryptedSync` per-memory levels
- **Cross-platform** — Python, Node.js, Rust, Swift, WASM

## Benchmark (LoCoMo)

| Category | **MemMe** | mem0 | mem0-graph |
|---|---|---|---|
| **Single-hop** | **80.50** | 67.13 | 65.71 |
| **Multi-hop** | **55.76** | 51.15 | 47.19 |
| **Temporal** | **59.38** | 55.51 | 58.13 |
| **Open-domain** | **74.55** | 72.93 | 75.71 |

## Links

- [GitHub Repository](https://github.com/vibeinging/MemMe)
- [Documentation](https://github.com/vibeinging/MemMe/blob/main/docs/TECHNICAL.md)
- [Rust Crate (memme-core)](https://crates.io/crates/memme-core)
- [npm Package (memme)](https://www.npmjs.com/package/memme)

## License

Apache-2.0. See [LICENSE](https://github.com/vibeinging/MemMe/blob/main/LICENSE) for details.

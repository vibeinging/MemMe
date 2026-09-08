<div align="center">

# MemMe

**Local long-term memory for AI pets.**

Remember the owner. Keep each pet relationship separate. Stay on the device.

[![CI](https://github.com/vibeinging/MemMe/actions/workflows/ci.yml/badge.svg)](https://github.com/vibeinging/MemMe/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/%40wjmwjmwb%2Fmemme.svg)](https://www.npmjs.com/package/@wjmwjmwb/memme)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Rust core** · **SQLite single file** · **VexDB-Lite** · **macOS / Linux** · **offline-first**

English | [中文](README_CN.md)

</div>

---

MemMe is a private, durable memory runtime for AI pets and companions. It helps
an AI pet remember its owner, preserve shared experiences, keep each
relationship separate, and continue working across restarts, model changes, and
network loss.

The engine is built for AI-pet behavior, safety, isolation, low latency, and
on-device use. Its Rust core can also be embedded in other companion products.

## Try it locally

This demo uses local inference and needs no API key. It requires macOS or
Linux (x64 / arm64), Rust, Python 3, and curl. The first run needs internet
access to download dependencies and the embedding model:

```bash
git clone --branch main https://github.com/vibeinging/MemMe.git
cd MemMe
bash demos/rest-demo.sh
```

The script downloads the pinned VexDB-Lite extension and ONNX Runtime, builds
the REST server with a local ONNX embedding model, writes memories for one
owner and two pets, fully restarts the process, and recalls again — so you can
verify restart persistence and per-pet isolation yourself. The first run also
downloads the embedding model before the server starts. Later runs reuse
the model cache in `~/.cache/memme-demo/.fastembed_cache`; inference works
offline once the dependencies and model are cached. Startup still loads the
model into memory.

The script checks seven outcomes and exits with an error if a check fails.
Set `MEMME_DEMO_PORT` if port 18070 is busy, `MEMME_DEMO_DIR` to keep the data
and model cache elsewhere, or `MEMME_DEMO_START_TIMEOUT` to change the model
startup timeout (default: 600 seconds). Startup details are in
`$MEMME_DEMO_DIR/server.log` (default: `~/.cache/memme-demo/server.log`).

**Source version:** this demo is maintained on `main`. The `v0.1.2` source
archive predates the demo and ONNX Runtime download scripts; use the clone
command above instead of running these instructions from that archive.
Record `git rev-parse HEAD` when reporting a result. Node.js and manual REST
paths are in the quick starts below.

## What an AI pet needs to remember

An AI pet should remember more than a bag of similar chat snippets:

- **Owner memory** — stable facts, preferences, boundaries, and safety rules
  shared across the owner’s pets.
- **Relationship memory** — names, inside jokes, shared experiences, and habits
  that belong to one owner-pet relationship and must not leak to another pet.
- **Fresh events** — what was said or done a moment ago, available before any
  background summarization finishes.
- **Changing truth** — expired or replaced facts must stop affecting replies,
  while their history remains auditable.
- **Durable local state** — memory survives restarts, model changes, network
  loss, and application upgrades.

## Core capabilities

- Search owner-global memory and the selected pet relationship together.
- Recall fresh events immediately inside the correct owner and pet scope.
- Keep owners and pet relationships isolated by default.
- Filter expired and superseded facts before they can affect a reply.
- Store authoritative data in one SQLite file with VexDB-Lite vector indexing.
- Combine vector, full-text, entity, temporal, and exact-identifier retrieval.
- Preserve history, immutable safety memories, backups, and portable exports.
- Run optional LLM extraction and consolidation outside the reply path.

## Memory model

```text
message / action
      │
      ▼
append-only events ───────────────► immediately searchable
      │
      ▼
session ── compact ──► episode ── meditate ──► durable facts
                                                   │
                         ┌─────────────────────────┴──────────────┐
                         │                                        │
                 owner-global memory                    pet relationship memory
                 agent_id = NULL                        agent_id = selected pet
                         │                                        │
                         └────────────── search ──────────────────┘
                                            │
                              VexDB vector + SQLite FTS
```

The normal reply path does not need an extra memory LLM call. It writes the raw
event, reads the small active scope, and retrieves a few useful memories. Heavy
fact extraction, reconciliation, graph work, and reflection can run after the
reply or while the device is idle.

SQLite tables are the source of truth. Vector and full-text indexes are derived
data and can be rebuilt.

## PetMemBench

`PetMemBench` is MemMe's product benchmark. Its scenarios cover owner safety,
per-pet relationships, fresh events, privacy isolation, Chinese retrieval,
expiration, correction, and deletion.

Release benchmark for `0.1.2`, using 2,000 memories and three independent runs:

| Metric | Result |
|---|---:|
| Required scenarios | 11 / 11 |
| Extended scenarios | 3 / 3 |
| Recall@10 | 100% |
| Search p50 | 1.999 ms |
| Search p95 | 3.062 ms |
| Search p99 | 17.128 ms |
| Write throughput | 445.7 memories/s |
| SQLite file size | 10.4 MB |

These numbers test the storage and retrieval contract. They do **not** yet prove
final reply quality or production performance: the benchmark uses deterministic
local test embeddings, 2,000 memories, and an x86_64 process under Rosetta on
Apple Silicon.

- [0.1.2 release report](docs/reports/2026-09-01_release-0.1.2.md)
- [Scenarios](benchmarks/petmem/scenarios.json)
- [AI-pet architecture research](docs/research/2026-09-01_ai-pet-memory-architecture.md)

## Quick start with Node.js

The current npm release supports macOS and Linux on arm64 and x64:

```bash
npm install @wjmwjmwb/memme
```

MemMe does not bundle the VexDB-Lite SQLite extension. Download the matching
trusted VexDB-Lite v0.0.17 library and provide its absolute path:

```bash
export MEMME_VEXDB_LITE_EXTENSION=/absolute/path/to/vexdb_lite.dylib
```

```javascript
const { MemoryStore } = require("@wjmwjmwb/memme");

const store = MemoryStore.newOpenai(
  process.env.OPENAI_API_KEY,
  "momo-memory.db",
);

// Owner-global memory: visible to the owner's selected pets.
await store.add("The owner has a severe peanut allergy.", "owner-001");

// Relationship memory: only Momo should retrieve it.
await store.add(
  "Momo and the owner first met under the ginkgo tree.",
  "owner-001",
  "momo",
);

const context = await store.search(
  "What should I remember when preparing Momo's birthday snack?",
  "owner-001",
  "momo",
  null,
  5,
);

console.log(context);
```

Use `MemoryStore.newMock()` for tests that must not call an embedding service.

## Quick start with Rust

Rust users can build the SQLite engine directly from this repository:

```bash
git clone https://github.com/vibeinging/MemMe.git
cd MemMe
export MEMME_VEXDB_LITE_EXTENSION="$(bash scripts/download-vexdb-lite-extension.sh)"
export ORT_DYLIB_PATH="$(bash scripts/download-onnx-runtime.sh)"
cargo build -p memme-core
cargo test -p memme-core
```

```rust
use std::sync::Arc;
use memme_core::{AddOptions, MemoryConfig, MemoryStore, SearchOptions};
use memme_embeddings::onnx::OnnxEmbedder;

fn main() -> memme_core::Result<()> {
    let config = MemoryConfig::new("momo-memory.db", 384);
    let embedder = Arc::new(OnnxEmbedder::new()?);
    let store = MemoryStore::new(config, embedder)?;

    store.add(
        "主人对花生严重过敏。",
        AddOptions::new("owner-001").immutable(true),
    )?;

    store.add(
        "默默和主人第一次见面是在银杏树下。",
        AddOptions::new("owner-001").agent_id("momo"),
    )?;

    let memories = store.search(
        "给默默准备生日零食，要注意什么？",
        SearchOptions::new("owner-001").agent_id("momo").limit(5),
    )?;

    for memory in memories {
        println!("{}", memory.content);
    }

    Ok(())
}
```

## Quick start with the REST server

The server uses the compact local `bge-small-zh-v1.5` ONNX model by default.
Choose `--onnx-embedding-model multilingual-e5-small` for multilingual data.
An LLM is optional: event
ingestion and recall work without one; `compact` and `meditate` need one.

```bash
export MEMME_VEXDB_LITE_EXTENSION="$(bash scripts/download-vexdb-lite-extension.sh)"
export ORT_DYLIB_PATH="$(bash scripts/download-onnx-runtime.sh)"
export MEMME_API_KEY=change-me
cargo run --release -p memme-server -- --db-path momo-memory.db
```

Write each message with a stable event ID. Retrying the same request is safe:

```bash
curl -s http://127.0.0.1:8080/v1/events \
  -H "Authorization: Bearer $MEMME_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id":"voice-session-001",
    "user_id":"owner-001",
    "agent_id":"momo",
    "app_id":"xiaozhi",
    "messages":[{
      "event_id":"voice-session-001-user-001",
      "role":"user",
      "content":"I bought Momo a blue whale toy."
    }]
  }'
```

Use the same `user_id` and `agent_id` when recalling. This keeps relationship
memory from different pets separate. The complete contract is in
[`docs/openapi.yaml`](docs/openapi.yaml).

The REST API also exposes complete user-data operations:

- `POST /v1/data/export` produces an unpaginated version 3 user export with
  lifecycle, correction, graph, audit, procedure, meditation, and recall data.
- `POST /v1/data/import` accepts only v3 exports for the same collection,
  validates owner and cross-layer references, then imports every layer in one
  transaction. The target must be free of imported IDs. Admission happens before
  the JSON body is read; explicit body, record, and vector-memory limits apply.
  Use a SQLite backup for larger databases or replacement restores.
- `DELETE /v1/users/{user_id}` removes that user's memories, raw events,
  sessions, episodes, graph, identity, and history after exact confirmation.
- `POST /v1/backups` uses SQLite's online backup API to create a consistent
  snapshot under `MEMME_BACKUP_DIR` while the service remains available.
- `POST /v1/backups/restore` opens the candidate with the current embedding,
  VexDB-Lite, and collection settings before replacement. It keeps the previous
  database until the replacement reopens successfully. Success and failure both
  restart the REST server on the same address; failure continues with the old DB.

If remote embedding is temporarily unavailable, `/v1/events` still stores the
raw text and returns `embedding_pending`. Replaying the same `event_id` after the
provider recovers backfills its vector instead of skipping it as a duplicate.

Only `/health` is public when `MEMME_API_KEY` is configured. `/diagnose` runs
real provider checks and therefore requires the same Bearer token as `/v1/*`.

A `session_id` is permanently bound to its first `user_id`, `agent_id`,
`app_id`, and `run_id`. Reusing it with another owner or pet returns `400`.

## VexDB-Lite storage

MemMe uses a normal SQLite database as its durable source of truth and
VexDB-Lite’s persistent `GRAPH_INDEX` for vector retrieval. `sqlite-vec` and
DuckDB are not used by the current source tree.

The repository helper downloads the pinned VexDB-Lite v0.0.17 release for the
current macOS/Linux architecture and verifies the archive and library with
SHA-256:

```bash
export MEMME_VEXDB_LITE_EXTENSION="$(bash scripts/download-vexdb-lite-extension.sh)"
```

Local ONNX embeddings additionally load a pinned ONNX Runtime 1.19.2 dynamic
library via `ORT_DYLIB_PATH`:

```bash
export ORT_DYLIB_PATH="$(bash scripts/download-onnx-runtime.sh)"
```

Current dynamic-extension support is macOS and Linux on x64 and arm64. Windows
does not have a matching VexDB-Lite v0.0.17 SQLite extension. Mobile and WASM
crates exist, but they are not yet wired to the same VexDB-Lite SQLite runtime.

Only load a trusted extension: SQLite extensions run as native code inside the
application process.

## Data and privacy

```text
momo-memory.db             all authoritative memory data
momo-memory.db.replica     optional local replica
```

- One owner’s memory never enters another owner’s query.
- One pet’s relationship memory never enters another pet’s query.
- Expired and superseded facts are filtered before result fusion.
- Immutable safety memory cannot be silently updated or deleted.
- `backup_to_path()` creates a portable SQLite snapshot.
- `full_export()` / `full_import()` and the REST data endpoints move every
  user-data layer without a cloud service.
- The host application controls sync, encryption, retention, and raw-audio
  policy. For voice toys, text events are enough for memory; raw audio does not
  need to become long-term memory.

## Packages and APIs

| Component | Current use |
|---|---|
| `memme-core` | Rust memory engine, lifecycle, search, graph, backup |
| `memme-embeddings` | ONNX, OpenAI-compatible, and Ollama embedders |
| `memme-llm` | Optional OpenAI, Anthropic, Gemini, and Ollama extraction |
| [`@wjmwjmwb/memme`](https://www.npmjs.com/package/@wjmwjmwb/memme) | Node.js / Electron binding for macOS and Linux |
| `memme-python` | PyO3 binding; build current SQLite version from source |
| `memme-ffi` | Swift/C UniFFI binding; VexDB-Lite mobile wiring is pending |
| `memme-server` | Self-hosted REST API |
| `memme-mcp` | MCP stdio server |

Use the scoped npm package for Node.js. For Rust and Python projects, pin a
source revision and build the SQLite engine from this repository.

## Run PetMemBench

```bash
export MEMME_VEXDB_LITE_EXTENSION="$(bash scripts/download-vexdb-lite-extension.sh)"
cargo run --release -p memme-core --example pet_memory_benchmark -- \
  --dataset benchmarks/petmem/scenarios.json \
  --output benchmarks/petmem/results/latest.json
```

## Contributing

AI-pet scenarios are the most useful contribution right now. A good test should
state the owner, pet, time, expected memory, forbidden memory, and privacy scope.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).

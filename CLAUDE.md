# MemMe

Edge-first AI memory engine — Rust core + DuckDB single-file storage.

## Quick Reference

- Language: Rust (edition 2021, MSRV 1.77)
- Build: `cargo build -p memme-core`
- Test: `cargo test -p memme-core`
- Lint: `cargo clippy -p memme-core -p memme-embeddings -p memme-llm -- -D warnings`
- Format: `cargo fmt --all`
- Format check: `cargo fmt --all -- --check`
- Build all bindings: `cargo build -p memme-ffi -p memme-node -p memme-wasm -p memme-server -p memme-mcp`

## Workspace Structure

| Crate | Purpose |
|---|---|
| `memme-core` | Core memory engine: CRUD, search, graph, dedup, forgetting curve, analytics |
| `memme-embeddings` | Embedding trait + ONNX/OpenAI/Ollama backends |
| `memme-llm` | LLM trait + Ollama/OpenAI/Anthropic/Gemini backends for fact extraction |
| `memme-python` | Python bindings via PyO3 + maturin |
| `memme-ffi` | Swift/C bindings via UniFFI |
| `memme-node` | Node.js bindings via NAPI-RS |
| `memme-wasm` | WASM bindings via wasm-bindgen |
| `memme-server` | REST API server (axum) |
| `memme-mcp` | MCP stdio server for Claude Desktop / Cursor |

## Architecture

### Storage

DuckDB single file. Tables: `memories`, `entities_{collection}`, `relationships_{collection}`, `memory_entities`, `history`, `procedures`, `sessions`, `sources`, `events`, `episodes`, `identity`, `associations`, `meditations`, `recalls`, `memme_config`.

### Data Flow (Session/Episode model)

1. **Stream** — raw events ingested via `ingest_event` / `append_events`
2. **Session / Episode** — events grouped into sessions, compacted into episodes
3. **Memory (Trace)** — atomic facts extracted from episodes, stored with embeddings
4. **Identity** — high-level personality traits distilled from memories
5. **Graph** — entity-relation knowledge graph extracted alongside memories

### Search Pipeline

Vector + BM25 FTS + Entity Spreading Activation + Temporal -> RRF Fusion -> Optional Rerank -> Forgetting Curve scoring.

RRF weights (configurable): vector 0.5, FTS 0.3, entity 0.2, temporal 0.15. Each channel retrieves `limit * candidate_multiplier` candidates before fusion.

### LLM Integration

Pluggable via `LlmProvider` trait (OpenAI, Anthropic, Gemini, Ollama). LLM is optional — core works without it using pure vector dedup path.

### Memory Lifecycle

add -> deduplicate (cosine distance < threshold) -> extract facts (if LLM) -> store with embedding -> search with forgetting curve

## Key Patterns

- **ConnectionPool**: `Single` mode (`Mutex<Connection>`) for `:memory:`, `Multi` mode (separate read/write connections) for file-backed DBs. Mutex poison recovery via `recover_lock`.
- **Deferred writes**: `access_count` increment and `stability` reinforcement queued in `Mutex<Vec<DeferredWrite>>`, flushed on next write operation to avoid acquiring write lock during reads.
- **FSRS forgetting curve**: `R(t, S) = (1 + t / (c * S))^(-p)` where c=5.0, p=0.5. Not classical Ebbinghaus exponential. Stability reinforcement: `new_S = S * (1 + growth_factor * (1 - R))`.
- **Entity matching**: Aho-Corasick automaton with word boundary checks (`entity_index.rs`).
- **Graph extraction**: single combined LLM call for entities + relationships.
- **Collection name**: validated alphanumeric + underscore only, used in table names via `format!()`.

## Error Types

`MemoryError` enum in `error.rs`:
- `DuckDb` — DuckDB driver errors
- `Embedding` — embedding provider errors
- `NotFound` — memory ID not found
- `Serialization` — JSON serde errors
- `Config` — invalid configuration
- `Llm` — LLM provider errors
- `ImmutableMemory` — attempt to modify/delete an immutable memory

## Conventions

- All SQL queries use parameterized `$N` placeholders (never string interpolation for values)
- Public API is in `lib.rs` (`MemoryStore`), storage internals are `pub(crate)`
- Tests use `:memory:` DuckDB with `MockEmbedder` / mock LLM
- Feature flags: `bundled` (default, compile DuckDB from source), `memme-db` (precompiled DuckDB + HNSW), `api-rerank`, `onnx-rerank`, `webhooks`
- Commit messages: `<type>: <summary>` (feat/fix/refactor/docs/test/ci/chore)
- CI runs on ubuntu + macOS, checks: fmt, clippy (-D warnings on core/embeddings/llm), unit tests, integration tests, edge cases, mobile scenarios, stress tests

## Common Tasks

- **Adding a new storage method**: implement in `storage/*.rs`, expose via `MemoryStore` methods in `memory/*.rs`, re-export from `lib.rs` if public
- **Adding a new LLM provider**: implement `LlmProvider` trait in `memme-llm/src/`
- **Adding a new embedder**: implement `Embedder` trait in `memme-embeddings/src/`
- **Running benchmarks**: see `benchmarks/` directory
- **Building Python package**: `cd crates/memme-python && maturin develop --release`

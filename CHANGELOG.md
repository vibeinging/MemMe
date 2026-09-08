# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-09-01

npm `@wjmwjmwb/memme@0.1.2` (macOS / Linux) was published from this source
line. PyPI and crates.io remain on `0.1.1` until registry credentials are
refreshed; both publish workflows run manually.

### Changed

- **SQLite migration**: Replaced DuckDB with SQLite as the sole storage backend. Single-file `.db` format, smaller binary, broader platform compatibility. All APIs remain unchanged.

### Added

- **Session/Episode architecture**: Four-layer memory model (Stream → Episode → Semantic → Identity) with session history, compact purification, meditation extraction, and session context retrieval
- **Multi LLM Provider support**: Pluggable provider trait with protocol adapter pattern; trait-HTTP separation for cleaner abstractions
- **Four-channel retrieval engine**: Vector, BM25, graph, and temporal channels merged via RRF
- **Temporal extraction**: Time dimension as first-class citizen with temporal-aware search
- **Recall API**: New recall store and recall types for structured memory recall
- **Session/Episode types**: `Session`, `ListSessionsOptions`, stream types, and identity types
- **Connection pool**: Read-write separation via `ConnectionPool`
- **Config persistence**: Runtime configuration stored via `config_store`
- **Interactive playground**: Local web demo (`demos/playground/`) with Remember/Recall/Chat modes, sample data loading, and localStorage config persistence
- **Apple native demo**: macOS/iOS SwiftUI demo application
- **CI/CD pipelines**: Cross-platform release workflows for PyPI, npm, crates.io, iOS, and Android
- **Benchmark tooling**: LoCoMo benchmark with rerank/temporal parameters; batch processing and judge improvements (45.7% → 63.9%)
- **Release profile**: LTO + strip optimizations for smaller binaries

### Changed

- **Refactored `smart.rs`**: Split 1374-line monolith into a module directory
- **Unified SQL queries**: Introduced `EVENT_COLS` constant to standardize column selection across storage layer
- **Trait-HTTP separation**: Decoupled trait definitions from HTTP transport for reduced binary size
- **Enhanced extraction prompt**: Improved LLM standard extraction with better time parsing
- **Removed Memory Index**: Reverted unified memory index experiment to focus on Session/Episode architecture
- **API naming**: Renamed `top_k` → `limit` across REST server, MCP server, cu-memme, and json_ops for consistency with core API

### Fixed

- **Transaction safety**: `add_smart()` now wrapped in BEGIN/COMMIT/ROLLBACK to prevent partial writes on LLM/embedding failure
- **Ollama embedding retry**: Added exponential backoff (3 retries) matching OpenAI provider behavior
- **Deferred writes flush**: Added time-based flush (30s interval) alongside the 500-item cap to prevent stale writes in read-heavy workloads
- **iOS compilation**: Method name alignment, `HybridSearch` adaptation, and `staticlib` target
- **Cross-platform packaging**: Isolated ONNX dependencies and added missing build deps
- **Rerank fallback**: Graceful degradation when reranker is unavailable
- **Deadlock fix**: Resolved concurrency issue in core engine
- **Security**: Removed hardcoded API keys; switched to environment variables

## [0.1.0] - 2026-03-18

### Core Engine

- Single-file DuckDB storage for vectors, knowledge graph, FTS index, and history
- Pluggable LLM: works without LLM (<10ms vector dedup) or with any LLM (fact extraction)
- Four-level scoping: `user_id` / `agent_id` / `app_id` / `run_id` isolation
- Immutable memories with optional TTL auto-expiration
- Battery-aware processing that defers heavy operations on low battery

### Search & Retrieval

- Hybrid search combining vector similarity and BM25 full-text search
- Reciprocal Rank Fusion (RRF) for result merging
- Advanced filters: Eq, Ne, Gt, Gte, Lt, Lte, In, Contains, IContains
- AND/OR filter combinators

### Knowledge Graph

- Entity and relationship extraction (smart mode)
- DuckPGQ-powered graph traversal
- Graph-augmented memory retrieval

### Memory Management

- Ebbinghaus forgetting curve with stability reinforcement on access
- Deduplication: vector similarity (no LLM) or LLM-driven smart dedup
- Per-memory privacy levels: `LocalOnly`, `Syncable`, `EncryptedSync`
- Full JSON export/import including memories, entities, and relationships

### Embeddings

- Pluggable embedding provider trait
- Mock embeddings for testing
- OpenAI embeddings support (behind `openai` feature)
- ONNX cross-encoder reranker (behind `onnx-rerank` feature)

### LLM Integration

- Pluggable LLM provider trait for smart mode
- OpenAI provider (behind `openai` feature)
- Fact extraction and memory consolidation

### Bindings

- `memme-ffi` -- C-compatible FFI layer
- `memme-node` -- Node.js bindings (NAPI-RS)
- `memme-python` -- Python bindings (PyO3)
- `memme-wasm` -- WebAssembly bindings
- `memme-server` -- HTTP server
- `memme-mcp` -- Model Context Protocol server

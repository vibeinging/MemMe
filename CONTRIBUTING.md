# Contributing to MemMe

Thank you for your interest in contributing to MemMe! This document provides guidelines and instructions for contributing.

## Prerequisites

- **Rust 1.77+** (install via [rustup](https://rustup.rs/))
- **Git** with submodule support
- **DuckDB** (optional -- the default `bundled` feature compiles DuckDB from source)

### Optional

- **CMake 3.20+** -- required only if building with the `memme-db` feature (DuckDB + MemMe-DB extension)
- **Node.js 18+** -- for `memme-node` binding development
- **Python 3.9+** -- for `memme-python` binding development
- **wasm-pack** -- for `memme-wasm` development

## Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/vibeinging/MemMe.git
cd MemMe

# Build the core crate (uses bundled DuckDB by default)
cargo build -p memme-core

# Build all binding crates
cargo build -p memme-ffi -p memme-node -p memme-wasm -p memme-server -p memme-mcp
```

## Running Tests

```bash
# Run core tests
cargo test -p memme-core

# Run embedding tests
cargo test -p memme-embeddings

# Run LLM integration tests
cargo test -p memme-llm

# Run integration tests
cargo test -p memme-core --test integration

# Run all workspace tests
cargo test --workspace
```

## Code Style

This project enforces code style via CI:

- **rustfmt** -- all code must pass `cargo fmt --all -- --check`
- **clippy** -- all code must pass `cargo clippy -- -D warnings`

Run both before submitting a PR:

```bash
cargo fmt --all
cargo clippy -p memme-core -p memme-embeddings -p memme-llm -- -D warnings
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** for any new functionality.
3. **Ensure CI passes** -- run `cargo fmt`, `cargo clippy`, and `cargo test` locally.
4. **Keep PRs focused** -- one feature or fix per pull request.
5. **Update documentation** if your change affects public APIs or user-facing behavior.
6. **Fill out the PR template** completely.

## Commit Message Format

Use clear, descriptive commit messages:

```
<type>: <short summary>

<optional body explaining why>
```

Types:
- `feat` -- new feature
- `fix` -- bug fix
- `refactor` -- code restructuring without behavior change
- `docs` -- documentation only
- `test` -- adding or updating tests
- `ci` -- CI/CD changes
- `chore` -- maintenance tasks

Examples:
```
feat: add TTL support for auto-expiring memories
fix: prevent duplicate entity insertion in knowledge graph
docs: update quick start for Python bindings
```

## Reporting Issues

- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) for bugs.
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) for ideas.
- Check existing issues before opening a new one.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

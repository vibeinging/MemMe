# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | Yes                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report vulnerabilities through one of the following channels:

1. **GitHub Security Advisory** -- use the "Report a vulnerability" button on the [Security Advisories page](https://github.com/vibeinging/MemMe/security/advisories/new).
2. **Email** -- send details to the maintainers at the email address listed in the repository profile.

### What to include

- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact
- Suggested fix (if any)

### Response timeline

- **Acknowledgment** -- within 48 hours
- **Initial assessment** -- within 5 business days
- **Fix and disclosure** -- coordinated with reporter; aim for 30 days

## Security Model

MemMe is designed as a **local-first, edge-deployed** memory engine. Its security model reflects this:

### Data Storage

- All data is stored in a local `.duckdb` file on the user's device.
- No data is sent to external servers unless the application explicitly configures a remote embedding or LLM provider.
- The `LocalOnly` privacy level ensures marked memories are never eligible for sync.

### SQL Injection Prevention

- All database queries use **parameterized SQL statements**.
- User-supplied values are never interpolated into query strings.

### Privacy Controls

- **Per-memory privacy levels**: `LocalOnly`, `Syncable`, `EncryptedSync`.
- **Four-level scoping** (`user_id` / `agent_id` / `app_id` / `run_id`) provides data isolation between tenants.
- Memory queries are scoped by default -- one user cannot access another user's memories.

### LLM and Embedding Providers

- When using external providers (OpenAI, etc.), memory content is sent to those services for processing.
- Users should review provider privacy policies before enabling smart mode or external embeddings.
- The `bundled` feature and mock providers allow fully offline operation with no external data transmission.

### Dependencies

- Dependencies are pinned via `Cargo.lock` and audited periodically.
- The project uses `cargo clippy` with `-D warnings` in CI to catch common issues.

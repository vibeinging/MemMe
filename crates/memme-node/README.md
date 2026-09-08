# MemMe Node.js SDK

Local long-term memory for AI pets. MemMe keeps owner-global memory, per-pet
relationship memory, and fresh events in one SQLite file.

```bash
npm install @wjmwjmwb/memme
```

## AI-pet scope

```js
const { MemoryStore } = require("@wjmwjmwb/memme");

const store = MemoryStore.newOpenai(
  process.env.OPENAI_API_KEY,
  "momo-memory.db",
);

// Shared owner memory.
await store.add("The owner has a severe peanut allergy.", "owner-001");

// Only Momo's relationship can retrieve this memory.
await store.add(
  "Momo and the owner first met under the ginkgo tree.",
  "owner-001",
  "momo",
);

const memories = await store.search(
  "What should I remember for Momo's birthday snack?",
  "owner-001",
  "momo",
  null,
  5,
);
```

## Supported platforms

The published package supports macOS and Linux on x64 and arm64. Windows is not
published because VexDB-Lite v0.0.17 does not provide a Windows SQLite extension.

The extension is not bundled in the npm package. Download the matching trusted
VexDB-Lite SQLite extension and provide its absolute path:

```bash
export MEMME_VEXDB_LITE_EXTENSION=/opt/vexdb-lite/vexdb_lite.so
```

```js
const store = MemoryStore.newMock("memory.db", 384);
```

Or pass the extension path directly as the last constructor argument:

```js
const store = MemoryStore.newMock(
  "memory.db",
  384,
  "/opt/vexdb-lite/vexdb_lite.so",
);
```

Mobile and WASM publication remains paused until VexDB-Lite uses the same SQLite
instance as MemMe.

See the [main project README](https://github.com/vibeinging/MemMe) for the
architecture, PetMemBench results, privacy boundary, and roadmap.

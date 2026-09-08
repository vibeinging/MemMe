<div align="center">

# MemMe

**给 AI 宠物用的本地长期记忆体。**

记住主人，分清每只宠物的关系，数据留在设备里。

[![CI](https://github.com/vibeinging/MemMe/actions/workflows/ci.yml/badge.svg)](https://github.com/vibeinging/MemMe/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/%40wjmwjmwb%2Fmemme.svg)](https://www.npmjs.com/package/@wjmwjmwb/memme)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Rust 内核** · **SQLite 单文件** · **VexDB-Lite** · **macOS / Linux** · **本地优先**

[English](README.md) | 中文

</div>

---

MemMe 是 AI 宠物和 AI 陪伴设备的本地长期记忆体。它让 AI 宠物记住主人、保留
共同经历、分清每一段关系，并在重启、断网或更换模型后继续保持记忆。

整个引擎围绕 AI 宠物的关系连续性、安全、隔离、低延迟和端侧运行来设计。Rust
内核也可以嵌入其他陪伴类产品。

## 在本地试用

最快的路径完全运行在你自己的机器上——不需要 API Key，不连云。需要
macOS 或 Linux（x64 / arm64）和 Rust 工具链：

```bash
git clone https://github.com/vibeinging/MemMe.git
cd MemMe
bash demos/rest-demo.sh
```

脚本会下载固定版本的 VexDB-Lite 扩展、构建使用本地 ONNX 向量模型的 REST
服务，写入一位主人和两只宠物的记忆，完整重启进程后再次检索——重启后记忆
还在、不同宠物不串记忆，这两件事你可以亲自验证。首次运行会下载约 100 MB
的本地向量模型（仅一次）；之后启动是即时的，并且完全离线。Node.js 和手动
REST 路径见下面的快速开始。

## AI 宠物真正需要记住什么

AI 宠物的记忆，不应该只是一堆相似的聊天片段：

- **主人记忆**：稳定资料、喜好、边界和安全信息，可以被主人的多只宠物共享。
- **关系记忆**：称呼、共同经历、只有彼此懂的梗和互动习惯，只属于一只宠物和
  主人的关系，不能串给另一只宠物。
- **即时事件**：刚刚说过的话、刚刚发生的动作，不需要等后台总结就能找到。
- **变化中的事实**：已经过期或被新事实替代的内容不能继续影响回答，但历史记录
  仍然保留。
- **本地持久状态**：重启、断网、换模型、升级应用后，记忆仍然存在。

## 核心能力

- 一次搜索合并主人全局记忆和当前宠物的关系记忆。
- 刚写入的事件可以立即在正确的主人和宠物范围内找回。
- 默认隔离不同主人、不同宠物的关系记忆。
- 过期和已经被替代的事实不会进入回答上下文。
- 使用 SQLite 单文件保存权威数据，VexDB-Lite 提供向量索引。
- 组合向量、全文、实体、时间和精确编号检索。
- 保留历史、不可变安全记忆、备份和可搬走的导出数据。
- 可选的 LLM 提取和整理放在回复热路径之外运行。

## 记忆模型

```text
消息 / 动作
    │
    ▼
只追加的事件流 ─────────────────► 立即可以搜索
    │
    ▼
会话 ── compact ──► 共同经历 ── meditate ──► 长期事实
                                               │
                     ┌─────────────────────────┴──────────────┐
                     │                                        │
               主人全局记忆                           当前宠物关系记忆
               agent_id = NULL                       agent_id = 当前宠物
                     │                                        │
                     └────────────── 搜索 ────────────────────┘
                                          │
                                VexDB 向量 + SQLite FTS
```

正常回复时，不需要额外调用一次“记忆 LLM”。热路径只负责保存原始事件、读取小块
状态并取回少量相关记忆。事实提取、冲突整理、图谱和反思可以在回复后或设备空闲时
运行。

SQLite 表是权威数据。向量索引和全文索引都是派生数据，可以重新生成。

## PetMemBench

`PetMemBench` 是 MemMe 的产品 Benchmark，场景覆盖主人安全、每只宠物的关系、
即时事件、隐私隔离、中文检索、过期、纠正和删除。

`0.1.2` 发布 Benchmark，2,000 条记忆，独立运行三次后的中位数：

| 指标 | 结果 |
|---|---:|
| 必须场景 | 11 / 11 |
| 扩展场景 | 3 / 3 |
| Recall@10 | 100% |
| 查询 p50 | 1.999 ms |
| 查询 p95 | 3.062 ms |
| 查询 p99 | 17.128 ms |
| 写入速度 | 445.7 条/秒 |
| SQLite 文件大小 | 10.4 MB |

这些数字只证明当前存储和检索约定已经跑通，**不能代表最终回答质量或生产性能**。
当前 Benchmark 使用确定性的本地测试向量，数据量为 2,000 条，并且在 Apple
Silicon 上用 x86_64/Rosetta 进程运行。

- [0.1.2 发布报告](docs/reports/2026-09-01_release-0.1.2.md)
- [测试场景](benchmarks/petmem/scenarios.json)
- [AI 宠物记忆架构调研](docs/research/2026-09-01_ai-pet-memory-architecture.md)

## Node.js 快速开始

当前 npm 包支持 macOS、Linux 的 arm64 和 x64：

```bash
npm install @wjmwjmwb/memme
```

MemMe 不会把 VexDB-Lite SQLite 扩展打进 npm 包。请下载匹配架构、来源可信的
VexDB-Lite v0.0.17 动态库，并提供绝对路径：

```bash
export MEMME_VEXDB_LITE_EXTENSION=/absolute/path/to/vexdb_lite.dylib
```

```javascript
const { MemoryStore } = require("@wjmwjmwb/memme");

const store = MemoryStore.newOpenai(
  process.env.OPENAI_API_KEY,
  "momo-memory.db",
);

// 主人全局记忆：主人选中的宠物都能取到。
await store.add("主人对花生严重过敏。", "owner-001");

// 关系记忆：只有默默能取到。
await store.add(
  "默默和主人第一次见面是在银杏树下。",
  "owner-001",
  "momo",
);

const context = await store.search(
  "给默默准备生日零食，要注意什么？",
  "owner-001",
  "momo",
  null,
  5,
);

console.log(context);
```

不想调用向量服务的测试，可以使用 `MemoryStore.newMock()`。

## Rust 快速开始

Rust 项目可以直接从本仓库构建 SQLite 引擎：

```bash
git clone https://github.com/vibeinging/MemMe.git
cd MemMe
export MEMME_VEXDB_LITE_EXTENSION="$(bash scripts/download-vexdb-lite-extension.sh)"
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

## REST 服务快速开始

服务默认使用体积较小的本地 `bge-small-zh-v1.5` ONNX 模型，不需要云端 Key。
多语言数据可以改用 `--onnx-embedding-model multilingual-e5-small`。LLM 是可选项：
事件写入和召回不需要 LLM；`compact` 和 `meditate` 需要配置 LLM。

```bash
export MEMME_VEXDB_LITE_EXTENSION="$(bash scripts/download-vexdb-lite-extension.sh)"
export MEMME_API_KEY=change-me
cargo run --release -p memme-server -- --db-path momo-memory.db
```

每条消息都使用稳定的 `event_id`。同一请求重试不会重复写入：

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
      "content":"我给默默买了一个蓝色鲸鱼玩具。"
    }]
  }'
```

召回时继续使用同一个 `user_id` 和 `agent_id`，不同宠物的关系记忆不会混在一起。
完整接口见 [`docs/openapi.yaml`](docs/openapi.yaml)。

REST API 也提供完整的用户数据操作：

- `POST /v1/data/export` 生成不受分页限制的 v3 用户导出，包含生命周期、
  纠错、图谱、审计、流程、冥想和召回数据。
- `POST /v1/data/import` 只接受同 collection 的 v3 导出，先校验主人、宠物和
  跨层引用，再用一个事务导入全部数据层；目标库不能已有待导入 ID。服务会在读取
  JSON 前取得唯一导入名额，并限制请求体、记录数和向量内存；更大的数据库或整库
  替换应使用 SQLite 备份。
- `DELETE /v1/users/{user_id}` 在精确确认后，删除该用户的记忆、原始事件、
  会话、情景、图谱、身份和历史。
- `POST /v1/backups` 使用 SQLite 在线备份接口，在服务可用时创建一致的快照。
- `POST /v1/backups/restore` 会先按当前向量维度、VexDB-Lite 和 collection 配置
  打开候选库。新库确认可用前会保留旧库；失败时继续使用旧库并自动重启 REST。

远程 embedding 暂时不可用时，`/v1/events` 仍会保存原始文字，并返回
`embedding_pending`。服务恢复后用同一个 `event_id` 重试，会补写向量，不会因为
幂等重放而跳过。

配置 `MEMME_API_KEY` 后，只有 `/health` 可以匿名访问。`/diagnose` 会真实检查
embedding 和 LLM，因此也必须携带 Bearer Token。

一个 `session_id` 会永久绑定第一次写入的 `user_id`、`agent_id`、`app_id` 和
`run_id`。换主人或换宠物复用同一个会话会返回 `400`。

## VexDB-Lite 存储

MemMe 使用普通 SQLite 数据库保存权威数据，使用 VexDB-Lite 的持久化
`GRAPH_INDEX` 做向量检索。当前源码不再使用 `sqlite-vec` 或 DuckDB。

仓库里的脚本会根据当前 macOS/Linux 架构下载固定的 VexDB-Lite v0.0.17，并
同时校验压缩包和动态库的 SHA-256：

```bash
export MEMME_VEXDB_LITE_EXTENSION="$(bash scripts/download-vexdb-lite-extension.sh)"
```

当前动态扩展支持 macOS、Linux 的 x64 和 arm64。VexDB-Lite v0.0.17 没有
Windows SQLite 扩展。移动端和 WASM crate 已经存在，但还没有接到同一个
VexDB-Lite SQLite 运行时。

只加载来源可信的扩展。SQLite 扩展会作为原生代码运行在应用进程里。

## 数据和隐私

```text
momo-memory.db             全部权威记忆数据
momo-memory.db.replica     可选的本地副本
```

- 一个主人的记忆不会进入另一个主人的查询。
- 一只宠物的关系记忆不会进入另一只宠物的查询。
- 过期和已被替代的事实会在融合前被过滤。
- 不可变的安全记忆不能被静默修改或删除。
- `backup_to_path()` 生成可以搬走的 SQLite 快照。
- `full_export()` / `full_import()` 和 REST 数据接口可以在不依赖云服务的情况下
  搬移全部用户数据层。
- 宿主应用决定是否同步、如何加密、保存多久、是否保留原始音频。语音玩具只保存
  文本事件也能做长期记忆，没有必要默认保存原始音频。

## 包和 API

| 组件 | 当前用途 |
|---|---|
| `memme-core` | Rust 记忆内核、生命周期、检索、图谱、备份 |
| `memme-embeddings` | ONNX、OpenAI 兼容接口、Ollama 向量 |
| `memme-llm` | 可选的 OpenAI、Anthropic、Gemini、Ollama 提取 |
| [`@wjmwjmwb/memme`](https://www.npmjs.com/package/@wjmwjmwb/memme) | macOS / Linux 的 Node.js、Electron 绑定 |
| `memme-python` | PyO3 绑定；当前 SQLite 版本需要从源码构建 |
| `memme-ffi` | Swift/C UniFFI；VexDB-Lite 移动端接线还没完成 |
| `memme-server` | 自托管 REST API |
| `memme-mcp` | MCP stdio 服务 |

Node.js 项目使用带作用域的 npm 包。Rust 和 Python 项目请固定一个源码版本，并从
本仓库构建 SQLite 引擎。

## 运行 PetMemBench

```bash
export MEMME_VEXDB_LITE_EXTENSION="$(bash scripts/download-vexdb-lite-extension.sh)"
cargo run --release -p memme-core --example pet_memory_benchmark -- \
  --dataset benchmarks/petmem/scenarios.json \
  --output benchmarks/petmem/results/latest.json
```

## 参与贡献

现在最有价值的贡献是 AI 宠物真实场景。一个好的测试应明确主人、宠物、时间、
应该找回的记忆、禁止出现的记忆和隐私范围。

参见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

Apache-2.0，参见 [LICENSE](LICENSE)。

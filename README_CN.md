[English](README.md) | 中文

# MemMe

**首个面向移动端和边缘设备的可嵌入 AI 记忆引擎。**

> [mem0](https://github.com/mem0ai/mem0) 的开源替代方案 — 离线优先、单文件部署、延迟低于 10ms。使用 Rust 构建，提供 Python、Node.js 和 Swift 绑定。

[![CI](https://github.com/vibeinging/MemMe/actions/workflows/ci.yml/badge.svg)](https://github.com/vibeinging/MemMe/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/memme-core.svg)](https://crates.io/crates/memme-core)

## MemMe 是什么？

MemMe 是一个 AI memory layer（AI 记忆层），为 AI agents、personal AI 助手和 conversational memory 系统提供 long-term memory（长期记忆）。基于 Rust 和 DuckDB 构建，将 vector database（向量数据库）、knowledge graph（知识图谱）和 full-text search（全文检索）整合在单个可嵌入文件中 —— 无需任何外部服务。

与云端方案不同，MemMe 专为 on-device AI 和 edge AI 场景设计，支持完整的 offline AI 能力。它在本地完成 embedding、RAG 风格的混合检索和记忆整合，非常适合移动应用、IoT 设备和隐私敏感的部署场景。

- **一体化存储** — 向量、知识图谱、全文索引和变更历史全部存储在单个文件中
- **离线可用** — LLM 为可选项，通过 `LlmProvider` trait 可插拔（Ollama、OpenAI、Anthropic 或不使用）
- **无 LLM 模式下延迟低于 10ms** — 纯向量去重路径，适用于延迟敏感场景
- **多平台绑定** — Python（PyO3）、Node.js（NAPI-RS）、Swift（UniFFI）、WASM

## 核心特性

- **单文件部署** -- 一个 `.duckdb` 文件包含一切：向量、图谱、FTS 索引、历史记录
- **可插拔 LLM** -- 无 LLM 模式（向量去重 <10ms）或接入任意 LLM（事实提取 + 智能去重）
- **知识图谱** -- 实体/关系提取，支持 DuckPGQ 图遍历
- **混合搜索** -- 向量相似度 + BM25 全文检索，通过 Reciprocal Rank Fusion 融合
- **遗忘曲线** -- 基于艾宾浩斯遗忘曲线的记忆衰减，访问时强化稳定性
- **四级作用域** -- `user_id` / `agent_id` / `app_id` / `run_id` 隔离
- **隐私控制** -- 每条记忆独立设置 `LocalOnly`、`Syncable`、`EncryptedSync` 隐私级别
- **电量感知处理** -- 设备电量不足时推迟重量级操作
- **不可变记忆 + TTL** -- 锁定记忆防止修改，或设置自动过期时间
- **高级过滤器** -- Eq/Ne/Gt/Gte/Lt/Lte/In/Contains/IContains，支持 AND/OR 组合
- **导出/导入** -- 完整 JSON 导出，包含记忆、实体和关系
- **多语言** -- Rust 核心 + Python（PyO3）+ Node.js（NAPI-RS）+ Swift（UniFFI）+ WASM

## 基准测试 — AI Memory 准确率（LoCoMo，1540 道题，gpt-4o-mini 评判）

| 类别 | **MemMe** | mem0 | mem0-graph | Zep |
|---|---|---|---|---|
| **单跳推理** | **80.50** | 67.13 | 65.71 | 61.70 |
| **多跳推理** | **55.76** | 51.15 | 47.19 | 41.35 |
| **时序推理** | **59.38** | 55.51 | 58.13 | 49.31 |
| **开放域** | **74.55** | 72.93 | 75.71 | 76.60 |

MemMe 在 [LoCoMo benchmark](https://github.com/snap-stanford/locomo) 的**全部四个类别**中均优于 mem0。检索流水线：4 路搜索（向量 + BM25 + 实体扩散 + 时序），RRF 融合后经 cross-encoder reranking。

## MemMe vs mem0 / Zep — 功能对比

| | **MemMe** | **mem0** | **Zep** |
|---|---|---|---|
| **部署方式** | 单个 `.duckdb` 文件 | 服务器 + Qdrant/Pinecone + Neo4j | 托管云服务 |
| **移动端 / iOS** | 原生支持（UniFFI） | 不支持 | 不支持 |
| **离线支持** | 完整离线 | 需要云端 API | 仅云端 |
| **无 LLM 延迟** | <10ms | 始终需要 LLM | 始终需要 LLM |
| **语言** | Rust 核心 | 仅 Python | Go（服务端） |
| **存储** | 嵌入式 DuckDB | 外部向量库 + 图数据库 | 托管 |
| **知识图谱** | 内置（DuckPGQ） | 外部 Neo4j | 无 |
| **FTS + 混合搜索** | 内置（RRF） | 无 | 部分支持 |
| **Reranking** | 内置（API / ONNX） | 可选（Cohere） | 无 |
| **遗忘曲线** | 内置 | 无 | 无 |
| **隐私分级** | 每条记忆独立设置 | 无 | SOC2/HIPAA（云端） |

## 快速开始

### Rust

```toml
[dependencies]
memme-core = "0.1"
memme-embeddings = { version = "0.1", features = ["onnx"] }
```

```rust
use std::sync::Arc;
use memme_core::{MemoryConfig, MemoryStore, AddOptions, SearchOptions};
use memme_embeddings::onnx::OnnxEmbedder;

fn main() -> memme_core::Result<()> {
    let config = MemoryConfig::new("memory.duckdb", 384);
    let embedder = Arc::new(OnnxEmbedder::new()?);
    let store = MemoryStore::new(config, embedder)?;

    store.add("User prefers dark mode", AddOptions::new("alice"))?;
    store.add("User drinks coffee every morning", AddOptions::new("alice"))?;

    let results = store.search("morning routine", SearchOptions::new("alice").limit(5))?;
    for r in &results {
        println!("{} (score: {:.4})", r.content, r.score.unwrap_or(0.0));
    }
    Ok(())
}
```

### Python

```bash
pip install memme
```

```python
from memme import MemoryStore

store = MemoryStore("memory.duckdb")  # 默认使用本地 ONNX embedding
store.add("User prefers dark mode", user_id="alice")
results = store.search("preferences", user_id="alice")
for r in results:
    print(r["content"], r["score"])
```

### Node.js

```bash
npm install memme
```

```javascript
const { MemoryStore } = require("memme");

const store = new MemoryStore("memory.duckdb");
await store.add("User prefers dark mode", { userId: "alice" });
const results = await store.search("preferences", { userId: "alice" });
console.log(results);
```

### Swift（UniFFI）

```swift
import MemMe

let store = try MemoryStore(dbPath: "memory.duckdb", embedder: "onnx")
try store.add("User prefers dark mode", userId: "alice")
let results = try store.search("preferences", userId: "alice")
```

## 架构 — MemMe 工作原理

```
┌──────────────────────────────────────────────────────────────┐
│                      语言绑定层                               │
│  Python (PyO3)  │  Swift (UniFFI)  │  Node.js (NAPI-RS)     │
│  memme-python      memme-ffi          memme-node             │
├──────────────────────────────────────────────────────────────┤
│                      memme-server (axum REST API)            │
│                      memme-mcp (MCP stdio server)            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                    memme-core (Rust)                          │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────────┐    │
│  │ MemoryStore │  │ GraphStore │  │ SearchEngine         │    │
│  │ add()       │  │ add_graph()│  │ vector search        │    │
│  │ search()    │  │ entities   │  │ FTS (BM25)           │    │
│  │ update_trace│  │ relations  │  │ hybrid RRF           │    │
│  │ delete_trace│  │ traverse   │  │ rerank (API / ONNX)   │    │
│  └─────┬──────┘  └─────┬──────┘  └──────────┬──────────┘    │
│        │               │                     │               │
│  ┌─────┴───────────────┴─────────────────────┴─────────┐     │
│  │          DuckDB 存储层（.duckdb 文件）               │     │
│  │  memories │ entities │ relationships │ history        │     │
│  │  FTS index │ vector index (MemMe-DB optional)      │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌──────────────────┐   ┌──────────────────────────────┐     │
│  │ memme-embeddings  │   │ memme-llm                    │     │
│  │ ├ OnnxEmbedder    │   │ ├ OllamaProvider             │     │
│  │ ├ OpenAIEmbedder  │   │ ├ OpenAIProvider              │     │
│  │ └ OllamaEmbedder  │   │ ├ AnthropicProvider           │     │
│  └──────────────────┘   │ ├ GeminiProvider              │     │
│                          │ └ NoopProvider (fallback)     │     │
│                          └──────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

**工作空间 crate 说明：**

| Crate | 用途 |
|---|---|
| `memme-core` | 核心记忆引擎：CRUD、搜索、图谱、去重、分析 |
| `memme-embeddings` | Embedding trait + ONNX/OpenAI/Ollama 后端 |
| `memme-llm` | LLM trait + Ollama/OpenAI/Anthropic/Gemini 后端，用于事实提取 |
| `memme-python` | Python 绑定，基于 PyO3 + maturin |
| `memme-ffi` | Swift/C 绑定，基于 UniFFI |
| `memme-node` | Node.js 绑定，基于 NAPI-RS |
| `memme-wasm` | WASM 绑定，基于 wasm-bindgen |
| `memme-server` | REST API 服务器（axum） |
| `memme-mcp` | MCP stdio 服务器，适配 Claude Desktop / Cursor |

## API 概览

| 方法 | 说明 |
|---|---|
| `add(content, AddOptions)` | 添加记忆，自动去重 |
| `add_smart(messages, AddOptions)` | LLM 驱动的事实提取和记忆管理 |
| `search(query, SearchOptions)` | 向量相似度搜索 |
| `hybrid_search(query, HybridSearchOptions)` | 向量 + FTS，通过 RRF 融合 |
| `get(id)` | 根据 ID 获取单条记忆 |
| `update_trace(id, content, UpdateOptions)` | 更新记忆内容 |
| `delete_trace(id)` | 删除记忆 |
| `list_traces(ListOptions)` | 按 user/agent/app/run 过滤列出记忆 |
| `history(memory_id)` | 获取记忆的变更历史 |
| `batch_update(ids, contents)` | 批量更新（跳过不可变记忆） |
| `batch_delete(ids)` | 批量删除（跳过不可变记忆） |
| `add_graph(text, user_id, llm)` | 提取实体/关系并写入知识图谱 |
| `search_graph(query, user_id)` | 搜索知识图谱 |
| `consolidate()` | 衰减保留率，清理已过期/低保留率的记忆 |
| `export(user_id) / import(data)` | 完整 JSON 导出/导入 |
| `user_stats(user_id)` | 用户维度的 OLAP 分析 |
| `memory_frequency(user_id)` | 记忆创建趋势 |
| `top_entities(user_id)` | 知识图谱中出现频率最高的实体 |
| `add_procedure / get_procedure` | 程序性记忆（技能、工作流） |
| `export_changes_since(version)` | 增量同步 delta |
| `storage_stats()` | 记忆数量、实体数量、估计数据库大小 |

## 配置

```rust
let config = MemoryConfig {
    db_path: "memory.duckdb".into(),       // 或 ":memory:"
    collection_name: "default".into(),      // 表前缀
    embedding_dims: 384,                    // 必须与 embedder 维度一致
    dedup_threshold: 0.15,                  // 去重用的余弦距离阈值
    default_limit: 10,                      // 默认搜索/列表返回条数
    enable_graph: true,                     // smart 模式下自动进行图谱提取
    enable_forgetting_curve: true,          // 艾宾浩斯衰减
    retention_weight: 0.7,                  // 保留率在评分中的权重
    max_memories_per_user: Some(1000),      // 超出后自动剪枝
    pruning_strategy: PruningStrategy::LRU, // 或 Importance、Decay
    auto_prune: true,                       // add() 超限时自动剪枝
    inclusion_prompt: Some("Extract work-related tasks".into()),
    exclusion_prompt: Some("Ignore passwords and secrets".into()),
    power_config: Some(PowerConfig {        // 移动端电量感知
        full_power_threshold: 0.5,
        power_save_threshold: 0.2,
        defer_when_critical: true,
    }),
    ..Default::default()
};
```

## REST API

启动服务器：

```bash
cargo run -p memme-server -- --db memory.duckdb --port 8080
```

请求示例：

```bash
# 添加记忆
curl -X POST http://localhost:8080/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "User likes dark mode", "user_id": "alice"}'

# 搜索
curl -X POST http://localhost:8080/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "UI preferences", "user_id": "alice", "limit": 5}'

# 根据 ID 获取
curl http://localhost:8080/v1/memories/{id}

# 更新
curl -X PUT http://localhost:8080/v1/memories/{id} \
  -H "Content-Type: application/json" \
  -d '{"content": "User switched to light mode"}'

# 删除
curl -X DELETE http://localhost:8080/v1/memories/{id}

# 列表
curl -X POST http://localhost:8080/v1/memories/list \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice"}'

# 历史记录
curl http://localhost:8080/v1/memories/{id}/history

# 导出 / 导入
curl -X POST http://localhost:8080/v1/memories/export \
  -d '{"user_id": "alice"}'
curl -X POST http://localhost:8080/v1/memories/import \
  -H "Content-Type: application/json" \
  -d @exported.json

# 健康检查
curl http://localhost:8080/health
```

## MCP 服务器

MemMe 内置 MCP（Model Context Protocol）stdio 服务器，可与 Claude Desktop、Cursor 及其他 MCP 客户端集成。

在 Claude Desktop 配置文件（`claude_desktop_config.json`）中添加：

```json
{
  "mcpServers": {
    "memme": {
      "command": "/path/to/memme-mcp",
      "args": ["--db", "memory.duckdb"]
    }
  }
}
```

可用 MCP 工具：`add_memory`、`search_memory`、`get_memory`、`update_memory`、`delete_memory`、`list_memories`、`delete_all_memories`。

构建 MCP 服务器：

```bash
cargo build -p memme-mcp --release
```

## 从源码构建

### 默认（捆绑 DuckDB）

```bash
git clone --recurse-submodules https://github.com/vibeinging/MemMe.git
cd MemMe
cargo build --release
```

### 使用 MemMe-DB（HNSW）

用于支持分区级过滤的高级向量索引：

```bash
# 1. 构建 DuckDB + MemMe-DB 静态库
./scripts/build_memme_db.sh

# 2. 启用 memme-db feature 构建 MemMe
export DUCKDB_LIB_DIR=<path>/build/release/src
export DUCKDB_INCLUDE_DIR=<path>/src/include
cargo build -p memme-core --no-default-features --features memme-db --release
```

### Feature 标志

| Feature | 说明 |
|---|---|
| `bundled`（默认） | 从源码编译 DuckDB |
| `memme-db` | 链接预编译的 DuckDB + MemMe-DB |
| `api-rerank` | 基于 API 的 cross-encoder reranker（Jina/Cohere/DashScope） |
| `onnx-rerank` | 通过 fastembed 在本地运行 ONNX cross-encoder reranker |

### 运行测试

```bash
cargo test                                    # 全部单元测试（244 个）
```

### 构建 Python 包

```bash
cd crates/memme-python
maturin develop --release
```

## 贡献

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献指南。

- [报告 Bug](https://github.com/vibeinging/MemMe/issues/new?template=bug_report.md)
- [提交功能请求](https://github.com/vibeinging/MemMe/issues/new?template=feature_request.md)
- [路线图](docs/ROADMAP.md)

## 应用场景

- **AI Agents** — 为聊天机器人和自主代理提供跨对话的持久记忆
- **Personal AI 助手** — 在设备端记住用户偏好、习惯和上下文
- **移动应用** — 离线优先的记忆系统，无需网络即可使用，联网时自动同步
- **RAG 管道** — 混合检索（vector + graph + FTS）构建本地知识库
- **Digital Twins（数字分身）** — 构建基于记忆的个人数字化表示

## 社区

- [GitHub Discussions](https://github.com/vibeinging/MemMe/discussions) — 问题与想法
- [Issue Tracker](https://github.com/vibeinging/MemMe/issues) — Bug 报告与功能请求

## 许可证

Apache-2.0 —— 详见 [LICENSE](LICENSE)。

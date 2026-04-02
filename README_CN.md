[English](README.md) | 中文

<div align="center">

# MemMe

**Memories that are actually yours.**

一个可嵌入的 AI 记忆引擎。一个文件。你的设备。你说了算。

[![CI](https://github.com/vibeinging/MemMe/actions/workflows/ci.yml/badge.svg)](https://github.com/vibeinging/MemMe/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/memme-core.svg)](https://crates.io/crates/memme-core)

</div>

---

你和 AI 助手聊了三个月。它知道你的工作、你的口味、你的思考方式。

然后有一天，平台修改了隐私政策。或者你想换一个模型。或者服务停了。

你的三个月记忆，蒸发了。

不是因为技术做不到。是因为**那些记忆从来就不属于你。**

MemMe 想改变这件事。

## 你的记忆，凭什么在别人的服务器上？

你的日记、照片、通讯录，存在你自己的手机里。你可以备份、迁移、删除。

但你和 AI 的全部对话记忆呢？存在 OpenAI 的服务器上。存在 Claude 的云端。你不知道谁能访问它们，不知道它们被用来训练了什么，不知道明天它们还在不在。

AI 记忆比普通数据敏感得多。它不只是你说了什么，更是你**是什么样的人**——你的思维模式、决策习惯、情感状态、人际关系。这是最私密的个人画像。

**这样的数据，应该放在你自己的设备上。**

## MemMe：一个文件，装下全部记忆

```
memory.duckdb              ← 你的全部记忆，一个文件
│
├── memories               内容 + 向量 + 元数据
├── entities / relationships   知识图谱（人、地、事 + 关系）
├── sessions / events      原始对话流
├── episodes               情景记忆（对话压缩后的故事）
├── identity               身份特征（你是谁）
├── procedures             程序性记忆（技能、习惯）
├── meditations            冥想日志（记忆整合记录）
├── history                变更审计（每次读写都有记录）
└── memme_config           运行时配置
```

要备份？复制文件。要迁移到新手机？复制文件。要彻底删除？删除文件。

Rust 写的内核。接上 LLM 做智能提取，不接也能跑——纯向量模式延迟低于 10ms。Python、Node.js、Swift/Kotlin 原生绑定（UniFFI），嵌入式设备和机器人也能跑，不是 HTTP 套壳。

应用倒了，记忆还在。模型换了，记忆还在。平台跑了，记忆还在。

## 试一下

```bash
pip install memme
python demos/playground/server.py
```

浏览器自动打开。存记忆、搜记忆、跟记忆对话。数据就在你的机器上。

## 跑分

[LoCoMo 基准测试](https://github.com/snap-stanford/locomo)（1540 题，GPT-4o-mini 评判）：

| 类别 | **MemMe** | mem0 | mem0-graph | Zep |
|------|-----------|------|------------|-----|
| 单跳 | **79.43** | 67.13 | 65.71 | 61.70 |
| 多跳 | **65.73** | 51.15 | 47.19 | 41.35 |
| 时序 | **70.83** | 55.51 | 58.13 | 49.31 |
| 开放域 | **82.28** | 72.93 | 75.71 | 76.60 |

## 它能做什么

### 四层记忆架构

模拟人类从感知到认知的完整层次：

- **Stream（感知流）** — 原始输入，忠实记录
- **Episode（情景记忆）** — 事件聚合成有意义的"故事"
- **Semantic（语义记忆）** — 从故事中提炼知识和事实
- **Identity（身份记忆）** — 最高阶抽象：你是谁

### "冥想"机制

人在睡眠时整理白天的经历。MemMe 也一样——空闲时自动把零散对话聚合成情景、从情景中提炼事实、用 LLM 裁决每条事实是新增/更新/删除、构建知识图谱并关联实体与记忆。内置 FSRS 遗忘曲线，三个月前随口提的餐厅自然淡出，反复提及的偏好越来越牢固。

### 反思与反馈学习

- **反思（Reflect）** — 基于近期记忆和身份特征，LLM 生成主题洞察和聚焦建议
- **反馈学习（Learn from Feedback）** — 从用户纠正中提炼行为原则，存为高重要度记忆和身份特征
- **健康诊断（Diagnose）** — 一键检测存储、Embedder、LLM 连通性，每项附带延迟报告

### 四通道混合检索

向量语义搜索 + BM25 全文搜索 + 实体图谱导航 + 时间维度，四路并行，RRF 融合后经 cross-encoder 重排序。

### 知识图谱

LLM 自动提取实体和关系，存在 DuckDB 里，SQL 直接查。不需要外挂 Neo4j。

### 隐私控制

每条记忆独立设置隐私级别——仅本地、可同步、加密同步。医疗记录锁在手机里，咖啡偏好同步到所有设备。完整审计日志，每次读写都有记录。

### 全端原生

| 平台 | 方式 |
|------|------|
| Mac / Linux / Windows | Rust 原生 |
| iPhone / iPad | Swift 绑定 (UniFFI) |
| Android | NDK 原生 |
| Web / Electron | Node.js 绑定 (NAPI-RS) |
| Python 生态 | PyO3 绑定 |
| 机器人 / IoT | Rust 编译到 ARM |

所有平台共享同一个 Rust 内核，同一个 `.duckdb` 文件格式。

### 生态集成

| 集成 | 状态 | 说明 |
|------|------|------|
| **Claude Desktop / Cursor** | 已完成 | MCP 协议接入，作为 AI 的长期记忆 |
| **REST API** | 已完成 | axum 服务，23 个端点，Bearer 认证 |
| **[YiYi](https://github.com/vibeinging/YiYi)** | 已集成 | 桌面 AI 个人助手——能操作电脑、执行任务、管理文件，记忆系统由 MemMe 驱动 |
| **OpenClaw** | 初步搭建 | 开源 Agent 框架的记忆插件 |
| **Dora-rs** | 初步搭建 | Rust 机器人框架的记忆节点 |
| **LeRobot** | 初步搭建 | Hugging Face 机器人框架的记忆封装 |
| **Copper-rs** | 初步搭建 | 实时机器人框架的 CuTask 集成 |
| **LangChain / LlamaIndex** | 规划中 | 主流 LLM 框架适配 |

## 对比

| | **MemMe** | **mem0** | **Zep** |
|---|---|---|---|
| 部署 | 一个 `.duckdb` 文件 | 服务器 + Qdrant + Neo4j | 托管云 |
| 移动端 | 原生支持 | 不支持 | 不支持 |
| 离线 | 完整支持 | 需要云 API | 仅云端 |
| 无 LLM 延迟 | <10ms | 必须有 LLM | 必须有 LLM |
| 语言 | Rust | 仅 Python | Go |
| 知识图谱 | 内置 | 外挂 Neo4j | 无 |
| 混合检索 | 四通道 + RRF | 无 | 部分 |
| 遗忘曲线 | 内置 | 无 | 无 |

## 谁应该用 MemMe

- **在意数据主权的人** — 记忆在你的设备上，不在别人的服务器里
- **数字分身开发者** — 跨年的人格一致性，应用迭代但"我是谁"不丢
- **端侧 AI 助手** — 手机上的私人助理，完全离线
- **隐私敏感场景** — 医疗、法律、金融，数据不出设备
- **非 Python 开发者** — Rust / Swift / Node.js 终于有了原生记忆引擎
- **具身智能** — 低延迟、可嵌入，一个文件扔进去就能跑

---

## 技术文档

以下是面向开发者的技术细节。

### 快速上手

**Python**

```python
from memme import MemoryStore

store = MemoryStore("memory.duckdb")
store.add("Alex 喜欢喝燕麦拿铁", user_id="alex")
store.add("女儿 Mia 的生日是 3 月 15 日", user_id="alex")

results = store.search("家人的生日", user_id="alex")
for r in results:
    print(r["content"], r["score"])
```

**Rust**

```rust
use std::sync::Arc;
use memme_core::{MemoryConfig, MemoryStore, AddOptions, SearchOptions};
use memme_embeddings::onnx::OnnxEmbedder;

fn main() -> memme_core::Result<()> {
    let config = MemoryConfig::new("memory.duckdb", 384);
    let embedder = Arc::new(OnnxEmbedder::new()?);
    let store = MemoryStore::new(config, embedder)?;

    store.add("Alex 喜欢喝燕麦拿铁", AddOptions::new("alex"))?;
    let results = store.search("饮品偏好", SearchOptions::new("alex").limit(5))?;
    Ok(())
}
```

**Node.js**

```javascript
const { MemoryStore } = require("memme");

// OpenAI embedding（或 newMock() 免 API 测试）
const store = MemoryStore.newOpenai(process.env.OPENAI_API_KEY, "memory.duckdb");
await store.add("Alex 喜欢喝燕麦拿铁", "alex");
const results = await store.search("饮品偏好", "alex");
```

**Swift (UniFFI)**

```swift
import MemMe

// 宿主 App 提供 HTTP 传输（URLSession / OkHttp）
let store = try MemoryStore.newWithHttpClient(
    dbPath: "memory.duckdb",
    httpClient: myHttpClient,  // 实现 HttpClient 协议
    apiKey: "sk-...",
    model: "text-embedding-3-small",
    dims: 1536
)
try store.add("Alex 喜欢喝燕麦拿铁", userId: "alex")
let results = try store.search("饮品偏好", userId: "alex")
```

### 架构

```
┌──────────────────────────────────────────────────────────┐
│                      语言绑定                              │
│  Python (PyO3)  │  Node.js (NAPI-RS)  │  Swift (UniFFI)  │
├──────────────────────────────────────────────────────────┤
│  REST API (axum)          │  MCP 服务 (stdio)             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│                   memme-core (Rust)                       │
│                                                          │
│  事件流 ──► 会话 ──► 片段 ──► 记忆                         │
│                                │                         │
│                      ┌─────────┤                         │
│                      ▼         ▼                         │
│                  身份特征     知识图谱                      │
│                                                          │
│  搜索：向量 + BM25 + 图谱 + 时间                           │
│        ──► RRF 融合 ──► 重排序                             │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  DuckDB（.duckdb 单文件）                            │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  memme-embeddings          memme-llm                     │
│  (ONNX / OpenAI / Ollama)  (OpenAI / Anthropic / Gemini  │
│                             / Ollama / Noop)             │
└──────────────────────────────────────────────────────────┘
```

### REST API

```bash
cargo run -p memme-server -- --db-path memory.duckdb --port 8080
```

```bash
# 存记忆
curl -X POST http://localhost:8080/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "Alex 喜欢喝燕麦拿铁", "user_id": "alex"}'

# 搜索
curl -X POST http://localhost:8080/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "饮品偏好", "user_id": "alex"}'
```

完整 API：[docs/openapi.yaml](docs/openapi.yaml) — 粘贴到 [Swagger Editor](https://editor.swagger.io) 浏览 23 个端点。

### MCP 服务

```json
{
  "mcpServers": {
    "memme": {
      "command": "/path/to/memme-mcp",
      "args": ["--db-path", "memory.duckdb"]
    }
  }
}
```

### 从源码构建

```bash
git clone --recurse-submodules https://github.com/vibeinging/MemMe.git
cd MemMe
cargo build --release
cargo test   # 450+ 测试
```

| Feature | 说明 |
|---|---|
| `bundled`（默认） | 从源码编译 DuckDB |
| `memme-db` | 预编译 DuckDB + MemMe-DB (HNSW) |
| `api-rerank` | API 重排序 (Jina/Cohere) |
| `onnx-rerank` | 本地 ONNX 重排序 |

---

## 参与贡献

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。[路线图](docs/ROADMAP.md)。

## 许可证

Apache-2.0 — 见 [LICENSE](LICENSE)。

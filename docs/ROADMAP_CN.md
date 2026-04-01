# MemMe 产品路线图

> 数据伴你终生。
>
> 端侧 AI 记忆引擎 —— Rust 内核，单 DuckDB 文件，多语言原生绑定。
>
> 最后更新：2026-04-01

---

## 已发布功能

MemMe 0.1.0 是一个功能完整的 AI 记忆引擎。以下是已经可用的能力：

### 核心引擎

| 功能 | 说明 |
|------|------|
| **Stream → Session → Episode → Memory** | 四层数据模型。原始事件压缩为片段，片段提取为语义记忆 |
| **冥想（Meditation）** | 记忆整合：遗忘曲线衰减 + 片段→记忆提取 + 图谱构建 + 身份特征蒸馏 |
| **知识图谱** | LLM 实体/关系提取，DuckDB 存储，扩散激活搜索 |
| **四通道混合检索** | 向量 + BM25 全文 + 实体图谱 + 时间维度，RRF 融合排序 |
| **重排序器** | LLM 重排序、ONNX 交叉编码器（fastembed）、API 重排序（Jina/Cohere 兼容） |
| **FSRS 遗忘曲线** | `R(t,S) = (1 + t/(c*S))^(-p)`，访问时强化稳定性 |
| **身份特征** | 从记忆语料中蒸馏高层人格特质 |
| **程序性记忆** | 技能/习惯/工作流的存储与检索 |
| **Smart 模式** | LLM 事实提取 → 向量去重 → ADD/UPDATE/DELETE 决策 |
| **分析（OLAP）** | 用户画像 / 记忆频率 / 热门实体 —— 基于 DuckDB 列式引擎 |
| **隐私控制** | 逐条记忆隐私级别：LocalOnly / Syncable / EncryptedSync |
| **四级作用域** | user_id + agent_id + app_id + run_id 隔离 |
| **生命周期管理** | TTL 过期、LRU/重要度/衰减裁剪、存储上限、延迟写入 |
| **增量同步** | export_changes_since + sync_version 追踪 |
| **Webhook 事件** | memory_add/update/delete HTTP POST 回调 |

### 多语言绑定

| 绑定 | 包名 | 状态 |
|------|------|------|
| **Rust** | crates.io `memme-core` | 发布中 |
| **Python** | PyPI `memme`（PyO3 + maturin） | 发布中 |
| **Node.js / TypeScript** | npm `memme`（NAPI-RS） | 发布中 |
| **Swift / Kotlin** | UniFFI .xcframework | 发布中 |
| **WebAssembly** | wasm-bindgen | 实验性 |

### 基础设施

| 组件 | 说明 |
|------|------|
| **REST API 服务** | axum，23 个端点，Bearer Token 认证 |
| **MCP 服务** | stdio JSON-RPC，支持 Claude Desktop / Cursor |
| **CI/CD** | GitHub Actions：fmt + clippy + 测试（Ubuntu/macOS） |
| **发布流水线** | crates.io、PyPI、npm、iOS/Android 二进制发布 |

### 基准测试（LoCoMo，1540 题，GPT-4o-mini 评判）

| 类别 | MemMe | mem0 | mem0-graph | Zep |
|------|-------|------|------------|-----|
| 单跳 | **80.50** | 67.13 | 65.71 | 61.70 |
| 多跳 | **55.76** | 51.15 | 47.19 | 41.35 |
| 时间 | **59.38** | 55.51 | 58.13 | 49.31 |
| 开放域 | **74.55** | 72.93 | 75.71 | 76.60 |

---

## 路线图

按优先级排列。每个里程碑为下一波用户增长扫清障碍。

### M0：生产加固（公开发布前）

> 把已有引擎打磨到可靠。不加新功能，只修脆弱的地方。

| 任务 | 优先级 | 状态 | 说明 |
|------|--------|------|------|
| LLM 事务安全 | P0 | ✅ 完成 | `add_smart()` 已包裹 BEGIN/COMMIT，失败时 ROLLBACK |
| LLM/Embedding 重试 | P0 | ✅ 完成 | 所有 Provider（OpenAI + Ollama）均有指数退避重试 |
| RRF 候选上限 | P1 | ✅ 完成 | 由 `rrf_candidate_multiplier` 配置控制 |
| LLM 调用超时 | P1 | ✅ 完成 | OpenAI 120s，Ollama 600s（HTTP client 级别） |
| 延迟写入刷新 | P1 | ⚠️ 部分 | 队列上限 500 条自动刷新，缺定时刷新机制 |
| API 命名统一 | P1 | ❌ 待做 | Server 用 `top_k`，Core 用 `limit`；跨绑定 f32/f64 不一致 |

### M1：开发者体验（发布后第 1-2 周）

> 让试用和接入变得极其简单。

| 任务 | 优先级 | 状态 | 说明 |
|------|--------|------|------|
| **公开 API Rustdoc** | P0 | ✅ 完成 | lib.rs 所有公开类型和方法已有 `///` 文档注释 |
| **错误类型文档** | P2 | ✅ 完成 | error.rs 所有 MemoryError 变体已有文档 |
| **OpenAPI 规范** | P0 | ✅ 完成 | docs/openapi.yaml — OpenAPI 3.1 覆盖全部 23 个端点 |
| **WASM Playground** | P1 | ❌ 待做 | 浏览器体验版，零安装，10 秒试用 MemMe |
| **快速上手指南** | P1 | ⚠️ 部分 | README 有各语言代码示例，缺独立的单页指南 |

### M2：数据导入（第 3-4 周）

> 带上你的历史数据。这是用户增长的杀手级驱动力。

| 任务 | 优先级 | 说明 |
|------|--------|------|
| **ChatGPT 历史导入** | P0 | 解析 ChatGPT 导出的 `conversations.json` → Session/Episode 模型 |
| **Claude 历史导入** | P0 | 解析 Claude 导出格式 → Session/Episode 模型 |
| **通用对话导入** | P1 | 标准 JSON/JSONL schema，支持任意聊天记录 |
| **导入 CLI 工具** | P1 | `memme import --format chatgpt --file conversations.json --db memory.duckdb` |
| **导入进度与统计** | P2 | 进度条 + 报告：导入对话数、提取记忆数、发现实体数 |

### M3：生态集成（第 5-8 周）

> 到开发者已经在的地方去。

| 任务 | 优先级 | 状态 | 说明 |
|------|--------|------|------|
| **OpenClaw 插件** | P1 | ⚠️ 部分 | Tools 已搭建（store/recall/forget），hooks 未接入 |
| **LangChain Memory** | P0 | ❌ 待做 | 实现 LangChain BaseMemory 接口 |
| **LlamaIndex 集成** | P1 | ❌ 待做 | MemMe 作为检索器/存储后端 |
| **CrewAI / AutoGen** | P1 | ❌ 待做 | 多 Agent 框架的共享记忆 |
| **Obsidian 插件** | P2 | ❌ 待做 | 双向同步：Obsidian 笔记 ↔ MemMe 记忆 |

### M4：端侧与具身智能（第 9-12 周）

> 护城河：MemMe 跑在别人跑不了的地方。

| 任务 | 优先级 | 状态 | 说明 |
|------|--------|------|------|
| **LeRobot 集成** | P0 | ⚠️ 部分 | MemoryRobot 封装 + episode 日志已搭建 |
| **Copper-rs CuTask** | P1 | ⚠️ 部分 | cu-memme crate 消息类型已有，需完整 CuTask 实现 |
| **iOS SDK（CocoaPods/SPM）** | P1 | ❌ 待做 | 预编译 .xcframework + 包管理器分发 |
| **Android SDK（Maven/Gradle）** | P1 | ❌ 待做 | AAR 包 + Kotlin DSL |
| **端侧 LLM** | P2 | ❌ 待做 | 集成 llama.cpp / MLX，完全离线 Smart 模式 |
| **树莓派基准测试** | P2 | ❌ 待做 | 用真实性能数据证明边缘部署能力 |

### M5：高级记忆科学（第 13-20 周）

> 推动 AI 记忆能做到的事情的边界。

| 任务 | 优先级 | 状态 | 说明 |
|------|--------|------|------|
| **Sleep-time 计算** | P1 | ⚠️ 部分 | Meditation 系统已有衰减+提取，缺后台自动调度 |
| **视觉记忆** | P1 | ❌ 待做 | 多模态：从图片中提取和存储记忆 |
| **因果推理** | P2 | ❌ 待做 | 跨记忆追踪因果链 |
| **情感标注** | P2 | ❌ 待做 | 情感/情绪提取与检索 |
| **记忆聚类** | P2 | ❌ 待做 | 自动发现主题簇，生成记忆地图 |
| **多 Agent 隔离** | P2 | ❌ 待做 | 完整的 per-agent 记忆空间 + 可选共享策略 |

### M6：规模与性能（持续进行）

| 任务 | 优先级 | 说明 |
|------|--------|------|
| **连接池优化** | P1 | WAL 模式实现读写并行 |
| **Async Rust API** | P1 | 核心方法原生异步 |
| **流式搜索** | P2 | 各通道结果到达即返回 |
| **分布式模式** | P3 | 多节点 MemMe，面向云端部署（远期） |

---

## 明确不做的事

- **托管云服务** —— MemMe 是引擎，不是平台。别人可以基于它构建 SaaS。
- **替代向量数据库** —— MemMe 是恰好用了向量的记忆引擎，不是通用向量数据库。
- **Python 优先** —— Rust 是唯一的事实来源。所有绑定都是生成的，不是手写的。

---

## 架构

```
        ┌─────────────────────────────────────────────┐
        │               语言绑定                        │
        │  Python │ Node.js │ Swift │ Kotlin │ WASM   │
        ├─────────────────────────────────────────────┤
        │  REST API (axum)  │  MCP 服务 (stdio)        │
        ├─────────────────────────────────────────────┤
        │                                             │
        │              memme-core (Rust)              │
        │                                             │
        │  事件流 ──► 会话 ──► 片段 ──► 记忆           │
        │                                │            │
        │                      ┌─────────┤            │
        │                      ▼         ▼            │
        │                   身份特征    知识图谱        │
        │                                             │
        │  搜索：向量 + BM25 + 图谱 + 时间             │
        │        ──► RRF 融合 ──► 重排序               │
        │                                             │
        │  ┌───────────────────────────────────────┐  │
        │  │   DuckDB（.duckdb 单文件）              │  │
        │  │   memories │ entities │ relationships  │  │
        │  │   sessions │ episodes │ events         │  │
        │  │   identity │ procedures │ history      │  │
        │  └───────────────────────────────────────┘  │
        │                                             │
        │  memme-embeddings    memme-llm              │
        │  (ONNX/OpenAI/       (Ollama/OpenAI/        │
        │   Ollama)             Anthropic/Gemini)      │
        └─────────────────────────────────────────────┘
```

---

## 如何参与贡献

详见 [CONTRIBUTING.md](../CONTRIBUTING.md)。我们特别欢迎：

- **Good first issues**：在 GitHub Issues 中标注
- **语言绑定**：将新的核心 API 暴露到 Python/Node/Swift
- **导入格式**：支持更多聊天记录格式
- **生态集成**：LangChain、LlamaIndex 及其他框架适配器
- **基准测试**：在新数据集或新硬件上运行 MemMe

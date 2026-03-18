# MemMe 产品路线图

> 基于 DuckDB + MemMe-DB 的端侧 AI 记忆引擎，Rust 核心，对标 mem0
>
> 最后更新：2026-03-17

---

## 功能完成状态

| mem0 功能 | 状态 | 说明 |
|-----------|------|------|
| **add（无 LLM）** | ✅ 完成 | 向量去重 + hash 快速匹配 + metadata 保留 |
| **add (smart 模式)** | ✅ 完成 | LLM 事实提取 → 向量搜索 → LLM 决策 ADD/UPDATE/DELETE |
| **search** | ✅ 完成 | 向量相似度搜索 + threshold 过滤 + limit |
| **get / update / delete** | ✅ 完成 | 完整 CRUD |
| **list** | ✅ 完成 | 按 user_id / agent_id 过滤 |
| **history** | ✅ 完成 | 变更审计（ADD/UPDATE/DELETE 事件记录） |
| **SQL 注入防护** | ✅ 完成 | 全部参数化查询 |
| **Config 校验** | ✅ 完成 | dims/threshold/collection/db_path 验证 |
| **app_id 四级作用域** | ✅ 完成 | user_id + agent_id + app_id + run_id 四级隔离（对标 mem0） |
| **不可变记忆 (immutable)** | ✅ 完成 | immutable=true 的记忆不可 update/delete |
| **TTL / 自动过期** | ✅ 完成 | expiration_date 字段 + consolidate 自动清理过期记忆 |
| **自定义分类 (categories)** | ✅ 完成 | VARCHAR[] 列表 + list_contains 过滤 |
| **高级过滤运算符** | ✅ 完成 | Eq/Ne/Gt/Gte/Lt/Lte/In/Contains/IContains + AND/OR 组合 |
| **自定义时间戳** | ✅ 完成 | UpdateOptions.timestamp 设置自定义更新时间 |
| **Inclusion/Exclusion Prompts** | ✅ 完成 | MemoryConfig 支持提取/忽略指导 |
| **Export / Import** | ✅ 完成 | 数据导入导出（JSON 格式，含 metadata/categories） |
| **单元测试** | ✅ 完成 | 164 个测试（storage/dedup/memory/config/types/smart/graph/search/analytics/rerank/embeddings/llm） |
| **集成测试** | ✅ 完成 | 6 个端到端测试 + 6 个 LLM 集成测试（OpenAI 格式 API） |
| **Python 测试** | ✅ 完成 | 31 个 pytest |
| **Example 程序** | ✅ 完成 | basic.rs + smart.rs |
| **Graph Memory** | ✅ 完成 | entities/relationships 表 + LLM 提取 + 递归 CTE 图遍历 |
| **FTS 全文搜索** | ✅ 完成 | DuckDB FTS 扩展 + BM25 搜索 |
| **混合检索 (向量+FTS+RRF)** | ✅ 完成 | Reciprocal Rank Fusion |
| **MemMe-DB 集成** | ✅ 完成 | `memme-db` feature + db_engine 自包含静态构建 + Rust 链接 |
| **Python 绑定** | ✅ 完成 | PyO3 + maturin, `pip install memme` |
| **Swift 绑定** | ✅ 完成 | UniFFI proc macros, memme-ffi crate |
| **JS/TS 绑定** | ✅ 完成 | NAPI-RS, memme-node crate |
| **WASM 绑定** | ✅ 完成 | wasm-bindgen, memme-wasm crate（需 DuckDB-WASM 集成） |
| **Reranker (LLM)** | ✅ 完成 | LLM reranker 已实现（4 个测试） |
| **Analytics (OLAP)** | ✅ 完成 | user_stats / memory_frequency / top_entities（6 个测试） |
| **REST API Server** | ✅ 完成 | axum HTTP 服务器，完整 CRUD + search + export/import |
| **MCP Server** | ✅ 完成 | Model Context Protocol stdio 服务器，Claude Desktop/Cursor 集成 |
| **keyword_search 标志** | ✅ 完成 | SearchOptions.keyword_search 控制 FTS+RRF 混合搜索 |
| **fields 字段选择** | ✅ 完成 | SearchOptions.fields 指定返回字段 |
| **Batch 批量操作** | ✅ 完成 | batch_update / batch_delete，跳过 immutable 记忆 |
| **Shared Memory** | ✅ 完成 | memory_type 字段（session/long_term/shared）+ 过滤 |
| **Webhook 事件系统** | ✅ 完成 | memory_add/update/delete HTTP POST 回调，fire-and-forget |
| **Node.js 异步 API** | ✅ 完成 | 18 个方法全部 async/Promise，tokio spawn_blocking |
| **ONNX 本地 Reranker** | ✅ 完成 | fastembed TextRerank 离线重排序，`onnx-rerank` feature |
| **Async Rust API** | ✅ 完成 | FFI 层 spawn_blocking 包装，UniFFI/NAPI async 导出 |
| **Procedural Memory** | ✅ 完成 | procedures 表 + CRUD（add/get/list/delete） |
| **Vision Message 支持** | ✅ 完成 | ChatMessage.image_url/image_type 字段（Vision LLM 提取预留） |
| **存储上限 + 自动裁剪** | ✅ 完成 | max_memories_per_user + LRU/Importance/Decay 策略 |
| **隐私控制** | ✅ 完成 | Privacy 枚举（LocalOnly/Syncable/EncryptedSync）+ export 过滤 |
| **电量感知处理** | ✅ 完成 | PowerConfig + 原子电量状态 + 关键电量自动延迟操作 |
| **延迟操作队列** | ✅ 完成 | DeferredOp 队列 + process_deferred() + deferred_count() |
| **增量同步原语** | ✅ 完成 | SyncDelta/SyncChange + export_changes_since + sync_version 追踪 |
| **存储统计** | ✅ 完成 | StorageStats（内存数/实体数/关系数/估算大小） |
| **单元测试** | ✅ 完成 | 244 个测试（237 lib + 6 integration + 1 doctest） |

---

## P0: 代码骨架 + 收尾 ✅ 已完成

### P0-Phase1: 代码骨架（已完成）

4 个 crate 并行开发：

- **memme-core** — 核心记忆引擎（CRUD + 向量去重 + DuckDB 存储层）
- **memme-embeddings** — Embedding 抽象层（Embedder trait + ONNX/OpenAI/Ollama/Mock 实现）
- **memme-llm** — LLM 抽象层（LlmProvider trait + Ollama/OpenAI/NoOp 实现 + mem0 Prompt 移植）
- **memme-ffi** — FFI 绑定占位

### P0-Phase2: 收尾（已完成）

| 任务 | 内容 |
|------|------|
| **Task 1: SQL 注入修复** | `insert_memory`/`update_memory`/`record_history` 全部参数化查询 |
| **Task 2: Bug 修复** | metadata 保留、hash 快速去重、score 文档 |
| **Task 3: Config 校验** | 5 项验证规则 + MemoryStore::new() 入口调用 |
| **Task 4: 单元测试** | 47 个测试覆盖 storage/dedup/memory/config/types |
| **Task 5: Smart 模式** | smart.rs + add_smart() + 5 个测试 |
| **Task 6: 集成测试** | 6 个端到端测试 |
| **Task 7: Example** | basic.rs + smart.rs |

---

## P1: 对标 mem0 完整功能 ✅ 已完成

### Task P1-1: MemMe-DB 预编译集成

**目标**: 替换 DuckDB bundled 模式，让 HNSW 真正生效

**方案**: 预编译 DuckDB + MemMe-DB 静态库，通过环境变量链接到 duckdb-rs

```bash
# 1. 预编译 DuckDB + MemMe-DB
cd duckdb
cmake -B build/for_rust -DCMAKE_BUILD_TYPE=Release -DBUILD_SHELL=OFF -DBUILD_UNITTESTS=OFF
cmake --build build/for_rust --target duckdb_static -j8

# 2. 链接到 MemMe
export DUCKDB_LIB_DIR=/.../build/for_rust/src
export DUCKDB_INCLUDE_DIR=/.../src/include
cargo build
```

**验证**:
- HNSW 创建成功
- HNSW 带 user_id 过滤搜索生效
- 向量搜索性能 benchmark

### Task P1-2: Graph Memory（知识图谱）

**目标**: 实体-关系图谱，对标 mem0 的 Neo4j Graph Memory

**依赖**: DuckPGQ 扩展（SQL/PGQ 标准语法）

**Schema**:
```sql
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    name VARCHAR NOT NULL,
    entity_type VARCHAR,
    embedding FLOAT[384],
    user_id VARCHAR NOT NULL,
    mentions INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE relationships (
    id UUID PRIMARY KEY,
    source_id UUID REFERENCES entities(id),
    target_id UUID REFERENCES entities(id),
    relation_type VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

-- DuckPGQ 属性图
CREATE PROPERTY GRAPH knowledge_graph
VERTEX TABLES (entities)
EDGE TABLES (
    relationships SOURCE KEY (source_id) REFERENCES entities (id)
                  DESTINATION KEY (target_id) REFERENCES entities (id)
);
```

**API**:
```rust
impl MemoryStore {
    fn add_graph(&self, text: &str, user_id: &str, llm: Arc<dyn LlmProvider>) -> Result<()>;
    fn search_graph(&self, query: &str, user_id: &str) -> Result<Vec<GraphResult>>;
}
```

**流程**:
1. LLM extract_entities(text) → `{entity, entity_type}`
2. LLM establish_relationships(entities) → `{source, relationship, destination}`
3. Embedding 做实体去重（cosine similarity >= 0.7 合并节点）
4. 搜索时：提取查询实体 → 向量匹配图节点 → 获取关系 → BM25 重排序三元组

### Task P1-3: FTS 全文搜索

**目标**: 关键词检索，补充向量搜索的不足

**依赖**: DuckDB FTS 扩展

```sql
-- 创建全文索引
PRAGMA create_fts_index('memories', 'id', 'content');

-- 全文搜索
SELECT id, content, fts_main_memories.match_bm25(id, 'coffee morning') AS bm25_score
FROM memories
WHERE bm25_score IS NOT NULL
ORDER BY bm25_score DESC
LIMIT 10;
```

### Task P1-4: 混合检索 + RRF

**目标**: 向量搜索 + FTS 搜索 + Reciprocal Rank Fusion 融合

```rust
pub struct HybridSearchOptions {
    pub user_id: String,
    pub limit: usize,
    pub vector_weight: f64,  // default 0.7
    pub fts_weight: f64,     // default 0.3
    pub k: usize,            // RRF constant, default 60
}

impl MemoryStore {
    fn hybrid_search(&self, query: &str, options: HybridSearchOptions) -> Result<Vec<MemoryResult>>;
}
```

**RRF 公式**: `score = Σ weight_i / (k + rank_i)`

### Task P1-5: 集成测试 — Smart + Ollama

**前提**: 本地运行 Ollama

- 真实 LLM 事实提取验证
- 真实向量搜索排序验证
- Smart 模式端到端（对话 → 事实 → 记忆 → 搜索）

---

## P2: 多语言绑定 ✅ 已完成

### Task P2-1: Python 绑定 (PyO3)

**目标**: `pip install memme`

```python
from memme import MemoryStore

store = MemoryStore("memory.duckdb")
store.add("I like coffee", user_id="alice")
results = store.search("beverages", user_id="alice")
```

**技术**: PyO3 + maturin 构建

### Task P2-2: memme-ffi (C ABI)

**目标**: 通过 C ABI 为 Swift/JS 等语言提供基础

**技术**: cbindgen 生成头文件

---

## P3: 移动端 + WASM ✅ 已完成

### Task P3-1: Swift 绑定 (UniFFI)

**目标**: iOS/macOS 原生集成

**关键**: 静态链接 DuckDB + MemMe-DB（iOS 不支持 dlopen）

```
编译流程:
  Cargo → build.rs → CMake 编译 DuckDB+MemMe-DB+DuckPGQ+FTS (静态)
  → 输出 .xcframework
```

### Task P3-2: JS/TS 绑定 (NAPI-RS)

**目标**: Node.js / Electron 集成

### Task P3-3: WASM 支持

**目标**: 浏览器内运行

---

## P4: 高级功能（部分完成）

### Task P4-1: Reranker ✅ 部分完成

- ~~LLM-as-reranker~~ ✅ 已实现
- Cohere reranker API — 未做
- HuggingFace cross-encoder (本地 ONNX) — 未做

### Task P4-2: Sleep-time Compute — 未做

借鉴 Letta/MemGPT：Agent 空闲时后台整理记忆（合并重复、提升重要度、衰减过期）

### Task P4-3: 记忆分析 (OLAP) ✅ 已完成

利用 DuckDB OLAP 能力：
- ~~记忆频率趋势~~ ✅ memory_frequency()
- ~~用户画像聚合~~ ✅ user_stats()
- ~~话题热度分析~~ ✅ top_entities()

### Task P4-4: 多 Agent 记忆隔离 — 未做

支持 per-agent 记忆空间（mem0 的已知痛点 [#4126](https://github.com/mem0ai/mem0/issues/4126)）

---

## P5: 多语言绑定完善 + 分析 + Reranker ✅ 已完成

| 任务 | 内容 |
|------|------|
| **Task P5-1: Node.js 绑定** | NAPI-RS 完整绑定（memme-node crate），支持 CRUD / search / smart / graph / hybrid / analytics |
| **Task P5-2: WASM 绑定** | wasm-bindgen 绑定（memme-wasm crate），浏览器端基础支持 |
| **Task P5-3: LLM Reranker** | rerank 模块 + LLM 重排序（search / hybrid_search 集成），4 个单元测试 |
| **Task P5-4: 记忆分析 (OLAP)** | analytics 模块：user_stats / memory_frequency / top_entities，6 个单元测试 |
| **Task P5-5: 测试扩充** | 单元测试从 74 → 133，新增 analytics / rerank / graph / memory / embeddings 测试 |

**已知限制**:
- Node.js API 全部同步阻塞，I/O 操作（add/search/add_smart/add_graph/hybrid_search）会阻塞事件循环
- WASM 需要 DuckDB-WASM 集成才能完整使用

---

## P6: 未来工作

| 优先级 | 任务 | 说明 |
|--------|------|------|
| 高 | **Node.js 异步 API** | 使用 NAPI-RS AsyncTask 将 I/O 方法改为返回 Promise |
| 中 | **更多 Reranker 后端** | Cohere API / HuggingFace cross-encoder / SentenceTransformer |
| 中 | **Async Rust API** | 核心 API 提供 async 版本，减少阻塞 |
| 低 | **Procedural Memory** | 程序性记忆：技能、习惯、工作流的学习与调用 |
| 低 | **Vision Message 支持** | 多模态消息记忆（图片内容理解与检索） |
| 低 | **Sleep-time Compute** | Agent 空闲时后台整理记忆（合并重复、提升重要度、衰减过期） |
| 低 | **多 Agent 记忆隔离** | per-agent 记忆空间 |

---

## 架构概览

```
┌─────────────────────────────────────────────────────┐
│                   Language Bindings                   │
│  Python (PyO3)  │  Swift (UniFFI)  │  JS (NAPI-RS)  │
├─────────────────────────────────────────────────────┤
│                                                       │
│              memme-core  (Rust 核心)                  │
│                                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌───────────────┐  │
│  │  MemoryStore │ │ GraphStore  │ │  SearchEngine  │  │
│  │  add()       │ │ add_entity()│ │  vector_search │  │
│  │  search()    │ │ add_rel()   │ │  fts_search    │  │
│  │  update()    │ │ traverse()  │ │  hybrid_rrf    │  │
│  │  delete()    │ │ search()    │ │  rerank        │  │
│  └──────┬──────┘ └──────┬──────┘ └───────┬───────┘  │
│         │               │                 │           │
│  ┌──────┴───────────────┴─────────────────┴───────┐  │
│  │         DuckDB + MemMe-DB Storage Layer       │  │
│  │  memories / entities / relationships / history  │  │
│  │  HNSW / FTS                          │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  ┌─────────────────┐  ┌──────────────────────────┐   │
│  │  memme-embeddings│  │  memme-llm               │   │
│  │  ├ OnnxEmbedder  │  │  ├ OllamaProvider        │   │
│  │  ├ OpenAIEmbedder│  │  ├ OpenAIProvider         │   │
│  │  └ OllamaEmbedder│  │  ├ AnthropicProvider      │   │
│  └─────────────────┘  │  ├ GeminiProvider         │   │
│                        │  └ NoopProvider           │   │
│                        └──────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 核心差异化（vs mem0）

| # | 差异点 | 说明 |
|---|--------|------|
| 1 | **单文件部署** | 一个 `.duckdb` 文件 = 向量 + 关系 + 图谱 + 历史 + 全文索引 |
| 2 | **Rust 核心 + 多语言绑定** | PyO3 / UniFFI / NAPI-RS，非 Python only |
| 3 | **可插拔 LLM** | 无 LLM（<10ms 向量去重）或接入任意 LLM（事实提取 + 智能去重） |
| 4 | **HNSW 过滤检索** | 分区向量索引，按 user_id 等标量列索引级过滤（非后过滤） |
| 5 | **DuckPGQ 图能力** | SQL/PGQ 标准语法 + PageRank/WCC/最短路径，零外部依赖 |
| 6 | **端侧原生** | 离线、低延迟、可嵌入移动端、单文件可迁移 |

---

## 目标市场优先级

1. **非 Python Agent 开发者**（Rust/Swift/JS/TS）— 市场空白最大
2. **具身智能 / 机器人**（2025 $4.4B → 2033 $67.6B）— 对端侧记忆有刚需
3. **隐私敏感场景**（医疗、金融、政府）— 数据不离设备
4. **移动端 / 桌面端应用**（iOS/Android/Electron）— mem0 无法原生嵌入

---

## 竞争窗口

- **Cognee-RS**（最大威胁）：已宣布 Rust 端侧版，种子轮资金拨款，但截至 2026.03 **无公开代码**
- **窗口期**: 6-12 个月
- **策略**: 快速推出 P1 完成对标 mem0 核心功能，建立先发优势

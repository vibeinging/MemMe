# MemMe V3 实现文档

> Date: 2026-04-13
> Status: Current (基于 commit efafff6)
> 说明：本文档基于实际代码阅读，描述 V3 架构的**现实现状态**，而非设计意图。

---

## 1. 核心设计原则

1. **Store-First**：事件通过 `append_events` 写入后立即可搜索，无需等待 compact
2. **多通道融合**：5 个弱信号通过 RRF 融合，比单一强信号更鲁棒
3. **非阻塞 LLM**：后台处理队列，append/search 不等待 LLM 返回
4. **本地优先**：所有检索离线完成，LLM 仅用于内容处理（compact/meditate）

---

## 2. 数据流总览

```
append_events()
│
├─ Embed batch（1 次 API 调用，N 条消息）
├─ 插入 events + vec0 索引（立即可搜索）
├─ 构建共现图（零 LLM，Aho-Corasick）
└─ [可选] 触发后台处理

compact(session_id)                         ← 1 次 LLM 调用
│
├─ 获取 unprocessed events
├─ LLM：净化 + 摘要 + 前瞻查询
├─ 创建 Episode
├─ 插入叙述 Trace（不可变，Resolution::Narrative）
└─ 标记事件为 processed

meditate(user_id)                           ← N 次 LLM 调用（每 Episode 1 次）
│
├─ 遗忘曲线衰减
├─ 循环（每批 20 个 Episode）：
│   ├─ LLM：提取事实 → Vec<ExtractedFact>
│   ├─ add_batch（含 embedding + dedup）
│   ├─ 图提取 + 实体链接
│   └─ 标记 Episode 已冥想
└─ 保存 MeditationRecord

search(query, user_id, limit)               ← 零 LLM（可选 rerank API）
│
├─ 5 通道并行检索
├─ RRF 融合
├─ [可选] Cross-encoder rerank
├─ [可选] 遗忘曲线评分
└─ 图增强（post-recall，补充相关记忆）
```

---

## 3. 搜索管道详解

### 3.1 五通道 RRF 融合

**文件：** `memory/search.rs:168-531`

| 通道 | 技术 | 默认权重 |
|------|------|---------|
| 1. 向量语义 | vec0 HNSW KNN | 0.30 |
| 2. BM25 全文 | FTS5（含实体扩展查询） | 0.35 |
| 3. 实体传播 | Aho-Corasick + 图遍历 + 向量排序 | 0.20 |
| 4. 词重叠 | Token overlap score（补充 tiebreaker） | 0.15 |
| 5. 时间通道 | 时间意图检测 + 过滤 + 权重提升 | 0.15 |

**RRF 公式**（`search.rs:58-86`）：
```
score(doc) = Σ  weight_i / (k + rank_i)
             i
其中 k=30，权重可通过 TuningConfig 调节
```

每个通道检索 `limit × candidate_multiplier` 个候选（默认 limit=10，multiplier=3），融合后截断到 limit。

**自适应加权**（`adaptive_rrf_alpha`，默认 0 = 禁用）：根据各通道分数分布（均值/标准差）动态调整权重，高置信度通道获得更高权重。

### 3.2 实体通道（Channel 3）

```
query → Aho-Corasick 提取实体 ["coffee", "Tuesday"]
     → 图遍历 1-hop → 扩展实体 ["espresso", "caffeine", "latte"]
     → 搜索链接到这些实体的记忆
     → 按向量相似度排序（实体链接的记忆可能语义较远）
```

BM25 通道同时使用图扩展实体作为 OR 条件（最多 10 个），提高 FTS 召回率。

### 3.3 时间感知

**检测时机**（`memory/search.rs:64-159`）：
- 时间解析器（`time_parser.rs`）优先：能提取具体时间范围
- 启发式 fallback：when/yesterday/上周/去年 等关键词（支持中英文）

**处理方式**（非破坏性）：
- 有时间戳且在范围内 → 分数 ×1.5
- 无时间戳的记忆始终保留（防止数据丢失）
- 有时间戳但范围外 → 正常参与排序（不剔除）

### 3.4 Rerank（可选）

- **API rerank**（Jina / Cohere / DashScope）：候选数 × `rerank_candidate_multiplier`（默认 3），调用远程 cross-encoder 重排
- **ONNX rerank**（本地）：feature flag `onnx-rerank`，离线执行
- 未配置时跳过，直接返回 RRF 结果

### 3.5 遗忘曲线评分

**公式**（幂律，非 Ebbinghaus 指数）：
```
retention(t, S) = (t / (5 × S) + 1) ^ -0.5
score = similarity × (w × retention + (1-w) × importance)
```
其中 `w = retention_weight`（默认 0.7），S = stability（随访问增长）。

**稳定性增长**：每次检索命中时 `S' = S × (1 + 2.5 × (1 - R))`，已遗忘的记忆每次访问获得更大增强。

**disabled for benchmark**：benchmark 中 `enable_forgetting_curve=False`，所有记忆平等参与排序。

### 3.6 图增强（Post-recall）

检索结果返回后，从这些结果提取实体，查找关联但未出现在结果中的记忆（最多 `graph_augmentation_limit` 个，默认 10），追加到结果列表末尾。0 = 禁用。

---

## 4. 写入流程

### 4.1 append_events（`memory/compact_ops.rs:26-128`）

```
① embed_batch(messages)         → 1 次 API 调用
② insert_event(×N)              → events 表 + vec0 HNSW 索引
③ append_structured_note()      → 追加到 session.notes
④ build_cooccurrence_graph()    → 共现实体图（零 LLM）
⑤ 检查 compact_needed           → 纯信息性，不自动触发
⑥ process_background()          → 处理 1 个后台任务（若有）
```

返回 `AppendEventsResult { events_appended, total_unprocessed, compact_needed }`。
`compact_needed` 仅为建议，客户端自行决定是否调 `compact()`。

### 4.2 compact（`memory/compact_ops.rs:143-403`）

**令牌估计 → 选择路径：**
- `estimated_tokens < compact_fallback_token_threshold`（默认 200）→ 跳过 LLM，用规则生成标题/摘要
- 否则 → 单一 LLM 调用，同时执行：净化 + 标题 + 摘要 + 显著性 + 前瞻查询

**前瞻查询**（Prospective Indexing）：LLM 生成用户未来可能问的 3-5 个问题，追加到 `narrative_content`，一并嵌入。目的是在 BM25 和向量通道中为未来查询预置索引。

**输出：**
- `Episode`：标题 + 摘要 + 时间范围 + 显著性
- `Narrative Trace`（不可变记忆，`Resolution::Narrative`）：episode 摘要的向量化形式

### 4.3 meditate（`memory/meditation_ops.rs`）

**触发条件：** 冷却期 ≥ 1 小时（默认），且有未冥想的 Episode。

**循环处理（每批 20 个 Episode）：**
1. `extract_facts_from_episode()` → LLM 提取原子事实
2. `add_batch(facts)` → embed + dedup（阈值 0.15 余弦距离） + 插入
3. `process_graph_batch()` → LLM 提取实体关系三元组 + 实体链接
4. `mark_episode_meditated()` → 标记 Episode

**dedup 策略：**
- 计算新事实 embedding 与已有记忆的余弦距离
- 距离 ≤ 0.15 → 视为重复，跳过或更新
- 后批次可见前批次新增的记忆，避免跨 Episode 重复

**每 Episode 超时 180 秒**，防止 LLM 卡住阻塞整个冥想。

---

## 5. 记忆类型

| Resolution | 来源 | 搜索权重 | 特征 |
|-----------|------|---------|------|
| `Granular` | meditate 事实提取 | 1.0 | 原子事实，可被更新/删除 |
| `Narrative` | compact episode 摘要 | 0.65 | 不可变（immutable=true） |
| `Identity` | identity_ops 个性特征 | 0.70 | 长期稳定的自我描述 |

---

## 6. 配置关键字段

```toml
# 检索
rrf_vector_weight = 0.30
rrf_fts_weight = 0.35
rrf_entity_weight = 0.20
rrf_word_overlap_weight = 0.15
rrf_temporal_weight = 0.15
rrf_k = 30
rrf_candidate_multiplier = 3    # 候选数 = limit × multiplier
graph_augmentation_limit = 10   # post-recall 图增强，0=禁用

# 遗忘曲线
enable_forgetting_curve = true
retention_weight = 0.70
stability_growth_factor = 2.5
prune_retention_threshold = 0.05

# LLM
llm_max_tokens = 2048           # reasoning 模型需调到 8000+
llm_temperature = 0.1
compact_fallback_token_threshold = 200

# Meditate
meditation_batch_size = 20
meditation_min_significance = 0.3
meditation_cooldown_hours = 1

# Dedup
dedup_threshold = 0.15          # 余弦距离阈值
```

---

## 7. Python API 速查

```python
store = memme.MemoryStore(
    db_path="mydb.sqlite",
    embedder="openai",
    api_key="...",
    embed_model="text-embedding-3-small",
    dims=1536,
    llm_api_key="...",
    llm_model="gpt-4o-mini",
    llm_base_url="https://api.openai.com/v1/chat/completions",
    rerank_api_key="...",           # 可选
    rerank_base_url="...",          # 可选
    rerank_model="gte-rerank-v2",   # 可选
    enable_forgetting_curve=False,
)

# 写入
result = store.append_events(
    session_id="session_001",
    messages=[("user", "text"), ("assistant", "response")],
    user_id="alice",
)
if result["compact_needed"]:
    store.compact("session_001")

# 冥想（后台）
store.meditate(user_id="alice", triggered_by="benchmark")

# 搜索
results = store.search(query="what did I drink last week", user_id="alice", limit=10)

# FTS 索引重建（ingest 完成后）
store.rebuild_fts_index()
```

---

## 8. Benchmark 配置（LoCoMo）

当前 benchmark (`run_benchmark_engine.py`) 使用的参数：

```
embed_model = text-embedding-3-small (1536d)
engine_llm_model = gpt-4o-mini
chat_model = gpt-4o-mini (answer)
judge_model = gpt-4o-mini (judge)
top_k = 30
rrf_vector_weight = 0.5 (benchmark 覆盖默认值)
rrf_fts_weight = 0.3
rrf_entity_weight = 0.2
rrf_temporal_weight = 0.15
enable_forgetting_curve = False
rerank = gte-rerank-v2 (DashScope)
```

**历史得分（LoCoMo 10 对话）：**

| 版本 | Judge (含 AD) | Judge (排除 AD) | 说明 |
|------|-------------|----------------|------|
| v20 (2026-04-06) | 72.16 | — | SQLite 迁移后，DuckDB→SQLite |
| V3 arch (2026-04-13) | 75.53 | **85.52** | 5 通道 RRF + rerank (10 conv, 1986 Q) |

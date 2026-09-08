# PetMemBench

PetMemBench 是 MemMe 面向 AI 宠物场景的离线基准。它不依赖云端 LLM，先回答三个基础问题：

1. 关键记忆能不能正确找回，是否会串用户、串宠物或读到过期内容。
2. 刚发生的事件能不能立刻被找回，更新和删除是否真的生效。
3. 在固定数据量下，写入和查询的延迟、文件大小、向量计算次数是多少。

这里使用确定性的词语/字片段向量，只用于稳定测量 MemMe 引擎和 VexDB-Lite SQLite 存储。它不能代替真实 embedding 模型的语义质量测试。

## 运行

```bash
MEMME_VEXDB_LITE_EXTENSION="$(bash scripts/download-vexdb-lite-extension.sh)" \
  cargo run --release -p memme-core --example pet_memory_benchmark -- \
  --dataset benchmarks/petmem/scenarios.json \
  --output benchmarks/petmem/results/latest.json \
  --memory-count 2000 \
  --query-count 100
```

也可以用 `--extension /absolute/path/to/vexdb_lite.dylib` 显式指定已验证的扩展。

退出码规则：硬门槛全部通过时为 0；任何硬门槛失败时为 2。`required: false` 的目标场景失败会写入结果，但不会改变退出码。

## 指标

- `required_scenarios`：当前版本必须满足的正确性和安全要求。
- `target_scenarios`：下一版架构要解决的问题。
- `lifecycle_checks`：不可变记忆、更新、删除、精确去重和冲突事实处理。
- `performance`：批量写入耗时、查询 p50/p95/p99、Recall@K、数据库文件大小。
- `embedding_counts`：单条和批量 embedding 调用次数，用来估算本地算力或云端费用。

结果只适合在相同机器、相同编译模式、相同数据量和相同 VexDB-Lite 版本之间比较。

## 宠物范围约定

- 全局用户记忆：写入时不设置 `agent_id`。
- 宠物关系记忆：通过 `AddOptions::agent_id()` 设置宠物范围。
- 原始会话事件：在 `append_events()` 的 metadata 中写入保留字段 `agent_id`，例如 `{"agent_id":"momo"}`。

指定 `SearchOptions::agent_id()` 后，搜索会合并全局用户记忆与当前宠物的关系记忆和即时事件，同时排除其他宠物的数据。

最新 Phase 1 结果与架构结论见 `docs/reports/2026-09-01_petmembench-phase-1.md`。

# AI 宠物 REST 接入优化报告

日期：2026-09-01  
范围：MemMe 核心事件层、`memme-server`、OpenAPI 和接入文档  
工作区：有其他未提交修改，本轮没有提交或推送

## 结论

MemMe 侧已经补齐小智类语音玩偶最重要的数据边界：对话事件先可靠落盘，重试不会
重复写入，同一个 session 不能跨主人、宠物、应用或连接复用；召回可以按这些范围
过滤。REST 现在也能完整导出、导入、删除一个用户的全部数据，并能受控恢复 SQLite
备份。

小智仓库本轮没有修改。它仍要把稳定的 `user_id`、`agent_id` 和 `event_id` 传给
这些接口，并在断线时等待写入完成或使用本地重试队列。

## 2026-09-02 数据安全二次优化

### 1. session 不再串用户或宠物

session 第一次创建后会固定绑定 `user_id`、`agent_id`、`app_id`、`run_id` 和
`source_id`。后续请求只要有一个值不同或缺失，就会在写事件前返回错误。校验同时
覆盖先读后写和并发创建后复查，避免两个请求同时创建同一 session 时绕过边界。

### 2. 完整数据迁移

新增：

- `POST /v1/data/export`
- `POST /v1/data/import`
- `DELETE /v1/users/{user_id}`

完整导出包含记忆、session、事件、episode、图谱、身份和 source，也包含
`local_only` 记忆。完整导入会恢复事件范围、source 所有者、隐私级别和稳定度，
并重新计算向量、重建 FTS；导入后不需要等下一次整理才能检索。整用户删除必须在
请求体中精确重复 `user_id`，删除范围覆盖全部数据层及派生索引。

旧的 `/v1/memories/export` 和 `/v1/memories/import` 保留为只处理 memory trace 的
小接口，OpenAPI 已改正，不再把它们描述成完整导出。

### 3. 受控备份恢复

新增 `POST /v1/backups/restore`：

- 只接受 `MEMME_BACKUP_DIR` 下的普通 `.db` 文件名。
- 文件本身必须是普通文件，目录和符号链接会被拒绝。
- 确认字段必须等于 `restore:<filename>`。
- 接受前执行 SQLite `integrity_check`，并确认关键 MemMe 表存在。
- 返回 `202` 后停止接收新请求，等待当前请求结束，关闭旧连接，原子恢复数据库，
  再在原地址自动启动服务。

真实 REST 流程已验证：备份前写入事件 A，备份后写入事件 B，发起恢复后服务自动
重启；再次完整导出只剩事件 A。

## 完成的优化

### 1. 幂等事件写入

新增 `MemoryStore::append_events_idempotent()` 和 `POST /v1/events`。

- 调用方给每条消息提供稳定的 `event_id`。
- 同一 ID、同一内容重试时不再计算向量，也不重复写入。
- 同一 ID 携带不同内容、身份、时间或 metadata 时返回错误。
- 返回值增加 `events_replayed`，接入方可以分清新写入和重试命中。

### 2. 宠物范围完整进入事件检索

`events` 表增加 `app_id` 和 `run_id`，旧数据从 JSON metadata 回填，并增加范围索引。
向量和 FTS 事件召回现在都支持：

- `user_id`
- `agent_id`
- `app_id`
- `run_id`

`/v1/recall` 也补齐相同字段。宠物产品应始终传 `agent_id`，否则查询含义是主人级
管理查询，会看到这个主人的全部宠物记忆。

### 3. 生命周期和数据安全接口

新增：

- `POST /v1/sessions/{session_id}/compact`
- `POST /v1/meditations`
- `POST /v1/backups`
- `GET /v1/replica/status`
- `POST /v1/replica/sync`

备份接口只能在 `MEMME_BACKUP_DIR` 下创建普通 `.db` 文件，不能传任意服务器路径。
耗时的诊断、compact、meditate、备份和副本操作放到阻塞线程执行，不会占住 Tokio
的异步工作线程。

### 4. 本地模型和 Key 分离

服务默认使用本地 `bge-small-zh-v1.5` ONNX 模型，首次下载约 90.46 MiB，不需要
云端 Key。多语言数据可以选择 `multilingual-e5-small`，但它的 ONNX 文件约
448.48 MiB，不适合低存储设备作为默认值。

远程模型改成独立配置：

- `EMBEDDING_API_KEY` 只用于向量服务。
- `LLM_API_KEY` 只用于 LLM。
- `MEMME_API_KEY` 只用于 REST 鉴权。

LLM Key 不再写入 SQLite。打开旧数据库时会删除旧版本留下的 `llm_api_key` 配置项。
模型名和 endpoint 可以保留，重启后仍需通过环境变量提供 Key。

### 5. 文档和测试

- OpenAPI 删除不存在的 smart 和 feedback 路由，并补齐当前路由。
- 增加 OpenAPI 路由清单测试。
- 增加 REST 端到端测试：幂等重试、两只宠物隔离、范围召回。
- 增加小智接入约定：`docs/design/2026-09-01_xiaozhi-rest-integration.md`。
- README 增加可以直接复制的 REST 启动和事件写入示例。

## 验证结果

### 自动测试

- `cargo fmt --all -- --check`：通过。
- `cargo clippy -p memme-core -p memme-embeddings -p memme-llm -p memme-server --all-targets -- -D warnings`：通过。
- `cargo test -p memme-core -p memme-server` 和新增聚焦测试：
  - 核心单元测试 354 / 354 通过。
  - edge cases 24 / 24 通过。
  - integration 6 / 6 通过。
  - mobile scenarios 19 通过、1 个原有 ignored。
  - stress 10 / 10 通过。
  - VexDB-Lite 11 / 11 通过。
  - REST 5 / 5 通过。
  - doc tests 3 / 3 通过。

### PetMemBench 回归

最新结果：`benchmarks/petmem/results/2026-09-01_data-safety-optimization.json`

| 指标 | 结果 |
|---|---:|
| 必须场景 | 11 / 11 |
| 目标场景 | 3 / 3 |
| Recall@10 | 100% |
| 数据量 | 2,000 条 |
| 查询 p50 | 1.875 ms |
| 查询 p95 | 2.228 ms |
| 查询 p99 | 14.789 ms |
| 写入速度 | 356.1 条/秒 |
| SQLite 文件大小 | 10,412,032 bytes |

这个基准使用确定性的本地测试向量，用来确认引擎和 VexDB-Lite 没有回退，不代表
真实模型的最终语义质量。

目标 Linux 的 2 CPU / 4 GB 容器验收本轮未生成结果。本机 Docker Desktop 在启动后
60 秒内没有提供可用 daemon，已退出，没有把 Mac 结果写成 Linux 结果。正式部署前
仍需在目标服务器运行同一条 PetMemBench 命令，并补启动时间和 RSS。

### 真实本地 ONNX 冒烟测试

默认 `bge-small-zh-v1.5` 完成首次下载并启动服务。缓存模型后的单次本机 HTTP
样本：

- 写入中文事件：26.8 ms。
- 使用不同说法召回中文事件：8.6 ms。
- 返回内容正确，范围为指定 `owner-onnx / pet-a / xiaozhi`。

这是一次冒烟样本，不是稳定性能统计。

## 仍在 MemMe 之外的工作

小智适配器还需要：

1. 不再用设备 MAC 直接代替主人身份。
2. 写入时生成稳定 `event_id`，断线重试时保持不变。
3. 查询时传 `user_id + agent_id`，长期召回不要传当前 `run_id`。
4. 断线时等待写入成功，或先进入本地持久队列。

真实 LLM 的 compact/meditate 本轮没有调用，因为没有使用用户的外部模型凭据；接口、
错误返回和无 LLM 降级路径已经覆盖。

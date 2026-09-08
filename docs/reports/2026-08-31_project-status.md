# MemMe 项目现状检查

检查日期：2026-08-31

## 结论

MemMe 已经不是原型：核心存储、检索、图谱、会话/情景、同步备份、多个绑定层和服务端入口都已落地，核心测试与绑定构建通过。

但当前工作区不适合直接发布。最直接的原因是项目自己的格式检查和 Clippy 门禁失败；此外，OpenAPI、README/架构说明与实际代码已有明显漂移。建议先完成一次小范围的“主分支恢复健康”整理，再继续增加能力。

## 当前项目轮廓

- 主分支：`main`，当前提交 `991139e`（2026-04-20）。本地分支未显示已配置的上游跟踪分支。
- Rust：edition 2021，MSRV 1.77，本地版本号 0.1.1。
- 主 workspace：9 个包：`memme-core`、`memme-embeddings`、`memme-llm`、`memme-ffi`、`memme-python`、`memme-wasm`、`memme-node`、`memme-server`、`memme-mcp`。
- `crates/` 下共有 11 个 Cargo 包；`memme-dora` 与 `cu-memme` 没有加入主 workspace。
- Rust 代码约 122 个文件、37,748 行，约 510 个 `#[test]` / `#[tokio::test]` 标记。
- 核心采用 SQLite 单文件存储，包含向量表、FTS5、记忆、事件、会话、情景、身份、图谱、历史、同步、备份等数据结构。
- 对外入口包括 Rust API、Python、Node.js、UniFFI、WASM、REST 服务和 MCP 服务。

## 本次验证结果

| 检查 | 结果 | 说明 |
|---|---|---|
| `cargo test -p memme-core` | 通过 | 397 项通过，24 项忽略；忽略项主要是真实 LLM/Embedding 测试 |
| `cargo test -p memme-embeddings -p memme-llm` | 通过 | 58 项通过 |
| 绑定与服务构建 | 通过 | `memme-ffi`、`memme-node`、`memme-wasm`、`memme-server`、`memme-mcp` 均构建成功 |
| `cargo fmt --all -- --check` | 失败 | `memme-core` 多个文件未按当前 rustfmt 格式化 |
| CI 同款 Clippy 命令 | 失败 | 5 类错误：未使用变量、未使用枚举分支、未使用方法、参数过多等 |

测试通过说明主要功能路径没有发现直接回归；但真实外部 LLM、真实模型下载/推理、移动设备产物和发布流程本次没有做线上或真机验收。

## 主要问题

### P0：当前 CI 门禁会失败

`.github/workflows/ci.yml` 会先执行格式检查和 `-D warnings` Clippy。当前本地执行相同命令时：

- 格式检查在 `contradiction.rs`、`compact_ops.rs`、`helpers.rs`、`lifecycle.rs`、`search.rs`、`storage/mod.rs`、`storage/query.rs`、`storage/stream.rs` 等文件发现差异。
- Clippy 在 `storage/query.rs`、`memory/background.rs`、`storage/session.rs` 等位置报错。
- 普通构建仍会产生 `memme-core` 和 `memme-ffi` 警告。

这表示“能编译、测试能过”，但还没有达到仓库自己定义的合入标准。

### P1：API 文档与实际路由不一致

`docs/openapi.yaml` 仍记录以下接口，但当前 `memme-server` 路由没有注册：

- `/v1/memories/smart`
- `/v1/memories/smart/messages`
- `/v1/recall/{recall_id}/feedback`

反过来，代码已有 `/diagnose`，OpenAPI 中没有。客户端若按 OpenAPI 生成代码，会遇到真实接口不存在或遗漏的问题。

### P1：检索架构说明与当前实现不一致

当前 `search()` 的 RRF 输入实际是：向量、BM25、实体、词面重合四路；时间信息作为可解析时间范围的结果过滤，不是独立 RRF 通道。

配置中仍保留 `rrf_temporal_weight`，但当前检索实现没有读取它；存储层的 `temporal_range_search()` 也没有接入主搜索流程。README、架构说明中“向量 + BM25 + 实体 + 时序四通道”的说法需要改成当前事实，或把时序通道重新接回实现。

### P1：工作区存在较大的未提交变化

当前有 22 个已跟踪的 benchmark 文件被删除，合计约 70,504 行，主要来自 `benchmarks/locomo` 与 `benchmarks/locomo-plus`；另有未跟踪的 `AGENTS.md` 和 `references/`。

这些变化可能是有意做 benchmark 清理和竞品代码归档，也可能是尚未整理完。正式发布或合并前应先确认边界。本次检查没有恢复、删除、暂存或提交这些内容。

### P2：仓库结构有遗留内容

- 根目录 `src/lib.rs` 仍是默认的 `add(2, 2)` 模板，且根 `Cargo.toml` 只有 workspace、没有根 package，因此该文件不会参与构建。
- `memme-dora` 与 `cu-memme` 使用 workspace/path 依赖但没有列入 workspace；它们也不在主 CI 的构建矩阵中。
- `CHANGELOG.md` 的大量 0.1.1 之后变化仍放在 `Unreleased`，而仓库已经存在 `v0.1.1` 标签，版本记录边界不清楚。

### P2：性能与基准结论本次无法复现

README 中的 `<10ms` 和 LoCoMo 82.92% 属于历史结果。本次没有重新跑 benchmark，而且相关 benchmark 文件当前正处于删除状态，因此不能把这些数字当作当前提交的重新验证结果。

## 做得好的部分

- SQLite、VexDB-Lite、FTS5、图谱和备份都收在本地单文件模型里，产品定位与代码主体一致。
- 核心测试覆盖 CRUD、隔离、SQL 注入、不可变记忆、同步、备份、图谱、时间解析、移动电量策略和压力场景。
- 外部 LLM 是可选边界，Mock 路径可让核心离线测试稳定运行。
- CI 已覆盖 Ubuntu/macOS、格式、Clippy、核心/边界/移动/压力测试和主要绑定构建。
- `.env`、模型缓存、日志和 `target/` 都已被忽略；本次没有读取 `.env` 内容。

## 建议顺序

1. 先只修格式和 Clippy，让 `main` 恢复为全绿状态。
2. 明确 temporal channel 的产品决定：接回独立检索通道，或删除未使用配置并改正文档。
3. 以 `memme-server` 当前路由为准更新 OpenAPI，并加一项路由与 OpenAPI 一致性测试。
4. 确认 benchmark 删除、`references/`、Dora/Copper 两个包的归属，再整理 workspace 与 CI。
5. 重新跑可复现 benchmark 后再更新 README 的性能和准确率数字。

## 检查边界

本次没有修改业务代码，没有运行真实付费 LLM 请求，没有真机验证移动产物，没有发布、提交或推送。唯一新增内容是本报告。

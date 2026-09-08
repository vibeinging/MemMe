# 公开入口修复实施报告（2026-09-08）

对应计划：`docs/plans/2026-04-01_github-10k-star-promotion-plan.md`（2026-09-08 修订版）
代码基点：`991139e` + 工作区未提交修改

## 完成事项

### P0：把正确的产品交到公开入口

- **CI 失败原因**：远端 main 最近一次失败（2026-05-10，`991139e`）只有 `cargo fmt --check` 和 `clippy -D warnings` 两个门禁，测试与构建均为绿色。当前工作区已修复并通过全部本地同款门禁（见"验证"）。
- **提交整理**：工作区改动整理为两次提交（0.1.2 源码补齐 + 公开入口修复），排除内部目录（`references/` 第三方代码、`.gstack/`、`deploy/`、`AGENTS.md`），并加入 `.gitignore`。提交前扫描未发现密钥或个人路径泄漏。
- **npm 0.1.2 溯源**：`v0.1.2` tag 指向源码补齐提交，GitHub Release 附带发布报告链接与产物 SHA-256。npm 发布工作流增加"已发布则跳过"守卫，重复推送 tag 不会因重复发布而变红。
- **crates.io / PyPI**：发布工作流改为手动触发（凭据失效期间本就需要手动、按依赖顺序发布）；Python README 明确 `pip install memme` 当前安装到的仍是 0.1.1 旧版（DuckDB），当前版本需从源码构建。
- **官网 404 根因**：GitHub Pages 从未配置（Pages API 404），不是部署损坏。新增 `pages.yml` 工作流从 `docs/` 部署，并通过 API 以 workflow 模式启用 Pages。

### P0：让第一次试用不依赖猜测

- 新增 `demos/rest-demo.sh`：一条完整首选路径。脚本检查系统与架构、下载并校验 VexDB-Lite 扩展与 ONNX Runtime、构建 REST 服务，然后执行 写入 → 检索 → 完整退出进程 → 重开 → 再检索，并验证双宠物隔离。
- 新增 `scripts/download-onnx-runtime.sh`：固定 ONNX Runtime 1.19.2（四个平台 SHA-256 校验、私有缓存目录）。这解决了实测发现的真实阻塞——本地 ONNX embedding 依赖 `ORT_DYLIB_PATH`，此前仓库没有任何获取方式，新用户会在第一步 panic。
- 两个 README 增加首屏"在本地试用"入口（标明系统要求、无 API Key、首启模型下载说明）。
- 旧 `demos/playground` 标记为 legacy 并指向新路径。

### 过程中发现并修复的产品 bug

**agent 检索丢失主人全局事件**（`crates/memme-core/src/storage/query.rs`）：

- `vector_search_index` 的 memories 分支正确使用 `agent_id = X OR agent_id IS NULL`，但 events 分支只过滤 `agent_id = X`，导致不带 agent 的主人全局事件在宠物检索中永远不可见。
- `fts_search` 存在同样的不对称。
- 两处已修复并对齐语义，新增回归测试 `test_agent_search_includes_owner_global_events`。
- 修复前 `rest-demo.sh` 实测 FAIL（过敏记忆检索不到），修复后 PASS。

### P1：首屏展示行为和适用对象

- `docs/index.html` 重构：AI 宠物定位首屏（一句话 + 试用入口 + 真实数据统计）、新增"本地试用"与"三个可验证能力"区块、各绑定标注真实状态（已发布/源码构建/接线中）、LoCoMo 竞品对比替换为 PetMemBench 结果与限制说明、License 由错误的 MIT 改为 Apache-2.0、Node 示例包名改正。
- `llms.txt` / `llms-full.txt` 同步：移除 LoCoMo/mem0/Zep 对比说法与过时 API（`add_smart_messages`）、补齐 VexDB-Lite/ONNX Runtime 运行时要求、绑定状态如实标注。
- GitHub About / Topics / homepage 通过 API 更新为 AI 宠物 + SQLite 定位（见"远端操作"）。

## 验证（2026-09-08 本机，x86_64-apple-darwin）

- `cargo fmt --all -- --check`：通过。
- `cargo clippy -p memme-core -p memme-embeddings -p memme-llm -- -D warnings`：通过。
- `cargo test -p memme-core`：372 通过、0 失败（1 项设备场景与 23 项真实 LLM 测试按设计忽略）；embeddings 9 项、llm 49 项通过。
- `demos/rest-demo.sh` 全新运行：7/7 检查 PASS（写入、Momo 检索、不泄漏到 Luna、Luna 检索、主人全局记忆对宠物可见、重启后记忆保持、重启后隔离保持）。
- 首次运行成本实测：server release 构建 2m34s（后续增量 0.5s）；VexDB-Lite 扩展与 ONNX Runtime 一次性下载并缓存；首次检索触发 embedding 模型下载。

## 远端操作

- 推送 main（两次提交）与 `v0.1.2` tag；创建 GitHub Release `v0.1.2`（注明 npm 0.1.2 溯源与 PyPI/crates 现状）。
- 更新仓库 About 描述、Topics（`ai-memory` `long-term-memory` `ai-companion` `ai-pets` `local-first` `sqlite` `rust`）、homepage 指向 Pages。
- 启用 GitHub Pages（workflow 模式）并验证官网返回 200。

## 未完成 / 后续

- PyPI 与 crates.io 的 0.1.2 发布：等待凭据刷新后手动触发工作流（顺序：embeddings → llm → core）。
- 30–60 秒真实演示视频/GIF：脚本与场景已就绪（rest-demo 的四幕结构），需要录制后放入首屏。
- 外部试用者（计划第 2 周）：需要在干净环境重放 `demos/rest-demo.sh` 并记录结果。
- 计划中的社区分享内容未发送，本次没有任何对外发帖或邀请。
- Windows 平台：VexDB-Lite v0.0.17 无对应扩展，快速开始已如实标注不支持。

# 公开入口修复实施报告（2026-09-08）

> 最新本机验收结果见末尾“复查问题修复与验收”。此前的实施和复查记录保留为历史；远端验证与部署状态以对应提交的 CI / Pages 结果为准。

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

## 验证（2026-09-08 本机，x86_64-apple-darwin，rustc 1.93.1）

- `cargo fmt --all -- --check`：通过。
- `cargo clippy -p memme-core -p memme-embeddings -p memme-llm -- -D warnings`：本机通过。
- `cargo test -p memme-core`：372 通过、0 失败（1 项设备场景与 23 项真实 LLM 测试按设计忽略）；embeddings 9 项、llm 49 项通过。
- `demos/rest-demo.sh` 全新运行：7/7 检查 PASS（写入、Momo 检索、不泄漏到 Luna、Luna 检索、主人全局记忆对宠物可见、重启后记忆保持、重启后隔离保持）。
- 首次运行成本实测：server release 构建 2m34s（后续增量 0.5s）；VexDB-Lite 扩展与 ONNX Runtime 一次性下载并缓存；首次检索触发 embedding 模型下载。

### CI 迭代记录（如实）

- 第一次推送（`c2cf6d3`）后远端 CI 失败：Format 与 Clippy 两个门禁。原因有二：
  1. 初次验证脚本用管道接 `tail`/`grep`，`$?` 取到的是管道末命令的退出码，掩盖了本机 fmt 的实际 diff（query.rs 缩进）——验证脚本本身的错误，已改为直接检查退出码。
  2. 远端 clippy 为 1.98.0，新 lint `unnecessary_sort_by` 报在既有代码 `search.rs:42`；本机 1.93.1 无此 lint。
- 修复后第二次推送（`93dbb9b`）远端 CI 全绿（含两个平台的全部测试矩阵与绑定构建）。教训：本机工具链落后于 CI 时，fmt/clippy 结论以远端为准。

## 远端操作（均已执行）

- 推送 main：`fc7c54c`（0.1.2 源码补齐）、`c2cf6d3`（入口修复）、`93dbb9b`（fmt/clippy 修复）；推送 `v0.1.2` tag。
- GitHub Release `v0.1.2` 已创建：[releases/tag/v0.1.2](https://github.com/vibeinging/MemMe/releases/tag/v0.1.2)，注明 npm 0.1.2 溯源、PyPI/crates 现状与已知限制。
- 仓库 About、Topics（`ai-memory` `long-term-memory` `ai-companion` `ai-pets` `local-first` `sqlite` `rust`，已移除 `duckdb`）、homepage 已更新。
- GitHub Pages 以 workflow 模式启用；`pages.yml` 首次运行因 Pages 尚未启用而失败，重新运行后部署成功；官网返回 200，页面关键内容（包名、License、PetMemBench、无竞品对比）已逐项核对。
- 远端 CI 在 `93dbb9b` 全绿。
- `v0.1.2` tag 只触发了 npm 发布工作流（crates/PyPI 已改为手动）；npm 工作流对五个已在 registry 的 0.1.2 包走跳过守卫。

## 遗留观察

- **macos-13 runner 已退役**：`v0.1.2` tag 触发的 npm 工作流中，`x86_64-apple-darwin` 构建在 macos-13 上排队 50+ 分钟无 runner 接单（同时 macos-14 秒级开跑）。已把该构建迁到 macos-14 交叉编译（`7a1ddab`），并为 publish-node 增加手动触发（可对既有 tag 重跑）。原死锁 run 已取消，v0.1.2 的发布工作流改由手动触发完成，publish 阶段按守卫跳过已发布的 0.1.2。
- 历史 dependabot 依赖升级 PR 的 CI 为红色（早于本次改动）。不影响 main 分支 CI，但会在 PR 列表形成负面观感，建议下一轮集中处理或关闭过期 PR。
- 本机工具链（1.93.1）落后于 CI（1.98.0）：本机 fmt/clippy 通过不能替代远端结论，建议升级本机 stable 或以远端为准。

## 未完成 / 后续

- PyPI 与 crates.io 的 0.1.2 发布：等待凭据刷新后手动触发工作流（顺序：embeddings → llm → core）。
- 30–60 秒真实演示视频/GIF：脚本与场景已就绪（rest-demo 的四幕结构），需要录制后放入首屏。
- 外部试用者（计划第 2 周）：需要在干净环境重放 `demos/rest-demo.sh` 并记录结果。
- 计划中的社区分享内容未发送，本次没有任何对外发帖或邀请。
- Windows 平台：VexDB-Lite v0.0.17 无对应扩展，快速开始已如实标注不支持。

---

## 独立复查（2026-09-08，当前窗口）

复查基点：`90d4418fa80d2582b7c8ed747ea7447aa5b2f05d`。开始复查时工作区干净。
本节补充原实施记录，区分已经上线的改动与新用户试用验收；不覆盖上方历史记录。

**结论：公开入口修复已有实际成果，但首选路径的首次模型下载仍会失败，暂不能按“第 1 周全部验收完成”处理。**
以下问题均未在本轮复查中修复；本轮只更新这份报告。

### 已独立确认

- GitHub About、Topics 已改为 AI 宠物 / 陪伴应用、SQLite 定位，`duckdb` 标签已移除。
- 官网、`styles.css`、`app.js` 均返回 HTTP 200，响应内容与本地当前文件逐字节一致。
- 真实浏览器能打开新官网，“Try it locally”按钮能跳转至试用区。此次仅检查桌面窗口，没有完成移动端验收。
- 当前提交的 [CI](https://github.com/vibeinging/MemMe/actions/runs/34186164838) 与 [Pages 部署](https://github.com/vibeinging/MemMe/actions/runs/34186164874) 均为 success。
- 新建临时数据库、使用独立端口，运行 `bash demos/rest-demo.sh`：7 条 PASS、退出码 0，总用时 44.58 秒，其中增量 release 构建 40.10 秒。此次运行复用了已有 Rust 依赖、原生运行库和 embedding 模型缓存，不能称为完全干净环境首装。
- [v0.1.2 Release](https://github.com/vibeinging/MemMe/releases/tag/v0.1.2) 已创建。PyPI 与 crates.io API 仍返回 0.1.1，与报告的未完成项一致。
- 本次核查使用 GitHub API 和实际 HTTP 响应；搜索工具返回了旧 README 快照，因此没有把搜索缓存当作最新发布状态。

### R1 · P1：空模型缓存下，首选 Demo 失败

位置：[ONNX 初始化](../../crates/memme-embeddings/src/onnx.rs#L85)、[锁定的下载依赖](../../Cargo.lock#L862)、[Demo 服务启动](../../demos/rest-demo.sh#L65)。

复现方法：保留已安装工具链和运行库，在一个新的临时工作目录中，通过绝对路径运行仓库的 `demos/rest-demo.sh`；设置新的 `MEMME_DEMO_DIR` 和未占用端口。
`fastembed` 默认在当前目录使用 `.fastembed_cache`，因此这一轮模型缓存为空，没有删除或移动原缓存。

结果：31.83 秒后退出码 1，没有进入写入和检索步骤。服务日志为：

```text
model initialization failed: failed to initialize ONNX model 'bge-small-zh-v1.5':
request error: Bad URL: failed to parse URL: RelativeUrlWithoutBase:
relative URL without a base
```

本次实际检查 [模型 config.json](https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/main/config.json) 和 [tokenizer_config.json](https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/main/tokenizer_config.json)，均返回 HTTP 307，`Location` 是 `/api/resolve-cache/models/...` 形式的相对路径。
`Cargo.lock` 锁定 `fastembed 3.6.1`、`hf-hub 0.3.2`；本机对应 `hf-hub` 源码的 `Api::metadata` 直接把 `Location` 交给 `.get(...)`，没有相对原 URL 解析。这与实测错误吻合。

影响：维护者机器上已有模型可以通过，新用户首次运行却可能无法启动；现有 CI 没有覆盖这条真实模型下载路径。
建议修复下载兼容性或提供固定版本、可校验的模型准备流程，再从空模型缓存重跑。不要通过复制维护者缓存来宣称首装已通过，也不要只延长健康检查等待时间来掩盖下载错误。

### R2 · P1：官网手动启动步骤仍漏了 ONNX Runtime

位置：[官网启动步骤](../index.html#L108)。

官网步骤 1 只设置 `MEMME_VEXDB_LITE_EXTENSION`，步骤 2 直接启动 REST server，遗漏：

```bash
export ORT_DYLIB_PATH="$(bash scripts/download-onnx-runtime.sh)"
```

本次在保留已缓存模型、移除 `ORT_DYLIB_PATH` 的子进程中启动同一个默认 ONNX 服务，复现退出码 101：

```text
An error occurred while attempting to load the ONNX Runtime binary
at `libonnxruntime.dylib`: ... (no such file)
```

README 的手动 REST 示例已经包含该环境变量，官网未同步。建议官网也以完整 `rest-demo.sh` 命令为首选路径；保留手动步骤时须补齐依赖和 recall / restart 指令，目前步骤 3 的代码块只有写入。

### R3 · P1：手动 npm 发布的构建与打包没有使用同一提交

位置：[构建 checkout](../../.github/workflows/publish-node.yml#L36)、[发布 checkout](../../.github/workflows/publish-node.yml#L97)。

构建任务使用 `ref: ${{ inputs.tag || github.ref }}`，发布任务的 checkout 没有同样的 `ref`，因而使用触发工作流的分支提交。
[checkout 官方说明](https://github.com/actions/checkout#usage) 也明确：省略 `ref` 时使用触发事件的 ref/SHA。

这次已经发生源码不一致：[手动发布运行](https://github.com/vibeinging/MemMe/actions/runs/34186138652) 的构建日志显示 `fc7c54c`，发布日志显示 `7a1ddab`。
本次两个提交的版本号均为 0.1.2，五个包又都已在 npm 上，因此全部被跳过，没有证据表明本次发布了错误包。
但以后 main 提升版本后重跑旧 tag，会把旧 tag 的二进制配上新分支的包元数据，可能发布到错误版本。

建议构建与发布固定同一解析后的提交，并核对 tag、五个 package.json 的版本及产物目标架构，再进入 publish。测试这一边界应使用“main 与目标 tag 版本不同”的情况，不需要真正发布包。

### R4 · P2：端口被占用时，Demo 会给出虚假的重启成功

位置：[服务健康检查](../../demos/rest-demo.sh#L65)、[停止服务](../../demos/rest-demo.sh#L78)。

本次用独立临时数据和端口，先启动一个仅用于复查的 MemMe 服务，再让 Demo 使用同一端口、另一数据库目录。
结果：脚本退出码 0、7 条 PASS，输出“重启后仍有记忆”；原服务 PID 一直存活，脚本声称写入的 `demo-memory.db` 实际没有创建。
复查完成后，已关闭该临时服务，没有操作用户原有服务。

原因：健康检查只检查端口上的 `/health`，没有确认响应来自刚启动的进程；`stop_server` 忽略终止失败。脚本实际上始终向先启动的服务读写。
建议启动前检查端口、检查子进程是否提前退出、使用每次运行独立的鉴权值，并在重启验收中确认旧进程已退出、目标数据库存在。
另外，初始化阶段的错误应及时暴露；当前即使服务已经退出，也会一直等满约 30 秒。

### R5 · P2：Python 源码安装示例在仓库根目录执行会失败

位置：[Python README](../../crates/memme-python/README.md#L15)、[官网 Python 卡片](../index.html#L585)。

两处都让用户在 `cd MemMe` 后直接运行 `maturin develop --release`。本次照做，退出码 1：

```text
Failed to parse Cargo.toml ...
missing field `package`
```

根目录是虚拟 Cargo workspace，实际 Python crate 位于 `crates/memme-python`。
建议在虚拟环境中指定 `--manifest-path crates/memme-python/Cargo.toml`，或进入该 crate 目录，并同步修正后续辅助脚本的相对路径。

### R6 · P2：v0.1.2 源码包包含新 README，却不包含它指向的 Demo

`v0.1.2^{commit}` 为 `fc7c54c`；该提交的 README 已推荐 `bash demos/rest-demo.sh`，但 `git ls-tree` 确认该 tag 没有 `demos/rest-demo.sh` 和 `scripts/download-onnx-runtime.sh`，两者在后续 `c2cf6d3` 才加入。

因此克隆 main 的路线有脚本，下载 Release 源码或切换到 v0.1.2 的用户则无法从该目录直接执行同一命令。
建议明确标注此 Demo 当前需要哪个提交，或者随下一补丁版本完整交付；不要把“已有 tag”视为演示已经能按固定版本复现。

### 对原方案的验收结论

| 项目 | 复查状态 |
|---|---|
| 官网、About、Topics、公开 README 同步 | 已落地；官网的部分操作步骤仍有错误 |
| 主分支 CI 与 Pages | 当前提交通过 |
| 有缓存的本地 Demo | 新数据库、独立端口下通过 |
| 首次模型下载与干净试用 | 未通过，R1 阻塞 |
| Demo 对重启事实的验证 | 有端口冲突误判，R4 待修 |
| 发布版本一致性 | 有 R3、R6 待修；PyPI/crates 仍为 0.1.1 |
| 演示视频、外部试用、持续使用、社区传播 | 原实施报告已列为后续，本轮没有新增完成证据 |

建议先处理 R1–R3，再修 R4–R6，然后重跑一套空模型缓存、独立数据、固定版本的试用验收，进入视频录制和外部试用阶段。
本轮没有更改产品代码、工作流或网站，没有 commit、push、tag、重新发布或对外发送消息。

---

## 复查问题修复与验收（2026-09-08，当前窗口）

授权：用户在复查后要求“继续，你来做”。本轮基于 `90d4418fa80d2582b7c8ed747ea7447aa5b2f05d` 修复 R1–R6。
开始时只有上面的复查记录未提交；本轮保留该记录。本节记录提交前的本机验收，当时尚未 commit、push、移动 tag、发布包、部署官网或对外发送消息。用户随后明确授权提交和推送。

**结论：阻塞首次试用的问题已修复，本机空模型缓存 Demo 和 Python 源码安装均通过。远端验证以修复提交对应的新 CI 矩阵结果为准。**

### 修复对应关系

| 问题 | 本轮改动 | 状态 |
|---|---|---|
| R1 首次模型下载失败 | 保留 `fastembed 3.6.1` 和现有 ORT，使用固定的 `hf-hub 0.4.3` 下载模型与 tokenizer 文件到兼容缓存，再由原推理层读取；也准备 E5-large 的外部权重文件 | 相对跳转回归测试、空模型缓存真实推理、缓存重跑均通过 |
| R2 官网缺少运行库且手动流程不完整 | 官网首选命令改为完整 Demo，自动准备 VexDB-Lite / ONNX Runtime；中英文 README、Python 指南与 LLM 入口说明同步，移除“即时启动”说法 | 本地浏览器及实际脚本通过 |
| R3 构建和 npm 打包来源不同 | 新增 `source` 任务，验证输入是版本 tag、解析提交、检查五个 npm 包和四个可选依赖版本；构建与发布都依赖这个任务并 checkout 同一 SHA | 实际前置脚本、版本反例和工作流结构检查通过；未执行发布 |
| R4 误用已有服务而声称重启成功 | 构建前检查端口；每次运行生成独立鉴权值；就绪请求必须鉴权；检查子进程存活、真实退出码和目标数据库文件；初始化失败及时显示日志 | 占用端口及提前退出回归通过，真实重启 Demo 通过 |
| R5 Python 安装命令不正确 | 创建虚拟环境，显式指定 Python crate 的 manifest；运行库按 Python 解释器架构选择，兼容 ARM64 Python 与 Rosetta Rust 并存 | 根目录安装、导入、写入、搜索和重新打开数据库通过 |
| R6 tag 缺少 Demo | 首选 clone 显式使用 `main`；说明 `v0.1.2` 源码包不含 Demo 和 ORT 下载脚本，要求反馈附带提交号 | 文档区分已完成；完整固定版本交付仍待下一补丁，不移动旧 tag |

另外：官网的 MCP 参数修正为实际识别的 `--db-path`，补充必需环境变量说明。试用区和绑定区的长命令原先会撑宽网格；改为允许卡片收缩、代码块内部滚动，行内长变量可以换行，并更新 CSS 资源版本以避开旧缓存。

### 本机验证结果

环境：Rust `1.93.1` / `x86_64-apple-darwin`；Python 安装验收使用新建虚拟环境中的 CPython `3.14` ARM64、maturin `1.15.0`。均使用合成测试数据。

| 检查 | 结果及范围 |
|---|---|
| `cargo fmt --all -- --check` | 通过 |
| `cargo clippy -p memme-core -p memme-embeddings -p memme-llm -- -D warnings` | 通过 |
| `cargo test -p memme-core -p memme-embeddings -p memme-llm` | 506 通过、0 失败、24 忽略；含 core 单元 372、集成/边界/设备场景/压力/VexDB-Lite 70、embeddings 9、LLM 49、文档示例 6 |
| ONNX 下载回归 | 本地 HTTP 服务返回相对 307 跳转；下载完成后关闭服务，再准备同一缓存仍成功 |
| 空模型缓存 Demo | 新临时目录、空 `.fastembed_cache`、新数据库和独立端口：7/7 PASS，退出码 0，31.90 秒 |
| 缓存重跑 | 同一数据目录，Cargo 离线模式、出站 HTTP(S) 指向不可达本地代理并允许 localhost：7/7 PASS，退出码 0，34.69 秒，包含增量构建 |
| `python3 -m unittest discover -s scripts/tests -v` | 7 项通过；覆盖每个平台包与依赖版本、旧 tag 配新主包、端口冲突、服务初始化提前退出 |
| npm 真实 tag 前置校验 | 在独立临时 checkout 执行工作流中实际的 shell：`v0.1.2` 通过，输出 `fc7c54c053f1f9c2afb0b9de72c8546f84b0b95b` / `0.1.2`；切到 main 后用旧 tag 校验，正确报 `Checkout is not the requested tag` |
| 工作流检查 | 两份 YAML 可解析；build / publish 都声明依赖 source，并使用 `needs.source.outputs.sha` |
| Python 安装 | 从项目根目录使用新虚拟环境，`maturin develop --manifest-path crates/memme-python/Cargo.toml --release` 构建和安装成功；执行文档中的架构选择和运行库下载命令后，实际导入、写入、搜索、重新打开数据库均通过；该读写示例使用 mock 向量 |
| 官网本地浏览器 | “Try it locally”可进入完整 Demo 区；1280px 桌面试用卡片宽约 337px，375px 窄屏试用及绑定卡片宽 327px，均在内容区域内；长命令在代码块中滚动 |
| 静态检查 | `bash -n demos/rest-demo.sh`、`git diff --check` 通过 |

试验中还实际暴露并修正了两个问题：Demo 新增显式模型参数时，应使用 clap 接受的 `bge-small-zh-v15`；Python ARM64 加载 Rust x64 架构的扩展会失败，因此安装说明不能只依赖 `rustc` 推断架构。

### 验收边界和后续

- 空模型缓存验收复用了已安装的 Rust 工具链、Cargo 依赖和原生库缓存，不能称为全新操作系统首装。31.90 秒不能作为新用户总安装耗时承诺。
- 新增 CI 的 Linux / macOS 空模型缓存 Demo 任务，以及 Demo / npm 回归任务。本机验收结束时尚无修复提交对应的远端 CI 结果，不能用本机结果替代。
- 24 项忽略测试仍是 1 项设备场景和 23 项真实 LLM 测试；本轮未完成真实设备或付费 LLM 验收。真实 ONNX 网络下载验收覆盖默认 Demo 模型；其余模型没有逐个下载。
- 浏览器验收针对修改后的试用和绑定入口；没有重新验收官网全部交互或在手机硬件上测试。
- `v0.1.2` tag 仍为 `fc7c54c`，本轮内容需要随后续提交和补丁版本交付。PyPI / crates 发布、演示视频、外部试用与社区传播仍是原计划后续项。

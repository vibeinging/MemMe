# MemMe VexDB-Lite SQLite 变更评审

日期：2026-09-01  
状态：`READY_FOR_RELEASE_CANDIDATE`  
结论：原评审发现的代码阻断项已经修复。macOS/Linux 的 Rust core、Python 和 Node 可以进入发布候选验收；Windows、iOS、Android 和 WASM 已明确暂停，不再生成看似可用但运行失败的发布产物。

## 评审范围

- 当前分支：`main`
- 当前提交：`991139e892fca4e41ef493eeda6290f10d1920b7`
- `origin/main`：同一个提交
- 评审对象：工作区中把 `sqlite-vec` 替换为 VexDB-Lite SQLite 的未提交改动，以及受影响的绑定和发布流程
- 未纳入本次结论：工作区中原有的 benchmark 大量删除、`references/`、`AGENTS.md`、旧项目状态报告等无关改动
- 当前没有 PR，因此本次是对 `main` 上未提交工作区变更的评审

## 范围结论

原目标是只保留 VexDB-Lite SQLite，不再保留 `sqlite-vec`。当前实现完成了 Rust core 的动态加载、Vex 虚拟表、基础增删改查、重开数据库和 CI 下载脚本。

后续优化已经把“只保留 VexDB-Lite”落实到事务、索引版本、旧数据清理、测试门禁、绑定 API 和发布矩阵。以下 P1/P2 内容保留为原始评审记录，处理结果见下一节。

## 优化结果

| 原问题 | 处理结果 |
|---|---|
| 源数据、Vex、FTS 分开写 | 增加同连接 `IMMEDIATE` 事务；memory/event 的新增、更新、删除和批量清理全部传播错误并整笔回滚 |
| 维度和格式未校验 | 构造阶段先检查权威 embedding 和现有 Vex 配置；保存 `vector_dimensions` 与 `vector_index_version` |
| 用户删除和旧 sqlite-vec 数据 | 用户删除、reset、裁剪、过期清理统一同步索引；核对 schema 后删除旧 vec0 shadow 数据 |
| 绑定包不可运行 | Python/Node 增加显式扩展路径；只发布有官方动态库的 macOS/Linux；Windows、移动和 WASM 暂停 |
| metadata 扫描 | 给 Vex shadow metadata 建唯一 ID、user 和常用 scope 组合索引 |
| 测试假绿 | 缺扩展直接失败；Vex 专项扩展为 10 项，包含失败注入、维度、scope、迁移和重开 |
| 不安全 `/tmp` | 改为私有用户缓存或 `RUNNER_TEMP`，检查 owner/符号链接，并校验压缩包和动态库 |
| 文档旧用法 | 根 README、绑定 README、Rustdoc、示例和设计文档同步新前置条件及平台边界 |
| `limit=0` | 在生成 embedding 和调用 Vex 前直接返回空结果 |
| 删除和 scope 过滤扫描 | 删除先通过唯一 metadata 索引取 rowid；agent/run/app 下推 Vex KNN |

迁移时会删除旧 sqlite-vec shadow 数据，但保留两条旧虚拟表 schema 记录。原因是缺少 sqlite-vec 模块时无法安全执行虚拟表 `DROP TABLE`，而直接编辑 `sqlite_schema` 有数据库损坏风险。回退旧版必须恢复迁移前备份。

## 原 P1：发版前必须处理（已关闭）

### 1. 源数据、Vex 索引和 FTS 不是同一个事务

证据：

- `crates/memme-core/src/storage/crud.rs:80-116` 先提交 `memories`，再写 Vex，最后写 FTS。
- `crates/memme-core/src/storage/crud.rs:337-369` 先更新源行，再单独删除、插入 Vex。
- `crates/memme-core/src/storage/stream.rs:187-218` 的 event 更新也是分开的，并且忽略 Vex 删除错误。
- `crates/memme-core/src/storage/backend.rs:54-60` 每次只锁定并执行一条语句，没有覆盖整次业务操作的事务接口。

已用真实 VexDB-Lite `v0.0.17` 复现：在 3 维索引上用 4 维配置更新 memory，API 返回维度错误，但源行已经更新，旧 Vex 行已经删除。下次启动只检查表和后端标记，不会修复这个缺口。

影响：搜索会永久漏数据、读到旧向量，或出现重复索引行。应用收到错误并不代表操作已经回滚。

建议：增加持有同一连接锁的事务 API，把源表、Vex 和 FTS 写入放进一次事务；不要忽略删除错误；增加失败注入和并发更新测试。启动时还应有 dirty marker 或一致性修复入口。

### 2. 索引维度没有持久化和校验

证据：

- `crates/memme-core/src/storage/mod.rs:67-103` 只记录 `vector_backend=vexdb-lite`。
- `crates/memme-core/src/storage/dialect_sqlite.rs:173-187` 使用 `CREATE VIRTUAL TABLE IF NOT EXISTS`，旧表存在时不会核对 `FLOAT[dims]`。

已用真实扩展复现：3 维 Vex 表用 4 维 DDL 重开不会报错，4 维写入和查询才报 `vector/query dimension 4 != index dimension 3`。

影响：服务启动和健康检查可能正常，第一次真实写入或搜索才失败。更糟时会触发第 1 项的半完成更新。

建议：在 `memme_config` 保存索引维度和索引格式版本。维度不一致时，在任何写入前明确拒绝；若要支持切换维度，必须先重新生成 embedding，再重建索引。

### 3. 用户删除会吞掉 Vex 错误，旧 sqlite-vec 数据也没有清理

证据：

- `crates/memme-core/src/storage/mod.rs:404-417` 用 `let _ =` 丢弃两次 Vex 删除错误，随后删除权威源行并返回成功。
- 旧版会创建 `vec_memories`、`vec_events` 及持久化 shadow tables；当前 `ensure_vector_index()` 只创建和重建新 Vex 表，没有删除旧表。
- `crates/memme-core/tests/vexdb_lite.rs:167-192` 创建的是 Vex 数据库，不是旧 sqlite-vec 数据库，因此没有覆盖这个迁移。

影响：用户收到“删除成功”时，向量和 `user_id` 仍可能留在 Vex 或旧 sqlite-vec shadow tables 中。旧版本回退也会读取已经过期的旧索引。

建议：把用户删除放进事务并传播所有错误。增加有版本号的单向迁移，使用真实 sqlite-vec `0.1.9` fixture 验证旧表和 shadow tables 的清理。回退必须恢复迁移前备份，不能原地回退到旧索引。

### 4. 当前发布矩阵会产生不能运行的绑定包

证据：

- Python `crates/memme-python/src/lib.rs:159`、Node `crates/memme-node/src/lib.rs:437-499`、WASM `crates/memme-wasm/src/lib.rs:28-33`、UniFFI `crates/memme-ffi/src/lib.rs:381-425` 都走依赖本地动态库路径的 `MemoryStore::new()`。
- Python/Node 发布流程只打包各自模块，没有把 VexDB-Lite 放进包。
- `publish-mobile.yml` 只构建 `libmemme_ffi`，没有静态注册 Vex。实际 iOS 产物能编译，但 `nm` 找不到 `sqlite3_vexdblite_init` 或其他 Vex 符号。
- Python 和 Node 仍发布 Windows x86_64，Android 仍发布 armv7。
- [VexDB-Lite v0.0.17 Release](https://github.com/VexDB-THU/VexDB-Lite/releases/tag/v0.0.17) 有 macOS、Linux、iOS、Android arm64/x86_64、OHOS 和 WASM 资产，但没有 Windows，也没有 Android armv7 资产。

影响：打全平台 tag 后，会出现“包能安装或库能编译，但创建 store 失败”的产物。Windows 目前连官方扩展资产都没有。

建议：如果第一阶段只支持桌面/服务端，就暂停 Windows、移动和 WASM 发布任务，并在包元数据中明确支持范围。Python/Node 要么把对应 Vex 动态库放进平台包并从包内解析可信路径，要么明确要求外部安装。移动端和 WASM 必须接入官方静态或专用资产后再恢复发布。

### 5. 主 KNN 路径按 user_id 过滤，但没有 metadata 索引

证据：

- `crates/memme-core/src/storage/dialect_sqlite.rs:175-185` 声明 `user_id` metadata，但没有创建 shadow metadata 索引。
- `crates/memme-core/src/storage/dialect_sqlite.rs:228-232` 每次 KNN 都加 `user_id = ?`。
- VexDB-Lite `v0.0.17` 的 `FilterKnn` 会先从 `<vtab>_vectors` 按 metadata 查允许的 rowid；默认只有 rowid 主键。

影响：每次搜索可能先扫描所有用户的 metadata，复杂度接近总向量数。数据越多、用户越多，延迟越明显。

建议：优先在 VexDB-Lite 提供正式的 metadata index 能力；若短期使用 shadow table 索引，必须固定并测试其兼容约束。增加多用户大数据量基准，确认无关用户数量增加时延迟不会线性上涨。

## 原 P2：应该随本次改动补齐（已关闭）

### 6. Vex 集成测试会假绿，迁移测试没有测试迁移

- `crates/memme-core/tests/vexdb_lite.rs:65-69` 和 `:168-172` 在环境变量缺失时直接 `return`。
- 实际执行 `env -u MEMME_VEXDB_LITE_EXTENSION cargo test -p memme-core --test vexdb_lite`，结果显示 `2 passed`，但两个测试都没有加载扩展。
- `default_constructor_uses_vexdb_lite_environment_path` 先用 Vex 创建数据库，再用 Vex 重开，不能证明 sqlite-vec 到 Vex 的迁移。
- `docs/design/2026-09-01_vexdb-lite-sqlite-backend.md:79` 当前把它写成已验证迁移，证据不成立。

建议：本地可选测试用 `#[ignore]`，CI 明确用 `--ignored` 运行；CI 缺扩展时必须失败。增加真实旧库 fixture，覆盖 memory、event、重复打开、失败回滚、维度变化和旧表清理。

### 7. 下载脚本使用可预测的共享 `/tmp` 路径

- `scripts/download-vexdb-lite-extension.sh:5` 默认路径是 `/tmp/memme-vexdb-lite-v0.0.17`。
- `:35-56` 信任已经存在的目录，只校验压缩包，不检查目录所有者、符号链接，也不重新校验解压后的动态库。

影响：多用户机器上的本地攻击者可以预先创建或替换该路径中的动态库，后续 `dlopen` 会执行被替换的本地代码。

建议：默认使用权限为 `0700` 的 `mktemp -d`，或使用严格检查所有者和符号链接的用户缓存目录。解压后立即校验实际动态库。

### 8. 公开文档和绑定说明仍有旧用法

- `crates/memme-core/src/memory/mod.rs:89`、`crates/memme-core/src/lib.rs:32`、`README_CN.md:263`、多个 examples 和 Python README 仍展示无前置条件的 `MemoryStore::new()`。
- 根英文 README 已说明环境变量，但各语言包用户通常只会看到包内 README。

建议：统一文档和示例，明确受支持平台、扩展安装方式、可信路径要求和第一阶段范围。

### 9. `limit=0` 行为发生回归

`crates/memme-core/src/storage/dialect_sqlite.rs:231` 会生成 `k = 0`。VexDB-Lite `v0.0.17` 返回 `k must be positive`，而旧 sqlite-vec 返回空结果。图搜索路径还可能吞掉这个错误，造成同一 API 在不同配置下行为不一致。

建议：在公开搜索入口对 `limit=0` 直接返回空数组，并增加真实 Vex 边界测试。

### 10. 删除和额外过滤仍会退化为全表扫描

- `crates/memme-core/src/storage/dialect_sqlite.rs:216-244` 按 `memory_id`、`event_id` metadata 删除，VexDB-Lite 当前只对 rowid 等值提供点查询计划。
- `crates/memme-core/src/storage/query.rs:230-280` 只要有 agent、run、app 或自定义 filter，就完全绕过 Vex，扫描源表 embedding。

建议：先用索引映射解析 Vex rowid 再删除；把常用过滤字段作为 Vex metadata 推下去。无法推下的复杂过滤，可以先取扩大后的 Vex 候选集再过滤，不要直接扫描全部 embedding。

## 当前验证

- 官方 VexDB-Lite `v0.0.17` Release 的 macOS/Linux SQLite 动态库可加载，`vexdb_version()` 可用。
- 设置真实扩展路径后，Vex 专项测试：11/11 通过，并确认 SQLite 查询计划使用 memory ID 和 user/agent metadata 索引。
- 完整 `cargo test -p memme-core`：408 个测试通过，24 个忽略。
- 四个绑定 `cargo check` 通过；Node x64/arm64 NAPI 构建通过；Python 3.12 和 Node 已完成显式路径/环境变量真实加载。
- `cargo check -p memme-ffi --target aarch64-apple-ios` 通过，但移动构造器仍保持阻断，不能解释为移动端可用。
- 下载脚本语法、安全缓存行为、YAML/JSON 和 `git diff --check` 通过。
- `cargo clippy -p memme-core -p memme-embeddings -p memme-llm -- -D warnings` 通过。
- `cargo fmt --all -- --check` 通过。

## 剩余外部验收

1. 在 GitHub Linux runner 上执行实际 wheel/npm 安装验收。
2. 用由 sqlite-vec `0.1.9` 真实生成的固定数据库再跑一次升级；当前测试使用等价的 vec0 schema/shadow fixture。
3. 增加多用户大数据量基准，确认 metadata 过滤延迟不会随无关用户数量线性增加。
4. 完成 Vex 静态注册后，再恢复 iOS、Android 和 WASM 发布并做真实设备/运行时验收。
5. Python 3.14 需要先升级 PyO3；当前 PyO3 0.24 支持到 Python 3.13。

# MemMe 数据安全加固验收

日期：2026-09-02

## 结论

`2026-08-31-memme-integration-research.md` 在 2026-09-02 复查中指出的 7 个
MemMe 数据安全问题，已经在当前工作区逐项修复并加上回归测试。普通事件写入、完整
导出、完整导入、整用户删除、在线备份和受控恢复可以继续用于小智适配器开发。

MemMe 随后已经接入婴喜爱生产环境：小智 `Memory_memme` provider、稳定身份映射、
持久重试队列、低权限召回边界和目标 Linux ONNX 链路都已完成。智控台可以选择
MemMe，但默认仍未切换，现有真实智能体也没有选择它。公开 Git 版本和包发布是代码
分发事项，不再阻断这台服务器上的当前集成。

后文保留每轮复查时的原始结论；其中“尚未接入小智”或“目标 Linux 未验收”只代表
当时状态，以本节和下一节为准。

## 婴喜爱生产验收

- 环境：CentOS 7、2 CPU、4 GB；MemMe 只监听 `127.0.0.1:8080`。
- 运行时：VexDB-Lite v0.0.17、ONNX Runtime 1.17.3、本地
  `bge-small-zh-v1.5` 512 维向量；LLM 关闭。
- 鉴权：`/health` 匿名 200，`/diagnose` 匿名 401；带 Bearer Key 时 storage 和
  embedder 检查均通过。
- 真实 provider 合成测试：20 次写入 p50/p95 为 75.220/84.981 ms；MemMe 重启后
  20 次召回 p50/p95 为 54.736/62.547 ms，20/20 找回目标内容，重试队列为 0。
- MemMe 重启后约 498 ms 恢复健康；服务内存低于 systemd 的 768 MB 上限。
- 合成用户与本地队列已经删除，删除后的 memory、entity、relationship 统计均为 0。
- MemMe、管理接口、语音服务和 Nginx 均为 active；数据库迁移已执行，MemMe 配置
  为启用、非默认，真实智能体选择数为 0。

当前生产路径写入并召回原始文字事件，`compact_on_save=false`，不依赖 DeepSeek。
启用 LLM 后的 compact/meditate 是可选后台能力。真实家庭数据启用前仍需明确授权，
并完成磁盘或应用层加密及密钥轮换。

## 本轮修复

1. 完整导出升级到 `3.0`，不再调用带默认分页的 list 方法。session、event 和
   episode 会全部导出；旧的无 owner source 只要被该用户的数据引用，也会带上。
2. 完整导入改为一个 SQLite `IMMEDIATE` 事务。任一数据层、向量索引或 FTS 重建
   失败，整批回滚；失败后数据库连接仍可继续写入。
3. 导入会校验已有 session 的 `user_id/source_id/agent_id/app_id/run_id`，并拒绝
   event 跨主人或跨宠物挂到旧 session。图关系、memory-entity link 和 association
   也会校验引用对象的 owner。
4. `FullExport` 补齐 memory 的纠错、有效期、session/episode、强度、证据、同步和
   pinned 字段，并增加 history、procedures、meditations、recalls、
   memory_entities、associations 六类数据。
5. `/diagnose` 进入和 `/v1/*` 相同的 Bearer Token 鉴权层。配置 API Key 后，只有
   `/health` 可匿名访问。
6. 备份改用 SQLite Online Backup API。恢复先复制到候选文件，再按当前向量维度、
   VexDB-Lite 和 collection 配置完整打开；旧主库保留到新库再次打开成功，失败时
   自动回退。
7. `/v1/data/import` 单独限制为 32 MiB；完整导入最多 50,000 条记录、最多约
   128 MiB 生成向量；embedding 每批最多 128 条；REST 同一时间只运行一个完整
   导入。更大的迁移明确走已校验的 SQLite 备份。

## 回归测试

新增或加强的测试覆盖：

- 25 个 session、105 个 event、105 个 episode 全量导出，证明不会落回旧的
  20/100/100 分页上限。
- 导入先写入新 session，随后遇到冲突 event；断言新 session 被回滚，旧 event
  未改变，并且失败后还能继续写 memory。
- 导入已有 session 的跨主人、跨宠物冲突；断言 event 未写入。
- 权威 memory 字段和 6 个补充数据层执行 export -> import -> export 往返校验。
- `/health` 匿名成功、`/diagnose` 匿名 401、携带正确 Token 成功。
- 超过 32 MiB 的完整导入请求返回 413。
- 备份向量维度或 collection 不匹配时，恢复被拒绝，原主库标记仍存在。

## PetMemBench

本机 release 模式、VexDB-Lite v0.0.17、2,000 条记忆、100 次查询，连续 3 次：

| 指标 | 结果 |
|---|---:|
| 必须场景 | 11 / 11 |
| 扩展场景 | 3 / 3 |
| Recall@K | 100% |
| 查询 p50 中位数 | 1.857 ms |
| 查询 p95 中位数 | 2.261 ms |
| 查询 p99 中位数 | 17.525 ms |
| 写入吞吐中位数 | 191.9 memories/s |
| SQLite 文件中位数 | 10,412,032 bytes |

原始结果：`benchmarks/petmem/results/2026-09-02_data-safety-final-run1.json` 到
`run3.json`。

这些数字只验证确定性测试 embedding 下的存储与检索，不代表目标 2 核 4 GB Linux
机器上的真实 ONNX、DeepSeek、语音链路或最终回复质量。

与 2026-09-01 三次基线相比，查询 p50/p95 没有变慢，p99 基本持平，但本轮写入吞吐
中位数从 445.7 降到 191.9 memories/s。测量时主机 load average 为 16.9 到 20.8，
有两个长期 bun 进程各占一个 CPU 核，同时有 FileProvider 和全盘 find 持续占用 IO；
因此不能把这组写入差异直接归因于代码，也不能写成“性能通过”。应在空闲机器或目标
Linux 设备上用同一命令再跑三次，确认写入吞吐后再设发布性能门槛。

## 仍需外部完成

- [已在小智未提交工作区完成] `Memory_memme` provider、稳定身份映射和持久重试队列。
- [已在小智未提交工作区完成] 召回内容作为低权限、不可信历史输入，不覆盖 system/tool 指令。
- 固定当前源码为可追溯的 Git tag/release，并完成需要的 crates.io/PyPI/server 发布。
- 在目标 2 核 4 GB Linux 机器上记录启动时间、RSS、写入和召回 P50/P95，并跑真实
  ONNX + DeepSeek compact/meditate。
- 第一轮只使用虚构数据和独立数据库；上述外部验收完成前不接入真实家庭数据。

## 第二轮复查后的补充修复

同日更深一层复查又发现 7 个问题。本轮继续修复如下：

1. `EMBEDDING_API_KEY`、`OPENAI_API_KEY`、`LLM_API_KEY` 和 `MEMME_API_KEY`
   都设置了 `hide_env_values = true`。单元测试检查四个参数，另用四个虚构 Key
   实跑 `memme-server --help`，帮助页不再出现值。此前已进入命令日志的真实
   `OPENAI_API_KEY` 仍需由账号持有人轮换；代码无法替用户完成这项外部操作。
2. portable import 现在要求目标库不存在任何待导入主键；任何 source、session、
   event、episode、memory、图谱或审计层 ID 冲突都会让整个事务失败，不再保留旧值
   却返回成功。memory 的 `session_id`、`episode_id`、`episode_ids` 和
   `superseded_by` 还会检查引用对象存在且属于同一用户。
3. 完整导入只接受格式 `3.0`，并要求 export 的 collection 与目标 collection
   完全一致。未知版本或错误 collection 会在生成向量和写库前失败。
4. REST 收到恢复请求并停服后，即使完整候选校验或安装失败，也不再用 `?` 退出
   进程。旧库由核心恢复流程保留或回退，服务循环随后继续用旧库重启。
5. 远程 embedding 失败时，原始 event 仍写 SQLite，但响应新增
   `embedding_pending`。同一个 `event_id` 精确重放时会检测空向量并重新生成、补写
   主表和 Vex 索引。Node 和 Python 返回值同步加入该字段及 `events_replayed`。
6. 完整导入的单任务门槛移到 Axum JSON extractor 之前。已有导入时，新请求在读取
   和反序列化 32 MiB JSON 前就返回 `503`。回归测试用非法 JSON 证明门槛先执行。
7. 完整导出的所有数据层改为在一个 SQLite deferred read transaction 中读取。
   WAL 回归测试在两次读取之间从另一连接写入，事务内第二次读取仍看到同一快照，
   事务结束后才看到新记录。

完整验证结果：

- `cargo test -p memme-core -p memme-server`：核心 376、edge 24、integration 6、
  mobile 19、stress 10、VexDB-Lite 11、REST 9、doc tests 3，全部通过；另有 1 个
  offline 场景和 23 个真实外部 LLM 测试按设计忽略。
- Node/Python Rust target 构建测试通过。
- core、embeddings、LLM、server、Node、Python 全目标 clippy `-D warnings` 通过。
- `cargo fmt --all -- --check`、`git diff --check HEAD` 通过。

第二轮修复后又连续运行三次 PetMemBench：必测 11/11、扩展 3/3、Recall@K 100%。
中位数为 p50 1.948 ms、p95 2.939 ms、p99 14.904 ms、写入 352.2 条/秒。原始结果为
`benchmarks/petmem/results/2026-09-02_second-hardening-run1.json` 到 `run3.json`。

本次测量前 load average 为 18.56；Python、`find` 和两个 `bun` 进程都在长期占用
CPU。因此 352.2 条/秒仍不能和空闲机的 445.7 条/秒基线直接比较。正确性门槛通过，
性能结论仍需在空闲机或目标 2 核 4 GB Linux 上复测。

## 小智 provider 接入进度

小智工作区已实现 `Memory_memme` provider、显式稳定的
`user_id/agent_id/app_id`、本地 SQLite 待发队列和不可信记忆提示边界。
小智新增测试 7/7 通过，当前整套测试 34 项和 12 个子测试全部通过。用真实 MemMe REST
服务和 mock embedding 写入、重放虚构事件后，得到 `events_replayed=2`、
`embedding_pending=0`，小智本地队列清空。

这些都是当前未提交工作区的结果，不等于已发布。默认仍是 `nomem`；
真实 ONNX + DeepSeek、目标 Linux 资源、稳定账号映射、加密存储、密钥轮换和
双仓库 commit/tag/release 仍是上线门槛。

## 第三轮复查后的补充修复

第三轮复查列出的 6 个 P1 问题已经在当前 MemMe 工作区修复，并增加了读取侧、
恢复侧和绑定侧的第二道保护。

1. portable import 会先为 memory、session、event、episode、entity、identity 等层
   建立 ID 到 owner 的映射，再检查 episode 的 event/session、relation 的 episode、
   identity 的 evidence、history 的 memory、memory-entity link 和 association。
   所有检查和目标主键冲突预检都在 embedding 前完成；事务内仍保留二次检查。
   episode 消息和 meditation 读取 event 时还会带上 owner 条件，旧坏数据也不能跨用户读。
2. 恢复增加 `MemoryError::RollbackFailed`。只有旧主库已经放回、完整打开并验证成功，
   server 才能继续重启；否则进程退出并保留 rollback 路径，避免 SQLite 在缺失路径上
   自动创建空库。
3. 整用户删除会在原数据仍存在时删除任一端命中该用户 memory、event、episode、
   identity 或 entity 的 association。测试逐层覆盖，并证明其他用户自己的 association
   仍保留。
4. REST LLM 地址只接受 HTTPS 公网目标，默认拒绝回环、内网、链路本地和保留地址，
   也不跟随 HTTP 重定向；校验后的 DNS 地址会固定到 HTTP client，避免校验后再解析
   导致 DNS rebinding。只有 operator 通过 `MEMME_LLM_ALLOWED_HOSTS` 显式列出的主机
   可以使用私网或 HTTP。上游错误正文不再进入公开错误信息。
5. LLM 非密钥配置在一个 SQLite 事务中保存，保存成功后才切换运行态。server 启动
   和恢复重启都会读取 SQLite 的 model/base_url，API Key 仍只在进程环境或当前运行态。
   同时修复了 blocking HTTP client 在 Tokio 异步任务内创建或销毁时可能 panic 的问题。
6. SQLite 主文件、WAL/SHM、候选恢复文件、replica 和 backup 文件会强制设为 `0600`；
   代码新建的数据库目录和 REST backup 目录为 `0700`。新增 systemd 示例设置
   `UMask=0077` 和 `StateDirectoryMode=0700`。磁盘加密仍是独立上线条件。

同轮还完成了这些合同修正：图片字段目前会明确返回“不支持”，不再静默丢弃；Axum
JSON 解析错误统一为 JSON 错误结构；完整导入忙时返回 `503` 和 `Retry-After: 1`；
restore 的完整性检查移到 blocking task；Bearer Token 使用固定时间比较；Node 和
Python 增加带稳定 event ID 的 `append_events_idempotent` 重放接口。

### 第三轮验证

- 标准 core/server 完整套件：core 382、edge 24、integration 6、mobile 19、stress 10、
  VexDB-Lite 11、server 13、doc tests 3，全部通过；1 个 offline 和 23 个真实外部
  LLM 测试按设计忽略。
- Node/Python test target 构建通过。
- 把 core、server、Node、Python 合并为一条 `cargo test` 时，Cargo feature union 额外
  启用了需要外部模型的 ONNX reranker 测试；该项超过 3 分钟没有结束后被停止，不计入
  通过数。标准 core/server 套件、相关 ONNX 构造测试以及 Node/Python target 分别验证。
- core、embeddings、LLM、server、Node、Python 全目标 clippy `-D warnings`、fmt 和
  `git diff --check HEAD` 通过。
- PetMemBench 连续三次均为必测 11/11、扩展 3/3、Recall@K 100%。中位数为 p50
  1.812 ms、p95 2.185 ms、p99 14.642 ms、写入 419.0 条/秒。原始结果在
  `benchmarks/petmem/results/2026-09-02_third-hardening-run1.json` 到 `run3.json`。

与第二轮三次测量的中位数相比，p50 从 1.948 ms 降到 1.812 ms，p95 从 2.939 ms
降到 2.185 ms，p99 从 14.904 ms 降到 14.642 ms，写入从 352.2 增到 419.0 条/秒。
这只说明本机确定性 benchmark 没有出现回归，不能代替目标 Linux 的真实 ONNX、
DeepSeek、RSS 和语音链路验收。

仍未完成的非 P1 项：完整导入的持久 `import_id + digest` 回执、超大导入时进一步缩短
唯一 SQLite 连接占用、应用层数据库加密、目标 Linux 真机测试，以及公开 commit/tag/
release。完成这些外部上线步骤前，继续使用虚构数据和独立数据库。

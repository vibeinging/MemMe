# MemMe 路线图

> Memories that are actually yours.
>
> 最后更新：2026-04-01

这份路线图说明已完成的功能、下一步计划，以及欢迎社区参与的方向。

---

## 已发布 (v0.1)

MemMe 0.1 是一个功能完整的 AI 记忆引擎：

- **四层数据模型** — 事件流 → 会话 → 片段 → 记忆
- **知识图谱** — LLM 实体/关系提取，存储在 DuckDB 中
- **四通道混合检索** — 向量 + BM25 + 实体 + 时间，RRF 融合，可选重排序
- **遗忘曲线** — 基于 FSRS 的记忆衰减，访问时强化稳定性
- **冥想** — 记忆整合：衰减 + 提取 + 图谱构建 + 身份特征蒸馏
- **Smart 模式** — LLM 事实提取 + 去重（或纯向量模式，延迟 <10ms）
- **隐私控制** — 逐条设置：仅本地 / 可同步 / 加密同步
- **多语言绑定** — Rust、Python (PyO3)、Node.js (NAPI-RS)、Swift/Kotlin (UniFFI)
- **REST API** — axum 服务，23 个端点，OpenAPI 规范
- **MCP 服务** — Claude Desktop / Cursor 集成
- **交互式 Playground** — 本地 Web 体验，Remember / Recall / Chat 三种模式

完整功能列表见 [README](../README_CN.md)。

---

## 下一步

### 数据导入

让用户方便地把已有的 AI 对话历史导入 MemMe。

- [ ] ChatGPT 历史导入（`conversations.json` → Session/Episode）
- [ ] Claude 历史导入
- [ ] 通用对话导入（JSON/JSONL 标准格式）
- [ ] CLI 工具：`memme import --format chatgpt --file conversations.json`

### 生态集成

到开发者已经在的地方去。

- [ ] **LangChain** — 实现 BaseMemory 接口
- [ ] **LlamaIndex** — MemMe 作为检索器/存储后端
- [ ] **CrewAI / AutoGen** — 多 Agent 框架共享记忆
- [ ] **Obsidian** — 笔记与记忆双向同步

### 移动 SDK 打包

超越原始 FFI 绑定的原生 SDK。

- [ ] iOS SDK（CocoaPods / Swift Package Manager）
- [ ] Android SDK（Maven / Gradle）
- [ ] 端侧 LLM 集成（llama.cpp / MLX）

### 高级记忆

推动 AI 记忆能力的边界。

- [ ] 后台整合（定时冥想）
- [ ] 视觉记忆 — 从图片中提取和存储记忆
- [ ] 记忆聚类 — 自动发现主题群组
- [ ] 情感标注
- [ ] 跨记忆因果推理

### 性能

- [ ] WAL 模式实现读写并发
- [ ] Rust 原生异步 API
- [ ] 流式搜索结果

---

## 进行中的集成

以下集成已有初步代码，但尚未完成：

| 集成 | 说明 | 状态 |
|------|------|------|
| [YiYi](https://github.com/vibeinging/YiYi) | 桌面 AI 个人助手 — MemMe 驱动其记忆系统 | 已集成 |
| OpenClaw | Agent 框架记忆插件 | 初步搭建 |
| Dora-rs | Rust 机器人框架记忆节点 | 初步搭建 |
| LeRobot | Hugging Face 机器人框架记忆封装 | 初步搭建 |
| Copper-rs | 实时机器人框架 CuTask | 初步搭建 |

---

## 明确不做的事

- **托管云服务** — MemMe 是引擎，不是平台
- **通用向量数据库** — MemMe 用向量但不跟 Qdrant/Milvus 竞争
- **Python 优先** — Rust 是唯一的事实来源，所有绑定都是生成的

---

## 参与贡献

以下方向特别欢迎社区参与：

- **导入格式** — 支持更多聊天记录格式（Gemini、Copilot 等）
- **框架适配** — LangChain、LlamaIndex、CrewAI 集成
- **语言绑定** — 将新的核心 API 暴露到 Python/Node/Swift
- **基准测试** — 在新数据集或新硬件上运行 MemMe

详见 [CONTRIBUTING.md](../CONTRIBUTING.md)。

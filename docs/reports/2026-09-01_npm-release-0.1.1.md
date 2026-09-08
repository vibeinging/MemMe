# npm 0.1.1 发布报告

日期：2026-09-01

## 发布结果

已发布以下平台包：

- `memme-darwin-arm64@0.1.1`
- `memme-darwin-x64@0.1.1`
- `memme-linux-arm64-gnu@0.1.1`
- `memme-linux-x64-gnu@0.1.1`

顶级包名 `memme` 被 npm 的相似名称规则拒绝。主包已按 npm 的建议改为
`@wjmwjmwb/memme@0.1.1` 并发布。其 `latest` 标签为 `0.1.1`，访问级别为
`public`。

## 支持范围

- macOS：arm64、x64
- Linux glibc：arm64、x64
- Windows：本版本不支持，因为 VexDB-Lite v0.0.17 没有 Windows SQLite
  扩展包
- VexDB-Lite 动态库不打进 npm 包。应用必须提供经过校验的动态库绝对路径，
  或设置 `MEMME_VEXDB_LITE_EXTENSION`

## 校验

- `cargo fmt --all -- --check`：通过
- `cargo clippy -p memme-core -p memme-embeddings -p memme-llm -p memme-node -- -D warnings`：通过
- `cargo test -p memme-core -p memme-embeddings -p memme-llm`：通过
- macOS arm64 本地 tarball 安装、写入、搜索：通过
- macOS x64 本地 tarball 安装、写入、搜索：通过
- 作用域主包本地 tarball 安装、写入、搜索：通过
- npm 公开索引安装：通过
- macOS arm64 从 npm 官方仓库全新安装、写入、搜索：通过
- macOS x64 从 npm 官方仓库全新安装、写入、搜索：通过

主包发布后经过 npm 的发布时安全扫描。扫描期间 `dist-tag` 和访问级别可见，
公开包信息暂时返回 404；扫描完成后公开安装恢复正常。

## tarball SHA-256

```text
7b1c48590193dd675cc95450742fa7e27667f332c7e43c9968be8ef55382ba4f  wjmwjmwb-memme-0.1.1.tgz
866922620c041b0d30ae8a9a79c9229e100247a1d5b6d9c0437ccf92664ab6ad  memme-darwin-arm64-0.1.1.tgz
aab4bc6166c826302336737e191357e7c40f26c3ce1a83d1f6825fe77f1eda75  memme-darwin-x64-0.1.1.tgz
4a8bcc65c04ec9862330e27a7aea590ea9fde33fcb0dfcf9db7bbe8dd317b555  memme-linux-arm64-gnu-0.1.1.tgz
3989632436e4f687c753e7f3150f483b6db342e6f2afb2fa34de113c98d520e3  memme-linux-x64-gnu-0.1.1.tgz
```

本次没有执行 Git commit 或 push。工作区中原有的其他改动未清理、未重置。

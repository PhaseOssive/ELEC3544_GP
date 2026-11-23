# Colab I/O 优化与 30k TEMP 部署

## 本次 30k TEMP 固定路径

- Fetch 根目录（持久原件）：`/content/drive/MyDrive/GP_Research_Papers`
- 输入主表：`data/processed/papers_master_temp.csv`
- 作者表：`data/processed/authors_arxiv_v1.csv`
- DBLP 辅助目录：`data/processed/dblp_author_features_v3_1_1`
- XML/DTD：`data/raw/dblp/dblp.xml.gz`、`data/raw/dblp/dblp.dtd`
- 持久 Cache：`/content/drive/MyDrive/dblp_cache_30000_temp`
- 持久输出：`/content/drive/MyDrive/GP_DBLP_Author_Recovery/recovery_output_30000_temp`
- Colab 本地工作区：`/content/GP_DBLP_Recovery_Work_30000_TEMP`

旧 599-paper `dblp_cache` 和 `recovery_output` 不会被本 Notebook 使用或覆盖。

## I/O 策略

Google Drive 不承担 SQLite 随机读写或 XML 流式扫描。`colab_runtime.prepare` 在启动前将以下内容复制到 `/content`：

1. `index.sqlite` 和已有 `candidate_history.sqlite`（若有）；
2. `dblp.xml.gz` 与 `dblp.dtd`；
3. `papers_master_temp.csv`、`authors_arxiv_v1.csv`；
4. `bootstrap.py` 会读取的 DBLP 辅助 CSV/JSON（存在的文件全部复制，包括较大的 author-detail/conflict/priority audit）。

复制按 8 MiB 分块，显示真实进度、SHA256 校验，并在复制前估算本地磁盘空间；输入文件只有在源 size/mtime 或本地副本变化时才重复制。同一 VM 重跑会复用本地文件。新 VM 会从 Drive 最新完整 Cache Snapshot 恢复，再重新 stage 输入文件。

业务运行时 `FETCH_DATA_ROOT`、XML、DTD、Profile SQLite 和 Derived SQLite 都指向本地磁盘。输入哈希、CSV 解析和源完整性复核也在本地完成，避免对数 GB Drive CSV 重复执行多遍读操作。新版 `recovery/bootstrap.py` 对大型辅助 CSV 使用分块读取：author-detail 只投影身份重验证所需字段，并只保留当前 30k 作者 roster 对应记录；identity-conflicts 分块读取并只保留当前论文/作者键对应的审计行；其他宽审计表只读取 `arxiv_id` 与 `dblp_record_key` 两列用于 Focal 提示。完整原始 CSV 不修改，仍记录完整 SHA256。

最终 CSV、报告和日志仍写入专用 Drive `OUTPUT_DIR`。这些是阶段性顺序导出，不是 SQLite 的随机读写。Cache 只在 Python 子进程正常退出或收到 SIGINT 并关闭 SQLite 后，作为完整冷快照写回 Drive；绝不复制活动 SQLite。

## 推荐运行顺序

1. 先运行 Notebook 的 PREPARE 单元：只复制/校验输入和 Cache，不启动 Recovery；检查本地磁盘余量、RAM 和大 CSV 表头。PREPARE 不把大 CSV 正文载入 RAM。正式 Bootstrap 使用分块/窄列读取，避免一次性把数 GB 宽表及其 Python 字典副本放入内存。
2. 运行 `stage="scan"`：Bootstrap、候选扩展、全量 XML 扫描和磁盘去重；退出后会写入 `runtime_snapshots/LATEST.json`。
3. 确认日志出现 `Persistent cache snapshot saved` 后，再运行 `stage="all"`：复用本地/持久 Cache，执行网络评分、身份恢复、特征、审计及自动 Missing Treatment。
4. `stage="all"` 正常完成后，读取 `recovery_output_30000_temp/report/final_recovery_summary.json` 和 `report/missing_treatment_summary.json`。

若 Colab 在 Stage 1 与 Stage 2 之间重启，新 VM 会从最新 snapshot 恢复；需要重新执行挂载、上传代码 ZIP、配置和 `stage="scan"` 单元，之后再运行 `stage="all"`。若运行中直接重置/强制回收 VM，未到达快照点的本地进度仍可能丢失。

## Cache 与兼容性

- 首次从 `dblp_cache_30000_temp/index.sqlite` 复制 Profile/Alias Index；不重建、不覆盖原件。
- Derived Cache 在新的 30k 专属目录生成，与 599 旧缓存隔离。
- Snapshot 是完整冷副本，额外占用 Drive 空间；不自动删除旧快照。
- 缓存签名、身份算法、XML 时间边界、Focal 排除、四个正式特征和 Missing Treatment 规则均不因本次 Colab 部署路径调整而改变。
- `colab_runtime.py` 负责 staging、路径映射、进度显示和快照。`recovery/bootstrap.py` 的输入加载改为内存有界的分块/窄列读取；作者身份判定规则、XML 扫描、Focal 排除、时间边界、网络回测、特征定义和 Missing Treatment 决策规则未改变。由于 bootstrap.py 属于算法签名的一部分，本版本应从 Stage 1 开始使用同一 ZIP 完成 Stage 1 与 Stage 2，不要在两阶段之间切换代码包。

## 限制

本地磁盘空间必须同时容纳 XML、Profile Index、输入 CSV、Derived Cache 及 SQLite 临时文件。Notebook 会在启动前估算已知复制量并保留至少 2 GiB 安全空间，但无法准确预测真实数据的 Derived Cache 增长；运行中应留意 `/content` 剩余空间。真实 DBLP 的运行时间不能从合成 30k benchmark 直接外推。

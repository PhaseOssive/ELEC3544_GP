# GP 论文数据采集项目

这是 notebook 的采集部分工程化版本：生成 arXiv v1 论文特征、DBLP 作者历史、Semantic Scholar 窗口引用标签和 OpenReview 公开评分，合并为一篇论文一行的主表。模型训练、预测和“成功”阈值计算属于 proposal 的后续阶段，本包不执行它们。

## 1. 安装与运行

建议 Python 3.12，至少 Python 3.10。解压后进入 **fetch 目录**：

```bash
cd fetch
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell 改用：.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python control/fetch.py --preflight-only --no-network
python control/fetch.py --config config.example.json
```

`requirement.txt` 是单数拼写兼容入口，也可 `pip install -r requirement.txt`。本次运行环境的精确依赖版本另见 `requirements-validated-py312.txt`；该文件需要 Python 3.12+。推荐在独立虚拟环境安装，避免与其他项目依赖混用。

也可以在 fetch 的**父目录**运行：

```bash
python -m fetch.control.fetch --config fetch/config.example.json
# 或 python -m fetch --config fetch/config.example.json
```

路径按启动命令时的工作目录解释。使用已有数据时建议给 `data_root` 和 `dblp_xml_path` 填绝对路径。

不需要 API Key，不会弹出密钥输入框。导入模块不挂载 Drive、不请求网络、不开始采集。默认 2017–2022（含首尾）、每年最多 100 篇，最多约 600 篇。v1 中没有 CS 类别的论文会按原 notebook 规则排除，因此最终数量可能低于采样数量。

**首次运行耗时**：arXiv 保留“全年逐月完整拉取，再按年随机抽样”的原方法。把 target_per_year 改成 1 并不会减少候选池下载量。DBLP 保留本地 XML 全量双遍扫描，首次可能下载 GB 级压缩文件并消耗较多时间和内存；请优先设置现有 `dblp_xml_path`。`minimum_free_gb` 是最低磁盘检查值，不是 DBLP 所需容量估算。

## 2. 先做真实小规模检查

```bash
python control/fetch.py --preflight-only --config config.example.json
```

网络预检只探测端点，不能保证后续所有分页可访问；返回 2 表示至少一个端点不可达。正式入口先检查配置、依赖和存储，再在实际请求时执行限速/重试，不额外强制探测网络，因此完整缓存可以离线复用。

若想快速验证自己的部署，使用已有论文 ID 清单，而不是重新扫描整个年度。`seed_csv` 格式：

```csv
arxiv_id,seed_query_year
1706.03762,2017
```

将 `seed_csv` 指向该文件，设置相应年份，再运行相同入口。**显式 seed_csv 使用文件中的完整清单，不再抽样**；正式研究仍使用默认年度抽样，或使用已经确认的历史 seed 文件。`sources` 可只选择某些下游来源，这会明确标记其他来源 `NOT_COLLECTED`，不会宣称全项目完成。

## 3. 配置与存储

所有可调整运行参数集中在 `control/config.py` 的 `Config`；`config.example.json` 展示全部默认值，`CONFIGURATION.md` 逐项说明。JSON 中未知字段直接拒绝，避免拼写错误悄悄被忽略。

默认 `data_root=./GP_Research_Papers`。数据根目录包含：

```text
GP_Research_Papers/
  data/raw/                     # arXiv XML、OpenReview 请求快照、DBLP 源文件
  data/intermediate/            # 原 V3.1.1 DBLP 两阶段索引
  data/processed/               # 论文表、作者表、引用边、评分表、Master、质量报告
  logs/pipeline.log             # 详细过程与错误回溯
  logs/errors.jsonl             # 可恢复失败与恢复事件
  state/latest.json             # 最近运行状态
  state/runs/                   # 每次运行的配置、版本和阶段摘要
  state/tasks/                  # 各阶段论文状态和待恢复任务
  state/processed_cache/        # 带指纹和校验和的最终表缓存
  run_summary.json              # 最终运行摘要
```

S2 的 raw 和 summary 保留 notebook 原路径：`data/processed/semantic_scholar_public/raw/` 与 `citation_3y_summary_cache/`，避免使已有缓存失效。

## 4. Colab Drive

把 fetch 目录放到 Colab 可访问的位置，先安装依赖，然后使用 `config.colab.example.json`：

```python
!pip install -r /content/fetch/requirements.txt
!python /content/fetch/control/fetch.py --config /content/fetch/config.colab.example.json
```

若 Drive 授权需要在 notebook 主进程弹窗，请改为同进程调用：

```python
import sys
sys.path.insert(0, "/content")
from fetch.control.config import Config
from fetch.control.fetch import run
cfg = Config.load("/content/fetch/config.colab.example.json")
result = run(cfg)
```

只有 `storage_mode=colab_drive` 才会调用 `google.colab.drive.mount`；其 data_root 必须位于指定挂载目录下。挂载可能需要 Google 的交互授权，这与数据源 API Key 无关。Runtime 结束后无法继续后台工作；下一次重新执行入口，从 Drive 中的缓存恢复。本次未在真实 Colab runtime 做挂载验收，详见验证报告。

## 5. 缓存、断点恢复与刷新

通常只需重新执行**相同配置和命令**：

```bash
python control/fetch.py --config config.example.json
```

- `reuse_valid`：校验来源、请求指纹、schema/算法版本、完整性和 SHA256 后复用。
- `refresh_source`：同时设置 `refresh_sources`，例如 `["openreview"]`，只刷新指定来源。
- `refresh_all`：明确请求所有来源更新，成本很高。
- `resume_enabled=false`：不跳过已完成阶段；raw/intermediate 有效缓存仍复用。若要重抓网络，请使用刷新策略。

写入采用临时文件替换；引用边先增量写临时 CSV，完成后替换。S2 每个成功分页立刻保存 raw；恢复时重建内存计数器并读回成功页，只请求缺失页。最终表损坏或被删除会触发重建。失败和部分阶段不会存成可直接跳过的成功 checkpoint。

旧 S2 raw cache、summary、DBLP V3.1.1 pickle 索引保留原格式和键。无 sidecar 的旧缓存首次按结构/原有指纹验证后补登记校验元数据，之后做 checksum 验证。不能追溯证明旧文件首次生成时的网络完整性；这不是将旧缓存当作全新实验结果。请只复用自己生成的 pickle 索引。

已有数据的迁移方式：把 `data_root` 指向原 `GP_Research_Papers`，尽量保留原目录及 DBLP XML 路径和时间戳。**原 DBLP 指纹包含源 XML 路径和 mtime**，搬到不同路径或重写 XML 时可能重建索引；这是保留原指纹规则的行为，旧文件不会无故删除。损坏索引会改名保留为 `.invalid.gz` 再重建。

OpenReview 成功响应默认无限期作为采集快照保留，抓取时间在 `.gp-meta.json` 中；显式刷新才能更新。新鲜度不等于覆盖率，公开搜索本身可能漏掉未索引的评审。

## 6. 网络与失败含义

每个来源独立限速；OpenReview 默认单 worker。429 优先使用 Retry-After（支持秒数和 HTTP 日期），否则指数退避加抖动，并在连续限流时降低速度。`max_retries=7` 表示初次请求之外最多再重试 7 次。断网默认在此预算内按 `reconnection_delay` 等待恢复；不会无限挂起。

预算耗尽后持久化状态，DBLP/S2/OpenReview 可以各自失败而其他独立来源继续。arXiv seed 或 v1 失败则停止依赖它们的下游。401/403 不自动判定为“无匹配”，也不绕过服务方认证要求。

| 状态 | 含义 |
|---|---|
| COMPLETE | 本论文采集计算完成；详细匹配/覆盖仍须查看来源字段 |
| ZERO_CONFIRMED | 依据累计引用为零等证据确定零值 |
| NO_MATCH | 请求成功但没有匹配/公开评分，数值保留缺失 |
| REQUEST_ERROR | 请求或阶段出错，必须看错误日志 |
| PARTIAL | 只得到部分数据、分页受限或存在未解决的覆盖问题 |
| PENDING | 尚未执行，或必要上游未完成 |
| NOT_COLLECTED | 本次配置未启用该来源 |

`state/tasks/*.json` 保存统一论文状态。原始科学状态（例如 `s2_c3_status`、`dblp_match_status`、`openreview_score_status`）保留，统一状态不替代它们。DBLP 完成扫描但作者无法确认时，详细身份字段仍保持未解决和特征缺失。

进程退出码：0 完成；1 启动配置/依赖错误；2 部分完成或失败；130 用户中断。运行时错误详情在日志，不在终端刷完整 traceback。同一个 data_root 只允许一个采集进程。

## 7. 输出与研究语义

主结果是 `data/processed/papers_master_raw_stage.csv` 和同名 JSONL；字段及每个来源文件见 `DATA_DICTIONARY.md`。

- t0 使用 arXiv **v1 published**，不是 updated。
- C3、IC3 同用 `[t0, t0 + 3 个日历年)`，右端不包含。仅年份/月的日期沿用 notebook 的保守判断。`citation_window_years` 可设其他正整数，兼容文件名和 `_3y` 列名保持不变，真实窗口会写入主表 metadata；正式 C3 研究请保持 3。
- DBLP 保留身份冲突、匹配置信度、时间排除、focal work 排除和双遍索引，不强行认定不确定身份。
- OpenReview 保留 reviewer-level 表；paper mean 对同一论坛去重后的总体评分求均值。多论坛、评分字段混用或记录冲突不擅自合并。部分抓取的观察均值保留，但 `openreview_score_status=PARTIAL`，不能当完整标签。
- C3/IC3 请求失败或数据不完整保留缺失，不补 0。引用边去重，并对完整标签核对 `C3 >= IC3 >= 0` 和边表计数。
- Master 从 arXiv v1 左连接；只合并 paper-level 表。一篇论文一行，作者、评委和引用边不直接扩展主表行数。
- OpenAlex 仍标记未采集；论文里尚未实现的机构、国家、全文方法特征没有伪造。

`coverage.csv` 给出逐字段覆盖率；`quality_report.json` 给出主键、标签、引用边、未完成阶段；`disk_usage.json` 给出文件大小、总占用和磁盘剩余空间。`run_summary.json` 区分 pipeline 状态、master_integrity_ok、master_complete、all_sources_collected、all_labels_available。**程序成功完成不代表每篇论文在所有平台都能匹配或拥有公开评分。**

errors.jsonl 事件含 timestamp、run_id、step、paper_id、error_type、message、retry_count、recovery_status。重新完成阶段会追加 RECOVERED 事件，保留旧错误审计记录；当前待处理任务以 state/tasks 和 latest.json 为准。

## 8. 目录与开发说明

`arxiv/`、`dblp/`、`semantic_scholar/`、`openreview/` 各自保留业务逻辑；每个 collect 明确接收 Config/HTTPClient/输入表。为尽量减少算法变动，原 notebook helper 在 collect 的局部作用域中运行，避免隐式全局变量和跨次运行污染。`control/fetch.py` 只协调顺序、结果和失败；`tools.py` 只负责通用 HTTP/限速/缓存/IO；`state.py` 管理状态；`combine.py` 保留 notebook 的论文级特征聚合。

各来源 run 返回统一 SourceResult（包含进程内 tables 和可序列化 summary），内部 collect 保留表级接口。统一阶段摘要字段：source、status、input_papers、processed_papers、failed_papers、cache_hits、network_requests、output_files、errors。摘要保存在 state 和 run_summary，不通过解析 print 判断成功。

按 note.md 顶部的特别要求，`tests/` 留空，没有加入 Golden Dataset 测试代码。开发期间实际运行过的检查及其结果单独列在 `VALIDATION_REPORT.md`。本包不包含模拟实验数据，也不把模拟数值作为研究输出。

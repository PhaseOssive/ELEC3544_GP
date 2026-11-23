# 最终 Missing Treatment 更新

本次只增加最终 Treatment 与新主表输出。身份算法、XML 扫描、Temporal/Focal 审计、回测和缓存模块均未修改。详见 [MISSING_TREATMENT.md](MISSING_TREATMENT.md)。

已有完整恢复结果时运行：

```bash
python main.py --config config.runtime.json --stage treatment
```

只读取原始主表和已有结果，不读取 XML/SQLite，不重扫历史。正常 `--stage all` 完成后也会自动执行 Treatment。

新主表：`OUTPUT_DIR/features/papers_master_dblp_imputed.csv`。Treatment 汇总：`OUTPUT_DIR/report/missing_treatment_summary.json` 和 `.md`。

# 性能优化版使用入口

本包的 30k TEMP Colab Notebook 使用 `papers_master_temp.csv`、独立 `dblp_cache_30000_temp` 和 `recovery_output_30000_temp`。运行时把 XML、DTD、SQLite 与 Fetch/DBLP 输入副本放到 `/content/GP_DBLP_Recovery_Work_30000_TEMP`；最终输出与冷 Cache Snapshot 持久保存到 Drive。请先阅读 [COLAB_PERFORMANCE.md](COLAB_PERFORMANCE.md)，再运行新版 `Colab_Deployment.ipynb`。不要用 Drive 路径配置直接启动 main.py 来代替新 Notebook 的 launch 单元。

完整身份算法、Focal 排除和四特征定义保持不变。优化重点为 I/O、查询批处理和安全复用，不能承诺真实 DBLP 固定一小时完成。

# GP_DBLP_Author_Recovery

基于第三版 Local 项目继续改造，直接读取 Fetch 原始输出，不依赖任何 Recovery 历史输出。保留完整候选扩展、流式 XML 全量目标检索、SQLite 候选历史、合作网络、序列对齐、隐藏 PID 回测、磁盘去重和严格特征计算，并整合此前的目标论文身份恢复和合并姓名拆分。

## 运行

Python 3.10+，Linux / Colab。在项目目录执行：

```bash
python -m pip install -r requirements.txt
python main.py --config config.local.example.json
```

先修改配置里的路径。相对路径相对于启动命令的工作目录。程序不下载 DBLP、不访问 DBLP API 或其他数据源。Semantic Scholar 仅生成后续待处理队列。

`--stage prepare` 仅 Bootstrap、索引校验/初始化及候选扩展；`--stage scan` 进一步完成 XML 扫描与历史缓存；默认 `--stage all` 完成身份、历史、四特征和导出。PREPARED/BLOCKED 不是恢复成功。

## 必须提前存在的文件

FETCH_DATA_ROOT 下：

- `data/processed/papers_master_temp.csv`（本次 30k TEMP 部署配置）：arxiv_id、title_v1、submission_date_v1，作者期望数量等原始字段。
- `data/processed/authors_arxiv_v1.csv`：arxiv_id、author_position、author_name_raw。

完整执行必须有配置指定的本地 DBLP XML（支持 gzip）及配套 DTD。XML 可以位于 Drive；建议手动复制 XML 和 DTD 到 Colab 本地工作目录以提高扫描速度。程序不会移动、覆盖或下载它们。

`data/processed/dblp_author_features_v3_1_1/` 下作者明细、团队特征、候选/冲突/优先级审计和 metadata 为可选输入，缺失记录警告。运行时检查真实列名和键，记录 schema 审计。大型辅助 CSV 采用分块/窄列读取，只保留当前作者 roster 所需的身份字段与冲突审计，不把数 GB 宽表整体转换为 Python 字典。无需 recovered_dblp_author_features_by_author.csv 或 papers_master_recovery_preview.csv。

## 配置与 Cache 生命周期

| 配置 | 含义 |
|---|---|
| FETCH_DATA_ROOT | 只读 Fetch 数据根目录 |
| DBLP_XML_PATH | 已有完整 XML 路径 |
| CACHE_DIR | 独立持久 Cache 根目录，不得位于 Fetch 数据内 |
| PROFILE_INDEX_PATH | null 时为 CACHE_DIR/index.sqlite |
| DTD_PATH | null 时优先 CACHE_DIR/dblp.dtd，否则 XML 同目录 dblp.dtd |
| DERIVED_CACHE_DIR | null 时为 CACHE_DIR/derived |
| OUTPUT_DIR | 新结果目录，与 Fetch 和 Cache 分离 |

首次没有 index.sqlite 时，使用 XML/DTD 在本地流式构建 Profile/Alias SQLite 索引。临时数据库保留检查点；完整解析、完整性及源指纹校验通过后原子发布。已有索引以只读方式打开并校验，绝不自动覆盖。Fetch pass1/pass2 pickle 不是这个索引，不能替代它。

候选历史与磁盘去重数据库自动写到 DERIVED_CACHE_DIR/candidate_history.sqlite。支持检查点恢复；恢复时会重新流式读取到检查点而不重复写入。完整 XML 仍需存在，部分历史缓存不能替代它。后续运行复用同一快照的 Profile 索引和同一输入/代码/配置签名的派生 Cache。换数据集或改变算法时使用新的 DERIVED_CACHE_DIR 子目录；快照不匹配时明确停止，不静默覆盖旧 Cache。首次构建、扫描和源 SHA256 校验均有实际 I/O 成本；全量 DBLP 性能以实际机器为准。

所有结果、日志写 OUTPUT_DIR；索引和派生 Cache 是唯一写入 CACHE_DIR 的文件。输入、XML、DTD、已有 Profile 索引保持只读，运行前后验证源文件哈希。

## 保守恢复规则

作者出现记录使用 arxiv_id + author_position，保留原姓名、原 PID 和完整 lineage，不按姓名去重。Fetch 先前确认记录重新核对当前 Profile、Alias、Redirect 与冲突；新增姓名候选不能仅凭模糊分数或 Initial 自动确认。明确反证输出审计；准备阶段未确认不是最终消歧结论。

目标论文标识、可靠作者列表、保守编辑修复和序列对齐支持 PID mismatch、歧义及合并作者恢复。多候选和证据不足继续 UNRESOLVED；网络回测使用先前确认身份作为伪标签，不是外部真值。回测支持不足或精度下界未达标时关闭网络自动确认。

P0 修复：目标身份可靠性与 Focal 排除完全独立。所有潜在目标命中、Fetch 审计中的候选记录，以及由强标识关联的版本均排除自身历史合作证据。缺少年份/未解析 PID 的目标也参与版本排除。独立 focal 审计和回归测试检查遗漏、合作证据泄漏及时间边界。这种保守排除可能减少历史记录，证据逐条可追溯。

严格保留 publication_year < t0_year；同年、未来和 Focal Work 均排除。沿用原规则的 DOI/arXiv/作品去重。完整扫描下 DBLP 可观察历史为零可以记 Count=0，但不等于证明现实世界从未发表；身份未确认或查询失败不填零。在恢复层，没有首发年份的 Academic Age 保持 Missing；最终 Treatment 仅对完整扫描认证的零历史作者将 Academic Age 设为 0，首次发表年份仍保持 Missing。Age 使用原定义 t0_year - first_year。四个正式团队均值/最大值要求完整有效的作者覆盖；Partial 独立标记，不能冒充完整值。原始非 DBLP 字段和 Y1/Y2/Y3 不变。

## 输出

| 子目录 | 主要文件 |
|---|---|
| diagnosis | suspicious_author_names.csv |
| matching | bootstrap_author_occurrences.csv、fetch_identity_revalidation.csv、final_author_identity_mapping.csv、final_unresolved_authors.csv、identity_conflicts.csv、candidate_history_evidence.csv、candidate_coauthor_evidence.csv，以及候选/对齐/回测审计 |
| history | final_author_publication_history.csv |
| features | final_author_features.csv、final_team_features.csv、final_papers_master_preview.csv |
| fallback | semantic_scholar_fallback_queue.csv |
| report | fetch_schema_audit.json、focal 排除审计、temporal_integrity_audit.json、final_recovery_summary.json、final_recovery_report.md |
| logs | 运行日志 |

准备或阻塞阶段只生成已经完成阶段的输出，不伪造 final 身份或特征文件。此前真实数据准备结果的结论保留在 validation/previous_delivery；优化版不重复打包大体积旧候选 CSV。

## 模块输入、输出与职责

| 模块 | 输入 → 输出 / 功能 |
|---|---|
| config.py / main.py | JSON 配置 → 路径保护、阶段启动 |
| bootstrap.py | Fetch CSV/JSON → 统一出现记录、schema/源审计、初始身份复核 |
| index_init.py / profile.py | XML/DTD 或现有 SQLite → Profile/Alias/Redirect 只读查询 |
| names.py / rules.py | 原姓名、出版记录 → Fetch 兼容规范化、标识与作品规则 |
| search.py | 论文、候选 PID、XML → 全量目标与候选历史 SQLite |
| database.py / dedup.py | 流式记录 → 断点数据库、磁盘作品等价类 |
| name_repair.py / evidence.py | Alias、可靠目标作者列表 → 名字候选、拆分与身份审计 |
| network.py | 严格历史合作记录 → 候选证据、隐藏 PID 回测和接受阈值 |
| alignment.py | 原作者序列、目标作者序列 → 保守序列对应及 lineage |
| focal.py | 目标、作品、合作证据 → 独立排除与时间审计 |
| history.py / features.py | 确认身份、严格历史 → 作者历史、四个团队特征 |
| workflow.py | 上述阶段 → 最终映射、预览、Fallback、汇总报告 |
| io.py / progress.py / common.py | 流式数据与状态 → 原子导出、真实计数/速度/日志 |

## 测试与真实验证

```bash
python -m unittest discover -s tests -v
python tests/stress.py --papers 30000 --root /tmp/dblp_recovery_stress --fresh-index
```

本次优化验证见 `validation/PERFORMANCE_REPORT.md`；此前验证记录单独保留在 `validation/previous_delivery`，不代表本次重新执行。真实 Fetch 599 篇/2,361 作者出现记录已执行 Bootstrap 和候选扩展，但提供的 cache.zip 没有完整 XML，因此真实完整恢复明确阻塞。不能把合成结果当作真实覆盖提升。

Colab 使用 `Colab_Deployment.ipynb`；Notebook 只负责挂载、配置、部署、安装和启动，不包含身份恢复业务算法。

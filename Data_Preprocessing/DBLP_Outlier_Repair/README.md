# DBLP Outlier Repair v3.1

用已有 DBLP 数据核验四个团队特征。Notebook 只运行 Audit Only，原始主表作为只读输入，changes.csv 记录建议值。

## Colab

打开 Colab_DBLP_Outlier_Repair_v3_1.ipynb。默认运行内嵌测试；要在 Drive 的真实数据上审计，将 RUN_FULL 改为 True 并顺序运行全部单元。核对 input_csv、recovery_root、xml_path、dtd_path；主表也可以上传原 CSV 或 data_quality_check.zip。保留 v3 使用的 cache_root 以复用 Crossref 响应及兼容的 DBLP 索引。baseline_v3_run 指向本次 v3 输出目录，使用相同主表及候选集合时会生成 paper_comparison_v3.csv 和 comparison_v3.json。

## 本地

使用 Python 3.10+，在项目目录执行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py --config config.local.example.json --audit-only
```

先在配置中填好本地路径。中断后重跑相同命令；若需要再次核验未决作者，可添加 --refresh-reviews。新版本独立保存作者结论。相同来源、完整且匹配范围的历史索引可以继续使用。CLI 默认 Audit Only；正式写入新结果表需要另外显式指定 --apply。

## 核验及输出

- 姓名按 Recovery 的论文 ID、作者位置及现有 PID 关联。保存 Recovery 原名与修复名、有序主表和可用团队名单的署名、精确 PID 的 XML 档案别名及来源。当前团队 CSV 只有人数与统计量，可用来校验长度。仅凭 PID 档案取得的姓名和名单冲突均受写入保护。
- audit_queue.csv 按已有 ORCID、署名和标题冲突及异常履历排列核验顺序。全部候选都会进入队列。符合绑定条件的 v3 日志提供调查线索。成功响应和已记录的 404 来自原 Crossref 缓存。外部查询继续使用 v3 的预算、去重及出版商身份门槛。
- author_verdicts.csv 记录姓名、位置、PID、来源、判断和历史覆盖数。work_evidence.csv 覆盖轻量筛查及深入核验两个分支，每一版本可按 occurrence_id、work_id、history_pid、record_key、record_position 追踪。作品数按作者出现位置单独去重。服务器预算延期、姓名或 DOI 缺失、缺少身份锚点和证据冲突可同时出现在 audit_flags_json。
- 如果提供原始 recovery_history_csv，recovery_history_ledger.csv 另外记录候选作者在原导出中的行号、work_id（如有）及 INCLUDED、FOCAL_WORK_EXCLUDED、SAME_OR_POST_T0_YEAR 等原始决定。原导出 work_id 与重新构建的筛选索引 work_id 按各自来源解释。
- summary.json 汇总作者、作品、版本明细及建议/实际修改数。progress.json 和 runtime_estimate.json 沿用 v3 的计时和 ETA 计算。完整记录明细及全局风险排序会增加 I/O，实际时间以运行日志为准。
- validation/ 包含真实导出的只读姓名回放表、计数、合成 Audit Only 明细及测试日志。测试报告说明了哪些结果已经在真实数据上验证。

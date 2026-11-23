# 最小改动：最终 Missing Treatment

## 规则

| 身份及历史证据 | 最终 Count | 最终 Academic Age |
|---|---|---|
| VERIFIED、有 PID，完整恢复层认证 ZERO_PRIOR_WORKS_IN_DBLP_SNAPSHOT，原 Count=0、首次年份为空 | 0 | 0 |
| VERIFIED，已有有效历史 | 保留原值 | 保留原值 |
| 身份未确认 | Missing | Missing |
| MISSING_YEAR / 年份无法验证 / 扫描未完成 / Alias 覆盖有歧义 | Missing | Missing |
| NO_OBSERVED_RECORDS（未找到记录，恢复层未认证可用零历史） | Missing | Missing |

零历史是完整 DBLP 快照内、严格 t0 之前、排除 Focal 后的合格历史为零，不代表现实世界绝对没有发表。完整恢复层认证的零值才参与 Treatment；不根据缺失值自行猜零。首次发表年份不填 0，原 history_status 和证据字段保留。

团队继续调用原 aggregate；一个作者身份未确认、有效历史不足、缺少作者记录或 PID 重复时，不以 partial 值代替正式四特征。存在完整身份但 Academic Age 不可验证的情况时，继续按原有每项特征完整性规则分别判断。没有团队均值插补。

## 已完成 Recovery：推荐只运行 Treatment

```bash
python main.py --config config.runtime.json --stage treatment
```

也可直接：

```bash
python missing_treatment.py --config config.runtime.json
```

Colab Notebook 新增最后一个独立单元：部署代码后可直接运行，跳过前面的 launch。它采用原 Drive 配置，既不复制 XML，也不恢复或打开 SQLite。Config 中 XML/索引路径可以仍指向原位置，Treatment 不要求它们实际存在。

必须已有完整成功的 final_recovery_summary.json（COMPLETE 或 COMPLETE_WITH_REVIEW_ITEMS、full_xml_scan_complete=true、source_files_unchanged=true）、Temporal/Focal 审计、source_manifest、final_author_features.csv、final_team_features.csv，以及与恢复时完全一致的原始 Fetch 主表/作者 CSV。只完成 prepare/scan 或上次运行失败时，不能用 Treatment 冒充完整恢复；请先完成 all。

尚未完成 Recovery：照旧通过 Optimized Notebook 的 launch 执行 all；本次 main.py 在原工作流成功返回后追加 Treatment。新步骤失败不会修改扫描缓存，修正输入后可单独重跑 treatment。

## 新输出

全部位于 OUTPUT_DIR：

- features/papers_master_dblp_imputed.csv：原 Master 全部行、顺序、唯一标识、列顺序和非四特征字段保留，只更新四特征。原表没有四特征列时才追加。
- features/treated_author_features.csv：最终 Count/Age、处理前值、Treatment 原因和策略，保留身份与历史证据。
- features/treated_team_features.csv：按新 Treatment 计算的四特征、完整性与 partial 状态。
- report/missing_treatment_summary.json / .md：零历史赋值作者出现记录数、唯一 PID 数、未确认作者数、历史缺失原因、最终有效/Missing 数、保留/删除论文数、Treatment 前后 Coverage。

原始 Master、原 final_author_features/final_team_features、原 preview、历史证据、Temporal/Focal 审计与 final_recovery_summary 均不覆盖。原 final 文件是 Treatment 前结果；新 treated 文件和新 Master 是 Treatment 后结果。新 Master 中其他 dblp_status 等旧字段按要求原样保留，可能不代表新 Treatment 完整性；最新状态看 treated_team_features.csv，不额外改写主表字段。

Treatment 前 Coverage 指同一恢复结果在新 Treatment 前的严格团队汇总；JSON 另列 original_master_coverage，避免混淆原 Fetch 基线与 Treatment 增益。

## 缓存保证及改动文件

- 新增 missing_treatment.py：只做末端处理、严格汇总、新表写入与结果验证。
- 修改 main.py：新增 --stage treatment；all 后自动追加 Treatment。
- 修改 Colab_Deployment.ipynb / README.md：新增独立运行说明；不改变 launch 或缓存步骤。
- 新增测试、MISSING_TREATMENT.md 与验证报告。

recovery 目录所有 Python 文件、config.py、common.py、colab_runtime.py、cache_compatibility.json 与 Optimized 版逐字节相同。因此原有算法哈希、缓存签名、迁移与 checkpoint 行为不变。没有新增 Cache，也不用把任何数据库移到新位置。

## 如何检查新主表

程序自动校验原始主表和作者 CSV 的源 SHA256、成功运行状态、Temporal/Focal 全零违规、作者/团队/主表 cohort 对齐、唯一键、期望作者数、作者与已有团队四特征一致性。

写出后重新读取 CSV，验证行顺序、原始非四特征字段和四特征数值。失败明确报错，不静默丢行。原始 599 篇输入应显示 retained_paper_count=599、excluded_paper_count=0；程序不硬编码 599，也支持其他完整数据集。

运行后查看 report/missing_treatment_summary.md。若无可认证的零历史作者，零填充数可以为 0；这不是失败。

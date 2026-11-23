# Missing Treatment 实际验证

本次仅最终 Treatment 更新，2026-11-07（香港）。

## 修改范围

新增 missing_treatment.py；main.py 增加 treatment 阶段及 all 成功后末端调用。Notebook 仅追加独立 Treatment 单元。README 与 MISSING_TREATMENT.md 更新说明，新增两份测试脚本。

recovery/*.py、config.py、common.py、colab_runtime.py、cache_compatibility.json、三个配置和 requirements 与 Optimized 版逐字节相同，见 unchanged_core_manifest.json。未改变扫描、匹配、历史计算、Focal、Temporal、回测、confidence、Cache schema、输入签名或 checkpoint。最终零值约定只在输出层施加。

## 自动化验证

70 项测试通过（原 57 项 + 新增 13 项），4.806 秒。覆盖：已认证零历史 Count/Age=0；未知身份即使存在数字仍 Missing；缺失年份/未扫描/无观察记录/歧义状态不误填零；矛盾零历史字段不填零；零与正值团队 Mean/Max；全零团队；一位未确认导致完整团队 Missing；删除 XML 且禁止 sqlite3.connect 仍成功；Cache 前后哈希不变；Master 原字段 round trip；重复 Treatment 幂等；prepare-only、源主表变化、审计失败、作者与团队导出不一致均拒绝处理。

## 599 篇合成集成验证（不是用户真实数据）

随机打乱原论文顺序，保留前导零标签、NA 字符串、含逗号和换行字段。先使用未改变的 Recovery 工作流生成完整结果，再移走 XML，单独运行 Treatment。

- 输入 599 篇，输出保留 599 篇，删除 0。
- 150 条作者出现记录被认证为零历史，Count=0、Academic Age=0。
- 150 条作者身份未确认，保持 Missing。
- 149 条作者历史存在缺失年份，保持 Missing。
- Publication Count Mean/Max：Treatment 前后有效均 300；最终 Missing 均 299。
- Academic Age Mean/Max：有效从 150 增至 300；最终 Missing 均 299；Coverage 从 25.04% 增至 50.08%。
- 原 Master、Profile SQLite、派生 SQLite、Temporal 审计前后 SHA256 不变。
- 该次 Treatment 约 2.633 秒；最终 main.py --stage treatment 在 XML 不存在时也已实际执行成功。

新 Master 仅替换四特征，其余字段、全部行、唯一标识及原顺序逐字段验证。例子输出单独放在 synthetic599/，不要作为真实论文结果使用。

## 真实数据限制

当前未取得用户在 Colab 最近运行产生的完整 final_author_features.csv、final_team_features.csv 和成功运行报告，无法报告真实 599 篇的零填充作者数或真实 Coverage 增益。部署后运行 --stage treatment 自动生成真实 summary；不会为生成报告而重新扫描 XML。

完整数据库与 Cache 不需新增或迁移。原 final_author_features/final_team_features、原 preview 和 final_recovery_summary 是 Treatment 前证据，保留不覆盖；新 treated_*、papers_master_dblp_imputed.csv 和 missing_treatment_summary 是最终 Treatment 结果。

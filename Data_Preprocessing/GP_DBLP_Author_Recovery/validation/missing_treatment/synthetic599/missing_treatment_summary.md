# Missing Treatment Summary

保留论文：599/599；删除：0。
确认零历史作者出现记录：150；Academic Age 新填 0：150。
身份未确认作者出现记录：150。

| 特征 | Treatment 前有效 | Treatment 后有效 | 最终 Missing | Coverage 前→后 |
|---|---:|---:|---:|---|
| dblp_team_mean_publication_count_pre_t0 | 300 | 300 | 299 | 50.08% → 50.08% |
| dblp_team_max_publication_count_pre_t0 | 300 | 300 | 299 | 50.08% → 50.08% |
| dblp_team_mean_academic_age_pre_t0 | 150 | 300 | 299 | 25.04% → 50.08% |
| dblp_team_max_academic_age_pre_t0 | 150 | 300 | 299 | 25.04% → 50.08% |

仅更新原始主表四个 DBLP 特征；所有行、顺序、其他字段保留。原有 dblp_status 等字段为源数据元数据，故不改写；最新完整性状态见 treated_team_features.csv。
未访问 XML/SQLite；Temporal/Focal 审计原件不修改；年份不明、扫描不完整或未观察到记录但不能认证零历史的作者保持 Missing。
首次发表年份不填 0。完整零历史的 Academic Age=0 是最终 Treatment 约定。

# 实际验证报告 — 2026-11-06

## 交付范围

直接修改第三版 Local 代码的延续副本；移除 Recovery 输出入口，新增 Fetch Bootstrap 和缺失 Profile 索引本地初始化。复用此前的保守姓名分段/拆分算法；第三版 rules.py 与原文件逐字节一致。原 Fetch、原项目压缩包、原输入及原 SQLite 未覆盖。没有下载 DBLP，没有调用 DBLP API、arXiv、Semantic Scholar 或 OpenReview。

## 测试结果

| 验证 | 实际结果 |
|---|---|
| 第一版原代码历史基线 | 15 tests OK |
| 第二版原代码历史基线 | 31 tests OK |
| 第三版原代码历史基线 | 26 tests OK |
| 最终统一架构回归 | 45 tests OK，2.208 秒 |
| 首次运行，无任何 Recovery 输出 | 通过 |
| Cache 缺失本地建索引/中断恢复 | 通过 |
| 已有 Cache 只读复用/错误 Cache 不覆盖 | 通过 |
| Focal 不可靠目标、无年份目标、强标识版本排除 | 通过 |
| 原始 schema、缺失可选输入、缺失作者、t0 冲突 | 通过 |
| PID mismatch、歧义、姓名修复、Redirect | 通过 |
| 严格四特征、网络回测、序列对齐、磁盘去重 | 通过 |

历史基线在各版原代码上独立执行，最终 45 项包含迁移后的接口和新架构回归；没有把旧 CLI 接口通过宣称为新架构通过。历史测试因环境无 requests/tqdm 使用禁用联网替身及第三版标准库进度适配器；最终代码直接运行，pandas/lxml 可用，tqdm 缺失时使用内置进度日志。附原始测试日志。

## 30,000 篇合成端到端验证

空 Profile Cache 起步，60,000 条作者出现记录、60,000 个 Profile、60,000 条发表记录，完整流式解析 120,000 个 XML 记录。176.678 秒完成；整个进程峰值 RSS 1,035.62 MiB（含合成 fixture 构造）；派生 SQLite 306,933,760 字节；10,000 次索引查询 0.053 秒。

最终 60,000 条作者确认、0 未解决，四项特征各 30,000 个有效值。时间边界与 Focal 泄漏计数均为 0。网络回测支持不足，自动网络确认正确关闭，目标论文证据仍完成恢复。详见 synthetic_30000_benchmark.json。

这是唯一姓名、每篇两名作者的合成数据，不能外推全量 DBLP 的扫描用时、重名候选爆炸场景或真实身份准确率。未宣称全量真实数据库性能测试通过。

## 真实 Fetch 数据验证

读取原始 Fetch 599 篇主表、2,361 条作者出现记录及实际 DBLP 明细/审计。完全不读取第一/第二版 Recovery 输出。复用 cache.zip 内已有 Profile/Alias SQLite，运行前后源哈希一致。

Fetch 原声称确认 1,768 条；Bootstrap 重新核对后暂时确认 1,767 条，594 条进入候选待验证。1 条先前确认记录有姓名不一致：1904.12631 第 2 作者 Jule-Raymond Tapamo，PID homepages/67/3647，原方法为 Initial+Surname。保留源身份及复核原因，等待目标论文证据，不能当作已证明原匹配错误，也不能在缺 XML 时宣称已完成最终降级。

展开候选/拒绝审计 1,084,900 行，合格候选涉及 22,969 个不同 PID。prepare 运行 66.575 秒，峰值 RSS 349.11 MiB。随后以最终代码执行 --stage all：57.693 秒后状态 BLOCKED_MISSING_INPUT，原因是完整 DBLP_XML_PATH 不存在；源文件未变、网络尝试为 0。具体原始结果见 real_full_run_blocked.json 和日志。

提供的 cache.zip 含 index.sqlite、DTD 和部分历史缓存，没有完整原始 XML。部分 Cache 不替代完整扫描。因此真实数据的最终身份数、恢复覆盖提升、历史与四特征尚未验证；未生成虚假的真实 final mapping/features。

原始主表四特征有效值数量：

```json
{
  "dblp_team_mean_publication_count_pre_t0": 295,
  "dblp_team_max_publication_count_pre_t0": 295,
  "dblp_team_mean_academic_age_pre_t0": 197,
  "dblp_team_max_academic_age_pre_t0": 197
}
```

## 首次运行与复用

必须提供原始 Fetch 主表/作者 CSV、完整 XML 和 DTD。DBLP 明细和审计为可选，缺失有警告。首次可没有 index.sqlite：程序流式构建临时 Profile/Alias SQLite，完整校验后原子发布；同时自动创建 derived/candidate_history.sqlite。已有合格索引只读复用，不加载 Fetch pickle，不覆盖旧 Cache。后续同快照同输入复用；输入、算法或语义配置改变需使用新的派生子目录。代码、配置、Notebook 和测试均在包内。

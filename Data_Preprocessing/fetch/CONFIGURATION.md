# 配置字典

配置仅从 `control/config.py` 定义，JSON 覆盖部分字段即可。JSON 不支持注释。

| 字段 | 默认值 | 说明 |
|---|---|---|
| `storage_mode` | `"local"` | local 或 colab_drive；后者才挂载 Drive。 |
| `data_root` | `"./GP_Research_Papers"` | 数据根目录；建议绝对路径。 |
| `start_year` | `2017` | 起始年份，包含。 |
| `end_year` | `2022` | 结束年份，包含。 |
| `target_per_year` | `100` | 每年最多抽样数量；不减少年度候选池扫描。 |
| `random_seed` | `3544` | 年度随机抽样种子。 |
| `citation_window_years` | `3` | 日历年整数；默认 3，改变时两种引用标签同时改变。 |
| `cache_policy` | `"reuse_valid"` | reuse_valid / refresh_source / refresh_all。 |
| `refresh_sources` | `[]` | refresh_source 的目标列表；arxiv / dblp / semantic_scholar / openreview。 |
| `resume_enabled` | `true` | 是否直接复用成功阶段的最终表缓存；false 不会禁止 raw 缓存。 |
| `max_retries` | `7` | 初次请求之后的最大重试次数，预算耗尽持久化失败。 |
| `request_timeout` | `120` | 每次 requests 请求超时秒数（连接/读取超时，不是整个阶段时限）。 |
| `wait_for_reconnection` | `true` | 连接中断时是否在重试预算内等待。 |
| `reconnection_delay` | `30` | 连接中断每次等待的最小秒数。 |
| `max_retry_delay` | `120` | 无 Retry-After 时指数退避上限；不截断服务器指定等待时间。 |
| `log_level` | `"INFO"` | 终端级别；完整细节始终保存 DEBUG 文件日志。 |
| `source_intervals` | `{"arxiv": 3.1, "dblp": 1.0, "semantic_scholar": 1.1, "openreview": 1.0}` | 各来源请求最小间隔秒数，429 时自动增加。 |
| `arxiv_page_size` | `2000` | 年度候选 API 单页大小，最多 2000。 |
| `arxiv_metadata_batch_size` | `100` | 每批 v1 ID 数量。 |
| `s2_batch_size` | `500` | 元数据 batch 大小，最多 500。 |
| `s2_page_size` | `1000` | 引用边每页大小，最多 1000。 |
| `s2_max_pages` | `10` | 每篇论文的分页上限；达到上限标记未完成。 |
| `s2_max_citations` | `9999` | 沿用 notebook 的 9999 条 API 访问预算，超限不冒充完整标签。 |
| `openreview_workers` | `1` | 检索 worker 数；默认 1，仍受统一来源限速锁约束。 |
| `openreview_batch_size` | `20` | 标题/论坛批次大小。 |
| `openreview_page_size` | `1000` | 检索页大小。 |
| `openreview_max_pages` | `30` | 检索分页预算，超限标记失败。 |
| `dblp_xml_path` | `null` | 已有 dblp.xml 或 dblp.xml.gz；null 时先搜索数据目录，再下载。 |
| `sources` | `["dblp", "semantic_scholar", "openreview"]` | 启用的下游来源列表；arXiv 是必需上游。 |
| `seed_csv` | `null` | 可选显式 ID 清单；填写后跳过年度候选抽样，使用清单全部行。 |
| `minimum_free_gb` | `1.0` | 启动时最小磁盘剩余容量 GB；不是首次 DBLP 下载/扫描容量估算。 |
| `user_agent` | `"GP-Research-Data-Collection/1.0 (academic research)"` | 公共请求标识，不存放密钥。 |
| `drive_mount_point` | `"/content/drive"` | 仅 Colab 存储模式使用。 |

引用窗口仅接受正整数日历年；不把 3 年近似为 1095 天。更改窗口后输出兼容列名仍是 `_3y`，以 metadata 的 citation_window_years 为准。

示例：只更新 OpenReview（其余参数沿用默认）：

```json
{"cache_policy":"refresh_source", "refresh_sources":["openreview"]}
```

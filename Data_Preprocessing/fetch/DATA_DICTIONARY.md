# 数据文件与字段字典

所有路径相对 `data/processed/`，另行注明的 raw/intermediate 除外。CSV 读取务必 `dtype={"arxiv_id": str}`；例如 `1701.00010` 不能让浮点转换丢失末尾 0。

## 表粒度与主键

| 文件 | 一行代表 / 主键 | 用途 |
|---|---|---|
| arxiv_seed_ids.csv | 候选论文 / arxiv_id | 年度随机样本（显式 seed_csv 时使用外部清单） |
| arxiv_monthly_candidate_counts.csv | 年月 | 候选池完整性审计 |
| papers_arxiv_v1.csv | 论文 / arxiv_id | 主表基底 |
| authors_arxiv_v1.csv | 论文署名作者 / arxiv_id + author_position | 作者明细，不能直接左连接主表 |
| arxiv_v1_excluded_non_cs.csv | 被排除论文 / arxiv_id | v1 不属于 CS |
| arxiv_v1_year_mismatch.csv | 年份冲突论文 / arxiv_id | 查询年份与 v1 年份审计 |
| dblp_author_features_v3_1_1/dblp_author_features.csv | 论文 / arxiv_id | DBLP 论文级特征 |
| dblp_author_features_v3_1_1/dblp_author_features_by_author.csv | 目标论文作者 / arxiv_id + author_position | 身份与历史明细 |
| dblp_author_features_v3_1_1/dblp_author_match_audit.csv | 作者匹配审计记录 | 原 V3.1.1 匹配审计 |
| dblp_author_features_v3_1_1/dblp_candidate_match_audit.csv | 候选匹配 | 选取/拒绝证据 |
| dblp_author_features_v3_1_1/dblp_identity_conflicts.csv | 作者身份冲突 | 未解决身份 |
| dblp_author_features_v3_1_1/dblp_identity_priority_audit.csv | 身份优先级审计 | 强弱证据冲突 |
| dblp_author_features_v3_1_1/dblp_temporal_exclusion_audit.csv | 时间排除审计 | focal work 与 t0 |
| dblp_author_features_v3_1_1/dblp_source_metadata.json | 文件级元数据 | XML 指纹、版本、索引命中、计数 |
| semantic_scholar_public/s2_metadata.csv | 论文 / arxiv_id | 包含未匹配论文的 S2 元数据 |
| semantic_scholar_public/s2_metadata_audit.csv | 论文 / arxiv_id | 匹配与请求状态 |
| semantic_scholar_public/semantic_scholar_citation_3y.csv | 论文 / arxiv_id | C3、IC3、状态和覆盖审计 |
| semantic_scholar_public/semantic_scholar_influential_citation_3y.csv | 论文 / arxiv_id | 上表的兼容副本 |
| semantic_scholar_public/semantic_scholar_citation_3y_progress.csv | 已处理论文 / arxiv_id | 每 10 篇更新的辅助进度；恢复以 raw 和 summary 为准 |
| semantic_scholar_public/semantic_scholar_citation_edges_3y.csv | 唯一引用关系 / arxiv_id + citing_paper_id | 保留日期与窗口状态，只有 IN_WINDOW 计入标签 |
| semantic_scholar_papers.csv | 已匹配论文 / arxiv_id | 旧接口兼容输出；最终合并仍使用全部目标论文 |
| openreview_ratings/openreview_submissions.csv | 论文/来源/投稿 | 匹配到的公开投稿 |
| openreview_ratings/openreview_reviews.csv | 论文关联的评审 Note | 原始评审内容与来源 |
| openreview_ratings/openreview_reviewer_ratings.csv | 论文/论坛/评审 Note/评分字段 | reviewer-level 原始评分，不能直接合并主表 |
| openreview_ratings/openreview_paper_ratings.csv | 论文 / arxiv_id | paper-level 均分 |
| openreview_ratings/openreview_missing_ratings.csv | 缺评分评审 | 缺失原因 |
| openreview_ratings/openreview_rating_audit.csv | 论文 / arxiv_id | 公开检索范围、状态、数量 |
| openreview_ratings/openreview_search_audit.csv | 搜索批次 | 网络与检索审计 |
| openreview_ratings/openreview_forum_notes.jsonl | host/forum/原始 Note | 保留实际获取到的 Notes |
| openreview_ratings/openreview_errors.json | 错误数组 | 来源内详细错误 |
| papers_master_raw_stage.csv / .jsonl | 论文 / arxiv_id | 最终主表，行数等于保留的 arXiv v1 论文数 |
| papers_master_metadata.json | 文件级元数据 | 字段、窗口、来源和生成时间 |
| coverage.csv | 主表字段 | total、non_missing、coverage |
| quality_report.json | 一次质量检查 | 完整性、未完成项、论文状态计数 |
| disk_usage.json | 文件清单 | bytes、free_bytes、files |

某些全空审计表沿用原 notebook 的空表格式，可能没有表头；读取时先检查文件大小。它们不是主表输入。

## 主表字段

以下是所有来源已执行时的列集合；某个来源整体失败/未启用时，该来源特征列可能不存在，但统一 `*_status` 列始终存在。NA 保留缺失，不能 blanket fillna(0)。

| 字段 | 定义 |
|---|---|
| `paper_id` | arXiv ID 的兼容副本。 |
| `arxiv_id` | 唯一主键，不含 v 后缀；必须按字符串读取。 |
| `arxiv_version` | 元数据版本，固定 v1。 |
| `submission_date_v1` | 首次提交时间 t0（UTC）。 |
| `v1_updated_date` | v1 返回的更新时间，不替代 t0。 |
| `title_v1` | v1 标题。 |
| `abstract_v1` | v1 摘要。 |
| `primary_category_v1` | v1 主分类。 |
| `categories_v1_json` | 全部 v1 分类 JSON 数组。 |
| `cs_categories_v1_json` | cs.* 分类子集 JSON 数组。 |
| `authors_v1_json` | v1 署名作者及可用机构信息 JSON 数组。 |
| `author_count_v1` | v1 署名作者数。 |
| `comment_v1` | arXiv comment。 |
| `journal_ref_v1` | arXiv journal reference。 |
| `doi_v1` | v1 API 返回 DOI；缺失不推断。 |
| `seed_query_year` | 候选采样所属年份。 |
| `submission_year` | 从 v1 t0 提取的年份。 |
| `t0_date` | DBLP 使用的 t0 日期副本。 |
| `t0_year` | DBLP 历史特征的年度截断。 |
| `dblp_match_status` | 论文候选匹配详细状态，未验证匹配不算匹配。 |
| `dblp_match_method` | 论文匹配证据类型。 |
| `dblp_record_key` | 选定 DBLP 记录键。 |
| `dblp_record_type` | article/inproceedings 等记录类型。 |
| `dblp_record_year` | 选定记录年份。 |
| `dblp_focal_work_id` | 合并版本后的当前研究工作 ID。 |
| `dblp_focal_work_earliest_year` | 当前工作最早 DBLP 年份。 |
| `dblp_temporal_status` | DBLP 时间顺序与排除审计状态。 |
| `dblp_previously_published_before_arxiv_v1` | 当前工作是否在 arXiv v1 前已发表。 |
| `dblp_focal_work_excluded_from_author_history` | 当前工作是否从作者历史中排除。 |
| `dblp_focal_work_excluded_author_count` | 执行当前工作排除的作者数。 |
| `dblp_candidate_record_count` | 候选 DBLP 记录数。 |
| `dblp_duplicate_version_group_size` | 选定工作所合并的版本数。 |
| `arxiv_author_count` | DBLP 输入的 arXiv 作者数。 |
| `dblp_author_match_count` | 确认匹配的作者数量。 |
| `dblp_author_identity_complete` | 所有目标作者身份是否完整。 |
| `dblp_high_confidence_author_count` | 高置信度作者数。 |
| `dblp_medium_confidence_author_count` | 中置信度作者数。 |
| `dblp_low_confidence_author_count` | 低置信度作者数。 |
| `dblp_unresolved_author_count` | 身份未确定作者数。 |
| `dblp_identity_conflict_author_count` | 存在身份冲突的作者数。 |
| `dblp_all_authors_high_confidence` | 是否所有作者高置信度。 |
| `dblp_all_authors_high_or_medium_confidence` | 是否所有作者至少中置信度。 |
| `dblp_total_author_publication_count_pre_t0` | 完整身份/历史条件下，作者 t0 前去重发表数之和；否则缺失。 |
| `dblp_total_academic_age_pre_t0` | 作者 t0 前学术年龄之和；不可计算则缺失。 |
| `dblp_publication_count_available` | 发表数特征是否可用。 |
| `dblp_academic_age_available` | 学术年龄特征是否可用。 |
| `dblp_academic_age_missing_author_count` | 缺失学术年龄的作者数。 |
| `_dblp_row_present` | 该来源左连接时是否实际提供了这一篇论文的行；不等于特征完整。 |
| `dblp_team_author_row_count` | 作者明细聚合后的团队署名行数。 |
| `dblp_team_author_matched_count` | 团队确认匹配作者数。 |
| `dblp_team_all_authors_matched` | 团队是否全部匹配。 |
| `dblp_team_high_confidence_author_count` | 团队高置信度作者数。 |
| `dblp_team_medium_confidence_author_count` | 团队中置信度作者数。 |
| `dblp_team_publication_features_status` | 团队发表特征完整性状态。 |
| `dblp_team_mean_publication_count_pre_t0` | 完整团队作者 t0 前发表数均值。 |
| `dblp_team_max_publication_count_pre_t0` | 完整团队作者 t0 前发表数最大值。 |
| `dblp_team_academic_age_features_status` | 团队学术年龄完整性状态。 |
| `dblp_team_mean_academic_age_pre_t0` | 完整团队 t0 前学术年龄均值。 |
| `dblp_team_max_academic_age_pre_t0` | 完整团队 t0 前学术年龄最大值。 |
| `_dblp_team_row_present` | 该来源左连接时是否实际提供了这一篇论文的行；不等于特征完整。 |
| `s2_paper_id` | Semantic Scholar 论文 ID。 |
| `s2_title` | S2 当前标题，仅匹配/审计，不替代 v1 标题。 |
| `s2_doi` | S2 DOI。 |
| `s2_citation_count_current` | 当前累计被引数，不能作为首次提交任务的 X。 |
| `s2_influential_citation_count_current` | 当前累计 influential 被引数，非 IC3。 |
| `s2_reference_count_current` | S2 当前参考文献数，可能是后续版本。 |
| `s2_publication_date` | S2 出版日期，不替代 t0。 |
| `s2_year` | S2 年份。 |
| `s2_authors_json` | S2 当前作者记录 JSON。 |
| `s2_citation_count_3y` | C3：窗口内唯一 citing paper 数；不完整为缺失。 |
| `s2_influential_citation_count_3y` | IC3：相同窗口内唯一 influential citing paper 数。 |
| `s2_c3_status` | C3 详细完整性状态。 |
| `s2_ic3_status` | IC3 详细完整性状态。 |
| `s2_metadata_status` | MATCHED、REQUEST_ERROR、ID 冲突、无记录等元数据审计状态。 |
| `_semantic_scholar_row_present` | 该来源左连接时是否实际提供了这一篇论文的行；不等于特征完整。 |
| `s2_match_status` | 合并后的 S2 匹配状态。 |
| `openreview_collection_status` | 公开检索/评分抓取状态，区分 FETCH_ERROR 和无匹配。 |
| `openreview_matched_forum_count` | 匹配到的论坛数。 |
| `openreview_raw_rating_rows` | 原始评分行数。 |
| `openreview_raw_review_notes` | 原始 reviewer Note 数。 |
| `openreview_mean_rating` | 单一论坛、单一总体评分字段的去重 reviewer 均值；无公开评分为缺失。 |
| `openreview_n_scored_reviews` | 参与聚合的唯一评分评审数。 |
| `openreview_scored_forum_count` | 有数值评分的论坛数。 |
| `openreview_scored_forum_id` | 被聚合的论坛 ID。 |
| `openreview_score_field` | 实际使用的总体评分字段。 |
| `openreview_score_status` | MEAN_AVAILABLE、冲突/多论坛/混合字段、无评分或 PARTIAL。 |
| `_openreview_row_present` | 该来源左连接时是否实际提供了这一篇论文的行；不等于特征完整。 |
| `openalex_collection_status` | 当前项目未采集 OpenAlex。 |
| `title_word_count_v1` | 按原 notebook 空白分词计算标题词数。 |
| `abstract_word_count_v1` | 按原 notebook 空白分词计算摘要词数。 |
| `s2_c3_label_valid` | C3 状态是否 COMPLETE / ZERO_CONFIRMED_BY_CUMULATIVE_COUNT。 |
| `s2_ic3_label_valid` | IC3 状态是否 COMPLETE / ZERO_CONFIRMED_BY_CUMULATIVE_COUNT。 |
| `dblp_status` | DBLP 阶段统一论文运行状态，不替代身份置信度。 |
| `semantic_scholar_status` | S2 统一论文状态。 |
| `openreview_status` | OpenReview 统一论文状态。 |

## 引用边与覆盖审计

边表字段为 arxiv_id、s2_paper_id、citing_paper_id、citing_publication_date、citing_year、date_precision、window_status、is_influential、date_conflict、influential_flag_conflict。window_status 用于区分 IN_WINDOW / OUTSIDE_WINDOW / 日期不确定；不能直接以边表总行数作为 C3。

引用论文表另有 s2_c3_pages_fetched、records_scanned、observed_in_window、year_only_in_window、ambiguous_date_records、missing_date_records、missing_id_records、uncertain_unique_ids，以及 IC3 的 observed_in_window、unknown_flags、missing_date_records、missing_id_records、flag_conflict_ids。观察到的计数是下界/审计值，不能替代缺失的完整标签。start_date 与 end_date_exclusive 保存实际时间边界；s2_c3_error 保存失败摘要。

DBLP 作者明细保留 author_position、arxiv_author_name_v1、dblp_person_id、dblp_identity_method、dblp_identity_confidence，以及 dblp_author_publication_count_pre_t0、dblp_author_academic_age_pre_t0 等原始字段。团队聚合只在相应完整性条件满足时提供正式特征。

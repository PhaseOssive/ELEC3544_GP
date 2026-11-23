# Read-only Data Quality Assessment Report

- Run time (UTC): 2026-11-09T16:39:53.947494+00:00
- Source CSV: `/Users/owenli/Desktop/handle_data_quality_check/papers_features.csv`
- Source shape: 29963 rows × 25 columns
- Assessed candidate X/Y columns: 14
- Ignored from detailed feature analysis: 11 columns
- Source data changed: **No** (read-only assessment)

## Assessment scope

Detailed completeness, validity-range, distribution, and outlier analysis is limited to candidate model inputs/targets. Identifier, URL/DOI, raw text/JSON, source identifiers, collection-status, and pipeline-audit fields are not plotted or treated as candidate predictors. The full source table is still used for shape, structural, duplicate-record, primary-key, and configured cross-field integrity checks.

### Provisional predictor candidates

`submission_year`, `primary_category_v1`, `author_count_v1`, `dblp_previously_published_before_arxiv_v1`, `dblp_team_mean_publication_count_pre_t0`, `dblp_team_max_publication_count_pre_t0`, `dblp_team_mean_academic_age_pre_t0`, `dblp_team_max_academic_age_pre_t0`, `s2_reference_count_current`, `title_word_count_v1`, `abstract_word_count_v1`

### Provisional target/outcome candidates

`s2_citation_count_3y`, `s2_influential_citation_count_3y`, `openreview_mean_rating`

### Fields ignored in detailed feature analysis (not deleted)

`paper_id`, `title_v1`, `abstract_v1`, `categories_v1_json`, `cs_categories_v1_json`, `authors_v1_json`, `comment_v1`, `journal_ref_v1`, `doi_v1`, `t0_date`, `dblp_temporal_status`

## Detection results

### Structural

| Check | Status | Issue detected |
|---|---|---|
| `check_shape` | info | False |
| `check_dtypes` | pass | False |
| `check_info` | info | False |
| `check_memory_usage` | info | False |

### Completeness

| Check | Status | Issue detected |
|---|---|---|
| `check_missing_count` | warning | True |
| `check_missing_ratio` | warning | True |
| `check_missing_pattern` | warning | True |
| `check_nonstandard_missing` | pass | False |

### Validity

| Check | Status | Issue detected |
|---|---|---|
| `check_value_range` | pass | False |
| `check_duplicate_rows` | pass | False |
| `check_primary_key_missing` | pass | False |
| `check_duplicate_paper_id` | pass | False |
| `check_category_consistency` | pass | False |
| `check_cross_field_consistency` | pass | False |

### Distributional

| Check | Status | Issue detected |
|---|---|---|
| `calculate_summary_statistics` | info | False |
| `plot_histogram` | info | False |
| `plot_boxplot` | info | False |
| `plot_countplot` | info | False |
| `plot_kde` | info | False |
| `check_iqr_outliers` | info | True |

## Feature-level overview

| Feature | Dtype | Missing % | Nonstandard missing | Out of range | IQR outliers | Mean | Median |
|---|---|---:|---:|---:|---:|---:|---:|
| `submission_year` | `int64` | 0.00% | 0 | 0 | 0 | 2020 | 2020 |
| `primary_category_v1` | `str` | 0.00% | 0 | 0 | 0 |  |  |
| `author_count_v1` | `int64` | 0.00% | 0 | 0 | 580 | 3.879 | 3 |
| `dblp_previously_published_before_arxiv_v1` | `bool` | 0.00% | 0 | 0 | 0 |  |  |
| `dblp_team_mean_publication_count_pre_t0` | `float64` | 49.02% | 0 | 0 | 815 | 45.57 | 30.5 |
| `dblp_team_max_publication_count_pre_t0` | `float64` | 49.02% | 0 | 0 | 998 | 107.7 | 66 |
| `dblp_team_mean_academic_age_pre_t0` | `float64` | 67.09% | 0 | 0 | 350 | 11.79 | 11 |
| `dblp_team_max_academic_age_pre_t0` | `float64` | 67.09% | 0 | 0 | 63 | 19.08 | 18 |
| `s2_reference_count_current` | `float64` | 0.36% | 0 | 0 | 1266 | 41.57 | 36 |
| `title_word_count_v1` | `int64` | 0.00% | 0 | 0 | 445 | 9.242 | 9 |
| `abstract_word_count_v1` | `int64` | 0.00% | 0 | 0 | 91 | 165.7 | 163 |
| `s2_citation_count_3y` | `float64` | 0.48% | 0 | 0 | 3243 | 19.91 | 6 |
| `s2_influential_citation_count_3y` | `float64` | 0.37% | 0 | 0 | 2806 | 2.588 | 0 |
| `openreview_mean_rating` | `float64` | 96.88% | 0 | 0 | 20 | 5.645 | 6 |

## Outlier interpretation

- Method: Tukey's IQR fences, Q1 − 1.5×IQR and Q3 + 1.5×IQR.
- Potential outlier feature-record flags: 10677.
- IQR flags are screening candidates, not automatic data errors. Genuine extreme academic teams or citation counts should normally be retained unless domain evidence indicates an error.
- No rows or values were corrected, imputed, clipped, or deleted.

## Output files

- `report/data_quality_report.json`: complete machine-readable checks.
- `report/data_quality_report.md`: this summary.
- `report/feature_level_report.csv`: candidate feature-level statistics.
- `report/record_issue_report.csv`: row-level issue index.
- `report/outlier_report.csv`: one row per potential outlier feature-record.
- `diagram/`: missingness and distribution plots.

## Interpretation notes

- A `skipped` check means the required rule or usable input was unavailable; it does not mean the data passed.
- The assessment scope is a provisional candidate X/Y list, not the final modelling feature list.
- The source CSV is never rewritten. All generated files are separate reports and figures.

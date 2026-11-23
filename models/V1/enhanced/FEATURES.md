# Feature and evaluation contract

This file is the reference for what enters the model, what is deliberately
excluded, and how the temporal boundary is enforced.  The source table is
`gp_data/papers_features.csv` (29,856 rows × 25 source columns, 2017–2022).

## Source columns and role

| Source column | Role | Used in X? |
|---|---|---|
| `paper_id` | Primary key | No |
| `title_v1` | t0 text | Through TF-IDF only |
| `abstract_v1` | t0 text | Through TF-IDF only |
| `primary_category_v1` | t0 category | One-hot |
| `categories_v1_json` | Full category list | No (redundant with primary category here) |
| `cs_categories_v1_json` | cs.* subset | No |
| `authors_v1_json` | Author JSON | No; `author_count_v1` is used |
| `author_count_v1` | Team size | Yes, as log1p |
| `comment_v1` | Optional comment/venue text | No |
| `journal_ref_v1` | Optional journal reference | No |
| `doi_v1` | Optional identifier | No |
| `submission_year` | t0 cohort | Yes, numeric cohort control; also split/threshold key |
| `t0_date` | First-submission time | No |
| `dblp_temporal_status` | DBLP matching audit | No |
| `dblp_previously_published_before_arxiv_v1` | DBLP audit indicator | No |
| `dblp_team_mean_publication_count_pre_t0` | Author history | Yes, log1p |
| `dblp_team_max_publication_count_pre_t0` | Author history | Yes, log1p |
| `dblp_team_mean_academic_age_pre_t0` | Author history | Yes, raw |
| `dblp_team_max_academic_age_pre_t0` | Author history | Yes, raw |
| `s2_reference_count_current` | Current-time metadata | No; post-t0 |
| `s2_citation_count_3y` | Y1 | Target only |
| `s2_influential_citation_count_3y` | Y2 | Target only |
| `openreview_mean_rating` | Y3 | Target only |
| `title_word_count_v1` | Title length | Yes |
| `abstract_word_count_v1` | Abstract length | Yes |

## Derived model features

The loader creates these columns; they are not in the source CSV:

| Derived column | Definition |
|---|---|
| `author_count_log1p` | `log1p(author_count_v1)` |
| `dblp_mean_publication_log1p` | `log1p(dblp_team_mean_publication_count_pre_t0)` |
| `dblp_max_publication_log1p` | `log1p(dblp_team_max_publication_count_pre_t0)` |
| `dblp_mean_academic_age` | Copy of the mean academic-age feature |
| `dblp_max_academic_age` | Copy of the max academic-age feature |
| `dblp_features_missing` | 1 if all four DBLP features are missing, else 0 |
| `text_all` | `title_v1 + " " + abstract_v1` for TF-IDF |

## Why these exclusions matter

- `s2_reference_count_current` is current-time metadata, not a t0
  observation.  Using it would violate the prediction-time boundary.
- `comment_v1`, `journal_ref_v1` and `doi_v1` are optional metadata that can
  reveal later venue or publication outcomes; they are not part of the
  approved paper/team/content feature set.
- DBLP matching/audit fields are not author-history features and can encode
  collection-process information.
- `paper_id` is a key, not a predictor.
- `submission_year` is a t0-available cohort control and is also used for the
  chronological split and annual thresholds.  Because the test years are
  later than the training years, its coefficient/leaf effect is an
  extrapolation control rather than a post-t0 signal.
- Target columns are never used as features.

## Preprocessing and temporal discipline

All learned transformations are fitted on 2017–2020 training rows only:

- median imputation and scaling for the Linear Regression numeric block;
- native missing-value handling for XGBoost (the DBLP indicator is retained);
- one-hot encoding of `primary_category_v1`, with rare categories grouped
  using training rows only;
- TF-IDF vocabulary over title + abstract, fitted on training rows only.

For the small Y3 training set, the TF-IDF block is capped at 1,500 terms
instead of the full 5,000.  Both model families use the same feature block
within a target.

## Targets and success rule

- Y1: three-year citation count, observed for 29,818 papers.
- Y2: three-year influential-citation count, observed for 29,852 papers.
- Y3: OpenReview mean rating, observed for 934 papers.
- Y1/Y2 are modelled on `log1p` scale, with a Duan smearing factor estimated
  from training residuals to correct retransformation bias.  Y3 is modelled on
  its observed rating scale.
- Each target model uses its own observed-label subset; missing labels are
  excluded, never imputed as zero.
- For each target and each training year (2017–2020), the success threshold is
  the 90th percentile of observed training labels.
- The held-out threshold is the mean of those four annual thresholds.
- For the held-out years, reference success labels use that year's observed
  90th percentile; this is computed only at evaluation time and never enters
  model fitting or threshold estimation.
- Per target: `predicted_success = predicted_value >= estimated_threshold`;
  `true_success = observed_value >= observed_test_year_p90`.
- Primary all-target success: rows with all three observed outcomes; success
  if at least one target reaches its threshold.  Its coverage is reported in
  `success_metrics.csv`.

## Reproducibility controls

- `paper_id` is read as a string.
- Chronological split: 2017–2020 train; 2021–2022 test.
- XGBoost validation: 2020; inner train: 2017–2019.
- Final models refit on 2017–2020 after validation selection.
- Fixed seed (42 by default); versions and config saved in `run_manifest.json`.
- No test data is used for imputation, encoding, vectorisation, scaling,
  hyper-parameter selection, or threshold estimation.

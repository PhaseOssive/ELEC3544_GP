# Data audit

Source: `C:\Users\Jack\Programming\DSH_Workspace\ELEC3544_Group_Project\gp_data\papers_features.csv`
Rows: 29,856
Source columns: 25
Derived columns: 7
All three targets observed: 929
DBLP four-feature missing: 4,269

## Target coverage

| Target | Column | Available | Missing | Rate |
|---|---|---:|---:|---:|
| Y1 | `s2_citation_count_3y` | 29,818 | 38 | 99.87% |
| Y2 | `s2_influential_citation_count_3y` | 29,852 | 4 | 99.99% |
| Y3 | `openreview_mean_rating` | 934 | 28,922 | 3.13% |

## Chronological split

Train: 2017, 2018, 2019, 2020  
Test: 2021, 2022  
XGBoost validation year: 2020  
XGBoost inner train years: 2017-2019

## Modelling contract

Numeric: submission_year (t0 cohort control), author_count_log1p, title/abstract word counts,
four DBLP features (counts log1p; ages raw), and dblp_features_missing.

Categorical: one-hot primary_category_v1, rare categories grouped using
training rows only.

Text: TF-IDF over title_v1 + abstract_v1, fitted on training rows only.

Excluded: paper_id, raw text as a numeric feature, authors/categories JSON,
comment/journal_ref/doi, t0_date, DBLP audit fields, s2_reference_count_current
(post-t0), and all targets.

See DATA_SPLIT.csv and DATA_TARGET_COVERAGE.csv for cohort detail.

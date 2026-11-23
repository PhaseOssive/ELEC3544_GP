"""Shared constants for the enhanced impact-model pipeline."""

from __future__ import annotations

TRAIN_YEARS = (2017, 2018, 2019, 2020)
TEST_YEARS = (2021, 2022)

# Expanding-window temporal validation folds.
FOLDS = (
    {"name": "2019", "train_years": (2017, 2018), "validation_year": 2019},
    {"name": "2020", "train_years": (2017, 2018, 2019), "validation_year": 2020},
)

TARGETS = {
    "Y1": {
        "column": "s2_citation_count_3y",
        "label": "C3 / three-year citations",
        "task": "count",
    },
    "Y2": {
        "column": "s2_influential_citation_count_3y",
        "label": "IC3 / three-year influential citations",
        "task": "count",
    },
    "Y3": {
        "column": "openreview_mean_rating",
        "label": "R / OpenReview mean rating",
        "task": "rating",
    },
}

REQUIRED_COLUMNS = (
    "paper_id", "title_v1", "abstract_v1", "primary_category_v1",
    "categories_v1_json", "cs_categories_v1_json", "authors_v1_json",
    "author_count_v1", "comment_v1", "journal_ref_v1", "doi_v1",
    "submission_year", "t0_date", "dblp_temporal_status",
    "dblp_previously_published_before_arxiv_v1",
    "dblp_team_mean_publication_count_pre_t0",
    "dblp_team_max_publication_count_pre_t0",
    "dblp_team_mean_academic_age_pre_t0",
    "dblp_team_max_academic_age_pre_t0",
    "s2_reference_count_current", "s2_citation_count_3y",
    "s2_influential_citation_count_3y", "openreview_mean_rating",
    "title_word_count_v1", "abstract_word_count_v1",
)

# Optional external columns populated by `scripts/fetch_external_features.py`
OPTIONAL_EXTERNAL_COLUMNS = (
    "oa_matched",
    "oa_institutions_count",
    "oa_countries_count",
    "oa_has_top_industry_lab",
    "oa_has_top_university",
    "oa_early_citations_t0_year",
    "oa_early_citations_t0_plus1",
    "oa_referenced_works_count",
    "oa_institutions_json",
    "s2_batch_matched",
    "s2_ref_citation_max",
    "s2_ref_citation_mean",
    "s2_ref_citation_median",
    "s2_ref_recent_2y_ratio",
    "s2_author_max_hindex",
    "s2_author_mean_hindex",
    "s2_author_max_citations",
    "pwc_has_official_code",
    "pwc_repo_count",
    "pwc_is_pytorch",
    "pwc_is_tensorflow",
    "pwc_mentioned_in_github",
)

DBLP_COLUMNS = (
    "dblp_team_mean_publication_count_pre_t0",
    "dblp_team_max_publication_count_pre_t0",
    "dblp_team_mean_academic_age_pre_t0",
    "dblp_team_max_academic_age_pre_t0",
)

# Strictly t0-observable structured features
STRICT_T0_NUMERIC_FEATURES = (
    "submission_year",
    "t0_month_sin",
    "t0_month_cos",
    "author_count_log1p",
    "has_author_affiliation",
    "category_count",
    "is_interdisciplinary",
    "title_word_count_v1",
    "abstract_word_count_v1",
    "dblp_mean_publication_log1p",
    "dblp_max_publication_log1p",
    "dblp_team_total_pub_log1p",
    "dblp_star_ratio",
    "dblp_productivity_rate",
    "dblp_mean_academic_age",
    "dblp_max_academic_age",
    "dblp_features_missing",
    "text_has_code_or_url",
    "text_has_sota_claim",
    "text_has_theory_claim",
    "text_is_survey_or_benchmark",
    "text_has_named_model_colon",
    "text_is_challenge_report",
)

# Strictly t0-observable external features (when `papers_features_enriched.csv` is loaded)
STRICT_T0_EXTERNAL_NUMERIC_FEATURES = (
    "oa_institutions_count_log1p",
    "oa_countries_count_log1p",
    "oa_has_top_industry_lab",
    "oa_has_top_university",
    "s2_ref_recent_2y_ratio",
    "pwc_has_official_code",
    "pwc_is_pytorch",
    "pwc_is_tensorflow",
)

# Extended table features available in papers_features.csv
EXTENDED_NUMERIC_FEATURES = STRICT_T0_NUMERIC_FEATURES + (
    "reference_count_log1p",
    "comment_present",
    "comment_pages_log1p",
    "comment_figures_log1p",
    "comment_accepted",
    "comment_top_venue",
    "comment_oral_spotlight",
    "comment_submitted_only",
    "has_journal_ref",
    "has_doi",
    "dblp_published_same_year",
    "dblp_published_after",
    "dblp_previously_published_int",
)

# Extended external features (when `papers_features_enriched.csv` is loaded)
EXTENDED_EXTERNAL_NUMERIC_FEATURES = STRICT_T0_EXTERNAL_NUMERIC_FEATURES + (
    "pwc_repo_count_log1p",
    "oa_early_citations_t0_year_log1p",
    "oa_early_citations_t0_plus1_log1p",
    "oa_referenced_works_count_log1p",
    "s2_ref_citation_max_log1p",
    "s2_ref_citation_mean_log1p",
    "s2_ref_citation_median_log1p",
    "s2_author_max_hindex_log1p",
    "s2_author_mean_hindex_log1p",
    "s2_author_max_citations_log1p",
    "pwc_mentioned_in_github",
)

NUMERIC_FEATURES = EXTENDED_NUMERIC_FEATURES

EXCLUDED_FROM_X = {
    "paper_id": "primary key",
    "title_v1": "raw text; consumed by TF-IDF/BGE and emerging-paradigm encoder",
    "abstract_v1": "raw text; consumed by TF-IDF/BGE and emerging-paradigm encoder",
    "categories_v1_json": "parsed into multi-hot category features",
    "cs_categories_v1_json": "parsed into multi-hot category features",
    "authors_v1_json": "parsed into training-only 2-hop collaboration graph & DBLP recovery",
    "oa_institutions_json": "parsed into training-only institutional prestige prior",
    "s2_citation_count_3y": "target Y1",
    "s2_influential_citation_count_3y": "target Y2",
    "openreview_mean_rating": "target Y3",
}

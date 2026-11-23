"""Local project paths and detection-only rules for papers_features.csv.

The input CSV is read-only. This module selects assessment columns and
defines checks; it never changes, drops, imputes, or rewrites source data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# All default paths are relative to this project folder.
PROJECT_DIR = Path(__file__).resolve().parent

# Place papers_features.csv directly beside main.py and this config file.
INPUT_FILE_NAME = "papers_features.csv"
DEFAULT_INPUT_FILE = PROJECT_DIR / INPUT_FILE_NAME

# Each run creates one self-contained output folder inside this project.
# If the folder already exists, the program creates data_quality_check_2,
# data_quality_check_3, and so on, without overwriting previous reports.
OUTPUT_FOLDER_NAME = "data_quality_check"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / OUTPUT_FOLDER_NAME
AUTO_INCREMENT_OUTPUT_FOLDER = True

# Only these fields are treated as potentially useful tabular model inputs
# or target candidates for detailed completeness/distribution/outlier review.
# Raw text, JSON payloads, identifiers, URLs/DOIs, source IDs, collection
# flags, matching diagnostics, and pipeline-status fields are deliberately
# excluded from detailed feature analysis, but remain untouched in the CSV.
#
# This is an assessment scope, NOT a final X/Y decision. Confirm the actual
# target and predictors during feature preparation.
PREDICTOR_CANDIDATE_COLUMNS = [
    "submission_year",
    "primary_category_v1",
    "author_count_v1",
    "dblp_previously_published_before_arxiv_v1",
    "dblp_team_mean_publication_count_pre_t0",
    "dblp_team_max_publication_count_pre_t0",
    "dblp_team_mean_academic_age_pre_t0",
    "dblp_team_max_academic_age_pre_t0",
    "s2_reference_count_current",
    "title_word_count_v1",
    "abstract_word_count_v1",
]

TARGET_CANDIDATE_COLUMNS = [
    "s2_citation_count_3y",
    "s2_influential_citation_count_3y",
    "openreview_mean_rating",
]

# Combined scope for quality assessment. The final X/Y choice is a later
# project decision; these are provisional predictor/target candidates.
ASSESSMENT_COLUMNS = PREDICTOR_CANDIDATE_COLUMNS + TARGET_CANDIDATE_COLUMNS

# Numeric columns for potential-outlier screening. Year and binary flags are
# intentionally omitted: IQR fences are not informative for those fields.
OUTLIER_COLUMNS = [
    "author_count_v1",
    "dblp_team_mean_publication_count_pre_t0",
    "dblp_team_max_publication_count_pre_t0",
    "dblp_team_mean_academic_age_pre_t0",
    "dblp_team_max_academic_age_pre_t0",
    "s2_reference_count_current",
    "s2_citation_count_3y",
    "s2_influential_citation_count_3y",
    "openreview_mean_rating",
    "title_word_count_v1",
    "abstract_word_count_v1",
]
IQR_MULTIPLIER = 1.5

# The only expected dtype assertion is on the stable primary identifier.
EXPECTED_DTYPES = {
    "paper_id": "string",
}

# Bounds are included only where the field definition supports them.
# These are validity checks, not outlier rules.
VALUE_RANGE_RULES: dict[str, dict[str, Any]] = {
    "author_count_v1": {"min": 1},
    "title_word_count_v1": {"min": 1},
    "abstract_word_count_v1": {"min": 1},
    "dblp_team_mean_publication_count_pre_t0": {"min": 0},
    "dblp_team_max_publication_count_pre_t0": {"min": 0},
    "dblp_team_mean_academic_age_pre_t0": {"min": 0},
    "dblp_team_max_academic_age_pre_t0": {"min": 0},
    "s2_reference_count_current": {"min": 0},
    "s2_citation_count_3y": {"min": 0},
    "s2_influential_citation_count_3y": {"min": 0},
}

CATEGORY_COLUMNS = [
    "primary_category_v1",
]

COUNT_PLOT_COLUMNS = [
    "primary_category_v1",
    "dblp_previously_published_before_arxiv_v1",
]


def _authors_json_matches_count(df: pd.DataFrame) -> pd.Series:
    """True when authors_v1_json is a JSON list matching author_count_v1."""
    def matches(row: pd.Series) -> bool:
        raw = row.get("authors_v1_json")
        count = row.get("author_count_v1")
        if pd.isna(raw) or pd.isna(count):
            return True  # Completeness reports missing values separately.
        try:
            authors = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return isinstance(authors, list) and len(authors) == int(count)

    return df.apply(matches, axis=1).astype("boolean")


def _date_is_parseable(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["t0_date"], errors="coerce", utc=True).notna()


def _submission_year_matches_t0(df: pd.DataFrame) -> pd.Series:
    parsed = pd.to_datetime(df["t0_date"], errors="coerce", utc=True)
    year = pd.to_numeric(df["submission_year"], errors="coerce")
    return (parsed.notna() & year.notna() & (parsed.dt.year == year)).astype("boolean")


def _max_publications_not_below_mean(df: pd.DataFrame) -> pd.Series:
    mean = pd.to_numeric(df["dblp_team_mean_publication_count_pre_t0"], errors="coerce")
    maximum = pd.to_numeric(df["dblp_team_max_publication_count_pre_t0"], errors="coerce")
    available = mean.notna() & maximum.notna()
    return ((~available) | (maximum >= mean)).astype("boolean")


def _max_academic_age_not_below_mean(df: pd.DataFrame) -> pd.Series:
    mean = pd.to_numeric(df["dblp_team_mean_academic_age_pre_t0"], errors="coerce")
    maximum = pd.to_numeric(df["dblp_team_max_academic_age_pre_t0"], errors="coerce")
    available = mean.notna() & maximum.notna()
    return ((~available) | (maximum >= mean)).astype("boolean")


def _influential_citations_not_above_all_citations(df: pd.DataFrame) -> pd.Series:
    influential = pd.to_numeric(df["s2_influential_citation_count_3y"], errors="coerce")
    citations = pd.to_numeric(df["s2_citation_count_3y"], errors="coerce")
    available = influential.notna() & citations.notna()
    return ((~available) | (influential <= citations)).astype("boolean")


def _dblp_temporal_flag_matches_status(df: pd.DataFrame) -> pd.Series:
    expected = df["dblp_temporal_status"].eq("PREVIOUSLY_PUBLISHED_BEFORE_ARXIV_V1")
    observed = df["dblp_previously_published_before_arxiv_v1"].astype("boolean")
    return (expected == observed).astype("boolean")


def build_cross_field_rules(columns: list[str]) -> dict[str, dict[str, Any]]:
    """Build only domain consistency rules whose required fields exist."""
    available = set(columns)
    rules: dict[str, dict[str, Any]] = {}
    candidates = [
        (
            "authors_json_matches_author_count",
            ["authors_v1_json", "author_count_v1"],
            _authors_json_matches_count,
            "The number of authors in authors_v1_json should equal author_count_v1.",
        ),
        (
            "t0_date_is_parseable",
            ["t0_date"],
            _date_is_parseable,
            "t0_date should be parseable as a date/time value.",
        ),
        (
            "submission_year_matches_t0_date",
            ["submission_year", "t0_date"],
            _submission_year_matches_t0,
            "The year in t0_date should equal submission_year.",
        ),
        (
            "team_max_publications_not_below_mean",
            ["dblp_team_mean_publication_count_pre_t0", "dblp_team_max_publication_count_pre_t0"],
            _max_publications_not_below_mean,
            "Team maximum pre-t0 publication count should not be below the team mean.",
        ),
        (
            "team_max_academic_age_not_below_mean",
            ["dblp_team_mean_academic_age_pre_t0", "dblp_team_max_academic_age_pre_t0"],
            _max_academic_age_not_below_mean,
            "Team maximum academic age should not be below the team mean.",
        ),
        (
            "influential_citations_not_above_all_citations",
            ["s2_influential_citation_count_3y", "s2_citation_count_3y"],
            _influential_citations_not_above_all_citations,
            "Influential citations should be a subset of all citations over the same three-year window.",
        ),
        (
            "dblp_temporal_flag_matches_status",
            ["dblp_temporal_status", "dblp_previously_published_before_arxiv_v1"],
            _dblp_temporal_flag_matches_status,
            "The boolean temporal flag should agree with the DBLP temporal status.",
        ),
    ]
    for name, required, check, description in candidates:
        if set(required).issubset(available):
            rules[name] = {
                "columns": required,
                "check": check,
                "description": description,
            }
    return rules

#!/usr/bin/env python3
"""Remove post-t0 / current-snapshot leaky external columns from `papers_features_enriched.csv`
while preserving all 25 original columns and the 11 strictly t0-safe external columns."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

ORIGINAL_25_COLUMNS = (
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

# 11 strictly t0-safe external columns (fixed at paper submission time t0)
CLEAN_T0_EXTERNAL_COLUMNS = (
    "oa_matched",
    "oa_institutions_count",
    "oa_countries_count",
    "oa_has_top_industry_lab",
    "oa_has_top_university",
    "oa_institutions_json",
    "s2_batch_matched",
    "s2_ref_recent_2y_ratio",
    "pwc_has_official_code",
    "pwc_is_pytorch",
    "pwc_is_tensorflow",
)

# 11 post-t0 / current-snapshot leaky external columns to drop
LEAKY_EXTERNAL_COLUMNS = (
    "oa_early_citations_t0_year",
    "oa_early_citations_t0_plus1",
    "oa_referenced_works_count",
    "s2_ref_citation_max",
    "s2_ref_citation_mean",
    "s2_ref_citation_median",
    "s2_author_max_hindex",
    "s2_author_mean_hindex",
    "s2_author_max_citations",
    "pwc_repo_count",
    "pwc_mentioned_in_github",
)


def clean_leaky_columns(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, dtype={"paper_id": "string"}, low_memory=False)
    missing_orig = [c for c in ORIGINAL_25_COLUMNS if c not in df.columns]
    if missing_orig:
        raise ValueError(f"Missing original 25 columns: {missing_orig}")

    dropped = [c for c in LEAKY_EXTERNAL_COLUMNS if c in df.columns]
    kept_ext = [c for c in CLEAN_T0_EXTERNAL_COLUMNS if c in df.columns]
    keep_cols = list(ORIGINAL_25_COLUMNS) + kept_ext

    cleaned = df[keep_cols].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(
        f"[CleanExternal] Dropped {len(dropped)} leaky external columns: {dropped}\n"
        f"[CleanExternal] Kept all 25 original columns + {len(kept_ext)} clean t0 external columns "
        f"({cleaned.shape[0]} rows x {cleaned.shape[1]} cols) -> {output_path}"
    )
    return cleaned


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop leaky external columns from papers_features_enriched.csv while keeping all 25 original columns and 11 clean t0 external columns."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("papers_features_enriched.csv"),
        help="Input enriched CSV path (default: papers_features_enriched.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("papers_features_enriched_clean.csv"),
        help="Output cleaned CSV path (default: papers_features_enriched_clean.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    clean_leaky_columns(args.input.expanduser().resolve(), args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

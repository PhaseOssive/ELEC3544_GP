"""显式调用的 Notebook 迁移模块。输入输出与主键见 README 和 DATA_DICTIONARY。"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fetch.control.config import Config
    from fetch.control.tools import HTTPClient
    import pandas as pd
from fetch.control.tools import atomic_csv, log_preview, log_print as print

def collect(config: Config, http: HTTPClient, papers_df=None, seed_df=None) -> dict[str, pd.DataFrame]:
    ROOT, RAW, PROCESSED = config.paths
    tables = papers_df
    # ============================================================
    # GP DATA COLLECTION — FINAL MASTER TABLE CELL (Notebook Cell 19)
    # ============================================================

    # ============================================================
    # 7. FINAL MASTER TABLE — ONE ROW PER ARXIV PAPER
    #
    # Inputs:
    #   arXiv v1: papers_df
    #   DBLP: dblp_author_features_df
    #   Semantic Scholar: s2_all_df (all input papers, including unmatched)
    #   OpenReview: openreview_paper_ratings_df
    #
    # Optional fallback: load the saved CSVs if an in-memory DataFrame
    # is unavailable. OpenAlex is NOT collected by this notebook version;
    # this cell will only merge it if openalex_work_df is explicitly present.
    #
    # Outputs:
    #   papers_master_raw_stage.csv
    #   papers_master_raw_stage.jsonl
    #   papers_master_metadata.json
    # ============================================================

    from pathlib import Path
    import json
    import re
    import numpy as np
    import pandas as pd

    def _master_normalize_arxiv_id(value):
        if pd.isna(value):
            return None
        value = str(value).strip()
        value = value.rsplit("/abs/", 1)[-1]
        value = re.sub(r"v\d+$", "", value)
        return value or None


    def _master_get_df(variable_name, csv_path=None):
        value = tables.get(variable_name)
        return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()



    def _master_prepare_source(df, source_name, base_columns):
        """Normalize arXiv key and enforce one row per paper."""
        if df is None or df.empty:
            return pd.DataFrame()

        source = df.copy()

        if "arxiv_id" not in source.columns:
            raise ValueError(
                f"{source_name}: source table has no arxiv_id column."
            )

        source["arxiv_id"] = source["arxiv_id"].map(
            _master_normalize_arxiv_id
        )
        source = source.dropna(subset=["arxiv_id"]).copy()

        duplicated = source["arxiv_id"].duplicated(keep=False)
        if duplicated.any():
            examples = (
                source.loc[duplicated, "arxiv_id"]
                .astype(str).unique()[:10].tolist()
            )
            raise ValueError(
                f"{source_name}: expected one row per arxiv_id, "
                f"but duplicates were found: {examples}. "
                "Do not merge a many-row table directly into the master."
            )

        # The arXiv v1 table is the canonical source for overlapping
        # paper metadata such as title_v1 and submission_date_v1.
        overlap = [
            c for c in source.columns
            if c != "arxiv_id" and c in base_columns
        ]
        if overlap:
            print(
                f"{source_name}: keeping canonical arXiv columns; "
                f"dropping duplicate source columns: {overlap}"
            )
            source = source.drop(columns=overlap)

        return source


    def _master_bool(series):
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)
        return (
            series.astype(str).str.strip().str.lower()
            .isin(["true", "1", "yes", "y"])
        )


    def _master_merge(master, source, source_name):
        if source.empty:
            print(f"{source_name}: no table available; skipped.")
            return master

        source = _master_prepare_source(
            source, source_name, set(master.columns)
        )
        if source.empty:
            print(f"{source_name}: no valid arxiv_id rows; skipped.")
            return master

        marker = f"_{source_name.lower()}_row_present"
        source[marker] = True

        before_rows = len(master)
        merged = master.merge(
            source,
            on="arxiv_id",
            how="left",
            validate="one_to_one"
        )
        merged[marker] = merged[marker].fillna(False).astype(bool)

        if len(merged) != before_rows:
            raise AssertionError(
                f"{source_name} merge changed master row count: "
                f"{before_rows} -> {len(merged)}"
            )

        print(
            f"{source_name}: merged {int(merged[marker].sum())}/"
            f"{len(merged)} paper rows."
        )
        return merged


    # ------------------------------------------------------------
    # A. Canonical arXiv v1 paper table
    # ------------------------------------------------------------

    papers_source = _master_get_df(
        "papers_df",
        PROCESSED / "papers_arxiv_v1.csv"
    )

    if papers_source.empty:
        raise FileNotFoundError(
            "Cannot find papers_df or papers_arxiv_v1.csv. "
            "Run the arXiv v1 collection cell first."
        )

    if "arxiv_id" not in papers_source.columns:
        raise ValueError("papers_df must contain arxiv_id.")

    papers_source = papers_source.copy()
    papers_source["arxiv_id"] = papers_source["arxiv_id"].map(
        _master_normalize_arxiv_id
    )
    papers_source = papers_source.dropna(subset=["arxiv_id"]).copy()

    if papers_source["arxiv_id"].duplicated().any():
        raise ValueError(
            "The arXiv v1 paper table contains duplicate arxiv_id values."
        )

    master_df = papers_source.copy()
    master_input_n = len(master_df)


    # ------------------------------------------------------------
    # B. DBLP paper-level author features
    # ------------------------------------------------------------

    dblp_dir = PROCESSED / "dblp_author_features_v3_1_1"

    dblp_paper_source = _master_get_df(
        "dblp_author_features_df",
        dblp_dir / "dblp_author_features.csv"
    )

    master_df = _master_merge(
        master_df,
        dblp_paper_source,
        "DBLP"
    )

    # Keep the DBLP author-level table separate; never merge it raw
    # because it contains multiple rows per paper.
    dblp_by_author = _master_get_df(
        "dblp_author_features_by_author_df",
        dblp_dir / "dblp_author_features_by_author.csv"
    )

    # Derive team Mean / Max features only when every arXiv v1 author
    # has a corresponding matched DBLP author row and a nonmissing value.
    # These are additional derived variables; the existing DBLP Sum
    # variables remain unchanged.
    if not dblp_by_author.empty and "arxiv_id" in dblp_by_author.columns:

        da = dblp_by_author.copy()
        da["arxiv_id"] = da["arxiv_id"].map(
            _master_normalize_arxiv_id
        )
        da = da.dropna(subset=["arxiv_id"]).copy()

        if {"arxiv_id", "author_position"}.issubset(da.columns):
            if da.duplicated(["arxiv_id", "author_position"]).any():
                raise ValueError(
                    "DBLP author-level table has duplicate "
                    "(arxiv_id, author_position) rows."
                )

        expected_author_counts = (
            master_df.set_index("arxiv_id")["author_count_v1"]
            if "author_count_v1" in master_df.columns
            else pd.Series(dtype="float64")
        )

        team_rows = []

        for aid, group in da.groupby("arxiv_id", sort=False):

            expected = expected_author_counts.get(aid, np.nan)
            expected = (
                int(expected)
                if pd.notna(expected)
                else None
            )

            row_count = len(group)

            if "dblp_author_matched_to_paper" in group.columns:
                matched_mask = _master_bool(
                    group["dblp_author_matched_to_paper"]
                )
            else:
                matched_mask = group.get(
                    "dblp_person_id",
                    pd.Series(index=group.index, dtype="object")
                ).notna()

            matched_count = int(matched_mask.sum())

            all_authors_matched = (
                expected is not None
                and row_count == expected
                and matched_count == expected
            )

            pub = pd.to_numeric(
                group.get(
                    "dblp_author_publication_count_pre_t0",
                    pd.Series(index=group.index, dtype="float64")
                ),
                errors="coerce"
            )

            age = pd.to_numeric(
                group.get(
                    "dblp_author_academic_age_pre_t0",
                    pd.Series(index=group.index, dtype="float64")
                ),
                errors="coerce"
            )

            pub_complete = (
                all_authors_matched
                and len(pub) == expected
                and pub.notna().all()
            )

            age_complete = (
                all_authors_matched
                and len(age) == expected
                and age.notna().all()
            )

            confidence = (
                group["dblp_identity_confidence"].astype(str)
                if "dblp_identity_confidence" in group.columns
                else pd.Series([], dtype="object")
            )

            team_rows.append({
                "arxiv_id": aid,
                "dblp_team_author_row_count": row_count,
                "dblp_team_author_matched_count": matched_count,
                "dblp_team_all_authors_matched": all_authors_matched,
                "dblp_team_high_confidence_author_count": int(
                    confidence.eq("HIGH").sum()
                ),
                "dblp_team_medium_confidence_author_count": int(
                    confidence.eq("MEDIUM").sum()
                ),
                "dblp_team_publication_features_status": (
                    "COMPLETE" if pub_complete
                    else "INCOMPLETE_AUTHOR_OR_HISTORY_COVERAGE"
                ),
                "dblp_team_mean_publication_count_pre_t0": (
                    float(pub.mean()) if pub_complete else np.nan
                ),
                "dblp_team_max_publication_count_pre_t0": (
                    float(pub.max()) if pub_complete else np.nan
                ),
                "dblp_team_academic_age_features_status": (
                    "COMPLETE" if age_complete
                    else "INCOMPLETE_AUTHOR_OR_HISTORY_COVERAGE"
                ),
                "dblp_team_mean_academic_age_pre_t0": (
                    float(age.mean()) if age_complete else np.nan
                ),
                "dblp_team_max_academic_age_pre_t0": (
                    float(age.max()) if age_complete else np.nan
                )
            })

        team_features_df = pd.DataFrame(team_rows)

        master_df = _master_merge(
            master_df,
            team_features_df,
            "DBLP_TEAM"
        )

    else:
        print(
            "DBLP author-level detail table unavailable; "
            "team Mean / Max features were not derived."
        )


    # ------------------------------------------------------------
    # C. Semantic Scholar: use ALL-paper table, not matched-only s2_df
    # ------------------------------------------------------------

    s2_output_dir = PROCESSED / "semantic_scholar_public"

    s2_source = _master_get_df("s2_all_df")

    # The saved semantic_scholar_papers.csv is matched-only in this
    # notebook and therefore is NOT the preferred fallback. Reconstruct
    # the all-paper table from s2_metadata.csv + C3/IC3 labels so papers
    # without an S2 match remain represented in the source table.
    if s2_source.empty:

        s2_metadata_source = _master_get_df(
            "s2_metadata_df",
            s2_output_dir / "s2_metadata.csv"
        )

        if not s2_metadata_source.empty:

            rename_map = {
                "title": "s2_title",
                "doi": "s2_doi",
                "citation_count": "s2_citation_count_current",
                "influential_citation_count":
                    "s2_influential_citation_count_current",
                "reference_count": "s2_reference_count_current",
                "publication_date": "s2_publication_date",
                "year": "s2_year",
                "authors_json": "s2_authors_json"
            }

            s2_metadata_source = s2_metadata_source.rename(
                columns={
                    k: v for k, v in rename_map.items()
                    if k in s2_metadata_source.columns
                }
            )

            s2_citation_source = _master_get_df(
                "s2_citation_3y_df",
                s2_output_dir / "semantic_scholar_citation_3y.csv"
            )

            if not s2_citation_source.empty:

                c3_columns = [
                    "arxiv_id",
                    "s2_citation_count_3y",
                    "s2_influential_citation_count_3y",
                    "s2_c3_status",
                    "s2_ic3_status"
                ]

                c3_columns = [
                    c for c in c3_columns
                    if c in s2_citation_source.columns
                ]

                s2_metadata_source = s2_metadata_source.merge(
                    s2_citation_source[c3_columns],
                    on="arxiv_id",
                    how="left",
                    validate="one_to_one"
                )

            s2_source = s2_metadata_source

        else:
            # Last-resort fallback. This file contains matched papers only;
            # the left join still preserves every arXiv row in master_df.
            s2_source = _master_get_df(
                "s2_df",
                s2_output_dir / "semantic_scholar_papers.csv"
            )

    # Attach metadata match audit status so API errors are not
    # misreported as genuine no-match cases.
    s2_audit_source = _master_get_df(
        "s2_audit_df",
        s2_output_dir / "s2_metadata_audit.csv"
    )

    if (
        not s2_source.empty
        and not s2_audit_source.empty
        and "status" in s2_audit_source.columns
    ):
        s2_audit_small = s2_audit_source[
            ["arxiv_id", "status"]
        ].copy().rename(
            columns={"status": "s2_metadata_status"}
        )

        s2_audit_small["arxiv_id"] = (
            s2_audit_small["arxiv_id"].map(
                _master_normalize_arxiv_id
            )
        )

        s2_source = s2_source.merge(
            s2_audit_small,
            on="arxiv_id",
            how="left",
            validate="one_to_one"
        )

    master_df = _master_merge(
        master_df,
        s2_source,
        "SEMANTIC_SCHOLAR"
    )

    if "s2_paper_id" in master_df.columns:
        s2_matched = master_df["s2_paper_id"].notna()
    else:
        s2_matched = pd.Series(False, index=master_df.index)

    if "s2_metadata_status" in master_df.columns:
        master_df["s2_match_status"] = master_df[
            "s2_metadata_status"
        ].fillna(
            pd.Series(
                np.where(s2_matched, "MATCHED", "NO_S2_RECORD_RETURNED"),
                index=master_df.index
            )
        )
    else:
        master_df["s2_match_status"] = np.where(
            s2_matched, "MATCHED", "NO_S2_RECORD_RETURNED"
        )


    # ------------------------------------------------------------
    # D. OpenReview: paper-level mean rating, one row per paper
    # ------------------------------------------------------------

    openreview_dir = PROCESSED / "openreview_ratings"

    openreview_source = _master_get_df(
        "openreview_paper_ratings_df",
        openreview_dir / "openreview_paper_ratings.csv"
    )

    master_df = _master_merge(
        master_df,
        openreview_source,
        "OPENREVIEW"
    )

    if "openreview_score_status" not in master_df.columns:
        master_df["openreview_score_status"] = (
            "NOT_COLLECTED_OR_AGGREGATION_NOT_RUN"
        )


    # ------------------------------------------------------------
    # E. Optional OpenAlex merge
    # ------------------------------------------------------------
    # This notebook currently contains an OpenAlex section heading,
    # but no executable OpenAlex collection cell. Do not fabricate
    # OpenAlex matches or treat missing OpenAlex data as zero.

    openalex_source = None

    if isinstance(openalex_source, pd.DataFrame) and not openalex_source.empty:

        master_df = _master_merge(
            master_df,
            openalex_source,
            "OPENALEX"
        )

        if "openalex_work_id" in master_df.columns:
            master_df["openalex_collection_status"] = np.where(
                master_df["openalex_work_id"].notna(),
                "MATCHED",
                "NOT_MATCHED"
            )
        else:
            master_df["openalex_collection_status"] = (
                "OPENALEX_TABLE_MERGED_ID_FIELD_UNAVAILABLE"
            )

    else:
        master_df["openalex_collection_status"] = (
            "NOT_COLLECTED_IN_CURRENT_NOTEBOOK"
        )


    # ------------------------------------------------------------
    # F. Deterministic arXiv v1 features
    # ------------------------------------------------------------

    if "title_v1" in master_df.columns:
        master_df["title_word_count_v1"] = (
            master_df["title_v1"]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
        )

    if "abstract_v1" in master_df.columns:
        master_df["abstract_word_count_v1"] = (
            master_df["abstract_v1"]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
        )

    if "author_count_v1" in master_df.columns:
        master_df["author_count_v1"] = pd.to_numeric(
            master_df["author_count_v1"],
            errors="coerce"
        )

    # Explicit label-validity flags. Zero-confirmed is a valid zero.
    valid_label_statuses = {
        "COMPLETE",
        "ZERO_CONFIRMED_BY_CUMULATIVE_COUNT"
    }

    if "s2_c3_status" in master_df.columns:
        master_df["s2_c3_label_valid"] = (
            master_df["s2_c3_status"].isin(valid_label_statuses)
        )

    if "s2_ic3_status" in master_df.columns:
        master_df["s2_ic3_label_valid"] = (
            master_df["s2_ic3_status"].isin(valid_label_statuses)
        )


    # ------------------------------------------------------------
    # G. Final integrity checks
    # ------------------------------------------------------------

    if len(master_df) != master_input_n:
        raise AssertionError(
            f"Master row count changed: {master_input_n} -> {len(master_df)}"
        )

    if master_df["arxiv_id"].isna().any():
        raise AssertionError("Master table contains missing arxiv_id.")

    if master_df["arxiv_id"].duplicated().any():
        raise AssertionError("Master table contains duplicate arxiv_id.")

    if {
        "s2_citation_count_3y",
        "s2_influential_citation_count_3y"
    }.issubset(master_df.columns):

        both_labels = master_df.dropna(
            subset=[
                "s2_citation_count_3y",
                "s2_influential_citation_count_3y"
            ]
        )

        invalid_order = both_labels[
            both_labels["s2_citation_count_3y"]
            < both_labels["s2_influential_citation_count_3y"]
        ]

        if not invalid_order.empty:
            raise AssertionError(
                "Found papers with C3 < IC3 in the merged master table."
            )


    # ------------------------------------------------------------
    # H. Save final one-row-per-paper master table
    # ------------------------------------------------------------

    master_csv_path = PROCESSED / "papers_master_raw_stage.csv"
    master_jsonl_path = PROCESSED / "papers_master_raw_stage.jsonl"
    master_metadata_path = PROCESSED / "papers_master_metadata.json"

    atomic_csv(master_df, 
        master_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    master_df.to_json(
        master_jsonl_path,
        orient="records",
        lines=True,
        force_ascii=False,
        date_format="iso"
    )

    master_metadata = {
        "dataset": "GP paper-level master table",
        "unit_of_observation": "one row per arXiv paper",
        "primary_key": "arxiv_id",
        "row_count": int(len(master_df)),
        "unique_arxiv_id_count": int(master_df["arxiv_id"].nunique()),
        "sources": {
            "arxiv_v1": "papers_arxiv_v1.csv",
            "dblp": "dblp_author_features_v3_1_1/dblp_author_features.csv",
            "semantic_scholar": "semantic_scholar_public",
            "openreview": "openreview_ratings/openreview_paper_ratings.csv",
            "openalex": (
                "Merged only if openalex_work_df exists; "
                "not collected by the current notebook cells."
            )
        },
        "label_definitions": {
            "C3": "Unique citing papers within the three-year window; incomplete labels remain missing.",
            "IC3": "Unique influential citing papers within the same three-year window; incomplete labels remain missing.",
            "R": "OpenReview paper-level mean rating; papers without a numeric public rating remain missing."
        },
        "merge_policy": "Left joins from the arXiv v1 paper table; every source must be one row per arxiv_id.",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat()
    }

    master_metadata_path.write_text(
        json.dumps(master_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("FINAL MASTER TABLE")
    print("=" * 65)
    print("Rows:", len(master_df))
    print("Unique arXiv IDs:", master_df["arxiv_id"].nunique())
    print("Columns:", len(master_df.columns))
    print("Duplicate arXiv IDs:", master_df["arxiv_id"].duplicated().sum())
    print("C3 valid labels:", int(master_df.get("s2_c3_label_valid", pd.Series(False, index=master_df.index)).sum()))
    print("IC3 valid labels:", int(master_df.get("s2_ic3_label_valid", pd.Series(False, index=master_df.index)).sum()))
    print("OpenReview mean ratings:", int(master_df.get("openreview_mean_rating", pd.Series(index=master_df.index, dtype="float64")).notna().sum()))
    print("OpenAlex:", master_df["openalex_collection_status"].iloc[0] if len(master_df) else "NO_ROWS")

    print("\nSaved:")
    print(master_csv_path)
    print(master_jsonl_path)
    print(master_metadata_path)

    log_preview(master_df.head(5))

    return {"master_df": master_df}

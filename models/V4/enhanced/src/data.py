"""Load, validate and derive the enhanced model's data features."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from .constants import (
    DBLP_COLUMNS,
    EXCLUDED_FROM_X,
    EXTENDED_EXTERNAL_NUMERIC_FEATURES,
    EXTENDED_NUMERIC_FEATURES,
    OPTIONAL_EXTERNAL_COLUMNS,
    REQUIRED_COLUMNS,
    STRICT_T0_EXTERNAL_NUMERIC_FEATURES,
    STRICT_T0_NUMERIC_FEATURES,
    TARGETS,
    TEST_YEARS,
    TRAIN_YEARS,
)


@dataclass(frozen=True)
class TaskData:
    key: str
    column: str
    label: str
    train: pd.DataFrame
    test: pd.DataFrame
    full: pd.DataFrame


def _parse_categories(value: Any, primary: str) -> List[str]:
    labels: List[str] = []
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                labels = [str(item) for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            labels = []
    if primary and primary not in labels:
        labels.append(str(primary))
    return sorted(set(labels))


def _parse_authors(value: Any) -> Tuple[List[str], int, List[str]]:
    names: List[str] = []
    affs: List[str] = []
    has_aff = 0
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        nm = str(item.get("name") or "").strip().lower()
                        if nm:
                            names.append(nm)
                        af = str(item.get("affiliation_raw") or "").strip().lower()
                        if af:
                            has_aff = 1
                            affs.append(af)
        except json.JSONDecodeError:
            pass
    return names, has_aff, affs


def _parse_json_str_list(value: Any) -> List[str]:
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x).strip().lower() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return []


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the 25-column (or 47-column enriched) table, validate contract, and derive base features.

    ``paper_id`` is always read as text; pandas numeric inference would destroy
    IDs with trailing zeroes such as 1701.00010.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, dtype={"paper_id": "string"}, low_memory=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing source columns: {missing}")
    allowed = set(REQUIRED_COLUMNS) | set(OPTIONAL_EXTERNAL_COLUMNS)
    unexpected = [column for column in df.columns if column not in allowed]
    if unexpected:
        raise ValueError(f"Unexpected source columns; refusing to guess: {unexpected}")

    df = df.copy()
    df["paper_id"] = df["paper_id"].astype("string").str.strip()
    if df["paper_id"].isna().any() or df["paper_id"].eq("").any():
        raise ValueError("paper_id contains missing/blank values.")
    if df["paper_id"].duplicated().any():
        raise ValueError(f"paper_id has {df['paper_id'].duplicated().sum()} duplicates.")

    for column in ("submission_year", "author_count_v1",
                   "title_word_count_v1", "abstract_word_count_v1"):
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["submission_year"] = df["submission_year"].astype(int)
    years = set(TRAIN_YEARS) | set(TEST_YEARS)
    if not set(df["submission_year"].unique()) <= years:
        raise ValueError("submission_year outside 2017-2022.")

    for key, spec in TARGETS.items():
        column = spec["column"]
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = df[column].notna() & numeric.isna()
        if invalid.any():
            examples = df.loc[invalid, column].astype(str).head(5).tolist()
            raise ValueError(f"{key} has non-numeric values: {examples}")
        df[column] = numeric.astype(float)
        if (df[column].dropna() < 0).any():
            raise ValueError(f"{key} has negative values.")
        if spec["task"] == "count" and (df[column].dropna() % 1 != 0).any():
            raise ValueError(f"{key} must contain integer counts.")

    y1 = df["s2_citation_count_3y"]
    y2 = df["s2_influential_citation_count_3y"]
    both = y1.notna() & y2.notna()
    if (y2[both] > y1[both]).any():
        raise ValueError("IC3 exceeds C3 for at least one row.")

    for mean_col, max_col in (
        ("dblp_team_mean_publication_count_pre_t0",
         "dblp_team_max_publication_count_pre_t0"),
        ("dblp_team_mean_academic_age_pre_t0",
         "dblp_team_max_academic_age_pre_t0"),
    ):
        present = df[mean_col].notna() & df[max_col].notna()
        if (df.loc[present, max_col] < df.loc[present, mean_col]).any():
            raise ValueError(f"{max_col} < {mean_col}.")

    if (df["author_count_v1"] < 1).any():
        raise ValueError("author_count_v1 < 1.")
    if (df["title_word_count_v1"] < 1).any():
        raise ValueError("title_word_count_v1 < 1.")
    if (df["abstract_word_count_v1"] < 1).any():
        raise ValueError("abstract_word_count_v1 < 1.")

    # Parse authors and affiliation presence
    parsed_authors: List[List[str]] = []
    has_aff_list: List[int] = []
    raw_aff_lists: List[List[str]] = []
    for raw_au in df["authors_v1_json"]:
        aus, aff, aff_names = _parse_authors(raw_au)
        parsed_authors.append(aus)
        has_aff_list.append(aff)
        raw_aff_lists.append(aff_names)
    df["parsed_authors"] = parsed_authors
    df["has_author_affiliation"] = np.asarray(has_aff_list, dtype=np.float32)

    # Parse t0_date seasonality and exact days since 2017-01-01 for As-Of Temporal Encoding
    t0_dt = pd.to_datetime(df["t0_date"], errors="coerce", utc=True)
    t0_month = t0_dt.dt.month.fillna(6).to_numpy(dtype=np.float64)
    df["t0_month"] = t0_month
    df["t0_month_sin"] = np.sin(2.0 * np.pi * t0_month / 12.0)
    df["t0_month_cos"] = np.cos(2.0 * np.pi * t0_month / 12.0)
    fallback_days = (df["submission_year"].to_numpy(dtype=np.float64) - 2017.0) * 365.25 + (t0_month - 0.5) * 30.4375
    raw_days = (t0_dt - pd.Timestamp("2017-01-01", tz="UTC")).dt.total_seconds() / 86400.0
    df["t0_days"] = np.where(np.isfinite(raw_days), raw_days, fallback_days)
    df["t0_bucket"] = np.floor(df["t0_days"] / 60.875).astype(int)

    # Parse categories
    cat_labels: List[List[str]] = []
    for all_labels, cs_labels, primary in zip(
        df["categories_v1_json"],
        df["cs_categories_v1_json"],
        df["primary_category_v1"].astype(str),
    ):
        c1 = _parse_categories(all_labels, primary)
        c2 = _parse_categories(cs_labels, primary)
        cat_labels.append(sorted(set(c1) | set(c2)))
    df["category_labels"] = cat_labels
    df["category_count"] = np.asarray([len(c) for c in cat_labels], dtype=np.float32)
    df["is_interdisciplinary"] = np.asarray(
        [1.0 if len({lbl.split(".")[0] for lbl in c}) > 1 else 0.0 for c in cat_labels],
        dtype=np.float32,
    )

    # Structured DBLP & team ratios
    au_cnt = df["author_count_v1"].astype(float)
    d_mean = df["dblp_team_mean_publication_count_pre_t0"].astype(float)
    d_max = df["dblp_team_max_publication_count_pre_t0"].astype(float)
    d_age_mean = df["dblp_team_mean_academic_age_pre_t0"].astype(float)
    d_age_max = df["dblp_team_max_academic_age_pre_t0"].astype(float)

    df["author_count_log1p"] = np.log1p(au_cnt)
    df["dblp_mean_publication_log1p"] = np.log1p(d_mean)
    df["dblp_max_publication_log1p"] = np.log1p(d_max)
    df["dblp_team_total_pub_log1p"] = np.log1p(au_cnt * d_mean)
    df["dblp_star_ratio"] = d_max / (d_mean + 1.0)
    df["dblp_productivity_rate"] = np.log1p(d_max / (d_age_max + 1.0))
    df["dblp_mean_academic_age"] = d_age_mean
    df["dblp_max_academic_age"] = d_age_max
    df["dblp_features_missing"] = (
        df[list(DBLP_COLUMNS)].isna().all(axis=1).astype(int)
    )

    # Text fields and lexical/domain indicators from title + abstract
    title_clean = df["title_v1"].fillna("").astype(str).str.strip()
    abstract_clean = df["abstract_v1"].fillna("").astype(str).str.strip()
    comment_clean = df["comment_v1"].fillna("").astype(str).str.strip()

    df["text_t0"] = title_clean + " [SEP] " + abstract_clean
    df["text_all"] = (
        title_clean
        + " [SEP] "
        + abstract_clean
        + np.where(comment_clean != "", " [SEP] " + comment_clean, "")
    )

    t0_lower = (title_clean + " " + abstract_clean).str.lower()
    df["text_has_code_or_url"] = t0_lower.str.contains(
        r"github\.com|https?://|code is available|open-source|open source|project page",
        regex=True,
    ).astype(float)
    df["text_has_sota_claim"] = t0_lower.str.contains(
        r"state-of-the-art|state of the art|outperforms|surpasses|new benchmark",
        regex=True,
    ).astype(float)
    df["text_has_theory_claim"] = t0_lower.str.contains(
        r"theorem|theoretical|provably|convergence|regret bound|minimax|optimal rate",
        regex=True,
    ).astype(float)
    df["text_is_survey_or_benchmark"] = t0_lower.str.contains(
        r"\bsurvey\b|\bbenchmark\b|\bdataset\b|\btoolkit\b|\boverview\b|\breview\b",
        regex=True,
    ).astype(float)
    df["text_has_named_model_colon"] = (
        title_clean.str.match(r"^[A-Za-z0-9\-\+]{2,20}:\s+")
        & (~title_clean.str.lower().str.match(
            r"^(revisiting|towards|on|beyond|understanding|learning|survey|review|comment|reply|erratum|corrigendum):"
        ))
    ).astype(float)
    df["text_is_challenge_report"] = (
        (title_clean.str.lower().str.contains("challenge") & (au_cnt >= 15))
        | (au_cnt > 55)
    ).astype(float)

    # Extended metadata features from papers_features.csv
    c_lower = comment_clean.str.lower()
    df["reference_count_log1p"] = np.log1p(
        pd.to_numeric(df["s2_reference_count_current"], errors="coerce")
        .clip(lower=0.0, upper=400.0)
    )
    df["comment_present"] = (comment_clean != "").astype(float)
    pages_extracted = c_lower.str.extract(r"(\d+)\s*pages?")[0].astype(float).clip(1.0, 80.0)
    figs_extracted = c_lower.str.extract(r"(\d+)\s*figures?")[0].astype(float).clip(0.0, 50.0)
    df["comment_pages_raw"] = pages_extracted
    df["comment_pages_log1p"] = np.log1p(pages_extracted)
    df["comment_figures_log1p"] = np.log1p(figs_extracted)
    df["comment_accepted"] = c_lower.str.contains(
        r"accepted|published as a conference|to appear in|camera-ready|camera ready|oral|spotlight|best paper",
        regex=True,
    ).astype(float)
    df["comment_top_venue"] = c_lower.str.contains(
        r"\biclr\b|\bneurips\b|\bnips\b|\bicml\b|\bcvpr\b|\biccv\b|\beccv\b|\bacl\b|\bemnlp\b|\bnaacl\b|\bkdd\b|\baaai\b|\bijcai\b",
        regex=True,
    ).astype(float)
    df["comment_oral_spotlight"] = c_lower.str.contains(
        r"\boral\b|\bspotlight\b|best paper|notable",
        regex=True,
    ).astype(float)
    df["comment_submitted_only"] = (
        c_lower.str.contains(r"submitted to", regex=True)
        & (~c_lower.str.contains(r"accepted|published", regex=True))
    ).astype(float)
    df["has_journal_ref"] = df["journal_ref_v1"].notna().astype(float)
    df["has_doi"] = df["doi_v1"].notna().astype(float)
    df["dblp_published_same_year"] = (
        df["dblp_temporal_status"] == "SAME_CALENDAR_YEAR_AS_ARXIV_V1"
    ).astype(float)
    df["dblp_published_after"] = (
        df["dblp_temporal_status"] == "PUBLISHED_AFTER_ARXIV_V1"
    ).astype(float)
    df["dblp_previously_published_int"] = (
        df["dblp_previously_published_before_arxiv_v1"].astype(bool).astype(float)
    )

    # Optional external features from `scripts/fetch_external_features.py` (supports both 47-col full and 36-col leak-free tables)
    if "oa_matched" in df.columns:
        def _col(col_name: str, default_val: float = 0.0) -> pd.Series:
            if col_name in df.columns:
                s = pd.to_numeric(df[col_name], errors="coerce")
                return s.fillna(default_val) if not math.isnan(default_val) else s
            return pd.Series(default_val, index=df.index, dtype=float)

        df["oa_institutions_count_log1p"] = np.log1p(_col("oa_institutions_count", 0.0).clip(0.0, 100.0))
        df["oa_countries_count_log1p"] = np.log1p(_col("oa_countries_count", 0.0).clip(0.0, 50.0))
        df["oa_has_top_industry_lab"] = _col("oa_has_top_industry_lab", 0.0).clip(0.0, 1.0)
        df["oa_has_top_university"] = _col("oa_has_top_university", 0.0).clip(0.0, 1.0)
        df["s2_ref_recent_2y_ratio"] = _col("s2_ref_recent_2y_ratio", math.nan).clip(0.0, 1.0)
        df["pwc_has_official_code"] = _col("pwc_has_official_code", 0.0).clip(0.0, 1.0)
        df["pwc_is_pytorch"] = _col("pwc_is_pytorch", 0.0).clip(0.0, 1.0)
        df["pwc_is_tensorflow"] = _col("pwc_is_tensorflow", 0.0).clip(0.0, 1.0)
        if "oa_referenced_works_count" in df.columns:
            oa_ref = _col("oa_referenced_works_count", 0.0).clip(0.0, 400.0)
            s2_ref = pd.to_numeric(df["s2_reference_count_current"], errors="coerce").fillna(0.0).clip(0.0, 400.0)
            eff_ref = np.maximum(s2_ref, oa_ref)
            df["s2_reference_count_current"] = eff_ref
            df["reference_count_log1p"] = np.log1p(eff_ref)

        if "oa_early_citations_t0_year" in df.columns:
            df["oa_early_citations_t0_year_log1p"] = np.log1p(_col("oa_early_citations_t0_year", 0.0).clip(0.0, 500.0))
            df["oa_early_citations_t0_plus1_log1p"] = np.log1p(_col("oa_early_citations_t0_plus1", 0.0).clip(0.0, 1500.0))
            df["oa_referenced_works_count_log1p"] = np.log1p(_col("oa_referenced_works_count", 0.0).clip(0.0, 400.0))
            df["s2_ref_citation_max_log1p"] = np.log1p(_col("s2_ref_citation_max", math.nan).clip(0.0, 200000.0))
            df["s2_ref_citation_mean_log1p"] = np.log1p(_col("s2_ref_citation_mean", math.nan).clip(0.0, 50000.0))
            df["s2_ref_citation_median_log1p"] = np.log1p(_col("s2_ref_citation_median", math.nan).clip(0.0, 25000.0))
            df["s2_author_max_hindex_log1p"] = np.log1p(_col("s2_author_max_hindex", math.nan).clip(0.0, 300.0))
            df["s2_author_mean_hindex_log1p"] = np.log1p(_col("s2_author_mean_hindex", math.nan).clip(0.0, 250.0))
            df["s2_author_max_citations_log1p"] = np.log1p(_col("s2_author_max_citations", math.nan).clip(0.0, 500000.0))
            df["pwc_repo_count_log1p"] = np.log1p(_col("pwc_repo_count", 0.0).clip(0.0, 50.0))
            df["pwc_mentioned_in_github"] = _col("pwc_mentioned_in_github", 0.0).clip(0.0, 1.0)

        parsed_insts: List[List[str]] = []
        raw_inst_col = df["oa_institutions_json"] if "oa_institutions_json" in df.columns else [None] * len(df)
        for raw_ij, fallback_affs in zip(raw_inst_col, raw_aff_lists):
            ij_list = _parse_json_str_list(raw_ij)
            parsed_insts.append(ij_list if ij_list else fallback_affs)
        df["parsed_institutions"] = parsed_insts

    return df


def task_frame(df: pd.DataFrame, key: str) -> TaskData:
    if key not in TARGETS:
        raise KeyError(key)
    spec = TARGETS[key]
    observed = df.loc[df[spec["column"]].notna()].copy()
    train = observed.loc[observed["submission_year"].isin(TRAIN_YEARS)].copy()
    test = observed.loc[observed["submission_year"].isin(TEST_YEARS)].copy()
    if train.empty or test.empty:
        raise ValueError(f"{key}: empty train/test.")
    return TaskData(
        key=key, column=spec["column"], label=spec["label"],
        train=train, test=test, full=observed,
    )


def annual_thresholds(frame: pd.DataFrame, column: str) -> Dict[int, float]:
    thresholds = {}
    for year in TRAIN_YEARS:
        values = frame.loc[frame["submission_year"] == year, column].dropna()
        if not values.empty:
            thresholds[year] = float(values.quantile(0.90))
    return thresholds


def success_labels(
    frame: pd.DataFrame,
    column: str,
    thresholds: Mapping[int, float],
) -> np.ndarray:
    per_row = frame["submission_year"].map(thresholds).to_numpy(float)
    return (frame[column].to_numpy(float) >= per_row).astype(int)


def test_reference_thresholds(
    frame: pd.DataFrame,
    column: str,
) -> Dict[int, float]:
    result = {}
    for year in TEST_YEARS:
        values = frame.loc[frame["submission_year"] == year, column].dropna()
        if not values.empty:
            result[year] = float(values.quantile(0.90))
    return result


def data_audit(df: pd.DataFrame, strict_t0: bool = False) -> Dict[str, Any]:
    has_ext = "oa_matched" in df.columns
    has_ext_full = "oa_early_citations_t0_year" in df.columns
    if strict_t0:
        num_feats = list(STRICT_T0_NUMERIC_FEATURES) + (
            list(STRICT_T0_EXTERNAL_NUMERIC_FEATURES) if has_ext else []
        )
    else:
        ext_list = (
            list(EXTENDED_EXTERNAL_NUMERIC_FEATURES)
            if has_ext_full
            else (list(STRICT_T0_EXTERNAL_NUMERIC_FEATURES) if has_ext else [])
        )
        num_feats = list(EXTENDED_NUMERIC_FEATURES) + ext_list
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "has_external_features": bool(has_ext),
        "paper_id_unique": bool(df["paper_id"].is_unique),
        "years": {
            str(year): int((df["submission_year"] == year).sum())
            for year in sorted(df["submission_year"].unique())
        },
        "targets": {
            key: {
                "column": spec["column"],
                "available": int(df[spec["column"]].notna().sum()),
                "missing": int(df[spec["column"]].isna().sum()),
                "rate": float(df[spec["column"]].notna().mean()),
            }
            for key, spec in TARGETS.items()
        },
        "all_three_available": int(
            df[[spec["column"] for spec in TARGETS.values()]].notna().all(axis=1).sum()
        ),
        "dblp_missing": int(df["dblp_features_missing"].sum()),
        "numeric_features": num_feats,
        "author_graph_features": "2-hop co-author collaboration graph with Lp-norm aggregation, shared-paper deduplication, challenge-paper cap, and cross-paper DBLP recovery (fitted on training rows only)",
        "emerging_paradigm_features": "Temporal emerging high-impact n-gram encoder contrasting late vs early training sub-windows",
        "category_source": "primary + categories_v1_json + cs_categories_v1_json, multi-hot",
        "text_source": "title_v1 + [SEP] + abstract_v1 (+ comment_v1 in full mode); TF-IDF and optional BGE embeddings",
        "excluded": dict(EXCLUDED_FROM_X),
        "split": {
            "train_years": list(TRAIN_YEARS),
            "test_years": list(TEST_YEARS),
            "validation_folds": [
                {"train": [2017, 2018], "validation": 2019},
                {"train": [2017, 2018, 2019], "validation": 2020},
            ],
        },
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, spec in TARGETS.items():
        for year in sorted(df["submission_year"].unique()):
            group = df.loc[df["submission_year"] == year]
            available = int(group[spec["column"]].notna().sum())
            rows.append({
                "target": key, "submission_year": int(year),
                "available": available, "total": int(len(group)),
                "rate": available / len(group) if len(group) else math.nan,
            })
    return pd.DataFrame(rows)

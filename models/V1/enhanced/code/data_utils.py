"""Data contract, temporal split, and success-threshold utilities.

The authoritative input is ``gp_data/papers_features.csv`` (29,856 x 25).
Rows are split chronologically: 2017-2020 train and 2021-2022 test.
The 2020 cohort is used as the chronological validation subset for XGBoost
hyper-parameter selection.  Nothing in this module fits a model or a learned
transformer, so it is safe to use for auditing before training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

TRAIN_YEARS = (2017, 2018, 2019, 2020)
TEST_YEARS = (2021, 2022)
VALIDATION_YEAR = 2020
INNER_TRAIN_YEARS = (2017, 2018, 2019)

TARGETS: Dict[str, Dict[str, Any]] = {
    "Y1": {
        "column": "s2_citation_count_3y",
        "label": "C3 (three-year citations)",
        "log_target": True,
        "clip_nonnegative": True,
    },
    "Y2": {
        "column": "s2_influential_citation_count_3y",
        "label": "IC3 (three-year influential citations)",
        "log_target": True,
        "clip_nonnegative": True,
    },
    "Y3": {
        "column": "openreview_mean_rating",
        "label": "R (OpenReview mean rating)",
        "log_target": False,
        "clip_nonnegative": False,
    },
}

TARGET_COLUMNS = tuple(spec["column"] for spec in TARGETS.values())

REQUIRED_COLUMNS = (
    "paper_id",
    "title_v1",
    "abstract_v1",
    "primary_category_v1",
    "categories_v1_json",
    "cs_categories_v1_json",
    "authors_v1_json",
    "author_count_v1",
    "comment_v1",
    "journal_ref_v1",
    "doi_v1",
    "submission_year",
    "t0_date",
    "dblp_temporal_status",
    "dblp_previously_published_before_arxiv_v1",
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
)

DBLP_COLUMNS = (
    "dblp_team_mean_publication_count_pre_t0",
    "dblp_team_max_publication_count_pre_t0",
    "dblp_team_mean_academic_age_pre_t0",
    "dblp_team_max_academic_age_pre_t0",
)

# Model inputs.  Raw text is consumed by the TF-IDF block, not as a numeric
# column; submission_year is a t0 cohort control and the split/threshold key.
NUMERIC_FEATURES = (
    "submission_year",
    "author_count_log1p",
    "title_word_count_v1",
    "abstract_word_count_v1",
    "dblp_mean_publication_log1p",
    "dblp_max_publication_log1p",
    "dblp_mean_academic_age",
    "dblp_max_academic_age",
    "dblp_features_missing",
)
DERIVED_COLUMNS = (
    "author_count_log1p",
    "dblp_mean_publication_log1p",
    "dblp_max_publication_log1p",
    "dblp_mean_academic_age",
    "dblp_max_academic_age",
    "dblp_features_missing",
    "text_all",
)
CATEGORICAL_COLUMN = "primary_category_v1"
TEXT_COLUMN = "text_all"

# Kept in the table for audit only, never in X.
EXCLUDED_FROM_X: Dict[str, str] = {
    "paper_id": "Primary key; not a feature.",
    "title_v1": "Raw text; used only through TF-IDF.",
    "abstract_v1": "Raw text; used only through TF-IDF.",
    "categories_v1_json": "Redundant with primary_category_v1.",
    "cs_categories_v1_json": "Redundant with primary_category_v1 for this baseline.",
    "authors_v1_json": "Identity metadata; author_count_v1 is the model feature.",
    "comment_v1": "Optional venue/status metadata; possible post-t0 information.",
    "journal_ref_v1": "Optional venue metadata; not part of the t0 content/team feature set.",
    "doi_v1": "Identifier/optional metadata; not predictive at t0.",
    "t0_date": "Time key used to construct the split; not a content feature.",
    "dblp_temporal_status": "DBLP matching/audit field.",
    "dblp_previously_published_before_arxiv_v1": "DBLP audit field; not a baseline predictor.",
    "s2_reference_count_current": "Current-time metadata; post-t0 and therefore leakage-prone.",
    "s2_citation_count_3y": "Target Y1.",
    "s2_influential_citation_count_3y": "Target Y2.",
    "openreview_mean_rating": "Target Y3.",
}


@dataclass(frozen=True)
class TaskFrame:
    key: str
    column: str
    label: str
    log_target: bool
    clip_nonnegative: bool
    train: pd.DataFrame
    test: pd.DataFrame


def find_project_root(start: Optional[Path] = None) -> Path:
    """Return the nearest directory containing ``gp_data`` or ``gp_model``.

    Entry scripts must also work when a user uploads them directly to
    ``/content`` in Colab, where ``Path.parents[2]`` does not exist.  This
    helper walks up at most the available ancestors and falls back to the
    script's own directory.
    """
    script_dir = (
        Path(start).expanduser().resolve()
        if start is not None
        else Path(__file__).resolve().parent
    )
    for candidate in [script_dir, *script_dir.parents]:
        if (candidate / "gp_data" / "papers_features.csv").is_file():
            return candidate
        if (candidate / "gp_model" / "code").is_dir():
            return candidate
    return script_dir



def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the latest GP feature table without coercing paper_id.

    ``paper_id`` is deliberately read as a string.  Reading it with pandas'
    default numeric inference would damage IDs such as ``1701.00010``.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, dtype={"paper_id": "string"}, low_memory=False)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    unexpected = [column for column in df.columns if column not in REQUIRED_COLUMNS]
    if unexpected:
        raise ValueError(
            "Unexpected columns detected; refusing to guess the schema: "
            f"{unexpected}"
        )

    df = df.copy()
    df["paper_id"] = df["paper_id"].astype("string").str.strip()
    if df["paper_id"].isna().any() or df["paper_id"].eq("").any():
        raise ValueError("paper_id contains missing or blank values.")
    if df["paper_id"].duplicated().any():
        duplicated = int(df["paper_id"].duplicated().sum())
        raise ValueError(f"paper_id contains {duplicated} duplicate values.")

    for column in ("submission_year", "author_count_v1", "title_word_count_v1",
                   "abstract_word_count_v1"):
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["submission_year"] = df["submission_year"].astype(int)
    valid_years = set(TRAIN_YEARS) | set(TEST_YEARS)
    observed_years = set(df["submission_year"].unique())
    if not observed_years <= valid_years:
        raise ValueError(
            f"submission_year outside the approved 2017-2022 range: "
            f"{sorted(observed_years - valid_years)}"
        )

    for key, spec in TARGETS.items():
        column = spec["column"]
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = df[column].notna() & numeric.isna()
        if invalid.any():
            examples = df.loc[invalid, column].astype(str).head(5).tolist()
            raise ValueError(
                f"{key} ({column}) contains non-numeric values: {examples}"
            )
        df[column] = numeric.astype(float)
        if (df[column].dropna() < 0).any():
            raise ValueError(f"{key} ({column}) contains negative values.")
        if spec["log_target"] and (df[column].dropna() % 1 != 0).any():
            raise ValueError(f"{key} ({column}) must contain integer counts.")

    y1 = df["s2_citation_count_3y"]
    y2 = df["s2_influential_citation_count_3y"]
    both = y1.notna() & y2.notna()
    bad_order = both & (y2 > y1)
    if bad_order.any():
        raise ValueError(
            f"IC3 exceeds C3 for {int(bad_order.sum())} rows; check label construction."
        )

    for mean_col, max_col in (
        ("dblp_team_mean_publication_count_pre_t0",
         "dblp_team_max_publication_count_pre_t0"),
        ("dblp_team_mean_academic_age_pre_t0",
         "dblp_team_max_academic_age_pre_t0"),
    ):
        both_present = df[mean_col].notna() & df[max_col].notna()
        if (df.loc[both_present, max_col] < df.loc[both_present, mean_col]).any():
            raise ValueError(f"{max_col} is below {mean_col} for at least one row.")

    if (df["author_count_v1"] < 1).any():
        raise ValueError("author_count_v1 must be >= 1.")
    if (df["title_word_count_v1"] < 1).any():
        raise ValueError("title_word_count_v1 must be >= 1.")
    if (df["abstract_word_count_v1"] < 1).any():
        raise ValueError("abstract_word_count_v1 must be >= 1.")

    df["author_count_log1p"] = np.log1p(df["author_count_v1"].astype(float))
    df["dblp_mean_publication_log1p"] = np.log1p(
        df["dblp_team_mean_publication_count_pre_t0"].astype(float)
    )
    df["dblp_max_publication_log1p"] = np.log1p(
        df["dblp_team_max_publication_count_pre_t0"].astype(float)
    )
    df["dblp_mean_academic_age"] = df[
        "dblp_team_mean_academic_age_pre_t0"
    ].astype(float)
    df["dblp_max_academic_age"] = df[
        "dblp_team_max_academic_age_pre_t0"
    ].astype(float)
    df["dblp_features_missing"] = (
        df[list(DBLP_COLUMNS)].isna().all(axis=1).astype(int)
    )
    df["text_all"] = (
        df["title_v1"].fillna("").astype(str).str.strip()
        + " "
        + df["abstract_v1"].fillna("").astype(str).str.strip()
    )
    return df


def task_frame(df: pd.DataFrame, key: str) -> TaskFrame:
    """Return the observed-label train/test subset for one target."""

    if key not in TARGETS:
        raise KeyError(f"Unknown task {key!r}; choose from {list(TARGETS)}")
    spec = TARGETS[key]
    observed = df.loc[df[spec["column"]].notna()].copy()
    train = observed.loc[observed["submission_year"].isin(TRAIN_YEARS)].copy()
    test = observed.loc[observed["submission_year"].isin(TEST_YEARS)].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"{key}: train and test must both be non-empty; "
            f"train={len(train)}, test={len(test)}"
        )
    return TaskFrame(
        key=key,
        column=spec["column"],
        label=spec["label"],
        log_target=bool(spec["log_target"]),
        clip_nonnegative=bool(spec["clip_nonnegative"]),
        train=train,
        test=test,
    )


def _p90(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(0.90))


def thresholds_for_task(task: TaskFrame) -> Dict[str, Any]:
    """Training-year p90 thresholds and the held-out threshold estimate.

    The estimated held-out threshold is the mean of the 2017-2020 annual
    thresholds for this target.  Test outcomes are used only to construct the
    *reference* success labels during final evaluation.
    """

    annual_train = {
        year: _p90(task.train.loc[task.train["submission_year"] == year, task.column])
        for year in TRAIN_YEARS
    }
    annual_test = {
        year: _p90(task.test.loc[task.test["submission_year"] == year, task.column])
        for year in TEST_YEARS
    }
    usable = [value for value in annual_train.values() if np.isfinite(value)]
    if not usable:
        raise ValueError(f"{task.key}: no training-year thresholds could be computed.")
    estimated = float(np.mean(usable))
    return {
        "target": task.key,
        "column": task.column,
        "annual_train": annual_train,
        "estimated_test": estimated,
        "annual_test_reference": annual_test,
    }


def threshold_rows(task: TaskFrame, thresholds: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year, value in thresholds["annual_train"].items():
        rows.append({
            "target": task.key,
            "year": year,
            "split": "train",
            "role": "annual_training_p90",
            "p90": value,
            "n": int(task.train.loc[task.train["submission_year"] == year, task.column]
                     .notna().sum()),
        })
    for year, value in thresholds["annual_test_reference"].items():
        rows.append({
            "target": task.key,
            "year": year,
            "split": "test",
            "role": "observed_test_p90_for_reference_labels_only",
            "p90": value,
            "n": int(task.test.loc[task.test["submission_year"] == year, task.column]
                     .notna().sum()),
        })
    rows.append({
        "target": task.key,
        "year": "2021-2022",
        "split": "test",
        "role": "estimated_heldout_threshold_mean_of_training_years",
        "p90": thresholds["estimated_test"],
        "n": int(len(task.test)),
    })
    return rows


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rows and target availability by cohort, for the run manifest."""
    rows: list[dict[str, Any]] = []
    for year in sorted(df["submission_year"].unique()):
        subset = df.loc[df["submission_year"] == year]
        row = {
            "submission_year": int(year),
            "split": "train" if year in TRAIN_YEARS else "test",
            "papers": int(len(subset)),
        }
        for key, spec in TARGETS.items():
            row[f"{key}_available"] = int(subset[spec["column"]].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def data_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """Serializable summary of the data contract and observed coverage."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "source_columns": int(len(REQUIRED_COLUMNS)),
        "derived_columns": list(DERIVED_COLUMNS),
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
                "available_rate": float(df[spec["column"]].notna().mean()),
            }
            for key, spec in TARGETS.items()
        },
        "all_three_targets_available": int(
            df[list(TARGET_COLUMNS)].notna().all(axis=1).sum()
        ),
        "dblp_features_missing": int(df["dblp_features_missing"].sum()),
        "primary_categories": int(df["primary_category_v1"].nunique()),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_feature": CATEGORICAL_COLUMN,
        "text_feature": "title_v1 + ' ' + abstract_v1 -> TF-IDF fitted on training rows only",
        "target_columns": list(TARGET_COLUMNS),
        "excluded_from_x": EXCLUDED_FROM_X,
        "split": {
            "train_years": list(TRAIN_YEARS),
            "test_years": list(TEST_YEARS),
            "xgboost_validation_year": VALIDATION_YEAR,
            "xgboost_inner_train_years": list(INNER_TRAIN_YEARS),
        },
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def coverage_by_target(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, spec in TARGETS.items():
        for year in sorted(df["submission_year"].unique()):
            subset = df.loc[df["submission_year"] == year]
            available = int(subset[spec["column"]].notna().sum())
            rows.append({
                "target": key,
                "submission_year": int(year),
                "available": available,
                "total": int(len(subset)),
                "rate": available / len(subset) if len(subset) else float("nan"),
            })
    return pd.DataFrame(rows)

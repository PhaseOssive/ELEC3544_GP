#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge the DBLP Missing Treatment output into papers_features.csv.

Default files are expected in the same folder as this script:
    papers_features.csv
    treated_team_features.csv

The merged output keeps the original 25-column papers_features schema.
The four DBLP feature columns are replaced by the official values from
treated_team_features.csv. The paper_id is restored from the canonical
arxiv_id in the recovery table.

If output overwrites the base input, a backup is created automatically.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_BASE = "papers_features.csv"
DEFAULT_DBLP = "treated_team_features.csv"
DEFAULT_OUTPUT = "papers_features.csv"

DBLP_FEATURES = [
    "dblp_team_mean_publication_count_pre_t0",
    "dblp_team_max_publication_count_pre_t0",
    "dblp_team_mean_academic_age_pre_t0",
    "dblp_team_max_academic_age_pre_t0",
]

BASE_REQUIRED = ["paper_id", *DBLP_FEATURES]
DBLP_REQUIRED = ["arxiv_id", *DBLP_FEATURES]


def resolve_path(value: str) -> Path:
    """Resolve relative paths against the folder containing this script."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def normalize_arxiv_id(value: object) -> str:
    """
    Normalize an arXiv ID for joining.

    In the supplied papers_features.csv, 2,997 modern arXiv IDs had lost
    trailing zero(s) from the five-digit sequence, e.g. 1701.0064.
    The treated DBLP table retains the canonical form, e.g. 1701.00640.

    For modern IDs only, a short numeric suffix is right-padded to five
    digits. Old-style IDs and other formats are kept unchanged.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = text.replace("https://arxiv.org/abs/", "")
    text = text.replace("http://arxiv.org/abs/", "")
    text = text.replace("https://export.arxiv.org/abs/", "")
    text = text.strip("/")
    if text.lower().startswith("arxiv:"):
        text = text.split(":", 1)[1].strip()
    if text.lower().endswith(".pdf"):
        text = text[:-4]
    import re
    text = re.sub(r"v\d+$", "", text)

    match = re.fullmatch(r"(\d{4})\.(\d{1,5})", text)
    if match:
        year_month, sequence = match.groups()
        text = f"{year_month}.{sequence.ljust(5, '0')}"

    return text


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def check_feature_values(dblp: pd.DataFrame) -> None:
    """Check that the four recovered feature columns are numeric and coherent."""
    for column in DBLP_FEATURES:
        numeric = pd.to_numeric(dblp[column], errors="coerce")
        invalid = dblp[column].notna() & numeric.isna()
        if invalid.any():
            raise ValueError(
                f"{column} contains {int(invalid.sum())} non-numeric, non-empty values."
            )
        negative = numeric < 0
        if negative.any():
            raise ValueError(
                f"{column} contains {int(negative.sum())} negative values."
            )

    pairs = [
        (
            "dblp_team_mean_publication_count_pre_t0",
            "dblp_team_max_publication_count_pre_t0",
        ),
        (
            "dblp_team_mean_academic_age_pre_t0",
            "dblp_team_max_academic_age_pre_t0",
        ),
    ]

    for mean_col, max_col in pairs:
        mean_values = pd.to_numeric(dblp[mean_col], errors="coerce")
        max_values = pd.to_numeric(dblp[max_col], errors="coerce")

        one_sided_missing = mean_values.isna() ^ max_values.isna()
        if one_sided_missing.any():
            raise ValueError(
                f"{mean_col} and {max_col} have "
                f"{int(one_sided_missing.sum())} one-sided missing pairs."
            )

        invalid_order = (
            mean_values.notna()
            & max_values.notna()
            & (mean_values > max_values + 1e-9)
        )
        if invalid_order.any():
            raise ValueError(
                f"{mean_col} exceeds {max_col} in "
                f"{int(invalid_order.sum())} rows."
            )


def next_backup_path(base_path: Path) -> Path:
    candidate = base_path.with_name(
        f"{base_path.stem}_before_dblp_merge{base_path.suffix}"
    )
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = base_path.with_name(
            f"{base_path.stem}_before_dblp_merge_{index}{base_path.suffix}"
        )
        if not candidate.exists():
            return candidate
        index += 1


def values_changed(old: pd.Series, new: pd.Series) -> int:
    old_num = pd.to_numeric(old, errors="coerce")
    new_num = pd.to_numeric(new, errors="coerce")
    both_missing = old_num.isna() & new_num.isna()
    both_present_equal = (
        old_num.notna()
        & new_num.notna()
        & np.isclose(
            old_num.fillna(0).to_numpy(dtype=float),
            new_num.fillna(0).to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        )
    )
    equal = both_missing | both_present_equal
    return int((~equal).sum())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge DBLP-treated features into papers_features.csv."
    )
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=f"Base table (default: {DEFAULT_BASE}, in the script folder).",
    )
    parser.add_argument(
        "--dblp",
        default=DEFAULT_DBLP,
        help=f"treated DBLP table (default: {DEFAULT_DBLP}, in the script folder).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT}, in the script folder).",
    )
    args = parser.parse_args()

    base_path = resolve_path(args.base)
    dblp_path = resolve_path(args.dblp)
    output_path = resolve_path(args.output)

    for path in [base_path, dblp_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    print("=" * 68)
    print("TREATED DBLP FEATURE MERGE")
    print("=" * 68)
    print("Base table:  ", base_path)
    print("Treated table:", dblp_path)
    print("Output file: ", output_path)

    # Read IDs as strings so that arXiv identifiers are never parsed as numbers.
    base = pd.read_csv(base_path, dtype={"paper_id": "string"})
    dblp = pd.read_csv(dblp_path, dtype={"arxiv_id": "string"})

    require_columns(base, BASE_REQUIRED, "Base table")
    require_columns(dblp, DBLP_REQUIRED, "treated DBLP table")

    if len(base.columns) != 25:
        raise ValueError(
            f"Expected the base table to have 25 columns, "
            f"but found {len(base.columns)}. No output was written."
        )

    if base["paper_id"].isna().any() or base["paper_id"].str.strip().eq("").any():
        raise ValueError("Base table contains missing/blank paper_id values.")

    if dblp["arxiv_id"].isna().any() or dblp["arxiv_id"].str.strip().eq("").any():
        raise ValueError("DBLP table contains missing/blank arxiv_id values.")

    if base["paper_id"].duplicated().any():
        raise ValueError("Base table contains duplicate paper_id values.")

    if dblp["arxiv_id"].duplicated().any():
        raise ValueError("DBLP table contains duplicate arxiv_id values.")

    if len(base) != len(dblp):
        raise ValueError(
            f"Row counts differ: base={len(base):,}, DBLP={len(dblp):,}. "
            "Please verify that these are the matching project files."
        )

    base = base.copy()
    dblp = dblp.copy()
    base["_merge_key"] = base["paper_id"].map(normalize_arxiv_id)
    dblp["_merge_key"] = dblp["arxiv_id"].map(normalize_arxiv_id)

    if base["_merge_key"].eq("").any() or dblp["_merge_key"].eq("").any():
        raise ValueError("One or more IDs could not be normalized.")

    if base["_merge_key"].duplicated().any():
        examples = base.loc[
            base["_merge_key"].duplicated(keep=False), "_merge_key"
        ].head(10).tolist()
        raise ValueError(
            f"Base IDs are not unique after normalization. Examples: {examples}"
        )

    if dblp["_merge_key"].duplicated().any():
        examples = dblp.loc[
            dblp["_merge_key"].duplicated(keep=False), "_merge_key"
        ].head(10).tolist()
        raise ValueError(
            f"DBLP IDs are not unique after normalization. Examples: {examples}"
        )

    base_keys = set(base["_merge_key"])
    dblp_keys = set(dblp["_merge_key"])
    base_only = sorted(base_keys - dblp_keys)
    dblp_only = sorted(dblp_keys - base_keys)

    if base_only or dblp_only:
        raise ValueError(
            "The two files do not contain exactly the same paper IDs after "
            "normalization.\n"
            f"Only in base ({len(base_only)}): {base_only[:10]}\n"
            f"Only in DBLP ({len(dblp_only)}): {dblp_only[:10]}\n"
            "No output was written."
        )

    check_feature_values(dblp)

    # Use the treated DBLP table as the authoritative source for the four
    # DBLP feature columns and canonical arXiv IDs. Keep all other base fields.
    dblp_lookup = dblp.set_index("_merge_key")
    merged = base.copy()

    original_values = {c: merged[c].copy() for c in DBLP_FEATURES}

    for column in DBLP_FEATURES:
        merged[column] = merged["_merge_key"].map(dblp_lookup[column])

    # Restore the canonical five-digit arXiv IDs from the recovery table.
    merged["paper_id"] = merged["_merge_key"].map(dblp_lookup["arxiv_id"])

    # Keep exactly the original 25 base columns, in their original order.
    output_columns = [c for c in base.columns if c != "_merge_key"]
    merged = merged[output_columns]

    if len(merged) != len(base) or len(merged.columns) != 25:
        raise RuntimeError("Final shape check failed; output was not written.")

    if merged["paper_id"].duplicated().any():
        raise RuntimeError("Final paper_id values are not unique.")

    # Write to a temporary file first; only replace the destination when the
    # complete output has been written successfully.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    merged.to_csv(temp_path, index=False, encoding="utf-8-sig")

    backup_path = None
    if output_path.resolve() == base_path.resolve():
        backup_path = next_backup_path(base_path)
        shutil.copy2(base_path, backup_path)

    temp_path.replace(output_path)

    repaired_ids = int(
        (base["paper_id"].astype(str) != merged["paper_id"].astype(str)).sum()
    )

    print("\nMerge completed successfully.")
    print(f"Rows: {len(merged):,}")
    print(f"Columns: {len(merged.columns)}")
    print(f"IDs restored to canonical form: {repaired_ids:,}")

    print("\nDBLP feature results:")
    for column in DBLP_FEATURES:
        missing_count = int(merged[column].isna().sum())
        changed_count = values_changed(original_values[column], merged[column])
        print(
            f"  {column}\n"
            f"    non-missing: {len(merged) - missing_count:,}\n"
            f"    missing:     {missing_count:,}\n"
            f"    changed:     {changed_count:,}"
        )

    if backup_path:
        print(f"\nOriginal base table backed up to:\n{backup_path}")

    print(f"\nFinal output:\n{output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

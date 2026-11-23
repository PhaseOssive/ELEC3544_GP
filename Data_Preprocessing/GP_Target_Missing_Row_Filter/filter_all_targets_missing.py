#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove rows only when all three prediction targets (Y1, Y2, Y3) are missing.

Default input:
    papers_features(4).csv

Default output:
    papers_features.csv

The script keeps a row if at least one of Y1, Y2, or Y3 is available.
Missing values in predictors and optional metadata do not cause row deletion.
The input file is not modified.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT = "papers_features.csv"
DEFAULT_OUTPUT = "papers_features.csv"

TARGETS = {
    "Y1": "s2_citation_count_3y",
    "Y2": "s2_influential_citation_count_3y",
    "Y3": "openreview_mean_rating",
}

EXPECTED_COLUMN_COUNT = 25


def resolve_path(value: str) -> Path:
    """Resolve relative paths against the folder containing this script."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")


def validate_target_columns(df: pd.DataFrame) -> None:
    """
    Ensure target columns contain numeric values or missing values.
    Do not silently turn unexpected text into missing values.
    """
    for label, column in TARGETS.items():
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = df[column].notna() & numeric.isna()
        if invalid.any():
            examples = df.loc[invalid, column].astype(str).head(5).tolist()
            raise ValueError(
                f"{label} ({column}) contains non-numeric, non-empty values. "
                f"Examples: {examples}"
            )


def next_backup_path(path: Path) -> Path:
    base = path.with_name(f"{path.stem}_before_target_filter{path.suffix}")
    if not base.exists():
        return base

    index = 2
    while True:
        candidate = path.with_name(
            f"{path.stem}_before_target_filter_{index}{path.suffix}"
        )
        if not candidate.exists():
            return candidate
        index += 1


def build_report(
    input_path: Path,
    output_path: Path,
    input_rows: int,
    output_rows: int,
    removed_rows: int,
    missing_counts: dict[str, int],
    all_missing_rows: int,
    both_or_more_available: int,
    exactly_one_available: int,
) -> str:
    lines = [
        "TARGET MISSINGNESS ROW FILTER REPORT",
        "=" * 48,
        f"Run time: {datetime.now().isoformat(timespec='seconds')}",
        f"Input: {input_path}",
        f"Output: {output_path}",
        "",
        f"Input rows: {input_rows:,}",
        f"Rows removed (Y1, Y2, and Y3 all missing): {removed_rows:,}",
        f"Output rows: {output_rows:,}",
        "",
        "Missing values before filtering:",
    ]
    for label, column in TARGETS.items():
        lines.append(f"  {label} ({column}): {missing_counts[label]:,}")

    lines.extend([
        "",
        f"Rows with all three targets missing: {all_missing_rows:,}",
        f"Rows with exactly one target available: {exactly_one_available:,}",
        f"Rows with at least two targets available: {both_or_more_available:,}",
        "",
        "Rule applied:",
        "  Remove a row only if Y1, Y2, and Y3 are all missing.",
        "  Keep a row if at least one of the three targets is available.",
        "  Missing predictor values and optional metadata do not trigger row deletion.",
        "",
        "The input file is not modified. The output retains the original columns and row order.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Keep papers with at least one available target among Y1, Y2, and Y3; "
            "remove only rows where all three targets are missing."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input CSV (default: {DEFAULT_INPUT}, in the script folder).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT}, in the script folder).",
    )
    parser.add_argument(
        "--report",
        default="target_missing_filter_report.txt",
        help="Summary report filename (default: target_missing_filter_report.txt).",
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    report_path = resolve_path(args.report)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print("=" * 68)
    print("TARGET MISSINGNESS ROW FILTER")
    print("=" * 68)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Read paper_id as text to preserve arXiv identifiers.
    df = pd.read_csv(input_path, dtype={"paper_id": "string"})

    require_columns(df, ["paper_id", *TARGETS.values()])

    if len(df.columns) != EXPECTED_COLUMN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COLUMN_COUNT} columns in the integrated main table, "
            f"but found {len(df.columns)}. No output was written."
        )

    if df["paper_id"].isna().any() or df["paper_id"].str.strip().eq("").any():
        raise ValueError("paper_id contains missing or blank values.")

    if df["paper_id"].duplicated().any():
        duplicate_count = int(df["paper_id"].duplicated().sum())
        raise ValueError(f"paper_id contains {duplicate_count:,} duplicate rows.")

    validate_target_columns(df)

    target_columns = list(TARGETS.values())
    missing_counts = {
        label: int(df[column].isna().sum())
        for label, column in TARGETS.items()
    }

    # The only deletion rule: all three outcome labels are missing.
    all_targets_missing = df[target_columns].isna().all(axis=1)
    keep_mask = ~all_targets_missing

    filtered = df.loc[keep_mask].copy()

    input_rows = len(df)
    output_rows = len(filtered)
    removed_rows = int(all_targets_missing.sum())

    # These two counts describe the target availability pattern in the kept table.
    available_per_row = df[target_columns].notna().sum(axis=1)
    exactly_one_available = int((available_per_row == 1).sum())
    both_or_more_available = int((available_per_row >= 2).sum())

    # Ensure the output has the same schema and preserves original row order.
    if list(filtered.columns) != list(df.columns):
        raise RuntimeError("Output schema changed unexpectedly.")
    if output_rows + removed_rows != input_rows:
        raise RuntimeError("Row-count reconciliation failed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file before moving it into place.
    temp_path = output_path.with_name(output_path.name + ".tmp")
    filtered.to_csv(temp_path, index=False, encoding="utf-8-sig")

    # If output already exists, back it up before replacing it.
    backup_path = None
    if output_path.exists():
        backup_path = next_backup_path(output_path)
        output_path.replace(backup_path)

    temp_path.replace(output_path)

    report_text = build_report(
        input_path=input_path,
        output_path=output_path,
        input_rows=input_rows,
        output_rows=output_rows,
        removed_rows=removed_rows,
        missing_counts=missing_counts,
        all_missing_rows=removed_rows,
        both_or_more_available=both_or_more_available,
        exactly_one_available=exactly_one_available,
    )
    report_path.write_text(report_text, encoding="utf-8")

    print("\nCompleted successfully.")
    print(f"Input rows: {input_rows:,}")
    print(f"Rows removed (all Y1/Y2/Y3 missing): {removed_rows:,}")
    print(f"Output rows: {output_rows:,}")
    print(f"Output columns: {len(filtered.columns)}")
    print("\nMissing target values before filtering:")
    for label, column in TARGETS.items():
        print(f"  {label}: {missing_counts[label]:,}")
    print(f"\nOutput CSV: {output_path}")
    print(f"Summary report: {report_path}")
    if backup_path:
        print(f"Previous output backed up to: {backup_path}")
    print("\nThe input file was not modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

#!/usr/bin/env python3
"""Audit the latest GP feature table and its modelling contract.

This script uses only pandas/numpy so it can run before installing the ML
dependencies.  It writes a JSON audit and a concise Markdown summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_utils import (
    DERIVED_COLUMNS,
    REQUIRED_COLUMNS,
    TEST_YEARS,
    TRAIN_YEARS,
    TARGETS,
    coverage_by_target,
    data_audit,
    find_project_root,
    load_dataset,
    split_summary,
)


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)
    parser = argparse.ArgumentParser(description="Audit papers_features.csv")
    parser.add_argument(
        "--data",
        default=str(first_existing([
            project_root / "gp_data" / "papers_features.csv",
            script_dir / "gp_data" / "papers_features.csv",
            script_dir / "papers_features.csv",
        ])),
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent),
        help="Directory for DATA_AUDIT.json and DATA_AUDIT.md.",
    )
    args = parser.parse_args()

    df = load_dataset(args.data)
    audit = data_audit(df)
    audit["source_rows"] = int(len(df))
    audit["derived_columns"] = list(DERIVED_COLUMNS)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "DATA_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    split = split_summary(df)
    coverage = coverage_by_target(df)
    split.to_csv(outdir / "DATA_SPLIT.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(outdir / "DATA_TARGET_COVERAGE.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Data audit",
        "",
        f"Source: `{Path(args.data).expanduser().resolve()}`",
        f"Rows: {audit['rows']:,}",
        f"Source columns: {len(REQUIRED_COLUMNS)}",
        f"Derived columns: {len(DERIVED_COLUMNS)}",
        f"All three targets observed: {audit['all_three_targets_available']:,}",
        f"DBLP four-feature missing: {audit['dblp_features_missing']:,}",
        "",
        "## Target coverage",
        "",
        "| Target | Column | Available | Missing | Rate |",
        "|---|---|---:|---:|---:|",
    ]
    for key, spec in TARGETS.items():
        lines.append(
            f"| {key} | `{spec['column']}` | {audit['targets'][key]['available']:,} | "
            f"{audit['targets'][key]['missing']:,} | "
            f"{audit['targets'][key]['available_rate']:.2%} |"
        )
    lines += [
        "",
        "## Chronological split",
        "",
        f"Train: {', '.join(map(str, TRAIN_YEARS))}  ",
        f"Test: {', '.join(map(str, TEST_YEARS))}  ",
        "XGBoost validation year: 2020  ",
        "XGBoost inner train years: 2017-2019",
        "",
        "## Modelling contract",
        "",
        "Numeric: submission_year (t0 cohort control), author_count_log1p, title/abstract word counts,",
        "four DBLP features (counts log1p; ages raw), and dblp_features_missing.",
        "",
        "Categorical: one-hot primary_category_v1, rare categories grouped using",
        "training rows only.",
        "",
        "Text: TF-IDF over title_v1 + abstract_v1, fitted on training rows only.",
        "",
        "Excluded: paper_id, raw text as a numeric feature, authors/categories JSON,",
        "comment/journal_ref/doi, t0_date, DBLP audit fields, s2_reference_count_current",
        "(post-t0), and all targets.",
        "",
        "See DATA_SPLIT.csv and DATA_TARGET_COVERAGE.csv for cohort detail.",
        "",
    ]
    (outdir / "DATA_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Audit written to {outdir}")
    print(f"Rows: {audit['rows']:,}; all-three targets: {audit['all_three_targets_available']:,}")
    print("Target coverage:")
    for key, info in audit["targets"].items():
        print(f"  {key}: {info['available']:,} / {audit['rows']:,} ({info['available_rate']:.2%})")


if __name__ == "__main__":
    main()

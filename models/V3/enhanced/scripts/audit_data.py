#!/usr/bin/env python3
"""Data contract audit for the enhanced repo (no model fitting)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import data_audit, load_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=str(REPO_ROOT.parents[1] / "gp_data" / "papers_features.csv")
    )
    parser.add_argument("--outdir", default=str(REPO_ROOT / "audit"))
    args = parser.parse_args()
    df = load_dataset(args.data)
    audit = data_audit(df)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "DATA_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Enhanced data audit",
        "",
        f"Rows: {audit['rows']:,}",
        f"All three targets: {audit['all_three_available']:,}",
        f"DBLP missing: {audit['dblp_missing']:,}",
        "",
        "| Target | Available | Rate |",
        "|---|---:|---:|",
    ]
    for key, info in audit["targets"].items():
        lines.append(f"| {key} | {info['available']:,} | {info['rate']:.2%} |")
    (out / "DATA_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'DATA_AUDIT.json'}")


if __name__ == "__main__":
    main()

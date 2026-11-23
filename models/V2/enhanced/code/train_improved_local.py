#!/usr/bin/env python3
"""Local entry point for the improved candidate pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from data_utils import find_project_root
from improve_model import DEFAULT_FEATURES, run


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = find_project_root(script_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=str(root / "gp_data" / "papers_features.csv")
    )
    parser.add_argument(
        "--outdir",
        default=str(root / "gp_model" / f"improved_{datetime.now():%Y%m%d_%H%M%S}"),
    )
    parser.add_argument("--max-features", type=int, default=DEFAULT_FEATURES)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    run(Path(args.data), Path(args.outdir), args.max_features, args.n_jobs)


if __name__ == "__main__":
    main()

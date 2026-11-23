#!/usr/bin/env python3
"""Google Colab entry point for the improved candidate pipeline.

Works with either the full code-folder layout or a flat /content layout.  The
default data file is /content/papers_features.csv; the output is under
/content/gp_model_runs/improved_<timestamp>.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from data_utils import find_project_root
from improve_model import DEFAULT_FEATURES, run


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = find_project_root(script_dir)
    content = Path("/content")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(first_existing([
            content / "papers_features.csv",
            script_dir / "papers_features.csv",
            root / "gp_data" / "papers_features.csv",
        ])),
    )
    parser.add_argument(
        "--outdir",
        default=str(
            (content if content.is_dir() else root)
            / "gp_model_runs"
            / f"improved_{datetime.now():%Y%m%d_%H%M%S}"
        ),
    )
    parser.add_argument("--max-features", type=int, default=DEFAULT_FEATURES)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    run(Path(args.data), Path(args.outdir), args.max_features, args.n_jobs)


if __name__ == "__main__":
    main()

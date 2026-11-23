#!/usr/bin/env python3
"""Google Colab runner for the enhanced model repository.

Expected: this folder is uploaded as ``/content/enhanced`` and the latest
CSV is at ``/content/papers_features.csv`` (or pass ``--data``).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.pipeline import run


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    content = Path("/content")
    project_root = repo_root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(first_existing([
            content / "papers_features.csv",
            repo_root / "papers_features.csv",
            project_root / "gp_data" / "papers_features.csv",
        ])),
    )
    parser.add_argument(
        "--outdir",
        default=str(
            (content if content.is_dir() else project_root)
            / "gp_model_runs"
            / f"enhanced_{datetime.now():%Y%m%d_%H%M%S}"
        ),
    )
    parser.add_argument("--no-bge", action="store_true")
    parser.add_argument("--strict-t0", action="store_true")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--bge-dim", type=int, default=128)
    parser.add_argument("--tfidf-max-features", type=int, default=8000)
    parser.add_argument("--category-min-count", type=int, default=15)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(
        args.data, args.outdir, use_bge=not args.no_bge,
        embedding_model=args.embedding_model, bge_dim=args.bge_dim,
        tfidf_max_features=args.tfidf_max_features,
        category_min_count=args.category_min_count,
        strict_t0=args.strict_t0,
        n_jobs=args.n_jobs, device=args.device,
    )


if __name__ == "__main__":
    main()

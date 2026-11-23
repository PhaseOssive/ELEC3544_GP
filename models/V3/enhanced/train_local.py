#!/usr/bin/env python3
"""Local runner for the enhanced model repository."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.pipeline import run


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    project_root = repo_root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=str(project_root / "gp_data" / "papers_features.csv")
    )
    parser.add_argument(
        "--outdir",
        default=str(project_root / "gp_model" / "enhanced_runs"
                    / f"enhanced_{datetime.now():%Y%m%d_%H%M%S}"),
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

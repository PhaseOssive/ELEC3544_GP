#!/usr/bin/env python3
"""Local training entry point.

Run from this directory (or from the project root):

    python gp_model/code/train_local.py

The default input is ``gp_data/papers_features.csv`` and the default output is
``gp_model/runs/local_<timestamp>``.  Use ``--data`` and ``--outdir`` to change
them; use ``--quick`` for a faster smoke run before the full run.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from data_utils import find_project_root
from model_core import cli_main


def first_existing(paths: list[Path]) -> Path:
    """Return the first existing path, or the first candidate as a clear default."""
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)
    default_data = first_existing([
        project_root / "gp_data" / "papers_features.csv",
        script_dir / "gp_data" / "papers_features.csv",
        script_dir / "papers_features.csv",
    ])
    default_out = (
        project_root / "gp_model" / "runs"
        if (project_root / "gp_model").is_dir()
        else script_dir / "gp_model" / "runs"
    )
    cli_main(
        default_data,
        default_out / f"local_{datetime.now():%Y%m%d_%H%M%S}",
    )


if __name__ == "__main__":
    main()

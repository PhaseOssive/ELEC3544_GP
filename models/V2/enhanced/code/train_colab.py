#!/usr/bin/env python3
"""Google Colab entry point.

Typical Colab flow (upload the code folder and the latest CSV first):

    !pip install -q -r requirements.txt
    !python train_colab.py --data /content/papers_features.csv \
        --outdir /content/gp_model_runs/colab_run

If ``/content/papers_features.csv`` exists it is used by default.  The script
is a thin Colab wrapper around the same tested core used by train_local.py.
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
    content = Path("/content")

    # Works whether the user uploads the whole code folder or places the
    # scripts and CSV directly in /content.  No parent-index assumption.
    default_data = first_existing([
        content / "papers_features.csv",
        script_dir / "papers_features.csv",
        project_root / "gp_data" / "papers_features.csv",
        script_dir / "gp_data" / "papers_features.csv",
    ])
    output_root = (
        content / "gp_model_runs"
        if content.is_dir()
        else project_root / "gp_model" / "runs"
    )
    default_out = output_root / f"colab_{datetime.now():%Y%m%d_%H%M%S}"
    cli_main(default_data, default_out, colab=True)


if __name__ == "__main__":
    main()

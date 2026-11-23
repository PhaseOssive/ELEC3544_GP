"""Completeness assessment checks (detection only)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from ..common import make_result, validate_dataframe, validate_columns
except ImportError:  # pragma: no cover
    from common import make_result, validate_dataframe, validate_columns


DEFAULT_SENTINELS: tuple[Any, ...] = (
    "", "na", "n/a", "nan", "null", "none", "missing", "-999", -999,
)


def check_missing_count(df: pd.DataFrame) -> dict[str, Any]:
    """Count true Pandas missing values (NaN, None, NaT, pd.NA) per column."""
    validate_dataframe(df)
    counts = df.isna().sum()
    details = [
        {"column": str(column), "missing_count": int(count)}
        for column, count in counts.items()
    ]
    issue_count = int((counts > 0).sum())
    return make_result(
        "check_missing_count", "completeness",
        summary={"columns_with_missing": issue_count,
                 "total_missing_cells": int(counts.sum()),
                 "total_cells": int(df.shape[0] * df.shape[1])},
        details=details,
        detected_issue=bool(counts.sum() > 0),
    )


def check_missing_ratio(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate the true missing-value ratio for every column."""
    validate_dataframe(df)
    ratios = df.isna().mean()
    counts = df.isna().sum()
    details = [
        {"column": str(column), "missing_count": int(counts[column]),
         "missing_ratio": float(ratio), "missing_percent": round(float(ratio) * 100, 4)}
        for column, ratio in ratios.items()
    ]
    details.sort(key=lambda item: item["missing_ratio"], reverse=True)
    return make_result(
        "check_missing_ratio", "completeness",
        summary={"columns_with_missing": int((ratios > 0).sum()),
                 "columns_over_5_percent": int((ratios > 0.05).sum()),
                 "columns_over_30_percent": int((ratios > 0.30).sum())},
        details=details,
        detected_issue=bool((ratios > 0).any()),
        description="The 5% and 30% cutoffs are lecture rules of thumb, not automatic drop-column decisions.",
    )


def check_missing_pattern(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    *,
    max_columns: int = 50,
    save_path: str | Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Visualize row/column patterns of true missing values using a matrix plot.

    If more than max_columns are selected, the columns with the most missing
    values are shown. The input DataFrame is never modified.
    """
    validate_dataframe(df)
    if max_columns < 1:
        raise ValueError("max_columns must be >= 1")
    selected = validate_columns(df, columns)
    missing_counts = df[selected].isna().sum().sort_values(ascending=False)
    plotted_columns = list(missing_counts.head(max_columns).index)
    matrix = df[plotted_columns].isna().to_numpy(dtype=bool) if plotted_columns else np.empty((len(df), 0), dtype=bool)

    import matplotlib.pyplot as plt

    width = max(8, min(18, 0.35 * max(len(plotted_columns), 1)))
    fig, ax = plt.subplots(figsize=(width, 6))
    if len(plotted_columns):
        ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks(range(len(plotted_columns)))
        ax.set_xticklabels(plotted_columns, rotation=90, fontsize=8)
        ax.set_ylabel("Row position")
        ax.set_xlabel("Columns (black = missing)")
        if not matrix.any():
            ax.set_title("Missingness pattern — no true missing values detected")
        else:
            ax.set_title("Missingness pattern")
    else:
        ax.text(0.5, 0.5, "No columns selected", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()

    saved_to = None
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved_to = str(path)
    if show:
        plt.show()

    details = [
        {"column": str(column), "missing_count": int(df[column].isna().sum()),
         "missing_ratio": float(df[column].isna().mean())}
        for column in plotted_columns
    ]
    return make_result(
        "check_missing_pattern", "completeness",
        summary={"row_count": int(len(df)), "selected_column_count": len(selected),
                 "plotted_column_count": len(plotted_columns),
                 "columns_omitted": max(0, len(selected) - len(plotted_columns)),
                 "missing_cell_count_in_plotted_columns": int(matrix.sum())},
        details=details,
        detected_issue=bool(matrix.any()),
        status="warning" if matrix.any() else "pass",
        artifacts=[{"type": "matplotlib_figure", "figure": fig, "saved_path": saved_to}],
    )


def check_nonstandard_missing(
    df: pd.DataFrame,
    sentinels: Iterable[Any] = DEFAULT_SENTINELS,
    *,
    case_insensitive: bool = True,
) -> dict[str, Any]:
    """Find blank strings and configured sentinel values that Pandas may not treat as NA.

    Sentinel matches are reported only; they are not replaced. Be careful with
    numeric sentinels such as -999 because they may be valid in some domains.
    """
    validate_dataframe(df)
    sentinels = tuple(sentinels)
    string_sentinels = {str(x).strip().casefold() if case_insensitive else str(x).strip() for x in sentinels if isinstance(x, str)}
    numeric_sentinels = [x for x in sentinels if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)]
    details: list[dict[str, Any]] = []
    total_matches = 0

    for column in df.columns:
        series = df[column]
        nonstandard_mask = pd.Series(False, index=df.index)
        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype) or isinstance(series.dtype, pd.CategoricalDtype):
            text = series.astype("string").str.strip()
            compare = text.str.casefold() if case_insensitive else text
            if string_sentinels:
                nonstandard_mask |= compare.isin(string_sentinels).fillna(False)
        if numeric_sentinels and pd.api.types.is_numeric_dtype(series.dtype):
            nonstandard_mask |= series.isin(numeric_sentinels)
        # True NA values are handled by check_missing_count, not double-counted here.
        nonstandard_mask &= series.notna()
        count = int(nonstandard_mask.sum())
        if count:
            total_matches += count
            examples = series[nonstandard_mask].astype(str).value_counts().head(10)
            details.append({
                "column": str(column),
                "nonstandard_missing_count": count,
                "examples": [{"value": str(value), "count": int(n)} for value, n in examples.items()],
                "row_indices": [str(idx) for idx in series.index[nonstandard_mask][:100]],
            })

    return make_result(
        "check_nonstandard_missing", "completeness",
        summary={"columns_with_nonstandard_missing": len(details),
                 "total_nonstandard_missing_cells": total_matches,
                 "sentinels_checked": [str(x) for x in sentinels]},
        details=details,
        detected_issue=total_matches > 0,
        description="Potential sentinel values are flagged for review; this function does not convert them to NA.",
    )

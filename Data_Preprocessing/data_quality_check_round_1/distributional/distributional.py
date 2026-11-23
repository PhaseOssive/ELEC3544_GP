"""Distributional assessment and visualization functions (detection only)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from ..common import make_result, validate_dataframe, validate_columns
except ImportError:  # pragma: no cover
    from common import make_result, validate_dataframe, validate_columns


def _numeric_columns(df: pd.DataFrame, columns: Iterable[str] | None) -> list[str]:
    if columns is not None:
        selected = validate_columns(df, columns)
        non_numeric = [c for c in selected if not pd.api.types.is_numeric_dtype(df[c].dtype)]
        if non_numeric:
            raise TypeError(f"These columns are not numeric: {non_numeric}")
        return selected
    return [c for c in df.select_dtypes(include=[np.number]).columns]


def _categorical_columns(df: pd.DataFrame, columns: Iterable[str] | None, max_categories: int) -> list[str]:
    if columns is not None:
        return validate_columns(df, columns)
    selected = []
    for column in df.columns:
        s = df[column]
        cat_like = (pd.api.types.is_object_dtype(s.dtype)
                    or pd.api.types.is_string_dtype(s.dtype)
                    or isinstance(s.dtype, pd.CategoricalDtype)
                    or pd.api.types.is_bool_dtype(s.dtype))
        if cat_like and 1 <= s.nunique(dropna=True) <= max_categories:
            selected.append(column)
    return selected


def _safe_filename(value: Any) -> str:
    """Make a column label safe for use in a filename."""
    import re
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text[:100] or "column"


def _save_figure(fig: Any, save_dir: str | Path | None, filename: str) -> str | None:
    if save_dir is None:
        return None
    directory = Path(save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return str(path)


def calculate_summary_statistics(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    percentiles: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> dict[str, Any]:
    """Calculate descriptive statistics for numeric features."""
    validate_dataframe(df)
    selected = _numeric_columns(df, columns)
    if any(p <= 0 or p >= 1 for p in percentiles):
        raise ValueError("percentiles must be strictly between 0 and 1")
    details: list[dict[str, Any]] = []
    for column in selected:
        s = pd.to_numeric(df[column], errors="coerce")
        non_null = s.dropna()
        record: dict[str, Any] = {
            "column": str(column),
            "count": int(non_null.count()),
            "missing_count": int(s.isna().sum()),
            "mean": None if non_null.empty else float(non_null.mean()),
            "median": None if non_null.empty else float(non_null.median()),
            "std": None if non_null.empty else (float(non_null.std()) if len(non_null) > 1 else 0.0),
            "min": None if non_null.empty else float(non_null.min()),
            "max": None if non_null.empty else float(non_null.max()),
            "skewness": None if len(non_null) < 3 else float(non_null.skew()),
            "kurtosis": None if len(non_null) < 4 else float(non_null.kurt()),
        }
        for p in percentiles:
            record[f"q{int(round(p * 100)):02d}"] = None if non_null.empty else float(non_null.quantile(p))
        details.append(record)
    return make_result(
        "calculate_summary_statistics", "distributional",
        summary={"numeric_columns_checked": len(selected), "row_count": int(len(df))},
        details=details,
        status="info",
        description="Descriptive statistics only; skewness and kurtosis are reported for interpretation, not automatic cleaning decisions.",
    )


def plot_histogram(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    *,
    bins: int = 30,
    save_dir: str | Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Create one histogram per selected numeric column."""
    validate_dataframe(df)
    if bins < 1:
        raise ValueError("bins must be >= 1")
    selected = _numeric_columns(df, columns)
    import matplotlib.pyplot as plt

    artifacts, details = [], []
    for column in selected:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(7, 4))
        if values.empty:
            ax.text(0.5, 0.5, "No numeric observations", ha="center", va="center")
        else:
            ax.hist(values, bins=bins, edgecolor="black", alpha=0.8)
            ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram: {column}")
        ax.set_xlabel(str(column))
        fig.tight_layout()
        saved = _save_figure(fig, save_dir, f"histogram_{_safe_filename(column)}.png")
        if show:
            plt.show()
        artifacts.append({"column": str(column), "type": "matplotlib_figure", "figure": fig, "saved_path": saved})
        details.append({"column": str(column), "observations": int(values.size),
                        "unique_values": int(values.nunique()), "saved_path": saved})
    return make_result("plot_histogram", "distributional",
                       summary={"columns_plotted": len(selected), "bins": bins},
                       details=details, status="info", artifacts=artifacts)


def plot_boxplot(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    *,
    save_dir: str | Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Create one box plot per selected numeric column."""
    validate_dataframe(df)
    selected = _numeric_columns(df, columns)
    import matplotlib.pyplot as plt

    artifacts, details = [], []
    for column in selected:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        if values.empty:
            ax.text(0.5, 0.5, "No numeric observations", ha="center", va="center")
        else:
            ax.boxplot(values, vert=False, showfliers=True)
        ax.set_title(f"Box Plot: {column}")
        ax.set_xlabel(str(column))
        fig.tight_layout()
        saved = _save_figure(fig, save_dir, f"boxplot_{_safe_filename(column)}.png")
        if show:
            plt.show()
        artifacts.append({"column": str(column), "type": "matplotlib_figure", "figure": fig, "saved_path": saved})
        details.append({"column": str(column), "observations": int(values.size), "saved_path": saved})
    return make_result("plot_boxplot", "distributional",
                       summary={"columns_plotted": len(selected)}, details=details,
                       status="info", artifacts=artifacts)


def plot_countplot(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    *,
    max_categories: int = 30,
    top_n: int = 20,
    save_dir: str | Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Create category-frequency bar plots; optionally group tail levels as Other.

    With columns=None, only low-cardinality categorical-like columns are auto-selected.
    Explicitly supplied columns are plotted even if they have high cardinality.
    """
    validate_dataframe(df)
    if max_categories < 1 or top_n < 1:
        raise ValueError("max_categories and top_n must be >= 1")
    selected = _categorical_columns(df, columns, max_categories)
    import matplotlib.pyplot as plt

    artifacts, details = [], []
    for column in selected:
        counts = df[column].astype("string").fillna("<MISSING>").value_counts(dropna=False)
        if len(counts) > top_n:
            shown = counts.head(top_n).copy()
            shown.loc["<OTHER>"] = int(counts.iloc[top_n:].sum())
        else:
            shown = counts
        fig, ax = plt.subplots(figsize=(max(7, min(14, 0.45 * len(shown))), 4.5))
        shown.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Category Counts: {column}")
        ax.set_xlabel("Count")
        ax.set_ylabel(str(column))
        fig.tight_layout()
        saved = _save_figure(fig, save_dir, f"countplot_{_safe_filename(column)}.png")
        if show:
            plt.show()
        artifacts.append({"column": str(column), "type": "matplotlib_figure", "figure": fig, "saved_path": saved})
        details.append({"column": str(column), "unique_categories": int(df[column].nunique(dropna=True)),
                        "categories_displayed": int(len(shown)), "counts": {str(k): int(v) for k, v in shown.items()},
                        "saved_path": saved})
    return make_result("plot_countplot", "distributional",
                       summary={"columns_plotted": len(selected), "top_n": top_n},
                       details=details, status="info", artifacts=artifacts)


def plot_kde(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    *,
    grid_size: int = 200,
    max_samples: int = 5000,
    random_state: int = 42,
    save_dir: str | Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Plot Gaussian KDE curves for numeric columns without requiring SciPy.

    Large columns are deterministically sampled to max_samples for plotting.
    Columns with fewer than two distinct numeric values are skipped and noted.
    """
    validate_dataframe(df)
    if grid_size < 20 or max_samples < 2:
        raise ValueError("grid_size must be >= 20 and max_samples must be >= 2")
    selected = _numeric_columns(df, columns)
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(random_state)
    artifacts, details = [], []
    for column in selected:
        values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        fig, ax = plt.subplots(figsize=(7, 4))
        status = "plotted"
        if len(values) < 2 or np.unique(values).size < 2:
            ax.text(0.5, 0.5, "KDE requires at least two distinct values", ha="center", va="center")
            status = "skipped_insufficient_variation"
        else:
            if len(values) > max_samples:
                values = rng.choice(values, size=max_samples, replace=False)
            std = float(np.std(values, ddof=1))
            bandwidth = 1.06 * std * (len(values) ** (-1 / 5))
            if not np.isfinite(bandwidth) or bandwidth <= 0:
                bandwidth = max(float(np.ptp(values)) / 100, 1e-8)
            padding = 3 * bandwidth
            grid = np.linspace(float(values.min() - padding), float(values.max() + padding), grid_size)
            # Chunk over grid points to keep temporary arrays bounded.
            density = np.empty_like(grid)
            normalizer = bandwidth * np.sqrt(2 * np.pi) * len(values)
            for start in range(0, len(grid), 25):
                chunk = grid[start:start + 25, None]
                z = (chunk - values[None, :]) / bandwidth
                density[start:start + 25] = np.exp(-0.5 * z * z).sum(axis=1) / normalizer
            ax.plot(grid, density)
            ax.set_ylabel("Density")
        ax.set_title(f"KDE: {column}")
        ax.set_xlabel(str(column))
        fig.tight_layout()
        saved = _save_figure(fig, save_dir, f"kde_{_safe_filename(column)}.png")
        if show:
            plt.show()
        artifacts.append({"column": str(column), "type": "matplotlib_figure", "figure": fig, "saved_path": saved})
        details.append({"column": str(column), "observations_used": int(len(values)),
                        "status": status, "saved_path": saved})
    return make_result("plot_kde", "distributional",
                       summary={"columns_requested": len(selected),
                                "columns_plotted": sum(d["status"] == "plotted" for d in details),
                                "columns_skipped": sum(d["status"] != "plotted" for d in details)},
                       details=details, status="info", artifacts=artifacts)


def check_iqr_outliers(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    *,
    multiplier: float = 1.5,
    min_observations: int = 4,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Flag potential numeric outliers using Tukey's 1.5*IQR fences.

    This is detection only: values and rows are never changed or removed.
    IQR flags are candidates for domain review, not proof of data errors.
    Columns with insufficient observations or zero IQR are reported as
    skipped because the usual fences are not informative for them.
    """
    validate_dataframe(df)
    if multiplier <= 0:
        raise ValueError("multiplier must be > 0")
    if min_observations < 1 or max_examples < 0:
        raise ValueError("min_observations must be >= 1 and max_examples >= 0")

    selected = _numeric_columns(df, columns)
    details: list[dict[str, Any]] = []
    total_outliers = 0

    for column in selected:
        numeric = pd.to_numeric(df[column], errors="coerce")
        finite_mask = pd.Series(np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)), index=df.index)
        values = numeric[finite_mask].dropna()

        base = {
            "column": str(column),
            "method": "Tukey IQR",
            "multiplier": float(multiplier),
            "observation_count": int(values.size),
            "outlier_count": 0,
            "outlier_percent": 0.0,
            "lower_outlier_count": 0,
            "upper_outlier_count": 0,
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_fence": None,
            "upper_fence": None,
            "outlier_positions": [],
            "examples": [],
        }

        if values.size < min_observations:
            base["status"] = "skipped_insufficient_observations"
            details.append(base)
            continue

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        base.update({"q1": q1, "q3": q3, "iqr": float(iqr)})

        if not np.isfinite(iqr) or iqr <= 0:
            base["status"] = "skipped_zero_or_invalid_iqr"
            details.append(base)
            continue

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        outlier_mask = finite_mask & ((numeric < lower) | (numeric > upper))
        positions = np.flatnonzero(outlier_mask.to_numpy())
        lower_positions = np.flatnonzero((finite_mask & (numeric < lower)).to_numpy())
        upper_positions = np.flatnonzero((finite_mask & (numeric > upper)).to_numpy())

        base.update({
            "status": "checked",
            "lower_fence": float(lower),
            "upper_fence": float(upper),
            "outlier_count": int(len(positions)),
            "outlier_percent": float(len(positions) / values.size * 100),
            "lower_outlier_count": int(len(lower_positions)),
            "upper_outlier_count": int(len(upper_positions)),
            "outlier_positions": [int(p) for p in positions],
            "examples": [
                {
                    "row_position": int(pos),
                    "row_index": str(df.index[pos]),
                    "value": float(numeric.iloc[pos]),
                }
                for pos in positions[:max_examples]
            ],
        })
        total_outliers += len(positions)
        details.append(base)

    return make_result(
        "check_iqr_outliers",
        "distributional",
        summary={
            "columns_checked": sum(d["status"] == "checked" for d in details),
            "columns_skipped": sum(d["status"] != "checked" for d in details),
            "potential_outlier_count_across_columns": int(total_outliers),
            "method": "Tukey IQR fences",
            "multiplier": float(multiplier),
        },
        details=details,
        detected_issue=total_outliers > 0,
        status="info",
        description=(
            "Potential outliers are flagged for investigation only. "
            "IQR flags do not imply invalid values; no records or values are modified."
        ),
    )

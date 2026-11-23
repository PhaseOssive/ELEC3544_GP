"""Shared helpers for Stage 1 data quality detection functions."""
from __future__ import annotations

from typing import Any, Iterable
import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> None:
    """Raise a clear error if the input is not a pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas.DataFrame, got {type(df).__name__}")


def make_result(
    check_name: str,
    dimension: str,
    summary: dict[str, Any] | None = None,
    details: list[dict[str, Any]] | dict[str, Any] | None = None,
    *,
    detected_issue: bool = False,
    status: str | None = None,
    description: str | None = None,
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the common, agent-friendly result schema used by every check.

    status is one of: pass, warning, info, skipped, error.
    """
    if status is None:
        status = "warning" if detected_issue else "pass"
    result: dict[str, Any] = {
        "check_name": check_name,
        "dimension": dimension,
        "status": status,
        "detected_issue": bool(detected_issue),
        "summary": summary or {},
        "details": details if details is not None else [],
    }
    if description:
        result["description"] = description
    if artifacts:
        result["artifacts"] = artifacts
    return result


def validate_columns(df: pd.DataFrame, columns: Iterable[str] | None) -> list[str]:
    """Return requested columns, or raise for unknown column names."""
    if columns is None:
        return list(df.columns)
    selected = list(columns)
    missing = [col for col in selected if col not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")
    return selected


def json_safe(value: Any) -> Any:
    """Convert common NumPy/Pandas scalar values into JSON-friendly values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value

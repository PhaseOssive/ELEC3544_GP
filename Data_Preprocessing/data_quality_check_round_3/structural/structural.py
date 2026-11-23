"""Structural assessment checks (detection only)."""
from __future__ import annotations

from io import StringIO
from typing import Any, Mapping

import pandas as pd

try:  # Support both package imports and running this file directly.
    from ..common import make_result, validate_dataframe, json_safe
except ImportError:  # pragma: no cover
    from common import make_result, validate_dataframe, json_safe


def check_shape(df: pd.DataFrame) -> dict[str, Any]:
    """Report the number of records and features in the DataFrame."""
    validate_dataframe(df)
    rows, columns = df.shape
    return make_result(
        "check_shape", "structural",
        summary={"row_count": int(rows), "column_count": int(columns)},
        details=[{"row_count": int(rows), "column_count": int(columns)}],
        status="info",
        description="Reports the dataset dimensions; it does not judge whether the size is appropriate.",
    )


def check_dtypes(
    df: pd.DataFrame,
    expected_dtypes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report actual column dtypes and optionally compare them with a schema.

    expected_dtypes example: {"paper_id": "string", "author_count_v1": "int64"}.
    Without expected_dtypes this is an inventory, not an automatic error check.
    """
    validate_dataframe(df)
    details: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    expected_dtypes = expected_dtypes or {}

    for column in df.columns:
        actual = df[column].dtype
        expected = expected_dtypes.get(column)
        matches = None
        expected_text = None
        if expected is not None:
            try:
                expected_dtype = pd.api.types.pandas_dtype(expected)
                matches = pd.api.types.is_dtype_equal(actual, expected_dtype)
                expected_text = str(expected_dtype)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid expected dtype for {column!r}: {expected!r}") from exc
        item = {
            "column": str(column),
            "actual_dtype": str(actual),
            "expected_dtype": expected_text,
            "matches_expected": matches,
            "non_null_count": int(df[column].notna().sum()),
            "null_count": int(df[column].isna().sum()),
        }
        details.append(item)
        if matches is False:
            mismatches.append(item)

    missing_schema_columns = [column for column in expected_dtypes if column not in df.columns]
    for column in missing_schema_columns:
        mismatches.append({
            "column": str(column),
            "actual_dtype": None,
            "expected_dtype": str(expected_dtypes[column]),
            "matches_expected": False,
            "non_null_count": 0,
            "null_count": None,
            "issue": "expected_column_missing",
        })
        details.append(mismatches[-1])

    return make_result(
        "check_dtypes", "structural",
        summary={"column_count": len(df.columns), "schema_columns_checked": len(expected_dtypes),
                 "dtype_mismatch_count": len(mismatches),
                 "missing_schema_columns": missing_schema_columns},
        details=details,
        detected_issue=bool(mismatches),
        description="Compares observed dtypes with an optional expected schema. Supply expected_dtypes to flag mismatches.",
    )


def check_info(df: pd.DataFrame) -> dict[str, Any]:
    """Capture the equivalent of DataFrame.info() without printing to console."""
    validate_dataframe(df)
    buffer = StringIO()
    df.info(buf=buffer, memory_usage="deep", show_counts=True)
    info_text = buffer.getvalue()
    details = [
        {
            "column": str(column),
            "dtype": str(df[column].dtype),
            "non_null_count": int(df[column].notna().sum()),
            "null_count": int(df[column].isna().sum()),
        }
        for column in df.columns
    ]
    return make_result(
        "check_info", "structural",
        summary={"row_count": int(len(df)), "column_count": int(len(df.columns)),
                 "non_null_cells": int(df.notna().sum().sum()),
                 "total_cells": int(df.shape[0] * df.shape[1]),
                 "info_text": info_text},
        details=details,
        status="info",
        description="Captured the DataFrame.info() overview without printing it to the console.",
    )


def check_memory_usage(df: pd.DataFrame, deep: bool = True) -> dict[str, Any]:
    """Report memory usage per column and for the whole DataFrame, in bytes."""
    validate_dataframe(df)
    memory = df.memory_usage(index=False, deep=deep)
    details = [
        {"column": str(column), "memory_bytes": int(memory[column])}
        for column in df.columns
    ]
    details.sort(key=lambda item: item["memory_bytes"], reverse=True)
    total = int(memory.sum())
    return make_result(
        "check_memory_usage", "structural",
        summary={"total_memory_bytes": total,
                 "total_memory_mib": round(total / (1024 ** 2), 4),
                 "deep": bool(deep)},
        details=details,
        status="info",
        description="Memory is reported in bytes; deep=True includes object/string payload estimates.",
    )

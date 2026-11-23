"""Validity assessment checks (detection only)."""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

try:
    from ..common import make_result, validate_dataframe, validate_columns
except ImportError:  # pragma: no cover
    from common import make_result, validate_dataframe, validate_columns


def check_value_range(
    df: pd.DataFrame,
    rules: Mapping[str, tuple[Any | None, Any | None] | Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Check numeric columns against caller-supplied domain bounds.

    rules examples:
      {"author_count_v1": {"min": 1}, "rating": {"min": 1, "max": 10}}
      {"citation_count": (0, None)}

    No universal bounds are guessed. Columns without rules are not evaluated.
    """
    validate_dataframe(df)
    if not rules:
        return make_result(
            "check_value_range", "validity",
            summary={"columns_checked": 0, "out_of_range_count": 0},
            details=[], status="skipped",
            description="No domain range rules supplied; pass a rules mapping to perform this check.",
        )

    details: list[dict[str, Any]] = []
    invalid_total = 0
    for column, rule in rules.items():
        if column not in df.columns:
            raise KeyError(f"Range rule references unknown column: {column}")
        if isinstance(rule, Mapping):
            lower = rule.get("min")
            upper = rule.get("max")
            min_inclusive = bool(rule.get("min_inclusive", True))
            max_inclusive = bool(rule.get("max_inclusive", True))
        else:
            if len(rule) != 2:
                raise ValueError(f"Range rule for {column!r} must be (min, max)")
            lower, upper = rule
            min_inclusive = max_inclusive = True

        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = pd.Series(False, index=df.index)
        comparable = numeric.notna()
        if lower is not None:
            invalid |= comparable & (numeric < lower if min_inclusive else numeric <= lower)
        if upper is not None:
            invalid |= comparable & (numeric > upper if max_inclusive else numeric >= upper)
        # Non-null, non-numeric values are outside the numeric domain too.
        non_numeric = df[column].notna() & numeric.isna()
        invalid |= non_numeric
        indices = list(df.index[invalid])
        invalid_total += len(indices)
        details.append({
            "column": str(column), "min": lower, "max": upper,
            "min_inclusive": min_inclusive, "max_inclusive": max_inclusive,
            "out_of_range_count": len(indices),
            "row_indices": [str(i) for i in indices[:200]],
            "examples": [str(v) for v in df.loc[indices, column].head(10).tolist()],
            "non_numeric_count": int(non_numeric.sum()),
        })

    return make_result(
        "check_value_range", "validity",
        summary={"columns_checked": len(rules), "out_of_range_count": invalid_total},
        details=details,
        detected_issue=invalid_total > 0,
        description="Only user-defined domain bounds are applied; values are not corrected or removed.",
    )


def check_duplicate_rows(
    df: pd.DataFrame,
    subset: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Find duplicate rows. By default, every column participates in comparison.

    All members of duplicate groups are listed (keep=False), while the number
    of extra copies is also reported (keep='first').
    """
    validate_dataframe(df)
    if subset is not None:
        subset = list(subset)
        validate_columns(df, subset)
    duplicate_mask = df.duplicated(subset=subset, keep=False)
    extra_mask = df.duplicated(subset=subset, keep="first")
    duplicate_positions = [int(pos) for pos in duplicate_mask.to_numpy().nonzero()[0]]
    details = []
    key_columns = subset if subset is not None else list(df.columns)
    for position in duplicate_positions[:500]:
        row = df.iloc[position]
        details.append({
            "row_position": position,
            "row_index": str(df.index[position]),
            "duplicate_key": {str(col): str(row[col]) for col in key_columns[:20]},
        })
    return make_result(
        "check_duplicate_rows", "validity",
        summary={"duplicate_group_member_count": int(duplicate_mask.sum()),
                 "extra_duplicate_count": int(extra_mask.sum()),
                 "comparison_columns": list(subset) if subset is not None else "all columns",
                 "details_truncated": len(duplicate_positions) > 500},
        details=details,
        detected_issue=bool(duplicate_mask.any()),
        description="Duplicate rows are reported for investigation; this function never drops records.",
    )


def _normalize_category(value: Any, case_insensitive: bool = True) -> str:
    text = re.sub(r"\s+", " ", str(value).strip())
    return text.casefold() if case_insensitive else text


def check_category_consistency(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    aliases: Mapping[str, Mapping[Any, Any]] | None = None,
    max_unique: int = 50,
    case_insensitive: bool = True,
) -> dict[str, Any]:
    """Find category labels that differ only by whitespace/case or supplied aliases.

    If columns is omitted, low-cardinality string/object/category/bool columns
    are selected. Free-text columns are excluded by the max_unique threshold.
    Alias maps are optional, e.g. {"gender": {"F": "Female", "f": "Female"}}.
    """
    validate_dataframe(df)
    if max_unique < 2:
        raise ValueError("max_unique must be >= 2")
    aliases = aliases or {}

    if columns is None:
        selected = []
        for column in df.columns:
            s = df[column]
            is_cat_like = (pd.api.types.is_object_dtype(s.dtype)
                           or pd.api.types.is_string_dtype(s.dtype)
                           or isinstance(s.dtype, pd.CategoricalDtype)
                           or pd.api.types.is_bool_dtype(s.dtype))
            nunique = int(s.nunique(dropna=True))
            if is_cat_like and 2 <= nunique <= max_unique:
                selected.append(column)
    else:
        selected = validate_columns(df, columns)

    findings: list[dict[str, Any]] = []
    for column in selected:
        s = df[column].dropna()
        if s.empty:
            continue
        alias_map = aliases.get(column, {})
        groups: dict[str, dict[str, int]] = {}
        for value, count in s.value_counts(dropna=True).items():
            canonical = alias_map.get(value, value)
            key = _normalize_category(canonical, case_insensitive=case_insensitive)
            raw = str(value)
            groups.setdefault(key, {})[raw] = int(count)
        for normalized, variants in groups.items():
            if len(variants) > 1:
                findings.append({
                    "column": str(column),
                    "normalized_category": normalized,
                    "variants": [{"value": value, "count": count} for value, count in variants.items()],
                    "variant_count": len(variants),
                })

    return make_result(
        "check_category_consistency", "validity",
        summary={"columns_checked": len(selected), "inconsistent_category_groups": len(findings)},
        details=findings,
        detected_issue=bool(findings),
        description="Detects formatting variants and optional alias-equivalent labels; semantic synonyms require an explicit alias map.",
    )


def check_cross_field_consistency(
    df: pd.DataFrame,
    rules: Mapping[str, Callable[[pd.DataFrame], pd.Series] | Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate caller-defined row-level consistency rules.

    Each rule maps a name to either a callable returning a boolean Series
    (True = valid) or a dict with keys 'check' (callable), 'columns' and
    optional 'description'. Rows returning NA are treated as unknown, not as
    invalid. Example: {"date_order": {"check": lambda d: d["t0_date"] >= d["submission_date_v1"], "columns": ["t0_date", "submission_date_v1"]}}.
    """
    validate_dataframe(df)
    if not rules:
        return make_result(
            "check_cross_field_consistency", "validity",
            summary={"rules_checked": 0, "inconsistent_rows": 0},
            details=[], status="skipped",
            description="Cross-field validity is domain-specific; supply explicit consistency rules.",
        )

    findings: list[dict[str, Any]] = []
    total_invalid = 0
    for rule_name, spec in rules.items():
        if callable(spec):
            check = spec
            columns: list[str] = []
            description = None
        elif isinstance(spec, Mapping):
            check = spec.get("check")
            columns = list(spec.get("columns", []))
            description = spec.get("description")
            if columns:
                validate_columns(df, columns)
        else:
            raise TypeError(f"Rule {rule_name!r} must be callable or a mapping")
        if not callable(check):
            raise TypeError(f"Rule {rule_name!r} must provide a callable 'check'")

        outcome = check(df.copy(deep=True))
        if not isinstance(outcome, pd.Series):
            outcome = pd.Series(outcome, index=df.index)
        if not outcome.index.equals(df.index):
            raise ValueError(f"Rule {rule_name!r} returned a Series with a different index")
        valid = outcome.astype("boolean")
        invalid_mask = valid.eq(False).fillna(False)
        indices = list(df.index[invalid_mask])
        total_invalid += len(indices)
        findings.append({
            "rule": str(rule_name),
            "description": description,
            "columns": columns,
            "invalid_row_count": len(indices),
            "row_indices": [str(i) for i in indices[:200]],
            "details_truncated": len(indices) > 200,
        })

    return make_result(
        "check_cross_field_consistency", "validity",
        summary={"rules_checked": len(rules), "inconsistent_rows_across_rules": total_invalid},
        details=findings,
        detected_issue=total_invalid > 0,
        description="Rules are supplied by the project/domain owner; this function does not invent business logic.",
    )


def check_primary_key_missing(
    df: pd.DataFrame,
    column: str = "paper_id",
) -> dict[str, Any]:
    """Report missing or blank primary-key values without modifying records."""
    validate_dataframe(df)
    if column not in df.columns:
        return make_result(
            "check_primary_key_missing", "validity",
            summary={"column": column, "missing_or_blank_count": None},
            details=[], status="skipped",
            description=f"Primary-key column {column!r} is not present.",
        )
    series = df[column]
    blank = series.astype("string").str.strip().eq("").fillna(False)
    missing = series.isna() | blank
    positions = [int(p) for p in missing.to_numpy().nonzero()[0]]
    return make_result(
        "check_primary_key_missing", "validity",
        summary={
            "column": column,
            "missing_or_blank_count": len(positions),
        },
        details=[{
            "column": column,
            "row_positions": positions,
        }] if positions else [],
        detected_issue=bool(positions),
        description="Checks that every source record has a nonblank primary key; no rows are removed.",
    )

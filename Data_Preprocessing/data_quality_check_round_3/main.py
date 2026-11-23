"""Read-only data quality assessment for papers_features.csv."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

try:  # package mode
    from .common import validate_dataframe
    from .structural.structural import check_shape, check_dtypes, check_info, check_memory_usage
    from .completeness.completeness import (
        check_missing_count,
        check_missing_ratio,
        check_missing_pattern,
        check_nonstandard_missing,
    )
    from .validity.validity import (
        check_value_range,
        check_duplicate_rows,
        check_category_consistency,
        check_cross_field_consistency,
        check_primary_key_missing,
    )
    from .distributional.distributional import (
        calculate_summary_statistics,
        plot_histogram,
        plot_boxplot,
        plot_countplot,
        plot_kde,
        check_iqr_outliers,
    )
    from .workflow_config import (
        DEFAULT_INPUT_FILE,
        DEFAULT_OUTPUT_DIR,
        AUTO_INCREMENT_OUTPUT_FOLDER,
        EXPECTED_DTYPES,
        VALUE_RANGE_RULES,
        CATEGORY_COLUMNS,
        COUNT_PLOT_COLUMNS,
        ASSESSMENT_COLUMNS,
        PREDICTOR_CANDIDATE_COLUMNS,
        TARGET_CANDIDATE_COLUMNS,
        OUTLIER_COLUMNS,
        IQR_MULTIPLIER,
        build_cross_field_rules,
    )
except ImportError:  # direct mode: python main.py
    from common import validate_dataframe
    from structural.structural import check_shape, check_dtypes, check_info, check_memory_usage
    from completeness.completeness import (
        check_missing_count,
        check_missing_ratio,
        check_missing_pattern,
        check_nonstandard_missing,
    )
    from validity.validity import (
        check_value_range,
        check_duplicate_rows,
        check_category_consistency,
        check_cross_field_consistency,
        check_primary_key_missing,
    )
    from distributional.distributional import (
        calculate_summary_statistics,
        plot_histogram,
        plot_boxplot,
        plot_countplot,
        plot_kde,
        check_iqr_outliers,
    )
    from workflow_config import (
        DEFAULT_INPUT_FILE,
        DEFAULT_OUTPUT_DIR,
        AUTO_INCREMENT_OUTPUT_FOLDER,
        EXPECTED_DTYPES,
        VALUE_RANGE_RULES,
        CATEGORY_COLUMNS,
        COUNT_PLOT_COLUMNS,
        ASSESSMENT_COLUMNS,
        PREDICTOR_CANDIDATE_COLUMNS,
        TARGET_CANDIDATE_COLUMNS,
        OUTLIER_COLUMNS,
        IQR_MULTIPLIER,
        build_cross_field_rules,
    )


def resolve_input_path(explicit_path: str | Path | None = None) -> Path:
    """Resolve the source CSV. No local sample fallback is used implicitly."""
    candidate = Path(explicit_path).expanduser() if explicit_path else Path(DEFAULT_INPUT_FILE)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.exists():
        if explicit_path is None:
            raise FileNotFoundError(
                "Default local feature table was not found:\n"
                f"  {candidate}\n"
                "Place papers_features.csv directly in the project folder, or update "
                "DEFAULT_INPUT_FILE in workflow_config.py / pass --input /path/to/papers_features.csv."
            )
        raise FileNotFoundError(f"Input CSV does not exist: {candidate}")
    if candidate.suffix.lower() != ".csv":
        raise ValueError(f"Input must be a CSV file: {candidate}")
    return candidate.resolve()


def _create_unique_output_directory(base_dir: Path) -> Path:
    """Create a new output folder, appending _2, _3, ... if needed."""
    base_dir = base_dir.expanduser().resolve()
    base_dir.parent.mkdir(parents=True, exist_ok=True)

    if not AUTO_INCREMENT_OUTPUT_FOLDER:
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    candidate = base_dir
    index = 2
    while True:
        try:
            # mkdir without exist_ok reserves the path and avoids overwrites.
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base_dir.with_name(f"{base_dir.name}_{index}")
            index += 1


def _prepare_output_directories(output_root: Path) -> dict[str, Path]:
    """Create report/diagram output folders without deleting prior outputs."""
    report_dir = output_root / "report"
    diagram_dir = output_root / "diagram"
    report_dir.mkdir(parents=True, exist_ok=True)
    diagram_dir.mkdir(parents=True, exist_ok=True)
    completeness_dir = diagram_dir / "completeness"
    distributional_dir = diagram_dir / "distributional"
    completeness_dir.mkdir(parents=True, exist_ok=True)
    distributional_dir.mkdir(parents=True, exist_ok=True)
    return {
        "report": report_dir,
        "diagram": diagram_dir,
        "completeness": completeness_dir,
        "distributional": distributional_dir,
    }


def _filter_rules(rules: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    return {name: rule for name, rule in rules.items() if name in columns}


def run_structural_detection(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Structural checks cover the full source table, not only candidate X/Y."""
    expected = _filter_rules(EXPECTED_DTYPES, list(df.columns))
    return [
        check_shape(df),
        check_dtypes(df, expected_dtypes=expected),
        check_info(df),
        check_memory_usage(df, deep=True),
    ]


def _close_figures(results: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    for result in results:
        for artifact in result.get("artifacts", []) or []:
            figure = artifact.get("figure") if isinstance(artifact, dict) else None
            if figure is not None:
                plt.close(figure)


def run_completeness_detection(
    analysis_df: pd.DataFrame,
    diagram_dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    """Completeness is focused on potentially useful model inputs/targets."""
    results = [
        check_missing_count(analysis_df),
        check_missing_ratio(analysis_df),
        check_missing_pattern(
            analysis_df,
            max_columns=50,
            save_path=diagram_dirs["completeness"] / "missing_pattern.png",
            show=False,
        ),
        check_nonstandard_missing(analysis_df),
    ]
    _close_figures(results)
    return results


def run_validity_detection(
    full_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Check model-relevant value rules plus whole-table record integrity."""
    range_rules = _filter_rules(VALUE_RANGE_RULES, list(analysis_df.columns))
    category_columns = [c for c in CATEGORY_COLUMNS if c in analysis_df.columns]
    cross_field_rules = build_cross_field_rules(list(full_df.columns))

    results = [
        check_value_range(analysis_df, rules=range_rules),
        # Exact duplicate and primary-key checks apply to the full source rows.
        check_duplicate_rows(full_df),
        check_primary_key_missing(full_df, column="paper_id"),
    ]
    if "paper_id" in full_df.columns:
        duplicate_id_result = check_duplicate_rows(full_df, subset=["paper_id"])
        duplicate_id_result["check_name"] = "check_duplicate_paper_id"
        duplicate_id_result["description"] = (
            "Checks uniqueness of paper_id independently of full-row duplication."
        )
        results.append(duplicate_id_result)

    results.extend([
        check_category_consistency(
            analysis_df,
            columns=category_columns,
            max_unique=100,
        ),
        # These cross-field rules validate internal consistency of meaningful
        # fields (e.g. author count, date/year, and paired team statistics).
        results_cross_field(full_df, cross_field_rules),
    ])
    return results


def results_cross_field(df: pd.DataFrame, rules: dict[str, Any]) -> dict[str, Any]:
    return check_cross_field_consistency(df, rules=rules)


def run_distributional_detection(
    analysis_df: pd.DataFrame,
    diagram_dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    """Distribution and outlier checks use only the configured assessment scope."""
    output_dir = diagram_dirs["distributional"]
    numeric_columns = [
        c for c in analysis_df.select_dtypes(include=[np.number]).columns
        if not pd.api.types.is_bool_dtype(analysis_df[c].dtype)
    ]
    count_columns = [c for c in COUNT_PLOT_COLUMNS if c in analysis_df.columns]
    outlier_columns = [
        c for c in OUTLIER_COLUMNS
        if c in analysis_df.columns and pd.api.types.is_numeric_dtype(analysis_df[c].dtype)
    ]

    results = [calculate_summary_statistics(analysis_df, columns=numeric_columns)]
    results.append(plot_histogram(
        analysis_df, columns=numeric_columns, bins=30,
        save_dir=output_dir / "histograms", show=False,
    ))
    _close_figures(results[-1:])
    results.append(plot_boxplot(
        analysis_df, columns=numeric_columns,
        save_dir=output_dir / "boxplots", show=False,
    ))
    _close_figures(results[-1:])
    results.append(plot_countplot(
        analysis_df, columns=count_columns, max_categories=100, top_n=30,
        save_dir=output_dir / "countplots", show=False,
    ))
    _close_figures(results[-1:])
    results.append(plot_kde(
        analysis_df, columns=numeric_columns, max_samples=5000, random_state=42,
        save_dir=output_dir / "kde", show=False,
    ))
    _close_figures(results[-1:])
    results.append(check_iqr_outliers(
        analysis_df,
        columns=outlier_columns,
        multiplier=IQR_MULTIPLIER,
    ))
    return results


def _json_safe(value: Any) -> Any:
    """Convert report objects into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if k != "figure"}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_feature_level_report(
    analysis_df: pd.DataFrame,
    assessment: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Create a compact report for the selected candidate X/Y columns."""
    rows: dict[str, dict[str, Any]] = {}
    for column in analysis_df.columns:
        rows[str(column)] = {
            "feature": str(column),
            "dtype": str(analysis_df[column].dtype),
            "row_count": int(len(analysis_df)),
            "non_null_count": int(analysis_df[column].notna().sum()),
            "missing_count": int(analysis_df[column].isna().sum()),
            "missing_ratio": float(analysis_df[column].isna().mean()) if len(analysis_df) else 0.0,
            "nonstandard_missing_count": 0,
            "out_of_range_count": 0,
            "category_inconsistency_groups": 0,
            "iqr_outlier_count": 0,
            "iqr_outlier_percent": 0.0,
            "iqr_lower_fence": None,
            "iqr_upper_fence": None,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "skewness": None,
            "kurtosis": None,
        }

    for result in assessment.get("completeness", []):
        name = result.get("check_name")
        for detail in result.get("details", []):
            column = detail.get("column")
            if column not in rows:
                continue
            if name == "check_missing_count":
                rows[column]["missing_count"] = int(detail.get("missing_count", 0))
            elif name == "check_nonstandard_missing":
                rows[column]["nonstandard_missing_count"] = int(
                    detail.get("nonstandard_missing_count", 0)
                )

    for result in assessment.get("validity", []):
        name = result.get("check_name")
        for detail in result.get("details", []):
            column = detail.get("column")
            if column not in rows:
                continue
            if name == "check_value_range":
                rows[column]["out_of_range_count"] = int(detail.get("out_of_range_count", 0))
            elif name == "check_category_consistency":
                rows[column]["category_inconsistency_groups"] += 1

    for result in assessment.get("distributional", []):
        name = result.get("check_name")
        for detail in result.get("details", []):
            column = detail.get("column")
            if column not in rows:
                continue
            if name == "calculate_summary_statistics":
                for key in ("mean", "median", "std", "min", "max", "skewness", "kurtosis"):
                    rows[column][key] = detail.get(key)
            elif name == "check_iqr_outliers":
                rows[column]["iqr_outlier_count"] = int(detail.get("outlier_count", 0))
                rows[column]["iqr_outlier_percent"] = float(detail.get("outlier_percent", 0.0))
                rows[column]["iqr_lower_fence"] = detail.get("lower_fence")
                rows[column]["iqr_upper_fence"] = detail.get("upper_fence")

    return pd.DataFrame(list(rows.values()))


def build_record_issue_report(
    full_df: pd.DataFrame,
    assessment_df: pd.DataFrame,
    assessment: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Build a row-level index for missingness, validity, and outlier findings."""
    issues: list[dict[str, Any]] = []
    id_values = full_df["paper_id"].astype("string") if "paper_id" in full_df.columns else None

    def add_issue(
        dimension: str,
        check_name: str,
        row_position: int,
        column: str | None = None,
        detail: Any = None,
    ) -> None:
        paper_id = None
        if id_values is not None and 0 <= row_position < len(id_values):
            paper_id = id_values.iloc[row_position]
        issues.append({
            "dimension": dimension,
            "check_name": check_name,
            "row_position": int(row_position),
            "paper_id": None if paper_id is None or pd.isna(paper_id) else str(paper_id),
            "column": column,
            "detail": json.dumps(_json_safe(detail), ensure_ascii=False),
        })

    # Missingness is reported only for the configured analysis columns.
    for column in assessment_df.columns:
        positions = np.flatnonzero(assessment_df[column].isna().to_numpy())
        for position in positions:
            add_issue("completeness", "missing_value", int(position), str(column), {"value": None})

    index_to_position = {str(index): pos for pos, index in enumerate(full_df.index)}
    for dimension, results in assessment.items():
        for result in results:
            name = result.get("check_name", "unknown_check")
            for detail in result.get("details", []):
                if not isinstance(detail, dict):
                    continue
                column = detail.get("column")
                if "row_position" in detail:
                    add_issue(dimension, name, int(detail["row_position"]), column, detail)
                for key in ("row_indices",):
                    for row_index in detail.get(key, []) or []:
                        position = index_to_position.get(str(row_index))
                        if position is not None:
                            add_issue(dimension, name, position, column, detail)
                for key in ("row_positions", "outlier_positions"):
                    for position in detail.get(key, []) or []:
                        add_issue(dimension, name, int(position), column, detail)
    return pd.DataFrame(
        issues,
        columns=["dimension", "check_name", "row_position", "paper_id", "column", "detail"],
    )


def build_outlier_report(
    full_df: pd.DataFrame,
    assessment_df: pd.DataFrame,
    assessment: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Expand IQR flags into one row per paper/feature; no source rows are removed."""
    records: list[dict[str, Any]] = []
    for result in assessment.get("distributional", []):
        if result.get("check_name") != "check_iqr_outliers":
            continue
        for detail in result.get("details", []):
            column = detail.get("column")
            for position in detail.get("outlier_positions", []) or []:
                pos = int(position)
                paper_id = full_df["paper_id"].iloc[pos] if "paper_id" in full_df.columns else None
                records.append({
                    "row_position": pos,
                    "paper_id": None if paper_id is None or pd.isna(paper_id) else str(paper_id),
                    "feature": column,
                    "value": assessment_df[column].iloc[pos],
                    "lower_fence": detail.get("lower_fence"),
                    "upper_fence": detail.get("upper_fence"),
                    "method": detail.get("method", "Tukey IQR"),
                    "interpretation": "Potential outlier; investigate domain context before any action.",
                })
    return pd.DataFrame(records, columns=[
        "row_position", "paper_id", "feature", "value",
        "lower_fence", "upper_fence", "method", "interpretation",
    ])


def write_markdown_report(
    report: dict[str, Any],
    feature_report: pd.DataFrame,
    outlier_report: pd.DataFrame,
    output_path: Path,
) -> None:
    workflow = report["workflow"]
    assessment = report["assessment"]
    lines = [
        "# Read-only Data Quality Assessment Report",
        "",
        f"- Run time (UTC): {workflow['run_at_utc']}",
        f"- Source CSV: `{workflow['input_file']}`",
        f"- Source shape: {workflow['row_count']} rows × {workflow['column_count']} columns",
        f"- Assessed candidate X/Y columns: {len(workflow['assessment_columns'])}",
        f"- Ignored from detailed feature analysis: {len(workflow['ignored_columns'])} columns",
        "- Source data changed: **No** (read-only assessment)",
        "",
        "## Assessment scope",
        "",
        "Detailed completeness, validity-range, distribution, and outlier analysis is limited to candidate model inputs/targets. "
        "Identifier, URL/DOI, raw text/JSON, source identifiers, collection-status, and pipeline-audit fields are not plotted or "
        "treated as candidate predictors. The full source table is still used for shape, structural, duplicate-record, primary-key, "
        "and configured cross-field integrity checks.",
        "",
        "### Provisional predictor candidates",
        "",
        ", ".join(f"`{c}`" for c in workflow["predictor_candidate_columns"]),
        "",
        "### Provisional target/outcome candidates",
        "",
        ", ".join(f"`{c}`" for c in workflow["target_candidate_columns"]),
        "",
        "### Fields ignored in detailed feature analysis (not deleted)",
        "",
        ", ".join(f"`{c}`" for c in workflow["ignored_columns"]),
        "",
        "## Detection results",
        "",
    ]
    for dimension in ("structural", "completeness", "validity", "distributional"):
        results = assessment.get(dimension, [])
        lines.extend([f"### {dimension.title()}", ""])
        if not results:
            lines.extend(["No checks were run.", ""])
            continue
        lines.extend(["| Check | Status | Issue detected |", "|---|---|---|"])
        for result in results:
            lines.append(
                f"| `{result.get('check_name')}` | {result.get('status')} | "
                f"{result.get('detected_issue')} |"
            )
        lines.append("")

    lines.extend([
        "## Feature-level overview",
        "",
        "| Feature | Dtype | Missing % | Nonstandard missing | Out of range | IQR outliers | Mean | Median |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in feature_report.iterrows():
        mean = "" if pd.isna(row["mean"]) else f"{row['mean']:.4g}"
        median = "" if pd.isna(row["median"]) else f"{row['median']:.4g}"
        lines.append(
            f"| `{row['feature']}` | `{row['dtype']}` | {row['missing_ratio'] * 100:.2f}% | "
            f"{int(row['nonstandard_missing_count'])} | {int(row['out_of_range_count'])} | "
            f"{int(row['iqr_outlier_count'])} | {mean} | {median} |"
        )

    lines.extend([
        "",
        "## Outlier interpretation",
        "",
        f"- Method: Tukey's IQR fences, Q1 − {IQR_MULTIPLIER}×IQR and Q3 + {IQR_MULTIPLIER}×IQR.",
        f"- Potential outlier feature-record flags: {len(outlier_report)}.",
        "- IQR flags are screening candidates, not automatic data errors. Genuine extreme academic teams or citation counts should normally be retained unless domain evidence indicates an error.",
        "- No rows or values were corrected, imputed, clipped, or deleted.",
        "",
        "## Output files",
        "",
        "- `report/data_quality_report.json`: complete machine-readable checks.",
        "- `report/data_quality_report.md`: this summary.",
        "- `report/feature_level_report.csv`: candidate feature-level statistics.",
        "- `report/record_issue_report.csv`: row-level issue index.",
        "- `report/outlier_report.csv`: one row per potential outlier feature-record.",
        "- `diagram/`: missingness and distribution plots.",
        "",
        "## Interpretation notes",
        "",
        "- A `skipped` check means the required rule or usable input was unavailable; it does not mean the data passed.",
        "- The assessment scope is a provisional candidate X/Y list, not the final modelling feature list.",
        "- The source CSV is never rewritten. All generated files are separate reports and figures.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_workflow(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read the configured feature table and run detection-only assessment."""
    raw_path = resolve_input_path(input_path)
    output_base = Path(output_dir).expanduser() if output_dir is not None else Path(DEFAULT_OUTPUT_DIR)
    output_root = _create_unique_output_directory(output_base)
    dirs = _prepare_output_directories(output_root)

    # Read-only: no output master table is created and no source values change.
    source_columns = list(pd.read_csv(raw_path, nrows=0).columns)
    dtype_map = {k: v for k, v in EXPECTED_DTYPES.items() if k in source_columns}
    if "arxiv_id" in source_columns:
        dtype_map["arxiv_id"] = "string"
    df = pd.read_csv(raw_path, dtype=dtype_map, low_memory=False)
    validate_dataframe(df)

    assessment_columns = [c for c in ASSESSMENT_COLUMNS if c in df.columns]
    missing_config = [c for c in ASSESSMENT_COLUMNS if c not in df.columns]
    if not assessment_columns:
        raise ValueError("None of ASSESSMENT_COLUMNS exist in the input CSV. Check workflow_config.py.")
    analysis_df = df.loc[:, assessment_columns].copy()
    ignored_columns = [c for c in df.columns if c not in assessment_columns]

    assessment: dict[str, list[dict[str, Any]]] = {}
    assessment["structural"] = run_structural_detection(df)
    assessment["completeness"] = run_completeness_detection(analysis_df, dirs)
    assessment["validity"] = run_validity_detection(df, analysis_df)
    assessment["distributional"] = run_distributional_detection(analysis_df, dirs)

    workflow = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(raw_path),
        "output_directory": str(output_root),
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "assessment_columns": assessment_columns,
        "predictor_candidate_columns": [c for c in PREDICTOR_CANDIDATE_COLUMNS if c in df.columns],
        "target_candidate_columns": [c for c in TARGET_CANDIDATE_COLUMNS if c in df.columns],
        "ignored_columns": ignored_columns,
        "configured_columns_not_found": missing_config,
        "source_file_size_bytes": int(raw_path.stat().st_size),
        "source_file_mtime": datetime.fromtimestamp(raw_path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "source_modified_by_workflow": False,
    }
    feature_report = build_feature_level_report(analysis_df, assessment)
    record_report = build_record_issue_report(df, analysis_df, assessment)
    outlier_report = build_outlier_report(df, analysis_df, assessment)

    report = {
        "workflow": workflow,
        "assessment": assessment,
        "outputs": {
            "json_report": str(dirs["report"] / "data_quality_report.json"),
            "markdown_report": str(dirs["report"] / "data_quality_report.md"),
            "feature_level_report": str(dirs["report"] / "feature_level_report.csv"),
            "record_issue_report": str(dirs["report"] / "record_issue_report.csv"),
            "outlier_report": str(dirs["report"] / "outlier_report.csv"),
            "diagram_directory": str(dirs["diagram"]),
        },
    }

    (dirs["report"] / "data_quality_report.json").write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    feature_report.to_csv(dirs["report"] / "feature_level_report.csv", index=False, encoding="utf-8-sig")
    record_report.to_csv(dirs["report"] / "record_issue_report.csv", index=False, encoding="utf-8-sig")
    outlier_report.to_csv(dirs["report"] / "outlier_report.csv", index=False, encoding="utf-8-sig")
    write_markdown_report(
        report,
        feature_report,
        outlier_report,
        dirs["report"] / "data_quality_report.md",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run read-only structural, completeness, validity, distribution, and outlier assessment."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional source CSV path. Defaults to DEFAULT_INPUT_FILE in workflow_config.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output base folder. A new numbered folder is created if it already exists.",
    )
    args = parser.parse_args()

    try:
        report = run_workflow(args.input, args.output_dir)
    except Exception as exc:
        print(f"[Assessment failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    workflow = report["workflow"]
    print("\n========== Read-only Data Quality Assessment Completed ==========")
    print(f"Source:       {workflow['input_file']}")
    print(f"Shape:        {workflow['row_count']} rows × {workflow['column_count']} columns")
    print(f"Assessed:     {len(workflow['assessment_columns'])} candidate X/Y columns")
    print(f"Ignored:      {len(workflow['ignored_columns'])} non-feature/metadata columns (not deleted)")
    print(f"Output:       {workflow['output_directory']}")
    print("Order:        Structural -> Completeness -> Validity -> Distributional + IQR Outliers")
    print("Source data:  READ ONLY; no columns or rows are removed or changed")
    print("===============================================================\n")


if __name__ == "__main__":
    main()

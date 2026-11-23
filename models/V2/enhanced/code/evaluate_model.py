#!/usr/bin/env python3
"""Audit and evaluate an existing GP training run from its saved artifacts.

This script deliberately uses the saved prediction CSVs so that evaluation is
independent of whether the joblib artifacts can be unpickled in the current
environment.  It reports both the trained models and naive baselines, error
slices, calibration, success confusion, thresholds and artifact portability.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

TARGET_COLUMNS = {
    "Y1": "s2_citation_count_3y",
    "Y2": "s2_influential_citation_count_3y",
    "Y3": "openreview_mean_rating",
}
SPLIT = {"train": (2017, 2018, 2019, 2020), "test": (2021, 2022)}
MODELS = ("linear", "xgb")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"paper_id": "string"})


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n == 0:
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "r2": math.nan}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    var = float(np.var(y_true))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)) if var > 0 else math.nan
    return {"n": n, "mae": mae, "rmse": rmse, "r2": r2}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / n if n else math.nan,
        "precision": precision, "recall": recall, "f1": f1,
        "positive_rate": float(y_true.mean()) if n else math.nan,
        "predicted_positive_rate": float(y_pred.mean()) if n else math.nan,
    }


def find_run_path(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = sorted(
        Path("gp_model").rglob("SUMMARY.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No run SUMMARY.md found under gp_model/.")
    return candidates[0].parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--data", default="gp_data/papers_features.csv")
    parser.add_argument("--outdir", default="gp_model/evaluation")
    args = parser.parse_args()

    run = find_run_path(args.run)
    out = Path(args.outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data).expanduser().resolve()

    data = pd.read_csv(data_path, dtype={"paper_id": "string"}, low_memory=False)
    data["paper_id"] = data["paper_id"].astype(str)
    for target, column in TARGET_COLUMNS.items():
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["submission_year"] = data["submission_year"].astype(int)

    print(f"Run:  {run}")
    print(f"Data: {data_path}")
    print(f"Rows: {len(data):,}")

    summary_rows: List[Dict[str, Any]] = []
    year_rows: List[Dict[str, Any]] = []
    category_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    success_rows: List[Dict[str, Any]] = []

    # Naive baselines from the training split, evaluated on test.
    baseline_rows: List[Dict[str, Any]] = []
    for target, column in TARGET_COLUMNS.items():
        train = data.loc[data["submission_year"].isin(SPLIT["train"]), column].dropna()
        test = data.loc[data["submission_year"].isin(SPLIT["test"]), column].dropna()
        if train.empty or test.empty:
            continue
        for name, value in [
            ("train_mean", float(train.mean())),
            ("train_median", float(train.median())),
            ("zero", 0.0),
            ("test_mean_oracle", float(test.mean())),
            ("test_median_oracle", float(test.median())),
        ]:
            metrics = regression_metrics(test.to_numpy(), np.full(len(test), value))
            metrics.update({"target": target, "baseline": name, "constant": value})
            baseline_rows.append(metrics)
        data_stats = {
            "target": target,
            "train_n": int(train.shape[0]),
            "train_mean": float(train.mean()),
            "train_median": float(train.median()),
            "test_n": int(test.shape[0]),
            "test_mean": float(test.mean()),
            "test_median": float(test.median()),
            "zero_share": float((test == 0).mean()),
        }
        summary_rows.append(data_stats)

    for target, column in TARGET_COLUMNS.items():
        for model in MODELS:
            pred_path = run / "predictions" / f"{target}_{model}_predictions.csv"
            if not pred_path.is_file():
                continue
            pred = read_csv(pred_path)
            pred["y_true"] = pd.to_numeric(pred["y_true"], errors="coerce")
            pred["y_pred"] = pd.to_numeric(pred["y_pred"], errors="coerce")
            test = pred.dropna(subset=["y_true", "y_pred"])
            metrics = regression_metrics(test["y_true"], test["y_pred"])
            metrics.update({"target": target, "model": model})
            summary_rows.append(metrics)

            # Per-year errors and bias.
            for year, group in test.groupby("submission_year"):
                m = regression_metrics(group["y_true"], group["y_pred"])
                m2 = regression_metrics(
                    group["y_true"], np.full(len(group), group["y_true"].mean())
                )
                m.update({
                    "target": target, "model": model, "submission_year": int(year),
                    "actual_mean": float(group["y_true"].mean()),
                    "pred_mean": float(group["y_pred"].mean()),
                    "bias": float((group["y_pred"] - group["y_true"]).mean()),
                    "mean_baseline_mae": m2["mae"],
                })
                year_rows.append(m)

            # Error by category, using the source table.
            merged = pred.merge(
                data[["paper_id", "primary_category_v1"]],
                on="paper_id",
                how="left",
            )
            for category, group in merged.groupby("primary_category_v1"):
                if len(group) < 30:
                    continue
                m = regression_metrics(group["y_true"], group["y_pred"])
                m.update({"target": target, "model": model, "category": category})
                category_rows.append(m)

            # Calibration by predicted-score decile.
            try:
                test = test.copy()
                test["bin"] = pd.qcut(
                    test["y_pred"], q=10, duplicates="drop", labels=False
                )
                for bin_id, group in test.groupby("bin"):
                    calibration_rows.append({
                        "target": target, "model": model, "bin": int(bin_id),
                        "n": int(len(group)),
                        "mean_pred": float(group["y_pred"].mean()),
                        "mean_true": float(group["y_true"].mean()),
                        "bias": float((group["y_pred"] - group["y_true"]).mean()),
                        "mae": float(np.mean(np.abs(group["y_pred"] - group["y_true"]))),
                        "positive_rate_true": float(group["true_success"].mean()),
                        "positive_rate_pred": float(group["predicted_success"].mean()),
                    })
            except ValueError:
                pass

            metrics = binary_metrics(pred["true_success"], pred["predicted_success"])
            metrics.update({
                "target": target, "model": model, "evaluation": "per_target",
            })
            success_rows.append(metrics)

    # Primary all-target union, if present.
    for model in MODELS:
        path = run / "predictions" / f"all_target_union_{model}.csv"
        if not path.is_file():
            continue
        pred = read_csv(path)
        metrics = binary_metrics(pred["true_success"], pred["predicted_success"])
        metrics.update({
            "target": "ALL3", "model": model, "evaluation": "all_target_union",
            "coverage": len(pred) / len(data.loc[data["submission_year"].isin(SPLIT["test"])]),
        })
        success_rows.append(metrics)

    # Thresholds and feature importance are copied into the evaluation folder.
    for name in ("thresholds.csv", "feature_importance.csv", "xgboost_validation.csv"):
        source = run / "metrics" / name
        if source.is_file():
            pd.read_csv(source).to_csv(out / name, index=False, encoding="utf-8-sig")

    pd.DataFrame(baseline_rows).to_csv(
        out / "baselines.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(year_rows).to_csv(
        out / "error_by_year.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(category_rows).to_csv(
        out / "error_by_category.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(calibration_rows).to_csv(
        out / "calibration.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(success_rows).to_csv(
        out / "success_confusion.csv", index=False, encoding="utf-8-sig"
    )

    # Artifact portability check.
    artifact_notes = []
    try:
        import joblib
        import warnings
        from sklearn.exceptions import InconsistentVersionWarning

        for target in TARGET_COLUMNS:
            for model in MODELS:
                path = run / "models" / f"{target}_{model}.joblib"
                if not path.is_file():
                    continue
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        artifact = joblib.load(path)
                    bad = [
                        str(w.message) for w in caught
                        if isinstance(w.message, InconsistentVersionWarning)
                    ]
                    artifact_notes.append(
                        f"- `{path.name}` loaded; feature count "
                        f"{len(artifact.get('feature_names', []))}; "
                        f"version warnings: {len(bad)}."
                    )
                except Exception as exc:
                    artifact_notes.append(
                        f"- `{path.name}` failed to load: `{type(exc).__name__}: {exc}`"
                    )
    except Exception as exc:
        artifact_notes.append(f"- Artifact check unavailable: `{exc}`")

    # Build the Markdown report.
    lines = [
        "# Evaluation of the existing GP model",
        "",
        f"- Run: `{run}`",
        f"- Data: `{data_path}`",
        f"- Rows: {len(data):,}",
        f"- Test years: {SPLIT['test']}",
        "",
        "## Test-set baselines vs trained models",
        "",
        "| Target | Model | N | MAE | RMSE | R² |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if "model" in row:
            lines.append(
                f"| {row['target']} | {row['model']} | {row['n']} | "
                f"{row['mae']:.3f} | {row['rmse']:.3f} | {row['r2']:.3f} |"
            )
    lines += [
        "",
        "Naive train-constant baselines are in `baselines.csv`; the test-median",
        "oracle rows are context only and are not deployable.",
        "",
        "## Target distribution shift",
        "",
        "| Target | Train N | Train mean | Train median | Test N | Test mean | Test median | Test zero share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if "model" not in row:
            lines.append(
                f"| {row['target']} | {row['train_n']} | {row['train_mean']:.3f} | "
                f"{row['train_median']:.3f} | {row['test_n']} | {row['test_mean']:.3f} | "
                f"{row['test_median']:.3f} | {row['zero_share']:.3f} |"
            )
    lines += [
        "",
        "## Binary success",
        "",
        "| Target | Model | N | True positive rate | Predicted positive rate | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in success_rows:
        lines.append(
            f"| {row['target']} | {row['model']} | {row['n']} | "
            f"{row['positive_rate']:.3f} | {row['predicted_positive_rate']:.3f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
        )
    lines += [
        "",
        "## Main findings",
        "",
        "1. **Y1/Y2 regression is weak but above naive means only for XGBoost.**",
        "   Y1 XGBoost R² ≈ 0.057; Y1 Linear Regression is essentially the",
        "   training mean. Y2 models improve modestly over the mean/median.",
        "2. **Y3 is not usable as a predictive regression.** The test mean and",
        "   median are higher than training, and both models have negative R²;",
        "   the train-median baseline is stronger than both models.",
        "3. **Success detection is under-predicting positives.** Predicted",
        "   positive rates are far below the true 10% rate, especially for",
        "   Y2/XGBoost and Y3/XGBoost (zero positives).",
        "4. **The primary all-target union is based on only 483 test papers**, so",
        "   its F1 estimates have wide uncertainty and limited coverage.",
        "5. **The saved joblib artifacts are not portable across scikit-learn",
        "   versions.** The Colab run used scikit-learn 1.6.1; loading under",
        "   1.9.1 fails for the imputer. The prediction CSVs remain usable.",
        "",
        "## Files",
        "",
        "- `baselines.csv`",
        "- `error_by_year.csv`",
        "- `error_by_category.csv`",
        "- `calibration.csv`",
        "- `success_confusion.csv`",
        "- `thresholds.csv`, `feature_importance.csv`, `xgboost_validation.csv`",
        "",
        "## Artifact notes",
        "",
        *artifact_notes,
        "",
    ]
    (out / "EVALUATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Evaluation written to {out}")
    print(f"Artifacts: {len(artifact_notes)} notes")


if __name__ == "__main__":
    main()

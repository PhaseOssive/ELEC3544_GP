#!/usr/bin/env python3
"""Post-hoc diagnostics for the improved run.

Reads the selected prediction CSVs and model artifacts, then writes error
slices, calibration, and feature importance into ``metrics/``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

TARGETS = {
    "Y1": "s2_citation_count_3y",
    "Y2": "s2_influential_citation_count_3y",
    "Y3": "openreview_mean_rating",
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    if n == 0:
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "r2": math.nan}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    var = float(np.var(y_true))
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)) if var else math.nan
    return {"n": n, "mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="gp_model/improved")
    parser.add_argument("--data", default="gp_data/papers_features.csv")
    args = parser.parse_args()
    run = Path(args.run)
    data = pd.read_csv(args.data, dtype={"paper_id": "string"}, low_memory=False)
    data["paper_id"] = data["paper_id"].astype(str)

    rows_year: List[Dict[str, Any]] = []
    rows_category: List[Dict[str, Any]] = []
    rows_calibration: List[Dict[str, Any]] = []
    rows_importance: List[Dict[str, Any]] = []

    for target, column in TARGETS.items():
        pred_path = run / "predictions" / f"{target}_selected_predictions.csv"
        if not pred_path.is_file():
            continue
        pred = pd.read_csv(pred_path, dtype={"paper_id": "string"})
        pred["y_true"] = pd.to_numeric(pred["y_true"], errors="coerce")
        pred["y_pred"] = pd.to_numeric(pred["y_pred"], errors="coerce")
        pred = pred.dropna(subset=["y_true", "y_pred"]).copy()

        for year, group in pred.groupby("submission_year"):
            m = metrics(group["y_true"], group["y_pred"])
            m.update({
                "target": target, "model": str(group["model"].iloc[0]),
                "submission_year": int(year),
                "actual_mean": float(group["y_true"].mean()),
                "pred_mean": float(group["y_pred"].mean()),
                "bias": float((group["y_pred"] - group["y_true"]).mean()),
            })
            rows_year.append(m)

        merged = pred.merge(
            data[["paper_id", "primary_category_v1"]], on="paper_id", how="left"
        )
        for category, group in merged.groupby("primary_category_v1"):
            if len(group) < 30:
                continue
            m = metrics(group["y_true"], group["y_pred"])
            m.update({
                "target": target, "model": str(group["model"].iloc[0]),
                "category": category,
            })
            rows_category.append(m)

        pred["bin"] = pd.qcut(pred["y_pred"], q=10, duplicates="drop", labels=False)
        for bin_id, group in pred.groupby("bin"):
            rows_calibration.append({
                "target": target, "bin": int(bin_id), "n": int(len(group)),
                "mean_pred": float(group["y_pred"].mean()),
                "mean_true": float(group["y_true"].mean()),
                "bias": float((group["y_pred"] - group["y_true"]).mean()),
                "mae": float(np.mean(np.abs(group["y_pred"] - group["y_true"]))),
            })

        artifact_path = run / "models" / f"{target}_selected.joblib"
        if artifact_path.is_file():
            artifact = joblib.load(artifact_path)
            model = artifact.get("model")
            names = artifact.get("feature_names")
            if names is None:
                names = []
            values = getattr(model, "feature_importances_", None)
            if values is not None and len(names) == len(values):
                order = np.argsort(-np.asarray(values))[:30]
                for index in order:
                    rows_importance.append({
                        "target": target,
                        "model": artifact.get("model_name", ""),
                        "feature": names[int(index)],
                        "importance": float(values[int(index)]),
                    })

    pd.DataFrame(rows_year).to_csv(
        run / "metrics" / "error_by_year.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(rows_category).to_csv(
        run / "metrics" / "error_by_category.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(rows_calibration).to_csv(
        run / "metrics" / "calibration.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(rows_importance).to_csv(
        run / "metrics" / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Diagnostics written to {run / 'metrics'}")


if __name__ == "__main__":
    main()

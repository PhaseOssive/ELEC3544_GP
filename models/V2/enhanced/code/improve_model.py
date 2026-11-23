#!/usr/bin/env python3
"""Improved model selection and final training.

Selection uses only the chronological 2020 validation cohort (2017-2019 inner
training).  The 2021-2022 test years are evaluated only after the candidate is
selected.  This keeps the test set out of model/threshold selection while
allowing a fair before/after comparison with the original run.

Candidates are motivated by ``gp_data/模型预选.docx``:
  * Y1/Y2: LightGBM Poisson/Tweedie, XGBoost Poisson/Tweedie, hurdle for Y2;
  * Y1/Y2: the original log-target XGBoost baseline;
  * Y3: Ridge (full and structured-only), plus constant mean/median fallbacks;
  * success: direct gradient-boosted classifiers with validation-selected
    thresholds, alongside the proposal's regression-threshold rule.

CatBoost, TabPFN/Nori and BGE embeddings from the preselection note are not
available in this environment; they are documented as external/future options
rather than silently replaced.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from data_utils import (
    NUMERIC_FEATURES,
    TARGETS,
    TEST_YEARS,
    TRAIN_YEARS,
    TaskFrame,
    coverage_by_target,
    data_audit,
    find_project_root,
    load_dataset,
    split_summary,
    task_frame,
    threshold_rows,
    thresholds_for_task,
)
from model_core import (
    FeaturePreprocessor,
    binary_metrics,
    duan_smearing_factor,
    inverse_prediction,
    regression_metrics,
    target_transform,
)

SEED = 42
N_JOBS = -1
INNER_YEARS = (2017, 2018, 2019)
VALIDATION_YEAR = 2020
DEFAULT_FEATURES = 5000
Y3_FEATURES = 1500

LGB_BASE: Dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

XGB_COUNT_BASE: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

# Best original XGBoost log-target configurations from the saved baseline run.
XGB_LOG_CONFIGS = {
    "Y1": {
        "n_estimators": 400, "max_depth": 3, "learning_rate": 0.05,
        "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 1,
        "reg_lambda": 1.0,
    },
    "Y2": {
        "n_estimators": 400, "max_depth": 3, "learning_rate": 0.05,
        "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 1,
        "reg_lambda": 1.0,
    },
}


@dataclass
class Prepared:
    task: TaskFrame
    inner: pd.DataFrame
    validation: pd.DataFrame
    pre_xgb_inner: FeaturePreprocessor
    X_inner: sp.csr_matrix
    X_validation: sp.csr_matrix
    pre_linear_inner: FeaturePreprocessor
    XL_inner: sp.csr_matrix
    XL_validation: sp.csr_matrix


def _prepare(task: TaskFrame, max_features: int) -> Prepared:
    inner = task.train.loc[task.train["submission_year"].isin(INNER_YEARS)].copy()
    validation = task.train.loc[task.train["submission_year"] == VALIDATION_YEAR].copy()
    if inner.empty or validation.empty:
        raise ValueError(f"{task.key}: inner/validation split is empty.")
    pre_xgb = FeaturePreprocessor(
        "xgb", max_features=max_features, min_df=5, cat_min_frequency=20
    ).fit(inner)
    pre_linear = FeaturePreprocessor(
        "linear", max_features=max_features, min_df=5, cat_min_frequency=20
    ).fit(inner)
    return Prepared(
        task=task,
        inner=inner,
        validation=validation,
        pre_xgb_inner=pre_xgb,
        X_inner=pre_xgb.transform(inner),
        X_validation=pre_xgb.transform(validation),
        pre_linear_inner=pre_linear,
        XL_inner=pre_linear.transform(inner),
        XL_validation=pre_linear.transform(validation),
    )


def _val_observed_threshold(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].quantile(0.90))


def _predict_log(model: Any, X: sp.csr_matrix, smearing: float, task: TaskFrame) -> np.ndarray:
    raw = model.predict(X)
    return inverse_prediction(
        raw, task.log_target, task.clip_nonnegative, smearing_factor=smearing
    )


def _fit_lgb_regressor(
    X: sp.csr_matrix,
    y: np.ndarray,
    objective: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> lgb.LGBMRegressor:
    params = {**LGB_BASE, "objective": objective}
    if extra:
        params.update(extra)
    return lgb.LGBMRegressor(**params).fit(X, y)


def _fit_xgb_regressor(
    X: sp.csr_matrix,
    y: np.ndarray,
    objective: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> XGBRegressor:
    params = {**XGB_COUNT_BASE, "objective": objective}
    if extra:
        params.update(extra)
    return XGBRegressor(**params).fit(X, y)


def _fit_xgb_log(
    X: sp.csr_matrix,
    y: np.ndarray,
    task_key: str,
) -> Tuple[XGBRegressor, float]:
    cfg = dict(XGB_LOG_CONFIGS[task_key])
    params = {
        **cfg,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": N_JOBS,
        "verbosity": 0,
    }
    model = XGBRegressor(**params).fit(X, np.log1p(y))
    smearing = duan_smearing_factor(np.log1p(y), model.predict(X))
    return model, smearing


def _fit_ridge(
    X: sp.csr_matrix,
    y: np.ndarray,
    alpha: float,
    log_target: bool,
) -> Ridge:
    return Ridge(alpha=alpha).fit(X, np.log1p(y) if log_target else y)


def _hurdle_predict(
    X_train: sp.csr_matrix,
    y_train: np.ndarray,
    X_apply: sp.csr_matrix,
    positive_objective: str = "poisson",
) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
    positive = (y_train > 0).astype(int)
    clf = lgb.LGBMClassifier(
        **{
            **LGB_BASE,
            "objective": "binary",
            "n_estimators": 400,
            "num_leaves": 31,
            "min_child_samples": 20,
        }
    ).fit(X_train, positive)
    p_positive = clf.predict_proba(X_apply)[:, 1]
    pos_mask = positive.astype(bool)
    reg = _fit_lgb_regressor(
        X_train[pos_mask], y_train[pos_mask], positive_objective
    )
    positive_count = np.clip(reg.predict(X_apply), 0.0, None)
    expected = p_positive * positive_count
    return expected, p_positive, clf, reg


def _label_success(
    frame: pd.DataFrame,
    column: str,
    thresholds_by_year: Mapping[int, float],
) -> np.ndarray:
    thresholds = frame["submission_year"].map(thresholds_by_year).to_numpy(float)
    return (frame[column].to_numpy(float) >= thresholds).astype(int)


def _fit_success_classifier(
    prepared: Prepared,
    max_features: int,
) -> Dict[str, Any]:
    task = prepared.task
    column = task.column
    inner = prepared.inner
    validation = prepared.validation
    annual = {
        year: float(inner.loc[inner["submission_year"] == year, column].quantile(0.90))
        for year in INNER_YEARS
    }
    validation_threshold = _val_observed_threshold(validation, column)
    y_inner = _label_success(inner, column, annual)
    y_validation = _label_success(validation, column, {VALIDATION_YEAR: validation_threshold})

    clf = lgb.LGBMClassifier(
        **{
            **LGB_BASE,
            "objective": "binary",
            "n_estimators": 400,
            "min_child_samples": 15,
        }
    ).fit(prepared.X_inner, y_inner)
    probabilities = clf.predict_proba(prepared.X_validation)[:, 1]
    validation_positive_rate = float(y_validation.mean())
    # Choose a prevalence-matched operating point rather than the raw F1
    # maximum.  The success definition is a top-10% rule, so the predicted
    # positive rate should not drift to 25-35% just to chase F1 on a weak
    # classifier.  F1 is retained as a tie-breaker.
    best_distance = float("inf")
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted_rate = float(np.mean(probabilities >= threshold))
        distance = abs(predicted_rate - validation_positive_rate)
        score = f1_score(y_validation, probabilities >= threshold)
        if (distance < best_distance - 1e-12) or (
            abs(distance - best_distance) <= 1e-12 and score > best_f1
        ):
            best_distance = distance
            best_threshold = float(threshold)
            best_f1 = float(score)
    validation_metrics = binary_metrics(y_validation, probabilities >= best_threshold)
    return {
        "model": clf,
        "threshold": best_threshold,
        "validation_f1": best_f1,
        "validation_positive_rate": validation_positive_rate,
        "validation_predicted_positive_rate": float(
            np.mean(probabilities >= best_threshold)
        ),
        "validation_metrics": validation_metrics,
        "validation_probabilities": probabilities,
        "validation_labels": y_validation,
        "annual_labels": annual,
        "validation_threshold": validation_threshold,
        "preprocessor": prepared.pre_xgb_inner,
    }


def _candidate_summary(
    name: str,
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    validation_observed_threshold: float,
    estimated_threshold: float,
) -> Dict[str, Any]:
    regression = regression_metrics(y_true, y_pred)
    success = binary_metrics(
        y_true >= validation_observed_threshold, y_pred >= estimated_threshold
    )
    return {
        "target": target,
        "candidate": name,
        **{f"reg_{key}": value for key, value in regression.items()},
        **{f"success_{key}": value for key, value in success.items()},
    }


def _structured_matrices(
    inner: pd.DataFrame,
    validation: pd.DataFrame,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, Any]:
    imputer = SimpleImputer(strategy="median").fit(
        inner[list(NUMERIC_FEATURES)].to_numpy(float)
    )
    scaler = StandardScaler().fit(
        imputer.transform(inner[list(NUMERIC_FEATURES)].to_numpy(float))
    )
    onehot = OneHotEncoder(
        handle_unknown="ignore", min_frequency=10, sparse_output=True
    ).fit(inner[["primary_category_v1"]].astype(str))

    def transform(frame: pd.DataFrame) -> sp.csr_matrix:
        numeric = scaler.transform(
            imputer.transform(frame[list(NUMERIC_FEATURES)].to_numpy(float))
        )
        category = onehot.transform(frame[["primary_category_v1"]].astype(str))
        return sp.hstack([sp.csr_matrix(numeric), category], format="csr")

    return transform(inner), transform(validation), {
        "imputer": imputer, "scaler": scaler, "encoder": onehot,
    }


def _save_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(payload), path)


def _write_report(
    outdir: Path,
    task_frames: Mapping[str, TaskFrame],
    validation_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    success_rows: Sequence[Mapping[str, Any]],
    selected: Mapping[str, str],
    baseline_metrics: pd.DataFrame,
) -> None:
    lines = [
        "# Improved GP model: selection and test results",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Selection protocol",
        "",
        "- Inner training: 2017–2019.",
        "- Validation: 2020.",
        "- Final training: 2017–2020.",
        "- Held-out test: 2021–2022.",
        "- Regression candidates are ranked by validation RMSE; Y3 also compares",
        "  deployable constant mean/median baselines by validation MAE because its",
        "  models have negative validation R².",
        "- Success classifiers choose their probability threshold on the 2020",
        "  validation labels only; the threshold targets the validation success",
        "  prevalence, with F1 used as a tie-breaker. The test-year p90 is used",
        "  only to construct reference labels after selection.",
        "",
        "## Selected regression models",
        "",
        "| Target | Selected model |",
        "|---|---|",
    ]
    for target, name in selected.items():
        lines.append(f"| {target} | {name} |")
    lines += [
        "",
        "## Validation candidates",
        "",
        "| Target | Candidate | Val N | MAE | RMSE | R² | Success F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in validation_rows:
        lines.append(
            f"| {row['target']} | {row['candidate']} | {row['reg_n']} | "
            f"{row['reg_mae']:.3f} | {row['reg_rmse']:.3f} | "
            f"{row['reg_r2']:.3f} | {row['success_f1']:.3f} |"
        )
    lines += [
        "",
        "## Final held-out test results",
        "",
        "| Target | Model | N | MAE | RMSE | R² |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in test_rows:
        lines.append(
            f"| {row['target']} | {row['model']} | {row['n']} | "
            f"{row['mae']:.3f} | {row['rmse']:.3f} | {row['r2']:.3f} |"
        )
    lines += [
        "",
        "## Success detection",
        "",
        "| Target | Method | N | Precision | Recall | F1 | Predicted positive rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in success_rows:
        lines.append(
            f"| {row['target']} | {row['method']} | {row['n']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['predicted_positive_rate']:.3f} |"
        )
    lines += [
        "",
        "## Original model baseline (test)",
        "",
        "| Target | Model | MAE | RMSE | R² |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in baseline_metrics.iterrows():
        lines.append(
            f"| {row['target']} | {row['model']} | {row['mae']:.3f} | "
            f"{row['rmse']:.3f} | {row['r2']:.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- CatBoost, TabPFN-3, Nori 1.5 and BGE-small are listed in the",
        "  preselection note but are not installed or available offline here;",
        "  they remain external/future candidates rather than silent substitutes.",
        "- The original joblib artifacts were trained with scikit-learn 1.6.1",
        "  and fail to transform under scikit-learn 1.9.1; the improved artifacts",
        "  are retrained and saved with the current environment.",
        "- Y3 has only 934 labels and 449 training labels; its conclusions are",
        "  correspondingly uncertain.",
        "",
    ]
    (outdir / "IMPROVEMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(data_path: Path, outdir: Path, max_features: int, n_jobs: int) -> Path:
    global N_JOBS
    N_JOBS = int(n_jobs)
    LGB_BASE["n_jobs"] = N_JOBS
    XGB_COUNT_BASE["n_jobs"] = N_JOBS
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "models").mkdir(exist_ok=True)
    (outdir / "metrics").mkdir(exist_ok=True)
    (outdir / "predictions").mkdir(exist_ok=True)

    df = load_dataset(data_path)
    audit = data_audit(df)
    (outdir / "metrics" / "data_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    task_frames = {key: task_frame(df, key) for key in TARGETS}
    prepared: Dict[str, Prepared] = {}
    validation_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    success_rows: List[Dict[str, Any]] = []
    selected: Dict[str, str] = {}
    selected_objects: Dict[str, Dict[str, Any]] = {}
    test_predictions: Dict[str, pd.DataFrame] = {}
    success_probs: Dict[str, np.ndarray] = {}
    success_thresholds: Dict[str, float] = {}
    threshold_rows_all: List[Dict[str, Any]] = []

    for key in ("Y1", "Y2"):
        task = task_frames[key]
        prepared[key] = _prepare(task, max_features)
        p = prepared[key]
        y_inner = p.inner[task.column].to_numpy(float)
        y_val = p.validation[task.column].to_numpy(float)
        val_threshold = _val_observed_threshold(p.validation, task.column)
        thresholds = thresholds_for_task(task)
        estimated = thresholds["estimated_test"]
        candidates: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}

        # LightGBM count objectives.
        for objective in ("poisson", "tweedie"):
            if objective == "tweedie":
                for power in (1.1, 1.5, 1.9):
                    name = f"lgb_tweedie_{power}"
                    model = _fit_lgb_regressor(
                        p.X_inner, y_inner, objective,
                        {"tweedie_variance_power": power},
                    )
                    pred = np.clip(model.predict(p.X_validation), 0, None)
                    candidates[name] = (
                        pred,
                        {"model": model, "family": "lgb", "objective": objective,
                         "extra": {"tweedie_variance_power": power}},
                    )
            else:
                name = f"lgb_{objective}"
                model = _fit_lgb_regressor(p.X_inner, y_inner, objective)
                pred = np.clip(model.predict(p.X_validation), 0, None)
                candidates[name] = (
                    pred, {"model": model, "family": "lgb", "objective": objective}
                )

        # XGBoost count objectives.
        for objective in ("count:poisson", "reg:tweedie"):
            name = f"xgb_{objective.replace(':', '_')}"
            extra = {"tweedie_variance_power": 1.5} if objective == "reg:tweedie" else {}
            model = _fit_xgb_regressor(p.X_inner, y_inner, objective, extra)
            pred = np.clip(model.predict(p.X_validation), 0, None)
            candidates[name] = (
                pred, {"model": model, "family": "xgb", "objective": objective,
                       "extra": extra},
            )

        # Original log-target XGBoost baseline.
        xgb_log, smear = _fit_xgb_log(p.X_inner, y_inner, key)
        pred_log = _predict_log(xgb_log, p.X_validation, smear, task)
        candidates["xgb_log"] = (
            pred_log,
            {"model": xgb_log, "family": "xgb", "objective": "log",
             "log_target": True, "smearing": smear},
        )

        # Ridge on the same feature matrix (log target for counts).
        for alpha in (1.0, 10.0, 100.0):
            model = _fit_ridge(p.XL_inner, y_inner, alpha, log_target=True)
            pred = inverse_prediction(
                model.predict(p.XL_validation), True, True, 1.0
            )
            candidates[f"ridge_log_{alpha:g}"] = (
                pred, {"model": model, "family": "ridge", "alpha": alpha,
                       "log_target": True},
            )

        # Hurdle for Y2 only (the preselection note's two-stage idea).
        if key == "Y2":
            expected, prob, clf, reg = _hurdle_predict(
                p.X_inner, y_inner, p.X_validation
            )
            candidates["hurdle_poisson"] = (
                expected,
                {"family": "hurdle", "classifier": clf, "regressor": reg,
                 "mode": "expected"},
            )

        # Rank by validation RMSE, with MAE as tie-breaker.
        scored = []
        for name, (pred, payload) in candidates.items():
            row = _candidate_summary(
                name, key, y_val, pred, val_threshold, estimated
            )
            validation_rows.append(row)
            scored.append((row["reg_rmse"], row["reg_mae"], name, pred, payload))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        best_rmse, best_mae, best_name, best_pred, best_payload = scored[0]
        selected[key] = best_name
        selected_objects[key] = {"validation_pred": best_pred, **best_payload}

        # Final fit on 2017-2020 using the chosen candidate.
        full = task.train
        pre_full = FeaturePreprocessor(
            "xgb" if best_payload["family"] != "ridge" else "linear",
            max_features=max_features, min_df=5, cat_min_frequency=20,
        ).fit(full)
        X_full = pre_full.transform(full)
        X_test = pre_full.transform(task.test)
        print(f"\n[{key}] selected {best_name} (val RMSE={best_rmse:.3f})", flush=True)

        if best_payload["family"] == "lgb":
            model = _fit_lgb_regressor(
                X_full, full[task.column].to_numpy(float),
                best_payload["objective"], best_payload.get("extra"),
            )
            test_pred = np.clip(model.predict(X_test), 0, None)
            artifact_model = model
        elif best_payload["family"] == "xgb" and best_payload["objective"] != "log":
            model = _fit_xgb_regressor(
                X_full, full[task.column].to_numpy(float),
                best_payload["objective"], best_payload.get("extra"),
            )
            test_pred = np.clip(model.predict(X_test), 0, None)
            artifact_model = model
        elif best_payload["family"] == "xgb" and best_payload["objective"] == "log":
            model, smear = _fit_xgb_log(X_full, full[task.column].to_numpy(float), key)
            test_pred = _predict_log(model, X_test, smear, task)
            artifact_model = model
        elif best_payload["family"] == "ridge":
            pre_linear_full = FeaturePreprocessor(
                "linear", max_features=max_features, min_df=5,
                cat_min_frequency=20,
            ).fit(full)
            X_full_linear = pre_linear_full.transform(full)
            X_test_linear = pre_linear_full.transform(task.test)
            model = _fit_ridge(
                X_full_linear, full[task.column].to_numpy(float),
                best_payload["alpha"], log_target=True,
            )
            test_pred = inverse_prediction(
                model.predict(X_test_linear), True, True, 1.0
            )
            pre_full = pre_linear_full
            artifact_model = model
        elif best_payload["family"] == "hurdle":
            expected, prob, clf, reg = _hurdle_predict(
                X_full, full[task.column].to_numpy(float), X_test
            )
            test_pred = expected
            artifact_model = {"classifier": clf, "regressor": reg}
        else:
            raise RuntimeError(f"Unsupported selected family: {best_payload['family']}")

        metrics = regression_metrics(
            task.test[task.column].to_numpy(float), test_pred
        )
        test_rows.append({"target": key, "model": best_name, **metrics})

        pred_frame = task.test[["paper_id", "submission_year", task.column]].copy()
        pred_frame["target"] = key
        pred_frame["model"] = best_name
        pred_frame["y_true"] = task.test[task.column].to_numpy(float)
        pred_frame["y_pred"] = test_pred
        pred_frame["predicted_success"] = test_pred >= estimated
        test_predictions[key] = pred_frame

        artifact = {
            "target": key,
            "target_column": task.column,
            "model_name": best_name,
            "model_family": best_payload["family"],
            "objective": best_payload.get("objective"),
            "extra": best_payload.get("extra"),
            "preprocessor": pre_full,
            "model": artifact_model,
            "feature_names": pre_full.get_feature_names_out(),
            "validation_rmse": best_rmse,
            "validation_mae": best_mae,
            "selected_on": "2020_validation",
            "seed": SEED,
        }
        _save_artifact(outdir / "models" / f"{key}_selected.joblib", artifact)
        pred_frame.to_csv(
            outdir / "predictions" / f"{key}_selected_predictions.csv",
            index=False, encoding="utf-8-sig",
        )

        # Original success rule is still evaluated for this selected model.
        success_rows.append({
            "target": key, "method": "regression_threshold",
            **binary_metrics(
                task.test[task.column].to_numpy(float)
                >= task.test["submission_year"].map(
                    thresholds["annual_test_reference"]
                ).to_numpy(float),
                test_pred >= estimated,
            ),
        })

        # Direct success classifier selected on the inner/validation split.
        sc = _fit_success_classifier(p, max_features)
        # Refit the classifier on all 2017-2020 data and use the validation
        # threshold for the held-out test predictions.  Always use a fresh
        # XGB-style preprocessor so the threshold and the final classifier use
        # the same feature space, even if the regression winner was Ridge.
        pre_cls = FeaturePreprocessor(
            "xgb", max_features=max_features, min_df=5, cat_min_frequency=20
        ).fit(full)
        X_full_cls = pre_cls.transform(full)
        X_test_cls = pre_cls.transform(task.test)
        annual_full = {
            year: float(
                full.loc[full["submission_year"] == year, task.column].quantile(0.90)
            )
            for year in TRAIN_YEARS
        }
        y_full_cls = _label_success(full, task.column, annual_full)
        clf_full = lgb.LGBMClassifier(
            **{
                **LGB_BASE,
                "objective": "binary",
                "n_estimators": 400,
                "min_child_samples": 15,
            }
        ).fit(X_full_cls, y_full_cls)
        prob_test = clf_full.predict_proba(X_test_cls)[:, 1]
        predicted_direct = prob_test >= sc["threshold"]
        true_direct = task.test[task.column].to_numpy(float) >= task.test[
            "submission_year"
        ].map(thresholds["annual_test_reference"]).to_numpy(float)
        success_probs[key] = prob_test
        success_thresholds[key] = float(sc["threshold"])
        success_rows.append({
            "target": key, "method": "direct_classifier",
            **binary_metrics(true_direct, predicted_direct),
            "validation_f1": sc["validation_f1"],
            "threshold": sc["threshold"],
        })
        _save_artifact(
            outdir / "models" / f"{key}_success_classifier.joblib",
            {
                "target": key,
                "target_column": task.column,
                "preprocessor": pre_cls,
                "model": clf_full,
                "threshold": sc["threshold"],
                "validation_f1": sc["validation_f1"],
                "feature_names": pre_cls.get_feature_names_out(),
            },
        )
        pd.DataFrame({
            "paper_id": task.test["paper_id"].to_numpy(),
            "submission_year": task.test["submission_year"].to_numpy(),
            "probability": prob_test,
        }).to_csv(
            outdir / "predictions" / f"{key}_success_probabilities.csv",
            index=False, encoding="utf-8-sig",
        )
        threshold_rows_all.append(pd.DataFrame(threshold_rows(task, thresholds)))

    # Y3: compare deployable constants and regularized/structured models.
    key = "Y3"
    task = task_frames[key]
    p = _prepare(task, Y3_FEATURES)
    y_inner = p.inner[task.column].to_numpy(float)
    y_val = p.validation[task.column].to_numpy(float)
    val_threshold = _val_observed_threshold(p.validation, task.column)
    thresholds = thresholds_for_task(task)
    estimated = thresholds["estimated_test"]
    candidates = {}

    for name, value in (
        ("train_mean", float(y_inner.mean())),
        ("train_median", float(np.median(y_inner))),
    ):
        pred = np.full(len(y_val), value)
        candidates[name] = (
            pred, {"family": "constant", "value": value, "name": name}
        )

    for alpha in (10.0, 100.0, 1000.0):
        model = _fit_ridge(p.XL_inner, y_inner, alpha, log_target=False)
        pred = model.predict(p.XL_validation)
        candidates[f"ridge_full_{alpha:g}"] = (
            pred, {"family": "ridge", "alpha": alpha, "model": model,
                   "feature_mode": "full"}
        )

    Xs_inner, Xs_validation, structured_parts = _structured_matrices(
        p.inner, p.validation
    )
    for alpha in (10.0, 100.0, 1000.0):
        model = _fit_ridge(Xs_inner, y_inner, alpha, log_target=False)
        pred = model.predict(Xs_validation)
        candidates[f"ridge_structured_{alpha:g}"] = (
            pred, {"family": "ridge", "alpha": alpha, "model": model,
                   "feature_mode": "structured", "parts": structured_parts}
        )

    scored = []
    for name, (pred, payload) in candidates.items():
        row = _candidate_summary(name, key, y_val, pred, val_threshold, estimated)
        validation_rows.append(row)
        scored.append((row["reg_mae"], row["reg_rmse"], name, pred, payload))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    best_mae, best_rmse, best_name, best_val_pred, best_payload = scored[0]
    selected[key] = best_name
    print(f"\n[Y3] selected {best_name} (val MAE={best_mae:.3f})", flush=True)

    # Final fit / prediction for the selected Y3 option.
    if best_payload["family"] == "constant":
        test_pred = np.full(len(task.test), best_payload["value"])
        pre_full = None
        artifact_model = {"constant": best_payload["value"]}
    elif best_payload["feature_mode"] == "full":
        pre_full = FeaturePreprocessor(
            "linear", max_features=Y3_FEATURES, min_df=5, cat_min_frequency=20
        ).fit(task.train)
        X_full = pre_full.transform(task.train)
        X_test = pre_full.transform(task.test)
        model = _fit_ridge(
            X_full, task.train[task.column].to_numpy(float),
            best_payload["alpha"], log_target=False,
        )
        test_pred = model.predict(X_test)
        artifact_model = model
    else:
        Xs_full, Xs_test, parts = _structured_matrices(task.train, task.test)
        model = _fit_ridge(
            Xs_full, task.train[task.column].to_numpy(float),
            best_payload["alpha"], log_target=False,
        )
        test_pred = model.predict(Xs_test)
        pre_full = None
        artifact_model = {"model": model, "parts": parts,
                          "feature_mode": "structured"}

    metrics = regression_metrics(task.test[task.column].to_numpy(float), test_pred)
    test_rows.append({"target": key, "model": best_name, **metrics})
    pred_frame = task.test[["paper_id", "submission_year", task.column]].copy()
    pred_frame["target"] = key
    pred_frame["model"] = best_name
    pred_frame["y_true"] = task.test[task.column].to_numpy(float)
    pred_frame["y_pred"] = test_pred
    pred_frame["predicted_success"] = test_pred >= estimated
    test_predictions[key] = pred_frame
    pred_frame.to_csv(
        outdir / "predictions" / f"{key}_selected_predictions.csv",
        index=False, encoding="utf-8-sig",
    )
    artifact = {
        "target": key,
        "target_column": task.column,
        "model_name": best_name,
        "model_family": best_payload["family"],
        "preprocessor": pre_full,
        "model": artifact_model,
        "feature_names": (
            pre_full.get_feature_names_out().tolist() if pre_full is not None else []
        ),
        "validation_mae": best_mae,
        "validation_rmse": best_rmse,
        "selected_on": "2020_validation",
        "seed": SEED,
    }
    _save_artifact(outdir / "models" / f"{key}_selected.joblib", artifact)
    success_rows.append({
        "target": key, "method": "regression_threshold",
        **binary_metrics(
            task.test[task.column].to_numpy(float)
            >= task.test["submission_year"].map(
                thresholds["annual_test_reference"]
            ).to_numpy(float),
            test_pred >= estimated,
        ),
    })

    # Y3 direct success classifier.
    sc = _fit_success_classifier(p, Y3_FEATURES)
    # Use the full preprocessor if one exists; otherwise fit a fresh xgb one.
    pre_full_xgb = FeaturePreprocessor(
        "xgb", max_features=Y3_FEATURES, min_df=5, cat_min_frequency=20
    ).fit(task.train)
    X_full_xgb = pre_full_xgb.transform(task.train)
    X_test_xgb = pre_full_xgb.transform(task.test)
    annual_full = {
        year: float(task.train.loc[task.train["submission_year"] == year, task.column].quantile(0.90))
        for year in TRAIN_YEARS
    }
    clf_full = lgb.LGBMClassifier(
        **{
            **LGB_BASE,
            "objective": "binary",
            "n_estimators": 400,
            "min_child_samples": 10,
        }
    ).fit(X_full_xgb, _label_success(task.train, task.column, annual_full))
    prob_test = clf_full.predict_proba(X_test_xgb)[:, 1]
    true_direct = task.test[task.column].to_numpy(float) >= task.test[
        "submission_year"
    ].map(thresholds["annual_test_reference"]).to_numpy(float)
    success_probs[key] = prob_test
    success_thresholds[key] = float(sc["threshold"])
    success_rows.append({
        "target": key, "method": "direct_classifier",
        **binary_metrics(true_direct, prob_test >= sc["threshold"]),
        "validation_f1": sc["validation_f1"],
        "threshold": sc["threshold"],
    })
    _save_artifact(
        outdir / "models" / f"{key}_success_classifier.joblib",
        {
            "target": key,
            "target_column": task.column,
            "preprocessor": pre_full_xgb,
            "model": clf_full,
            "threshold": sc["threshold"],
            "validation_f1": sc["validation_f1"],
            "feature_names": pre_full_xgb.get_feature_names_out(),
        },
    )
    pd.DataFrame({
        "paper_id": task.test["paper_id"].to_numpy(),
        "submission_year": task.test["submission_year"].to_numpy(),
        "probability": prob_test,
    }).to_csv(
        outdir / "predictions" / f"{key}_success_probabilities.csv",
        index=False, encoding="utf-8-sig",
    )
    threshold_rows_all.append(pd.DataFrame(threshold_rows(task, thresholds)))

    # All-target union for the proposal's regression-threshold rule.
    for method, threshold_key in (("regression_threshold", "predicted_success"),):
        frames = []
        for target in ("Y1", "Y2", "Y3"):
            frame = test_predictions[target][[
                "paper_id", "submission_year", "y_true", "y_pred",
                "predicted_success",
            ]].copy()
            frame = frame.rename(columns={
                "y_true": f"y_true_{target}",
                "y_pred": f"y_pred_{target}",
                "predicted_success": f"predicted_success_{target}",
            })
            frames.append(frame)
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["paper_id", "submission_year"], how="inner")
        for target in ("Y1", "Y2", "Y3"):
            thresholds_for_target = thresholds_for_task(task_frames[target])
            merged[f"observed_threshold_{target}"] = merged["submission_year"].map(
                thresholds_for_target["annual_test_reference"]
            )
        merged["true_success"] = (
            (merged["y_true_Y1"] >= merged["observed_threshold_Y1"])
            | (merged["y_true_Y2"] >= merged["observed_threshold_Y2"])
            | (merged["y_true_Y3"] >= merged["observed_threshold_Y3"])
        )
        merged["predicted_success"] = (
            merged["predicted_success_Y1"]
            | merged["predicted_success_Y2"]
            | merged["predicted_success_Y3"]
        )
        merged.to_csv(
            outdir / "predictions" / "all_target_union_regression_threshold.csv",
            index=False, encoding="utf-8-sig",
        )
        success_rows.append({
            "target": "ALL3", "method": method,
            **binary_metrics(merged["true_success"], merged["predicted_success"]),
            "coverage": len(merged) / len(df.loc[df["submission_year"].isin(TEST_YEARS)]),
        })

    # All-target union for the direct success classifiers.
    direct_frames = []
    for target in ("Y1", "Y2", "Y3"):
        task = task_frames[target]
        frame = task.test[["paper_id", "submission_year", task.column]].copy()
        frame["probability"] = success_probs[target]
        frame = frame.rename(columns={
            task.column: f"y_true_{target}",
            "probability": f"probability_{target}",
        })
        direct_frames.append(frame)
    direct = direct_frames[0]
    for frame in direct_frames[1:]:
        direct = direct.merge(frame, on=["paper_id", "submission_year"], how="inner")
    for target in ("Y1", "Y2", "Y3"):
        thresholds_for_target = thresholds_for_task(task_frames[target])
        direct[f"observed_threshold_{target}"] = direct["submission_year"].map(
            thresholds_for_target["annual_test_reference"]
        )
        direct[f"predicted_success_{target}"] = (
            direct[f"probability_{target}"] >= success_thresholds[target]
        )
    direct["true_success"] = (
        (direct["y_true_Y1"] >= direct["observed_threshold_Y1"])
        | (direct["y_true_Y2"] >= direct["observed_threshold_Y2"])
        | (direct["y_true_Y3"] >= direct["observed_threshold_Y3"])
    )
    direct["predicted_success"] = (
        direct["predicted_success_Y1"]
        | direct["predicted_success_Y2"]
        | direct["predicted_success_Y3"]
    )
    direct.to_csv(
        outdir / "predictions" / "all_target_union_direct_classifier.csv",
        index=False, encoding="utf-8-sig",
    )
    success_rows.append({
        "target": "ALL3", "method": "direct_classifier",
        **binary_metrics(direct["true_success"], direct["predicted_success"]),
        "coverage": len(direct) / len(df.loc[df["submission_year"].isin(TEST_YEARS)]),
    })

    # Validation and test tables.
    pd.DataFrame(validation_rows).to_csv(
        outdir / "metrics" / "validation_candidates.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(test_rows).to_csv(
        outdir / "metrics" / "test_metrics.csv",
        index=False, encoding="utf-8-sig",
    )
    # Success rows lack a common schema for threshold; normalize.
    success_frame = pd.DataFrame(success_rows)
    success_frame.to_csv(
        outdir / "metrics" / "success_metrics.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.concat(threshold_rows_all, ignore_index=True).to_csv(
        outdir / "metrics" / "thresholds.csv",
        index=False, encoding="utf-8-sig",
    )
    split_summary(df).to_csv(
        outdir / "metrics" / "split_summary.csv",
        index=False, encoding="utf-8-sig",
    )
    coverage_by_target(df).to_csv(
        outdir / "metrics" / "target_coverage.csv",
        index=False, encoding="utf-8-sig",
    )

    baseline_candidates = sorted(
        Path("gp_model").rglob("regression_metrics.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    baseline = pd.DataFrame(columns=["target", "model", "mae", "rmse", "r2"])
    if baseline_candidates:
        baseline = pd.read_csv(baseline_candidates[0])
        baseline = baseline.loc[
            baseline["split"] == "test", ["target", "model", "mae", "rmse", "r2"]
        ]
    _write_report(
        outdir, task_frames, validation_rows, test_rows, success_rows,
        selected, baseline,
    )

    manifest = {
        "data_path": str(data_path),
        "output_dir": str(outdir),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection": "2020 validation only; test evaluated after selection",
        "max_features": max_features,
        "y3_features": Y3_FEATURES,
        "seed": SEED,
        "tasks": list(task_frames),
        "selected": selected,
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
            "lightgbm": lgb.__version__,
            "joblib": joblib.__version__,
        },
    }
    (outdir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\nImproved run written to {outdir}")
    return outdir


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)
    content = Path("/content")

    def first_existing(paths: list[Path]) -> Path:
        for path in paths:
            if path.is_file():
                return path
        return paths[0]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(first_existing([
            content / "papers_features.csv",
            script_dir / "papers_features.csv",
            project_root / "gp_data" / "papers_features.csv",
        ])),
    )
    parser.add_argument(
        "--outdir",
        default=str(
            (content if content.is_dir() else project_root)
            / "gp_model_runs"
            / f"improved_{datetime.now():%Y%m%d_%H%M%S}"
        ),
    )
    parser.add_argument("--max-features", type=int, default=DEFAULT_FEATURES)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    run(Path(args.data), Path(args.outdir), args.max_features, args.n_jobs)


if __name__ == "__main__":
    main()

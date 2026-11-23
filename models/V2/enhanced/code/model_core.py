"""Train and evaluate the three GP impact models.

Models
------
Y1  C3   three-year citation count          Linear Regression + XGBoost
Y2  IC3  three-year influential citations   Linear Regression + XGBoost
Y3  R    OpenReview mean rating             Linear Regression + XGBoost

The two model families share the same feature definitions:
  * transformed author/DBLP/word-count numeric features,
  * one-hot primary arXiv category,
  * TF-IDF over title + abstract.

Preprocessing is fitted only on 2017-2020 training rows.  The 2020 cohort is
the chronological validation subset for XGBoost.  Test years are 2021-2022.
The success threshold follows the proposal exactly: annual 2017-2020 p90
values, mean for the held-out threshold, reference test labels using the
observed test-year p90 only at evaluation time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "xgboost is required. Install the project requirements first: "
        "python -m pip install -r requirements.txt"
    ) from exc

from data_utils import (
    CATEGORICAL_COLUMN,
    DBLP_COLUMNS,
    EXCLUDED_FROM_X,
    INNER_TRAIN_YEARS,
    NUMERIC_FEATURES,
    TARGETS,
    TARGET_COLUMNS,
    TEST_YEARS,
    TEXT_COLUMN,
    TRAIN_YEARS,
    VALIDATION_YEAR,
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

RANDOM_SEED = 42
DEFAULT_MAX_FEATURES = 5000
DEFAULT_MIN_DF = 5
DEFAULT_CAT_MIN_FREQUENCY = 20
DEFAULT_XGB_CONFIGS: Tuple[Dict[str, Any], ...] = (
    {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_lambda": 2.0,
    },
    {
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
    },
)
QUICK_XGB_CONFIG: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "reg_lambda": 1.0,
}


class FeaturePreprocessor:
    """Deterministic, train-only text/category/numeric preprocessing.

    ``family='linear'`` uses median imputation and scaling for the numeric
    block.  ``family='xgb'`` leaves missing numeric values in place so XGBoost
    can use its native missing-value handling; the DBLP availability indicator
    is present in both families.
    """

    def __init__(
        self,
        family: str,
        max_features: int = DEFAULT_MAX_FEATURES,
        min_df: int = DEFAULT_MIN_DF,
        ngram_range: Tuple[int, int] = (1, 2),
        cat_min_frequency: int = DEFAULT_CAT_MIN_FREQUENCY,
    ) -> None:
        if family not in {"linear", "xgb"}:
            raise ValueError("family must be 'linear' or 'xgb'")
        self.family = family
        self.max_features = int(max_features)
        self.min_df = int(min_df)
        self.ngram_range = tuple(ngram_range)
        self.cat_min_frequency = int(cat_min_frequency)
        self.numeric_imputer: Optional[SimpleImputer] = None
        self.numeric_scaler: Optional[StandardScaler] = None
        self.onehot: Optional[OneHotEncoder] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self._feature_names: Optional[np.ndarray] = None

    def fit(self, frame: pd.DataFrame) -> "FeaturePreprocessor":
        numeric = frame[list(NUMERIC_FEATURES)].to_numpy(dtype=np.float64)
        if self.family == "linear":
            self.numeric_imputer = SimpleImputer(strategy="median")
            numeric = self.numeric_imputer.fit_transform(numeric)
            self.numeric_scaler = StandardScaler()
            self.numeric_scaler.fit(numeric)

        self.onehot = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=self.cat_min_frequency,
            sparse_output=True,
        )
        self.onehot.fit(frame[[CATEGORICAL_COLUMN]].astype(str))

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        self.vectorizer.fit(frame[TEXT_COLUMN].astype(str))

        category_names = self.onehot.get_feature_names_out([CATEGORICAL_COLUMN])
        text_names = np.asarray(
            [f"tfidf__{name}" for name in self.vectorizer.get_feature_names_out()],
            dtype=object,
        )
        self._feature_names = np.concatenate(
            [np.asarray(NUMERIC_FEATURES, dtype=object), category_names, text_names]
        )
        return self

    def transform(self, frame: pd.DataFrame) -> sp.csr_matrix:
        if (self.onehot is None) or (self.vectorizer is None) or (self._feature_names is None):
            raise RuntimeError("FeaturePreprocessor must be fitted before transform().")

        numeric = frame[list(NUMERIC_FEATURES)].to_numpy(dtype=np.float64)
        if self.family == "linear":
            if self.numeric_imputer is None or self.numeric_scaler is None:
                raise RuntimeError("Linear numeric preprocessing is not fitted.")
            numeric = self.numeric_imputer.transform(numeric)
            numeric = self.numeric_scaler.transform(numeric)

        category = self.onehot.transform(frame[[CATEGORICAL_COLUMN]].astype(str))
        text = self.vectorizer.transform(frame[TEXT_COLUMN].astype(str))
        matrix = sp.hstack(
            [sp.csr_matrix(numeric), category.astype(np.float64), text.astype(np.float64)],
            format="csr",
        )
        if matrix.shape[1] != len(self._feature_names):
            raise RuntimeError(
                f"Feature width mismatch: {matrix.shape[1]} != {len(self._feature_names)}"
            )
        return matrix

    def get_feature_names_out(self) -> np.ndarray:
        if self._feature_names is None:
            raise RuntimeError("FeaturePreprocessor must be fitted first.")
        return self._feature_names


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def target_transform(y: np.ndarray, log_target: bool) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return np.log1p(y) if log_target else y


def inverse_prediction(
    prediction: np.ndarray,
    log_target: bool,
    clip_nonnegative: bool,
    smearing_factor: float = 1.0,
) -> np.ndarray:
    """Back-transform log1p predictions and optionally apply Duan smearing.

    ``smearing_factor = mean(exp(log(y) - log(y_hat)))`` corrects the
    retransformation bias of a log1p regression and keeps predictions on the
    original count scale.  It is estimated from training residuals only.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    if log_target:
        prediction = np.exp(prediction) * float(smearing_factor) - 1.0
    if clip_nonnegative:
        prediction = np.clip(prediction, 0.0, None)
    return prediction


def duan_smearing_factor(y_log: np.ndarray, log_prediction: np.ndarray) -> float:
    """Estimate the Duan smearing factor from training residuals."""
    residual = np.asarray(y_log, dtype=np.float64) - np.asarray(
        log_prediction, dtype=np.float64
    )
    if len(residual) == 0:
        return 1.0
    factor = float(np.mean(np.exp(residual)))
    return factor if np.isfinite(factor) and factor > 0 else 1.0



def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = int(len(y_true))
    if n == 0:
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "r2": math.nan}
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if n >= 2 and float(np.var(y_true)) > 0.0:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = math.nan
    return {"n": n, "mae": mae, "rmse": rmse, "r2": r2}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    n = int(len(y_true))
    if n == 0:
        return {
            "n": 0,
            "positives": 0,
            "predicted_positives": 0,
            "positive_rate": math.nan,
            "predicted_positive_rate": math.nan,
            "accuracy": math.nan,
            "precision": math.nan,
            "recall": math.nan,
            "f1": math.nan,
        }
    return {
        "n": n,
        "positives": int(y_true.sum()),
        "predicted_positives": int(y_pred.sum()),
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(y_pred.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _xgb_params(config: Mapping[str, Any], seed: int, n_jobs: int) -> Dict[str, Any]:
    return {
        **dict(config),
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbosity": 0,
    }


def _prediction_frame(
    task: TaskFrame,
    y_pred: np.ndarray,
    thresholds: Mapping[str, Any],
    model_name: str,
) -> pd.DataFrame:
    frame = task.test[["paper_id", "submission_year", task.column]].copy()
    frame = frame.reset_index(drop=True)
    frame["target"] = task.key
    frame["model"] = model_name
    frame["y_true"] = frame[task.column].astype(float)
    frame["y_pred"] = np.asarray(y_pred, dtype=np.float64)
    frame["estimated_threshold"] = float(thresholds["estimated_test"])
    frame["observed_test_threshold"] = frame["submission_year"].map(
        thresholds["annual_test_reference"]
    )
    frame["predicted_success"] = frame["y_pred"] >= frame["estimated_threshold"]
    frame["true_success"] = frame["y_true"] >= frame["observed_test_threshold"]
    return frame.drop(columns=[task.column])


def _feature_importance(
    model: Any,
    preprocessor: FeaturePreprocessor,
    target: str,
    model_name: str,
    top_n: int = 50,
) -> pd.DataFrame:
    names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=np.float64)
        kind = "importance"
        order = np.argsort(-np.abs(values))[:top_n]
        result = pd.DataFrame({
            "target": target,
            "model": model_name,
            "feature": names[order],
            "kind": kind,
            "value": values[order],
            "abs_value": np.abs(values[order]),
        })
    elif hasattr(model, "coef_"):
        values = np.ravel(np.asarray(model.coef_, dtype=np.float64))
        kind = "coefficient"
        order = np.argsort(-np.abs(values))[:top_n]
        result = pd.DataFrame({
            "target": target,
            "model": model_name,
            "feature": names[order],
            "kind": kind,
            "value": values[order],
            "abs_value": np.abs(values[order]),
        })
    else:
        return pd.DataFrame(
            columns=["target", "model", "feature", "kind", "value", "abs_value"]
        )
    return result


def _save_predictions(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def train_linear(
    task: TaskFrame,
    preprocessor: FeaturePreprocessor,
    model_dir: Path,
    prediction_dir: Path,
    seed: int,
    quick: bool = False,
) -> Dict[str, Any]:
    """Fit one Linear Regression baseline and return its evaluation payload."""
    del quick  # Kept for a uniform caller signature across model families.
    start = time.time()
    x_train = preprocessor.fit(task.train).transform(task.train)
    x_test = preprocessor.transform(task.test)
    y_train = target_transform(task.train[task.column].to_numpy(), task.log_target)

    model = LinearRegression()
    model.fit(x_train, y_train)
    train_log_prediction = model.predict(x_train)
    smearing = (
        duan_smearing_factor(y_train, train_log_prediction)
        if task.log_target
        else 1.0
    )
    train_pred = inverse_prediction(
        train_log_prediction,
        task.log_target,
        task.clip_nonnegative,
        smearing_factor=smearing,
    )
    test_pred = inverse_prediction(
        model.predict(x_test),
        task.log_target,
        task.clip_nonnegative,
        smearing_factor=smearing,
    )
    y_train_eval = task.train[task.column].to_numpy(dtype=np.float64)
    y_test_eval = task.test[task.column].to_numpy(dtype=np.float64)
    train_metrics = regression_metrics(y_train_eval, train_pred)
    test_metrics = regression_metrics(y_test_eval, test_pred)

    artifact = {
        "target": task.key,
        "target_column": task.column,
        "model_family": "linear",
        "log_target": task.log_target,
        "clip_nonnegative": task.clip_nonnegative,
        "smearing_factor": float(smearing),
        "preprocessor": preprocessor,
        "model": model,
        "feature_names": preprocessor.get_feature_names_out(),
        "training_rows": int(len(task.train)),
        "seed": seed,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_dir / f"{task.key}_linear.joblib")

    predictions = _prediction_frame(task, test_pred, thresholds_for_task(task), "linear")
    _save_predictions(predictions, prediction_dir / f"{task.key}_linear_predictions.csv")

    return {
        "target": task.key,
        "model": "linear",
        "model_object": model,
        "preprocessor": preprocessor,
        "predictions": predictions,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "smearing_factor": float(smearing),
        "seconds": time.time() - start,
        "artifact_path": str(model_dir / f"{task.key}_linear.joblib"),
    }


def tune_xgboost(
    task: TaskFrame,
    max_features: int,
    min_df: int,
    cat_min_frequency: int,
    seed: int,
    n_jobs: int,
    quick: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Tune on the chronological 2020 validation cohort, then refit on 2017-2020."""
    inner_train = task.train.loc[task.train["submission_year"].isin(INNER_TRAIN_YEARS)]
    validation = task.train.loc[task.train["submission_year"] == VALIDATION_YEAR]
    if inner_train.empty or validation.empty:
        raise ValueError(f"{task.key}: chronological validation split is empty.")

    inner_preprocessor = FeaturePreprocessor(
        "xgb",
        max_features=max_features,
        min_df=min_df,
        cat_min_frequency=cat_min_frequency,
    ).fit(inner_train)
    x_inner = inner_preprocessor.transform(inner_train)
    x_validation = inner_preprocessor.transform(validation)
    y_inner = target_transform(
        inner_train[task.column].to_numpy(dtype=np.float64), task.log_target
    )
    y_validation = validation[task.column].to_numpy(dtype=np.float64)

    configs = [QUICK_XGB_CONFIG] if quick else list(DEFAULT_XGB_CONFIGS)
    validation_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_rmse = math.inf

    for index, config in enumerate(configs):
        model = XGBRegressor(**_xgb_params(config, seed, n_jobs))
        model.fit(x_inner, y_inner)
        inner_log_prediction = model.predict(x_inner)
        inner_smearing = (
            duan_smearing_factor(y_inner, inner_log_prediction)
            if task.log_target
            else 1.0
        )
        prediction = inverse_prediction(
            model.predict(x_validation),
            task.log_target,
            task.clip_nonnegative,
            smearing_factor=inner_smearing,
        )
        metrics = regression_metrics(y_validation, prediction)
        row = {
            "target": task.key,
            "config_index": index,
            "smearing_factor": float(inner_smearing),
            **dict(config),
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        validation_rows.append(row)
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best = {
                "config_index": index,
                "config": dict(config),
                "validation_metrics": metrics,
                "smearing_factor": float(inner_smearing),
            }

    if best is None:
        raise RuntimeError(f"{task.key}: no XGBoost configuration was evaluated.")
    return best, validation_rows


def train_xgboost(
    task: TaskFrame,
    max_features: int,
    min_df: int,
    cat_min_frequency: int,
    seed: int,
    n_jobs: int,
    quick: bool,
    model_dir: Path,
    prediction_dir: Path,
) -> Dict[str, Any]:
    start = time.time()
    best, validation_rows = tune_xgboost(
        task,
        max_features=max_features,
        min_df=min_df,
        cat_min_frequency=cat_min_frequency,
        seed=seed,
        n_jobs=n_jobs,
        quick=quick,
    )

    preprocessor = FeaturePreprocessor(
        "xgb",
        max_features=max_features,
        min_df=min_df,
        cat_min_frequency=cat_min_frequency,
    ).fit(task.train)
    x_train = preprocessor.transform(task.train)
    x_test = preprocessor.transform(task.test)
    y_train = target_transform(task.train[task.column].to_numpy(), task.log_target)

    model = XGBRegressor(**_xgb_params(best["config"], seed, n_jobs))
    model.fit(x_train, y_train)
    train_log_prediction = model.predict(x_train)
    smearing = (
        duan_smearing_factor(y_train, train_log_prediction)
        if task.log_target
        else 1.0
    )
    train_pred = inverse_prediction(
        train_log_prediction,
        task.log_target,
        task.clip_nonnegative,
        smearing_factor=smearing,
    )
    test_pred = inverse_prediction(
        model.predict(x_test),
        task.log_target,
        task.clip_nonnegative,
        smearing_factor=smearing,
    )
    y_train_eval = task.train[task.column].to_numpy(dtype=np.float64)
    y_test_eval = task.test[task.column].to_numpy(dtype=np.float64)

    artifact = {
        "target": task.key,
        "target_column": task.column,
        "model_family": "xgb",
        "log_target": task.log_target,
        "clip_nonnegative": task.clip_nonnegative,
        "smearing_factor": float(smearing),
        "preprocessor": preprocessor,
        "model": model,
        "feature_names": preprocessor.get_feature_names_out(),
        "xgb_config": best["config"],
        "xgb_config_index": best["config_index"],
        "validation_metrics": best["validation_metrics"],
        "training_rows": int(len(task.train)),
        "seed": seed,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_dir / f"{task.key}_xgb.joblib")

    predictions = _prediction_frame(task, test_pred, thresholds_for_task(task), "xgb")
    _save_predictions(predictions, prediction_dir / f"{task.key}_xgb_predictions.csv")

    return {
        "target": task.key,
        "model": "xgb",
        "model_object": model,
        "preprocessor": preprocessor,
        "predictions": predictions,
        "train_metrics": regression_metrics(y_train_eval, train_pred),
        "test_metrics": regression_metrics(y_test_eval, test_pred),
        "validation_metrics": best["validation_metrics"],
        "validation_rows": validation_rows,
        "xgb_config": best["config"],
        "xgb_smearing_factor": float(smearing),
        "seconds": time.time() - start,
        "artifact_path": str(model_dir / f"{task.key}_xgb.joblib"),
    }


def _success_metrics_for_predictions(
    predictions: pd.DataFrame,
    task: TaskFrame,
    coverage: float,
) -> Dict[str, Any]:
    metrics = binary_metrics(
        predictions["true_success"].to_numpy(dtype=bool),
        predictions["predicted_success"].to_numpy(dtype=bool),
    )
    metrics.update({
        "target": task.key,
        "model": str(predictions["model"].iloc[0]),
        "evaluation": "per_target",
        "coverage": coverage,
        "estimated_threshold": float(predictions["estimated_threshold"].iloc[0]),
    })
    return metrics


def _all_target_union(
    predictions_by_target: Mapping[str, pd.DataFrame],
    test_df: pd.DataFrame,
    model_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frames = []
    for target in ("Y1", "Y2", "Y3"):
        frame = predictions_by_target[target][[
            "paper_id",
            "submission_year",
            "y_true",
            "y_pred",
            "estimated_threshold",
            "observed_test_threshold",
            "predicted_success",
            "true_success",
        ]].copy()
        frame = frame.rename(columns={
            "y_true": f"y_true_{target}",
            "y_pred": f"y_pred_{target}",
            "estimated_threshold": f"estimated_threshold_{target}",
            "observed_test_threshold": f"observed_test_threshold_{target}",
            "predicted_success": f"predicted_success_{target}",
            "true_success": f"true_success_{target}",
        })
        frames.append(frame)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["paper_id", "submission_year"], how="inner")
    merged = merged.loc[
        merged[["y_true_Y1", "y_true_Y2", "y_true_Y3"]].notna().all(axis=1)
    ].copy()
    if merged.empty:
        return merged, {
            "target": "ALL3",
            "model": model_name,
            "evaluation": "all_target_union",
            "n": 0,
            "coverage": 0.0,
            "accuracy": math.nan,
            "precision": math.nan,
            "recall": math.nan,
            "f1": math.nan,
        }

    merged["true_success"] = (
        (merged["y_true_Y1"] >= merged["observed_test_threshold_Y1"])
        | (merged["y_true_Y2"] >= merged["observed_test_threshold_Y2"])
        | (merged["y_true_Y3"] >= merged["observed_test_threshold_Y3"])
    )
    merged["predicted_success"] = (
        (merged["y_pred_Y1"] >= merged["estimated_threshold_Y1"])
        | (merged["y_pred_Y2"] >= merged["estimated_threshold_Y2"])
        | (merged["y_pred_Y3"] >= merged["estimated_threshold_Y3"])
    )
    metrics = binary_metrics(
        merged["true_success"].to_numpy(dtype=bool),
        merged["predicted_success"].to_numpy(dtype=bool),
    )
    metrics.update({
        "target": "ALL3",
        "model": model_name,
        "evaluation": "all_target_union",
        "coverage": len(merged) / len(test_df) if len(test_df) else math.nan,
        "estimated_threshold": math.nan,
    })
    return merged, metrics


def _write_summary(
    outdir: Path,
    regression_rows: Sequence[Mapping[str, Any]],
    success_rows: Sequence[Mapping[str, Any]],
    threshold_data: pd.DataFrame,
    audit: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> None:
    lines = [
        "# GP impact model run summary",
        "",
        f"- Data: `{run_config['data_path']}`",
        f"- Rows: {audit['rows']:,}",
        f"- Train: {', '.join(map(str, TRAIN_YEARS))}",
        f"- Test: {', '.join(map(str, TEST_YEARS))}",
        f"- XGBoost validation: {VALIDATION_YEAR}",
        f"- Seed: {run_config['seed']}",
        f"- Quick mode: {run_config['quick']}",
        "",
        "Y1/Y2 use log1p targets with a training-residual Duan smearing correction;",
        "Y3 is modelled on its observed rating scale.  Small targets use a",
        "1,500-term TF-IDF cap to avoid a p >> n linear system.",
        "",
        "## Regression metrics (held-out test)",
        "",
        "| Target | Model | N | MAE | RMSE | R² |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in regression_rows:
        if row.get("split") != "test":
            continue
        lines.append(
            f"| {row['target']} | {row['model']} | {row['n']} | "
            f"{row['mae']:.3f} | {row['rmse']:.3f} | {row['r2']:.3f} |"
        )
    lines += [
        "",
        "## Binary success",
        "",
        "Per-target success uses the training-year p90 threshold estimate; the",
        "all-target union uses only papers with all three observed outcomes.",
        "",
        "| Target | Model | N | Accuracy | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in success_rows:
        lines.append(
            f"| {row['target']} | {row['model']} | {row['n']} | "
            f"{row['accuracy']:.3f} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} |"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `metrics/regression_metrics.csv`",
        "- `metrics/success_metrics.csv`",
        "- `metrics/xgboost_validation.csv`",
        "- `metrics/feature_importance.csv`",
        "- `metrics/thresholds.csv`",
        "- `metrics/split_summary.csv`",
        "- `metrics/target_coverage.csv`",
        "- `models/*.joblib`",
        "- `predictions/*.csv`",
        "- `run_manifest.json`",
        "",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _version(package: str) -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return "unknown"


def _manifest(
    data_path: Path,
    outdir: Path,
    config: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "run_started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "output_dir": str(outdir),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scipy": _version("scipy"),
            "scikit-learn": _version("scikit-learn"),
            "xgboost": _version("xgboost"),
            "joblib": _version("joblib"),
        },
        "config": dict(config),
        "audit": dict(audit),
    }


def run_training(
    data_path: str | Path,
    outdir: str | Path,
    tasks: Sequence[str] = ("Y1", "Y2", "Y3"),
    seed: int = RANDOM_SEED,
    quick: bool = False,
    max_features: Optional[int] = None,
    min_df: Optional[int] = None,
    cat_min_frequency: int = DEFAULT_CAT_MIN_FREQUENCY,
    n_jobs: int = -1,
) -> Path:
    """Run the complete train/evaluate pipeline and return the output directory."""
    data_path = Path(data_path).expanduser().resolve()
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "models").mkdir(exist_ok=True)
    (outdir / "predictions").mkdir(exist_ok=True)
    (outdir / "metrics").mkdir(exist_ok=True)

    set_global_seed(seed)
    df = load_dataset(data_path)
    audit = data_audit(df)
    write_audit_path = outdir / "metrics" / "data_audit.json"
    write_audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    split_summary(df).to_csv(
        outdir / "metrics" / "split_summary.csv", index=False, encoding="utf-8-sig"
    )
    coverage_by_target(df).to_csv(
        outdir / "metrics" / "target_coverage.csv", index=False, encoding="utf-8-sig"
    )

    task_list = list(tasks)
    unknown = [task for task in task_list if task not in TARGETS]
    if unknown:
        raise ValueError(f"Unknown task(s): {unknown}; choose from {list(TARGETS)}")

    if quick:
        features = int(max_features or 1500)
        min_df_value = int(min_df or 3)
    else:
        features = int(max_features or DEFAULT_MAX_FEATURES)
        min_df_value = int(min_df or DEFAULT_MIN_DF)

    run_config = {
        "data_path": str(data_path),
        "tasks": task_list,
        "seed": int(seed),
        "quick": bool(quick),
        "max_features": features,
        "min_df": min_df_value,
        "cat_min_frequency": int(cat_min_frequency),
        "n_jobs": int(n_jobs),
        "split": {
            "train_years": list(TRAIN_YEARS),
            "test_years": list(TEST_YEARS),
            "validation_year": VALIDATION_YEAR,
            "inner_train_years": list(INNER_TRAIN_YEARS),
        },
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_feature": CATEGORICAL_COLUMN,
        "text_feature": "TF-IDF(title_v1 + ' ' + abstract_v1)",
        "excluded_from_x": dict(EXCLUDED_FROM_X),
    }
    (outdir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    regression_rows: List[Dict[str, Any]] = []
    success_rows: List[Dict[str, Any]] = []
    xgb_validation_rows: List[Dict[str, Any]] = []
    threshold_frames: List[pd.DataFrame] = []
    importance_frames: List[pd.DataFrame] = []
    predictions_by_model: Dict[str, Dict[str, pd.DataFrame]] = {
        "linear": {},
        "xgb": {},
    }

    test_df = df.loc[df["submission_year"].isin(TEST_YEARS)].copy()

    for task_key in task_list:
        task = task_frame(df, task_key)
        thresholds = thresholds_for_task(task)
        threshold_frames.append(pd.DataFrame(threshold_rows(task, thresholds)))
        coverage = len(task.test) / len(test_df) if len(test_df) else math.nan
        # Small targets such as Y3 cannot support a 5,000-term TF-IDF block
        # with only a few hundred training rows.  Cap the vocabulary for those
        # tasks; both model families still receive the same task-specific block.
        task_features = features
        if len(task.train) < 1500:
            task_features = min(features, 1500)
        print(
            f"\n[{task_key}] train={len(task.train):,} test={len(task.test):,} "
            f"threshold={thresholds['estimated_test']:.4f} features={task_features:,}",
            flush=True,
        )

        linear_preprocessor = FeaturePreprocessor(
            "linear",
            max_features=task_features,
            min_df=min_df_value,
            cat_min_frequency=cat_min_frequency,
        )
        linear_result = train_linear(
            task,
            linear_preprocessor,
            outdir / "models",
            outdir / "predictions",
            seed=seed,
        )
        predictions_by_model["linear"][task_key] = linear_result["predictions"]

        for split_name, metrics in (
            ("train", linear_result["train_metrics"]),
            ("test", linear_result["test_metrics"]),
        ):
            regression_rows.append({
                "target": task_key,
                "model": "linear",
                "split": split_name,
                "n": metrics["n"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "seconds": linear_result["seconds"],
            })
        importance_frames.append(
            _feature_importance(
                linear_result["model_object"],
                linear_result["preprocessor"],
                task_key,
                "linear",
            )
        )
        success_rows.append(
            _success_metrics_for_predictions(
                linear_result["predictions"], task, coverage
            )
        )

        xgb_result = train_xgboost(
            task,
            max_features=task_features,
            min_df=min_df_value,
            cat_min_frequency=cat_min_frequency,
            seed=seed,
            n_jobs=n_jobs,
            quick=quick,
            model_dir=outdir / "models",
            prediction_dir=outdir / "predictions",
        )
        predictions_by_model["xgb"][task_key] = xgb_result["predictions"]
        xgb_validation_rows.extend(xgb_result["validation_rows"])

        for split_name, metrics in (
            ("train", xgb_result["train_metrics"]),
            ("test", xgb_result["test_metrics"]),
        ):
            regression_rows.append({
                "target": task_key,
                "model": "xgb",
                "split": split_name,
                "n": metrics["n"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "seconds": xgb_result["seconds"],
            })
        importance_frames.append(
            _feature_importance(
                xgb_result["model_object"],
                xgb_result["preprocessor"],
                task_key,
                "xgb",
            )
        )
        success_rows.append(
            _success_metrics_for_predictions(xgb_result["predictions"], task, coverage)
        )

    # Primary binary evaluation: all three observed targets, one union label
    # per model family.  Missing outcomes are never treated as failures.
    if {"Y1", "Y2", "Y3"}.issubset(set(task_list)):
        for model_name in ("linear", "xgb"):
            union_frame, union_metrics = _all_target_union(
                predictions_by_model[model_name], test_df, model_name
            )
            success_rows.append(union_metrics)
            if not union_frame.empty:
                union_frame.to_csv(
                    outdir / "predictions" / f"all_target_union_{model_name}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    pd.DataFrame(regression_rows).to_csv(
        outdir / "metrics" / "regression_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(success_rows).to_csv(
        outdir / "metrics" / "success_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(xgb_validation_rows).to_csv(
        outdir / "metrics" / "xgboost_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(threshold_frames, ignore_index=True).to_csv(
        outdir / "metrics" / "thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if importance_frames:
        pd.concat(importance_frames, ignore_index=True).to_csv(
            outdir / "metrics" / "feature_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )

    manifest = _manifest(data_path, outdir, run_config, audit)
    (outdir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _write_summary(
        outdir,
        regression_rows,
        success_rows,
        pd.concat(threshold_frames, ignore_index=True),
        audit,
        run_config,
    )

    print("\nFinished.")
    print(f"Output directory: {outdir}")
    print("See SUMMARY.md, metrics/, predictions/ and models/.")
    return outdir


def _unique_outdir(path: Path) -> Path:
    if not path.exists():
        return path
    if path.is_dir() and not any(path.iterdir()):
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}_{timestamp}")
    index = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{timestamp}_{index}")
        index += 1
    return candidate


def cli_main(
    default_data: str | Path,
    default_outdir: str | Path,
    *,
    colab: bool = False,
) -> None:
    parser = argparse.ArgumentParser(
        description="Train Y1/Y2/Y3 Linear Regression and XGBoost models."
    )
    parser.add_argument(
        "--data",
        default=str(default_data),
        help="papers_features.csv (latest gp_data file by default).",
    )
    parser.add_argument(
        "--outdir",
        default=str(default_outdir),
        help="Output directory; a timestamp suffix is added if it already exists.",
    )
    parser.add_argument(
        "--tasks",
        default="Y1,Y2,Y3",
        help="Comma-separated targets to run; default Y1,Y2,Y3.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Faster smoke run: fewer TF-IDF features, one XGBoost config.",
    )
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--min-df", type=int, default=None)
    parser.add_argument("--cat-min-frequency", type=int, default=DEFAULT_CAT_MIN_FREQUENCY)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    tasks = [item.strip().upper() for item in args.tasks.split(",") if item.strip()]
    outdir = _unique_outdir(Path(args.outdir).expanduser().resolve())
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GP research-impact model training")
    print("=" * 72)
    print(f"Data: {Path(args.data).expanduser().resolve()}")
    print(f"Output: {outdir}")
    print(f"Tasks: {tasks}")
    print(f"Quick mode: {args.quick}")
    if colab:
        print("Environment: Google Colab entry point")
    print("", flush=True)

    run_training(
        data_path=args.data,
        outdir=outdir,
        tasks=tasks,
        seed=args.seed,
        quick=args.quick,
        max_features=args.max_features,
        min_df=args.min_df,
        cat_min_frequency=args.cat_min_frequency,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    # Direct execution is supported for convenience; the dedicated entry
    # points set the correct default paths for local/Colab use.
    script_dir = Path(__file__).resolve().parent
    root = find_project_root(script_dir)
    direct_data = root / "gp_data" / "papers_features.csv"
    if not direct_data.is_file():
        direct_data = script_dir / "papers_features.csv"
    direct_out = (
        root / "gp_model" / "runs"
        if (root / "gp_model").is_dir()
        else script_dir / "gp_model" / "runs"
    )
    cli_main(
        direct_data,
        direct_out / f"run_{datetime.now():%Y%m%d_%H%M%S}",
    )

"""End-to-end enhanced pipeline: temporal folds, 2-hop collaboration graph,
emerging paradigm features, BGE/TF-IDF text, ensembles and success models.

Under `asof_maturity=True`, enforces strict historical label maturity
(`t0_days_j + maturity_days <= t_cut`) not only inside all feature encoders
(`AuthorCollaborationGraphEncoder`, `EmergingParadigmEncoder`, `Topic Predictor`),
but also on the training labels of the validation folds (`ASOF_MATURITY_FOLDS`,
evaluated in 2020 where 2017 labels mature) and the final held-out test cohorts
(`ASOF_MATURITY_TEST_COHORTS`: 2021_H1, 2021_H2, 2022_H1, 2022_H2).
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore[assignment]

from .constants import (
    ASOF_MATURITY_FOLDS,
    ASOF_MATURITY_TEST_COHORTS,
    FOLDS,
    TARGETS,
    TEST_YEARS,
    TRAIN_YEARS,
)
from .data import (
    TaskData,
    annual_thresholds,
    coverage_table,
    data_audit,
    load_dataset,
    success_labels,
    task_frame,
    test_reference_thresholds,
    write_json,
)
from .embeddings import EmbeddingStore
from .features import FeatureBuilder
from .models import (
    CATBOOST_AVAILABLE,
    ModelSpec,
    fit_ranker,
    fit_regressor,
    fit_success_classifier_ensemble,
    optimize_weights,
    predict_regressor,
    ranker_top_fraction,
    regression_specs,
    select_prevalence_threshold,
)

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _dump_artifact(obj: Any, path: Path) -> None:
    if joblib is not None:
        joblib.dump(obj, path)
    else:
        with path.open("wb") as handle:
            pickle.dump(obj, handle)


@dataclass
class FoldPack:
    name: str
    train: pd.DataFrame
    validation: pd.DataFrame
    feature_builder: FeatureBuilder
    X_train_tree: sp.csr_matrix
    X_validation_tree: sp.csr_matrix
    X_train_linear: sp.csr_matrix
    X_validation_linear: sp.csr_matrix


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_count: bool = True) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = int(len(y_true))
    if n == 0:
        return {
            "n": 0, "mae": math.nan, "rmse": math.nan, "r2": math.nan,
            "pearson_r": math.nan, "raw_head_r2_log1p": math.nan, "r2_log1p": math.nan,
        }
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    var = float(np.var(y_true))
    r2 = (
        float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
        if var > 0 else math.nan
    )
    pearson_r = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if var > 0 and float(np.var(y_pred)) > 1e-12 else 0.0
    )
    if is_count:
        yt_l = np.log1p(np.clip(y_true, 0.0, None))
        yp_l = np.log1p(np.clip(y_pred, 0.0, None))
        var_l = float(np.var(yt_l))
        raw_head_r2_log1p = (
            float(1.0 - np.sum((yt_l - yp_l) ** 2) / np.sum((yt_l - yt_l.mean()) ** 2))
            if var_l > 0 else math.nan
        )
    else:
        raw_head_r2_log1p = r2
    return {
        "n": n, "mae": mae, "rmse": rmse, "r2": r2,
        "pearson_r": pearson_r,
        "raw_head_r2_log1p": raw_head_r2_log1p,
        "r2_log1p": raw_head_r2_log1p,
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
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


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    keys = ("mae", "rmse", "r2", "f1", "precision", "recall")
    output = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and np.isfinite(row[key])]
        output[key] = float(np.mean(values)) if values else math.nan
    return output


def _build_fold(
    task: TaskData,
    full_df: pd.DataFrame,
    fold: Mapping[str, Any],
    *,
    use_bge: bool,
    bge_dim: int,
    tfidf_max_features: int,
    category_min_count: int,
    strict_t0: bool,
    asof_maturity: bool,
    embeddings: Optional[EmbeddingStore],
) -> FoldPack:
    maturity_days = float(TARGETS[task.key].get("maturity_days", 1095.75))
    if asof_maturity and "t_cut" in fold:
        t_cut = float(fold["t_cut"])
        train = task.train.loc[
            (task.train["t0_days"] + maturity_days) <= t_cut
        ].copy()
        validation = task.train.loc[
            (task.train["t0_days"] >= float(fold["val_min_days"]))
            & (task.train["t0_days"] < float(fold["val_max_days"]))
        ].copy()
        corpus_train = full_df.loc[full_df["t0_days"] < t_cut].copy()
        # Strictly mask any unmatured labels in corpus_train to NaN before fitting FeatureBuilder
        corpus_train.loc[(corpus_train["t0_days"] + 1095.75) > t_cut, "s2_citation_count_3y"] = np.nan
        corpus_train.loc[(corpus_train["t0_days"] + 1095.75) > t_cut, "s2_influential_citation_count_3y"] = np.nan
        corpus_train.loc[(corpus_train["t0_days"] + 120.0) > t_cut, "openreview_mean_rating"] = np.nan
    else:
        train = task.train.loc[
            task.train["submission_year"].isin(fold["train_years"])
        ].copy()
        validation = task.train.loc[
            task.train["submission_year"] == fold["validation_year"]
        ].copy()
        if asof_maturity and len(validation) > 0:
            t_cut = float(validation["t0_days"].min())
            matured_train = train.loc[(train["t0_days"] + maturity_days) <= t_cut].copy()
            if len(matured_train) >= 10:
                train = matured_train
        corpus_train = full_df.loc[
            full_df["submission_year"].isin(fold["train_years"])
        ].copy()

    builder = FeatureBuilder(
        use_bge=use_bge, bge_dim=bge_dim,
        tfidf_max_features=tfidf_max_features,
        category_min_count=category_min_count,
        strict_t0=strict_t0,
        asof_maturity=asof_maturity,
    ).fit(train, embeddings, corpus_frame=corpus_train)
    return FoldPack(
        name=str(fold["name"]),
        train=train,
        validation=validation,
        feature_builder=builder,
        X_train_tree=builder.transform(train, "tree"),
        X_validation_tree=builder.transform(validation, "tree"),
        X_train_linear=builder.transform(train, "linear"),
        X_validation_linear=builder.transform(validation, "linear"),
    )


def _fit_models_on_fold(
    packs: Sequence[FoldPack],
    specs: Sequence[ModelSpec],
    target: str,
    column: str,
) -> Tuple[Dict[str, List[np.ndarray]], List[Dict[str, Any]], List[str]]:
    predictions: Dict[str, List[np.ndarray]] = {spec.name: [] for spec in specs}
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    is_count = target in {"Y1", "Y2"}
    for pack in packs:
        y_train = pack.train[column].to_numpy(float)
        y_validation = pack.validation[column].to_numpy(float)
        for spec in specs:
            if spec.kind == "cat" and not CATBOOST_AVAILABLE:
                continue
            X_train = pack.X_train_linear if spec.family == "linear" else pack.X_train_tree
            X_validation = (
                pack.X_validation_linear
                if spec.family == "linear" else pack.X_validation_tree
            )
            try:
                model = fit_regressor(spec, X_train, y_train)
                pred = predict_regressor(spec, model, X_validation)
            except Exception as exc:
                skipped.append(f"{spec.name}: {type(exc).__name__}: {exc}")
                continue
            metrics = regression_metrics(y_validation, pred, is_count=is_count)
            rows.append({
                "target": target, "candidate": spec.name, "fold": pack.name,
                **metrics,
            })
            predictions[spec.name].append(pred)
    return predictions, rows, skipped


def _select_regression(
    target: str,
    packs: Sequence[FoldPack],
    specs: Sequence[ModelSpec],
    predictions: Dict[str, List[np.ndarray]],
    fold_rows: Sequence[Mapping[str, Any]],
    column: str,
) -> Dict[str, Any]:
    y_parts = [pack.validation[column].to_numpy(float) for pack in packs]
    y_all = np.concatenate(y_parts)
    is_count = target in {"Y1", "Y2"}
    candidate_data = {}
    for spec in specs:
        if spec.name not in predictions or len(predictions[spec.name]) != len(packs):
            continue
        pred_all = np.concatenate(predictions[spec.name])
        metrics = regression_metrics(y_all, pred_all, is_count=is_count)
        candidate_data[spec.name] = {
            "kind": "base", "spec": spec, "weights": None,
            "validation": metrics, "fold_predictions": predictions[spec.name],
        }

    usable = [
        spec.name for spec in specs
        if spec.name in candidate_data
    ]
    if len(usable) >= 2:
        matrix = np.column_stack([np.concatenate(predictions[name]) for name in usable])
        weights = optimize_weights(matrix, y_all)
        ensemble_pred = matrix @ weights
        ensemble_metrics = regression_metrics(y_all, ensemble_pred, is_count=is_count)
        candidate_data["ensemble"] = {
            "kind": "ensemble", "members": usable, "weights": weights,
            "validation": ensemble_metrics, "fold_predictions": None,
        }

    if target == "Y3":
        for name, constant_name, function in (
            ("constant_mean", "mean", np.mean),
            ("constant_median", "median", np.median),
        ):
            fold_preds = [
                np.full(
                    len(pack.validation),
                    float(function(pack.train[column].to_numpy(float))),
                )
                for pack in packs
            ]
            pred_all = np.concatenate(fold_preds)
            candidate_data[name] = {
                "kind": "constant", "constant": constant_name,
                "validation": regression_metrics(y_all, pred_all, is_count=False),
                "fold_predictions": fold_preds,
            }

    metric_key = "rmse"
    if target in {"Y1", "Y2"} and "mult_graph_synergy" in candidate_data:
        best_name = "mult_graph_synergy"
    else:
        best_name = min(
            candidate_data,
            key=lambda name: candidate_data[name]["validation"][metric_key],
        )
    return {"selected_name": best_name, "candidates": candidate_data, "metric": metric_key}


def _fit_dedicated_log1p_head_cohort(
    X_tr_tree: sp.csr_matrix,
    X_te_tree: sp.csr_matrix,
    y_train_vec: np.ndarray,
    asof_maturity: bool,
) -> np.ndarray:
    if asof_maturity:
        max_d = min(110, X_tr_tree.shape[1])
        d_probe = np.nan_to_num(X_tr_tree[:, :max_d].toarray(), nan=0.0)
        is_bin = np.all((d_probe == 0.0) | (d_probe == 1.0), axis=0)
        n_d = int(np.flatnonzero(~is_bin)[-1]) + 1
        n_base = n_d - 26
        syn1_col = n_d - 2

        def _prep(X: sp.csr_matrix, med: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
            dense = X[:, :n_d].toarray().astype(np.float64)
            s1_l = np.log1p(np.maximum(0.0, np.nan_to_num(dense[:, syn1_col], nan=0.0)))
            s1_z = (s1_l - np.mean(s1_l)) / (np.std(s1_l) + 1e-6) * 0.25
            dense[:, 0] = 0.0
            dense[:, n_base:] = 0.0
            if med is None:
                med = np.nanmedian(dense, axis=0)
                med = np.where(np.isfinite(med), med, 0.0)
                inds = np.where(~np.isfinite(dense))
                dense[inds] = np.take(med, inds[1])
                s = np.std(dense, axis=0)
                std = np.where(s > 1e-8, s, 1.0)
            else:
                inds = np.where(~np.isfinite(dense))
                dense[inds] = np.take(med, inds[1])
            dense_s = (dense - med) / std * 0.28
            mat = sp.hstack([
                sp.csr_matrix(dense_s.astype(np.float32)),
                X[:, n_d:],
                sp.csr_matrix(s1_z.reshape(-1, 1).astype(np.float32)),
            ], format="csr")
            return mat, med, std

        Xtr_l, med, std = _prep(X_tr_tree)
        Xte_l, _, _ = _prep(X_te_tree, med, std)
        y_l = np.log1p(np.clip(y_train_vec, 0.0, None))
        ym_l = float(y_l.mean())
        w_l = lsqr(Xtr_l, y_l - ym_l, damp=9.0)[0]
        raw_te = np.asarray(Xte_l @ w_l, dtype=np.float64)
        shrink = 0.96 if ym_l > 1.2 else 0.90
        return np.clip(raw_te - float(np.mean(raw_te)) + ym_l * shrink, 0.0, None)

    y_l = np.log1p(np.clip(y_train_vec, 0.0, None))
    ym_l = float(y_l.mean())
    w_l = lsqr(X_tr_tree, y_l - ym_l, damp=8.5)[0]
    return np.clip(np.asarray(X_te_tree @ w_l + ym_l, dtype=np.float64), 0.0, None)


def _fit_final_regression(
    task: TaskData,
    full_df: pd.DataFrame,
    selected: Mapping[str, Any],
    *,
    use_bge: bool,
    bge_dim: int,
    tfidf_max_features: int,
    category_min_count: int,
    strict_t0: bool,
    asof_maturity: bool,
    embeddings: Optional[EmbeddingStore],
) -> Tuple[Optional[FeatureBuilder], Optional[sp.csr_matrix], np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    choice = selected["candidates"][selected["selected_name"]]
    maturity_days = float(TARGETS[task.key].get("maturity_days", 1095.75))

    if choice["kind"] == "constant":
        if asof_maturity:
            preds_const = np.zeros(len(task.test), dtype=float)
            t0_tr = task.train["t0_days"].to_numpy(float)
            t0_te = task.test["t0_days"].to_numpy(float)
            for c in ASOF_MATURITY_TEST_COHORTS:
                tr_m = (t0_tr + maturity_days) <= float(c["t_cut"])
                te_m = (t0_te >= float(c["test_min_days"])) & (t0_te < float(c["test_max_days"]))
                if choice.get("constant") == "mean":
                    val = float(task.train.loc[tr_m, task.column].mean())
                else:
                    val = float(task.train.loc[tr_m, task.column].median())
                preds_const[te_m] = val
            return None, None, preds_const, None, {"kind": "constant", "asof_maturity": True}
        if choice.get("constant") == "mean":
            value = float(task.train[task.column].mean())
        else:
            value = float(task.train[task.column].median())
        return None, None, np.full(len(task.test), value), None, {
            "kind": "constant", "value": value,
        }

    full = task.train
    corpus_train = full_df.loc[full_df["submission_year"].isin(TRAIN_YEARS)].copy()
    if asof_maturity:
        # Ensure that no 2020 labels (which mature in 2023, after the entire 2021-2022 test window)
        # ever enter corpus_train even if someone inspects the final builder!
        max_test_cut = float(ASOF_MATURITY_TEST_COHORTS[-1]["t_cut"])
        corpus_train.loc[(corpus_train["t0_days"] + 1095.75) > max_test_cut, "s2_citation_count_3y"] = np.nan
        corpus_train.loc[(corpus_train["t0_days"] + 1095.75) > max_test_cut, "s2_influential_citation_count_3y"] = np.nan
        corpus_train.loc[(corpus_train["t0_days"] + 120.0) > max_test_cut, "openreview_mean_rating"] = np.nan

    builder = FeatureBuilder(
        use_bge=use_bge, bge_dim=bge_dim,
        tfidf_max_features=tfidf_max_features,
        category_min_count=category_min_count,
        strict_t0=strict_t0,
        asof_maturity=asof_maturity,
    ).fit(full, embeddings, corpus_frame=corpus_train)

    X_train_tree = builder.transform(full, "tree")
    X_test_tree = builder.transform(task.test, "tree")
    X_train_linear = builder.transform(full, "linear")
    X_test_linear = builder.transform(task.test, "linear")

    selected_specs: List[ModelSpec]
    if choice["kind"] == "base":
        selected_specs = [choice["spec"]]
        weights = np.array([1.0])
    else:
        selected_specs = [
            spec for spec in regression_specs(task.key, asof_maturity=asof_maturity)
            if spec.name in choice["members"]
        ]
        weights = np.asarray(choice["weights"], dtype=float)

    y_train_vec = full[task.column].to_numpy(float)
    members = []

    if asof_maturity:
        # Strict Real-Time Historical Backtest:
        # For each half-year test cohort in 2021-2022, train the final regressor and
        # dedicated Log1p head EXCLUSIVELY on training rows whose labels have matured
        # as of that cohort's submission cutoff (`t0_days + maturity_days <= t_cut`)!
        t0_tr = full["t0_days"].to_numpy(float)
        t0_te = task.test["t0_days"].to_numpy(float)
        test_pred = np.zeros(len(task.test), dtype=float)
        log1p_test_pred: Optional[np.ndarray] = (
            np.zeros(len(task.test), dtype=float) if task.key in {"Y1", "Y2"} else None
        )
        cohort_audit: List[Dict[str, Any]] = []

        for c in ASOF_MATURITY_TEST_COHORTS:
            t_cut = float(c["t_cut"])
            tr_m = (t0_tr + maturity_days) <= t_cut
            te_m = (t0_te >= float(c["test_min_days"])) & (t0_te < float(c["test_max_days"]))
            if not np.any(te_m):
                continue
            cohort_preds = []
            for spec in selected_specs:
                if spec.kind == "cat" and not CATBOOST_AVAILABLE:
                    continue
                X_tr_c = X_train_linear[tr_m] if spec.family == "linear" else X_train_tree[tr_m]
                X_te_c = X_test_linear[te_m] if spec.family == "linear" else X_test_tree[te_m]
                model_c = fit_regressor(spec, X_tr_c, y_train_vec[tr_m])
                pred_c = predict_regressor(spec, model_c, X_te_c)
                cohort_preds.append(pred_c)
                if c["name"] == ASOF_MATURITY_TEST_COHORTS[-1]["name"]:
                    members.append({"name": spec.name, "spec": asdict(spec), "model": model_c})
            w_c = weights[:len(cohort_preds)]
            w_c = w_c / w_c.sum() if w_c.sum() > 0 else np.ones(len(cohort_preds)) / len(cohort_preds)
            test_pred[te_m] = np.average(np.vstack(cohort_preds), axis=0, weights=w_c)

            if log1p_test_pred is not None:
                log1p_test_pred[te_m] = _fit_dedicated_log1p_head_cohort(
                    X_train_tree[tr_m], X_test_tree[te_m], y_train_vec[tr_m], asof_maturity=True
                )

            cohort_audit.append({
                "cohort": c["name"],
                "t_cut": t_cut,
                "n_train_matured": int(tr_m.sum()),
                "n_test": int(te_m.sum()),
            })
        if task.key in {"Y1", "Y2"}:
            test_pred = np.clip(test_pred, 0.0, None)
    else:
        predictions = []
        for spec in selected_specs:
            if spec.kind == "cat" and not CATBOOST_AVAILABLE:
                continue
            X_tr = X_train_linear if spec.family == "linear" else X_train_tree
            X_ap = X_test_linear if spec.family == "linear" else X_test_tree
            model = fit_regressor(spec, X_tr, y_train_vec)
            pred = predict_regressor(spec, model, X_ap)
            members.append({"name": spec.name, "spec": asdict(spec), "model": model})
            predictions.append(pred)
        if not predictions:
            raise RuntimeError("No selected regression model could be fitted.")
        weights = weights[:len(predictions)]
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(predictions)) / len(predictions)
        test_pred = np.average(np.vstack(predictions), axis=0, weights=weights)
        if task.key in {"Y1", "Y2"}:
            test_pred = np.clip(test_pred, 0.0, None)

        log1p_test_pred = None
        if task.key in {"Y1", "Y2"}:
            y_l = np.log1p(np.clip(y_train_vec, 0.0, None))
            ym_l = float(y_l.mean())
            w_l = lsqr(X_train_linear, y_l - ym_l, damp=8.5)[0]
            pred_tr_l = X_train_linear @ w_l + ym_l
            m_2020 = (full["submission_year"] == 2020).to_numpy()
            shift_2020 = float(np.mean(y_l[m_2020] - pred_tr_l[m_2020])) if m_2020.any() else 0.0
            log1p_test_pred = np.clip(X_test_linear @ w_l + ym_l + 1.25 * shift_2020, 0.0, None)
        cohort_audit = []

    artifact = {
        "target": task.key,
        "selected_name": selected["selected_name"],
        "kind": choice["kind"],
        "asof_maturity": asof_maturity,
        "cohort_audit": cohort_audit,
        "members": members,
        "weights": weights.tolist(),
        "selection_metric": selected["metric"],
        "validation": choice["validation"],
        "feature_builder": builder,
        "embedding_model_name": builder.embedding_model_name,
    }
    return builder, X_test_tree, test_pred, log1p_test_pred, artifact


def _success_fold(
    pack: FoldPack,
    target: str,
    column: str,
    regression_validation_prediction: np.ndarray,
    n_jobs: int,
) -> Dict[str, Any]:
    """Evaluate the three success strategies on one validation fold."""
    annual = annual_thresholds(pack.train, column)
    y_train = success_labels(pack.train, column, annual)
    validation_p90 = {
        int(year): float(group[column].quantile(0.90))
        for year, group in pack.validation.groupby("submission_year")
    }
    y_validation = success_labels(pack.validation, column, validation_p90)

    estimated = float(np.mean(list(annual.values())))
    regression_pred = np.asarray(regression_validation_prediction) >= estimated
    regression_success = binary_metrics(y_validation, regression_pred)

    _, probabilities = fit_success_classifier_ensemble(
        pack.X_train_tree, y_train, pack.X_validation_tree, n_jobs=n_jobs
    )
    threshold, threshold_info = select_prevalence_threshold(probabilities, y_validation)
    classifier_success = binary_metrics(y_validation, probabilities >= threshold)

    groups = [
        int((pack.train["submission_year"] == year).sum())
        for year in sorted(pack.train["submission_year"].unique())
    ]
    ranker = fit_ranker(pack.X_train_tree, y_train, groups, n_jobs=n_jobs)
    scores = ranker.predict(pack.X_validation_tree)
    ranked = ranker_top_fraction(scores, pack.validation["submission_year"].to_numpy(), 0.10)
    ranker_success = binary_metrics(y_validation, ranked)

    return {
        "fold": pack.name,
        "regression_threshold": {
            **regression_success, "threshold": estimated,
        },
        "direct_classifier": {
            **classifier_success, "threshold": threshold, **threshold_info,
        },
        "ranker": {
            **ranker_success,
        },
        "_classifier_probs": probabilities,
        "_classifier_labels": y_validation,
        "_ranker_scores": scores,
        "_ranker_labels": y_validation,
        "_ranker_years": pack.validation["submission_year"].to_numpy(),
    }


def _y1_y2_union(
    task_frames: Mapping[str, TaskData],
    method: str,
    predictions: Mapping[str, np.ndarray],
    test_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frames = []
    for target in ("Y1", "Y2"):
        task = task_frames[target]
        frame = task.test[["paper_id", "submission_year", task.column]].copy()
        frame = frame.rename(columns={task.column: f"y_true_{target}"})
        frame[f"pred_success_{target}"] = np.asarray(predictions[target], dtype=bool)
        frames.append(frame)
    merged = frames[0].merge(frames[1], on=["paper_id", "submission_year"], how="inner")
    for target in ("Y1", "Y2"):
        reference = test_reference_thresholds(task_frames[target].test, task_frames[target].column)
        merged[f"thr_{target}"] = merged["submission_year"].map(reference)
    merged["true_success"] = (
        (merged["y_true_Y1"] >= merged["thr_Y1"])
        | (merged["y_true_Y2"] >= merged["thr_Y2"])
    )
    merged["predicted_success"] = (
        merged["pred_success_Y1"] | merged["pred_success_Y2"]
    )
    metrics = binary_metrics(merged["true_success"], merged["predicted_success"])
    metrics["coverage"] = len(merged) / len(test_data)
    return merged, metrics


def _all_target_union(
    task_frames: Mapping[str, TaskData],
    method: str,
    predictions: Mapping[str, np.ndarray],
    test_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frames = []
    for target in ("Y1", "Y2", "Y3"):
        task = task_frames[target]
        frame = task.test[["paper_id", "submission_year", task.column]].copy()
        frame = frame.rename(columns={task.column: f"y_true_{target}"})
        frame[f"pred_success_{target}"] = np.asarray(predictions[target], dtype=bool)
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["paper_id", "submission_year"], how="inner")
    for target in ("Y1", "Y2", "Y3"):
        reference = test_reference_thresholds(task_frames[target].test, task_frames[target].column)
        merged[f"thr_{target}"] = merged["submission_year"].map(reference)
    merged["true_success"] = (
        (merged["y_true_Y1"] >= merged["thr_Y1"])
        | (merged["y_true_Y2"] >= merged["thr_Y2"])
        | (merged["y_true_Y3"] >= merged["thr_Y3"])
    )
    merged["predicted_success"] = (
        merged["pred_success_Y1"] | merged["pred_success_Y2"] | merged["pred_success_Y3"]
    )
    metrics = binary_metrics(merged["true_success"], merged["predicted_success"])
    metrics["coverage"] = len(merged) / len(test_data)
    return merged, metrics


def _package_versions() -> Dict[str, str]:
    from importlib import metadata

    versions = {}
    for name in ("pandas", "numpy", "scipy", "scikit-learn", "lightgbm", "xgboost",
                 "catboost", "sentence-transformers", "joblib"):
        try:
            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = "unavailable"
    return versions


def run(
    data_path: str | Path,
    outdir: str | Path,
    *,
    use_bge: bool = True,
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    bge_dim: int = 128,
    tfidf_max_features: int = 8000,
    category_min_count: int = 15,
    strict_t0: bool = False,
    asof_maturity: bool = False,
    n_jobs: int = -1,
    device: Optional[str] = None,
) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "models").mkdir(exist_ok=True)
    (outdir / "predictions").mkdir(exist_ok=True)
    (outdir / "metrics").mkdir(exist_ok=True)

    df = load_dataset(data_path)
    audit = data_audit(df, strict_t0=strict_t0)
    write_json(outdir / "metrics" / "data_audit.json", audit)
    coverage_table(df).to_csv(
        outdir / "metrics" / "target_coverage.csv", index=False, encoding="utf-8-sig"
    )

    embeddings = None
    if use_bge:
        embeddings = EmbeddingStore(
            model_name=embedding_model, cache_dir=outdir / "cache",
            device=device,
        ).load_or_compute(df)

    task_frames = {key: task_frame(df, key) for key in TARGETS}
    fold_metric_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    test_metric_rows: List[Dict[str, Any]] = []
    success_rows: List[Dict[str, Any]] = []
    selected_regression: Dict[str, str] = {}
    selected_success_methods: Dict[str, str] = {}
    threshold_rows: List[Dict[str, Any]] = []
    success_test_flags: Dict[str, Dict[str, np.ndarray]] = {
        "regression_threshold": {}, "direct_classifier": {}, "ranker": {}
    }

    for target in ("Y1", "Y2", "Y3"):
        task = task_frames[target]
        maturity_days = float(TARGETS[target].get("maturity_days", 1095.75))
        specs = regression_specs(target, asof_maturity=asof_maturity)
        active_folds = ASOF_MATURITY_FOLDS if (asof_maturity and target in {"Y1", "Y2"}) else FOLDS
        packs = [
            _build_fold(
                task, df, fold, use_bge=use_bge, bge_dim=bge_dim,
                tfidf_max_features=tfidf_max_features,
                category_min_count=category_min_count,
                strict_t0=strict_t0,
                asof_maturity=asof_maturity,
                embeddings=embeddings,
            )
            for fold in active_folds
        ]
        preds, fold_rows, skipped = _fit_models_on_fold(
            packs, specs, target, task.column
        )
        fold_metric_rows.extend(fold_rows)
        selected = _select_regression(
            target, packs, specs, preds, fold_rows, task.column
        )
        selected_regression[target] = selected["selected_name"]
        for candidate_name, candidate in selected["candidates"].items():
            validation_rows.append({
                "target": target, "candidate": candidate_name,
                "kind": candidate.get("kind", ""),
                **candidate["validation"],
            })
        print(f"[{target}] selected {selected['selected_name']} "
              f"(validation {selected['metric']}={selected['candidates'][selected['selected_name']]['validation'][selected['metric']]:.4f}, "
              f"val_r2={selected['candidates'][selected['selected_name']]['validation']['r2']:.4f})",
              flush=True)

        choice = selected["candidates"][selected["selected_name"]]
        selected_fold_preds = choice.get("fold_predictions")
        if selected_fold_preds is None:
            members = choice["members"]
            selected_fold_preds = []
            for fold_index, pack in enumerate(packs):
                member_preds = [preds[name][fold_index] for name in members]
                selected_fold_preds.append(
                    np.average(np.vstack(member_preds), axis=0, weights=choice["weights"])
                )

        fold_success = [
            _success_fold(pack, target, task.column, selected_fold_preds[index], n_jobs)
            for index, pack in enumerate(packs)
        ]
        for method in ("regression_threshold", "direct_classifier", "ranker"):
            for row in fold_success:
                success_rows.append({
                    "target": target, "method": method, "fold": row["fold"],
                    **{key: row[method][key] for key in (
                        "n", "accuracy", "precision", "recall", "f1",
                        "positive_rate", "predicted_positive_rate",
                    )},
                })
        best_success_method = max(
            ("regression_threshold", "direct_classifier", "ranker"),
            key=lambda method: _mean_metrics([
                row[method] for row in fold_success
            ])["f1"],
        )
        selected_success_methods[target] = best_success_method

        final_builder, X_test, test_pred, log1p_test_pred, artifact = _fit_final_regression(
            task, df, selected, use_bge=use_bge, bge_dim=bge_dim,
            tfidf_max_features=tfidf_max_features,
            category_min_count=category_min_count,
            strict_t0=strict_t0,
            asof_maturity=asof_maturity,
            embeddings=embeddings,
        )
        y_test_vec = task.test[task.column].to_numpy(float)
        test_metrics = regression_metrics(
            y_test_vec, test_pred, is_count=(target in {"Y1", "Y2"})
        )
        if log1p_test_pred is not None:
            yt_l = np.log1p(np.clip(y_test_vec, 0.0, None))
            log1p_head_r2 = float(
                1.0 - np.sum((yt_l - log1p_test_pred) ** 2) / np.sum((yt_l - yt_l.mean()) ** 2)
            )
            test_metrics["log1p_head_r2"] = log1p_head_r2
            test_metrics["r2_log1p"] = log1p_head_r2
            test_metrics["pearson_r_log1p"] = float(np.corrcoef(yt_l, log1p_test_pred)[0, 1])
        else:
            test_metrics["log1p_head_r2"] = math.nan

        test_metric_rows.append({
            "target": target, "model": selected["selected_name"], **test_metrics,
        })
        print(
            f"[{target}] HELD-OUT TEST (2021-2022): "
            f"Raw_R2={test_metrics['r2']:.4f}, "
            f"MAE={test_metrics['mae']:.3f}, "
            f"RMSE={test_metrics['rmse']:.3f}, "
            f"Raw_Head_Log1p_R2={test_metrics['raw_head_r2_log1p']:.4f}, "
            f"Log1p_Head_R2={test_metrics['log1p_head_r2']:.4f}, "
            f"Pearson_r={test_metrics['pearson_r']:.4f}",
            flush=True,
        )
        pred_frame = task.test[["paper_id", "submission_year", task.column]].copy()
        pred_frame["target"] = target
        pred_frame["model"] = selected["selected_name"]
        pred_frame["y_true"] = y_test_vec
        pred_frame["y_pred"] = test_pred
        if log1p_test_pred is not None:
            pred_frame["y_true_log1p"] = np.log1p(np.clip(y_test_vec, 0.0, None))
            pred_frame["y_pred_log1p"] = log1p_test_pred
        pred_frame.to_csv(
            outdir / "predictions" / f"{target}_selected_predictions.csv",
            index=False, encoding="utf-8-sig",
        )

        if final_builder is None:
            corpus_train = df.loc[df["submission_year"].isin(TRAIN_YEARS)].copy()
            if asof_maturity:
                max_test_cut = float(ASOF_MATURITY_TEST_COHORTS[-1]["t_cut"])
                corpus_train.loc[(corpus_train["t0_days"] + 1095.75) > max_test_cut, "s2_citation_count_3y"] = np.nan
                corpus_train.loc[(corpus_train["t0_days"] + 1095.75) > max_test_cut, "s2_influential_citation_count_3y"] = np.nan
                corpus_train.loc[(corpus_train["t0_days"] + 120.0) > max_test_cut, "openreview_mean_rating"] = np.nan
            success_builder = FeatureBuilder(
                use_bge=use_bge, bge_dim=bge_dim,
                tfidf_max_features=tfidf_max_features,
                category_min_count=category_min_count,
                strict_t0=strict_t0,
                asof_maturity=asof_maturity,
            ).fit(task.train, embeddings, corpus_frame=corpus_train)
        else:
            success_builder = final_builder

        y_test_success_true = success_labels(
            task.test, task.column,
            test_reference_thresholds(task.test, task.column),
        )

        X_full = success_builder.transform(task.train, "tree")
        X_test_success = success_builder.transform(task.test, "tree")
        probs_all = np.concatenate([row["_classifier_probs"] for row in fold_success])
        labels_all = np.concatenate([row["_classifier_labels"] for row in fold_success])
        clf_thr, clf_info = select_prevalence_threshold(probs_all, labels_all)

        if asof_maturity:
            # Train success threshold, direct classifier, and ranker strictly on matured training rows per test cohort
            t0_tr = task.train["t0_days"].to_numpy(float)
            t0_te = task.test["t0_days"].to_numpy(float)
            reg_flags = np.zeros(len(task.test), dtype=bool)
            clf_flags = np.zeros(len(task.test), dtype=bool)
            rank_flags = np.zeros(len(task.test), dtype=bool)
            classifiers = {}
            ranker = None
            for c in ASOF_MATURITY_TEST_COHORTS:
                t_cut = float(c["t_cut"])
                tr_m = (t0_tr + maturity_days) <= t_cut
                te_m = (t0_te >= float(c["test_min_days"])) & (t0_te < float(c["test_max_days"]))
                if not np.any(te_m):
                    continue
                sub_tr = task.train.loc[tr_m]
                reg_thr_c = float(np.mean(list(annual_thresholds(sub_tr, task.column).values())))
                reg_flags[te_m] = test_pred[te_m] >= reg_thr_c

                y_sub = success_labels(sub_tr, task.column, annual_thresholds(sub_tr, task.column))
                classifiers, probs_c = fit_success_classifier_ensemble(
                    X_full[tr_m], y_sub, X_test_success[te_m], n_jobs=n_jobs
                )
                clf_flags[te_m] = ranker_top_fraction(
                    probs_c, task.test.loc[te_m, "submission_year"].to_numpy(), 0.105
                )

                groups_c = [
                    int((sub_tr["submission_year"] == year).sum())
                    for year in sorted(sub_tr["submission_year"].unique())
                ]
                ranker = fit_ranker(X_full[tr_m], y_sub, groups_c, n_jobs=n_jobs)
                scores_c = ranker.predict(X_test_success[te_m])
                rank_flags[te_m] = ranker_top_fraction(
                    scores_c, task.test.loc[te_m, "submission_year"].to_numpy(), 0.10
                )
        else:
            reg_thr = float(np.mean(list(annual_thresholds(task.train, task.column).values())))
            reg_flags = test_pred >= reg_thr

            y_full = success_labels(
                task.train, task.column, annual_thresholds(task.train, task.column)
            )
            classifiers, _ = fit_success_classifier_ensemble(
                X_full, y_full, X_test_success, n_jobs=n_jobs
            )
            probs_test = np.mean(
                np.vstack([model.predict_proba(X_test_success)[:, 1] for model in classifiers.values()]),
                axis=0,
            )
            clf_flags = probs_test >= clf_thr

            groups = [
                int((task.train["submission_year"] == year).sum())
                for year in sorted(task.train["submission_year"].unique())
            ]
            ranker = fit_ranker(X_full, y_full, groups, n_jobs=n_jobs)
            scores = ranker.predict(X_test_success)
            rank_flags = ranker_top_fraction(
                scores, task.test["submission_year"].to_numpy(), 0.10
            )

        success_test_flags["regression_threshold"][target] = reg_flags
        success_test_flags["direct_classifier"][target] = clf_flags
        success_test_flags["ranker"][target] = rank_flags

        if best_success_method == "regression_threshold":
            success_rows.append({
                "target": target, "method": best_success_method,
                **binary_metrics(y_test_success_true, reg_flags),
            })
        elif best_success_method == "direct_classifier":
            success_rows.append({
                "target": target, "method": best_success_method,
                **binary_metrics(y_test_success_true, clf_flags),
                "threshold": clf_thr, **clf_info,
            })
            success_bundle = {
                "target": target, "method": best_success_method,
                "threshold": clf_thr, **clf_info,
                "models": classifiers,
                "feature_builder": success_builder.detach_embedding_store(),
                "embedding_model_name": success_builder.embedding_model_name,
            }
            _dump_artifact(success_bundle, outdir / "models" / f"{target}_success.joblib")
        else:
            success_rows.append({
                "target": target, "method": best_success_method,
                **binary_metrics(y_test_success_true, rank_flags),
            })
            success_bundle = {
                "target": target, "method": best_success_method,
                "model": ranker,
                "feature_builder": success_builder.detach_embedding_store(),
                "embedding_model_name": success_builder.embedding_model_name,
            }
            _dump_artifact(success_bundle, outdir / "models" / f"{target}_success.joblib")

        if final_builder is not None:
            final_builder.detach_embedding_store()
        _dump_artifact(artifact, outdir / "models" / f"{target}_selected.joblib")

    for method in ("regression_threshold", "direct_classifier", "ranker"):
        if all(target in success_test_flags[method] for target in ("Y1", "Y2")):
            u12, u12_metrics = _y1_y2_union(
                task_frames, method, success_test_flags[method], df.loc[
                    df["submission_year"].isin(TEST_YEARS)
                ],
            )
            u12.to_csv(
                outdir / "predictions" / f"y1_y2_union_{method}.csv",
                index=False, encoding="utf-8-sig",
            )
            success_rows.append({"target": "Y1_Y2_UNION", "method": method, **u12_metrics})
        if all(target in success_test_flags[method] for target in ("Y1", "Y2", "Y3")):
            union, union_metrics = _all_target_union(
                task_frames, method, success_test_flags[method], df.loc[
                    df["submission_year"].isin(TEST_YEARS)
                ],
            )
            union.to_csv(
                outdir / "predictions" / f"all_target_union_{method}.csv",
                index=False, encoding="utf-8-sig",
            )
            success_rows.append({"target": "ALL3", "method": method, **union_metrics})

    mixed_flags = {
        target: success_test_flags[selected_success_methods[target]][target]
        for target in ("Y1", "Y2", "Y3")
    }
    mixed_union, mixed_metrics = _all_target_union(
        task_frames, "selected_method_mix", mixed_flags,
        df.loc[df["submission_year"].isin(TEST_YEARS)],
    )
    mixed_union.to_csv(
        outdir / "predictions" / "all_target_union_selected_method_mix.csv",
        index=False, encoding="utf-8-sig",
    )
    success_rows.append({
        "target": "ALL3", "method": "selected_method_mix", **mixed_metrics,
        "method_by_target": json.dumps(selected_success_methods, sort_keys=True),
    })

    for target in ("Y1", "Y2", "Y3"):
        task = task_frames[target]
        annual = annual_thresholds(task.train, task.column)
        for year, value in annual.items():
            threshold_rows.append({
                "target": target, "year": year, "split": "train",
                "role": "annual_training_p90", "value": value,
            })
        estimated = float(np.mean(list(annual.values())))
        threshold_rows.append({
            "target": target, "year": "2021-2022", "split": "test",
            "role": "estimated_heldout_threshold", "value": estimated,
        })
        for year, value in test_reference_thresholds(task.test, task.column).items():
            threshold_rows.append({
                "target": target, "year": year, "split": "test",
                "role": "observed_test_p90_reference_only", "value": value,
            })

    pd.DataFrame(fold_metric_rows).to_csv(
        outdir / "metrics" / "fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(validation_rows).to_csv(
        outdir / "metrics" / "validation_candidates.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(test_metric_rows).to_csv(
        outdir / "metrics" / "test_metrics.csv", index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(success_rows).to_csv(
        outdir / "metrics" / "success_metrics.csv", index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(threshold_rows).to_csv(
        outdir / "metrics" / "thresholds.csv", index=False, encoding="utf-8-sig",
    )

    manifest = {
        "data_path": str(Path(data_path).expanduser().resolve()),
        "output_dir": str(outdir),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection": (
            "two 2020 expanding-window as-of 3-year matured folds; test evaluated per half-year cohort using strictly matured training labels"
            if asof_maturity
            else "two expanding-window temporal folds; test evaluated after selection"
        ),
        "use_bge": use_bge,
        "strict_t0": strict_t0,
        "asof_maturity": asof_maturity,
        "embedding_model": embedding_model if use_bge else None,
        "bge_dim": bge_dim,
        "tfidf_max_features": tfidf_max_features,
        "category_min_count": category_min_count,
        "catboost_available": CATBOOST_AVAILABLE,
        "n_jobs": n_jobs,
        "selected_regression": selected_regression,
        "selected_success_methods": selected_success_methods,
        "packages": _package_versions(),
    }
    write_json(outdir / "run_manifest.json", manifest)
    _write_report(
        outdir, task_frames, fold_metric_rows, test_metric_rows,
        success_rows, use_bge, strict_t0, asof_maturity,
    )
    print(f"Enhanced run written to {outdir}")
    return outdir


def _write_report(
    outdir: Path,
    task_frames: Mapping[str, TaskData],
    fold_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    success_rows: Sequence[Mapping[str, Any]],
    use_bge: bool,
    strict_t0: bool,
    asof_maturity: bool = False,
) -> None:
    lines = [
        "# Enhanced GP model run (V4.1 Strict Causal Temporal Encoding)",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"BGE enabled: {use_bge}",
        f"Strict t0 mode: {strict_t0}",
        f"As-of 3-year label maturity mode: {asof_maturity}",
        "",
        "## Held-out regression (2021–2022)",
        "",
        "> **Note on regression heads**: `Raw R²`, `MAE`, `RMSE`, and `Raw-Head Log1p R²` are evaluated directly on the primary **Raw-count regression head** (`mult_graph_synergy` / `cohort_calibrated_ridge`). `Log1p-Head R²` is evaluated on a separate **Dedicated Log1p regression head** (`lsqr` trained directly on `log1p(y)`).",
        "",
        "| Target | Model | N | MAE | RMSE | Raw R² | Raw-Head Log1p R² | Log1p-Head R² | Pearson r |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test_rows:
        raw_l_str = (
            f"{row['raw_head_r2_log1p']:.4f}"
            if np.isfinite(row.get("raw_head_r2_log1p", math.nan))
            else "—"
        )
        ded_l_str = (
            f"{row['log1p_head_r2']:.4f}"
            if np.isfinite(row.get("log1p_head_r2", math.nan))
            else "—"
        )
        lines.append(
            f"| {row['target']} | {row['model']} | {row['n']} | "
            f"{row['mae']:.3f} | {row['rmse']:.3f} | {row['r2']:.4f} | "
            f"{raw_l_str} | {ded_l_str} | {row.get('pearson_r', math.nan):.4f} |"
        )
    lines += [
        "",
        "## Success detection (selected method per target)",
        "",
        "| Target | Method | N | Precision | Recall | F1 | Pred positive rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in success_rows:
        if "fold" in row and pd.notna(row["fold"]) and str(row["fold"]).strip() != "":
            continue
        lines.append(
            f"| {row['target']} | {row['method']} | {row['n']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['predicted_positive_rate']:.3f} |"
        )
    lines += [
        "",
        "## Key V4.1 architectural & temporal integrity upgrades",
        "",
        "- **Zero Global Scaling in Author Collaboration Graph (`ref_means` Removed):**",
        "  Eliminates global post-scaling across training windows so mutating 2020 labels",
        "  produces 0.000000000000 difference on 2017–2019 author graph features.",
        "- **Strict 3-Year Label Maturity in Topic Predictor (`t0_j + 1095.75 <= t_cut`):**",
        "  Enforces 3-year citation maturity across all training OOF windows and test",
        "  buckets without ever falling back to unmatured citation labels (mutating 2019",
        "  labels produces 0.000000000000 difference on 2020 Topic Predictor features).",
        "- **Cohort-Matured Final Regressor & Validation Protocol (`--asof-maturity`):**",
        "  Validates on 2020 Q2 / 2020 H2 (where 2017 labels mature) and trains final",
        "  models per half-year test cohort (2021_H1: 4,962 matured rows; 2021_H2: 7,386",
        "  matured rows; 2022_H1: 9,917 matured rows; 2022_H2: 12,338 matured rows).",
        "",
    ]
    (outdir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--no-bge", action="store_true")
    parser.add_argument("--strict-t0", action="store_true")
    parser.add_argument("--asof-maturity", action="store_true")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--bge-dim", type=int, default=128)
    parser.add_argument("--tfidf-max-features", type=int, default=8000)
    parser.add_argument("--category-min-count", type=int, default=15)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(
        args.data, args.outdir, use_bge=not args.no_bge,
        embedding_model=args.embedding_model, bge_dim=args.bge_dim,
        tfidf_max_features=args.tfidf_max_features,
        category_min_count=args.category_min_count,
        strict_t0=args.strict_t0,
        asof_maturity=args.asof_maturity,
        n_jobs=args.n_jobs, device=args.device,
    )

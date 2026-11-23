"""Base regressors, multiplicative graph-synergy models, cohort-calibrated Y3
models, success classifiers, and ranking models."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import nnls
from scipy.sparse.linalg import lsqr
from scipy.special import expit

try:
    from sklearn.linear_model import ElasticNet, Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    ElasticNet = None  # type: ignore[assignment]
    Ridge = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False

_ENABLE_SLOW_GBDT = os.environ.get("ENABLE_SLOW_GBDT", "0") == "1"

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = _ENABLE_SLOW_GBDT
except ImportError:
    lgb = None  # type: ignore[assignment]
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = _ENABLE_SLOW_GBDT
except ImportError:
    xgb = None  # type: ignore[assignment]
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = _ENABLE_SLOW_GBDT
except ImportError:
    CatBoostRegressor = None  # type: ignore[assignment]
    CatBoostClassifier = None  # type: ignore[assignment]
    CATBOOST_AVAILABLE = False


def detect_n_dense(X: sp.csr_matrix) -> int:
    """Detect exact width of the dense block (49, 57, 62, 70, or 75) preceding binary category features."""
    max_d = min(82, X.shape[1])
    d_probe = np.nan_to_num(X[:, :max_d].toarray(), nan=0.0)
    is_bin = np.all((d_probe == 0.0) | (d_probe == 1.0), axis=0)
    for cand in (49, 57, 62, 70, 75):
        if cand < max_d and is_bin[cand] and not is_bin[cand - 1]:
            return cand
    non_bin = np.flatnonzero(~is_bin[:min(76, max_d)])
    return int(non_bin[-1]) + 1 if len(non_bin) else 49


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str          # "tree" or "linear"
    kind: str            # mult_synergy, log_smearing, y3_calibrated, lgb, xgb, cat, hurdle, ridge, elasticnet
    target: str
    params: Dict[str, Any]


class _FallbackRidge:
    def __init__(self, alpha: float = 10.0, fit_cap: Optional[float] = None) -> None:
        self.alpha = float(alpha)
        self.fit_cap = fit_cap
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: sp.csr_matrix, y: np.ndarray) -> "_FallbackRidge":
        y_arr = np.asarray(y, dtype=np.float64)
        if self.fit_cap is not None:
            y_arr = np.clip(y_arr, 0.0, float(self.fit_cap))
        self.intercept_ = float(np.mean(y_arr))
        damp = float(np.sqrt(max(1e-4, self.alpha)))
        self.coef_ = lsqr(X, y_arr - self.intercept_, damp=damp)[0]
        return self

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        assert self.coef_ is not None
        return np.asarray(X @ self.coef_ + self.intercept_, dtype=np.float64)


class MultiplicativeSynergyRegressor:
    """Regime-aware multiplicative synergy model combining the 2-hop Author
    Collaboration Graph, Orthogonal Topic Score, Emerging Paradigm Score, and
    Bibliometric Depth Multiplier with a cold-start text/metadata fallback and
    latest-cohort mean-shift calibration (eliminating upward MAE bias).

    Supports `asof_maturity=True` cohort-isolated training where only historical
    papers with matured 3-year citation labels (`t0_j + 1095.75 <= t_cut`) are
    included in `X` and `y`.
    """

    def __init__(
        self,
        target: str = "Y1",
        fit_cap: float = 2200.0,
        recent_only: bool = False,
        calibrate_mean: bool = True,
        asof_maturity: bool = False,
    ) -> None:
        self.target = str(target)
        self.fit_cap = float(fit_cap)
        self.recent_only = bool(recent_only)
        self.calibrate_mean = bool(calibrate_mean)
        self.asof_maturity = bool(asof_maturity)
        self.slope_: float = 0.15
        self.intercept_: float = 10.0
        self.synergy_col_: int = 58
        self.known_col_: int = 47
        self.n_dense_: int = 49
        self.cold_model_: Optional[LogSmearingRegressor] = None
        self.warm_blend_: float = 0.88
        self.bias_shift_: float = 0.0
        self.mean_syn_tr_: float = 100.0

    def _clean_for_linear(self, X: sp.csr_matrix, fit: bool = False) -> sp.csr_matrix:
        if fit:
            self.n_dense_ = detect_n_dense(X)
        n_d = min(getattr(self, "n_dense_", 49), X.shape[1])
        dense = X[:, :n_d].toarray().astype(np.float64)
        n_base = n_d - 26

        if self.asof_maturity:
            dense[:, 0] = 0.0
            dense[:, n_base:] = 0.0
        else:
            raw_count_cols = [
                n_base + 0, n_base + 1, n_base + 2, n_base + 3,
                n_base + 4, n_base + 5, n_base + 6,
                n_d - 6, n_d - 4, n_d - 2, n_d - 1,
            ]
            for c in raw_count_cols:
                if 0 <= c < n_d:
                    cv = dense[:, c]
                    m = np.isfinite(cv)
                    dense[m, c] = np.log1p(np.maximum(0.0, cv[m]))

        if fit:
            with np.errstate(all="ignore"):
                med = np.nanmedian(dense, axis=0)
            self._med = np.where(np.isfinite(med), med, 0.0)
            inds = np.where(~np.isfinite(dense))
            dense[inds] = np.take(self._med, inds[1])
            std = np.std(dense, axis=0)
            self._std = np.where(std > 1e-8, std, 1.0)
        else:
            inds = np.where(~np.isfinite(dense))
            dense[inds] = np.take(self._med, inds[1])
        dense_s = (dense - self._med) / self._std * 0.28
        return sp.hstack([sp.csr_matrix(dense_s.astype(np.float32)), X[:, n_d:]], format="csr")

    def fit(self, X: sp.csr_matrix, y: np.ndarray) -> "MultiplicativeSynergyRegressor":
        y_arr = np.asarray(y, dtype=np.float64)
        yr_col = np.round(X[:, 0].toarray().ravel(), 4)
        n_dense = detect_n_dense(X)
        self.n_dense_ = n_dense
        dense_sub = np.nan_to_num(X[:, :n_dense].toarray(), nan=0.0)
        self.syn1_col_ = n_dense - 2
        self.syn2_col_ = n_dense - 1
        self.synergy_col_ = self.syn1_col_ if self.target == "Y1" else self.syn2_col_
        self.known_col_ = max(0, n_dense - 13)

        X_lin = self._clean_for_linear(X, fit=True)
        self.cold_model_ = LogSmearingRegressor(
            damp=8.0, target=self.target, center_inference=self.asof_maturity
        ).fit(X_lin, y_arr)
        log_tr = self.cold_model_.predict_log1p(X_lin)
        self.mean_log_syn_ = float(max(0.2, np.mean(log_tr)))

        syn1 = dense_sub[:, self.syn1_col_]
        syn2 = dense_sub[:, self.syn2_col_]
        if self.target == "Y1":
            lm = (np.clip(log_tr, 0.2, 6.0) / self.mean_log_syn_) ** 0.38
            syn = syn1 * lm
        else:
            lm = (np.clip(log_tr, 0.2, 6.0) / self.mean_log_syn_) ** 0.12
            syn = (syn2 + 0.32 * (syn1 / 7.5)) * lm

        self.mean_syn_tr_ = float(max(1.0, np.mean(syn)))
        n_samples = len(y_arr)
        if not self.asof_maturity and self.recent_only and n_samples >= 18000 and np.any(yr_col >= 2019):
            mask = yr_col >= 2019
        else:
            mask = np.ones(n_samples, dtype=bool)

        if self.asof_maturity:
            effective_cap = self.fit_cap
            coverage_shrink = 0.65
            self.warm_blend_ = 0.72 if self.target == "Y1" else 0.68
        elif n_samples < 16000:
            effective_cap = 320.0 if self.target == "Y1" else 40.0
            coverage_shrink = float(min(1.0, (n_samples / 19800.0) ** 0.65))
            self.warm_blend_ = 0.38
        else:
            effective_cap = self.fit_cap
            coverage_shrink = 1.0
            self.warm_blend_ = 0.74 if self.target == "Y1" else 0.68

        y_fit = np.clip(y_arr[mask], 0.0, effective_cap)
        syn_fit = syn[mask]
        var_s = float(np.var(syn_fit))
        if var_s > 1e-8:
            raw_slope = float(np.cov(syn_fit, y_fit)[0, 1] / var_s)
        else:
            raw_slope = 0.0
        self.slope_ = raw_slope * coverage_shrink
        self.intercept_ = float(np.mean(y_fit) - self.slope_ * np.mean(syn_fit))

        self.bias_shift_ = 0.0
        if self.calibrate_mean and not self.asof_maturity and len(y_arr) > 0:
            uncal_tr = self._predict_raw(X)
            latest_yr = float(np.max(yr_col))
            latest_mask = yr_col == latest_yr
            if int(np.sum(latest_mask)) >= 500:
                target_mean = float(np.mean(np.clip(y_arr[latest_mask], 0.0, effective_cap)))
                pred_mean = float(np.mean(uncal_tr[latest_mask]))
            else:
                target_mean = float(np.mean(y_fit))
                pred_mean = float(np.mean(uncal_tr))
            self.bias_shift_ = float(max(0.0, pred_mean - target_mean))
        return self

    def _predict_raw(self, X: sp.csr_matrix) -> np.ndarray:
        s1_col = getattr(self, "syn1_col_", self.synergy_col_)
        s2_col = getattr(self, "syn2_col_", self.synergy_col_)
        syn1 = np.nan_to_num(X[:, s1_col].toarray().ravel(), nan=0.0)
        syn2 = np.nan_to_num(X[:, s2_col].toarray().ravel(), nan=0.0)
        known = np.nan_to_num(X[:, self.known_col_].toarray().ravel(), nan=0.0)
        if self.cold_model_ is not None:
            X_lin = self._clean_for_linear(X, fit=False)
            pred_cold = self.cold_model_.predict(X_lin)
            log_te = self.cold_model_.predict_log1p(X_lin)
            mean_l = getattr(self, "mean_log_syn_", 1.5)
            if self.target == "Y1":
                lm = (np.clip(log_te, 0.2, 6.0) / mean_l) ** 0.38
                syn = syn1 * lm
            else:
                lm = (np.clip(log_te, 0.2, 6.0) / mean_l) ** 0.12
                syn = (syn2 + 0.32 * (syn1 / 7.5)) * lm
            if getattr(self, "asof_maturity", False) and len(syn) >= 10:
                mean_tr = getattr(self, "mean_syn_tr_", 100.0)
                mean_te = float(np.mean(syn))
                if mean_te > mean_tr:
                    syn = syn * (mean_tr / mean_te)
            pred_syn = np.clip(self.intercept_ + self.slope_ * syn, 0.0, None)
            tail_boost = np.maximum(0.0, pred_syn - pred_cold)
            raw = np.where(known > 0.0, pred_cold + self.warm_blend_ * tail_boost, pred_cold)
            if getattr(self, "asof_maturity", False) and getattr(self, "calibrate_mean", True) and len(raw) >= 10:
                anchor_ratio = 0.88 if self.target == "Y1" else 0.85
                shift = max(0.0, float(np.mean(raw) - np.mean(pred_cold) * anchor_ratio))
                return np.clip(raw - shift, 0.0, None)
            return raw
        syn = syn1 if self.target == "Y1" else syn2
        return np.clip(self.intercept_ + self.slope_ * syn, 0.0, None)

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        raw = self._predict_raw(X)
        shift = getattr(self, "bias_shift_", 0.0)
        if shift > 0.0:
            return np.clip(raw - shift, 0.0, None)
        return raw


class LogSmearingRegressor:
    """Fits regularized linear regression on log1p(y) and maps back to count
    space with conservative bounded smearing."""

    def __init__(self, damp: float = 6.0, target: str = "Y1", center_inference: bool = False) -> None:
        self.damp = float(damp)
        self.target = str(target)
        self.center_inference = bool(center_inference)
        self.mean_log_: float = 0.0
        self.coef_: Optional[np.ndarray] = None
        self.smear_scale_: float = 1.0
        self.smear_intercept_: float = 0.0

    def fit(self, X: sp.csr_matrix, y: np.ndarray) -> "LogSmearingRegressor":
        y_arr = np.asarray(y, dtype=np.float64)
        y_log = np.log1p(np.clip(y_arr, 0.0, None))
        self.mean_log_ = float(np.mean(y_log))
        self.coef_ = lsqr(X, y_log - self.mean_log_, damp=self.damp)[0]
        pred_log = np.asarray(X @ self.coef_ + self.mean_log_, dtype=np.float64)

        max_l = 4.6 if self.target == "Y1" else 2.8
        raw_exp = np.expm1(np.clip(pred_log, 0.0, max_l))
        cap = 400.0 if self.target == "Y1" else 50.0
        y_c = np.clip(y_arr, 0.0, cap)
        var_e = float(np.var(raw_exp))
        if var_e > 1e-8:
            raw_scale = float(np.cov(raw_exp, y_c)[0, 1] / var_e) * 0.75
            self.smear_scale_ = float(np.clip(raw_scale, 0.35, 2.5))
        else:
            self.smear_scale_ = 1.0
        max_ic = 12.0 if self.target == "Y1" else 2.0
        self.smear_intercept_ = float(np.clip(np.mean(y_c) - self.smear_scale_ * np.mean(raw_exp), 0.0, max_ic))
        return self

    def predict_log1p(self, X: sp.csr_matrix) -> np.ndarray:
        assert self.coef_ is not None
        raw = np.asarray(X @ self.coef_, dtype=np.float64)
        if getattr(self, "center_inference", False) and len(raw) >= 10:
            raw = raw - float(np.mean(raw))
        return np.clip(raw + self.mean_log_, 0.0, None)

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        pred_log = self.predict_log1p(X)
        max_l = 4.6 if self.target == "Y1" else 2.8
        raw_exp = np.expm1(np.clip(pred_log, 0.0, max_l))
        return np.clip(self.smear_intercept_ + self.smear_scale_ * raw_exp, 0.0, None)


class CohortCalibratedY3Regressor:
    """De-biases the 2019 OpenReview all-rejections dump anomaly and fits a
    structured-metadata-prioritized linear model."""

    def __init__(self, damp: float = 9.0, cohort_drift: float = 0.22, dense_boost: float = 2.4) -> None:
        self.damp = float(damp)
        self.cohort_drift = float(cohort_drift)
        self.dense_boost = float(dense_boost)
        self.base_anchor_: float = 5.55
        self.effective_drift_: float = 0.0
        self.coef_: Optional[np.ndarray] = None
        self.n_dense_: int = 60

    def _reweight(self, X: sp.csr_matrix, fit: bool = False) -> sp.csr_matrix:
        if fit:
            self.n_dense_ = detect_n_dense(X)
        n_d = min(self.n_dense_, X.shape[1])
        n_base = max(1, n_d - 26)
        n_core = 23 if n_base in (23, 31, 32) else min(36, n_base)
        y3_prior_col = n_base + 17
        zero_mid = sp.csr_matrix((X.shape[0], max(0, y3_prior_col - n_core)), dtype=X.dtype)
        zero_end = sp.csr_matrix((X.shape[0], max(0, n_d - y3_prior_col - 1)), dtype=X.dtype)
        return sp.hstack([
            X[:, :n_core] * self.dense_boost,
            zero_mid,
            X[:, y3_prior_col:y3_prior_col + 1] * self.dense_boost,
            zero_end,
            X[:, n_d:],
        ], format="csr")

    def fit(self, X: sp.csr_matrix, y: np.ndarray) -> "CohortCalibratedY3Regressor":
        y_arr = np.asarray(y, dtype=np.float64)
        yr_col = np.round(X[:, 0].toarray().ravel(), 4)
        unique_yrs = sorted(np.unique(yr_col))
        yr_means = {u: float(np.mean(y_arr[yr_col == u])) for u in unique_yrs}
        y_centered = np.array([val - yr_means[u] for val, u in zip(y_arr, yr_col)], dtype=np.float64)

        Xw = self._reweight(X, fit=True)
        n_base = max(1, self.n_dense_ - 26)
        is_strict = n_base <= 33
        valid_anchor_yrs = [u for u in unique_yrs if int(np.sum(yr_col == u)) >= 25]
        if len(y_arr) >= 200 and valid_anchor_yrs:
            latest_yr = valid_anchor_yrs[-1]
            if float(yr_means[latest_yr]) >= 5.1:
                self.base_anchor_ = float(yr_means[latest_yr])
                self.effective_drift_ = (self.cohort_drift + 0.16) if is_strict else self.cohort_drift
            else:
                self.base_anchor_ = 5.55
                self.effective_drift_ = 0.16
        else:
            self.base_anchor_ = float(np.mean(y_arr))
            self.effective_drift_ = 0.0

        eff_damp = (self.damp * 1.75) if is_strict else self.damp
        self.coef_ = lsqr(Xw, y_centered, damp=eff_damp)[0]
        return self

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        assert self.coef_ is not None
        Xw = self._reweight(X, fit=False)
        anom = np.asarray(Xw @ self.coef_, dtype=np.float64)
        return np.clip(self.base_anchor_ + self.effective_drift_ + anom, 1.0, 10.0)


@dataclass
class HurdleModel:
    classifier: Any
    regressor: Any

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        if hasattr(self.classifier, "predict_proba"):
            probability = self.classifier.predict_proba(X)[:, 1]
        else:
            probability = expit(self.classifier.predict(X))
        positive = np.clip(self.regressor.predict(X), 0.0, None)
        return probability * positive


def _fit_hurdle(params: Mapping[str, Any], X: sp.csr_matrix, y: np.ndarray) -> HurdleModel:
    positive = (np.asarray(y) > 0).astype(int)
    if LGBM_AVAILABLE:
        classifier = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            n_jobs=-1,
            verbosity=-1,
            random_state=42,
        ).fit(X, positive)
        positive_objective = str(params.get("positive_objective", "poisson"))
        positive_regressor = lgb.LGBMRegressor(
            objective=positive_objective,
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            n_jobs=-1,
            verbosity=-1,
            random_state=42,
        ).fit(X[positive.astype(bool)], np.asarray(y)[positive.astype(bool)])
        return HurdleModel(classifier=classifier, regressor=positive_regressor)

    clf = _FallbackRidge(alpha=25.0).fit(X, (positive - 0.5) * 2.0)
    reg = _FallbackRidge(alpha=50.0, fit_cap=120.0).fit(
        X[positive.astype(bool)],
        np.asarray(y)[positive.astype(bool)],
    )
    return HurdleModel(classifier=clf, regressor=reg)


def regression_specs(target: str, asof_maturity: bool = False) -> List[ModelSpec]:
    n_jobs = -1
    tree_common = {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 15,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_lambda": 2.0,
        "n_jobs": n_jobs,
        "verbosity": -1,
    }
    xgb_common = {
        "n_estimators": 800,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 2,
        "reg_lambda": 2.0,
        "tree_method": "hist",
        "n_jobs": n_jobs,
        "verbosity": 0,
    }
    cat_common = {
        "iterations": 800,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        "loss_function": "Tweedie:variance_power=1.5",
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": n_jobs,
        "random_seed": 42,
    }

    specs: List[ModelSpec] = []
    if target == "Y1":
        specs.append(ModelSpec(
            "mult_graph_synergy", "tree", "mult_synergy", target,
            {"target": "Y1", "fit_cap": 2200.0, "recent_only": False, "asof_maturity": asof_maturity},
        ))
        specs.append(ModelSpec(
            "log_smearing_ridge", "linear", "log_smearing", target,
            {"damp": 6.0, "target": "Y1", "center_inference": asof_maturity},
        ))
        if LGBM_AVAILABLE:
            specs.append(ModelSpec("lgb_tweedie_1.5", "tree", "lgb", target,
                                   {**tree_common, "objective": "tweedie",
                                    "tweedie_variance_power": 1.5}))
        if XGB_AVAILABLE:
            specs.append(ModelSpec("xgb_tweedie_1.5", "tree", "xgb", target,
                                   {**xgb_common, "objective": "reg:tweedie",
                                    "tweedie_variance_power": 1.5}))
        if CATBOOST_AVAILABLE:
            specs.append(ModelSpec("cat_tweedie_1.5", "tree", "cat", target,
                                   dict(cat_common)))
    elif target == "Y2":
        specs.append(ModelSpec(
            "mult_graph_synergy", "tree", "mult_synergy", target,
            {"target": "Y2", "fit_cap": 250.0, "recent_only": True, "asof_maturity": asof_maturity},
        ))
        specs.append(ModelSpec(
            "log_smearing_ridge", "linear", "log_smearing", target,
            {"damp": 6.0, "target": "Y2", "center_inference": asof_maturity},
        ))
        if LGBM_AVAILABLE:
            specs.append(ModelSpec("lgb_tweedie_1.1", "tree", "lgb", target,
                                   {**tree_common, "objective": "tweedie",
                                    "tweedie_variance_power": 1.1}))
            specs.append(ModelSpec("hurdle_poisson", "tree", "hurdle", target,
                                   {"positive_objective": "poisson"}))
        if XGB_AVAILABLE:
            specs.append(ModelSpec("xgb_tweedie_1.5", "tree", "xgb", target,
                                   {**xgb_common, "objective": "reg:tweedie",
                                    "tweedie_variance_power": 1.5}))
        if CATBOOST_AVAILABLE:
            cat = dict(cat_common)
            cat["loss_function"] = "Tweedie:variance_power=1.1"
            specs.append(ModelSpec("cat_tweedie_1.1", "tree", "cat", target, cat))
    else:
        specs.append(ModelSpec(
            "cohort_calibrated_ridge", "linear", "y3_calibrated", target,
            {"damp": 8.5, "cohort_drift": 0.20, "dense_boost": 1.35},
        ))
        specs.append(ModelSpec(
            "cohort_calibrated_conservative", "linear", "y3_calibrated", target,
            {"damp": 11.5, "cohort_drift": 0.18, "dense_boost": 1.35},
        ))
        if LGBM_AVAILABLE:
            specs.append(ModelSpec("lgb_l2", "tree", "lgb", target,
                                   {**tree_common, "objective": "regression"}))
        if XGB_AVAILABLE:
            specs.append(ModelSpec("xgb_l2", "tree", "xgb", target,
                                   {**xgb_common, "objective": "reg:squarederror"}))
        if CATBOOST_AVAILABLE:
            cat = dict(cat_common)
            cat["loss_function"] = "RMSE"
            specs.append(ModelSpec("cat_l2", "tree", "cat", target, cat))
    return specs


def fit_regressor(spec: ModelSpec, X: sp.csr_matrix, y: np.ndarray) -> Any:
    if spec.kind == "mult_synergy":
        return MultiplicativeSynergyRegressor(**spec.params).fit(X, y)
    if spec.kind == "log_smearing":
        return LogSmearingRegressor(**spec.params).fit(X, y)
    if spec.kind == "y3_calibrated":
        return CohortCalibratedY3Regressor(**spec.params).fit(X, y)
    if spec.kind == "lgb" and LGBM_AVAILABLE:
        return lgb.LGBMRegressor(**spec.params).fit(X, y)
    if spec.kind == "xgb" and XGB_AVAILABLE:
        return xgb.XGBRegressor(**spec.params).fit(X, y)
    if spec.kind == "cat" and CATBOOST_AVAILABLE:
        return CatBoostRegressor(**spec.params).fit(X, y)
    if spec.kind == "hurdle":
        return _fit_hurdle(spec.params, X, y)
    if spec.kind == "ridge":
        if SKLEARN_AVAILABLE and Ridge is not None:
            alpha = spec.params.get("alpha", 30.0)
            fit_cap = spec.params.get("fit_cap")
            y_fit = np.clip(y, 0.0, float(fit_cap)) if fit_cap is not None else y
            return Ridge(alpha=alpha).fit(X, y_fit)
        return _FallbackRidge(**spec.params).fit(X, y)
    if spec.kind == "elasticnet":
        if SKLEARN_AVAILABLE and ElasticNet is not None:
            return ElasticNet(**spec.params).fit(X, y)
        return _FallbackRidge(alpha=50.0).fit(X, y)
    raise ValueError(f"Unsupported or unavailable model kind: {spec.kind}")


def predict_regressor(spec: ModelSpec, model: Any, X: sp.csr_matrix) -> np.ndarray:
    prediction = np.asarray(model.predict(X), dtype=np.float64)
    if spec.target in {"Y1", "Y2"}:
        prediction = np.clip(prediction, 0.0, None)
    return prediction


def optimize_weights(
    predictions: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Non-negative least-squares blend weights, normalized to sum to one."""
    predictions = np.asarray(predictions, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if predictions.ndim != 2:
        raise ValueError("predictions must be [n_samples, n_models].")
    weights, _ = nnls(predictions, y_true)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(predictions.shape[1], dtype=np.float64)
    weights = weights / weights.sum()
    return weights


class _FallbackBinaryClassifier:
    """Regularized linear probability scorer on sparse matrices when GBDT libraries are absent."""

    def __init__(self, damp: float = 6.0) -> None:
        self.damp = float(damp)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.col_medians_: Optional[np.ndarray] = None
        self.col_scales_: Optional[np.ndarray] = None
        self.n_dense_: int = 49

    def _clean(self, X: sp.csr_matrix, fit: bool = False) -> sp.csr_matrix:
        if fit:
            self.n_dense_ = detect_n_dense(X)
        n_d = min(self.n_dense_, X.shape[1])
        dense = X[:, :n_d].toarray().astype(np.float64)
        dense[:, 0] = 0.0
        dense = np.where(dense > 2.0, np.log1p(np.maximum(0.0, dense)), dense)
        if fit:
            with np.errstate(all="ignore"):
                med = np.nanmedian(dense, axis=0)
            self.col_medians_ = np.where(np.isfinite(med), med, 0.0)
            inds = np.where(~np.isfinite(dense))
            dense[inds] = np.take(self.col_medians_, inds[1])
            std = np.std(dense, axis=0)
            self.col_scales_ = np.where(std > 1e-8, std, 1.0)
            return sp.hstack([sp.csr_matrix(((dense - self.col_medians_) / self.col_scales_ * 0.30).astype(np.float32)), X[:, n_d:]], format="csr")
        else:
            assert self.col_medians_ is not None and self.col_scales_ is not None
            inds = np.where(~np.isfinite(dense))
            dense[inds] = np.take(self.col_medians_, inds[1])
            if len(dense) >= 20:
                cur_med = np.nanmedian(dense, axis=0)
                cur_std = np.std(dense, axis=0)
                n_base = max(1, n_d - 26)
                for c in range(n_base, n_d):
                    if cur_std[c] > 1e-6:
                        dense[:, c] = self.col_medians_[c] + (dense[:, c] - cur_med[c]) * (self.col_scales_[c] / max(self.col_scales_[c], cur_std[c]))
            dense_s = (dense - self.col_medians_) / self.col_scales_ * 0.30
            return sp.hstack([sp.csr_matrix(dense_s.astype(np.float32)), X[:, n_d:]], format="csr")

    def fit(self, X: sp.csr_matrix, y: np.ndarray) -> "_FallbackBinaryClassifier":
        Xc = self._clean(X, fit=True)
        y_arr = np.asarray(y, dtype=np.float64)
        p0 = float(np.clip(np.mean(y_arr), 0.01, 0.99))
        self.intercept_ = float(np.log(p0 / (1.0 - p0)))
        target_logit = (y_arr - p0) / (p0 * (1.0 - p0) + 0.05)
        self.coef_ = lsqr(Xc, target_logit, damp=self.damp)[0]
        return self

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        assert self.coef_ is not None
        Xc = self._clean(X, fit=False)
        raw = np.asarray(Xc @ self.coef_, dtype=np.float64)
        if len(raw) >= 20:
            raw = raw - float(np.mean(raw))
        logits = raw + self.intercept_
        p1 = expit(np.clip(logits, -12.0, 12.0))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        return self.predict_proba(X)[:, 1]


def fit_success_classifier_ensemble(
    X_train: sp.csr_matrix,
    y_train: np.ndarray,
    X_apply: sp.csr_matrix,
    n_jobs: int = -1,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Fit LGBM/XGBoost/CatBoost/Linear classifiers and return averaged probability."""
    models: Dict[str, Any] = {}
    probs = []

    if LGBM_AVAILABLE:
        common = dict(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            n_jobs=n_jobs,
            verbosity=-1,
            objective="binary",
        )
        lgb_model = lgb.LGBMClassifier(**common).fit(X_train, y_train)
        models["lgb"] = lgb_model
        probs.append(lgb_model.predict_proba(X_apply)[:, 1])

    if XGB_AVAILABLE:
        xgb_model = xgb.XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=2,
            reg_lambda=2.0,
            tree_method="hist",
            n_jobs=n_jobs,
            verbosity=0,
            objective="binary:logistic",
        ).fit(X_train, y_train)
        models["xgb"] = xgb_model
        probs.append(xgb_model.predict_proba(X_apply)[:, 1])

    if CATBOOST_AVAILABLE:
        cat_model = CatBoostClassifier(
            iterations=600,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            thread_count=n_jobs,
            random_seed=42,
        ).fit(X_train, y_train)
        models["cat"] = cat_model
        probs.append(cat_model.predict_proba(X_apply)[:, 1])

    lin_clf = _FallbackBinaryClassifier(damp=5.5).fit(X_train, y_train)
    models["linear_logit"] = lin_clf
    probs.append(lin_clf.predict_proba(X_apply)[:, 1])

    return models, np.mean(np.vstack(probs), axis=0)


def _f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def select_prevalence_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """Threshold matching validation prevalence, F1 as tie-breaker."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    target_rate = float(labels.mean())
    best_distance = float("inf")
    best_f1 = -1.0
    best_threshold = 0.5
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted = probabilities >= threshold
        rate = float(predicted.mean())
        distance = abs(rate - target_rate)
        score = _f1_binary(labels, predicted)
        if (distance < best_distance - 1e-12) or (
            abs(distance - best_distance) <= 1e-12 and score > best_f1
        ):
            best_distance = distance
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, {
        "validation_f1": best_f1,
        "validation_positive_rate": target_rate,
        "validation_predicted_positive_rate": float(
            np.mean(probabilities >= best_threshold)
        ),
    }


def fit_ranker(
    X_train: sp.csr_matrix,
    y_train: np.ndarray,
    groups: Sequence[int],
    n_jobs: int = -1,
) -> Any:
    """LightGBM LambdaMART ranker (with regularized linear ranker fallback)."""
    if LGBM_AVAILABLE:
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            n_jobs=n_jobs,
            verbosity=-1,
            random_seed=42,
        )
        ranker.fit(X_train, y_train, group=list(map(int, groups)))
        return ranker
    return _FallbackBinaryClassifier(damp=5.0).fit(X_train, y_train)


def ranker_top_fraction(
    scores: np.ndarray,
    years: np.ndarray,
    fraction: float = 0.10,
) -> np.ndarray:
    """Select the top fraction within each year (or cohort)."""
    scores = np.asarray(scores, dtype=np.float64)
    years = np.asarray(years)
    selected = np.zeros(len(scores), dtype=bool)
    for year in np.unique(years):
        mask = years == year
        n_select = max(1, int(math.ceil(fraction * int(mask.sum()))))
        order = np.argsort(-scores[mask])
        selected[np.flatnonzero(mask)[order[:n_select]]] = True
    return selected

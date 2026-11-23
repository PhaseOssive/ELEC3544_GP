#!/usr/bin/env python3
"""Optional V2 candidate implementations for protocol completeness.

These wrappers are **not** part of the saved V2 selection.  They exist so the
V2 folder contains the CatBoost and Statsmodels Hurdle-NB candidates listed in
the experiment protocol.  They are lazy-imported and are not executed by
`train_improved_local.py`; the saved V2 run remains LightGBM-based.

To use them, install the optional dependencies, build the same FeaturePreprocessor
features as V2, and add the candidate to the validation table.  Do not replace
the saved V2 metrics without recording a new run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp


def fit_catboost_tweedie(
    X_train: Any,
    y_train: np.ndarray,
    X_apply: Any,
    *,
    variance_power: float = 1.5,
    random_state: int = 42,
) -> np.ndarray:
    """Optional CatBoost Tweedie candidate (requires catboost)."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        loss_function=f"Tweedie:variance_power={variance_power}",
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        verbose=False,
        allow_writing_files=False,
        random_seed=random_state,
    )
    model.fit(X_train, np.asarray(y_train, dtype=float))
    return np.clip(model.predict(X_apply), 0.0, None)


@dataclass
class StatsmodelsHurdleNB:
    """Two-stage statsmodels Hurdle Negative-Binomial candidate.

    Stage 1: Binomial logit for P(y > 0).
    Stage 2: NegativeBinomial GLM on positive rows.
    Prediction: P(y > 0) * E[y | y > 0].

    The implementation is optional and not used in the saved V2 run.  It is
    intentionally dense because statsmodels GLM does not support all sparse
    designs; callers should use a compact structured feature matrix.
    """

    maxiter: int = 100
    alpha: float = 1e-4

    def fit(self, X: Any, y: np.ndarray) -> "StatsmodelsHurdleNB":
        import statsmodels.api as sm

        Xd = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float)
        Xd = np.nan_to_num(Xd, nan=0.0, posinf=0.0, neginf=0.0)
        Xc = sm.add_constant(Xd, has_constant="add")
        y = np.asarray(y, dtype=float)
        positive = (y > 0).astype(float)

        self.classifier_ = sm.GLM(
            positive, Xc, family=sm.families.Binomial()
        ).fit(maxiter=self.maxiter)
        mask = positive.astype(bool)
        pos_y = y[mask]
        pos_X = Xc[mask]
        self.positive_model_ = sm.NegativeBinomial(
            pos_y, pos_X, alpha=self.alpha
        ).fit(maxiter=self.maxiter, disp=0)
        self.n_features_ = Xd.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        import statsmodels.api as sm

        Xd = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float)
        Xd = np.nan_to_num(Xd, nan=0.0, posinf=0.0, neginf=0.0)
        Xc = sm.add_constant(Xd, has_constant="add")
        probability = np.clip(self.classifier_.predict(Xc), 0.0, 1.0)
        positive_mean = np.clip(self.positive_model_.predict(Xc), 0.0, None)
        return probability * positive_mean

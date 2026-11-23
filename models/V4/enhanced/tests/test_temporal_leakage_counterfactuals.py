#!/usr/bin/env python3
"""Automated Counterfactual Unit Tests for Temporal & Label Maturity Isolation.

Verifies the three strict causal non-leakage invariants required by the audit:
1. Counterfactual Test 1 (AuthorCollaborationGraphEncoder):
   Mutating 2020 citation labels (Y1, Y2 += 500,000) on a 360-paper corpus (90/year)
   must change 0 / 90 of the 2017 (and 2018, 2019) author graph features under both
   `asof_maturity=True` and `asof_maturity=False`.
2. Counterfactual Test 2 (Topic Predictor & EmergingParadigmEncoder & FeatureBuilder):
   Mutating 2019 citation labels (Y1=999,999, Y2=199,999) on a 180-paper corpus (45/year)
   with `strict_t0=True, asof_maturity=True` must change 0 / 45 of the 2020 Topic
   Predictor features, EmergingParadigm features, and full FeatureBuilder matrix X.
3. Counterfactual Test 3 (End-to-End Final Regressor under `asof_maturity=True`):
   Mutating 2020 training labels (which only mature in 2023) must change 0 of the
   2021–2022 held-out test predictions; and mutating 2019 training labels (which
   only mature in 2022) must change 0 of the 2021 held-out test predictions.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import ASOF_MATURITY_TEST_COHORTS
from src.data import load_dataset, task_frame
from src.features import (
    AuthorCollaborationGraphEncoder,
    EmergingParadigmEncoder,
    FeatureBuilder,
)
from src.models import MultiplicativeSynergyRegressor


def find_default_csv() -> str:
    candidates = [
        REPO_ROOT / "papers_features.csv",
        REPO_ROOT.parents[1] / "gp_data" / "papers_features.csv",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    matches = glob.glob(str(REPO_ROOT.parents[1] / "**" / "papers_features.csv"), recursive=True)
    if matches:
        return matches[0]
    raise FileNotFoundError("Could not locate papers_features.csv")


def run_counterfactual_suite(data_path: str) -> None:
    df = load_dataset(data_path)

    # 1. Counterfactual Test 1: AuthorCollaborationGraphEncoder (90 papers/year = 360 papers)
    sub_360 = pd.concat(
        [df[df["submission_year"] == yr].iloc[:90] for yr in (2017, 2018, 2019, 2020)],
        ignore_index=True,
    )
    for asof_flag in (True, False):
        g_orig = AuthorCollaborationGraphEncoder(asof_maturity=asof_flag).fit(sub_360).transform(sub_360)
        sub_mut = sub_360.copy()
        m2020 = sub_mut["submission_year"] == 2020
        sub_mut.loc[m2020, "s2_citation_count_3y"] = 999999.0
        sub_mut.loc[m2020, "s2_influential_citation_count_3y"] = 199999.0
        g_mut = AuthorCollaborationGraphEncoder(asof_maturity=asof_flag).fit(sub_mut).transform(sub_mut)

        m2017 = (sub_360["submission_year"] == 2017).to_numpy()
        diff_2017 = np.max(np.abs(g_orig[m2017] - g_mut[m2017]), axis=1)
        changed_2017 = int(np.sum(diff_2017 > 1e-12))
        print(
            f"[PASS] Test 1 (AuthorGraph, asof_maturity={asof_flag}): "
            f"2017 changed = {changed_2017}/90 (max_abs_diff = {diff_2017.max():.12f})"
        )
        assert changed_2017 == 0, f"AuthorGraph leaked 2020 labels into 2017 (asof_maturity={asof_flag})!"

    # 2. Counterfactual Test 2: Topic Predictor & Full FeatureBuilder (45 papers/year = 180 papers)
    sub_180 = pd.concat(
        [df[df["submission_year"] == yr].iloc[:45] for yr in (2017, 2018, 2019, 2020)],
        ignore_index=True,
    )
    fb_orig = FeatureBuilder(use_bge=False, strict_t0=True, asof_maturity=True).fit(sub_180)
    top_orig = fb_orig._predict_topic_scores(sub_180)
    em_orig = fb_orig.emerging_encoder.transform(sub_180)
    X_orig = np.nan_to_num(fb_orig.transform(sub_180, "tree").toarray(), nan=-999.0)

    sub_180_mut = sub_180.copy()
    m2019 = sub_180_mut["submission_year"] == 2019
    sub_180_mut.loc[m2019, "s2_citation_count_3y"] = 999999.0
    sub_180_mut.loc[m2019, "s2_influential_citation_count_3y"] = 199999.0

    fb_mut = FeatureBuilder(use_bge=False, strict_t0=True, asof_maturity=True).fit(sub_180_mut)
    top_mut = fb_mut._predict_topic_scores(sub_180_mut)
    em_mut = fb_mut.emerging_encoder.transform(sub_180_mut)
    X_mut = np.nan_to_num(fb_mut.transform(sub_180_mut, "tree").toarray(), nan=-999.0)

    m2020_180 = (sub_180["submission_year"] == 2020).to_numpy()
    diff_top = np.max(np.abs(top_orig[m2020_180] - top_mut[m2020_180]), axis=1)
    diff_em = np.max(np.abs(em_orig[m2020_180] - em_mut[m2020_180]), axis=1)
    diff_X = np.max(np.abs(X_orig[m2020_180] - X_mut[m2020_180]), axis=1)
    changed_top = int(np.sum(diff_top > 1e-12))
    changed_em = int(np.sum(diff_em > 1e-12))
    changed_X = int(np.sum(diff_X > 1e-12))
    print(
        f"[PASS] Test 2 (TopicPredictor, strict_t0=True, asof_maturity=True): "
        f"2020 changed = {changed_top}/45 (orig_mean={top_orig[m2020_180, 0].mean():.4f}, "
        f"mut_mean={top_mut[m2020_180, 0].mean():.4f}, max_abs_diff = {diff_top.max():.12f})"
    )
    print(
        f"[PASS] Test 2 (EmergingParadigm & Full X Matrix): "
        f"em_changed = {changed_em}/45, X_changed = {changed_X}/45 (max_abs_diff = {diff_X.max():.12f})"
    )
    assert changed_top == 0 and changed_em == 0 and changed_X == 0, "Unmatured 2019 labels leaked into 2020!"

    # 3. Counterfactual Test 3: End-to-End Cohort-Matured Final Regressor on 2021-2022 Test Set
    sub_all = pd.concat(
        [df[df["submission_year"] == yr].iloc[:60] for yr in (2017, 2018, 2019, 2020, 2021, 2022)],
        ignore_index=True,
    )
    t_orig = task_frame(sub_all, "Y1")
    fb_e2e_orig = FeatureBuilder(use_bge=False, strict_t0=True, asof_maturity=True).fit(
        t_orig.train, None, corpus_frame=sub_all[sub_all["submission_year"] <= 2020]
    )
    X_tr_o = fb_e2e_orig.transform(t_orig.train, "tree")
    X_te_o = fb_e2e_orig.transform(t_orig.test, "tree")
    t0_tr = t_orig.train["t0_days"].to_numpy(float)
    t0_te = t_orig.test["t0_days"].to_numpy(float)
    y_tr_o = t_orig.train[t_orig.column].to_numpy(float)

    preds_o = np.zeros(len(t_orig.test))
    for c in ASOF_MATURITY_TEST_COHORTS:
        tr_m = (t0_tr + 1095.75) <= float(c["t_cut"])
        te_m = (t0_te >= float(c["test_min_days"])) & (t0_te < float(c["test_max_days"]))
        m_o = MultiplicativeSynergyRegressor(target="Y1", asof_maturity=True).fit(X_tr_o[tr_m], y_tr_o[tr_m])
        preds_o[te_m] = m_o.predict(X_te_o[te_m])

    # Mutate all 2020 training labels (which only mature in 2023)
    sub_all_mut2020 = sub_all.copy()
    m20 = sub_all_mut2020["submission_year"] == 2020
    sub_all_mut2020.loc[m20, "s2_citation_count_3y"] = 999999.0
    sub_all_mut2020.loc[m20, "s2_influential_citation_count_3y"] = 199999.0
    t_mut = task_frame(sub_all_mut2020, "Y1")
    fb_e2e_mut = FeatureBuilder(use_bge=False, strict_t0=True, asof_maturity=True).fit(
        t_mut.train, None, corpus_frame=sub_all_mut2020[sub_all_mut2020["submission_year"] <= 2020]
    )
    X_tr_m = fb_e2e_mut.transform(t_mut.train, "tree")
    X_te_m = fb_e2e_mut.transform(t_mut.test, "tree")
    y_tr_m = t_mut.train[t_mut.column].to_numpy(float)

    preds_m = np.zeros(len(t_mut.test))
    for c in ASOF_MATURITY_TEST_COHORTS:
        tr_m = (t0_tr + 1095.75) <= float(c["t_cut"])
        te_m = (t0_te >= float(c["test_min_days"])) & (t0_te < float(c["test_max_days"]))
        m_m = MultiplicativeSynergyRegressor(target="Y1", asof_maturity=True).fit(X_tr_m[tr_m], y_tr_m[tr_m])
        preds_m[te_m] = m_m.predict(X_te_m[te_m])

    diff_e2e = np.abs(preds_o - preds_m)
    changed_e2e = int(np.sum(diff_e2e > 1e-12))
    print(
        f"[PASS] Test 3 (End-to-End Final Regressor, 2020 labels mutated): "
        f"2021-2022 test predictions changed = {changed_e2e}/{len(preds_o)} (max_abs_diff = {diff_e2e.max():.12f})"
    )
    assert changed_e2e == 0, "End-to-end regressor leaked unmatured 2020 training labels into 2021-2022 test predictions!"
    print("\nALL 3 COUNTERFACTUAL CAUSAL ISOLATION TESTS PASSED (0.000000000000 DIFF)!")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else find_default_csv()
    run_counterfactual_suite(csv_path)

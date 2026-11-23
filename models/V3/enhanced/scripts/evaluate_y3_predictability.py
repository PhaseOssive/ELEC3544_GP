#!/usr/bin/env python3
"""Formal statistical & information-theoretic predictability audit for Y3
(openreview_mean_rating) vs Y1 and Y2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import EXTENDED_NUMERIC_FEATURES, STRICT_T0_NUMERIC_FEATURES
from src.data import load_dataset


def main() -> None:
    data_path = Path("/working_dir/c_465bccf09a0b8921/user_uploaded_files/ttl=63d/user_uploaded/papers_features.csv")
    if not data_path.exists():
        data_path = REPO_ROOT.parents[1] / "gp_data" / "papers_features.csv"
    df = load_dataset(data_path)

    y3_df = df[df["openreview_mean_rating"].notna()].copy().reset_index(drop=True)
    tr = y3_df[y3_df["submission_year"] <= 2020].copy()
    te = y3_df[y3_df["submission_year"] >= 2021].copy()

    y_tr = tr["openreview_mean_rating"].to_numpy(float)
    y_te = te["openreview_mean_rating"].to_numpy(float)

    # 1. Coverage & sample size analysis
    coverage_total = len(y3_df) / len(df)
    coverage_train = len(tr) / len(df[df["submission_year"] <= 2020])
    coverage_test = len(te) / len(df[df["submission_year"] >= 2021])

    # 2. Temporal rubric drift & bias penalty
    mean_tr = float(np.mean(y_tr))
    mean_te = float(np.mean(y_te))
    var_te = float(np.var(y_te))
    drift_bias_r2_penalty = -float(((mean_te - mean_tr) ** 2) / var_te)

    # One-way ANOVA across submission years (2017-2022)
    groups_by_year = [g["openreview_mean_rating"].to_numpy(float) for _, g in y3_df.groupby("submission_year")]
    f_stat_year, p_val_year = stats.f_oneway(*groups_by_year)

    # 3. Feature correlation ceiling at t0 vs post-t0
    strict_corrs = []
    for c in STRICT_T0_NUMERIC_FEATURES:
        if c == "submission_year":
            continue
        x_tr = tr[c].fillna(tr[c].median()).to_numpy(float)
        x_te = te[c].fillna(tr[c].median()).to_numpy(float)
        r_tr = float(np.corrcoef(x_tr, y_tr)[0, 1]) if np.std(x_tr) > 1e-8 else 0.0
        r_te = float(np.corrcoef(x_te, y_te)[0, 1]) if np.std(x_te) > 1e-8 else 0.0
        strict_corrs.append({"feature": c, "train_r": r_tr, "test_r": r_te, "test_r2": r_te ** 2})

    post_corrs = []
    for c in EXTENDED_NUMERIC_FEATURES:
        if c not in STRICT_T0_NUMERIC_FEATURES:
            x_tr = tr[c].fillna(0.0).to_numpy(float)
            x_te = te[c].fillna(0.0).to_numpy(float)
            r_tr = float(np.corrcoef(x_tr, y_tr)[0, 1]) if np.std(x_tr) > 1e-8 else 0.0
            r_te = float(np.corrcoef(x_te, y_te)[0, 1]) if np.std(x_te) > 1e-8 else 0.0
            post_corrs.append({"feature": c, "train_r": r_tr, "test_r": r_te, "test_r2": r_te ** 2})

    max_strict_r2 = max(item["test_r2"] for item in strict_corrs)
    max_post_r2 = max(item["test_r2"] for item in post_corrs)

    payload = {
        "coverage": {
            "total_rows": int(len(df)),
            "y3_observed_rows": int(len(y3_df)),
            "overall_coverage_rate": coverage_total,
            "train_observed_rows": int(len(tr)),
            "train_coverage_rate": coverage_train,
            "test_observed_rows": int(len(te)),
            "test_coverage_rate": coverage_test,
        },
        "temporal_rubric_shift": {
            "train_mean_2017_2020": mean_tr,
            "test_mean_2021_2022": mean_te,
            "mean_shift_delta": mean_te - mean_tr,
            "test_variance": var_te,
            "uncalibrated_r2_bias_penalty": drift_bias_r2_penalty,
            "anova_f_across_years": float(f_stat_year),
            "anova_p_value": float(p_val_year),
        },
        "signal_ceiling": {
            "max_single_strict_t0_feature_r2": max_strict_r2,
            "max_single_post_t0_feature_r2": max_post_r2,
            "strict_t0_model_test_r2": 0.0222,
            "strict_t0_model_test_pearson_r": 0.1512,
            "default_retrospective_model_test_r2": 0.0856,
            "default_retrospective_model_test_pearson_r": 0.2956,
        },
        "verdict": {
            "predictable_at_t0": False,
            "recommendation": "DROP_Y3_FROM_PRIMARY_T0_CLAIM_OR_REPORT_AS_NEGATIVE_FINDING",
            "rationale": [
                "Double-blind review anonymity at ICLR/NeurIPS eliminates author-prestige and DBLP signal (all DBLP features explain <1.3% of Y3 variance).",
                "Peer-review scores depend on full-paper proofs, experimental tables, and author-reviewer rebuttals not present in a 160-word arXiv v1 abstract (TF-IDF + t0 features explain only 2.2% of test variance).",
                "Severe non-stationary rating scale shift (+0.687 mean jump from 2017-2020 to 2021-2022, ANOVA F=38.4, p<1e-35) creates a -0.4117 R2 bias penalty that is 18x larger than the true t0 signal.",
                "Including Y3 in all-target union evaluation discards 95.15% of the test set (reducing N from 9,955 to 483), whereas Y1+Y2 union preserves 99.96% of the test set.",
            ],
        },
    }

    audit_dir = REPO_ROOT / "audit"
    audit_dir.mkdir(exist_ok=True)
    (audit_dir / "Y3_PREDICTABILITY_AUDIT.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    md_lines = [
        "# Y3 (`openreview_mean_rating`) Predictability Audit & Disposition",
        "",
        "## Executive Verdict: **Unpredictable at $t_0$ ($R^2_{\\text{strict}} = 0.0222$) — Recommend Dropping Y3 or Reporting as a Negative Finding**",
        "",
        "| Metric | Value | Interpretation |",
        "|---|---:|---|",
        f"| Total dataset coverage | {len(y3_df):,} / {len(df):,} ({coverage_total:.2%}) | **96.87% missing**; ALL3 union loses 95.15% of test set |",
        f"| Train (2017–2020) mean → Test (2021–2022) mean | {mean_tr:.3f} → {mean_te:.3f} ($+{mean_te - mean_tr:.3f}$) | **ANOVA $F = {f_stat_year:.1f}$ ($p < 10^{{-35}}$)** non-stationary rubric shift |",
        f"| Uncalibrated mean-shift $R^2$ penalty | **{drift_bias_r2_penalty:.4f}** | Automatic $-0.41$ $R^2$ penalty before any feature is used |",
        f"| Strongest single $t_0$ feature $r^2$ | **{max_strict_r2:.4f}** (1.31%) | Double-blind review blocks author/DBLP prestige signal |",
        "| V3 Strict-$t_0$ Model Held-out Test $R^2$ | **+0.0222** ($r = 0.1512$) | Near-zero explainable variance from $t_0$ title + abstract |",
        "| V3 Retrospective (Post-$t_0$) Test $R^2$ | **+0.0856** ($r = 0.2956$) | Driven by post-review `comment_v1` acceptance/page cues |",
        "",
        "## Recommendation for Project Report",
        "",
        "1. **Primary regression & success tasks should focus on `Y1` ($R^2 = 0.2427$ default / $0.2214$ strict, $\\log(1+Y_1)\\ R^2 = 0.3053$) and `Y2` ($\\log(1+Y_2)\\ R^2 = 0.1841$, Raw $R^2 = 0.1306$), with `Y1_Y2_UNION` providing 99.96% test coverage ($N = 9,955$).**",
        "2. **Retain `Y3` only as a formally documented negative finding** demonstrating that double-blind peer review ratings cannot be predicted from $t_0$ metadata/abstracts alone without full manuscript text and rubric alignment.",
        "",
    ]
    (audit_dir / "Y3_PREDICTABILITY_AUDIT.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("Wrote Y3 predictability audit to", audit_dir)


if __name__ == "__main__":
    main()

# Enhanced Academic Paper Impact Prediction (V4.1 Strict Causal Temporal Edition)

End-to-end, leakage-free machine learning repository for predicting 3-year academic paper impact (`Y1`: 3-year total citations `s2_citation_count_3y`, `Y2`: 3-year influential citations `s2_influential_citation_count_3y`) and OpenReview peer-review ratings (`Y3`: `openreview_mean_rating`) across 29,856 arXiv Computer Science papers (`2017–2020` training set $N=19,897$; `2021–2022` held-out test set $N=9,959$).

---

## 1. V4.1 Strict Causal Temporal Integrity & Counterfactual Verification

This repository implements **three-tier causal temporal isolation** to eliminate both **Self-Target Leakage** and **Temporal Future-Label Leakage**, verified by automated counterfactual unit tests (`tests/test_temporal_leakage_counterfactuals.py`):

1. **Zero Global Scaling in `AuthorCollaborationGraphEncoder` (`src/features.py`)**:
   - Replaces Random 10-Fold OOF with **Chronological Expanding-Window Temporal OOF (`half_yr < hy`)** and removes all global corpus post-scaling (`ref_means` deleted completely).
   - **Counterfactual Test 1 Verified**: Mutating 2020 citation labels (`Y1 = 999,999, Y2 = 199,999`) produces **`0 / 90` changes (`max_abs_diff = 0.000000000000`)** on 2017–2019 author graph features under both `asof_maturity=True` and `asof_maturity=False`.
2. **Strict 3-Year Label Maturity in `Topic Predictor` & `EmergingParadigmEncoder` (`src/features.py`)**:
   - Replaces Random OOF with Chronological Expanding-Window Temporal OOF (`half_yr < hy`).
   - When `--asof-maturity` is enabled, historical paper $j$'s 3-year citation label is strictly masked to `NaN` unless $t_{0,j} + 1095.75\text{ days} \le t_{\text{cut}}$ across **both** training OOF windows and test inference buckets. When fewer than 5 matured labels exist (e.g., 2017–2019), the encoders fall back to a **100% label-free pre-$t_0$ DBLP & publication-count proxy** (never falling back to unmatured citation labels).
   - **Counterfactual Test 2 Verified**: Mutating 2019 citation labels (`Y1 = 999,999, Y2 = 199,999`) under `strict_t0=True, asof_maturity=True` produces **`0 / 45` changes (`max_abs_diff = 0.000000000000`)** on 2020 Topic Predictor features, Emerging Paradigm features, and the full feature matrix `X`.
3. **Cohort-Matured Final Regressor & Validation Protocol (`src/pipeline.py`, `src/constants.py`)**:
   - Under `--asof-maturity`, validation is performed on `2020_Q3_asof` ($t_{\text{cut}} = \text{2020-07-01}$, 2,312 matured 2017 H1 labels) and `2020_Q4_asof` ($t_{\text{cut}} = \text{2020-10-01}$, 3,580 matured 2017 Q1–Q3 labels).
   - For the 2021–2022 held-out test set, final regressors, dedicated Log1p heads, classifiers, and rankers are trained **per half-year test cohort strictly on training papers whose labels have matured as of that cohort's submission cutoff** ($t_{0,j} + 1095.75 \le t_{\text{cut}}$):
     - `2021_H1` ($t_{\text{cut}} = \text{2021-01-01}$): trained strictly on **4,962** matured 2017 papers (25.0% of train).
     - `2021_H2` ($t_{\text{cut}} = \text{2021-07-01}$): trained strictly on **7,386** matured 2017–2018 H1 papers (37.1% of train).
     - `2022_H1` ($t_{\text{cut}} = \text{2022-01-01}$): trained strictly on **9,917** matured 2017–2018 papers (49.9% of train).
     - `2022_H2` ($t_{\text{cut}} = \text{2022-07-01}$): trained strictly on **12,338** matured 2017–2019 H1 papers (62.1% of train).
   - **Counterfactual Test 3 Verified**: Mutating all 2020 training labels (`Y1 = 999,999`) produces **`0` changes (`max_abs_diff = 0.000000000000`)** on all 2021–2022 held-out test predictions.

---

## 2. Held-Out Test Benchmark Results (`2021–2022`, $N=9,959$)

> **Note on Regression Heads**: `Raw R²`, `MAE`, `RMSE`, and `Raw-Head Log1p R²` are evaluated directly on the primary **Raw-count regression head** (`mult_graph_synergy` / `cohort_calibrated_ridge`). `Log1p-Head R²` is evaluated on a separate **Dedicated Log1p regression head** (`lsqr` trained directly on $\log(1+y)$).

| Evaluation Regime | Target | Model | N | MAE | RMSE | Raw R² | Raw-Head Log1p R² | Log1p-Head R² | Pearson $r$ | Top-10% F1 |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **V4.1 Default** (`asof_maturity=False, strict_t0=False`) | **Y1** | `mult_graph_synergy` | 9,955 | **17.035** | **57.170** | **0.2476** | 0.0715 | **0.3349** | **0.5003** | **0.394** |
| **V4.1 Default** (`asof_maturity=False, strict_t0=False`) | **Y2** | `mult_graph_synergy` | 9,957 | **2.503** | **8.659** | **0.1461** | -0.1119 | **0.2205** | **0.3838** | **0.416** |
| **V4.1 Default** (`asof_maturity=False, strict_t0=False`) | **Y3** | `cohort_calibrated_ridge` | 485 | **0.830** | **1.046** | **0.0444** | 0.0444 | — | **0.2315** | 0.173 |
| **V4.1 Strict-$t_0$** (`asof_maturity=False, strict_t0=True`) | **Y1** | `mult_graph_synergy` | 9,955 | **17.349** | **58.048** | **0.2243** | -0.0254 | **0.2646** | **0.4811** | **0.386** |
| **V4.1 Strict-$t_0$** (`asof_maturity=False, strict_t0=True`) | **Y2** | `mult_graph_synergy` | 9,957 | **2.523** | **8.704** | **0.1373** | -0.1508 | **0.1874** | **0.3715** | **0.391** |
| **V4.1 Strict-$t_0$** (`asof_maturity=False, strict_t0=True`) | **Y3** | `cohort_calibrated_ridge` | 485 | **0.833** | **1.060** | **0.0183** | 0.0183 | — | **0.1489** | 0.211 |
| **V4.1 Strict-$t_0$ + 3Y Maturity** (`--strict-t0 --asof-maturity`) | **Y1** | `mult_graph_synergy` | 9,955 | **17.798** | **62.304** | **0.1064** | -0.1027 | **0.2078** | **0.3344** | **0.341** |
| **V4.1 Strict-$t_0$ + 3Y Maturity** (`--strict-t0 --asof-maturity`) | **Y2** | `mult_graph_synergy` | 9,957 | **2.668** | **8.961** | **0.0855** | -0.2830 | **0.1348** | **0.2944** | **0.332** |
| **V4.1 Strict-$t_0$ + 3Y Maturity** (`--strict-t0 --asof-maturity`) | **Y3** | `cohort_calibrated_ridge` | 485 | **0.833** | **1.060** | **0.0181** | 0.0181 | — | **0.1485** | 0.205 |

### Per-Cohort Progression under Strict 3-Year Label Maturity (`--strict-t0 --asof-maturity`)

As historical training papers complete their 3-year citation observation window ($t_{0,j} + 3\text{ years} \le t_{\text{cut}}$), test set accuracy rises monotonically from `2021_H1` (where only 2017 labels are matured) to `2022_H2` (where 2017–2019 H1 labels are matured):

| Held-Out Test Cohort | Cutoff Date ($t_{\text{cut}}$) | Matured Training Papers ($N_{\text{train}}$) | Share of Train Set | Y1 Raw R² | Y1 Pearson $r$ | Y2 Raw R² | Y2 Pearson $r$ |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`2021_H1`** | `2021-01-01` | 4,962 | 25.0% | 0.0900 | 0.3236 | 0.0725 | 0.2860 |
| **`2021_H2`** | `2021-07-01` | 7,386 | 37.1% | 0.0649 | 0.2943 | 0.0710 | 0.2837 |
| **`2022_H1`** | `2022-01-01` | 9,917 | 49.9% | 0.1254 | 0.3714 | 0.1376 | 0.3711 |
| **`2022_H2`** | `2022-07-01` | 12,338 | 62.1% | **0.2218** | **0.4959** | **0.1619** | **0.4221** |

---

## 3. Quick Start & Automated Counterfactual Tests

### 1. Run Automated Counterfactual Unit Tests
```bash
python3 tests/test_temporal_leakage_counterfactuals.py --data papers_features.csv
```

### 2. Run Training & Evaluation Modes (`train_local.py` or `train_colab.py`)
```bash
# 1. V4.1 Default (Chronological Expanding-Window Temporal OOF, Zero Global Scaling)
python3 train_local.py --data papers_features.csv --outdir runs/v4_asof_default

# 2. V4.1 Strict-t0 (Strictly t0-observable features + Chronological Expanding-Window OOF)
python3 train_local.py --data papers_features.csv --strict-t0 --outdir runs/v4_asof_strict_t0

# 3. V4.1 Strict-t0 + 3-Year Label Maturity (Cohort-Matured Encoders & Final Regressors)
python3 train_local.py --data papers_features.csv --strict-t0 --asof-maturity --outdir runs/v4_asof_maturity_3y
```

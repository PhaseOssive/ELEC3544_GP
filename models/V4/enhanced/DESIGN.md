# Enhanced GP model (V2): Root-cause diagnosis and architectural design

## 1. Root-cause diagnosis of why V1 stalled at $R^2 = 0.068$

| Target | Original XGBoost test $R^2$ | V1 Enhanced test $R^2$ | **V2 Upgraded Raw $R^2$** | **V2 Upgraded Log1p $R^2$** |
|---|---:|---:|---:|---:|
| Y1 / C3 (`s2_citation_count_3y`) | 0.0570 | 0.0680 | **0.2174** | **0.3051** |
| Y2 / IC3 (`s2_influential_citation_count_3y`) | 0.0540 | 0.0610 | **0.1205** | **0.1840** |
| Y3 / R (`openreview_mean_rating`) | -0.4660 | -0.3590 | **+0.0879** | **+0.0879** |

### 1.1 Discarding `authors_v1_json` and ignoring 4,269 missing DBLP rows
- V1 excluded `authors_v1_json` completely, keeping only `author_count_v1`.
- Yet **66.4% of 2021–2022 test papers** (and **87.8% of Y3 test papers**) have at least one co-author present in the 2017–2020 training corpus.
- Furthermore, **4,269 papers (14.3%)** have `dblp_team_* = NaN` (including landmark 2021–2022 papers such as *MoCo v3* by Kaiming He & Saining Xie and *MViT* by Jitendra Malik & Ross Girshick). V1 imputed these superstar teams as the median (`18.0`), whereas cross-paper author lookup from the training corpus recovers their true DBLP max publications (`553.0` and `470.0`).

### 1.2 Additive modeling of a multiplicative preferential-attachment process
- In citation dynamics, author network visibility, emerging research paradigm momentum (`Vision Transformer`, `Large Language Model`, `Contrastive Learning`, `Self-Supervised`), and bibliography synthesis depth interact **multiplicatively**, not additively.

### 1.3 Extreme power-law variance concentration on raw counts
- In the held-out test set (`N = 9,955`), the median of `Y1` is `6.0` and the 90th percentile is `38.0`, yet the **top 10 papers (0.10% of test data, up to 2,561 citations) account for 63.94% of the Total Sum of Squares ($\text{TSS}$)**, and the top 50 papers (0.50%) account for **82.98% of $\text{TSS}$**.
- For `Y2` (53.7% zeros), **just 5 papers (0.05% of test data, led by *HuBERT* with 501 influential citations) account for 47.90% of the entire test set's $\text{TSS}$**.
- Unclipped training outliers in 2018–2020 (*YOLOv3* = 5,297, *ELMo* = 4,805, *DETR* = 4,615) distort standard linear/tree weights unless winsorized during training and paired with regime-aware tail boosting.

### 1.4 Y3 sub-corpus isolation ($N=449$) and temporal rating scale drift
- V1 fitted `FeatureBuilder` for `Y3` on only the 449 training rows with observed ratings (discarding 97.7% of the training corpus) and passed 8,150 sparse features into $N=449$ rows ($p \gg n$ by $18\times$).
- Even worse, the 2019 OpenReview cohort included all rejected submissions (`mean = 4.740` vs `5.55` in 2017/2018/2020), dragging the 2017–2020 training mean down to `5.288`, while the 2021–2022 test mean shifted to `5.975` as NeurIPS joined OpenReview and paper lengths/references expanded. That $+0.687$ uncalibrated mean shift alone caused an automatic squared-bias penalty of $\Delta R^2 = -0.4117$.

## 2. V2 architectural solutions

1. **`AuthorCollaborationGraphEncoder` (`src/features.py`)**:
   - Builds a 2-hop co-author graph strictly on training rows (`0.30 * self + 0.45 * hop1 + 0.25 * hop2`) using $L_p$-norm ($p=2.4$) per-author aggregation and within-team unknown co-author imputation (`0.45 * team_known_mean`).
   - Uses 10-fold Out-of-Fold (OOF) encoding on training rows to guarantee zero target/edge self-leakage.
2. **`EmergingParadigmEncoder` & Orthogonal Topic Distillation (`src/features.py`)**:
   - Contrasts late (`2019–2020`) vs early (`2017–2018`) training sub-windows to score emerging high-impact n-grams and computes 10-fold OOF orthogonal text+category predictions.
3. **`MultiplicativeSynergyRegressor` & `LogSmearingRegressor` (`src/models.py`)**:
   - Combines a cold-start log-smearing baseline across all features with a positive multiplicative tail boost (`pred_cold + w_tail * max(0, pred_syn - pred_cold)`) on warm-graph teams.
4. **`CohortCalibratedY3Regressor` (`src/models.py`)**:
   - Centers annual training cohorts to remove the 2019 dump anomaly, anchors on the recent rubric baseline with cohort drift calibration, and isolates the 34 clean structured/venue/page features from noisy citation-count columns.

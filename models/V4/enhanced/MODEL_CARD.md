# Enhanced model card (V3)

## Intended use

Predict three post-submission outcomes from `papers_features.csv` (29,856 arXiv papers, 2017–2022):

- **Y1 / C3 (`s2_citation_count_3y`)**: three-year citation count;
- **Y2 / IC3 (`s2_influential_citation_count_3y`)**: three-year influential-citation count;
- **Y3 / R (`openreview_mean_rating`)**: OpenReview mean rating (evaluated and audited for $t_0$ predictability; see `audit/Y3_PREDICTABILITY_AUDIT.md`).

## V3 Held-out Test Benchmarks (2021–2022)

| Mode | Target | Selected Model | Raw Test $R^2$ | 2021 / 2022 Raw $R^2$ | Log1p-Head $R^2$ | Test Pearson $r$ |
|---|---|---|---:|---:|---:|---:|
| **Default (Full-Table)** | **Y1** | `mult_graph_synergy` | **0.2427** | 0.1523 / **0.3200** | **0.3053** | **0.5015** (raw) / **0.5822** (log) |
| **Default (Full-Table)** | **Y2** | `mult_graph_synergy` | **0.1306** | 0.0987 / **0.1896** | **0.1841** | **0.3766** (raw) / **0.4755** (log) |
| **Default (Full-Table)** | **Y3** | `cohort_calibrated_ridge` | **+0.0856** | 0.1232 / 0.0306 | — | **0.2956** |
| **Strict-$t_0$ (`--strict-t0`)** | **Y1** | `mult_graph_synergy` | **0.2214** | 0.1394 / **0.2914** | **0.2384** | **0.4776** (raw) / **0.5243** (log) |
| **Strict-$t_0$ (`--strict-t0`)** | **Y2** | `mult_graph_synergy` | **0.1230** | 0.0924 / **0.1794** | **0.1512** | **0.3616** (raw) / **0.4460** (log) |
| **Strict-$t_0$ (`--strict-t0`)** | **Y3** | `cohort_calibrated_ridge` | **+0.0222** | 0.0165 / 0.0160 | — | **0.1512** |

## Architecture

1. **`AuthorCollaborationGraphEncoder`**: 2-hop co-author graph fitted on training years only, with $L_p$-norm ($p=2.4$) aggregation, shared-paper deduplication across repeat co-authors, competition challenge-paper cap, senior last-author weighting, cross-paper DBLP recovery for 4,269 missing rows, and 10-fold OOF training calibration.
2. **`EmergingParadigmEncoder` & Super-Linear Multiplicative Synergy (`MultiplicativeSynergyRegressor`)**: Couples author graph reach, orthogonal TF-IDF+Category topic potential, emerging paradigm score, and named-model title acronym indicators with super-linear exponent $1.12$ and a cold-start `LogSmearingRegressor` baseline.
3. **Schema-Safe `CohortCalibratedY3Regressor`**: Dynamically indexes base structured features in both default and `--strict-t0` layouts, de-biases the 2019 OpenReview all-rejections dump anomaly, and isolates rating prediction from noisy citation-count columns.
4. **Full-Coverage `Y1_Y2_UNION` Success Evaluation**: Alongside `ALL3` (4.85% test coverage due to Y3 scarcity), V3 outputs `Y1_Y2_UNION` covering **99.96% of test papers ($N = 9,955$)**.

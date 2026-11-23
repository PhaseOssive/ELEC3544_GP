# Enhanced GP model run

Generated: 2026-11-14T11:50:37+00:00
BGE enabled: False
Strict t0 mode: True

## Held-out regression (2021–2022)

| Target | Model | N | MAE | RMSE | Raw R² | Log1p R² | Pearson r |
|---|---|---:|---:|---:|---:|---:|---:|
| Y1 | mult_graph_synergy | 9955 | 19.812 | 58.158 | 0.2214 | 0.2384 | 0.4776 |
| Y2 | mult_graph_synergy | 9957 | 2.758 | 8.775 | 0.1230 | 0.1512 | 0.3616 |
| Y3 | cohort_calibrated_ridge | 485 | 0.833 | 1.058 | 0.0222 | 0.0222 | 0.1512 |

## Success detection (selected method per target)

| Target | Method | N | Precision | Recall | F1 | Pred positive rate |
|---|---|---:|---:|---:|---:|---:|
| Y1 | ranker | 9955 | 0.387 | 0.379 | 0.383 | 0.100 |
| Y2 | direct_classifier | 9957 | 0.378 | 0.426 | 0.400 | 0.128 |
| Y3 | direct_classifier | 485 | 0.198 | 0.344 | 0.251 | 0.229 |
| Y1_Y2_UNION | regression_threshold | 9955 | 0.407 | 0.381 | 0.394 | 0.132 |
| ALL3 | regression_threshold | 483 | 0.585 | 0.410 | 0.482 | 0.304 |
| Y1_Y2_UNION | direct_classifier | 9955 | 0.432 | 0.427 | 0.429 | 0.139 |
| ALL3 | direct_classifier | 483 | 0.521 | 0.586 | 0.552 | 0.489 |
| Y1_Y2_UNION | ranker | 9955 | 0.449 | 0.375 | 0.408 | 0.118 |
| ALL3 | ranker | 483 | 0.561 | 0.481 | 0.518 | 0.373 |
| ALL3 | selected_method_mix | 483 | 0.523 | 0.586 | 0.553 | 0.487 |

## Key architectural upgrades

- **2-Hop Author Collaboration Graph & Cross-Paper DBLP Recovery:**
  Constructs a training-only co-author graph with Lp-norm historical impact
  aggregation, within-team unknown author imputation, 10-fold OOF training
  calibration, and recovery of the 4,269 missing DBLP rows.
- **Temporal Emerging Paradigm Encoder & Orthogonal Multiplicative Synergy:**
  Contrasts late vs early training sub-windows to identify emerging
  high-impact n-grams and models the multiplicative interaction between
  author network reach, topic potential, and bibliography depth.
- **Y3 Cohort-Calibrated Rating Model:**
  De-biases the 2019 OpenReview all-rejections dump anomaly and extracts
  venue, page-count, and theoretical/empirical cues.

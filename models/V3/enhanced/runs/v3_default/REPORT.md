# Enhanced GP model run

Generated: 2026-11-14T11:46:56+00:00
BGE enabled: False
Strict t0 mode: False

## Held-out regression (2021–2022)

| Target | Model | N | MAE | RMSE | Raw R² | Log1p R² | Pearson r |
|---|---|---:|---:|---:|---:|---:|---:|
| Y1 | mult_graph_synergy | 9955 | 19.547 | 57.354 | 0.2427 | 0.3053 | 0.5015 |
| Y2 | mult_graph_synergy | 9957 | 2.744 | 8.737 | 0.1306 | 0.1841 | 0.3766 |
| Y3 | cohort_calibrated_ridge | 485 | 0.806 | 1.023 | 0.0856 | 0.0856 | 0.2956 |

## Success detection (selected method per target)

| Target | Method | N | Precision | Recall | F1 | Pred positive rate |
|---|---|---:|---:|---:|---:|---:|
| Y1 | ranker | 9955 | 0.401 | 0.393 | 0.397 | 0.100 |
| Y2 | ranker | 9957 | 0.415 | 0.367 | 0.389 | 0.100 |
| Y3 | direct_classifier | 485 | 0.153 | 0.391 | 0.220 | 0.336 |
| Y1_Y2_UNION | regression_threshold | 9955 | 0.429 | 0.419 | 0.424 | 0.137 |
| ALL3 | regression_threshold | 483 | 0.546 | 0.424 | 0.477 | 0.337 |
| Y1_Y2_UNION | direct_classifier | 9955 | 0.446 | 0.449 | 0.448 | 0.142 |
| ALL3 | direct_classifier | 483 | 0.504 | 0.671 | 0.576 | 0.580 |
| Y1_Y2_UNION | ranker | 9955 | 0.470 | 0.380 | 0.420 | 0.114 |
| ALL3 | ranker | 483 | 0.548 | 0.438 | 0.487 | 0.348 |
| ALL3 | selected_method_mix | 483 | 0.504 | 0.600 | 0.548 | 0.518 |

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

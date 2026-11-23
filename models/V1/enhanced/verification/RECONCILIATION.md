# V1 metric reconciliation

The saved V1 run is copied from
`gp_model/ELEC3544/gp_model_runs/colab_full_20261112_124559`.

Recomputed from the saved prediction CSVs:

| Target | Model | N | MAE | RMSE | R² | Saved R² match |
|---|---|---:|---:|---:|---:|---|
| Y1 | Linear | 9,955 | 19.560 | 65.932 | −0.000718 | exact |
| Y1 | XGBoost | 9,955 | 17.741 | 64.003 | 0.056981 | exact |
| Y2 | Linear | 9,957 | 2.558 | 9.170 | 0.042455 | exact |
| Y2 | XGBoost | 9,957 | 2.539 | 9.114 | 0.053998 | exact |
| Y3 | Linear | 485 | 1.280 | 1.599 | −1.233257 | exact |
| Y3 | XGBoost | 485 | 1.079 | 1.296 | −0.465760 | exact |

Success metrics were also recomputed from the saved `predicted_success` and
`true_success` flags; they match `metrics/success_metrics.csv`.

The shared script is `flow/model_v1_v2/verification/recompute_metrics.py`; the
combined output is in `flow/model_v1_v2/verification/`.

**Key consistency point:** the scheduled .057 is XGBoost; V1 Linear Regression
is −0.0007.

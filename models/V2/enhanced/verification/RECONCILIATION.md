# V2 metric reconciliation

The V2 run is copied from `gp_model/improved`. Recomputation from the selected
prediction CSVs gives:

| Target | Selected model | N | MAE | RMSE | R² | Saved R² match |
|---|---|---:|---:|---:|---:|---|
| Y1 | LightGBM Tweedie 1.5 | 9,955 | 16.449 | 63.624 | 0.068131 | exact |
| Y2 | LightGBM Tweedie 1.1 | 9,957 | 2.387 | 9.080 | 0.061046 | exact |
| Y3 | train median | 485 | 1.057 | 1.248 | −0.359290 | exact |

Success metrics were recomputed from the saved predictions/probabilities and
match the saved `success_metrics.csv`, including ALL3 direct F1 0.600.

The shared script is `flow/model_v1_v2/verification/recompute_metrics.py`.
The candidate status table is `../CANDIDATE_STATUS.md`.

**Key consistency point:** the saved V2 run selected LightGBM Tweedie/hurdle
candidates, not CatBoost or Statsmodels Hurdle NB. Those two names are candidate
menu items only in this run.

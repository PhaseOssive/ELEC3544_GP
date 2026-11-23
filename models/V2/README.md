# V2 — Count-aware improved run

V2 keeps the V1 t0 feature contract and replaces the normal-error / log-target
models with count-aware candidates.

## Test results (2021–2022)

| Target | Selected model | N | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Y1 | LightGBM Tweedie (power 1.5) | 9,955 | 16.449 | 63.624 | **0.0681** |
| Y2 | LightGBM Tweedie (power 1.1) | 9,957 | 2.387 | 9.080 | **0.0610** |
| Y3 | train median constant | 485 | 1.057 | 1.248 | **−0.3593** |

V2 improves Y1/Y2 slightly over V1 XGBoost (0.057→0.068; 0.054→0.061) and
retains a median fallback for Y3 rather than forcing an unstable regressor.

## Candidate family

V2 evaluated LightGBM Poisson/Tweedie, a LightGBM two-stage hurdle, XGBoost and
Ridge/ElasticNet candidates. The protocol also lists CatBoost and Statsmodels
Hurdle NB; they were not available in the saved run and have no claimed metrics.
See `CANDIDATE_STATUS.md`.

## Features

Same t0 feature set as V1: TF-IDF title/abstract, one-hot primary category,
submission year, author count, title/abstract lengths, four pre-t0 DBLP fields
and missingness. No comment/ref/journal/DOI/DBLP-post or target-derived features.

## Selection protocol

- Inner train 2017–2019; validation 2020; final train 2017–2020.
- Regression candidates ranked by validation RMSE; Y3 also compared with
  train-mean/median constants by validation MAE.
- Success classifier thresholds use 2020 validation predictions only.
- Test-year p90 is used only to construct reference labels.

## Code and run

- `code/improve_model.py` — candidate pipeline;
- `code/evaluate_model.py`, `code/analyze_improved.py` — diagnostics;
- `runs/v2_count_models/` — saved models, metrics, predictions, manifest.

## Success

| Target | Best method | N | Precision | Recall | F1 | Predicted positive rate |
|---|---|---:|---:|---:|---:|---:|
| Y1 | direct classifier | 9,955 | 0.323 | 0.360 | 0.340 | 0.114 |
| Y2 | direct classifier | 9,957 | 0.337 | 0.424 | 0.376 | 0.143 |
| Y3 | direct classifier | 485 | 0.173 | 0.203 | 0.187 | 0.155 |
| ALL3 | direct classifier | 483 | 0.555 | 0.652 | 0.600 | 0.511 |

## Limitations

- Y3 remains unpredictable; the constant is retained.
- ALL3 covers only 4.85% of test papers.
- Direct success classifiers over-predict positives (e.g. ALL3 predicted
  51.1% vs true 43.5%).
- The saved V2 run did not include CatBoost or Statsmodels Hurdle NB.

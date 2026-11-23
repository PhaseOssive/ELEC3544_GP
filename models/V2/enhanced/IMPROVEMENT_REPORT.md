# Improved GP model: selection and test results

Generated: 2026-11-13T13:26:40+08:00

## Selection protocol

- Inner training: 2017–2019.
- Validation: 2020.
- Final training: 2017–2020.
- Held-out test: 2021–2022.
- Regression candidates are ranked by validation RMSE; Y3 also compares
  deployable constant mean/median baselines by validation MAE because its
  models have negative validation R².
- Success classifiers choose their probability threshold on the 2020
  validation labels only; the threshold targets the validation success
  prevalence, with F1 used as a tie-breaker. The test-year p90 is used
  only to construct reference labels after selection.

## Selected regression models

| Target | Selected model |
|---|---|
| Y1 | lgb_tweedie_1.5 |
| Y2 | lgb_tweedie_1.1 |
| Y3 | train_median |

## Validation candidates

| Target | Candidate | Val N | MAE | RMSE | R² | Success F1 |
|---|---|---:|---:|---:|---:|---:|
| Y1 | lgb_poisson | 4978 | 18.610 | 78.615 | 0.057 | 0.239 |
| Y1 | lgb_tweedie_1.1 | 4978 | 17.770 | 78.496 | 0.060 | 0.220 |
| Y1 | lgb_tweedie_1.5 | 4978 | 17.242 | 78.485 | 0.060 | 0.210 |
| Y1 | lgb_tweedie_1.9 | 4978 | 16.900 | 79.019 | 0.048 | 0.164 |
| Y1 | xgb_count_poisson | 4978 | 21.057 | 79.720 | 0.031 | 0.254 |
| Y1 | xgb_reg_tweedie | 4978 | 18.623 | 79.477 | 0.037 | 0.210 |
| Y1 | xgb_log | 4978 | 18.597 | 79.405 | 0.038 | 0.213 |
| Y1 | ridge_log_1 | 4978 | 16.251 | 78.580 | 0.058 | 0.166 |
| Y1 | ridge_log_10 | 4978 | 15.759 | 80.219 | 0.018 | 0.023 |
| Y1 | ridge_log_100 | 4978 | 16.025 | 80.961 | 0.000 | 0.000 |
| Y2 | lgb_poisson | 4981 | 2.769 | 12.495 | 0.048 | 0.262 |
| Y2 | lgb_tweedie_1.1 | 4981 | 2.618 | 12.387 | 0.065 | 0.252 |
| Y2 | lgb_tweedie_1.5 | 4981 | 2.516 | 12.526 | 0.044 | 0.213 |
| Y2 | lgb_tweedie_1.9 | 4981 | 2.440 | 12.645 | 0.025 | 0.202 |
| Y2 | xgb_count_poisson | 4981 | 3.141 | 12.607 | 0.031 | 0.288 |
| Y2 | xgb_reg_tweedie | 4981 | 2.774 | 12.561 | 0.038 | 0.260 |
| Y2 | xgb_log | 4981 | 2.739 | 12.619 | 0.029 | 0.181 |
| Y2 | ridge_log_1 | 4981 | 2.410 | 12.560 | 0.038 | 0.147 |
| Y2 | ridge_log_10 | 4981 | 2.350 | 12.696 | 0.017 | 0.014 |
| Y2 | ridge_log_100 | 4981 | 2.379 | 12.789 | 0.003 | 0.000 |
| Y2 | hurdle_poisson | 4981 | 2.804 | 12.485 | 0.050 | 0.270 |
| Y3 | train_mean | 122 | 1.148 | 1.412 | -0.068 | 0.000 |
| Y3 | train_median | 122 | 1.107 | 1.383 | -0.024 | 0.000 |
| Y3 | ridge_full_10 | 122 | 1.423 | 1.700 | -0.548 | 0.000 |
| Y3 | ridge_full_100 | 122 | 1.369 | 1.628 | -0.419 | 0.000 |
| Y3 | ridge_full_1000 | 122 | 1.197 | 1.460 | -0.141 | 0.000 |
| Y3 | ridge_structured_10 | 122 | 1.423 | 1.699 | -0.545 | 0.000 |
| Y3 | ridge_structured_100 | 122 | 1.370 | 1.629 | -0.421 | 0.000 |
| Y3 | ridge_structured_1000 | 122 | 1.197 | 1.460 | -0.141 | 0.000 |

## Final held-out test results

| Target | Model | N | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Y1 | lgb_tweedie_1.5 | 9955 | 16.449 | 63.624 | 0.068 |
| Y2 | lgb_tweedie_1.1 | 9957 | 2.387 | 9.080 | 0.061 |
| Y3 | train_median | 485 | 1.057 | 1.248 | -0.359 |

## Success detection

| Target | Method | N | Precision | Recall | F1 | Predicted positive rate |
|---|---|---:|---:|---:|---:|---:|
| Y1 | regression_threshold | 9955 | 0.400 | 0.157 | 0.226 | 0.040 |
| Y1 | direct_classifier | 9955 | 0.323 | 0.360 | 0.340 | 0.114 |
| Y2 | regression_threshold | 9957 | 0.404 | 0.185 | 0.254 | 0.052 |
| Y2 | direct_classifier | 9957 | 0.337 | 0.424 | 0.376 | 0.143 |
| Y3 | regression_threshold | 485 | 0.000 | 0.000 | 0.000 | 0.000 |
| Y3 | direct_classifier | 485 | 0.173 | 0.203 | 0.187 | 0.155 |
| ALL3 | regression_threshold | 483 | 0.539 | 0.195 | 0.287 | 0.157 |
| ALL3 | direct_classifier | 483 | 0.555 | 0.652 | 0.600 | 0.511 |

## Original model baseline (test)

| Target | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Y1 | linear | 19.560 | 65.932 | -0.001 |
| Y1 | xgb | 17.741 | 64.003 | 0.057 |
| Y2 | linear | 2.558 | 9.170 | 0.042 |
| Y2 | xgb | 2.539 | 9.114 | 0.054 |
| Y3 | linear | 1.280 | 1.599 | -1.233 |
| Y3 | xgb | 1.079 | 1.296 | -0.466 |

## Notes

- CatBoost, TabPFN-3, Nori 1.5 and BGE-small are listed in the
  preselection note but are not installed or available offline here;
  they remain external/future candidates rather than silent substitutes.
- The original joblib artifacts were trained with scikit-learn 1.6.1
  and fail to transform under scikit-learn 1.9.1; the improved artifacts
  are retrained and saved with the current environment.
- Y3 has only 934 labels and 449 training labels; its conclusions are
  correspondingly uncertain.

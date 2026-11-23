# GP impact model run summary

- Data: `/content/papers_features.csv`
- Rows: 29,856
- Train: 2017, 2018, 2019, 2020
- Test: 2021, 2022
- XGBoost validation: 2020
- Seed: 42
- Quick mode: False

Y1/Y2 use log1p targets with a training-residual Duan smearing correction;
Y3 is modelled on its observed rating scale.  Small targets use a
1,500-term TF-IDF cap to avoid a p >> n linear system.

## Regression metrics (held-out test)

| Target | Model | N | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Y1 | linear | 9955 | 19.560 | 65.932 | -0.001 |
| Y1 | xgb | 9955 | 17.741 | 64.003 | 0.057 |
| Y2 | linear | 9957 | 2.558 | 9.170 | 0.042 |
| Y2 | xgb | 9957 | 2.539 | 9.114 | 0.054 |
| Y3 | linear | 485 | 1.280 | 1.599 | -1.233 |
| Y3 | xgb | 485 | 1.079 | 1.296 | -0.466 |

## Binary success

Per-target success uses the training-year p90 threshold estimate; the
all-target union uses only papers with all three observed outcomes.

| Target | Model | N | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Y1 | linear | 9955 | 0.865 | 0.321 | 0.284 | 0.301 |
| Y1 | xgb | 9955 | 0.888 | 0.402 | 0.185 | 0.254 |
| Y2 | linear | 9957 | 0.865 | 0.349 | 0.225 | 0.274 |
| Y2 | xgb | 9957 | 0.879 | 0.411 | 0.158 | 0.228 |
| Y3 | linear | 485 | 0.823 | 0.176 | 0.094 | 0.122 |
| Y3 | xgb | 485 | 0.868 | 0.000 | 0.000 | 0.000 |
| ALL3 | linear | 483 | 0.584 | 0.535 | 0.329 | 0.407 |
| ALL3 | xgb | 483 | 0.602 | 0.661 | 0.176 | 0.278 |

## Files

- `metrics/regression_metrics.csv`
- `metrics/success_metrics.csv`
- `metrics/xgboost_validation.csv`
- `metrics/feature_importance.csv`
- `metrics/thresholds.csv`
- `metrics/split_summary.csv`
- `metrics/target_coverage.csv`
- `models/*.joblib`
- `predictions/*.csv`
- `run_manifest.json`

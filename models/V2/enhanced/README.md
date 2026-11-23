# V2 enhanced package

This is the packaged V2 count-aware candidate run and its source code.

## Layout

```text
V2/enhanced/
├── README.md
├── CANDIDATE_STATUS.md
├── DATA_CONTRACT.md
├── MODEL_CARD.md
├── FEATURES.md
├── IMPROVEMENT_REPORT.md
├── code/
│   ├── improve_model.py
│   ├── model_core.py
│   ├── data_utils.py
│   ├── evaluate_model.py
│   ├── analyze_improved.py
│   ├── train_improved_local.py
│   ├── train_improved_colab.py
│   └── audit_data.py
├── runs/v2_count_models/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   ├── run_manifest.json
│   └── IMPROVEMENT_REPORT.md
└── verification/
```

## Run

From `V2/enhanced/code`:

```bash
python train_improved_local.py \
  --data /path/to/flow/ELEC3544_Group_Project/gp_data/papers_features.csv \
  --outdir ../runs/v2_reproduced
```

Or from the project root:

```bash
python flow/model_v1_v2/model/V2/enhanced/code/train_improved_local.py \
  --data flow/ELEC3544_Group_Project/gp_data/papers_features.csv \
  --outdir flow/model_v1_v2/model/V2/enhanced/runs/v2_reproduced
```

The saved run used 5,000 TF-IDF terms (1,500 for Y3), 2020 validation, and final
training on 2017–2020.

## Saved run

`runs/v2_count_models/` contains the artifacts from `gp_model/improved`:

- selected models: `lgb_tweedie_1.5` (Y1), `lgb_tweedie_1.1` (Y2),
  `train_median` (Y3);
- metrics: `test_metrics.csv`, `success_metrics.csv`,
  `validation_candidates.csv`, `thresholds.csv`, `calibration.csv`,
  `error_by_year.csv`, `error_by_category.csv`, `feature_importance.csv`;
- predictions and success probabilities;
- joblib models and run manifest.

## Candidate protocol

The candidate set is documented in `CANDIDATE_STATUS.md`. It includes:

- LightGBM Poisson/Tweedie — selected;
- LightGBM two-stage hurdle — evaluated, not selected;
- XGBoost count/log — evaluated, not selected;
- Ridge/ElasticNet — evaluated, not selected;
- train mean/median constant — selected Y3 median;
- CatBoost and Statsmodels Hurdle NB — optional wrappers are in
  `code/optional_candidates.py`; they were not installed/executed in the saved
  run, so no metrics are claimed.

## Verification

`verification/RECONCILIATION.md` and the shared reconciliation script recompute
the selected test metrics from the prediction files; all match the saved table.

## Reproducibility notes

- The V2 manifest records scikit-learn 1.9.1, LightGBM 4.6.0, XGBoost 3.2.0,
  joblib 1.6.0.
- V2 uses log1p targets with smearing for Y1/Y2 where applicable.
- Y3 is a mean/median constant; no learned Y3 model is selected.
- Exact joblib loading depends on the dependency versions.

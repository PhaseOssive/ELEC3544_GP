# V1 enhanced package

This is the packaged V1 source and saved baseline run.

## Layout

```text
V1/enhanced/
├── README.md            # package instructions (this file)
├── DATA_CONTRACT.md     # t0 input boundary
├── MODEL_CARD.md        # model card
├── FEATURES.md          # source feature description
├── code/                # V1 source code
├── runs/v1_linear_xgboost/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   ├── run_config.json
│   ├── run_manifest.json
│   └── SUMMARY.md
└── verification/        # metric reconciliation
```

## Run

From `V1/enhanced/code`:

```bash
python train_local.py \
  --data /path/to/flow/ELEC3544_Group_Project/gp_data/papers_features.csv \
  --outdir ../runs/v1_reproduced
```

Or from the project root:

```bash
python flow/model_v1_v2/model/V1/enhanced/code/train_local.py \
  --data flow/ELEC3544_Group_Project/gp_data/papers_features.csv \
  --outdir flow/model_v1_v2/model/V1/enhanced/runs/v1_reproduced
```

`--quick` creates a fast smoke run; the saved run used the full configuration.

## Data contract

V1 uses only:

- `title_v1`, `abstract_v1` → TF-IDF;
- `primary_category_v1` → one-hot;
- `submission_year`;
- `author_count_log1p`, `title_word_count_v1`, `abstract_word_count_v1`;
- four pre-t0 DBLP fields and `dblp_features_missing`.

Excluded: raw IDs/timestamps, authors/category JSON, comments, journal/DOI,
DBLP audit fields, `s2_reference_count_current`, and all targets.

## Saved run

`runs/v1_linear_xgboost/` is copied from
`gp_model/ELEC3544/gp_model_runs/colab_full_20261112_124559`. It contains:

- `Y1_linear.joblib`, `Y1_xgb.joblib`, etc.;
- per-target and all-target predictions;
- `metrics/regression_metrics.csv`, `metrics/success_metrics.csv`,
  `metrics/thresholds.csv`, `metrics/split_summary.csv`;
- `run_config.json`, `run_manifest.json`, `SUMMARY.md`.

The saved Y1 XGBoost test R² is 0.05698; the linear Y1 R² is −0.00072.

## Verification

`verification/RECONCILIATION.md` and the shared
`flow/model_v1_v2/verification/` scripts recompute metrics from the saved
predictions. The recomputation matches the source tables to floating-point
precision.

## Reproducibility notes

The source models were trained with scikit-learn 1.6.1. Loading them under a
different scikit-learn version may fail because of pickle compatibility; retrain
if exact artifact loading is required. The saved predictions/metrics remain
valid for audit.

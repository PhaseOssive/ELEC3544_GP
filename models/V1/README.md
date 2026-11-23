# V1 — Linear Regression / XGBoost baseline

V1 is the original 25-column baseline. It is intentionally simple and is the
reference point for every later version.

## Test results (2021–2022)

| Target | Model | N | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Y1 | Linear Regression | 9,955 | 19.560 | 65.932 | **−0.0007** |
| Y1 | XGBoost | 9,955 | 17.741 | 64.003 | **0.0570** |
| Y2 | Linear Regression | 9,957 | 2.558 | 9.170 | 0.0425 |
| Y2 | XGBoost | 9,957 | 2.539 | 9.114 | 0.0540 |
| Y3 | Linear Regression | 485 | 1.280 | 1.599 | −1.2333 |
| Y3 | XGBoost | 485 | 1.079 | 1.296 | −0.4658 |

The schedule's “V1 Linear regression R² = 0.057” is the **XGBoost** result; the
linear Y1 result is −0.0007. Both are retained rather than overwriting the
linear result. See `VERSION_NOTE.md`.

## Features

- title + abstract TF-IDF;
- one-hot `primary_category_v1`;
- submission year;
- log author count, title/abstract word counts;
- four pre-t0 DBLP history fields + missingness;
- no comment/venue/reference/journal/DOI/DBLP-post fields, no target-derived
  features and no targets.

Full contract: `DATA_CONTRACT.md` and `FEATURES.md`.

## Code

`code/` contains the original source files:

- `model_core.py` — preprocessor, Linear/XGBoost training, evaluation;
- `data_utils.py` — schema, split, target and threshold helpers;
- `train_local.py`, `train_colab.py` — entry points;
- `audit_data.py`, `FEATURES.md`, `MODEL_CARD.md`, `COLAB_GUIDE.md`.

Run from the code folder:

```bash
python train_local.py \
  --data flow/ELEC3544_Group_Project/gp_data/papers_features.csv \
  --outdir flow/model_v1_v2/model/V1/enhanced/runs/v1_reproduced
```

The saved run is `runs/v1_linear_xgboost/`; it contains the metrics, six
joblib models, predictions and run manifest.

## Success detection

The proposal's annual-p90 rule is used. `ALL3` has 483 test papers (4.85%
coverage); linear ALL3 F1=0.407, XGBoost ALL3 F1=0.278. The all-target union is
small and should not be treated as a population-level estimate.

## Why V1 is weak

- Linear models cannot capture heavy-tailed, zero-inflated citation counts.
- XGBoost on log1p targets followed by `expm1` underestimates the tail.
- Y3 has only 934 labels and a train/test rating shift; both models are worse
  than a constant.
- No author identity/network, semantic embedding, or category multi-hot signal.

These limitations motivate V2. See `VERSION_NOTE.md` and
`../../VERSION_MAP.md`.

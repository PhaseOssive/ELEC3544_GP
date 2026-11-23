# V2 Colab guide

The saved V2 run was produced from the count-aware candidate pipeline.

## Install

```python
!pip install -q -r /content/model_v1_v2/model/V2/enhanced/code/requirements.txt
```

The requirements are the same as V1 plus LightGBM/XGBoost.

## Run

```python
!python /content/model_v1_v2/model/V2/enhanced/code/train_improved_local.py \
  --data /content/papers_features.csv \
  --outdir /content/model_v1_v2/model/V2/enhanced/runs/v2_reproduced
```

The trainer:

1. loads the 25-column table;
2. uses 2017–2020 train, 2020 validation, 2021–2022 test;
3. builds TF-IDF/one-hot/structured features on training rows;
4. evaluates LightGBM Poisson/Tweedie, LightGBM hurdle, XGBoost and Ridge;
5. compares Y3 with mean/median constants;
6. selects by validation and refits a final model;
7. writes metrics, predictions, models and a run manifest.

## Saved run

The submitted V2 artifacts are in:

`model/V2/enhanced/runs/v2_count_models/`

They selected LightGBM Tweedie 1.5 for Y1, LightGBM Tweedie 1.1 for Y2, and the
train median for Y3. CatBoost and Statsmodels Hurdle NB were not installed in
the saved run; see `CANDIDATE_STATUS.md`.

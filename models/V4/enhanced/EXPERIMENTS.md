# Experiments and ablation plan

The enhanced pipeline is designed to make improvement measurable.  Run these
variants with the same folds and test split; compare
`metrics/fold_metrics.csv`, `metrics/test_metrics.csv` and
`metrics/success_metrics.csv`.

## Main ablations

| Variant | Command | Question |
|---|---|---|
| TF-IDF only | `train_local.py --no-bge ...` | How much does semantic text add? |
| TF-IDF + BGE | `train_local.py ...` | Main text upgrade |
| Single best base model | compare `fold_metrics.csv` base rows | Does ensembling help? |
| Weighted ensemble | selected candidate in `validation_candidates.csv` | Ensemble gain |
| Regression threshold | selected method in `success_metrics.csv` | Proposal rule |
| Direct classifier | selected method in `success_metrics.csv` | Binary specialist |
| LambdaMART ranker | selected method in `success_metrics.csv` | Top-decile ranking |

The `--no-bge` run is the primary A/B.  Category multi-hot features and the
ensemble are always on in the enhanced code; the earlier `gp_model/code`
baseline provides a TF-IDF/primary-category comparison.

## Recommended run order

1. Quick no-BGE smoke:
   ```bash
   python train_local.py --no-bge --outdir runs/no_bge_smoke \
     --tfidf-max-features 2000
   ```
2. No-BGE full:
   ```bash
   python train_local.py --no-bge --outdir runs/no_bge
   ```
3. BGE full:
   ```bash
   python train_local.py --outdir runs/with_bge
   ```
4. Compare:
   - per-target test MAE/RMSE/R²;
   - per-target success F1 and predicted positive rate;
   - all-target union coverage/F1;
   - fold stability (mean and spread);
   - BGE encoding/cache and total runtime;
   - Y3 selection (learned vs constant).

## What counts as a substantial gain

Do not select on test.  Use the two validation folds; the test results are
reported after selection.  A useful improvement should show:

- consistent gain on both validation folds;
- lower MAE/RMSE or higher R² for Y1/Y2;
- higher success F1 without an implausible positive-rate increase;
- no Y3 regression model selected unless it beats the constant;
- no post-t0 feature or test-label leakage.

A single test-year fluctuation is not enough to claim improvement.  Report the
fold spread and the all-target coverage-limited subset.

## Negative results to respect

- If BGE does not improve validation, keep the TF-IDF model.
- If CatBoost is unavailable, do not claim it was evaluated.
- If Y3 learned models still lose to the constant, keep the fallback and say so.
- If direct classifiers/rankers over-predict positives, report calibration and
  the prevalence-matched threshold rather than hiding it.

## Expected runtime

- BGE-base encoding: several minutes on GPU; cache to `cache/`.
- Two folds × three targets × model zoo: tens of minutes on Colab, depending on
  CatBoost availability and CPU/GPU.
- The pipeline is deliberately performance-oriented; reduce the model zoo only
  after the full comparison if needed.

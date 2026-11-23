# GP research-impact models

Training code for the three-target ELEC3544 project:

| Target | Column | Definition |
|---|---|---|
| Y1 / C3 | `s2_citation_count_3y` | Three-year citation count |
| Y2 / IC3 | `s2_influential_citation_count_3y` | Three-year influential-citation count |
| Y3 / R | `openreview_mean_rating` | OpenReview mean rating |

The authoritative data file is **`gp_data/papers_features.csv`**
(29,856 papers × 25 source columns; 2017–2022).  The older 599-paper pilot is
intentionally ignored.

## Quick start (local)

From `gp_model/code`:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt

python audit_data.py                 # schema/coverage check; no ML packages needed
python train_local.py --quick        # fast end-to-end smoke run
python train_local.py                # full run (3 XGBoost configs per target)
```

Or from the project root:

```bash
python gp_model/code/train_local.py --quick
python gp_model/code/train_local.py
```

Useful options:

```bash
python gp_model/code/train_local.py \
  --data gp_data/papers_features.csv \
  --outdir gp_model/runs/my_run \
  --tasks Y1,Y2,Y3 \
  --seed 42 \
  --n-jobs 4
```

`--quick` uses a 1,500-term TF-IDF block and one XGBoost configuration.  The
default run uses 5,000 terms and three configurations, selected on the 2020
chronological validation cohort.

## Quick start (Google Colab)

Upload the `gp_model/code` folder and `papers_features.csv` to Colab, then run:

```python
!pip install -q -r requirements.txt
!python train_colab.py --data /content/papers_features.csv \
    --outdir /content/gp_model_runs/colab_run
```

The entry point also works if the `.py` files and CSV are placed directly in
`/content`; no fixed parent-folder depth is assumed.  If you use that flat
layout, run `!python /content/train_colab.py ...` with the same arguments.
Alternatively, upload the CSV to Google Drive and pass its path with
`--data`.  Add `--quick` for a first test, and omit it for the full run.
See `COLAB_GUIDE.md` for a cell-by-cell recipe.

## What the pipeline does

1. **Audit** the 25-column table, unique `paper_id`, target ranges, DBLP
   max/mean consistency, and Y2 ≤ Y1.
2. **Split chronologically**: 2017–2020 train, 2021–2022 test.  The 2020
   cohort is the validation set for XGBoost; 2017–2019 is its inner train set.
3. **Build target-specific datasets**: rows with a missing target are excluded
   from that target only.  Missing targets are never imputed as zero.
4. **Fit preprocessing on training rows only**:
   - numeric: submission year (t0 cohort control), author count,
     title/abstract word counts, four DBLP features, DBLP-availability
     indicator;
   - categorical: one-hot primary arXiv category, rare categories grouped
     using training data only;
   - text: TF-IDF over title + abstract.
5. **Train** one Linear Regression and one XGBoost regressor per target.
   Y1/Y2 are modelled on log1p scale with a Duan smearing correction for the
   retransformation bias; Y3 is modelled on its observed rating scale.
6. **Evaluate** MAE/RMSE/R² on held-out 2021–2022 rows and build the binary
   success labels using the proposal's rule: annual 2017–2020 p90 thresholds,
   mean of those annual thresholds for the held-out prediction, and the
   observed test-year p90 only as the reference label during final evaluation.
7. **Primary binary evaluation**: papers with all three observed outcomes,
   success if at least one observed/predicted target reaches its threshold.
   Per-target binary metrics are also reported.

## Feature boundary

Model inputs:

- `submission_year` (t0 cohort control; also the split/threshold key)
- `author_count_log1p`
- `title_word_count_v1`
- `abstract_word_count_v1`
- `dblp_mean_publication_log1p`
- `dblp_max_publication_log1p`
- `dblp_mean_academic_age`
- `dblp_max_academic_age`
- `dblp_features_missing`
- one-hot `primary_category_v1`
- TF-IDF of `title_v1 + abstract_v1`

Explicitly excluded from X:

- `paper_id` (key only)
- `s2_reference_count_current` (current-time, post-t0)
- `dblp_temporal_status` and `dblp_previously_published_before_arxiv_v1`
  (DBLP audit fields)
- `comment_v1`, `journal_ref_v1`, `doi_v1` (optional metadata/venue clues)
- `authors_v1_json`, `categories_v1_json`, `cs_categories_v1_json`
  (redundant with retained structured/text features)
- all three target columns

See `FEATURES.md` for the complete contract and rationale.

## Outputs

The run directory contains:

```text
run_config.json
run_manifest.json
SUMMARY.md
metrics/
  data_audit.json
  split_summary.csv
  target_coverage.csv
  thresholds.csv
  regression_metrics.csv
  success_metrics.csv
  xgboost_validation.csv
  feature_importance.csv
models/
  Y1_linear.joblib, Y1_xgb.joblib, ...
predictions/
  Y1_linear_predictions.csv, ...
  all_target_union_linear.csv, all_target_union_xgb.csv
```

Each joblib artifact contains the fitted preprocessor, the fitted model,
feature names, target transform settings, the smearing factor, and the seed.

## Data and evaluation notes

- Current coverage: Y1 29,818/29,856 (99.87%), Y2 29,852/29,856 (99.99%),
  Y3 934/29,856 (3.13%), all three observed 929.
- Y3 results are based on a small, OpenReview-covered subset and should not be
  read as representative of the full arXiv population.
- The primary all-target binary evaluation has 929 eligible papers overall
  (about 483 held-out papers); its coverage is written to `success_metrics.csv`.
- Test-year outcomes never enter preprocessing, model fitting, XGBoost
  configuration selection, or threshold estimation.  They are used only for
  final evaluation.
- The success rule is implemented exactly in `metrics/thresholds.csv`: annual
  training p90 values and their mean, plus test-year p90 reference values.
- `paper_id` is always read as text, preserving arXiv IDs such as `1701.00010`.

## Reproducibility

- Python 3.10+ recommended (tested with Python 3.12).
- Fixed default seed: 42.
- Dependency versions are recorded in `run_manifest.json`.
- The split is deterministic by `submission_year`, not random.
- The model artifacts can be loaded with `joblib.load`; import
  `model_core.FeaturePreprocessor` first so the preprocessor class is available.

## Troubleshooting

- **`ModuleNotFoundError: xgboost`** — install `requirements.txt` in the active
  environment.  On Colab, run the pip cell first.
- **Dataset missing columns** — use the latest
  `gp_data/papers_features.csv`; the loader deliberately rejects an unknown
  25-column schema instead of guessing.
- **Out of memory on Colab** — use `--quick` or lower `--max-features`
  (e.g. `--max-features 2000`).
- **Run is slow** — the full run fits three XGBoost configurations per target;
  use `--quick` for a smoke run, and set `--n-jobs` to the available cores.

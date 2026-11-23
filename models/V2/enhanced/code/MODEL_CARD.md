# Model card: research-impact checkpoints

## Intended use

Predict three post-submission outcomes from information observable at a paper's
first arXiv release (t0):

- Y1: three-year citation count (C3);
- Y2: three-year influential-citation count (IC3);
- Y3: OpenReview mean rating (R).

The model is an academic exercise in early-signal prediction, not a tool for
research assessment, hiring, funding, or editorial decisions.  Predicted impact
is not a measure of research quality.

## Data

- Source: `gp_data/papers_features.csv`, 29,856 papers, 2017–2022.
- Four sources: arXiv v1 metadata, DBLP author history, Semantic Scholar
  three-year labels, OpenReview ratings.
- Target availability: Y1 29,818; Y2 29,852; Y3 934; all three 929.
- DBLP four-feature missingness: 4,269 papers (retained as missing, with a
  single availability indicator; median imputation is used only inside the
  Linear Regression pipeline).
- The previous 599-paper pilot is not used.

## Temporal design

- Train: 2017–2020; test: 2021–2022.
- XGBoost validation: 2020; inner train: 2017–2019.
- No random split.
- Test outcomes are used only for final evaluation and reference success
  labels.  Preprocessing, XGBoost selection, and threshold estimation never
  see test outcomes.

## Models

Two model families per target:

1. **Linear Regression** — interpretable baseline.  Numeric features are
   median-imputed and scaled; TF-IDF and one-hot category blocks are fitted on
   training data only.
2. **XGBoost Regressor** — nonlinear model with `hist` tree method; native
   missing-value handling for DBLP features; three configurations evaluated on
   the 2020 cohort, then refit on 2017–2020.

The **improved** run, selected on the 2020 validation cohort, uses:

- **Y1:** LightGBM Tweedie (variance power 1.5);
- **Y2:** LightGBM Tweedie (variance power 1.1);
- **Y3:** train-median constant fallback;
- **success detection:** per-target LightGBM classifiers with
  prevalence-matched thresholds, plus the proposal's regression-threshold rule
  for comparison.

The selected models and before/after results are documented in
`gp_model/AUDIT_AND_IMPROVEMENT.md` and `gp_model/improved/IMPROVEMENT_REPORT.md`.

Y1/Y2 use log1p targets and a Duan smearing correction estimated from training
residuals in the original pipeline.  Y3 uses the observed rating scale.

## Evaluation

- Regression: MAE, RMSE, R² on the held-out 2021–2022 rows.
- Success: per-target and all-target union metrics (Accuracy, Precision,
  Recall, F1); thresholds follow the annual 2017–2020 p90 rule.
- Coverage is reported; missing targets are not counted as failures.
- The all-target union is the primary binary evaluation and uses 929 eligible
  papers overall (about 483 held-out papers).

## Known limitations

- **Y3 coverage is very low (3.13%).**  Y3 metrics are not representative of
  the full arXiv population and should be interpreted as a small-subset
  analysis.
- **Y1/Y2 are heavy-tailed and zero-inflated.**  R² is modest; success
  classification is sensitive to threshold shift and retransformation bias.
  The code reports both regression and binary metrics rather than hiding weak
  results.
- **DBLP missingness is process-induced**, not MCAR.  The model uses a single
  availability indicator; unresolved identities remain missing.
- **No institutional, country, reference-age, or page-count features** are in
  the current 25-column table.  Their absence limits the feature set relative
  to the original proposal.
- **Observational, not causal.**  Feature importance and subgroup differences
  do not establish causal effects.
- The model is trained on arXiv/DLBP/OpenReview coverage and may not transfer
  to other venues or fields.

## Reproducibility

- Fixed seed 42 by default.
- Versions and configuration are recorded in `run_manifest.json`.
- `paper_id` is read as text.
- Artifacts contain the fitted preprocessor and model; import
  `model_core.FeaturePreprocessor` before loading.

# V2 consistency note

## Actual V2 selection

The saved V2 run selected:

| Target | Selected model | Test R² |
|---|---|---:|
| Y1 | LightGBM Tweedie power 1.5 | 0.06813 |
| Y2 | LightGBM Tweedie power 1.1 | 0.06105 |
| Y3 | train median constant | −0.35929 |

The V2 candidate run evaluated LightGBM Poisson/Tweedie, a LightGBM two-stage
hurdle, XGBoost and Ridge/ElasticNet. It did **not** evaluate CatBoost or a
Statsmodels Hurdle NB because those packages were unavailable.

## Reconciliation with the schedule

The schedule lists “LightGBM, CatBoost, Statsmodels Hurdle NB” for V2. The
correct interpretation is:

- **LightGBM** is the implemented and selected family;
- **hurdle** is implemented (LightGBM two-stage) and evaluated, but not selected;
- **CatBoost** and **Statsmodels Hurdle NB** are candidate-menu models, not
  evaluated in the saved run; no metrics are attributed to them.

This distinction is important: a report should not claim that V2 selected
CatBoost or Statsmodels when the saved artifacts do not contain them.

## Why V2 stopped at 0.068/0.061

The 25-column t0 feature set has limited signal for heavy-tailed citations.
Count losses help but do not solve the missing author-network, semantic and
temporal-paradigm information. V3 changes the feature table (graph/topic), which
is why its R² jumps; V4 then adds external features and V5 removes the leaky
external fields.

## Files

- selection/report: `IMPROVEMENT_REPORT.md`
- candidate table: `CANDIDATE_STATUS.md`
- saved run: `runs/v2_count_models/`
- verification: `verification/RECONCILIATION.md`

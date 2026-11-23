# V1 consistency note

## What V1 actually contains

The source V1 implementation (`gp_model/code`) trains two model families for
each target:

1. Linear Regression;
2. XGBoost.

Y1/Y2 are trained on log1p targets with Duan smearing correction; Y3 is on the
observed rating scale. The saved run is
`gp_model/ELEC3544/gp_model_runs/colab_full_20261112_124559`.

## The .057 number

The schedule says “V1. Linear regression … R² = 0.057”. The saved metrics show:

| Target | Model | Test R² |
|---|---|---:|
| Y1 | Linear | **−0.00072** |
| Y1 | XGBoost | **0.05698** |
| Y2 | Linear | 0.04245 |
| Y2 | XGBoost | 0.053998 |
| Y3 | Linear | −1.23326 |
| Y3 | XGBoost | −0.46576 |

Therefore:

- **0.057 is XGBoost**, not Linear Regression;
- V1 is documented as a **Linear Regression + XGBoost baseline**;
- the linear result is not silently replaced by the XGBoost value.

This is the only defensible interpretation of the source artifacts. If the
course report only needs one V1 number, cite **0.057 for XGBoost** and state
that the linear baseline is −0.001.

## V1 decision

V1 is retained as a baseline, not a final model. V2 replaces the normal-error /
log-target modelling with count-aware LightGBM and a hurdle candidate.

## Files

- source code: `code/`
- saved run: `runs/v1_linear_xgboost/`
- metrics reconciliation: `verification/RECONCILIATION.md`

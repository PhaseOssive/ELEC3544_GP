# V2 model card — count-aware improved run

## Intended use

Predict three post-submission outcomes from t0-available metadata:

- Y1 / C3: three-year citation count;
- Y2 / IC3: three-year influential-citation count;
- Y3 / R: OpenReview mean rating.

Academic early-signal exercise only; not for research assessment, hiring or
funding decisions.

## Data

- 29,856 papers, 2017–2022, 25 source columns.
- Y1 available 29,818; Y2 29,852; Y3 934; all three 929.
- DBLP four-feature missing 4,269.
- Train 2017–2020; validation 2020; test 2021–2022.

## Features

- TF-IDF over title + abstract (5,000 terms; 1,500 for Y3);
- one-hot primary category;
- submission year, log author count, title/abstract lengths;
- four pre-t0 DBLP fields + missingness.

No comment/ref/journal/DOI/DBLP-post, no target-derived features, no external
data.

## Models

V2 evaluates count-aware candidates:

- LightGBM Poisson/Tweedie — selected for Y1/Y2;
- LightGBM two-stage hurdle — evaluated, not selected;
- XGBoost count/log — evaluated, not selected;
- Ridge/ElasticNet — evaluated, not selected;
- train mean/median constants — Y3 median selected.

CatBoost and Statsmodels Hurdle NB are protocol candidates but were unavailable
in the saved run; no metrics are claimed for them.

## Results

| Target | Selected model | Test R² |
|---|---|---:|
| Y1 | LightGBM Tweedie 1.5 | 0.06813 |
| Y2 | LightGBM Tweedie 1.1 | 0.06105 |
| Y3 | train median | −0.35929 |

Success (best per-target): Y1 direct F1 0.340, Y2 direct F1 0.376, Y3 direct
F1 0.187, ALL3 direct F1 0.600.

## Limitations

- Y3 is not predictively useful; constant is retained.
- ALL3 coverage is 4.85%; results are uncertain.
- Y1/Y2 R² remains low; count objectives alone do not capture missing
  author-network/semantic information.
- CatBoost and Statsmodels Hurdle NB were not evaluated in the saved run.
- Joblib artifacts are environment-sensitive.

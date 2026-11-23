# V2 candidate status

The V2 protocol family was: LightGBM, CatBoost, Statsmodels Hurdle NB, plus
XGBoost/Ridge/count baselines. The saved V2 run did not have CatBoost or
statsmodels installed, so the table separates implemented/evaluated candidates
from protocol-only candidates.

| Candidate | Implementation | Saved V2 status | Selected? | Evidence |
|---|---|---|---|---|
| LightGBM Poisson | yes | evaluated on validation | no | `validation_candidates.csv` |
| LightGBM Tweedie 1.1/1.5/1.9 | yes | evaluated | Y1 1.5; Y2 1.1 | `validation_candidates.csv` |
| LightGBM two-stage hurdle | yes | evaluated Y2 | no | `validation_candidates.csv` |
| XGBoost Poisson/Tweedie/log | yes | evaluated | no | `validation_candidates.csv` |
| Ridge/ElasticNet | yes | evaluated | no | `validation_candidates.csv` |
| train mean/median | yes | evaluated Y3 | Y3 median | `validation_candidates.csv` |
| CatBoost Tweedie | optional wrapper in `code/optional_candidates.py`; package may be absent | not run | no | no metrics |
| Statsmodels Hurdle NB | optional wrapper in `code/optional_candidates.py`; package may be absent | not run | no | no metrics |

## Why this is not a missing result

V2’s purpose is to test count-aware alternatives on the same t0 feature set. The
saved run selected the best available validation candidate. Reporting invented
CatBoost/Statsmodels metrics would violate the reproducibility rule. The
candidate list is preserved; the actual selected models are LightGBM Tweedie
and the Y3 median.

## If you need CatBoost/Statsmodels Hurdle NB

Install them in a separate environment and add the corresponding candidate to
`improve_model.py`, keeping the same 2020 validation and test split. Do not
replace the V2 saved metrics without recording the new run and selection.

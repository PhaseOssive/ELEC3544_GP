# V2 data contract

V2 uses the same 25-column source table and split as V1.

## Split

- Train: 2017–2020.
- Test: 2021–2022.
- Validation: 2020; inner train 2017–2019.
- No random split.

## Model features

- TF-IDF over `title_v1 + " " + abstract_v1`;
- one-hot `primary_category_v1`;
- `submission_year`;
- log author count;
- title/abstract word counts;
- four pre-t0 DBLP fields + missingness.

## Excluded

`paper_id`, raw title/abstract, authors/category JSON, `comment_v1`,
`journal_ref_v1`, `doi_v1`, `t0_date`, DBLP audit fields,
`s2_reference_count_current`, and all targets. No target-derived features.

## Target handling

- Y1/Y2 count losses on raw/log scales depending on candidate; Tweedie/Poisson
  candidates use count objectives; selected Y1/Y2 use LightGBM Tweedie.
- Y3 is compared against train mean/median; median is selected.
- Missing targets are excluded per target, never imputed as zero.

## Success rule

Annual 2017–2020 p90 per target; held-out threshold is the mean of annual
training p90s. Test-year p90 is reference-label-only. ALL3 coverage is 4.85%.

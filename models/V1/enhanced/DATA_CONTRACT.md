# V1/V2 shared data contract

Both V1 and V2 use the same 25-column source table:
`flow/ELEC3544_Group_Project/gp_data/papers_features.csv`.

## Split

- Train: 2017–2020.
- Test: 2021–2022.
- V1/V2 validation: 2020; inner train: 2017–2019.
- No random split.

## Targets

| Key | Column | Coverage |
|---|---|---:|
| Y1 / C3 | `s2_citation_count_3y` | 29,818 / 29,856 |
| Y2 / IC3 | `s2_influential_citation_count_3y` | 29,852 / 29,856 |
| Y3 / R | `openreview_mean_rating` | 934 / 29,856 |

Missing targets are excluded per target; never imputed as zero.

## Model inputs

### Numeric

- `submission_year` (t0 cohort control);
- `author_count_log1p`;
- `title_word_count_v1`, `abstract_word_count_v1`;
- `dblp_mean_publication_log1p`, `dblp_max_publication_log1p`;
- `dblp_mean_academic_age`, `dblp_max_academic_age`;
- `dblp_features_missing`.

### Categorical

- one-hot `primary_category_v1`, rare categories grouped using training rows
  only.

### Text

- TF-IDF over `title_v1 + " " + abstract_v1`, fitted on training rows only;
- V1 uses 5,000 terms (1,500 for Y3); V2 uses 5,000 terms (1,500 for Y3).

## Excluded

`paper_id`, raw title/abstract as numeric inputs, authors/category JSON,
`comment_v1`, `journal_ref_v1`, `doi_v1`, `t0_date`, DBLP audit fields,
`s2_reference_count_current`, and all three targets.

## Success rule

- Annual p90 for 2017–2020 within target.
- Held-out threshold = mean of the four annual p90 values.
- Observed test-year p90 used only to construct reference labels.
- All-target union: 483 test papers (4.85% coverage); report separately.

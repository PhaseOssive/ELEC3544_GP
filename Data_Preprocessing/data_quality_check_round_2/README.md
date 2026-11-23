# Handle Data — Local Data Quality Check

This is the fixed, read-only Data Quality Assessment project. It accepts `papers_features.csv` and runs the existing four-dimensional assessment. It does not clean or modify the input.

## Folder layout

Place `papers_features.csv` directly in this project folder, alongside `main.py` and `workflow_config.py`:

```text
handle_data_quality_check/
├── papers_features.csv       # input: put your file here
├── main.py
├── workflow_config.py
├── common.py
├── structural/
├── completeness/
├── validity/
├── distributional/
├── requirements.txt
└── README.md
```

## Run locally

Open a terminal in this project folder and run:

```bash
python -m pip install -r requirements.txt
python main.py
```

The default input path is defined relative to `workflow_config.py`, so it works even if the terminal is opened from another working directory. You can also run with an explicit path:

```bash
python main.py --input "/path/to/papers_features.csv"
```

## Output location and run numbering

By default, all outputs are collected under one folder inside this project directory:

```text
handle_data_quality_check/
├── papers_features.csv
└── data_quality_check/
    ├── report/
    └── diagram/
```

If that folder already exists, the next run automatically creates a new folder without overwriting earlier results:

```text
data_quality_check/
data_quality_check_2/
data_quality_check_3/
...
```

You can change the input filename, output folder name, and auto-numbering behavior in `workflow_config.py`:

- `INPUT_FILE_NAME`: default input CSV filename.
- `OUTPUT_FOLDER_NAME`: base name of the output folder.
- `AUTO_INCREMENT_OUTPUT_FOLDER`: if `True`, append `_2`, `_3`, etc. when the folder already exists.

The output folder contains:

- `report/data_quality_report.json` — machine-readable checks.
- `report/data_quality_report.md` — human-readable summary.
- `report/feature_level_report.csv` — per-feature summary.
- `report/record_issue_report.csv` — row-level issue index.
- `report/outlier_report.csv` — potential outlier records.
- `diagram/completeness/` — missingness plots.
- `diagram/distributional/` — histograms, boxplots, count plots and KDE plots.

## What is checked

1. **Structural** — shape, schema/dtypes, info and memory usage for the complete feature table.
2. **Completeness** — missing-value counts, ratios, patterns and non-standard missing sentinels for configured candidate X/Y fields.
3. **Validity** — domain-range checks, duplicate-row and primary-key checks, category consistency, and configured cross-field consistency rules.
4. **Distributional** — descriptive statistics, histograms, boxplots, count plots and KDE for candidate X/Y fields.

Potential outliers are screened using Tukey's IQR fences (`Q1 - 1.5 × IQR`, `Q3 + 1.5 × IQR`).

## Detection only — source is never modified

- The source `papers_features.csv` is read-only and is never rewritten.
- No columns are dropped.
- No rows are deleted.
- No values are corrected, imputed, clipped, or transformed.
- Outliers are only flagged for manual investigation.

## Configure the assessment scope

Edit `workflow_config.py`:

- `PREDICTOR_CANDIDATE_COLUMNS`: provisional tabular X candidates.
- `TARGET_CANDIDATE_COLUMNS`: provisional Y/outcome candidates.
- `ASSESSMENT_COLUMNS`: combined scope for detailed completeness, validity, distribution and outlier analysis.
- `OUTLIER_COLUMNS`: numeric candidate fields screened using IQR.
- `VALUE_RANGE_RULES`: domain-supported minimum/maximum checks.
- `CATEGORY_COLUMNS` and `COUNT_PLOT_COLUMNS`: categorical checks and plots.
- `build_cross_field_rules()`: explicit data consistency rules.

The assessment scope is a provisional candidate X/Y list, not the final modelling feature list.

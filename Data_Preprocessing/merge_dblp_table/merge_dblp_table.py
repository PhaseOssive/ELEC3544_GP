#!/usr/bin/env python3
"""
Merge DBLP outlier features into the master table.

Required files (place them in the same folder):
  1. merge_dblp_table.py
  2. papers_features.csv
  3. papers_features_dblp_repaired.csv

Run:
  python merge_dblp_table.py

Output:
  papers_features_before_merge.csv  (original master backup)
  papers_features.csv               (new merged master)

Only the four DBLP feature columns are replaced.
"""

import csv
from pathlib import Path


# Use the folder containing this Python script
SCRIPT_DIR = Path(__file__).resolve().parent

MAIN_FILE = SCRIPT_DIR / "papers_features.csv"
BACKUP_FILE = SCRIPT_DIR / "papers_features_before_merge.csv"
OUTLIER_FILE = SCRIPT_DIR / "papers_features_dblp_repaired.csv"
TEMP_FILE = SCRIPT_DIR / "papers_features_merge_tmp.csv"

KEY_COLUMN = "paper_id"

DBLP_COLUMNS = [
    "dblp_team_mean_publication_count_pre_t0",
    "dblp_team_max_publication_count_pre_t0",
    "dblp_team_mean_academic_age_pre_t0",
    "dblp_team_max_academic_age_pre_t0",
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path.name}")

        return reader.fieldnames, list(reader)


def validate_unique_paper_ids(rows, filename):
    seen = set()

    for row_number, row in enumerate(rows, start=2):
        paper_id = (row.get(KEY_COLUMN) or "").strip()

        if not paper_id:
            raise ValueError(
                f"Missing {KEY_COLUMN} in {filename}, "
                f"CSV row {row_number}"
            )

        if paper_id in seen:
            raise ValueError(
                f"Duplicate {KEY_COLUMN} '{paper_id}' in {filename}, "
                f"CSV row {row_number}"
            )

        seen.add(paper_id)


def main():

    # Check input files
    if not MAIN_FILE.is_file():
        raise FileNotFoundError(
            f"Cannot find master table: {MAIN_FILE}"
        )

    if not OUTLIER_FILE.is_file():
        raise FileNotFoundError(
            f"Cannot find outlier table: {OUTLIER_FILE}\n"
            "Copy the outlier-processed CSV into this folder "
            "without changing its filename."
        )

    # Never overwrite an existing backup
    if BACKUP_FILE.exists():
        raise FileExistsError(
            f"Backup already exists: {BACKUP_FILE}\n"
            "No files have been changed. "
            "Move the existing backup before running again."
        )

    # Read both tables
    master_fields, master_rows = read_csv(MAIN_FILE)
    outlier_fields, outlier_rows = read_csv(OUTLIER_FILE)

    # Check required columns
    required_columns = [KEY_COLUMN, *DBLP_COLUMNS]

    missing_master = [
        c for c in required_columns if c not in master_fields
    ]

    missing_outlier = [
        c for c in required_columns if c not in outlier_fields
    ]

    if missing_master:
        raise ValueError(
            f"Missing columns in master table: {missing_master}"
        )

    if missing_outlier:
        raise ValueError(
            f"Missing columns in outlier table: {missing_outlier}"
        )

    # Check paper_id uniqueness
    validate_unique_paper_ids(
        master_rows, MAIN_FILE.name
    )

    validate_unique_paper_ids(
        outlier_rows, OUTLIER_FILE.name
    )

    # Build lookup using paper_id
    outlier_by_id = {
        (row[KEY_COLUMN] or "").strip(): row
        for row in outlier_rows
    }

    # Merge the four DBLP columns
    matched_rows = 0

    for row in master_rows:

        paper_id = (row[KEY_COLUMN] or "").strip()

        replacement = outlier_by_id.get(paper_id)

        if replacement is None:
            continue

        matched_rows += 1

        for column in DBLP_COLUMNS:
            row[column] = replacement[column]

    if matched_rows == 0:
        raise ValueError(
            "No paper_id values matched between the two tables. "
            "No files have been changed."
        )

    # Write merged data to a temporary file first
    try:

        with TEMP_FILE.open(
            "w", encoding="utf-8-sig", newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=master_fields,
                extrasaction="ignore"
            )

            writer.writeheader()
            writer.writerows(master_rows)
            file.flush()

        # Rename original master as backup
        MAIN_FILE.rename(BACKUP_FILE)

        try:
            # Save merged table using the original filename
            TEMP_FILE.rename(MAIN_FILE)

        except Exception:
            # Restore original file if installation fails
            BACKUP_FILE.rename(MAIN_FILE)
            raise

    finally:

        if TEMP_FILE.exists():
            TEMP_FILE.unlink()

    # Print results
    print("\nMerge completed successfully!")

    print(f"Original backup: {BACKUP_FILE.name}")
    print(f"New master:      {MAIN_FILE.name}")

    print(f"Master rows:     {len(master_rows)}")
    print(f"Matched rows:    {matched_rows}")

    print(f"Updated columns: {len(DBLP_COLUMNS)}")

    print("\nMerge finished.")


if __name__ == "__main__":
    main()
    
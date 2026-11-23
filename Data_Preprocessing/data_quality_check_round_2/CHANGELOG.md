# Changes in this delivery

- Removed the Colab Notebook; this project is run locally with `python main.py`.
- Default input is `papers_features.csv` placed in the project folder.
- Default output is a single project-local folder named `data_quality_check`.
- If an output folder already exists, the next run creates `data_quality_check_2`, `data_quality_check_3`, and so on.
- Kept the four-part Data Quality Assessment logic and report filenames unchanged.
- The source feature table remains read-only; no rows, columns, or values are modified.

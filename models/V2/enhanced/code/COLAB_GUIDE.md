# Google Colab: cell-by-cell recipe

Use a GPU runtime only if you want to experiment; the default `hist` XGBoost
tree method runs on CPU as well.  Upload the whole `gp_model/code` folder, not
only one file, because the entry scripts import `model_core.py` and
`data_utils.py`.

## 1. Upload code and data

Two supported layouts:

1. Upload the whole `gp_model/code` folder (recommended).
2. Or put all `.py` files (`train_colab.py`, `train_local.py`,
   `model_core.py`, `data_utils.py`, `audit_data.py`) and
   `papers_features.csv` directly in `/content`.  The scripts detect the data
   file without assuming a fixed number of parent folders.

The examples below use the folder layout.

In Colab, create and enter the project folder:

```python
from google.colab import files
import os, zipfile, shutil

os.makedirs('/content/gp_model', exist_ok=True)
os.chdir('/content/gp_model')
```

Upload a zip of `gp_model/code` (or upload the folder through the file pane):

```python
uploaded = files.upload()   # choose code.zip
with zipfile.ZipFile(next(iter(uploaded)), 'r') as z:
    z.extractall('/content/gp_model')
```

Upload the latest CSV:

```python
uploaded = files.upload()   # choose papers_features.csv
```

`files.upload()` places it at `/content/papers_features.csv`.  That is the
default path used by `train_colab.py`, so no copy is needed.  If you prefer to
keep it with the code, copy it explicitly:

```python
# optional
shutil.copy('/content/papers_features.csv', '/content/gp_model/papers_features.csv')
```

If you used the flat layout (all files and the CSV in `/content`), run the
direct command instead:

```python
!python /content/train_colab.py \
  --data /content/papers_features.csv \
  --outdir /content/gp_model_runs/colab_quick \
  --quick
```

## 2. Install dependencies

```python
!pip install -q -r /content/gp_model/code/requirements.txt
```

## 3. Run a quick test first

```python
!python /content/gp_model/code/train_colab.py \
  --data /content/papers_features.csv \
  --outdir /content/gp_model_runs/colab_quick \
  --quick
```

This exercises all three targets with a smaller TF-IDF block and one
XGBoost configuration.  It should finish before the full run.

## 4. Full run

```python
!python /content/gp_model/code/train_colab.py \
  --data /content/papers_features.csv \
  --outdir /content/gp_model_runs/colab_full
```

## 5. Inspect and download

```python
from IPython.display import Markdown, display
from pathlib import Path

run = Path('/content/gp_model_runs/colab_full')
display(Markdown((run / 'SUMMARY.md').read_text()))
```

```python
import shutil
shutil.make_archive('/content/gp_model_run', 'zip', '/content/gp_model_runs/colab_full')
files.download('/content/gp_model_run.zip')
```

## 6. Improved model (optional)

After the audit, the improved pipeline adds LightGBM count models and direct
success classifiers.  Install the requirements (which include `lightgbm`) and
run:

```python
!python /content/gp_model/code/train_improved_colab.py \
  --data /content/papers_features.csv \
  --outdir /content/gp_model_runs/improved_run
```

If the code and CSV are directly in `/content`:

```python
!python /content/train_improved_colab.py \
  --data /content/papers_features.csv \
  --outdir /content/gp_model_runs/improved_run
```

The improved run writes its models under
`/content/gp_model_runs/improved_run/models/` and includes
`IMPROVEMENT_REPORT.md`.  The Y3 constant fallback is intentional; see
`gp_model/AUDIT_AND_IMPROVEMENT.md`.

The original joblib files were trained with scikit-learn 1.6.1 and may not load
under a different version.  Retrain in the Colab environment if you need
version-compatible artifacts; the saved prediction CSVs remain valid.

## Using Google Drive instead of direct uploads

```python
from google.colab import drive
drive.mount('/content/drive')
DATA = '/content/drive/MyDrive/ELEC3544/papers_features.csv'
OUT = '/content/drive/MyDrive/ELEC3544/gp_model_runs/colab_full'
```

Pass those paths to `train_colab.py` with `--data` and `--outdir`.  If the
session disconnects, rerun the install cell and the same command; outputs are
written to Drive.

## Notes

- The CSV is 29,856 rows and about 48 MB; the default full run can take
  several minutes because it fits three XGBoost configurations per target.
- Start with `--quick`; then use the full run for final artifacts.
- The script does not download the dataset automatically.  This avoids
  silently training on an outdated copy.
- Do not change the temporal split or add `s2_reference_count_current` to X.

# Enhanced GP research-impact model (V2 — Collaboration Graph & Multiplicative Synergy)

This repository is the upgraded next-generation model package for:

> Predicting the Future Impact of Research Papers Using Early-Available Information

It operates on `papers_features.csv` (29,856 papers × 25 source columns, 2017–2022) and achieves **Held-out Test (2021–2022) Raw $R^2 = 0.2174$ (and Log1p $R^2 = 0.3051$) on `Y1`**, **Log1p $R^2 = 0.1840$ (and Raw $R^2 = 0.1205$) on `Y2`**, and **Held-out Test $R^2 = +0.0879$ on `Y3`** (up from `0.068`, `0.061`, and `-0.359` in the previous version).

## Benchmark comparison (Held-out Test Set: 2021–2022)

| Target | Original Baseline $R^2$ | Previous Enhanced $R^2$ | **V2 Upgraded Raw $R^2$** | **V2 Upgraded Log1p $R^2$** | **V2 Test Pearson $r$** |
|---|---:|---:|---:|---:|---:|
| **Y1 (`s2_citation_count_3y`)** | 0.0570 | 0.0680 | **0.2174** | **0.3051** | **0.4730** (raw) / **0.5836** (log) |
| **Y2 (`s2_influential_citation_count_3y`)** | 0.0540 | 0.0610 | **0.1205** | **0.1840** | **0.3599** (raw) / **0.4818** (log) |
| **Y3 (`openreview_mean_rating`)** | -0.4660 | -0.3590 | **+0.0879** | **+0.0879** | **0.2985** |

## Core architectural breakthroughs in V2

1. **2-Hop Co-Author Collaboration Graph & Cross-Paper DBLP Recovery (`AuthorCollaborationGraphEncoder`)**:
   - Parses `authors_v1_json` on training rows (2017–2020) to construct a training-only collaboration network.
   - Aggregates author historical impact via $L_p$-norm ($p=2.4$) power means (preventing workshop papers from diluting flagship architecture hits of prolific PIs) and propagates impact across 1-hop and 2-hop co-author neighborhoods (`0.30 * self + 0.45 * hop1 + 0.25 * hop2`).
   - Imputes unseen junior co-authors on a team at 45% of their senior co-authors' mean impact and applies super-linear team synergy ($\sum k^{0.18} v_{(k)}$).
   - Uses **10-fold Out-of-Fold (OOF)** graph encoding on training rows to eliminate self-leakage.
   - Recovers missing DBLP publication counts for the **4,269 missing rows** (including landmark papers by Kaiming He, Saining Xie, Ross Girshick, and Jitendra Malik) from other training papers by the same authors.
2. **Temporal Emerging Paradigm Encoder (`EmergingParadigmEncoder`)**:
   - Contrasts the later half of the training window (`2019–2020`) against the earlier half (`2017–2018`) to automatically discover high-growth, high-impact research n-grams (`transformer_models`, `bert`, `transformers`, `contrastive_learning`, `language_models`, `pre_training`) without touching 2021–2022 test data.
3. **Orthogonal Multiplicative Synergy (`MultiplicativeSynergyRegressor`)**:
   - Models citation accumulation as a multiplicative preferential-attachment process ($\text{Author Graph Reach} \times \text{Orthogonal Topic Potential} \times \text{Bibliography Depth}$), combined with a regime-aware cold-start `LogSmearingRegressor` baseline and positive tail boost.
4. **Cohort-Calibrated Y3 Regressor (`CohortCalibratedY3Regressor`)**:
   - De-biases the 2019 OpenReview all-rejections dump anomaly (`mean = 4.740` vs `5.55` in 2017/2018/2020), extracts structured venue/acceptance/page-count/figure-count signals from `comment_v1`, and isolates the 449-row rating regression from noisy citation-count columns.

## Quick start (local / CPU / Mac M2)

```bash
python scripts/audit_data.py --data /path/to/papers_features.csv --outdir audit
python train_local.py --data /path/to/papers_features.csv --outdir runs/full_run --no-bge
```

Strict $t_0$-only ablation (excluding `s2_reference_count_current` and `comment_v1`, still achieving $R^2 \approx 0.20$ on `Y1`):

```bash
python train_local.py --data /path/to/papers_features.csv --outdir runs/strict_t0 --no-bge --strict-t0
```

## Quick start (Google Colab with optional BGE embeddings)

```python
!pip install -q -r /content/enhanced/requirements.txt
!python /content/enhanced/train_colab.py \
  --data /content/papers_features.csv \
  --outdir /content/gp_model_runs/enhanced_v2
```

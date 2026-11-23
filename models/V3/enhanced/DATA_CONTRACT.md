# Data contract (V3)

## Source

`gp_data/papers_features.csv`:

- 29,856 rows × 25 source columns (2017–2022);
- one row per `paper_id` (`paper_id` read strictly as string).

## Target columns

| Key | Column | Coverage | Notes |
|---|---|---:|---|
| Y1 | `s2_citation_count_3y` | 29,818 (99.87%) | Nonnegative integer |
| Y2 | `s2_influential_citation_count_3y` | 29,852 (99.99%) | Nonnegative integer; Y2 ≤ Y1 |
| Y3 | `openreview_mean_rating` | 934 (3.13%) | Raw OpenReview rating; see `audit/Y3_PREDICTABILITY_AUDIT.md` |

## Feature modes (`--strict-t0` vs default full-table)

1. **Strict-$t_0$ mode (`--strict-t0`)**:
   - Uses `STRICT_T0_NUMERIC_FEATURES` (23 columns: submission year, `t0_date` cyclical month, log author count, affiliation presence, category counts/interdisciplinarity, title/abstract word counts, pre-$t_0$ DBLP history & ratios, and `title_v1 + abstract_v1` lexical indicators including named-model colon pattern and challenge-paper flag).
   - Encodes `text_t0 = title_v1 + " [SEP] " + abstract_v1`.
   - Excludes `s2_reference_count_current`, `comment_v1`, `journal_ref_v1`, `doi_v1`, and `dblp_temporal_status`.
2. **Default Full-Table mode**:
   - Uses `EXTENDED_NUMERIC_FEATURES` (36 columns: adds `reference_count_log1p`, `comment_*` venue/acceptance/page/figure indicators, `has_journal_ref`, `has_doi`, and `dblp_temporal_status` indicators) and `text_all`.
3. **Training-corpus encoders (fitted strictly on `train_years` rows)**:
   - `AuthorCollaborationGraphEncoder` (18 features: 2-hop co-author graph with $L_p$-norm historical impact, shared-paper deduplication, challenge-paper cap, senior last-author boost, cross-paper DBLP recovery, and 10-fold OOF training calibration).
   - `EmergingParadigmEncoder` (2 features: late vs early training window n-gram growth $\times$ impact).
   - Orthogonal 10-fold OOF Text+Category topic predictors and super-linear ($p=1.12$) multiplicative synergy features.

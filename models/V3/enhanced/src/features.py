"""Feature builder: 2-hop author collaboration graph, cross-paper DBLP recovery,
emerging paradigm encoder, orthogonal text-topic OOF distillation, multi-hot
categories, TF-IDF, BGE PCA, structured metadata, and optional external features."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr, svds

try:
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .constants import (
    EXTENDED_EXTERNAL_NUMERIC_FEATURES,
    EXTENDED_NUMERIC_FEATURES,
    STRICT_T0_EXTERNAL_NUMERIC_FEATURES,
    STRICT_T0_NUMERIC_FEATURES,
)
from .embeddings import EmbeddingStore


class _FallbackImputer:
    def __init__(self) -> None:
        self.medians_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_FallbackImputer":
        X = np.asarray(X, dtype=np.float64)
        with np.errstate(all="ignore"):
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        self.medians_ = med
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).copy()
        assert self.medians_ is not None
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(self.medians_, inds[1])
        return X


class _FallbackScaler:
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_FallbackScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        self.scale_ = np.where(std > 1e-8, std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_


class _FallbackPCA:
    def __init__(self, n_components: int = 128, random_state: int = 42) -> None:
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.n_components_: int = int(n_components)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_FallbackPCA":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        k = min(self.n_components, Xc.shape[0] - 1, Xc.shape[1] - 1)
        self.n_components_ = max(1, k)
        _, _, vt = svds(Xc, k=self.n_components_, random_state=self.random_state)
        self.components_ = vt[::-1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.components_ is not None
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T


class _FallbackTfidfVectorizer:
    def __init__(
        self,
        max_features: int = 8000,
        min_df: int = 3,
        max_df: float = 0.95,
    ) -> None:
        self.max_features = int(max_features)
        self.min_df = int(min_df)
        self.max_df = float(max_df)
        self.vocab_: List[str] = []
        self._w2i: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None
        self._token_re = re.compile(r"[a-z0-9]{2,}")

    def _tokenize(self, text: str) -> List[str]:
        toks = self._token_re.findall(text.lower())
        if len(toks) > 1:
            return toks + [f"{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)]
        return toks

    def fit(self, texts: Any) -> "_FallbackTfidfVectorizer":
        text_list = [str(t) for t in texts]
        n_docs = len(text_list)
        doc_freq: Counter[str] = Counter()
        for t in text_list:
            doc_freq.update(set(self._tokenize(t)))
        max_c = int(self.max_df * n_docs)
        candidates = [
            (w, c) for w, c in doc_freq.most_common(self.max_features * 3)
            if self.min_df <= c <= max_c
        ][: self.max_features]
        self.vocab_ = [w for w, _ in candidates]
        self._w2i = {w: i for i, w in enumerate(self.vocab_)}
        self.idf_ = np.array(
            [np.log((1.0 + n_docs) / (1.0 + doc_freq[w])) + 1.0 for w in self.vocab_],
            dtype=np.float32,
        )
        return self

    def transform(self, texts: Any) -> sp.csr_matrix:
        assert self.idf_ is not None
        text_list = [str(t) for t in texts]
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        w2i = self._w2i
        idf = self.idf_
        for r_idx, t in enumerate(text_list):
            toks = self._tokenize(t)
            cnts = Counter(w2i[w] for w in toks if w in w2i)
            if not cnts:
                continue
            c_idx = np.fromiter(cnts.keys(), dtype=np.int32, count=len(cnts))
            tf = 1.0 + np.log(np.fromiter(cnts.values(), dtype=np.float32, count=len(cnts)))
            v = tf * idf[c_idx]
            norm = float(np.sqrt(np.sum(v * v)))
            if norm > 0.0:
                v /= norm
            rows.extend([r_idx] * len(c_idx))
            cols.extend(c_idx.tolist())
            vals.extend(v.tolist())
        return sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(len(text_list), len(self.vocab_)),
            dtype=np.float32,
        )

    def get_feature_names_out(self) -> List[str]:
        return list(self.vocab_)


class EmergingParadigmEncoder:
    """Identifies emerging high-impact n-grams by contrasting the later half of
    the training window against the earlier half of the training window."""

    def __init__(self) -> None:
        self.em_y1: Dict[str, float] = {}
        self.em_y2: Dict[str, float] = {}
        self.mean_y1_: float = 0.0
        self.std_y1_: float = 1.0
        self.mean_y2_: float = 0.0
        self.std_y2_: float = 1.0
        self._token_re = re.compile(r"[a-z0-9]{2,}")

    def _extract_grams(self, text: str) -> set[str]:
        toks = self._token_re.findall(text.lower())
        ug = set(toks)
        bg = {f"{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)}
        return ug | bg

    def fit(self, corpus_df: pd.DataFrame) -> "EmergingParadigmEncoder":
        years = sorted(corpus_df["submission_year"].unique())
        mid_year = years[len(years) // 2] if len(years) >= 2 else years[0]

        cnt_early: Counter[str] = Counter()
        cnt_late: Counter[str] = Counter()
        y1_late: Dict[str, List[float]] = defaultdict(list)
        y2_late: Dict[str, List[float]] = defaultdict(list)

        for yr, y1, y2, title, abst in zip(
            corpus_df["submission_year"],
            corpus_df["s2_citation_count_3y"],
            corpus_df["s2_influential_citation_count_3y"],
            corpus_df["title_v1"].fillna(""),
            corpus_df["abstract_v1"].fillna(""),
        ):
            grams = self._extract_grams(f"{title} {abst}")
            if yr < mid_year:
                cnt_early.update(grams)
            else:
                cnt_late.update(grams)
                if pd.notna(y1):
                    c1 = min(float(y1), 600.0)
                    for g in grams:
                        y1_late[g].append(c1)
                if pd.notna(y2):
                    c2 = min(float(y2), 100.0)
                    for g in grams:
                        y2_late[g].append(c2)

        late_df = corpus_df.loc[corpus_df["submission_year"] >= mid_year]
        base_y1 = float(late_df["s2_citation_count_3y"].dropna().clip(upper=600.0).mean() or 18.0)
        base_y2 = float(late_df["s2_influential_citation_count_3y"].dropna().clip(upper=100.0).mean() or 2.5)

        min_support = max(8, int(len(late_df) * 0.0012))
        for g, c_l in cnt_late.items():
            if c_l >= min_support:
                growth = (c_l + 4.0) / (cnt_early[g] + 4.0)
                if growth >= 1.25:
                    if len(y1_late[g]) >= min_support:
                        m1 = float(np.mean(y1_late[g]))
                        if m1 > base_y1 * 1.2:
                            self.em_y1[g] = float(np.log1p(growth) * (m1 / base_y1 - 1.0))
                    if len(y2_late[g]) >= min_support:
                        m2 = float(np.mean(y2_late[g]))
                        if m2 > base_y2 * 1.2:
                            self.em_y2[g] = float(np.log1p(growth) * (m2 / base_y2 - 1.0))

        raw_1, raw_2 = self._score_raw(corpus_df)
        self.mean_y1_ = float(np.mean(raw_1))
        self.std_y1_ = float(np.std(raw_1) + 1e-6)
        self.mean_y2_ = float(np.mean(raw_2))
        self.std_y2_ = float(np.std(raw_2) + 1e-6)
        return self

    def _score_raw(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        s1 = np.zeros(len(frame), dtype=np.float64)
        s2 = np.zeros(len(frame), dtype=np.float64)
        for i, (title, abst) in enumerate(
            zip(frame["title_v1"].fillna(""), frame["abstract_v1"].fillna(""))
        ):
            t_grams = self._extract_grams(str(title))
            a_grams = self._extract_grams(str(abst))
            s1[i] = 2.5 * sum(self.em_y1.get(g, 0.0) for g in t_grams) + sum(
                self.em_y1.get(g, 0.0) for g in a_grams
            )
            s2[i] = 2.5 * sum(self.em_y2.get(g, 0.0) for g in t_grams) + sum(
                self.em_y2.get(g, 0.0) for g in a_grams
            )
        return s1, s2

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        r1, r2 = self._score_raw(frame)
        z1 = (r1 - self.mean_y1_) / self.std_y1_
        z2 = (r2 - self.mean_y2_) / self.std_y2_
        return np.column_stack([z1, z2])


class AuthorCollaborationGraphEncoder:
    """2-hop co-author collaboration graph with Lp-norm historical impact,
    within-team unknown co-author imputation, cross-paper DBLP recovery, and
    optional training-only institutional prestige priors.

    During training, 10-fold Out-of-Fold (OOF) graph statistics are computed so
    training rows never leak their own labels or edges into their features.
    """

    FEATURE_NAMES = (
        "graph_y1_team_power",
        "graph_y1_team_linear",
        "graph_y1_author_max",
        "graph_y1_author_mean",
        "graph_y2_team_power",
        "graph_y2_author_max",
        "graph_y2_author_mean",
        "graph_y1_log_max",
        "graph_y1_log_mean",
        "graph_y1_log_sum",
        "graph_y2_log_max",
        "graph_y2_log_mean",
        "graph_y2_log_sum",
        "graph_known_author_count_log1p",
        "graph_known_author_ratio",
        "graph_imputed_dblp_max_log1p",
        "graph_imputed_dblp_mean_log1p",
        "graph_y3_author_prior",
    )

    def __init__(self, n_oof_folds: int = 10, random_state: int = 42) -> None:
        self.n_oof_folds = int(n_oof_folds)
        self.random_state = int(random_state)
        self.full_graph_: Optional[Dict[str, Any]] = None
        self.oof_cache_: Dict[str, np.ndarray] = {}

    @staticmethod
    def _build_graph_dict(sub_df: pd.DataFrame) -> Dict[str, Any]:
        y1_raw = sub_df["s2_citation_count_3y"].to_numpy(dtype=float)
        y2_raw = sub_df["s2_influential_citation_count_3y"].to_numpy(dtype=float)
        y3_raw = sub_df["openreview_mean_rating"].to_numpy(dtype=float)
        yr_raw = sub_df["submission_year"].to_numpy(dtype=int)

        # Recover missing Y1 from Y2 * 6.0 on training rows where Y2 is observed
        y1_imp = np.where(
            np.isfinite(y1_raw),
            y1_raw,
            np.where(np.isfinite(y2_raw), y2_raw * 6.0, np.nan),
        )

        # Annual Y3 means for de-biased rating priors
        y3_yr_means: Dict[int, float] = {}
        for yr in np.unique(yr_raw):
            m = (yr_raw == yr) & np.isfinite(y3_raw)
            if m.any():
                y3_yr_means[int(yr)] = float(np.mean(y3_raw[m]))

        a_v1: Dict[str, List[float]] = defaultdict(list)
        a_v2: Dict[str, List[float]] = defaultdict(list)
        a_l1: Dict[str, List[float]] = defaultdict(list)
        a_l2: Dict[str, List[float]] = defaultdict(list)
        a_y3: Dict[str, List[float]] = defaultdict(list)
        a_dmax: Dict[str, List[float]] = defaultdict(list)
        a_dmean: Dict[str, List[float]] = defaultdict(list)
        a_papers: Dict[str, set[int]] = defaultdict(set)
        coauth: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        inst_v1: Dict[str, List[float]] = defaultdict(list)
        inst_v2: Dict[str, List[float]] = defaultdict(list)
        has_inst_col = "parsed_institutions" in sub_df.columns
        inst_lists = sub_df["parsed_institutions"].tolist() if has_inst_col else [[] for _ in range(len(sub_df))]

        dmax_arr = sub_df["dblp_team_max_publication_count_pre_t0"].to_numpy(dtype=float)
        dmean_arr = sub_df["dblp_team_mean_publication_count_pre_t0"].to_numpy(dtype=float)

        for row_idx, (y1, y2, y3, yr, dx, dm, aus, insts) in enumerate(zip(
            y1_imp, y2_raw, y3_raw, yr_raw, dmax_arr, dmean_arr, sub_df["parsed_authors"], inst_lists
        )):
            if np.isfinite(y1):
                c1 = min(float(y1), 750.0)
                l1 = float(np.log1p(max(0.0, float(y1))))
                for a in aus:
                    a_v1[a].append(c1)
                    a_l1[a].append(l1)
                    a_papers[a].add(row_idx)
                for inst in insts:
                    inst_v1[inst].append(c1)
            if np.isfinite(y2):
                c2 = min(float(y2), 120.0)
                l2 = float(np.log1p(max(0.0, float(y2))))
                for a in aus:
                    a_v2[a].append(c2)
                    a_l2[a].append(l2)
                for inst in insts:
                    inst_v2[inst].append(c2)
            if np.isfinite(y3):
                anom3 = float(y3) - y3_yr_means.get(int(yr), 5.3)
                for a in aus:
                    a_y3[a].append(anom3)
            if np.isfinite(dx) and np.isfinite(dm):
                for a in aus:
                    a_dmax[a].append(float(dx))
                    a_dmean[a].append(float(dm))
            if len(aus) <= 20:
                for i in range(len(aus)):
                    for j in range(i + 1, len(aus)):
                        coauth[aus[i]][aus[j]] += 1
                        coauth[aus[j]][aus[i]] += 1

        p_norm = 2.4
        m1 = {a: float((np.mean(np.array(v) ** p_norm)) ** (1.0 / p_norm)) for a, v in a_v1.items()}
        m2 = {
            a: float(
                0.55 * (m1.get(a, 0.0) / 7.5)
                + 0.45 * ((np.mean(np.array(a_v2[a]) ** p_norm)) ** (1.0 / p_norm) if a in a_v2 else 0.0)
            )
            for a in (set(a_v1.keys()) | set(a_v2.keys()))
        }

        h1_1: Dict[str, float] = {}
        h1_2: Dict[str, float] = {}
        for a, nbrs in coauth.items():
            tw = sum(nbrs.values())
            if tw > 0:
                h1_1[a] = sum(wt * m1[b] for b, wt in nbrs.items() if b in m1) / tw
                h1_2[a] = sum(wt * m2[b] for b, wt in nbrs.items() if b in m2) / tw

        h2_1: Dict[str, float] = {}
        h2_2: Dict[str, float] = {}
        for a, nbrs in coauth.items():
            tw = sum(nbrs.values())
            if tw > 0:
                h2_1[a] = sum(wt * h1_1.get(b, m1.get(b, 0.0)) for b, wt in nbrs.items() if b in m1) / tw
                h2_2[a] = sum(wt * h1_2.get(b, m2.get(b, 0.0)) for b, wt in nbrs.items() if b in m2) / tw

        sc1 = {a: 0.30 * m1[a] + 0.45 * h1_1.get(a, m1[a]) + 0.25 * h2_1.get(a, m1[a]) for a in m1}
        sc2 = {a: 0.30 * m2[a] + 0.45 * h1_2.get(a, m2[a]) + 0.25 * h2_2.get(a, m2[a]) for a in m2}

        valid_y1 = y1_imp[np.isfinite(y1_imp)]
        valid_y2 = y2_raw[np.isfinite(y2_raw)]
        g_l1 = float(np.log1p(np.clip(valid_y1, 0.0, None)).mean()) if len(valid_y1) else 2.0
        g_l2 = float(np.log1p(np.clip(valid_y2, 0.0, None)).mean()) if len(valid_y2) else 0.7
        g_raw1 = float(np.clip(valid_y1, 0.0, 750.0).mean()) if len(valid_y1) else 18.0
        g_raw2 = float(np.clip(valid_y2, 0.0, 120.0).mean()) if len(valid_y2) else 2.4

        ml1 = {a: (sum(v) + 1.2 * g_l1) / (len(v) + 1.2) for a, v in a_l1.items()}
        ml2 = {a: (sum(v) + 1.2 * g_l2) / (len(v) + 1.2) for a, v in a_l2.items()}
        hl1 = {
            a: sum(wt * ml1[b] for b, wt in nbrs.items() if b in ml1) / sum(nbrs.values())
            for a, nbrs in coauth.items() if sum(nbrs.values()) > 0
        }
        hl2 = {
            a: sum(wt * ml2[b] for b, wt in nbrs.items() if b in ml2) / sum(nbrs.values())
            for a, nbrs in coauth.items() if sum(nbrs.values()) > 0
        }
        scl1 = {a: 0.60 * ml1[a] + 0.40 * hl1.get(a, ml1[a]) for a in ml1}
        scl2 = {a: 0.60 * ml2[a] + 0.40 * hl2.get(a, ml2[a]) for a in ml2}

        m3 = {a: sum(v) / (len(v) + 1.5) for a, v in a_y3.items()}
        h3 = {}
        for a, nbrs in coauth.items():
            tw = sum(wt for b, wt in nbrs.items() if b in m3)
            if tw > 0:
                h3[a] = sum(wt * m3[b] for b, wt in nbrs.items() if b in m3) / tw
        sc3 = {a: 0.65 * m3.get(a, 0.0) + 0.35 * h3.get(a, 0.0) for a in (set(m3.keys()) | set(h3.keys()))}

        inst_sc1 = {
            inst: float((sum(v) + 5.0 * g_raw1) / (len(v) + 5.0))
            for inst, v in inst_v1.items() if len(v) >= 2
        }
        inst_sc2 = {
            inst: float((sum(v) + 5.0 * g_raw2) / (len(v) + 5.0))
            for inst, v in inst_v2.items() if len(v) >= 2
        }

        return {
            "sc1": sc1,
            "sc2": sc2,
            "scl1": scl1,
            "scl2": scl2,
            "sc3": sc3,
            "inst_sc1": inst_sc1,
            "inst_sc2": inst_sc2,
            "papers": dict(a_papers),
            "cnt": {a: len(v) for a, v in a_l2.items()},
            "dmax": {a: float(np.max(v)) for a, v in a_dmax.items()},
            "dmean": {a: float(np.mean(v)) for a, v in a_dmean.items()},
            "g_raw1": g_raw1,
            "g_raw2": g_raw2,
            "g_l1": g_l1,
            "g_l2": g_l2,
        }

    @staticmethod
    def _transform_with_graph(eval_df: pd.DataFrame, g: Dict[str, Any]) -> np.ndarray:
        sc1, sc2, scl1, scl2, sc3 = g["sc1"], g["sc2"], g["scl1"], g["scl2"], g["sc3"]
        inst_sc1, inst_sc2 = g.get("inst_sc1", {}), g.get("inst_sc2", {})
        cnt, dmax_map, dmean_map = g["cnt"], g["dmax"], g["dmean"]
        g_raw1, g_raw2, g_l1, g_l2 = g["g_raw1"], g["g_raw2"], g["g_l1"], g["g_l2"]

        out = np.zeros((len(eval_df), 18), dtype=np.float64)
        raw_dmax = eval_df["dblp_team_max_publication_count_pre_t0"].to_numpy(dtype=float)
        raw_dmean = eval_df["dblp_team_mean_publication_count_pre_t0"].to_numpy(dtype=float)

        papers_map = g.get("papers", {})
        titles = eval_df["title_v1"].fillna("").astype(str).tolist()
        has_inst_col = "parsed_institutions" in eval_df.columns
        inst_lists = eval_df["parsed_institutions"].tolist() if has_inst_col else [[] for _ in range(len(eval_df))]

        for i, (aus, title, insts) in enumerate(zip(eval_df["parsed_authors"], titles, inst_lists)):
            known_aus = [a for a in aus if a in sc1]
            last_set = set(aus[-2:]) if len(aus) >= 3 else set()
            kv1 = [sc1[a] * (1.12 if a in last_set else 1.0) for a in known_aus]
            kv2 = [sc2[a] * (1.12 if a in last_set else 1.0) for a in known_aus if a in sc2]
            kvl1 = [scl1[a] for a in aus if a in scl1]
            kvl2 = [scl2[a] for a in aus if a in scl2]
            kv3 = [sc3[a] for a in aus if a in sc3]
            kc = [cnt[a] for a in aus if a in cnt]

            dx = raw_dmax[i]
            dm = raw_dmean[i]
            if not np.isfinite(dx) or not np.isfinite(dm):
                kdx = [dmax_map[a] for a in aus if a in dmax_map]
                kdm = [dmean_map[a] for a in aus if a in dmean_map]
                dx = float(np.max(kdx)) if kdx else 18.0
                dm = float(np.mean(kdm)) if kdm else 8.0

            dx_log = float(np.log1p(max(0.0, dx)))
            dm_log = float(np.log1p(max(0.0, dm)))
            s3_val = float(np.mean(kv3)) if kv3 else 0.0

            n_total = max(1, len(aus))

            if kv1:
                if len(known_aus) >= 2 and papers_map:
                    union_p = set().union(*(papers_map.get(a, set()) for a in known_aus))
                    sum_p = sum(len(papers_map.get(a, set())) for a in known_aus)
                    div_factor = float((len(union_p) / max(1, sum_p)) ** 0.45) if sum_p > 0 else 1.0
                else:
                    div_factor = 1.0
                is_challenge = ("challenge" in title.lower() and len(aus) >= 15) or (len(aus) > 55)
                max_k = 8 if is_challenge else 52
                n_unk = max(0, min(max_k, len(aus)) - len(known_aus))
                supp = 0.20 if (len(known_aus) == 1 and len(aus) >= 7) else 1.0

                kv1_adj = [v * div_factor for v in kv1]
                tm1 = float(np.mean(kv1_adj))
                all1 = sorted(kv1_adj + [0.38 * tm1 * supp] * n_unk, reverse=True)[:max_k]
                r1 = np.arange(1, len(all1) + 1, dtype=np.float64) ** 0.16
                s1_pow = float(np.sum(np.asarray(all1) * r1))
                s1_lin = float(sum(kv1_adj) + n_unk * (0.38 * tm1 * supp))

                kv2_adj = [v * div_factor for v in kv2] if kv2 else [tm1 / 7.5]
                tm2 = float(np.mean(kv2_adj))
                all2 = sorted(kv2_adj + [0.35 * tm2 * supp] * n_unk, reverse=True)[:max_k]
                r2 = np.arange(1, len(all2) + 1, dtype=np.float64) ** 0.08
                s2_pow = float(np.sum(np.asarray(all2) * r2))

                out[i] = [
                    s1_pow,
                    s1_lin,
                    float(np.max(kv1)),
                    tm1,
                    s2_pow,
                    float(np.max(kv2)) if kv2 else tm2,
                    tm2,
                    float(np.max(kvl1)) if kvl1 else g_l1,
                    float(np.mean(kvl1)) if kvl1 else g_l1,
                    float(np.sum(kvl1)) if kvl1 else g_l1,
                    float(np.max(kvl2)) if kvl2 else g_l2,
                    float(np.mean(kvl2)) if kvl2 else g_l2,
                    float(np.sum(kvl2)) if kvl2 else g_l2,
                    float(np.log1p(np.sum(kc))),
                    float(len(kv1) / n_total),
                    dx_log,
                    dm_log,
                    s3_val,
                ]
            else:
                dblp_scale = float(np.clip(dx_log / np.log1p(25.0), 0.5, 2.2))
                iv1 = [min(inst_sc1[inst], g_raw1 * 1.8) for inst in insts if inst in inst_sc1]
                iv2 = [min(inst_sc2[inst], g_raw2 * 1.8) for inst in insts if inst in inst_sc2]
                c_raw1 = (0.25 * float(np.mean(iv1)) + 0.75 * g_raw1 * dblp_scale) if iv1 else (g_raw1 * dblp_scale)
                c_raw2 = (0.25 * float(np.mean(iv2)) + 0.75 * g_raw2 * dblp_scale) if iv2 else (g_raw2 * dblp_scale)
                out[i] = [
                    c_raw1,
                    c_raw1,
                    c_raw1,
                    g_raw1,
                    c_raw2,
                    c_raw2,
                    g_raw2,
                    g_l1,
                    g_l1,
                    g_l1,
                    g_l2,
                    g_l2,
                    g_l2,
                    0.0,
                    0.0,
                    dx_log,
                    dm_log,
                    s3_val,
                ]
        return out

    def fit(self, corpus_df: pd.DataFrame) -> "AuthorCollaborationGraphEncoder":
        corpus_df = corpus_df.reset_index(drop=True)
        self.full_graph_ = self._build_graph_dict(corpus_df)
        self.oof_cache_ = {}

        n = len(corpus_df)
        if n >= self.n_oof_folds:
            rng = np.random.RandomState(self.random_state)
            idx = np.arange(n)
            rng.shuffle(idx)
            folds = np.array_split(idx, self.n_oof_folds)
            oof_mat = np.zeros((n, len(self.FEATURE_NAMES)), dtype=np.float64)
            for f_idx in folds:
                mask = np.ones(n, dtype=bool)
                mask[f_idx] = False
                g_fold = self._build_graph_dict(corpus_df.iloc[mask])
                oof_mat[f_idx] = self._transform_with_graph(corpus_df.iloc[f_idx], g_fold)
            for pid, row_vec in zip(corpus_df["paper_id"].astype(str), oof_mat):
                self.oof_cache_[pid] = row_vec
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.full_graph_ is None:
            raise RuntimeError("AuthorCollaborationGraphEncoder is not fitted.")
        pids = frame["paper_id"].astype(str).tolist()
        if all(pid in self.oof_cache_ for pid in pids):
            return np.vstack([self.oof_cache_[pid] for pid in pids])

        full_mat = self._transform_with_graph(frame, self.full_graph_)
        for i, pid in enumerate(pids):
            if pid in self.oof_cache_:
                full_mat[i] = self.oof_cache_[pid]
        return full_mat


class FeatureBuilder:
    """Fit all learned text/category/graph/reduction steps on training rows only.

    ``transform(frame, family='tree')`` keeps numeric NaNs for tree models while
    providing imputed interaction features.  ``family='linear'`` median-imputes
    and scales the dense block for Ridge/ElasticNet.
    """

    DERIVED_INTERACTION_NAMES = (
        "em_paradigm_y1",
        "em_paradigm_y2",
        "oof_topic_y1_raw",
        "oof_topic_y1_log",
        "oof_topic_y2_raw",
        "oof_topic_y2_log",
        "mult_y1_synergy",
        "mult_y2_synergy",
    )
    _CORPUS_CACHE: Dict[Tuple[int, str, str, bool, bool, int, int], Dict[str, Any]] = {}

    def __init__(
        self,
        use_bge: bool = True,
        bge_dim: int = 128,
        tfidf_max_features: int = 8000,
        tfidf_min_df: int = 3,
        category_min_count: int = 15,
        strict_t0: bool = False,
        random_state: int = 42,
    ) -> None:
        self.use_bge = bool(use_bge)
        self.bge_dim = int(bge_dim)
        self.tfidf_max_features = int(tfidf_max_features)
        self.tfidf_min_df = int(tfidf_min_df)
        self.category_min_count = int(category_min_count)
        self.strict_t0 = bool(strict_t0)
        self.random_state = int(random_state)

        self.numeric_cols: List[str] = list(
            STRICT_T0_NUMERIC_FEATURES if self.strict_t0 else EXTENDED_NUMERIC_FEATURES
        )
        self.has_external_: bool = False
        self.text_col: str = "text_t0" if self.strict_t0 else "text_all"

        self.category_vocab: list[str] = []
        self._category_to_index: dict[str, int] = {}
        self.tfidf: Any = None
        self.bge_pca: Any = None
        self.numeric_imputer: Any = None
        self.numeric_scaler: Any = None
        self.embedding_store: Optional[EmbeddingStore] = None
        self.embedding_model_name: Optional[str] = None

        self.author_graph: Optional[AuthorCollaborationGraphEncoder] = None
        self.emerging_encoder: Optional[EmergingParadigmEncoder] = None

        # Orthogonal text+category ridge weights for multiplicative topic synergy
        self._tc_w_y1_raw: Optional[np.ndarray] = None
        self._tc_m_y1_raw: float = 18.0
        self._tc_w_y1_log: Optional[np.ndarray] = None
        self._tc_m_y1_log: float = 2.0
        self._tc_w_y2_raw: Optional[np.ndarray] = None
        self._tc_m_y2_raw: float = 2.4
        self._tc_w_y2_log: Optional[np.ndarray] = None
        self._tc_m_y2_log: float = 0.7
        self._oof_topic_cache: Dict[str, np.ndarray] = {}
        self._ref_mean_raw: float = 38.0

        self._feature_names: Optional[list[str]] = None

    def _category_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = np.zeros((len(frame), len(self.category_vocab)), dtype=np.float32)
        for row, labels in enumerate(frame["category_labels"]):
            for label in labels:
                index = self._category_to_index.get(str(label))
                if index is not None:
                    matrix[row, index] = 1.0
        return matrix

    def _build_tc_sparse(self, frame: pd.DataFrame) -> Tuple[sp.csr_matrix, np.ndarray, sp.csr_matrix]:
        category = self._category_matrix(frame)
        tfidf = self.tfidf.transform(frame[self.text_col].astype(str)).astype(np.float32)
        tc_mat = sp.hstack([tfidf, sp.csr_matrix(category * 0.35)], format="csr")
        return tc_mat, category, tfidf

    def fit(
        self,
        frame: pd.DataFrame,
        embeddings: Optional[EmbeddingStore] = None,
        corpus_frame: Optional[pd.DataFrame] = None,
    ) -> "FeatureBuilder":
        if frame.empty:
            raise ValueError("Cannot fit FeatureBuilder on an empty frame.")

        ref_corpus = corpus_frame if corpus_frame is not None else frame
        ref_corpus = ref_corpus.reset_index(drop=True)
        self.has_external_ = "oa_matched" in frame.columns
        has_ext_full = "oa_early_citations_t0_year_log1p" in frame.columns
        if self.strict_t0:
            self.numeric_cols = list(STRICT_T0_NUMERIC_FEATURES) + (
                list(STRICT_T0_EXTERNAL_NUMERIC_FEATURES) if self.has_external_ else []
            )
        else:
            ext_cols = (
                list(EXTENDED_EXTERNAL_NUMERIC_FEATURES)
                if has_ext_full
                else (list(STRICT_T0_EXTERNAL_NUMERIC_FEATURES) if self.has_external_ else [])
            )
            self.numeric_cols = list(EXTENDED_NUMERIC_FEATURES) + ext_cols

        cache_key = (
            len(ref_corpus),
            str(ref_corpus["paper_id"].iloc[0]),
            str(ref_corpus["paper_id"].iloc[-1]),
            self.strict_t0,
            self.has_external_,
            self.tfidf_max_features,
            self.category_min_count,
        )
        if cache_key in self._CORPUS_CACHE:
            cached = self._CORPUS_CACHE[cache_key]
            self.author_graph = cached["author_graph"]
            self.emerging_encoder = cached["emerging_encoder"]
            self._ref_mean_raw = cached["ref_mean_raw"]
            self.category_vocab = cached["category_vocab"]
            self._category_to_index = cached["category_to_index"]
            self.tfidf = cached["tfidf"]
            self._tc_m_y1_raw = cached["tc_m_y1_raw"]
            self._tc_m_y1_log = cached["tc_m_y1_log"]
            self._tc_m_y2_raw = cached["tc_m_y2_raw"]
            self._tc_m_y2_log = cached["tc_m_y2_log"]
            self._tc_w_y1_raw = cached["tc_w_y1_raw"]
            self._tc_w_y1_log = cached["tc_w_y1_log"]
            self._tc_w_y2_raw = cached["tc_w_y2_raw"]
            self._tc_w_y2_log = cached["tc_w_y2_log"]
            self._oof_topic_cache = cached["oof_topic_cache"]
        else:
            self._fit_corpus_level(ref_corpus)
            self._CORPUS_CACHE[cache_key] = {
                "author_graph": self.author_graph,
                "emerging_encoder": self.emerging_encoder,
                "ref_mean_raw": self._ref_mean_raw,
                "category_vocab": self.category_vocab,
                "category_to_index": self._category_to_index,
                "tfidf": self.tfidf,
                "tc_m_y1_raw": self._tc_m_y1_raw,
                "tc_m_y1_log": self._tc_m_y1_log,
                "tc_m_y2_raw": self._tc_m_y2_raw,
                "tc_m_y2_log": self._tc_m_y2_log,
                "tc_w_y1_raw": self._tc_w_y1_raw,
                "tc_w_y1_log": self._tc_w_y1_log,
                "tc_w_y2_raw": self._tc_w_y2_raw,
                "tc_w_y2_log": self._tc_w_y2_log,
                "oof_topic_cache": self._oof_topic_cache,
            }

        # 4. Build full dense block on `frame` and fit imputer + scaler
        dense_train = self._build_dense_block(frame)
        dense_train_lin = self._log_compress_dense(dense_train)
        self.numeric_imputer = _FallbackImputer().fit(dense_train_lin)
        imputed = self.numeric_imputer.transform(dense_train_lin)
        self.numeric_scaler = _FallbackScaler().fit(imputed)

        if self.use_bge:
            if embeddings is None:
                raise ValueError("use_bge=True requires an EmbeddingStore.")
            self.embedding_store = embeddings
            self.embedding_model_name = embeddings.model_name
            raw = embeddings.get(ref_corpus)
            n_components = min(self.bge_dim, raw.shape[1], max(2, len(ref_corpus) - 1))
            if SKLEARN_AVAILABLE:
                self.bge_pca = PCA(
                    n_components=n_components, random_state=self.random_state
                ).fit(raw)
            else:
                self.bge_pca = _FallbackPCA(
                    n_components=n_components, random_state=self.random_state
                ).fit(raw)

        dense_names = (
            list(self.numeric_cols)
            + list(AuthorCollaborationGraphEncoder.FEATURE_NAMES)
            + list(self.DERIVED_INTERACTION_NAMES)
        )
        cat_names = [f"cat__{label}" for label in self.category_vocab]
        tfidf_names = [f"tfidf__{token}" for token in self.tfidf.get_feature_names_out()]
        bge_names = (
            [f"bge__pc{index}" for index in range(self.bge_pca.n_components_)]
            if self.bge_pca is not None else []
        )
        self._feature_names = dense_names + cat_names + tfidf_names + bge_names
        return self

    def _fit_corpus_level(self, ref_corpus: pd.DataFrame) -> None:
        # 1. Fit 2-hop author collaboration graph & emerging paradigm encoder on ref_corpus
        self.author_graph = AuthorCollaborationGraphEncoder(
            n_oof_folds=10, random_state=self.random_state
        ).fit(ref_corpus)
        self.emerging_encoder = EmergingParadigmEncoder().fit(ref_corpus)
        self._ref_mean_raw = float(
            ref_corpus["s2_reference_count_current"].fillna(35.0).clip(0.0, 250.0).mean()
        )

        # 2. Fit multi-hot category vocabulary & TF-IDF on ref_corpus
        counts = Counter(
            str(label) for labels in ref_corpus["category_labels"] for label in labels
        )
        self.category_vocab = sorted(
            label for label, count in counts.items()
            if count >= self.category_min_count
        )
        self._category_to_index = {
            label: index for index, label in enumerate(self.category_vocab)
        }

        if SKLEARN_AVAILABLE:
            self.tfidf = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                ngram_range=(1, 2),
                min_df=self.tfidf_min_df,
                max_df=0.95,
                sublinear_tf=True,
                stop_words="english",
                dtype=np.float32,
            ).fit(ref_corpus[self.text_col].astype(str))
        else:
            self.tfidf = _FallbackTfidfVectorizer(
                max_features=self.tfidf_max_features,
                min_df=self.tfidf_min_df,
                max_df=0.95,
            ).fit(ref_corpus[self.text_col].astype(str))

        # 3. Fit OOF orthogonal text+category topic predictors on ref_corpus
        tc_all, _, _ = self._build_tc_sparse(ref_corpus)
        y1_raw = ref_corpus["s2_citation_count_3y"].to_numpy(dtype=float)
        y2_raw = ref_corpus["s2_influential_citation_count_3y"].to_numpy(dtype=float)
        y1_imp = np.where(
            np.isfinite(y1_raw),
            y1_raw,
            np.where(np.isfinite(y2_raw), y2_raw * 6.0, 18.0),
        )
        y2_imp = np.where(np.isfinite(y2_raw), y2_raw, 2.0)

        y1_cap = np.clip(y1_imp, 0.0, 200.0)
        y1_log = np.log1p(np.clip(y1_imp, 0.0, None))
        y2_cap = np.clip(y2_imp, 0.0, 35.0)
        y2_log = np.log1p(np.clip(y2_imp, 0.0, None))

        self._tc_m_y1_raw = float(np.mean(y1_cap))
        self._tc_m_y1_log = float(np.mean(y1_log))
        self._tc_m_y2_raw = float(np.mean(y2_cap))
        self._tc_m_y2_log = float(np.mean(y2_log))

        self._tc_w_y1_raw = lsqr(tc_all, y1_cap - self._tc_m_y1_raw, damp=8.0)[0]
        self._tc_w_y1_log = lsqr(tc_all, y1_log - self._tc_m_y1_log, damp=6.0)[0]
        self._tc_w_y2_raw = lsqr(tc_all, y2_cap - self._tc_m_y2_raw, damp=8.0)[0]
        self._tc_w_y2_log = lsqr(tc_all, y2_log - self._tc_m_y2_log, damp=6.0)[0]

        self._oof_topic_cache = {}
        n_c = len(ref_corpus)
        if n_c >= 50:
            rng = np.random.RandomState(self.random_state)
            idx = np.arange(n_c)
            rng.shuffle(idx)
            folds = np.array_split(idx, 5)
            oof_t = np.zeros((n_c, 4), dtype=np.float64)
            for f_idx in folds:
                m = np.ones(n_c, dtype=bool)
                m[f_idx] = False
                tc_tr = tc_all[m]
                tc_va = tc_all[f_idx]

                m1r = float(np.mean(y1_cap[m]))
                w1r = lsqr(tc_tr, y1_cap[m] - m1r, damp=8.0)[0]
                m1l = float(np.mean(y1_log[m]))
                w1l = lsqr(tc_tr, y1_log[m] - m1l, damp=6.0)[0]
                m2r = float(np.mean(y2_cap[m]))
                w2r = lsqr(tc_tr, y2_cap[m] - m2r, damp=8.0)[0]
                m2l = float(np.mean(y2_log[m]))
                w2l = lsqr(tc_tr, y2_log[m] - m2l, damp=6.0)[0]

                oof_t[f_idx, 0] = tc_va @ w1r + m1r
                oof_t[f_idx, 1] = tc_va @ w1l + m1l
                oof_t[f_idx, 2] = tc_va @ w2r + m2r
                oof_t[f_idx, 3] = tc_va @ w2l + m2l

            for pid, vec in zip(ref_corpus["paper_id"].astype(str), oof_t):
                self._oof_topic_cache[pid] = vec

    def _predict_topic_scores(
        self,
        frame: pd.DataFrame,
        tc_mat: Optional[sp.csr_matrix] = None,
    ) -> np.ndarray:
        pids = frame["paper_id"].astype(str).tolist()
        if all(pid in self._oof_topic_cache for pid in pids):
            return np.vstack([self._oof_topic_cache[pid] for pid in pids])

        if tc_mat is None:
            tc_mat, _, _ = self._build_tc_sparse(frame)
        assert (
            self._tc_w_y1_raw is not None
            and self._tc_w_y1_log is not None
            and self._tc_w_y2_raw is not None
            and self._tc_w_y2_log is not None
        )
        out = np.column_stack([
            tc_mat @ self._tc_w_y1_raw + self._tc_m_y1_raw,
            tc_mat @ self._tc_w_y1_log + self._tc_m_y1_log,
            tc_mat @ self._tc_w_y2_raw + self._tc_m_y2_raw,
            tc_mat @ self._tc_w_y2_log + self._tc_m_y2_log,
        ])
        for i, pid in enumerate(pids):
            if pid in self._oof_topic_cache:
                out[i] = self._oof_topic_cache[pid]
        return out

    def _build_dense_block(
        self,
        frame: pd.DataFrame,
        tc_mat: Optional[sp.csr_matrix] = None,
    ) -> np.ndarray:
        assert self.author_graph is not None and self.emerging_encoder is not None
        base_num = frame[self.numeric_cols].to_numpy(dtype=np.float64)
        graph_feats = self.author_graph.transform(frame)
        em_feats = self.emerging_encoder.transform(frame)
        topic_feats = self._predict_topic_scores(frame, tc_mat=tc_mat)

        # Multiplicative synergy between 2-hop Author Graph, Topic Score, Emerging Paradigm, and Reference Depth
        g_y1 = graph_feats[:, 0]
        g_y2 = graph_feats[:, 4]
        t_y1 = topic_feats[:, 0]
        t_y2 = topic_feats[:, 2]
        em_y1 = em_feats[:, 0]
        em_y2 = em_feats[:, 1]

        colon_flag = frame["text_has_named_model_colon"].to_numpy(dtype=np.float64)
        code_flag = frame["text_has_code_or_url"].to_numpy(dtype=np.float64)

        ext_boost_1 = np.zeros(len(frame), dtype=np.float64)
        ext_boost_2 = np.zeros(len(frame), dtype=np.float64)
        if self.has_external_ and "oa_matched" in frame.columns:
            ind_flag = frame["oa_has_top_industry_lab"].to_numpy(dtype=np.float64)
            pwc_code = frame["pwc_has_official_code"].to_numpy(dtype=np.float64)
            rec_ratio = np.nan_to_num(frame["s2_ref_recent_2y_ratio"].to_numpy(dtype=np.float64), nan=0.35)
            ext_boost_1 = 0.08 * ind_flag + 0.05 * pwc_code + 0.06 * (rec_ratio - 0.35)
            ext_boost_2 = 0.10 * ind_flag + 0.07 * pwc_code + 0.08 * (rec_ratio - 0.35)
            if not self.strict_t0 and "oa_early_citations_t0_year_log1p" in frame.columns:
                early_c = frame["oa_early_citations_t0_year_log1p"].to_numpy(dtype=np.float64)
                ext_boost_1 = ext_boost_1 + 0.08 * early_c
                ext_boost_2 = ext_boost_2 + 0.10 * early_c

        topic_mult_1 = np.clip(
            1.0 + 0.25 * (t_y1 - self._tc_m_y1_raw) / max(1.0, self._tc_m_y1_raw) + 0.06 * em_y1 + 0.06 * colon_flag + 0.06 * code_flag + ext_boost_1,
            0.45,
            2.8,
        )
        topic_mult_2 = np.clip(
            1.0 + 0.45 * (t_y2 - self._tc_m_y2_raw) / max(0.5, self._tc_m_y2_raw) + 0.08 * em_y2 + 0.16 * colon_flag + 0.08 * code_flag + ext_boost_2,
            0.40,
            3.0,
        )

        t_l1 = np.clip(topic_feats[:, 1], 0.1, 6.0)
        t_l2 = np.clip(topic_feats[:, 3], 0.1, 6.0)
        l1_factor = (t_l1 / max(0.5, self._tc_m_y1_log)) ** 0.25
        l2_factor = (t_l2 / max(0.2, self._tc_m_y2_log)) ** 0.25

        if not self.strict_t0:
            ref_raw = (
                pd.to_numeric(frame["s2_reference_count_current"], errors="coerce")
                .fillna(self._ref_mean_raw)
                .clip(0.0, 250.0)
                .to_numpy(dtype=np.float64)
            )
            rm = max(10.0, self._ref_mean_raw)
            ref_mult_1 = np.clip(1.0 + 0.16 * (ref_raw - rm) / rm, 0.5, 2.5)
            ref_mult_2 = np.clip(1.0 + 0.18 * (ref_raw - rm) / rm, 0.5, 2.5)
            base_m1 = np.maximum(0.0, g_y1 * topic_mult_1 * ref_mult_1 + 0.25 * np.clip(t_y1 - self._tc_m_y1_raw, -10.0, 120.0) + 0.12 * (ref_raw - rm))
            base_m2 = np.maximum(0.0, g_y2 * topic_mult_2 * ref_mult_2 + 0.90 * np.clip(t_y2 - self._tc_m_y2_raw, -2.0, 20.0))
        else:
            base_m1 = np.maximum(0.0, g_y1 * topic_mult_1 + 0.25 * np.clip(t_y1 - self._tc_m_y1_raw, -10.0, 120.0))
            base_m2 = np.maximum(0.0, g_y2 * topic_mult_2 + 0.90 * np.clip(t_y2 - self._tc_m_y2_raw, -2.0, 20.0))

        mult_1 = (base_m1 ** 1.12) * l1_factor
        mult_2 = (base_m2 ** 1.12) * l2_factor

        derived = np.column_stack([
            em_feats,
            topic_feats,
            mult_1,
            mult_2,
        ])
        return np.hstack([base_num, graph_feats, derived])

    def _log_compress_dense(self, dense: np.ndarray) -> np.ndarray:
        out = dense.copy()
        n_base = len(self.numeric_cols)
        # Only log-compress true raw count columns: graph raw cols (0..6 of graph block) and raw topic/synergy cols (-6, -4, -2, -1)
        raw_count_cols = [
            n_base + 0, n_base + 1, n_base + 2, n_base + 3,
            n_base + 4, n_base + 5, n_base + 6,
            out.shape[1] - 6, out.shape[1] - 4, out.shape[1] - 2, out.shape[1] - 1,
        ]
        for c in raw_count_cols:
            col_v = out[:, c]
            m = np.isfinite(col_v)
            out[m, c] = np.log1p(np.maximum(0.0, col_v[m]))
        return out

    def extract_raw_dense(self, frame: pd.DataFrame) -> np.ndarray:
        """Return median-imputed unscaled dense features for specialized multiplicative/calibrated models."""
        if self.tfidf is None or self.numeric_imputer is None:
            raise RuntimeError("FeatureBuilder is not fitted.")
        tc_mat, _, _ = self._build_tc_sparse(frame)
        dense = self._build_dense_block(frame, tc_mat=tc_mat)
        return self.numeric_imputer.transform(dense)

    def transform(
        self,
        frame: pd.DataFrame,
        family: str = "tree",
        embeddings: Optional[EmbeddingStore] = None,
    ) -> sp.csr_matrix:
        if self.tfidf is None or self.numeric_imputer is None or self.numeric_scaler is None:
            raise RuntimeError("FeatureBuilder is not fitted.")
        if family not in {"tree", "linear"}:
            raise ValueError("family must be 'tree' or 'linear'.")

        tc_mat, category, tfidf = self._build_tc_sparse(frame)
        dense = self._build_dense_block(frame, tc_mat=tc_mat)
        if family == "linear":
            dense = self._log_compress_dense(dense)
            dense = self.numeric_imputer.transform(dense)
            dense = self.numeric_scaler.transform(dense) * 0.28

        blocks = [
            sp.csr_matrix(dense.astype(np.float32)),
            sp.csr_matrix(category),
            tfidf.astype(np.float32),
        ]
        if self.use_bge:
            store = embeddings or self.embedding_store
            if self.bge_pca is None or store is None:
                raise RuntimeError(
                    "BGE PCA/store is not available. Attach the EmbeddingStore "
                    "to the builder or pass embeddings=... to transform()."
                )
            bge = self.bge_pca.transform(
                np.asarray(store.get(frame), dtype=np.float32)
            )
            blocks.append(sp.csr_matrix(bge.astype(np.float32)))
        matrix = sp.hstack(blocks, format="csr").astype(np.float32)
        if self._feature_names is not None and matrix.shape[1] != len(self._feature_names):
            raise RuntimeError(
                f"Feature width mismatch: {matrix.shape[1]} vs {len(self._feature_names)}."
            )
        return matrix

    def attach_embedding_store(self, embeddings: EmbeddingStore) -> "FeatureBuilder":
        self.embedding_store = embeddings
        return self

    def detach_embedding_store(self) -> "FeatureBuilder":
        self.embedding_store = None
        return self

    def get_feature_names_out(self) -> list[str]:
        if self._feature_names is None:
            raise RuntimeError("FeatureBuilder is not fitted.")
        return list(self._feature_names)

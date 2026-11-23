"""Leakage-safe, strictly causal As-Of Temporal feature engineering for Y1/Y2/Y3.

Implements V4.1 Strict Causal Temporal Encoding across all target-derived encoders:
1. EmergingParadigmEncoder:
   - Eliminates self-target leakage and future-window leakage via Chronological
     Expanding-Window Temporal OOF.
   - Under `asof_maturity=True`, historical paper j's 3-year citation count is
     strictly masked unless `t0_days_j + 1095.75 <= t_cut` (falling back to a
     100% label-free pre-t0 ngram burstiness + DBLP seniority prior when <10
     matured papers exist).
2. AuthorCollaborationGraphEncoder:
   - Replaces Random 10-fold OOF with Chronological Expanding-Window Temporal OOF
     with ZERO global post-scaling (`ref_means` removed completely), guaranteeing
     that mutating future years' citation labels has zero effect (0.0 diff) on
     past years' author graph features.
   - Under `asof_maturity=True`, enforces `t0_days_j + 1095.75 <= t_cut` for all
     3-year citation labels (`Y1`, `Y2`) and `t0_days_j + 120 <= t_cut` for
     OpenReview ratings (`Y3`), while using observed pre-t0 co-authorship edges
     and pre-t0 DBLP/arXiv publication counts (`t0_days_j < t_cut`).
3. FeatureBuilder Topic Predictor:
   - Replaces Random 5-fold OOF with Chronological Expanding-Window Temporal OOF,
     and under `asof_maturity=True` strictly enforces `t0_days_j + 1095.75 <= t_cut`
     across both training OOF buckets and test inference buckets (falling back to
     a 100% label-free pre-t0 DBLP proxy when <5 matured labels exist, never
     falling back to unmatured citation labels).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr, svds

from .constants import (
    EXTENDED_EXTERNAL_NUMERIC_FEATURES,
    EXTENDED_NUMERIC_FEATURES,
    STRICT_T0_EXTERNAL_NUMERIC_FEATURES,
    STRICT_T0_NUMERIC_FEATURES,
)
from .embeddings import EmbeddingStore

try:
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    PCA = None  # type: ignore[assignment]
    TfidfVectorizer = None  # type: ignore[assignment]
    SimpleImputer = None  # type: ignore[assignment]
    RobustScaler = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False


_ENGLISH_STOP_WORDS = frozenset({
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves",
    "paper", "propose", "proposed", "method", "methods", "approach", "model",
    "models", "based", "using", "used", "use", "results", "show", "performance",
    "problem", "tasks", "task", "data", "learning", "network", "networks",
    "state", "art", "existing", "novel", "new", "work", "study", "provide",
    "achieve", "experimental", "experiments", "demonstrate", "effectiveness",
})
_TOKEN_RE = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]+\b")


def _array_digest(arr: np.ndarray) -> str:
    a = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=-999.0)
    return hashlib.md5(a.tobytes()).hexdigest()[:12]


class _FallbackImputer:
    def __init__(self) -> None:
        self.statistics_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_FallbackImputer":
        with np.errstate(all="ignore"):
            medians = np.nanmedian(X, axis=0)
        self.statistics_ = np.where(np.isfinite(medians), medians, 0.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.statistics_ is not None
        out = np.array(X, dtype=np.float64, copy=True)
        inds = np.where(~np.isfinite(out))
        out[inds] = np.take(self.statistics_, inds[1])
        return out


class _FallbackScaler:
    def __init__(self) -> None:
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_FallbackScaler":
        self.center_ = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        scale = q75 - q25
        self.scale_ = np.where(scale > 1e-8, scale, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.center_ is not None and self.scale_ is not None
        return (X - self.center_) / self.scale_


class _FallbackPCA:
    def __init__(self, n_components: int, random_state: int = 42) -> None:
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.n_components_: int = 0

    def fit(self, X: np.ndarray) -> "_FallbackPCA":
        X_f = np.asarray(X, dtype=np.float64)
        self.mean_ = X_f.mean(axis=0)
        Xc = X_f - self.mean_
        k = min(self.n_components, Xc.shape[0] - 1, Xc.shape[1] - 1)
        _, s, vt = svds(Xc, k=k, random_state=self.random_state)
        order = np.argsort(-s)
        self.components_ = vt[order]
        self.n_components_ = int(self.components_.shape[0])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.components_ is not None
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T


class _FallbackTfidfVectorizer:
    def __init__(self, max_features: int = 8000, min_df: int = 3, max_df: float = 0.95) -> None:
        self.max_features = int(max_features)
        self.min_df = int(min_df)
        self.max_df = float(max_df)
        self.vocab_: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = [
            t.lower() for t in _TOKEN_RE.findall(text)
            if t.lower() not in _ENGLISH_STOP_WORDS
        ]
        bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def fit(self, texts: Sequence[str]) -> "_FallbackTfidfVectorizer":
        n_docs = len(texts)
        df_counts: Counter[str] = Counter()
        tf_totals: Counter[str] = Counter()
        for text in texts:
            toks = self._tokenize(str(text))
            tf_totals.update(toks)
            df_counts.update(set(toks))
        max_doc_count = int(self.max_df * n_docs)
        candidates = [
            (tok, tf_totals[tok])
            for tok, dfc in df_counts.items()
            if self.min_df <= dfc <= max_doc_count
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        selected = [tok for tok, _ in candidates[: self.max_features]]
        selected.sort()
        self.feature_names_ = selected
        self.vocab_ = {tok: idx for idx, tok in enumerate(selected)}
        df_vec = np.array([df_counts[tok] for tok in selected], dtype=np.float64)
        self.idf_ = np.log((1.0 + n_docs) / (1.0 + df_vec)) + 1.0
        return self

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        assert self.idf_ is not None
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        vocab = self.vocab_
        idf = self.idf_
        for r_idx, text in enumerate(texts):
            counts: Counter[int] = Counter()
            for tok in self._tokenize(str(text)):
                c_idx = vocab.get(tok)
                if c_idx is not None:
                    counts[c_idx] += 1
            if not counts:
                continue
            c_indices = np.fromiter(counts.keys(), dtype=np.int32)
            c_tfs = np.fromiter(counts.values(), dtype=np.float64)
            weights = (1.0 + np.log(c_tfs)) * idf[c_indices]
            norm = float(np.sqrt(np.sum(weights * weights)))
            if norm > 0:
                weights /= norm
            rows.extend([r_idx] * len(c_indices))
            cols.extend(c_indices.tolist())
            vals.extend(weights.astype(np.float32).tolist())
        return sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(len(texts), len(self.feature_names_)),
            dtype=np.float32,
        )

    def get_feature_names_out(self) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)


class EmergingParadigmEncoder:
    """Identifies emerging high-impact n-grams using strictly causal Expanding-Window
    Temporal OOF (zero self-target leakage and zero future-window leakage).
    """

    _GRAM_CACHE: Dict[str, Tuple[set[str], set[str], set[str]]] = {}

    def __init__(self, asof_maturity: bool = False, random_state: int = 42) -> None:
        self.asof_maturity = bool(asof_maturity)
        self.random_state = int(random_state)
        self.em_y1: Dict[str, float] = {}
        self.em_y2: Dict[str, float] = {}
        self.mean_y1_: float = 0.0
        self.std_y1_: float = 1.0
        self.mean_y2_: float = 0.0
        self.std_y2_: float = 1.0
        self.oof_cache_: Dict[str, np.ndarray] = {}
        self.bucket_models_: Dict[int, Tuple[Dict[str, float], Dict[str, float], float, float, float, float]] = {}
        self._corpus_df_: Optional[pd.DataFrame] = None

    @staticmethod
    def _extract_grams(text: str) -> set[str]:
        toks = [
            t.lower() for t in _TOKEN_RE.findall(text)
            if len(t) >= 3 and t.lower() not in _ENGLISH_STOP_WORDS
        ]
        bigrams = [f"{toks[i]} {toks[i + 1]}" for i in range(len(toks) - 1)]
        return set(toks + bigrams)

    @classmethod
    def _get_paper_grams(cls, frame: pd.DataFrame) -> List[Tuple[set[str], set[str], set[str]]]:
        pids = frame["paper_id"].astype(str).tolist()
        titles = frame["title_v1"].fillna("").astype(str).tolist()
        absts = frame["abstract_v1"].fillna("").astype(str).tolist()
        out: List[Tuple[set[str], set[str], set[str]]] = []
        cache = cls._GRAM_CACHE
        for pid, title, abst in zip(pids, titles, absts):
            entry = cache.get(pid)
            if entry is None:
                tg = cls._extract_grams(title)
                ag = cls._extract_grams(abst)
                entry = (tg, ag, tg | ag)
                cache[pid] = entry
            out.append(entry)
        return out

    def _fit_lexicon_on_subset(
        self,
        sub_df: pd.DataFrame,
        t_cut: Optional[float] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], float, float, float, float]:
        if sub_df.empty:
            return {}, {}, 0.0, 1.0, 0.0, 1.0

        t0_d = sub_df["t0_days"].to_numpy(dtype=float)
        if t_cut is not None:
            valid_time = t0_d < t_cut
            if not np.any(valid_time):
                return {}, {}, 0.0, 1.0, 0.0, 1.0
            sub_df = sub_df.loc[valid_time].reset_index(drop=True)
            t0_d = sub_df["t0_days"].to_numpy(dtype=float)

        mid_d = float(np.median(t0_d))
        cnt_early: Counter[str] = Counter()
        cnt_late: Counter[str] = Counter()
        y1_src: Dict[str, List[float]] = defaultdict(list)
        y2_src: Dict[str, List[float]] = defaultdict(list)

        y1_arr = sub_df["s2_citation_count_3y"].to_numpy(dtype=float).copy()
        y2_arr = sub_df["s2_influential_citation_count_3y"].to_numpy(dtype=float).copy()
        dmax_arr = (
            pd.to_numeric(sub_df["dblp_team_max_publication_count_pre_t0"], errors="coerce")
            .fillna(18.0)
            .to_numpy(dtype=float)
        )

        if self.asof_maturity:
            eff_cut = float(t_cut) if t_cut is not None else float(t0_d.max())
            mat_mask = (t0_d + 1095.75) <= eff_cut
            y1_arr[~mat_mask] = np.nan
            y2_arr[~mat_mask] = np.nan
            has_enough_mat = int(np.sum(mat_mask & np.isfinite(y1_arr))) >= 10
        else:
            mat_mask = np.ones(len(sub_df), dtype=bool)
            has_enough_mat = int(np.sum(np.isfinite(y1_arr))) >= 50

        gram_triples = self._get_paper_grams(sub_df)

        for (d_val, is_m, y1, y2, dx, (_, _, grams)) in zip(
            t0_d, mat_mask, y1_arr, y2_arr, dmax_arr, gram_triples
        ):
            if d_val < mid_d:
                cnt_early.update(grams)
            else:
                cnt_late.update(grams)

            if has_enough_mat:
                use_for_y = is_m if self.asof_maturity else (d_val >= mid_d)
                if use_for_y:
                    if np.isfinite(y1):
                        c1 = min(float(y1), 600.0)
                        for g in grams:
                            y1_src[g].append(c1)
                    if np.isfinite(y2):
                        c2 = min(float(y2), 100.0)
                        for g in grams:
                            y2_src[g].append(c2)
            elif d_val >= mid_d:
                proxy_1 = 18.0 * float(np.clip(np.log1p(dx) / np.log1p(18.0), 0.5, 2.5))
                proxy_2 = 2.4 * float(np.clip(np.log1p(dx) / np.log1p(18.0), 0.5, 2.5))
                for g in grams:
                    y1_src[g].append(proxy_1)
                    y2_src[g].append(proxy_2)

        n_early = max(1, int(np.sum(t0_d < mid_d)))
        n_late = max(1, int(np.sum(t0_d >= mid_d)))
        scale_ratio = float(n_early) / float(n_late)

        all_v1 = [v for lst in y1_src.values() for v in lst]
        all_v2 = [v for lst in y2_src.values() for v in lst]
        base_y1 = float(np.mean(all_v1)) if all_v1 else 18.0
        base_y2 = float(np.mean(all_v2)) if all_v2 else 2.5

        em_y1: Dict[str, float] = {}
        em_y2: Dict[str, float] = {}
        min_support = max(3, int(n_late * 0.0012))
        for g, c_l in cnt_late.items():
            if c_l >= min_support:
                growth = (c_l * scale_ratio + 4.0) / (cnt_early[g] + 4.0)
                if growth >= 1.25:
                    if len(y1_src[g]) >= min_support:
                        m1 = float(np.mean(y1_src[g]))
                        if m1 > base_y1 * 1.15:
                            em_y1[g] = float(np.log1p(growth) * (m1 / base_y1 - 1.0))
                    else:
                        em_y1[g] = float(0.25 * np.log1p(growth))
                    if len(y2_src[g]) >= min_support:
                        m2 = float(np.mean(y2_src[g]))
                        if m2 > base_y2 * 1.15:
                            em_y2[g] = float(np.log1p(growth) * (m2 / base_y2 - 1.0))
                    else:
                        em_y2[g] = float(0.25 * np.log1p(growth))

        r1, r2 = self._score_with_lexicon(sub_df, em_y1, em_y2)
        m1_s, s1_s = float(np.mean(r1)), float(np.std(r1) + 1e-6)
        m2_s, s2_s = float(np.mean(r2)), float(np.std(r2) + 1e-6)
        return em_y1, em_y2, m1_s, s1_s, m2_s, s2_s

    def _score_with_lexicon(
        self,
        frame: pd.DataFrame,
        em_y1: Dict[str, float],
        em_y2: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        s1 = np.zeros(len(frame), dtype=np.float64)
        s2 = np.zeros(len(frame), dtype=np.float64)
        gram_triples = self._get_paper_grams(frame)
        for i, (t_grams, a_grams, _) in enumerate(gram_triples):
            s1[i] = 2.5 * sum(em_y1.get(g, 0.0) for g in t_grams) + sum(
                em_y1.get(g, 0.0) for g in a_grams
            )
            s2[i] = 2.5 * sum(em_y2.get(g, 0.0) for g in t_grams) + sum(
                em_y2.get(g, 0.0) for g in a_grams
            )
        return s1, s2

    def fit(self, corpus_df: pd.DataFrame) -> "EmergingParadigmEncoder":
        corpus_df = corpus_df.reset_index(drop=True)
        self._corpus_df_ = corpus_df
        self._get_paper_grams(corpus_df)
        max_cut = float(corpus_df["t0_days"].max() + 1.0)
        (
            self.em_y1,
            self.em_y2,
            self.mean_y1_,
            self.std_y1_,
            self.mean_y2_,
            self.std_y2_,
        ) = self._fit_lexicon_on_subset(corpus_df, t_cut=max_cut)

        self.oof_cache_ = {}
        self.bucket_models_ = {}
        half_yr = (
            (corpus_df["submission_year"].to_numpy(dtype=int) - 2017) * 2
            + (corpus_df["t0_month"].to_numpy(dtype=float) > 6.0).astype(int)
        )
        oof_mat = np.zeros((len(corpus_df), 2), dtype=np.float64)
        unique_hy = sorted(np.unique(half_yr))

        for hy in unique_hy:
            m_hy = half_yr == hy
            idx_hy = np.flatnonzero(m_hy)
            if hy <= unique_hy[0]:
                sub0 = corpus_df.iloc[idx_hy].reset_index(drop=True)
                if len(sub0) >= 10:
                    rng = np.random.RandomState(self.random_state)
                    perm = np.arange(len(sub0))
                    rng.shuffle(perm)
                    for f_idx in np.array_split(perm, min(10, len(sub0))):
                        mk = np.ones(len(sub0), dtype=bool)
                        mk[f_idx] = False
                        t_c = float(sub0["t0_days"].min()) if self.asof_maturity else float(sub0["t0_days"].max() + 1.0)
                        ey1, ey2, my1, sy1, my2, sy2 = self._fit_lexicon_on_subset(
                            sub0.iloc[mk], t_cut=t_c
                        )
                        r1, r2 = self._score_with_lexicon(sub0.iloc[f_idx], ey1, ey2)
                        oof_mat[idx_hy[f_idx], 0] = (r1 - my1) / sy1
                        oof_mat[idx_hy[f_idx], 1] = (r2 - my2) / sy2
            else:
                prev_mask = half_yr < hy
                prev_df = corpus_df.loc[prev_mask].reset_index(drop=True)
                t_c = float(corpus_df.loc[m_hy, "t0_days"].min())
                ey1, ey2, my1, sy1, my2, sy2 = self._fit_lexicon_on_subset(prev_df, t_cut=t_c)
                r1, r2 = self._score_with_lexicon(corpus_df.iloc[idx_hy], ey1, ey2)
                oof_mat[idx_hy, 0] = (r1 - my1) / sy1
                oof_mat[idx_hy, 1] = (r2 - my2) / sy2

        for pid, vec in zip(corpus_df["paper_id"].astype(str), oof_mat):
            self.oof_cache_[pid] = vec

        return self

    def _get_bucket_model(self, b: int) -> Tuple[Dict[str, float], Dict[str, float], float, float, float, float]:
        b_key = int(np.clip(b, 18, 36))
        if b_key not in self.bucket_models_:
            assert self._corpus_df_ is not None
            t_c = b_key * 60.875
            self.bucket_models_[b_key] = self._fit_lexicon_on_subset(self._corpus_df_, t_cut=t_c)
        return self.bucket_models_[b_key]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        pids = frame["paper_id"].astype(str).tolist()
        if all(pid in self.oof_cache_ for pid in pids):
            return np.vstack([self.oof_cache_[pid] for pid in pids])

        out = np.zeros((len(frame), 2), dtype=np.float64)
        if self.asof_maturity and "t0_bucket" in frame.columns and self._corpus_df_ is not None:
            buckets = frame["t0_bucket"].to_numpy(dtype=int)
            for b in np.unique(buckets):
                mb = buckets == b
                ey1, ey2, my1, sy1, my2, sy2 = self._get_bucket_model(int(b))
                r1, r2 = self._score_with_lexicon(frame.iloc[mb], ey1, ey2)
                out[mb, 0] = (r1 - my1) / sy1
                out[mb, 1] = (r2 - my2) / sy2
        else:
            r1, r2 = self._score_with_lexicon(frame, self.em_y1, self.em_y2)
            out[:, 0] = (r1 - self.mean_y1_) / self.std_y1_
            out[:, 1] = (r2 - self.mean_y2_) / self.std_y2_

        for i, pid in enumerate(pids):
            if pid in self.oof_cache_:
                out[i] = self.oof_cache_[pid]
        return out


class AuthorCollaborationGraphEncoder:
    """As-of Temporal 2-hop co-author collaboration graph with Lp-norm historical
    impact, within-team unknown co-author imputation, cross-paper DBLP recovery,
    and optional training-only institutional prestige priors.

    - Replaces Random OOF with Chronological Expanding-Window Temporal OOF with
      ZERO global post-scaling (`ref_means` removed completely), guaranteeing
      that future years' labels have zero effect on earlier training features.
    - When `asof_maturity=True`, historical citation labels (`Y1`, `Y2`) are strictly
      masked to `NaN` unless `t0_days_j + 1095.75 <= t_cut` (`t0_date_j + 3 years <= t_i`),
      while co-authorship edges and pre-t0 DBLP/publication counts use `t0_days_j < t_cut`.
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
        "graph_recovered_dblp_max_log1p",
        "graph_recovered_dblp_mean_log1p",
        "graph_y3_author_prior",
    )

    def __init__(
        self,
        n_oof_folds: int = 10,
        asof_maturity: bool = False,
        random_state: int = 42,
    ) -> None:
        self.n_oof_folds = int(n_oof_folds)
        self.asof_maturity = bool(asof_maturity)
        self.random_state = int(random_state)
        self.full_graph_: Optional[Dict[str, Any]] = None
        self.bucket_graphs_: Dict[int, Dict[str, Any]] = {}
        self.oof_cache_: Dict[str, np.ndarray] = {}
        self._corpus_df_: Optional[pd.DataFrame] = None

    @staticmethod
    def _build_graph_dict(
        sub_df: pd.DataFrame,
        t_cut: Optional[float] = None,
        enforce_maturity: bool = False,
    ) -> Dict[str, Any]:
        if t_cut is not None and "t0_days" in sub_df.columns:
            pre_df = sub_df.loc[sub_df["t0_days"] < t_cut].copy()
        else:
            pre_df = sub_df.copy()

        if pre_df.empty:
            return {
                "sc1": {}, "sc2": {}, "scl1": {}, "scl2": {}, "sc3": {},
                "inst_sc1": {}, "inst_sc2": {}, "papers": {}, "cnt": {}, "pre_cnt": {},
                "dmax": {}, "dmean": {}, "g_raw1": 18.0, "g_raw2": 2.4, "g_l1": 2.0, "g_l2": 0.7,
            }

        y1_raw = pre_df["s2_citation_count_3y"].to_numpy(dtype=float).copy()
        y2_raw = pre_df["s2_influential_citation_count_3y"].to_numpy(dtype=float).copy()
        y3_raw = pre_df["openreview_mean_rating"].to_numpy(dtype=float).copy()
        yr_raw = pre_df["submission_year"].to_numpy(dtype=int)

        if enforce_maturity and "t0_days" in pre_df.columns:
            eff_cut = float(t_cut) if t_cut is not None else float(pre_df["t0_days"].max())
            t0_d = pre_df["t0_days"].to_numpy(dtype=float)
            unmat_3y = (t0_d + 1095.75) > eff_cut
            unmat_y3 = (t0_d + 120.0) > eff_cut
            y1_raw[unmat_3y] = np.nan
            y2_raw[unmat_3y] = np.nan
            y3_raw[unmat_y3] = np.nan

        y1_imp = np.where(
            np.isfinite(y1_raw),
            y1_raw,
            np.where(np.isfinite(y2_raw), y2_raw * 6.0, np.nan),
        )

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
        pre_cnt: Dict[str, int] = defaultdict(int)
        coauth: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        inst_v1: Dict[str, List[float]] = defaultdict(list)
        inst_v2: Dict[str, List[float]] = defaultdict(list)
        has_inst_col = "parsed_institutions" in pre_df.columns
        inst_lists = pre_df["parsed_institutions"].tolist() if has_inst_col else [[] for _ in range(len(pre_df))]

        dmax_arr = pre_df["dblp_team_max_publication_count_pre_t0"].to_numpy(dtype=float)
        dmean_arr = pre_df["dblp_team_mean_publication_count_pre_t0"].to_numpy(dtype=float)

        for row_idx, (y1, y2, y3, yr, dx, dm, aus, insts) in enumerate(zip(
            y1_imp, y2_raw, y3_raw, yr_raw, dmax_arr, dmean_arr, pre_df["parsed_authors"], inst_lists
        )):
            for a in aus:
                pre_cnt[a] += 1
                a_papers[a].add(row_idx)
            if np.isfinite(y1):
                c1 = min(float(y1), 750.0)
                l1 = float(np.log1p(max(0.0, float(y1))))
                for a in aus:
                    a_v1[a].append(c1)
                    a_l1[a].append(l1)
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

        valid_y1 = y1_imp[np.isfinite(y1_imp)]
        valid_y2 = y2_raw[np.isfinite(y2_raw)]
        g_l1 = float(np.log1p(np.clip(valid_y1, 0.0, None)).mean()) if len(valid_y1) else 2.0
        g_l2 = float(np.log1p(np.clip(valid_y2, 0.0, None)).mean()) if len(valid_y2) else 0.7
        g_raw1 = float(np.clip(valid_y1, 0.0, 750.0).mean()) if len(valid_y1) else 18.0
        g_raw2 = float(np.clip(valid_y2, 0.0, 120.0).mean()) if len(valid_y2) else 2.4

        dmax_map = {a: float(np.max(v)) for a, v in a_dmax.items()}
        dmean_map = {a: float(np.mean(v)) for a, v in a_dmean.items()}

        p_norm = 2.4
        min_matured_rows = 5 if enforce_maturity else 50
        if len(valid_y1) >= min_matured_rows:
            m1 = {a: float((np.mean(np.array(v) ** p_norm)) ** (1.0 / p_norm)) for a, v in a_v1.items()}
            m2 = {
                a: float(
                    0.55 * (m1.get(a, 0.0) / 7.5)
                    + 0.45 * ((np.mean(np.array(a_v2[a]) ** p_norm)) ** (1.0 / p_norm) if a in a_v2 else 0.0)
                )
                for a in (set(a_v1.keys()) | set(a_v2.keys()))
            }
        else:
            # 100% label-free pre-t0 DBLP + corpus activity prior when 0 matured labels exist (2017-2019 under 3y maturity)
            m1, m2 = {}, {}
            for a, pc in pre_cnt.items():
                dx = dmax_map.get(a, 18.0)
                mult = float(np.clip(((np.log1p(dx) / np.log1p(18.0)) ** 0.70) * (1.0 + 0.25 * np.log1p(pc)), 0.45, 2.8))
                m1[a] = 18.0 * mult
                m2[a] = 2.4 * mult

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

        if enforce_maturity and len(valid_y1) >= min_matured_rows:
            for a in list(sc1.keys()):
                pc = pre_cnt.get(a, 1)
                dx = dmax_map.get(a, 18.0)
                pf1 = g_raw1 * ((np.log1p(dx) / np.log1p(18.0)) ** 0.75) * (1.0 + 0.30 * np.log1p(pc))
                pf2 = g_raw2 * ((np.log1p(dx) / np.log1p(18.0)) ** 0.75) * (1.0 + 0.30 * np.log1p(pc))
                sc1[a] = 0.75 * sc1[a] + 0.25 * pf1
                sc2[a] = 0.75 * sc2[a] + 0.25 * pf2

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
            "pre_cnt": dict(pre_cnt),
            "dmax": dmax_map,
            "dmean": dmean_map,
            "g_raw1": g_raw1,
            "g_raw2": g_raw2,
            "g_l1": g_l1,
            "g_l2": g_l2,
        }

    @staticmethod
    def _transform_with_graph(eval_df: pd.DataFrame, g: Dict[str, Any]) -> np.ndarray:
        sc1, sc2, scl1, scl2, sc3 = g["sc1"], g["sc2"], g["scl1"], g["scl2"], g["sc3"]
        inst_sc1, inst_sc2 = g.get("inst_sc1", {}), g.get("inst_sc2", {})
        cnt = g.get("pre_cnt") or g["cnt"]
        dmax_map, dmean_map = g["dmax"], g["dmean"]
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
            pre_known_aus = [a for a in aus if a in cnt]
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
                eff_known = pre_known_aus if len(pre_known_aus) >= len(known_aus) else known_aus
                if len(eff_known) >= 2 and papers_map:
                    union_p = set().union(*(papers_map.get(a, set()) for a in eff_known))
                    sum_p = sum(len(papers_map.get(a, set())) for a in eff_known)
                    div_factor = float((len(union_p) / max(1, sum_p)) ** 0.45) if sum_p > 0 else 1.0
                else:
                    div_factor = 1.0
                is_challenge = ("challenge" in title.lower() and len(aus) >= 15) or (len(aus) > 55)
                max_k = 8 if is_challenge else 52
                n_unk = max(0, min(max_k, len(aus)) - len(known_aus))
                supp = 0.20 if (len(eff_known) == 1 and len(aus) >= 7) else 1.0

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
                    float(len(eff_known) / n_total),
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
                    float(np.log1p(np.sum(kc))),
                    float(len(pre_known_aus) / n_total),
                    dx_log,
                    dm_log,
                    s3_val,
                ]
        return out

    def fit(self, corpus_df: pd.DataFrame) -> "AuthorCollaborationGraphEncoder":
        corpus_df = corpus_df.reset_index(drop=True)
        self._corpus_df_ = corpus_df
        max_cut = float(corpus_df["t0_days"].max() + 1.0)
        self.full_graph_ = self._build_graph_dict(
            corpus_df, t_cut=max_cut, enforce_maturity=self.asof_maturity
        )
        self.oof_cache_ = {}
        self.bucket_graphs_ = {}

        half_yr = (
            (corpus_df["submission_year"].to_numpy(dtype=int) - 2017) * 2
            + (corpus_df["t0_month"].to_numpy(dtype=float) > 6.0).astype(int)
        )
        unique_hy = sorted(np.unique(half_yr))
        oof_mat = np.zeros((len(corpus_df), len(self.FEATURE_NAMES)), dtype=np.float64)

        for hy in unique_hy:
            m_hy = half_yr == hy
            idx_hy = np.flatnonzero(m_hy)
            if hy <= unique_hy[0]:
                sub0 = corpus_df.iloc[idx_hy].reset_index(drop=True)
                if len(sub0) >= 2:
                    n_folds = min(self.n_oof_folds, len(sub0))
                    rng = np.random.RandomState(self.random_state)
                    perm = np.arange(len(sub0))
                    rng.shuffle(perm)
                    mat0 = np.zeros((len(sub0), len(self.FEATURE_NAMES)), dtype=np.float64)
                    t_c0 = float(sub0["t0_days"].min()) if self.asof_maturity else float(sub0["t0_days"].max() + 1.0)
                    for f_idx in np.array_split(perm, n_folds):
                        mk = np.ones(len(sub0), dtype=bool)
                        mk[f_idx] = False
                        g0 = self._build_graph_dict(
                            sub0.iloc[mk],
                            t_cut=t_c0,
                            enforce_maturity=self.asof_maturity,
                        )
                        mat0[f_idx] = self._transform_with_graph(sub0.iloc[f_idx], g0)
                    oof_mat[idx_hy] = mat0
            else:
                prev_df = corpus_df.loc[half_yr < hy].reset_index(drop=True)
                t_c = float(corpus_df.loc[m_hy, "t0_days"].min())
                g_prev = self._build_graph_dict(
                    prev_df, t_cut=t_c, enforce_maturity=self.asof_maturity
                )
                oof_mat[idx_hy] = self._transform_with_graph(corpus_df.iloc[idx_hy], g_prev)

        # Note: NO global ref_means scaling is applied here, preserving 100% causal isolation
        # so future windows can never alter historical windows' features.
        for pid, row_vec in zip(corpus_df["paper_id"].astype(str), oof_mat):
            self.oof_cache_[pid] = row_vec

        return self

    def _get_bucket_graph(self, b: int) -> Dict[str, Any]:
        b_key = int(np.clip(b, 18, 36))
        if b_key not in self.bucket_graphs_:
            assert self._corpus_df_ is not None
            t_c = b_key * 60.875
            self.bucket_graphs_[b_key] = self._build_graph_dict(
                self._corpus_df_, t_cut=t_c, enforce_maturity=True
            )
        return self.bucket_graphs_[b_key]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.full_graph_ is None:
            raise RuntimeError("AuthorCollaborationGraphEncoder is not fitted.")
        pids = frame["paper_id"].astype(str).tolist()
        if all(pid in self.oof_cache_ for pid in pids):
            return np.vstack([self.oof_cache_[pid] for pid in pids])

        if self.asof_maturity and "t0_bucket" in frame.columns and self._corpus_df_ is not None:
            full_mat = np.zeros((len(frame), len(self.FEATURE_NAMES)), dtype=np.float64)
            buckets = frame["t0_bucket"].to_numpy(dtype=int)
            for b in np.unique(buckets):
                mb = buckets == b
                full_mat[mb] = self._transform_with_graph(
                    frame.iloc[mb], self._get_bucket_graph(int(b))
                )
        else:
            full_mat = self._transform_with_graph(frame, self.full_graph_)

        for i, pid in enumerate(pids):
            if pid in self.oof_cache_:
                full_mat[i] = self.oof_cache_[pid]
        return full_mat


class FeatureBuilder:
    """Fit all learned text/category/graph/reduction steps on training rows only
    using strictly causal As-Of Temporal Encoding.
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
    _CORPUS_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def __init__(
        self,
        use_bge: bool = True,
        bge_dim: int = 128,
        tfidf_max_features: int = 8000,
        tfidf_min_df: int = 3,
        category_min_count: int = 15,
        strict_t0: bool = False,
        asof_maturity: bool = False,
        random_state: int = 42,
    ) -> None:
        self.use_bge = bool(use_bge)
        self.bge_dim = int(bge_dim)
        self.tfidf_max_features = int(tfidf_max_features)
        self.tfidf_min_df = int(tfidf_min_df)
        self.category_min_count = int(category_min_count)
        self.strict_t0 = bool(strict_t0)
        self.asof_maturity = bool(asof_maturity)
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

        self._tc_w_y1_raw: Optional[np.ndarray] = None
        self._tc_m_y1_raw: float = 18.0
        self._tc_w_y1_log: Optional[np.ndarray] = None
        self._tc_m_y1_log: float = 2.0
        self._tc_w_y2_raw: Optional[np.ndarray] = None
        self._tc_m_y2_raw: float = 2.4
        self._tc_w_y2_log: Optional[np.ndarray] = None
        self._tc_m_y2_log: float = 0.7
        self._tc_bucket_models_: Dict[int, Tuple[np.ndarray, float, np.ndarray, float, np.ndarray, float, np.ndarray, float]] = {}
        self._oof_topic_cache: Dict[str, np.ndarray] = {}
        self._ref_mean_raw: float = 38.0
        self._frame_block_cache: Dict[Tuple[int, str, str], Tuple[np.ndarray, sp.csr_matrix, np.ndarray]] = {}

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
        blocks = [tfidf, sp.csr_matrix(category * 0.35)]
        if self.use_bge and self.bge_pca is not None and self.embedding_store is not None:
            bge = self.bge_pca.transform(
                np.asarray(self.embedding_store.get(frame), dtype=np.float32)
            )
            blocks.append(sp.csr_matrix((bge * 0.35).astype(np.float32)))
        tc_mat = sp.hstack(blocks, format="csr")
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

        if self.use_bge:
            if embeddings is None:
                raise ValueError("use_bge=True requires an EmbeddingStore.")
            self.embedding_store = embeddings
            self.embedding_model_name = embeddings.model_name

        y1_digest = _array_digest(ref_corpus["s2_citation_count_3y"].to_numpy(dtype=float))
        y2_digest = _array_digest(ref_corpus["s2_influential_citation_count_3y"].to_numpy(dtype=float))
        cache_key = (
            len(ref_corpus),
            str(ref_corpus["paper_id"].iloc[0]),
            str(ref_corpus["paper_id"].iloc[-1]),
            y1_digest,
            y2_digest,
            self.strict_t0,
            self.asof_maturity,
            self.has_external_,
            self.use_bge,
            self.bge_dim,
            self.tfidf_max_features,
            self.category_min_count,
        )
        if cache_key in self._CORPUS_CACHE:
            cached = self._CORPUS_CACHE[cache_key]
            self.bge_pca = cached["bge_pca"]
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
            self._tc_bucket_models_ = cached["tc_bucket_models"]
            self._oof_topic_cache = cached["oof_topic_cache"]
        else:
            if self.use_bge and self.embedding_store is not None:
                raw = self.embedding_store.get(ref_corpus)
                n_components = min(self.bge_dim, raw.shape[1], max(2, len(ref_corpus) - 1))
                if SKLEARN_AVAILABLE:
                    self.bge_pca = PCA(
                        n_components=n_components, random_state=self.random_state
                    ).fit(raw)
                else:
                    self.bge_pca = _FallbackPCA(
                        n_components=n_components, random_state=self.random_state
                    ).fit(raw)
            self._fit_corpus_level(ref_corpus)
            self._CORPUS_CACHE[cache_key] = {
                "bge_pca": self.bge_pca,
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
                "tc_bucket_models": self._tc_bucket_models_,
                "oof_topic_cache": self._oof_topic_cache,
            }

        self._frame_block_cache = {}
        dense_train, _, _ = self._get_or_build_blocks(frame)
        dense_train_lin = self._log_compress_dense(dense_train)
        if self.asof_maturity:
            min_yr = int(frame["submission_year"].min())
            early_mask = (frame["submission_year"] == min_yr).to_numpy()
            fit_slice = dense_train_lin[early_mask].copy() if int(np.sum(early_mask)) >= 5 else dense_train_lin.copy()
            self.numeric_imputer = _FallbackImputer().fit(fit_slice)
            imputed = self.numeric_imputer.transform(fit_slice)
            self.numeric_scaler = _FallbackScaler().fit(imputed)
            # For purely t0-observable base metadata columns (0..n_base-1), use full frame's t0 scale (zero label dependence)
            n_base = len(self.numeric_cols)
            full_base_imp = _FallbackImputer().fit(dense_train_lin[:, :n_base]).transform(dense_train_lin[:, :n_base])
            full_base_sc = _FallbackScaler().fit(full_base_imp)
            self.numeric_imputer.statistics_[:n_base] = np.nanmedian(dense_train_lin[:, :n_base], axis=0)
            self.numeric_scaler.center_[:n_base] = full_base_sc.center_
            self.numeric_scaler.scale_[:n_base] = full_base_sc.scale_
        else:
            self.numeric_imputer = _FallbackImputer().fit(dense_train_lin)
            imputed = self.numeric_imputer.transform(dense_train_lin)
            self.numeric_scaler = _FallbackScaler().fit(imputed)

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

    @staticmethod
    def _solve_tc_lsqr(
        tc_sub: sp.csr_matrix,
        v1_c: np.ndarray,
        v1_l: np.ndarray,
        v2_c: np.ndarray,
        v2_l: np.ndarray,
    ) -> Tuple[np.ndarray, float, np.ndarray, float, np.ndarray, float, np.ndarray, float]:
        m1r = float(np.mean(v1_c))
        w1r = lsqr(tc_sub, v1_c - m1r, damp=8.0)[0]
        m1l = float(np.mean(v1_l))
        w1l = lsqr(tc_sub, v1_l - m1l, damp=6.0)[0]
        m2r = float(np.mean(v2_c))
        w2r = lsqr(tc_sub, v2_c - m2r, damp=8.0)[0]
        m2l = float(np.mean(v2_l))
        w2l = lsqr(tc_sub, v2_l - m2l, damp=6.0)[0]
        return w1r, m1r, w1l, m1l, w2r, m2r, w2l, m2l

    def _fit_corpus_level(self, ref_corpus: pd.DataFrame) -> None:
        self.author_graph = AuthorCollaborationGraphEncoder(
            n_oof_folds=10,
            asof_maturity=self.asof_maturity,
            random_state=self.random_state,
        ).fit(ref_corpus)
        self.emerging_encoder = EmergingParadigmEncoder(
            asof_maturity=self.asof_maturity,
            random_state=self.random_state,
        ).fit(ref_corpus)
        self._ref_mean_raw = float(
            ref_corpus["s2_reference_count_current"].fillna(35.0).clip(0.0, 250.0).mean()
        )

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

        tc_all, _, _ = self._build_tc_sparse(ref_corpus)
        t0_d = ref_corpus["t0_days"].to_numpy(dtype=float)
        max_cut = float(t0_d.max() + 1.0)

        y1_raw = ref_corpus["s2_citation_count_3y"].to_numpy(dtype=float).copy()
        y2_raw = ref_corpus["s2_influential_citation_count_3y"].to_numpy(dtype=float).copy()
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

        dmax_arr = (
            pd.to_numeric(ref_corpus["dblp_team_max_publication_count_pre_t0"], errors="coerce")
            .fillna(18.0)
            .to_numpy(dtype=float)
        )
        p1_cap = np.clip(18.0 * ((np.log1p(dmax_arr) / np.log1p(18.0)) ** 0.70), 2.0, 120.0)
        p1_log = np.log1p(p1_cap)
        p2_cap = np.clip(2.4 * ((np.log1p(dmax_arr) / np.log1p(18.0)) ** 0.70), 0.2, 20.0)
        p2_log = np.log1p(p2_cap)

        def _fit_at_cutoff(pool_mask: np.ndarray, t_c: float) -> Tuple[np.ndarray, float, np.ndarray, float, np.ndarray, float, np.ndarray, float]:
            if self.asof_maturity:
                mat_mask = pool_mask & ((t0_d + 1095.75) <= t_c) & np.isfinite(y1_raw)
                if int(np.sum(mat_mask)) >= 5:
                    return self._solve_tc_lsqr(
                        tc_all[mat_mask],
                        y1_cap[mat_mask], y1_log[mat_mask],
                        y2_cap[mat_mask], y2_log[mat_mask],
                    )
                pre_m = pool_mask & (t0_d < max(1.0, t_c))
                if not np.any(pre_m):
                    pre_m = pool_mask
                return self._solve_tc_lsqr(
                    tc_all[pre_m],
                    p1_cap[pre_m], p1_log[pre_m],
                    p2_cap[pre_m], p2_log[pre_m],
                )
            return self._solve_tc_lsqr(
                tc_all[pool_mask],
                y1_cap[pool_mask], y1_log[pool_mask],
                y2_cap[pool_mask], y2_log[pool_mask],
            )

        all_mask = np.ones(len(ref_corpus), dtype=bool)
        (
            self._tc_w_y1_raw, self._tc_m_y1_raw,
            self._tc_w_y1_log, self._tc_m_y1_log,
            self._tc_w_y2_raw, self._tc_m_y2_raw,
            self._tc_w_y2_log, self._tc_m_y2_log,
        ) = _fit_at_cutoff(all_mask, max_cut)

        self._oof_topic_cache = {}
        self._tc_bucket_models_ = {}
        n_c = len(ref_corpus)
        if n_c >= 5:
            half_yr = (
                (ref_corpus["submission_year"].to_numpy(dtype=int) - 2017) * 2
                + (ref_corpus["t0_month"].to_numpy(dtype=float) > 6.0).astype(int)
            )
            unique_hy = sorted(np.unique(half_yr))
            oof_t = np.zeros((n_c, 4), dtype=np.float64)

            for hy in unique_hy:
                m_hy = half_yr == hy
                idx_hy = np.flatnonzero(m_hy)
                if hy <= unique_hy[0]:
                    rng = np.random.RandomState(self.random_state)
                    perm = np.arange(len(idx_hy))
                    rng.shuffle(perm)
                    t_c0 = float(ref_corpus.loc[m_hy, "t0_days"].min()) if self.asof_maturity else float(ref_corpus.loc[m_hy, "t0_days"].max() + 1.0)
                    n_splits = min(5, len(idx_hy))
                    if n_splits >= 2:
                        for f_sub in np.array_split(perm, n_splits):
                            f_idx = idx_hy[f_sub]
                            tr_idx = np.setdiff1d(idx_hy, f_idx)
                            tr_mask = np.zeros(n_c, dtype=bool)
                            tr_mask[tr_idx] = True
                            w1r, m1r, w1l, m1l, w2r, m2r, w2l, m2l = _fit_at_cutoff(tr_mask, t_c0)
                            tc_va = tc_all[f_idx]
                            oof_t[f_idx, 0] = tc_va @ w1r + m1r
                            oof_t[f_idx, 1] = tc_va @ w1l + m1l
                            oof_t[f_idx, 2] = tc_va @ w2r + m2r
                            oof_t[f_idx, 3] = tc_va @ w2l + m2l
                else:
                    tr_mask = half_yr < hy
                    t_c = float(ref_corpus.loc[m_hy, "t0_days"].min())
                    w1r, m1r, w1l, m1l, w2r, m2r, w2l, m2l = _fit_at_cutoff(tr_mask, t_c)
                    tc_va = tc_all[idx_hy]
                    oof_t[idx_hy, 0] = tc_va @ w1r + m1r
                    oof_t[idx_hy, 1] = tc_va @ w1l + m1l
                    oof_t[idx_hy, 2] = tc_va @ w2r + m2r
                    oof_t[idx_hy, 3] = tc_va @ w2l + m2l

            for pid, vec in zip(ref_corpus["paper_id"].astype(str), oof_t):
                self._oof_topic_cache[pid] = vec

        if self.asof_maturity:
            # Precompute the 4 half-year test cutoffs (buckets 24, 27, 30, 33) and map bi-monthly buckets 18..36
            key_cuts = {18: 1095.75, 21: 1277.0, 24: 1461.0, 27: 1642.5, 30: 1826.0, 33: 2007.5}
            computed_cuts = {k: _fit_at_cutoff(all_mask, tc_val) for k, tc_val in key_cuts.items()}
            for b in range(18, 37):
                nearest_k = max(k for k in key_cuts if k <= b)
                self._tc_bucket_models_[b] = computed_cuts[nearest_k]

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

        if self.asof_maturity and "t0_bucket" in frame.columns and self._tc_bucket_models_:
            out = np.zeros((len(frame), 4), dtype=np.float64)
            buckets = frame["t0_bucket"].to_numpy(dtype=int)
            for b in np.unique(buckets):
                mb = buckets == b
                b_key = int(np.clip(b, min(self._tc_bucket_models_.keys()), max(self._tc_bucket_models_.keys())))
                w1r, m1r, w1l, m1l, w2r, m2r, w2l, m2l = self._tc_bucket_models_[b_key]
                tc_sub = tc_mat[mb]
                out[mb, 0] = tc_sub @ w1r + m1r
                out[mb, 1] = tc_sub @ w1l + m1l
                out[mb, 2] = tc_sub @ w2r + m2r
                out[mb, 3] = tc_sub @ w2l + m2l
        else:
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

    def _get_or_build_blocks(self, frame: pd.DataFrame) -> Tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
        if not hasattr(self, "_frame_block_cache") or self._frame_block_cache is None:
            self._frame_block_cache = {}
        key = (len(frame), str(frame["paper_id"].iloc[0]), str(frame["paper_id"].iloc[-1]))
        cached = self._frame_block_cache.get(key)
        if cached is not None:
            return cached
        tc_mat, category, tfidf = self._build_tc_sparse(frame)
        dense = self._build_dense_block(frame, tc_mat=tc_mat)
        res = (dense, tfidf, category)
        self._frame_block_cache[key] = res
        return res

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

        anchor_m1_raw = 18.0
        anchor_m1_log = 2.0
        anchor_m2_raw = 2.4
        anchor_m2_log = 0.7

        topic_mult_1 = np.clip(
            1.0 + 0.25 * (t_y1 - anchor_m1_raw) / anchor_m1_raw + 0.06 * em_y1 + 0.06 * colon_flag + 0.06 * code_flag + ext_boost_1,
            0.45,
            2.8,
        )
        topic_mult_2 = np.clip(
            1.0 + 0.45 * (t_y2 - anchor_m2_raw) / anchor_m2_raw + 0.08 * em_y2 + 0.16 * colon_flag + 0.08 * code_flag + ext_boost_2,
            0.40,
            3.0,
        )

        t_l1 = np.clip(topic_feats[:, 1], 0.1, 6.0)
        t_l2 = np.clip(topic_feats[:, 3], 0.1, 6.0)
        l1_factor = (t_l1 / anchor_m1_log) ** 0.25
        l2_factor = (t_l2 / anchor_m2_log) ** 0.25

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
            base_m1 = np.maximum(0.0, g_y1 * topic_mult_1 * ref_mult_1 + 0.25 * np.clip(t_y1 - anchor_m1_raw, -10.0, 120.0) + 0.12 * (ref_raw - rm))
            base_m2 = np.maximum(0.0, g_y2 * topic_mult_2 * ref_mult_2 + 0.90 * np.clip(t_y2 - anchor_m2_raw, -2.0, 20.0))
        else:
            base_m1 = np.maximum(0.0, g_y1 * topic_mult_1 + 0.25 * np.clip(t_y1 - anchor_m1_raw, -10.0, 120.0))
            base_m2 = np.maximum(0.0, g_y2 * topic_mult_2 + 0.90 * np.clip(t_y2 - anchor_m2_raw, -2.0, 20.0))

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
        if self.tfidf is None or self.numeric_imputer is None:
            raise RuntimeError("FeatureBuilder is not fitted.")
        dense, _, _ = self._get_or_build_blocks(frame)
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

        dense_raw, tfidf, category = self._get_or_build_blocks(frame)
        dense = dense_raw.copy()
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
        self._frame_block_cache = {}
        if self.author_graph is not None:
            self.author_graph._corpus_df_ = None
        if self.emerging_encoder is not None:
            self.emerging_encoder._corpus_df_ = None
        return self

    def get_feature_names_out(self) -> list[str]:
        if self._feature_names is None:
            raise RuntimeError("FeatureBuilder is not fitted.")
        return list(self._feature_names)

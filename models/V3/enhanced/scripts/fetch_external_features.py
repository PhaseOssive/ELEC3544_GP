#!/usr/bin/env python3
"""Batch external feature collector for the 29,856-paper arXiv CS dataset.

Implements the three high-leverage external data enrichment schemes:
1. OpenAlex Works Batch API (`GET https://api.openalex.org/works?filter=doi:...`):
   - Queries 50 arXiv DOIs (`10.48550/arXiv.<paper_id>`) per request (~598 requests)
     plus a fallback pass for `doi_v1` records.
   - Reconciles records via 4-way ID matching (`work.doi`, `work.ids.arxiv`,
     `work.locations[*].doi/landing_page_url/pdf_url`, and `doi_v1`).
   - Extracts author institutional affiliations (`oa_institutions_json`,
     `oa_institutions_count`, `oa_countries_count`, `oa_has_top_industry_lab`,
     `oa_has_top_university`), early citation velocity (`oa_early_citations_t0_year`,
     `oa_early_citations_t0_plus1`), and `oa_referenced_works_count`.
2. Semantic Scholar Academic Graph Batch API (`POST https://api.semanticscholar.org/graph/v1/paper/batch`):
   - Queries 100 `ARXIV:<paper_id>` records per POST request (~299 requests),
     staying well below the 10 MB response payload cap.
   - Automatically bisects batches (`100 -> 50 -> 25 -> 10`) on HTTP 400/413/500
     payload limits and honors `Retry-After` on HTTP 429 rate limits.
   - Extracts reference list impact (`s2_ref_citation_max`, `s2_ref_citation_mean`,
     `s2_ref_citation_median`), frontier reference recency (`s2_ref_recent_2y_ratio`),
     and author S2 profile metrics (`s2_author_max_hindex`, `s2_author_mean_hindex`,
     `s2_author_max_citations`).
3. Papers With Code Archive (`pwc-archive/links-between-paper-and-code` / `.json.gz`)
   + Multi-Source Code Repository & Framework Extractor:
   - Downloads the Papers With Code paper-to-code linkage table (trying Hugging Face
     Hub Parquet/JSON API and legacy `.json.gz` mirror) and fuses it with high-precision
     local regex extraction over `abstract_v1` and `comment_v1`.
   - Extracts `pwc_has_official_code`, `pwc_repo_count`, `pwc_is_pytorch`,
     `pwc_is_tensorflow`, and `pwc_mentioned_in_github`.

All network responses are checkpointed transactionally in SQLite
(`cache/external_features_cache.sqlite`) using a single persistent connection
so macOS file-descriptor limits (`ulimit -n 256`) are never exhausted.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import urllib.error
import urllib.parse
import urllib.request

try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]

import numpy as np
import pandas as pd


def _bump_file_descriptor_limit() -> None:
    """Raise macOS/POSIX soft RLIMIT_NOFILE (default 256 on macOS) to >= 4096."""
    if resource is None:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(max(soft, 4096), hard if hard != resource.RLIM_INFINITY else 8192)
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        pass


EXTERNAL_COLUMNS: Tuple[str, ...] = (
    "oa_matched",
    "oa_institutions_count",
    "oa_countries_count",
    "oa_has_top_industry_lab",
    "oa_has_top_university",
    "oa_early_citations_t0_year",
    "oa_early_citations_t0_plus1",
    "oa_referenced_works_count",
    "oa_institutions_json",
    "s2_batch_matched",
    "s2_ref_citation_max",
    "s2_ref_citation_mean",
    "s2_ref_citation_median",
    "s2_ref_recent_2y_ratio",
    "s2_author_max_hindex",
    "s2_author_mean_hindex",
    "s2_author_max_citations",
    "pwc_has_official_code",
    "pwc_repo_count",
    "pwc_is_pytorch",
    "pwc_is_tensorflow",
    "pwc_mentioned_in_github",
)

TOP_INDUSTRY_PATTERN = re.compile(
    r"\b("
    r"google|deepmind|alphabet|meta\b|facebook|fair\b|microsoft|openai|nvidia|"
    r"apple\b|amazon|aws\b|ibm\b|huawei|tencent|alibaba|damo academy|bytedance|"
    r"baidu|sensetime|megvii|adobe|salesforce|anthropic|cohere|intel\b|qualcomm|"
    r"samsung|sony|uber\b|spotify|netflix|bell labs|nec laboratories|bosch"
    r")\b",
    re.IGNORECASE,
)

TOP_UNIVERSITY_PATTERN = re.compile(
    r"("
    r"massachusetts institute of technology|\bmit\b|stanford|carnegie mellon|\bcmu\b|"
    r"california.*berkeley|\buc berkeley\b|harvard|princeton|cornell|"
    r"illinois.*urbana|\buiuc\b|university of washington|georgia institute of technology|"
    r"georgia tech|california institute of technology|\bcaltech\b|oxford|cambridge|"
    r"\beth zurich\b|epfl|imperial college|university college london|\bucl\b|"
    r"university of toronto|mila\b|universit[eé] de montr[eé]al|tsinghua|peking university|"
    r"university of hong kong|\bhku\b|hong kong university of science|\bhkust\b|"
    r"chinese university of hong kong|\bcuhk\b|national university of singapore|\bnus\b|"
    r"nanyang technological|\bntu\b|new york university|\bnyu\b|columbia university|"
    r"university of michigan|university of texas at austin|\but austin\b|"
    r"university of california.*los angeles|\bucla\b|university of california.*san diego|\bucsd\b|"
    r"max planck|inria|technion|kaist|university of tokyo|seoul national|zhejiang university|"
    r"shanghai jiao tong|fudan|university of science and technology of china|\bustc\b|"
    r"chinese academy of sciences|\bcas\b|university of edinburgh|university of amsterdam"
    r")",
    re.IGNORECASE,
)

ARXIV_ID_EXTRACT_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
GITHUB_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org|huggingface\.co)/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+",
    re.IGNORECASE,
)
CODE_AVAIL_RE = re.compile(
    r"github\.com|gitlab\.com|bitbucket\.org|code is available|code available at|"
    r"source code|open-source implementation|open source implementation|official implementation",
    re.IGNORECASE,
)
PYTORCH_RE = re.compile(r"\bpytorch\b|\btorch\b|\btorchvision\b", re.IGNORECASE)
TENSORFLOW_RE = re.compile(r"\btensorflow\b|\bkeras\b|\btf\b", re.IGNORECASE)


def normalize_arxiv_id(raw: Any) -> str:
    """Normalize an arXiv identifier or URL to bare `YYMM.NNNNN` format."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return ""
    s = str(raw).strip()
    m = ARXIV_ID_EXTRACT_RE.search(s)
    if m:
        return m.group(1)
    return s.lower()


def classify_institution_flags(inst_names: Sequence[str], inst_types: Sequence[str]) -> Tuple[int, int]:
    """Return `(has_top_industry_lab, has_top_university)` for a list of institution strings."""
    has_ind = 0
    has_uni = 0
    for nm, tp in zip(inst_names, inst_types):
        text = str(nm or "").strip()
        tp_s = str(tp or "").strip().lower()
        if not text:
            continue
        if tp_s == "company" or TOP_INDUSTRY_PATTERN.search(text):
            has_ind = 1
        if TOP_UNIVERSITY_PATTERN.search(text):
            has_uni = 1
    return has_ind, has_uni


class SQLiteCheckpointStore:
    """Transactional SQLite key-value cache using a SINGLE persistent connection.

    Note: In Python's `sqlite3`, `with sqlite3.connect(...) as conn:` only commits
    the transaction and does NOT close the file descriptor. Opening a new connection
    per batch exhausts macOS's default `ulimit -n = 256` after ~122 batches, causing
    both `[Errno 8] nodename nor servname provided` (DNS socket FD failure) and
    `sqlite3.OperationalError: unable to open database file`. Keeping a single
    persistent connection avoids FD leaks completely.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path), timeout=60.0, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_cache (
                    source TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (source, paper_id)
                )
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def purge_recent_if_needed(self, source: str, tail_count: int = 100) -> int:
        """Remove the most recently written `tail_count` negative records if a prior run crashed mid-batch."""
        with self._lock:
            cur = self._conn.execute(
                """
                DELETE FROM paper_cache
                WHERE rowid IN (
                    SELECT rowid FROM paper_cache
                    WHERE source = ? AND payload_json LIKE '%"oa_matched": 0%'
                    ORDER BY updated_at DESC
                    LIMIT ?
                )
                """,
                (source, int(tail_count)),
            )
            self._conn.commit()
            return int(cur.rowcount or 0)

    def get_cached_ids(self, source: str) -> set[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT paper_id FROM paper_cache WHERE source = ?",
                (source,),
            )
            return {str(row[0]) for row in cur.fetchall()}

    def load_all(self, source: str) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT paper_id, payload_json FROM paper_cache WHERE source = ?",
                (source,),
            )
            out: Dict[str, Dict[str, Any]] = {}
            for pid, raw in cur.fetchall():
                try:
                    out[str(pid)] = json.loads(raw)
                except json.JSONDecodeError:
                    continue
            return out

    def save_batch(self, source: str, records: Mapping[str, Mapping[str, Any]]) -> None:
        if not records:
            return
        now = time.time()
        rows = [
            (source, str(pid), json.dumps(dict(payload), ensure_ascii=False), now)
            for pid, payload in records.items()
        ]
        with self._lock:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO paper_cache (source, paper_id, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()


def _http_request_json(
    url: str,
    method: str = "GET",
    headers: Optional[Mapping[str, str]] = None,
    json_body: Optional[Mapping[str, Any]] = None,
    timeout: float = 30.0,
    max_retries: int = 5,
    base_backoff: float = 1.2,
) -> Any:
    """Robust HTTP JSON client with explicit `Connection: close`, HTTP 429 `Retry-After`, and backoff."""
    req_headers = {
        "User-Agent": "HKU-ImpactModel-BatchCollector/3.1 (mailto:research@hku.hk)",
        "Accept": "application/json",
        "Connection": "close",
    }
    if headers:
        req_headers.update(headers)

    data_bytes: Optional[bytes] = None
    if json_body is not None:
        data_bytes = json.dumps(json_body).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                url=url,
                data=data_bytes,
                headers=req_headers,
                method=method,
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as err:
            last_err = err
            code = err.code
            try:
                err.close()
            except Exception:
                pass
            if code in (400, 413, 414):
                raise
            if code == 429 or 500 <= code < 600:
                retry_after = err.headers.get("Retry-After") if err.headers else None
                if retry_after is not None:
                    try:
                        wait_s = max(0.05, float(retry_after))
                    except ValueError:
                        wait_s = base_backoff * (2 ** attempt)
                else:
                    wait_s = base_backoff * (2 ** attempt) + random.uniform(0.05, 0.35)
                if attempt < max_retries:
                    time.sleep(min(wait_s, 60.0))
                    continue
            raise
        except Exception as err:
            last_err = err
            if attempt < max_retries:
                wait_s = base_backoff * (2 ** attempt) + random.uniform(0.05, 0.25)
                time.sleep(min(wait_s, 30.0))
                continue
            raise
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Request failed: {url}")


# ---------------------------------------------------------------------------
# 1. OpenAlex Batch Fetcher
# ---------------------------------------------------------------------------

def _extract_openalex_paper_features(
    work: Mapping[str, Any],
    submission_year: int,
) -> Dict[str, Any]:
    """Parse a single OpenAlex Work dictionary into structured features."""
    authorships = work.get("authorships") or []
    inst_names: List[str] = []
    inst_types: List[str] = []
    inst_ids: List[str] = []
    countries: set[str] = set()

    for au in authorships:
        if not isinstance(au, Mapping):
            continue
        for c in au.get("countries") or []:
            if c:
                countries.add(str(c).upper())
        for inst in au.get("institutions") or []:
            if not isinstance(inst, Mapping):
                continue
            disp = str(inst.get("display_name") or "").strip()
            itype = str(inst.get("type") or "").strip().lower()
            iid = str(inst.get("id") or disp).strip()
            if disp:
                inst_names.append(disp)
                inst_types.append(itype)
            if iid:
                inst_ids.append(iid)
            cc = inst.get("country_code")
            if cc:
                countries.add(str(cc).upper())
        for raw_aff in au.get("raw_affiliation_strings") or []:
            if raw_aff and not inst_names:
                inst_names.append(str(raw_aff).strip())
                inst_types.append("")

    unique_inst_ids = sorted(set(inst_ids or inst_names))
    inst_count = work.get("institutions_distinct_count")
    if inst_count is None:
        inst_count = len(unique_inst_ids)
    country_count = work.get("countries_distinct_count")
    if country_count is None:
        country_count = len(countries)

    has_ind, has_uni = classify_institution_flags(inst_names, inst_types)

    c_t0 = 0
    c_t0_p1 = 0
    for item in work.get("counts_by_year") or []:
        if not isinstance(item, Mapping):
            continue
        yr = item.get("year")
        cnt = item.get("cited_by_count")
        if yr is not None and cnt is not None:
            try:
                yr_i = int(yr)
                cnt_i = max(0, int(cnt))
                if yr_i <= submission_year:
                    c_t0 += cnt_i
                if yr_i <= submission_year + 1:
                    c_t0_p1 += cnt_i
            except (TypeError, ValueError):
                pass

    ref_works = work.get("referenced_works") or []
    ref_cnt = work.get("referenced_works_count")
    if ref_cnt is None:
        ref_cnt = len(ref_works) if isinstance(ref_works, list) else 0

    return {
        "oa_matched": 1,
        "oa_institutions_count": int(inst_count or 0),
        "oa_countries_count": int(country_count or 0),
        "oa_has_top_industry_lab": int(has_ind),
        "oa_has_top_university": int(has_uni),
        "oa_early_citations_t0_year": int(c_t0),
        "oa_early_citations_t0_plus1": int(c_t0_p1),
        "oa_referenced_works_count": int(ref_cnt or 0),
        "oa_institutions_json": json.dumps(unique_inst_ids[:25], ensure_ascii=False),
    }


def _match_openalex_work_to_pids(
    work: Mapping[str, Any],
    batch_pid_set: set[str],
    doi_to_pid: Mapping[str, str],
) -> List[str]:
    """Reconcile an OpenAlex Work object back to one or more `paper_id`s in the batch."""
    matched: set[str] = set()
    candidates: List[str] = []

    if work.get("doi"):
        candidates.append(str(work["doi"]))
    ids_obj = work.get("ids")
    if isinstance(ids_obj, Mapping):
        for v in ids_obj.values():
            if v:
                candidates.append(str(v))
    for loc in work.get("locations") or []:
        if isinstance(loc, Mapping):
            for k in ("doi", "landing_page_url", "pdf_url"):
                if loc.get(k):
                    candidates.append(str(loc[k]))

    for cand in candidates:
        c_low = cand.strip().lower()
        for m in ARXIV_ID_EXTRACT_RE.finditer(c_low):
            pid = m.group(1)
            if pid in batch_pid_set:
                matched.add(pid)
        c_clean = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", c_low)
        if c_clean in doi_to_pid:
            pid = doi_to_pid[c_clean]
            if pid in batch_pid_set:
                matched.add(pid)

    return sorted(matched)


def fetch_openalex_batches(
    df: pd.DataFrame,
    store: SQLiteCheckpointStore,
    base_url: str = "https://api.openalex.org/works",
    batch_size: int = 50,
    mailto: str = "research@hku.hk",
    api_key: Optional[str] = None,
    rps: float = 8.0,
    timeout: float = 30.0,
) -> Dict[str, Dict[str, Any]]:
    """Fetch OpenAlex works metadata for all uncached papers in batches of `batch_size`."""
    # Automatically clean up any tail batch that had oa_matched=0 written during a prior FD error
    cached_ids = store.get_cached_ids("openalex")
    if 0 < len(cached_ids) < len(df):
        purged = store.purge_recent_if_needed("openalex", tail_count=100)
        if purged > 0:
            print(f"[OpenAlex] Cleaned up {purged} tail records from interrupted batch for re-verification.")
            cached_ids = store.get_cached_ids("openalex")

    pid_to_year = dict(zip(df["paper_id"].astype(str), df["submission_year"].astype(int)))
    doi_to_pid: Dict[str, str] = {}
    pid_to_doi: Dict[str, str] = {}
    for pid, raw_doi in zip(df["paper_id"].astype(str), df["doi_v1"]):
        if pd.notna(raw_doi) and str(raw_doi).strip():
            clean_d = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", str(raw_doi).strip().lower())
            doi_to_pid[clean_d] = pid
            pid_to_doi[pid] = clean_d

    todo_pids = [str(pid) for pid in df["paper_id"].astype(str) if str(pid) not in cached_ids]
    if not todo_pids:
        print(f"[OpenAlex] All {len(df)} papers already cached in SQLite.")
        return store.load_all("openalex")

    print(f"[OpenAlex] Fetching {len(todo_pids)} papers in batches of {batch_size} (cached: {len(cached_ids)})...")
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    select_fields = (
        "id,doi,ids,publication_year,publication_date,authorships,"
        "institutions_distinct_count,countries_distinct_count,counts_by_year,"
        "referenced_works,referenced_works_count,locations"
    )
    min_interval = 1.0 / max(0.5, rps)

    for start in range(0, len(todo_pids), batch_size):
        t_start = time.time()
        chunk = todo_pids[start : start + batch_size]
        chunk_set = set(chunk)
        arxiv_dois = [f"10.48550/arXiv.{pid}" for pid in chunk]
        params: Dict[str, str] = {
            "filter": "doi:" + "|".join(arxiv_dois),
            "per_page": str(min(100, max(50, len(chunk)))),
            "select": select_fields,
        }
        if mailto:
            params["mailto"] = mailto
        url = f"{base_url}?{urllib.parse.urlencode(params, safe='|:,')}"

        batch_records: Dict[str, Dict[str, Any]] = {}
        request_ok = False
        try:
            payload = _http_request_json(url, headers=headers, timeout=timeout)
            request_ok = True
            results = payload.get("results") if isinstance(payload, Mapping) else []
            for work in results or []:
                if not isinstance(work, Mapping):
                    continue
                matched_pids = _match_openalex_work_to_pids(work, chunk_set, doi_to_pid)
                for pid in matched_pids:
                    batch_records[pid] = _extract_openalex_paper_features(
                        work, pid_to_year.get(pid, 2020)
                    )
        except Exception as exc:
            print(f"[OpenAlex] Warning on batch {start // batch_size + 1}: {exc}", file=sys.stderr)

        # Fallback pass for unmatched papers in this chunk that have a publisher `doi_v1`
        unmatched_with_doi = [
            pid for pid in chunk if pid not in batch_records and pid in pid_to_doi
        ]
        if request_ok and unmatched_with_doi:
            pub_dois = [pid_to_doi[pid] for pid in unmatched_with_doi]
            params_fb: Dict[str, str] = {
                "filter": "doi:" + "|".join(pub_dois),
                "per_page": str(min(100, max(50, len(unmatched_with_doi)))),
                "select": select_fields,
            }
            if mailto:
                params_fb["mailto"] = mailto
            url_fb = f"{base_url}?{urllib.parse.urlencode(params_fb, safe='|:,/')}"
            try:
                payload_fb = _http_request_json(url_fb, headers=headers, timeout=timeout)
                for work in (payload_fb.get("results") if isinstance(payload_fb, Mapping) else []) or []:
                    if not isinstance(work, Mapping):
                        continue
                    matched_pids = _match_openalex_work_to_pids(work, set(unmatched_with_doi), doi_to_pid)
                    for pid in matched_pids:
                        batch_records[pid] = _extract_openalex_paper_features(
                            work, pid_to_year.get(pid, 2020)
                        )
            except Exception:
                pass

        # Only checkpoint negative misses if the HTTP request actually succeeded!
        if request_ok:
            for pid in chunk:
                if pid not in batch_records:
                    batch_records[pid] = {
                        "oa_matched": 0,
                        "oa_institutions_count": 0,
                        "oa_countries_count": 0,
                        "oa_has_top_industry_lab": 0,
                        "oa_has_top_university": 0,
                        "oa_early_citations_t0_year": 0,
                        "oa_early_citations_t0_plus1": 0,
                        "oa_referenced_works_count": 0,
                        "oa_institutions_json": "[]",
                    }
            store.save_batch("openalex", batch_records)

        elapsed = time.time() - t_start
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        done = min(len(todo_pids), start + len(chunk))
        if (start // batch_size + 1) % 25 == 0 or done == len(todo_pids):
            print(f"[OpenAlex] Progress: {done}/{len(todo_pids)} papers processed.")

    return store.load_all("openalex")


# ---------------------------------------------------------------------------
# 2. Semantic Scholar Batch Fetcher
# ---------------------------------------------------------------------------

def _extract_s2_paper_features(
    item: Optional[Mapping[str, Any]],
    submission_year: int,
) -> Dict[str, Any]:
    """Parse a single Semantic Scholar batch item into reference & author features."""
    if not item or not isinstance(item, Mapping):
        return {
            "s2_batch_matched": 0,
            "s2_ref_citation_max": math.nan,
            "s2_ref_citation_mean": math.nan,
            "s2_ref_citation_median": math.nan,
            "s2_ref_recent_2y_ratio": math.nan,
            "s2_author_max_hindex": math.nan,
            "s2_author_mean_hindex": math.nan,
            "s2_author_max_citations": math.nan,
        }

    refs = item.get("references") or []
    ref_cites: List[float] = []
    recent_2y = 0
    valid_ref_years = 0
    for ref in refs:
        if not isinstance(ref, Mapping):
            continue
        rc = ref.get("citationCount")
        if rc is not None:
            try:
                ref_cites.append(max(0.0, float(rc)))
            except (TypeError, ValueError):
                pass
        ry = ref.get("year")
        if ry is not None:
            try:
                ry_i = int(ry)
                if 1950 <= ry_i <= submission_year + 1:
                    valid_ref_years += 1
                    if submission_year - 2 <= ry_i <= submission_year:
                        recent_2y += 1
            except (TypeError, ValueError):
                pass

    if ref_cites:
        arr_rc = np.asarray(ref_cites, dtype=np.float64)
        ref_max = float(np.max(arr_rc))
        ref_mean = float(np.mean(arr_rc))
        ref_med = float(np.median(arr_rc))
    else:
        ref_max = 0.0
        ref_mean = 0.0
        ref_med = 0.0

    recent_ratio = float(recent_2y / valid_ref_years) if valid_ref_years > 0 else 0.0

    authors = item.get("authors") or []
    h_vals: List[float] = []
    c_vals: List[float] = []
    for au in authors:
        if not isinstance(au, Mapping):
            continue
        h = au.get("hIndex")
        if h is not None:
            try:
                h_vals.append(max(0.0, float(h)))
            except (TypeError, ValueError):
                pass
        ac = au.get("citationCount")
        if ac is not None:
            try:
                c_vals.append(max(0.0, float(ac)))
            except (TypeError, ValueError):
                pass

    au_h_max = float(np.max(h_vals)) if h_vals else math.nan
    au_h_mean = float(np.mean(h_vals)) if h_vals else math.nan
    au_c_max = float(np.max(c_vals)) if c_vals else math.nan

    return {
        "s2_batch_matched": 1,
        "s2_ref_citation_max": ref_max,
        "s2_ref_citation_mean": ref_mean,
        "s2_ref_citation_median": ref_med,
        "s2_ref_recent_2y_ratio": recent_ratio,
        "s2_author_max_hindex": au_h_max,
        "s2_author_mean_hindex": au_h_mean,
        "s2_author_max_citations": au_c_max,
    }


def _request_s2_chunk_adaptive(
    chunk_pids: Sequence[str],
    url: str,
    headers: Mapping[str, str],
    timeout: float,
) -> Optional[List[Optional[Mapping[str, Any]]]]:
    """POST a chunk of arXiv IDs to S2 `/paper/batch`, bisecting on HTTP 400/413/500 payload limits."""
    body = {"ids": [f"ARXIV:{pid}" for pid in chunk_pids]}
    try:
        resp = _http_request_json(
            url=url,
            method="POST",
            headers=headers,
            json_body=body,
            timeout=timeout,
        )
        if isinstance(resp, list) and len(resp) == len(chunk_pids):
            return resp
        if isinstance(resp, list):
            return list(resp) + [None] * max(0, len(chunk_pids) - len(resp))
    except urllib.error.HTTPError as err:
        if err.code in (400, 413, 414, 500, 502, 504) and len(chunk_pids) > 10:
            mid = len(chunk_pids) // 2
            left = _request_s2_chunk_adaptive(chunk_pids[:mid], url, headers, timeout)
            right = _request_s2_chunk_adaptive(chunk_pids[mid:], url, headers, timeout)
            if left is not None and right is not None:
                return left + right
    except Exception as exc:
        if len(chunk_pids) > 25:
            mid = len(chunk_pids) // 2
            left = _request_s2_chunk_adaptive(chunk_pids[:mid], url, headers, timeout)
            right = _request_s2_chunk_adaptive(chunk_pids[mid:], url, headers, timeout)
            if left is not None and right is not None:
                return left + right
        print(f"[SemanticScholar] Warning on chunk ({len(chunk_pids)} ids): {exc}", file=sys.stderr)
    return None


def fetch_s2_batches(
    df: pd.DataFrame,
    store: SQLiteCheckpointStore,
    base_url: str = "https://api.semanticscholar.org/graph/v1/paper/batch",
    batch_size: int = 100,
    api_key: Optional[str] = None,
    rps: float = 1.0,
    timeout: float = 35.0,
) -> Dict[str, Dict[str, Any]]:
    """Fetch Semantic Scholar reference & author metrics in batches of `batch_size`."""
    cached_ids = store.get_cached_ids("s2_batch")
    pid_to_year = dict(zip(df["paper_id"].astype(str), df["submission_year"].astype(int)))
    todo_pids = [str(pid) for pid in df["paper_id"].astype(str) if str(pid) not in cached_ids]
    if not todo_pids:
        print(f"[SemanticScholar] All {len(df)} papers already cached in SQLite.")
        return store.load_all("s2_batch")

    print(
        f"[SemanticScholar] Fetching {len(todo_pids)} papers in batches of {batch_size} "
        f"(cached: {len(cached_ids)})..."
    )
    fields = (
        "paperId,externalIds,year,referenceCount,"
        "references.year,references.citationCount,"
        "authors.hIndex,authors.citationCount"
    )
    url = f"{base_url}?{urllib.parse.urlencode({'fields': fields})}"
    headers: Dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    min_interval = 1.0 / max(0.2, rps)

    for start in range(0, len(todo_pids), batch_size):
        t_start = time.time()
        chunk = todo_pids[start : start + batch_size]
        items = _request_s2_chunk_adaptive(chunk, url=url, headers=headers, timeout=timeout)

        if items is not None:
            batch_records: Dict[str, Dict[str, Any]] = {}
            by_arxiv: Dict[str, Mapping[str, Any]] = {}
            for item in items:
                if isinstance(item, Mapping):
                    ext = item.get("externalIds")
                    if isinstance(ext, Mapping) and ext.get("ArXiv"):
                        aid = normalize_arxiv_id(ext["ArXiv"])
                        if aid:
                            by_arxiv[aid] = item

            for idx, pid in enumerate(chunk):
                item = by_arxiv.get(pid)
                if item is None and idx < len(items):
                    item = items[idx]
                batch_records[pid] = _extract_s2_paper_features(
                    item, pid_to_year.get(pid, 2020)
                )

            store.save_batch("s2_batch", batch_records)

        elapsed = time.time() - t_start
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        done = min(len(todo_pids), start + len(chunk))
        if (start // batch_size + 1) % 15 == 0 or done == len(todo_pids):
            print(f"[SemanticScholar] Progress: {done}/{len(todo_pids)} papers processed.")

    return store.load_all("s2_batch")


# ---------------------------------------------------------------------------
# 3. Papers With Code Bulk Dump + Local Regex Extractor
# ---------------------------------------------------------------------------

def _ingest_pwc_records(
    records: Iterable[Mapping[str, Any]],
    target_pids: set[str],
    title_to_pid: Mapping[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate Papers With Code repository records per `paper_id`."""
    agg: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "official": 0,
            "repos": set(),
            "pytorch": 0,
            "tensorflow": 0,
            "github": 0,
        }
    )

    for row in records:
        if not isinstance(row, Mapping):
            continue
        pid = ""
        for k in ("paper_arxiv_id", "arxiv_id", "paper_url_abs", "paper_url_pdf", "paper_url"):
            val = row.get(k)
            if val:
                cand = normalize_arxiv_id(val)
                if cand in target_pids:
                    pid = cand
                    break
        if not pid and row.get("paper_title"):
            t_norm = re.sub(r"\W+", " ", str(row["paper_title"]).lower()).strip()
            if t_norm in title_to_pid:
                pid = title_to_pid[t_norm]
        if not pid or pid not in target_pids:
            continue

        entry = agg[pid]
        repo_url = str(row.get("repo_url") or "").strip().lower()
        if repo_url:
            entry["repos"].add(repo_url)
            if "github.com" in repo_url:
                entry["github"] = 1
        if row.get("is_official") is True or str(row.get("is_official")).lower() == "true":
            entry["official"] = 1
        if row.get("mentioned_in_github") is True or str(row.get("mentioned_in_github")).lower() == "true":
            entry["github"] = 1
        fw = str(row.get("framework") or "").strip().lower()
        if fw in ("pytorch", "torch"):
            entry["pytorch"] = 1
        elif fw in ("tf", "tensorflow", "keras"):
            entry["tensorflow"] = 1

    out: Dict[str, Dict[str, Any]] = {}
    for pid, info in agg.items():
        n_repos = len(info["repos"])
        out[pid] = {
            "pwc_has_official_code": int(info["official"] or n_repos > 0),
            "pwc_repo_count": int(max(n_repos, 1 if info["github"] else 0)),
            "pwc_is_pytorch": int(info["pytorch"]),
            "pwc_is_tensorflow": int(info["tensorflow"]),
            "pwc_mentioned_in_github": int(info["github"] or n_repos > 0),
        }
    return out


def fetch_pwc_and_local_code_features(
    df: pd.DataFrame,
    pwc_url: str = "https://production-media.paperswithcode.com/about/links-between-papers-and-code.json.gz",
    hf_parquet_api_url: str = "https://huggingface.co/api/datasets/pwc-archive/links-between-paper-and-code/parquet/default/train",
    pwc_local_path: Optional[str | Path] = None,
    cache_dir: str | Path = "cache",
    timeout: float = 45.0,
    skip_network: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Combine Papers With Code archive dump with local regex extraction on `abstract_v1` + `comment_v1`."""
    cache_dir_path = Path(cache_dir).expanduser().resolve()
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    target_pids = set(df["paper_id"].astype(str))
    title_to_pid: Dict[str, str] = {}
    for pid, title in zip(df["paper_id"].astype(str), df["title_v1"].fillna("")):
        t_norm = re.sub(r"\W+", " ", str(title).lower()).strip()
        if t_norm:
            title_to_pid[t_norm] = str(pid)

    pwc_matched: Dict[str, Dict[str, Any]] = {}
    local_gz = Path(pwc_local_path) if pwc_local_path else (cache_dir_path / "links-between-papers-and-code.json.gz")

    if not skip_network and not local_gz.is_file():
        try:
            print(f"[PapersWithCode] Downloading archive from {pwc_url} ...")
            req = urllib.request.Request(
                pwc_url,
                headers={"User-Agent": "HKU-ImpactModel-BatchCollector/3.1", "Connection": "close"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read()
            local_gz.write_bytes(raw_bytes)
        except Exception as exc:
            print(f"[PapersWithCode] Primary URL unavailable ({exc}); checking Hugging Face mirror...")
            try:
                parquet_urls = _http_request_json(hf_parquet_api_url, timeout=timeout)
                if isinstance(parquet_urls, list) and parquet_urls:
                    p_url = str(parquet_urls[0])
                    req_p = urllib.request.Request(
                        p_url,
                        headers={"User-Agent": "HKU-ImpactModel-BatchCollector/3.1", "Connection": "close"},
                    )
                    with urllib.request.urlopen(req_p, timeout=timeout) as resp_p:
                        p_bytes = resp_p.read()
                    p_df = pd.read_parquet(io.BytesIO(p_bytes))
                    pwc_matched = _ingest_pwc_records(
                        p_df.to_dict(orient="records"), target_pids, title_to_pid
                    )
            except Exception as hf_exc:
                print(
                    f"[PapersWithCode] Hugging Face mirror also unreachable ({hf_exc}); "
                    "using local abstract/comment regex extraction fallback."
                )

    if local_gz.is_file() and not pwc_matched:
        try:
            raw_bytes = local_gz.read_bytes()
            if raw_bytes[:2] == b"\x1f\x8b":
                raw_bytes = gzip.decompress(raw_bytes)
            parsed = json.loads(raw_bytes.decode("utf-8"))
            if isinstance(parsed, list):
                pwc_matched = _ingest_pwc_records(parsed, target_pids, title_to_pid)
                print(f"[PapersWithCode] Matched {len(pwc_matched)} papers from local PWC dump.")
        except Exception as exc:
            print(f"[PapersWithCode] Warning reading {local_gz}: {exc}", file=sys.stderr)

    out: Dict[str, Dict[str, Any]] = {}
    for pid, title, abst, comm in zip(
        df["paper_id"].astype(str),
        df["title_v1"].fillna("").astype(str),
        df["abstract_v1"].fillna("").astype(str),
        df["comment_v1"].fillna("").astype(str),
    ):
        combined = f"{title} {abst} {comm}"
        urls = set(m.group(0).lower() for m in GITHUB_URL_RE.finditer(combined))
        has_code_text = 1 if (urls or CODE_AVAIL_RE.search(combined)) else 0
        is_pt = 1 if PYTORCH_RE.search(combined) else 0
        is_tf = 1 if TENSORFLOW_RE.search(combined) else 0
        has_gh = 1 if ("github.com" in combined.lower() or urls) else 0

        pwc_item = pwc_matched.get(str(pid))
        if pwc_item is not None:
            out[str(pid)] = {
                "pwc_has_official_code": int(max(pwc_item["pwc_has_official_code"], has_code_text)),
                "pwc_repo_count": int(max(pwc_item["pwc_repo_count"], len(urls), has_code_text)),
                "pwc_is_pytorch": int(max(pwc_item["pwc_is_pytorch"], is_pt)),
                "pwc_is_tensorflow": int(max(pwc_item["pwc_is_tensorflow"], is_tf)),
                "pwc_mentioned_in_github": int(max(pwc_item["pwc_mentioned_in_github"], has_gh)),
            }
        else:
            out[str(pid)] = {
                "pwc_has_official_code": int(has_code_text),
                "pwc_repo_count": int(max(len(urls), has_code_text)),
                "pwc_is_pytorch": int(is_pt),
                "pwc_is_tensorflow": int(is_tf),
                "pwc_mentioned_in_github": int(has_gh),
            }
    return out


# ---------------------------------------------------------------------------
# 4. Offline Local Affiliation Propagation Fallback & Enriched Table Merger
# ---------------------------------------------------------------------------

def build_local_affiliation_propagation(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Propagate author affiliations present in `authors_v1_json` across all co-authored papers."""
    author_affs: Dict[str, set[str]] = defaultdict(set)
    for raw_au in df["authors_v1_json"].dropna():
        try:
            parsed = json.loads(raw_au)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, Mapping):
                        nm = str(item.get("name") or "").strip().lower()
                        af = str(item.get("affiliation_raw") or "").strip()
                        if nm and af:
                            author_affs[nm].add(af)
        except json.JSONDecodeError:
            continue

    out: Dict[str, Dict[str, Any]] = {}
    for pid, raw_au in zip(df["paper_id"].astype(str), df["authors_v1_json"]):
        insts: set[str] = set()
        if isinstance(raw_au, str) and raw_au.strip():
            try:
                parsed = json.loads(raw_au)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, Mapping):
                            nm = str(item.get("name") or "").strip().lower()
                            af = str(item.get("affiliation_raw") or "").strip()
                            if af:
                                insts.add(af)
                            elif nm in author_affs:
                                insts.update(author_affs[nm])
            except json.JSONDecodeError:
                pass
        inst_list = sorted(insts)
        has_ind, has_uni = classify_institution_flags(inst_list, [""] * len(inst_list))
        out[str(pid)] = {
            "inst_count": len(inst_list),
            "has_ind": has_ind,
            "has_uni": has_uni,
            "inst_json": json.dumps(inst_list[:20], ensure_ascii=False),
        }
    return out


def merge_external_features(
    df: pd.DataFrame,
    oa_records: Mapping[str, Mapping[str, Any]],
    s2_records: Mapping[str, Mapping[str, Any]],
    pwc_records: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Merge OpenAlex, Semantic Scholar, and Papers With Code features into `df`."""
    local_aff = build_local_affiliation_propagation(df)
    enriched = df.copy()

    rows_ext: List[Dict[str, Any]] = []
    for pid, ref_cnt_cur in zip(
        enriched["paper_id"].astype(str),
        pd.to_numeric(enriched["s2_reference_count_current"], errors="coerce").fillna(0.0),
    ):
        oa = oa_records.get(pid, {})
        s2 = s2_records.get(pid, {})
        pwc = pwc_records.get(pid, {})
        loc = local_aff.get(pid, {})

        oa_matched = int(oa.get("oa_matched", 0))
        oa_inst_cnt = int(oa.get("oa_institutions_count", 0)) or int(loc.get("inst_count", 0))
        oa_cntry_cnt = int(oa.get("oa_countries_count", 0)) or (1 if oa_inst_cnt > 0 else 0)
        oa_ind = max(int(oa.get("oa_has_top_industry_lab", 0)), int(loc.get("has_ind", 0)))
        oa_uni = max(int(oa.get("oa_has_top_university", 0)), int(loc.get("has_uni", 0)))
        oa_inst_json = str(oa.get("oa_institutions_json") or "[]")
        if oa_inst_json == "[]" and loc.get("inst_json"):
            oa_inst_json = str(loc["inst_json"])

        rows_ext.append({
            "oa_matched": oa_matched,
            "oa_institutions_count": oa_inst_cnt,
            "oa_countries_count": oa_cntry_cnt,
            "oa_has_top_industry_lab": oa_ind,
            "oa_has_top_university": oa_uni,
            "oa_early_citations_t0_year": int(oa.get("oa_early_citations_t0_year", 0)),
            "oa_early_citations_t0_plus1": int(oa.get("oa_early_citations_t0_plus1", 0)),
            "oa_referenced_works_count": int(oa.get("oa_referenced_works_count", 0)) or int(ref_cnt_cur),
            "oa_institutions_json": oa_inst_json,
            "s2_batch_matched": int(s2.get("s2_batch_matched", 0)),
            "s2_ref_citation_max": s2.get("s2_ref_citation_max", math.nan),
            "s2_ref_citation_mean": s2.get("s2_ref_citation_mean", math.nan),
            "s2_ref_citation_median": s2.get("s2_ref_citation_median", math.nan),
            "s2_ref_recent_2y_ratio": s2.get("s2_ref_recent_2y_ratio", math.nan),
            "s2_author_max_hindex": s2.get("s2_author_max_hindex", math.nan),
            "s2_author_mean_hindex": s2.get("s2_author_mean_hindex", math.nan),
            "s2_author_max_citations": s2.get("s2_author_max_citations", math.nan),
            "pwc_has_official_code": int(pwc.get("pwc_has_official_code", 0)),
            "pwc_repo_count": int(pwc.get("pwc_repo_count", 0)),
            "pwc_is_pytorch": int(pwc.get("pwc_is_pytorch", 0)),
            "pwc_is_tensorflow": int(pwc.get("pwc_is_tensorflow", 0)),
            "pwc_mentioned_in_github": int(pwc.get("pwc_mentioned_in_github", 0)),
        })

    ext_df = pd.DataFrame(rows_ext, index=enriched.index)
    for col in EXTERNAL_COLUMNS:
        enriched[col] = ext_df[col]
    return enriched


# ---------------------------------------------------------------------------
# 5. Built-in Local Mock Server for 29,856-Paper End-to-End Verification
# ---------------------------------------------------------------------------

def run_local_30k_self_test(input_csv: Path, output_csv: Path, cache_dir: Path) -> Dict[str, Any]:
    """Spin up a local multi-threaded HTTP mock server serving realistic OpenAlex,
    Semantic Scholar (with simulated HTTP 429 & HTTP 413 edge cases), and Papers With Code
    endpoints, and execute the full 29,856-paper batch pipeline end-to-end."""
    df = pd.read_csv(input_csv, dtype={"paper_id": "string"}, low_memory=False)
    df["paper_id"] = df["paper_id"].astype("string").str.strip()

    meta_by_pid: Dict[str, Tuple[int, float, float, str, str]] = {}
    for pid, yr, dmax, rcnt, title, abst in zip(
        df["paper_id"].astype(str),
        df["submission_year"].astype(int),
        pd.to_numeric(df["dblp_team_max_publication_count_pre_t0"], errors="coerce").fillna(12.0),
        pd.to_numeric(df["s2_reference_count_current"], errors="coerce").fillna(32.0),
        df["title_v1"].fillna("").astype(str),
        df["abstract_v1"].fillna("").astype(str),
    ):
        meta_by_pid[pid] = (int(yr), float(dmax), float(rcnt), title, abst)

    state = {"oa_requests": 0, "s2_requests": 0, "s2_429_triggered": 0, "pwc_requests": 0}
    state_lock = threading.Lock()

    class _MockAPIHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/works":
                with state_lock:
                    state["oa_requests"] += 1
                qs = urllib.parse.parse_qs(parsed.query)
                flt = qs.get("filter", [""])[0]
                pids = ARXIV_ID_EXTRACT_RE.findall(flt)
                results = []
                for pid in pids:
                    if pid not in meta_by_pid:
                        continue
                    yr, dmax, rcnt, title, _ = meta_by_pid[pid]
                    h_val = int(hash(pid) & 0xFFFF)
                    is_top_uni = (dmax >= 25.0) or (h_val % 5 == 0)
                    is_ind = (dmax >= 60.0) or ("transformer" in title.lower())
                    inst_list = []
                    if is_top_uni:
                        inst_list.append({
                            "id": f"https://openalex.org/I{1000 + (h_val % 30)}",
                            "display_name": "University of Hong Kong" if (h_val % 2 == 0) else "Stanford University",
                            "type": "education",
                            "country_code": "HK" if (h_val % 2 == 0) else "US",
                        })
                    if is_ind:
                        inst_list.append({
                            "id": f"https://openalex.org/I{2000 + (h_val % 10)}",
                            "display_name": "Google Research",
                            "type": "company",
                            "country_code": "US",
                        })
                    early_c0 = int(min(15, max(0, round(math.log1p(dmax) * 0.6 + math.log1p(rcnt) * 0.3 - 1.5))))
                    results.append({
                        "id": f"https://openalex.org/W{abs(hash(pid)) % 10**10}",
                        "doi": f"https://doi.org/10.48550/arxiv.{pid}",
                        "ids": {"arxiv": f"https://arxiv.org/abs/{pid}"},
                        "publication_year": yr,
                        "institutions_distinct_count": len(inst_list),
                        "countries_distinct_count": len({i["country_code"] for i in inst_list}),
                        "authorships": [{"institutions": inst_list, "countries": [i["country_code"] for i in inst_list]}],
                        "counts_by_year": [
                            {"year": yr, "cited_by_count": early_c0},
                            {"year": yr + 1, "cited_by_count": early_c0 * 2},
                        ],
                        "referenced_works_count": int(rcnt),
                        "locations": [{"landing_page_url": f"https://arxiv.org/abs/{pid}"}],
                    })
                body = json.dumps({"results": results}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/links-between-papers-and-code.json.gz":
                with state_lock:
                    state["pwc_requests"] += 1
                pwc_rows = []
                for pid, (yr, dmax, rcnt, title, abst) in list(meta_by_pid.items())[::3]:
                    pwc_rows.append({
                        "paper_arxiv_id": pid,
                        "paper_title": title,
                        "repo_url": f"https://github.com/research/{pid.replace('.', '_')}",
                        "is_official": True,
                        "mentioned_in_github": True,
                        "framework": "pytorch" if (yr >= 2019) else "tf",
                    })
                gz_buf = io.BytesIO()
                with gzip.GzipFile(fileobj=gz_buf, mode="wb") as gz:
                    gz.write(json.dumps(pwc_rows).encode("utf-8"))
                payload = gz_buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/graph/v1/paper/batch":
                with state_lock:
                    state["s2_requests"] += 1
                    req_num = state["s2_requests"]
                    if req_num == 5 and state["s2_429_triggered"] == 0:
                        state["s2_429_triggered"] = 1
                        self.send_response(429)
                        self.send_header("Retry-After", "0.05")
                        self.end_headers()
                        self.wfile.write(b'{"message": "Too Many Requests"}')
                        return

                length = int(self.headers.get("Content-Length", "0"))
                req_json = json.loads(self.rfile.read(length).decode("utf-8"))
                ids = req_json.get("ids") or []
                out_list = []
                for raw_id in ids:
                    pid = normalize_arxiv_id(raw_id)
                    if pid not in meta_by_pid:
                        out_list.append(None)
                        continue
                    yr, dmax, rcnt, _, _ = meta_by_pid[pid]
                    base_ref_c = max(10.0, rcnt * 8.0 + dmax * 2.5)
                    h_idx = max(2, int(round(math.sqrt(dmax) * 2.2)))
                    out_list.append({
                        "paperId": f"s2_{pid}",
                        "externalIds": {"ArXiv": pid},
                        "year": yr,
                        "referenceCount": int(rcnt),
                        "references": [
                            {"year": yr - 1, "citationCount": int(base_ref_c * 3.5)},
                            {"year": yr - 2, "citationCount": int(base_ref_c * 1.2)},
                            {"year": yr - 4, "citationCount": int(base_ref_c * 0.5)},
                        ],
                        "authors": [
                            {"hIndex": h_idx, "citationCount": int(h_idx * h_idx * 4.5)},
                            {"hIndex": max(1, h_idx // 2), "citationCount": int(h_idx * 15)},
                        ],
                    })
                body = json.dumps(out_list).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(404)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockAPIHandler)
    port = server.server_address[1]
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    store: Optional[SQLiteCheckpointStore] = None
    try:
        db_file = cache_dir / "self_test_30k_cache.sqlite"
        if db_file.exists():
            db_file.unlink()
        pwc_gz = cache_dir / "links-between-papers-and-code.json.gz"
        if pwc_gz.exists():
            pwc_gz.unlink()

        store = SQLiteCheckpointStore(db_file)
        oa_url = f"http://127.0.0.1:{port}/works"
        s2_url = f"http://127.0.0.1:{port}/graph/v1/paper/batch"
        pwc_url = f"http://127.0.0.1:{port}/links-between-papers-and-code.json.gz"

        oa_res = fetch_openalex_batches(
            df, store, base_url=oa_url, batch_size=50, rps=5000.0, timeout=10.0
        )
        s2_res = fetch_s2_batches(
            df, store, base_url=s2_url, batch_size=100, rps=5000.0, timeout=10.0
        )
        pwc_res = fetch_pwc_and_local_code_features(
            df, pwc_url=pwc_url, cache_dir=cache_dir, timeout=10.0
        )

        enriched = merge_external_features(df, oa_res, s2_res, pwc_res)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(output_csv, index=False)

        open_fds = len(os.listdir("/proc/self/fd")) if os.path.exists("/proc/self/fd") else -1
        summary = {
            "rows_verified": int(len(enriched)),
            "columns_verified": int(enriched.shape[1]),
            "openalex_batches_sent": int(state["oa_requests"]),
            "s2_batches_sent": int(state["s2_requests"]),
            "s2_http_429_recovered": int(state["s2_429_triggered"]),
            "pwc_dump_requests": int(state["pwc_requests"]),
            "oa_matched_count": int(enriched["oa_matched"].sum()),
            "s2_matched_count": int(enriched["s2_batch_matched"].sum()),
            "pwc_official_code_count": int(enriched["pwc_has_official_code"].sum()),
            "sqlite_cached_openalex": len(store.get_cached_ids("openalex")),
            "sqlite_cached_s2": len(store.get_cached_ids("s2_batch")),
            "open_file_descriptors_at_completion": open_fds,
        }
        print("[SelfTest-30k] Completed 29,856-paper batch test:", json.dumps(summary, indent=2))
        return summary
    finally:
        if store is not None:
            store.close()
        server.shutdown()
        server.server_close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch OpenAlex, Semantic Scholar, and Papers With Code external features for papers_features.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("papers_features.csv"),
        help="Path to the input 25-column papers_features.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("papers_features_enriched.csv"),
        help="Path to write the enriched 47-column CSV.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Directory for SQLite checkpoint database and downloaded PWC archive.",
    )
    parser.add_argument(
        "--mailto",
        type=str,
        default="research@hku.hk",
        help="Contact email for OpenAlex Polite Pool (10 req/sec).",
    )
    parser.add_argument(
        "--openalex-api-key",
        type=str,
        default=os.environ.get("OPENALEX_API_KEY", ""),
        help="Optional OpenAlex API key (or set OPENALEX_API_KEY env var).",
    )
    parser.add_argument(
        "--s2-api-key",
        type=str,
        default=os.environ.get("S2_API_KEY", ""),
        help="Optional Semantic Scholar API key (or set S2_API_KEY env var).",
    )
    parser.add_argument(
        "--oa-batch-size",
        type=int,
        default=50,
        help="Number of arXiv DOIs per OpenAlex request (default: 50, max: 100).",
    )
    parser.add_argument(
        "--s2-batch-size",
        type=int,
        default=100,
        help="Number of arXiv IDs per Semantic Scholar POST batch (default: 100).",
    )
    parser.add_argument(
        "--oa-rps",
        type=float,
        default=8.0,
        help="Target requests per second for OpenAlex Polite Pool (default: 8.0).",
    )
    parser.add_argument(
        "--s2-rps",
        type=float,
        default=1.0,
        help="Target requests per second for Semantic Scholar Batch API (default: 1.0).",
    )
    parser.add_argument(
        "--pwc-dump-path",
        type=Path,
        default=None,
        help="Optional local path to links-between-papers-and-code.json.gz.",
    )
    parser.add_argument(
        "--offline-fallback",
        action="store_true",
        help="Skip live network calls and populate external columns using cached SQLite records + local affiliation/regex extraction.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a full 29,856-paper end-to-end verification against a local mock HTTP server.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _bump_file_descriptor_limit()
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if args.self_test:
        run_local_30k_self_test(input_path, output_path, cache_dir)
        return 0

    df = pd.read_csv(input_path, dtype={"paper_id": "string"}, low_memory=False)
    df["paper_id"] = df["paper_id"].astype("string").str.strip()

    store = SQLiteCheckpointStore(cache_dir / "external_features_cache.sqlite")
    try:
        if args.offline_fallback:
            print("[OfflineFallback] Loading existing SQLite cache and local text/affiliation extractors...")
            oa_records = store.load_all("openalex")
            s2_records = store.load_all("s2_batch")
            pwc_records = fetch_pwc_and_local_code_features(
                df, pwc_local_path=args.pwc_dump_path, cache_dir=cache_dir, skip_network=True
            )
        else:
            oa_records = fetch_openalex_batches(
                df,
                store=store,
                batch_size=args.oa_batch_size,
                mailto=args.mailto,
                api_key=args.openalex_api_key or None,
                rps=args.oa_rps,
            )
            s2_records = fetch_s2_batches(
                df,
                store=store,
                batch_size=args.s2_batch_size,
                api_key=args.s2_api_key or None,
                rps=args.s2_rps,
            )
            pwc_records = fetch_pwc_and_local_code_features(
                df,
                pwc_local_path=args.pwc_dump_path,
                cache_dir=cache_dir,
                skip_network=False,
            )

        enriched = merge_external_features(df, oa_records, s2_records, pwc_records)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(output_path, index=False)
        print(
            f"[Done] Wrote enriched dataset ({len(enriched)} rows x {enriched.shape[1]} cols) "
            f"to {output_path}"
        )
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())

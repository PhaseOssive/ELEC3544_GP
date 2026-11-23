"""显式调用的 Notebook 迁移模块。输入输出与主键见 README 和 DATA_DICTIONARY。"""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections import deque
if TYPE_CHECKING:
    from fetch.control.config import Config
    from fetch.control.tools import HTTPClient
    import pandas as pd
from fetch.control.tools import DeferredRequest, atomic_csv, log_preview, log_print as print


def _run_deferred_queue(records, fetch_one, http, max_deferred_retries=10, on_progress=None):
    """主队列先跑；429 论文延后，连续限流时优先探测重试队列。"""
    main_queue = deque(records)
    retry_queue = deque()
    results = {}
    edges_by_id = {}
    retry_counts = {}
    attempts = 0
    deferred_events = 0
    summary_cache_hits = 0
    exhausted_ids = set()

    while main_queue or retry_queue:
        # 全局限流期间只从重试队列取一个任务作为探测，不继续冲击主队列。
        if getattr(http, "s2_pause_requested", False) and retry_queue:
            row, _ = retry_queue.popleft()
            queue_name = "429 probe"
        elif main_queue:
            row = main_queue.popleft()
            queue_name = "main"
        else:
            row, _ = retry_queue.popleft()
            queue_name = "retry"

        aid = str(row["arxiv_id"])
        output, edges, cache_hit, deferred_429 = fetch_one(row)
        attempts += 1
        summary_cache_hits += int(cache_hit)
        results[aid] = output
        # 重试时以最新一轮重建出的完整边集合替换旧集合，避免重复追加。
        edges_by_id[aid] = edges

        if deferred_429:
            deferred_events += 1
            retry_count = retry_counts.get(aid, 0) + 1
            retry_counts[aid] = retry_count
            if retry_count <= max_deferred_retries:
                retry_queue.append((row, retry_count))
            else:
                exhausted_ids.add(aid)
                previous_error = output.get("s2_c3_error") or "HTTP_429"
                output["s2_c3_error"] = (
                    f"{previous_error}; deferred retry budget exhausted "
                    f"after {max_deferred_retries} retries; rerun to resume"
                )
                results[aid] = output
        else:
            retry_counts.pop(aid, None)

        if on_progress is not None:
            on_progress(
                results=results,
                edges_by_id=edges_by_id,
                attempts=attempts,
                deferred_events=deferred_events,
                queue_name=queue_name
            )

    return {
        "results": results,
        "edges_by_id": edges_by_id,
        "attempts": attempts,
        "deferred_events": deferred_events,
        "summary_cache_hits": summary_cache_hits,
        "exhausted_ids": exhausted_ids
    }


def collect(config: Config, http: HTTPClient, papers_df=None, seed_df=None) -> dict[str, pd.DataFrame]:
    ROOT, RAW, PROCESSED = config.paths

    # ============================================================
    # SEMANTIC SCHOLAR: METADATA + THREE-YEAR C3 + IC3
    #
    # C3  = All unique citing papers within 3 years
    # IC3 = Unique influential citing papers within 3 years
    #
    # Replaces the previous merged Semantic Scholar Cell.
    #
    # Dependencies:
    #   - PROCESSED
    #   - request_with_retry() from Cell 5
    #   - papers_df from arXiv v1 collection
    #
    # Outputs:
    #   1. s2_metadata.csv
    #   2. s2_metadata_audit.csv
    #   3. semantic_scholar_citation_3y.csv
    #   4. semantic_scholar_influential_citation_3y.csv
    #   5. semantic_scholar_citation_edges_3y.csv
    #   6. semantic_scholar_citation_3y_progress.csv
    #   7. semantic_scholar_papers.csv
    #   8. request_log.json
    # ============================================================

    import hashlib
    import json
    import re
    import time
    import urllib.parse

    from pathlib import Path
    from datetime import date, timedelta

    import pandas as pd
    from fetch.control.tools import quiet_tqdm as tqdm



    # ------------------------------------------------------------
    # 0. ENVIRONMENT CHECK + AUTO INITIALIZATION
    # ------------------------------------------------------------

    import os
    import time
    import requests
    from pathlib import Path

    S2 = "https://api.semanticscholar.org/graph/v1"

    S2_OUTPUT_DIR = Path(PROCESSED) / "semantic_scholar_public"

    S2_CACHE_DIR = S2_OUTPUT_DIR / "raw"

    # New summary cache: do not reuse old IC3-only summaries.
    S2_SUMMARY_CACHE_DIR = (
        S2_OUTPUT_DIR / "citation_3y_summary_cache"
    )

    S2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    S2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    S2_SUMMARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    S2_REFRESH_CACHE = config.refresh("semantic_scholar")
    S2_ALLOW_LEGACY_RAW_CACHE = True
    S2_BATCH_SIZE = config.s2_batch_size
    S2_PAGE_SIZE = config.s2_page_size
    S2_MAX_CITATIONS = config.s2_max_citations
    S2_MAX_PAGES = config.s2_max_pages
    # Request interval and API-key header are handled centrally by HTTPClient.

    S2_METADATA_FIELDS = (
        "title,externalIds,authors,citationCount,"
        "influentialCitationCount,referenceCount,"
        "year,publicationDate"
    )

    # New request: include year as a fallback when
    # publicationDate is unavailable.
    S2_CITATION_FIELDS = (
        "isInfluential,"
        "citingPaper.paperId,"
        "citingPaper.publicationDate,"
        "citingPaper.year"
    )

    # Exact old field list, for legacy raw-cache reuse.
    S2_LEGACY_CITATION_FIELDS = (
        "isInfluential,"
        "citingPaper.paperId,"
        "citingPaper.publicationDate"
    )

    S2_TERMINAL_STATUSES = {
        "COMPLETE",
        "INCOMPLETE_DATE_OR_ID_COVERAGE",
        "INCOMPLETE_INFLUENTIAL_COVERAGE",
        "API_9999_LIMIT_REACHED",
        "PAGE_LIMIT_REACHED",
        "NO_S2_MATCH",
        "MISSING_T0",
        "ZERO_CONFIRMED_BY_CUMULATIVE_COUNT"
    }

    S2_REQUEST_LOG = []


    # ------------------------------------------------------------
    # 2. CACHED API REQUEST FUNCTION
    # ------------------------------------------------------------

    def _s2_cache_path(url, payload=None):
        key = hashlib.sha256(
            json.dumps(
                [url, payload],
                sort_keys=True
            ).encode()
        ).hexdigest()

        return S2_CACHE_DIR / f"{key}.json"


    def s2_call(url, payload=None, legacy_urls=None):
        return http.s2(url, payload, legacy_urls, S2_CACHE_DIR)



    # ------------------------------------------------------------
    # 3. BASIC HELPERS
    # ------------------------------------------------------------

    def normalise_arxiv_id(value):

        if pd.isna(value):
            return None

        return re.sub(
            r"v\d+$",
            "",
            str(value).strip()
        )


    def optional_int(value):

        x = pd.to_numeric(
            pd.Series([value]),
            errors="coerce"
        ).iloc[0]

        return None if pd.isna(x) else int(x)


    def safe_string(value):

        if value is None or pd.isna(value):
            return None

        return str(value)


    def parse_year(value):

        if value is None or pd.isna(value):
            return None

        try:
            year = int(float(str(value).strip()))

            if 1800 <= year <= 2100:
                return year

        except Exception:
            pass

        return None


    # ------------------------------------------------------------
    # 4. PREPARE ARXIV INPUT
    # ------------------------------------------------------------

    s2_input = papers_df[
        ["arxiv_id", "submission_date_v1"]
    ].copy()

    s2_input["arxiv_id"] = (
        s2_input["arxiv_id"].map(normalise_arxiv_id)
    )

    s2_input = (
        s2_input
        .dropna(subset=["arxiv_id"])
        .drop_duplicates("arxiv_id")
        .reset_index(drop=True)
    )


    # ------------------------------------------------------------
    # 5. SEMANTIC SCHOLAR METADATA MATCHING
    # ------------------------------------------------------------

    def fetch_s2_metadata(input_df):

        rows_by_id = {}
        audit_by_id = {}
        batch_tasks = []

        for start in range(0, len(input_df), S2_BATCH_SIZE):
            ids = input_df["arxiv_id"].iloc[start:start + S2_BATCH_SIZE].tolist()
            batch_tasks.append((start, ids))

        main_queue = deque(batch_tasks)
        retry_queue = deque()
        retry_counts = {}
        batch_attempts = 0
        deferred_batches = 0
        max_deferred_retries = 10

        def empty_metadata_row(aid):
            return {
                "arxiv_id": aid,
                "s2_paper_id": None,
                "title": None,
                "doi": None,
                "citation_count": None,
                "influential_citation_count": None,
                "reference_count": None,
                "publication_date": None,
                "year": None,
                "authors_json": None
            }

        def store_batch(ids, data):
            if not isinstance(data, list) or len(data) != len(ids):
                raise ValueError("S2 batch response length mismatch.")

            for aid, paper in zip(ids, data):
                row = empty_metadata_row(aid)
                if paper is None:
                    status = "S2_NO_RECORD_RETURNED"
                else:
                    ext = paper.get("externalIds") or {}
                    found_id = normalise_arxiv_id(ext.get("ArXiv", ""))
                    if found_id and found_id != aid:
                        status = "ARXIV_ID_CONFLICT"
                    else:
                        row.update({
                            "s2_paper_id": paper.get("paperId"),
                            "title": paper.get("title"),
                            "doi": ext.get("DOI"),
                            "citation_count": paper.get("citationCount"),
                            "influential_citation_count": paper.get("influentialCitationCount"),
                            "reference_count": paper.get("referenceCount"),
                            "publication_date": paper.get("publicationDate"),
                            "year": paper.get("year"),
                            "authors_json": json.dumps(
                                paper.get("authors") or [], ensure_ascii=False
                            )
                        })
                        status = "MATCHED" if row["s2_paper_id"] else "NO_S2_PAPER_ID"

                rows_by_id[aid] = row
                audit_by_id[aid] = {
                    "arxiv_id": aid,
                    "status": status,
                    "error": None
                }

        while main_queue or retry_queue:
            if getattr(http, "s2_pause_requested", False) and retry_queue:
                start, ids = retry_queue.popleft()
                queue_name = "429 probe"
            elif main_queue:
                start, ids = main_queue.popleft()
                queue_name = "main"
            else:
                start, ids = retry_queue.popleft()
                queue_name = "retry"

            url = S2 + "/paper/batch?" + urllib.parse.urlencode({
                "fields": S2_METADATA_FIELDS
            })

            batch_attempts += 1
            try:
                data = s2_call(
                    url,
                    payload={"ids": ["ARXIV:" + aid for aid in ids]}
                )
                store_batch(ids, data)
                retry_counts.pop(start, None)

            except DeferredRequest as error:
                deferred_batches += 1
                retry_count = retry_counts.get(start, 0) + 1
                retry_counts[start] = retry_count
                if retry_count <= max_deferred_retries:
                    retry_queue.append((start, ids))
                else:
                    for aid in ids:
                        rows_by_id[aid] = empty_metadata_row(aid)
                        audit_by_id[aid] = {
                            "arxiv_id": aid,
                            "status": "REQUEST_ERROR",
                            "error": (
                                f"{error}; 429 deferred retry budget exhausted; "
                                "rerun to resume"
                            )[:1000]
                        }

            except Exception as error:
                for aid in ids:
                    rows_by_id[aid] = empty_metadata_row(aid)
                    audit_by_id[aid] = {
                        "arxiv_id": aid,
                        "status": "REQUEST_ERROR",
                        "error": str(error)[:1000]
                    }

            processed = len(rows_by_id)
            print(
                f"S2 metadata: {processed}/{len(input_df)} "
                f"(batch attempts={batch_attempts}, deferred 429={deferred_batches})"
            )

        metadata_df = pd.DataFrame(
            [rows_by_id[aid] for aid in input_df["arxiv_id"].tolist()]
        )
        audit_df = pd.DataFrame(
            [audit_by_id[aid] for aid in input_df["arxiv_id"].tolist()]
        )

        atomic_csv(metadata_df,
            S2_OUTPUT_DIR / "s2_metadata.csv",
            index=False
        )

        atomic_csv(audit_df,
            S2_OUTPUT_DIR / "s2_metadata_audit.csv",
            index=False
        )

        print(
            "S2 metadata queue summary:",
            {
                "papers": len(input_df),
                "batch_attempts": batch_attempts,
                "deferred_429_batches": deferred_batches,
                "retry_budget_exhausted": sum(
                    1 for item in audit_by_id.values()
                    if item["status"] == "REQUEST_ERROR"
                    and "retry budget exhausted" in str(item.get("error", ""))
                )
            }
        )

        return metadata_df, audit_df


    s2_metadata_df, s2_audit_df = fetch_s2_metadata(
        s2_input
    )


    # ------------------------------------------------------------
    # 6. DATE CLASSIFICATION
    # ------------------------------------------------------------

    def classify_citation_date(
        citing_paper,
        start_day,
        end_day
    ):
        """
        Classify a citing paper's date relative to:

            [t0, t0 + 3 years)

        Supports:
          - Exact day: YYYY-MM-DD
          - Month only: YYYY-MM
          - Year only: YYYY or citingPaper.year

        A year/month is counted only if its entire
        possible period is inside the three-year window.

        Partial overlap is marked AMBIGUOUS_DATE.
        """

        raw_date = citing_paper.get(
            "publicationDate"
        )

        year = parse_year(
            citing_paper.get("year")
        )

        precision = None
        period_start = None
        period_end = None
        display_date = None

        raw = (
            str(raw_date).strip()
            if raw_date is not None
            else ""
        )

        # Exact day.
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):

            try:

                parsed_day = date.fromisoformat(raw)

                period_start = parsed_day
                period_end = parsed_day + timedelta(days=1)

                year = parsed_day.year
                precision = "DAY"
                display_date = raw

            except Exception:
                pass

        # Month-only date.
        elif re.fullmatch(r"\d{4}-\d{2}", raw):

            try:

                y, m = map(int, raw.split("-"))

                period_start = date(y, m, 1)

                if m == 12:
                    period_end = date(y + 1, 1, 1)
                else:
                    period_end = date(y, m + 1, 1)

                year = y
                precision = "MONTH"
                display_date = raw

            except Exception:
                pass

        # Year-only publication date.
        elif re.fullmatch(r"\d{4}", raw):

            year = parse_year(raw)

        # Fallback: parse a full date if possible.
        elif raw:

            parsed = pd.to_datetime(
                raw,
                errors="coerce",
                utc=True
            )

            if not pd.isna(parsed):

                parsed_day = parsed.date()

                period_start = parsed_day
                period_end = parsed_day + timedelta(days=1)

                year = parsed.year
                precision = "DAY"
                display_date = parsed_day.isoformat()

        # If no exact date/month was available,
        # use the year field as a year-only period.
        if period_start is None and year is not None:

            period_start = date(year, 1, 1)
            period_end = date(year + 1, 1, 1)

            precision = "YEAR"

            if not display_date:
                display_date = str(year)

        if period_start is None:

            return {
                "citing_publication_date": None,
                "citing_year": year,
                "date_precision": "MISSING",
                "window_status": "MISSING_DATE"
            }

        # Entire possible date period is inside the window.
        if (
            period_start >= start_day
            and period_end <= end_day
        ):

            window_status = "IN_WINDOW"

        # Entire possible date period is outside.
        elif (
            period_end <= start_day
            or period_start >= end_day
        ):

            window_status = "OUTSIDE_WINDOW"

        # Only part of the month/year overlaps the window.
        else:

            window_status = "AMBIGUOUS_DATE"

        return {
            "citing_publication_date": display_date,
            "citing_year": year,
            "date_precision": precision,
            "window_status": window_status
        }


    # ------------------------------------------------------------
    # 7. PAPER-LEVEL C3 + IC3 COLLECTION
    # ------------------------------------------------------------

    def fetch_s2_citation_metrics_one(row):

        aid = str(row["arxiv_id"])

        s2id = row["s2_paper_id"]

        t0 = pd.to_datetime(
            row["submission_date_v1"],
            errors="coerce",
            utc=True
        )

        current_citation_count = optional_int(
            row["citation_count"]
        )

        current_influential_count = optional_int(
            row["influential_citation_count"]
        )

        base = {

            "arxiv_id": aid,

            "s2_paper_id": safe_string(s2id),

            "submission_date_v1": (
                None if pd.isna(t0)
                else t0.isoformat()
            ),

            "s2_citation_count_current": (
                current_citation_count
            ),

            "s2_influential_citation_count_current": (
                current_influential_count
            )
        }

        def make_result(
            c3,
            ic3,
            c3_status,
            ic3_status,
            **extra
        ):

            output = {

                **base,

                "s2_citation_count_3y": c3,

                "s2_influential_citation_count_3y": ic3,

                "s2_c3_status": c3_status,

                "s2_ic3_status": ic3_status,

                "s2_c3_pages_fetched": 0,

                "s2_c3_records_scanned": 0,

                "s2_c3_observed_in_window": 0,

                "s2_c3_year_only_in_window": 0,

                "s2_c3_ambiguous_date_records": 0,

                "s2_c3_missing_date_records": 0,

                "s2_c3_missing_id_records": 0,

                "s2_c3_uncertain_unique_ids": 0,

                "s2_ic3_observed_in_window": 0,

                "s2_ic3_unknown_flags": 0,

                "s2_ic3_missing_date_records": 0,

                "s2_ic3_missing_id_records": 0,

                "s2_ic3_flag_conflict_ids": 0,

                "s2_c3_start_date": None,

                "s2_c3_end_date_exclusive": None,

                "s2_c3_error": None,

                **extra
            }

            return output

        # --------------------------------------------------------
        # A. No S2 match
        # --------------------------------------------------------

        metadata_status = row.get("metadata_status")
        if metadata_status not in ("MATCHED", "S2_NO_RECORD_RETURNED"):
            status = "REQUEST_ERROR" if metadata_status == "REQUEST_ERROR" else "PARTIAL"
            return make_result(None, None, status, status), [], False, False

        if pd.isna(s2id):

            return make_result(
                None,
                None,
                "NO_S2_MATCH",
                "NO_S2_MATCH"
            ), [], False, False

        # --------------------------------------------------------
        # B. Zero cumulative citation shortcut
        #
        # If the current total citation count is zero,
        # both C3 and IC3 must also be zero.
        #
        # An influential count of zero alone is NOT
        # sufficient to conclude that C3 is zero.
        # --------------------------------------------------------

        if (
            current_citation_count == 0
            and (
                current_influential_count is None
                or current_influential_count == 0
            )
        ):

            return make_result(
                0,
                0,
                "ZERO_CONFIRMED_BY_CUMULATIVE_COUNT",
                "ZERO_CONFIRMED_BY_CUMULATIVE_COUNT"
            ), [], False, False

        # --------------------------------------------------------
        # C. Missing t0
        # --------------------------------------------------------

        if pd.isna(t0):

            return make_result(
                None,
                None,
                "MISSING_T0",
                "MISSING_T0"
            ), [], False, False

        # --------------------------------------------------------
        # D. Three-year window
        # --------------------------------------------------------

        t1 = t0 + pd.DateOffset(years=config.citation_window_years)

        start_day = t0.date()

        end_day = t1.date()

        # The API filter is inclusive at the end.
        # Client-side classification applies the exclusive end.
        #
        # Actual research window:
        # [start_day, end_day)
        # --------------------------------------------------------

        fingerprint_data = {

            "version": "S2_C3_IC3_v1",

            "arxiv_id": aid,

            "s2_paper_id": str(s2id),

            "t0": t0.isoformat(),

            "current_citation_count": current_citation_count,

            "current_influential_count": current_influential_count,

            "start": start_day.isoformat(),

            "end": end_day.isoformat(),

            "citation_fields": S2_CITATION_FIELDS,

            "page_size": S2_PAGE_SIZE,

            "max_pages": S2_MAX_PAGES,

            "max_citations": S2_MAX_CITATIONS
        }

        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_data,
                sort_keys=True
            ).encode()
        ).hexdigest()

        safe_id = re.sub(
            r"[^A-Za-z0-9_.-]",
            "_",
            aid
        )

        summary_cache_path = (
            S2_SUMMARY_CACHE_DIR
            / f"c3_ic3_{safe_id}.json"
        )

        # --------------------------------------------------------
        # E. Load new summary cache
        # --------------------------------------------------------

        if not S2_REFRESH_CACHE:

            try:

                cached = http.cache.read_json_fast(
                    summary_cache_path,
                    validator=lambda x: (
                        isinstance(x, dict)
                        and isinstance(x.get("edges"), list)
                        and isinstance(x.get("result"), dict)
                    )
                ) or {}

                previous = cached.get("result", {})

                if (
                    cached.get("fingerprint") == fingerprint
                    and previous.get("s2_c3_status")
                        in S2_TERMINAL_STATUSES
                    and previous.get("s2_ic3_status")
                        in S2_TERMINAL_STATUSES
                ):

                    return (
                        previous,
                        cached.get("edges", []),
                        True,
                        False
                    )

            except Exception:
                pass

        # --------------------------------------------------------
        # F. Citation counters
        # --------------------------------------------------------

        # All unique citing papers within the window.
        c3_ids = set()

        # Unique influential citing papers within the window.
        ic3_ids = set()

        # IDs with uncertain date membership.
        c3_uncertain_ids = set()

        # IDs with uncertain influential status/date.
        ic3_uncertain_ids = set()

        # Conflicting records for the same citing paper.
        date_conflict_ids = set()

        flag_conflict_ids = set()

        # Citation edge records, one per unique citing paper ID.
        edge_map = {}

        pages = 0
        scanned = 0

        year_only_in_window = 0
        ambiguous_date_records = 0
        missing_date_records = 0

        c3_missing_id_records = 0
        ic3_missing_id_records = 0

        unknown_flags = 0
        malformed_records = 0

        complete = False
        api_limit = False
        page_limit = False

        error_message = None
        deferred_429 = False

        offset = 0

        # --------------------------------------------------------
        # G. Pagination
        # --------------------------------------------------------

        try:

            for _ in range(S2_MAX_PAGES):

                params = {

                    "fields": S2_CITATION_FIELDS,

                    "publicationDateOrYear": (
                        f"{start_day.isoformat()}:"
                        f"{end_day.isoformat()}"
                    ),

                    "offset": offset,

                    "limit": S2_PAGE_SIZE
                }

                url = (
                    f"{S2}/paper/"
                    f"{urllib.parse.quote(str(s2id), safe='')}"
                    f"/citations?"
                    f"{urllib.parse.urlencode(params)}"
                )

                # Build the old URL for backward-compatible
                # raw-cache reuse.
                legacy_params = dict(params)

                legacy_params["fields"] = (
                    S2_LEGACY_CITATION_FIELDS
                )

                legacy_url = (
                    f"{S2}/paper/"
                    f"{urllib.parse.quote(str(s2id), safe='')}"
                    f"/citations?"
                    f"{urllib.parse.urlencode(legacy_params)}"
                )

                data = s2_call(
                    url,
                    legacy_urls=[legacy_url]
                )

                page = data.get("data")

                if not isinstance(page, list):

                    raise ValueError(
                        "S2 citation response has no data list."
                    )

                pages += 1

                scanned += len(page)

                # ------------------------------------------------
                # Process each citation edge
                # ------------------------------------------------

                for edge in page:

                    if not isinstance(edge, dict):

                        malformed_records += 1

                        continue

                    citing = edge.get("citingPaper") or {}

                    if not isinstance(citing, dict):
                        citing = {}

                    flag = edge.get("isInfluential")

                    if flag is not True and flag is not False:
                        flag = None

                    citing_id = citing.get("paperId")

                    if citing_id:
                        citing_id = str(citing_id).strip()
                    else:
                        citing_id = None

                    date_info = classify_citation_date(
                        citing,
                        start_day,
                        end_day
                    )

                    window_status = date_info[
                        "window_status"
                    ]

                    # --------------------------------------------
                    # Date precision audit
                    # --------------------------------------------

                    if date_info["date_precision"] == "YEAR":
                        if window_status == "IN_WINDOW":
                            year_only_in_window += 1

                    if window_status == "AMBIGUOUS_DATE":
                        ambiguous_date_records += 1

                    if window_status == "MISSING_DATE":
                        missing_date_records += 1

                    # --------------------------------------------
                    # C3: all citations
                    # --------------------------------------------

                    if window_status == "IN_WINDOW":

                        if citing_id:
                            c3_ids.add(citing_id)
                        else:
                            c3_missing_id_records += 1

                    elif window_status in (
                        "AMBIGUOUS_DATE",
                        "MISSING_DATE"
                    ):

                        if citing_id:
                            c3_uncertain_ids.add(citing_id)
                        else:
                            c3_missing_id_records += 1

                    # --------------------------------------------
                    # IC3: influential citations only
                    # --------------------------------------------

                    if window_status != "OUTSIDE_WINDOW":

                        if flag is True:

                            if window_status == "IN_WINDOW":

                                if citing_id:
                                    ic3_ids.add(citing_id)
                                else:
                                    ic3_missing_id_records += 1

                            elif citing_id:

                                ic3_uncertain_ids.add(citing_id)

                            else:

                                ic3_missing_id_records += 1

                        elif flag is None:

                            unknown_flags += 1

                            if citing_id:
                                ic3_uncertain_ids.add(citing_id)
                            else:
                                ic3_missing_id_records += 1

                    # --------------------------------------------
                    # Citation edge aggregation
                    # One unique citing paper per target paper.
                    # --------------------------------------------

                    if citing_id:

                        candidate = {

                            "arxiv_id": aid,

                            "s2_paper_id": str(s2id),

                            "citing_paper_id": citing_id,

                            "citing_publication_date": (
                                date_info[
                                    "citing_publication_date"
                                ]
                            ),

                            "citing_year": (
                                date_info["citing_year"]
                            ),

                            "date_precision": (
                                date_info["date_precision"]
                            ),

                            "window_status": window_status,

                            "is_influential": flag,

                            "date_conflict": False,

                            "influential_flag_conflict": False
                        }

                        if citing_id not in edge_map:

                            edge_map[citing_id] = candidate

                        else:

                            old = edge_map[citing_id]

                            # Detect conflicting dates.
                            old_date = old[
                                "citing_publication_date"
                            ]

                            new_date = candidate[
                                "citing_publication_date"
                            ]

                            if (
                                old_date
                                and new_date
                                and old_date != new_date
                            ):

                                old["date_conflict"] = True

                                date_conflict_ids.add(citing_id)

                            old_window = old["window_status"]

                            if (
                                old_window != window_status
                                and old_window != "MISSING_DATE"
                                and window_status != "MISSING_DATE"
                            ):

                                old["date_conflict"] = True

                                old["window_status"] = "CONFLICT"

                                date_conflict_ids.add(citing_id)

                            # Prefer a more precise date.
                            if (
                                old["date_precision"] != "DAY"
                                and date_info["date_precision"] == "DAY"
                            ):

                                old["citing_publication_date"] = (
                                    date_info[
                                        "citing_publication_date"
                                    ]
                                )

                                old["citing_year"] = (
                                    date_info["citing_year"]
                                )

                                old["date_precision"] = "DAY"

                            # Detect conflicting influential flags.
                            old_flag = old["is_influential"]

                            if (
                                old_flag is not None
                                and flag is not None
                                and old_flag != flag
                            ):

                                old[
                                    "influential_flag_conflict"
                                ] = True

                                flag_conflict_ids.add(citing_id)

                            # True takes precedence for observed IC3.
                            if flag is True:

                                old["is_influential"] = True

                            elif old_flag is None:

                                old["is_influential"] = flag

                # ------------------------------------------------
                # Pagination control
                # ------------------------------------------------

                next_offset = data.get("next")

                if next_offset is None:

                    if scanned >= S2_MAX_CITATIONS:

                        api_limit = True

                    else:

                        complete = True

                    break

                if (
                    isinstance(next_offset, bool)
                    or not isinstance(next_offset, int)
                    or next_offset <= offset
                ):

                    raise ValueError(
                        "S2 citation pagination stalled."
                    )

                if next_offset >= S2_MAX_CITATIONS:

                    api_limit = True

                    break

                offset = next_offset

            else:

                page_limit = True

        except DeferredRequest as error:

            # 429 不在当前论文上原地等待；把论文放回延迟队列。
            # 已成功的分页已写入 raw cache，下一轮从头重建时会直接命中缓存，
            # 因此真正重新访问网络的只有未成功的分页。
            deferred_429 = True
            error_message = str(error)[:1000]

        except Exception as error:

            error_message = str(error)[:1000]

        # --------------------------------------------------------
        # H. Determine coverage and label validity
        # --------------------------------------------------------

        observed_c3 = len(c3_ids)

        observed_ic3 = len(ic3_ids)

        # An uncertain record for an ID already confirmed
        # inside the window does not necessarily prevent counting
        # that unique citing paper.
        unresolved_c3_ids = (
            c3_uncertain_ids - c3_ids
        )

        unresolved_ic3_ids = (
            ic3_uncertain_ids - ic3_ids
        )

        unresolved_c3_ids |= date_conflict_ids

        unresolved_ic3_ids |= (
            date_conflict_ids | flag_conflict_ids
        )

        c3_incomplete = bool(
            unresolved_c3_ids
            or c3_missing_id_records
            or malformed_records
        )

        ic3_incomplete = bool(
            unresolved_ic3_ids
            or ic3_missing_id_records
            or malformed_records
        )

        # API-level completeness takes priority.
        if api_limit:

            c3_status = "API_9999_LIMIT_REACHED"
            ic3_status = "API_9999_LIMIT_REACHED"

        elif page_limit:

            c3_status = "PAGE_LIMIT_REACHED"
            ic3_status = "PAGE_LIMIT_REACHED"

        elif not complete:

            c3_status = (
                "REQUEST_ERROR_PARTIAL"
                if pages
                else "REQUEST_ERROR"
            )

            ic3_status = c3_status

        else:

            c3_status = (
                "INCOMPLETE_DATE_OR_ID_COVERAGE"
                if c3_incomplete
                else "COMPLETE"
            )

            ic3_status = (
                "INCOMPLETE_INFLUENTIAL_COVERAGE"
                if ic3_incomplete
                else "COMPLETE"
            )

        # Do not present incomplete counts as final labels.
        c3_value = (
            observed_c3
            if c3_status == "COMPLETE"
            else None
        )

        ic3_value = (
            observed_ic3
            if ic3_status == "COMPLETE"
            else None
        )

        # --------------------------------------------------------
        # I. Final paper-level output
        # --------------------------------------------------------

        output = make_result(
            c3_value,
            ic3_value,
            c3_status,
            ic3_status,

            s2_c3_start_date=start_day.isoformat(),

            s2_c3_end_date_exclusive=end_day.isoformat(),

            s2_c3_pages_fetched=pages,

            s2_c3_records_scanned=scanned,

            s2_c3_observed_in_window=observed_c3,

            s2_c3_year_only_in_window=year_only_in_window,

            s2_c3_ambiguous_date_records=ambiguous_date_records,

            s2_c3_missing_date_records=missing_date_records,

            s2_c3_missing_id_records=c3_missing_id_records,

            s2_c3_uncertain_unique_ids=len(
                unresolved_c3_ids
            ),

            s2_ic3_observed_in_window=observed_ic3,

            s2_ic3_unknown_flags=unknown_flags,

            s2_ic3_missing_date_records=missing_date_records,

            s2_ic3_missing_id_records=ic3_missing_id_records,

            s2_ic3_flag_conflict_ids=len(
                flag_conflict_ids
            ),

            s2_c3_error=error_message
        )

        edges = list(edge_map.values())

        # --------------------------------------------------------
        # J. Save new summary cache
        # --------------------------------------------------------

        if (
            c3_status in S2_TERMINAL_STATUSES
            and ic3_status in S2_TERMINAL_STATUSES
        ):

            http.cache.write_json_fast(
                summary_cache_path,
                {"fingerprint": fingerprint, "result": output, "edges": edges}
            )

        return output, edges, False, deferred_429


    # ------------------------------------------------------------
    # 8. RUN C3 + IC3 FOR ALL PAPERS
    # ------------------------------------------------------------

    citation_input = s2_input.merge(
        s2_metadata_df[
            [
                "arxiv_id",
                "s2_paper_id",
                "citation_count",
                "influential_citation_count"
            ]
        ],
        on="arxiv_id",
        how="left",
        validate="one_to_one"
    )

    citation_input = citation_input.merge(s2_audit_df[["arxiv_id", "status"]].rename(columns={"status": "metadata_status"}), on="arxiv_id", validate="one_to_one")

    cache_hits = 0

    edge_columns = [
        "arxiv_id",
        "s2_paper_id",
        "citing_paper_id",
        "citing_publication_date",
        "citing_year",
        "date_precision",
        "window_status",
        "is_influential",
        "date_conflict",
        "influential_flag_conflict"
    ]

    edge_tmp_path = (
        S2_OUTPUT_DIR
        / "semantic_scholar_citation_edges_3y.tmp.csv"
    )

    edge_final_path = (
        S2_OUTPUT_DIR
        / "semantic_scholar_citation_edges_3y.csv"
    )

    # Do not delete an earlier edge snapshot at startup. Each completed paper
    # is already persisted in its per-paper Summary Cache; the old snapshot is
    # retained until a complete replacement can be written at the end.
    input_records = citation_input.to_dict("records")
    input_ids = [str(row["arxiv_id"]) for row in input_records]
    progress_path = (
        S2_OUTPUT_DIR / "semantic_scholar_citation_3y_progress.csv"
    )

    # Preserve any earlier progress rows in case this run is interrupted before
    # it reaches the next checkpoint. New results override old rows by arXiv ID.
    prior_progress = {}
    if progress_path.is_file():
        try:
            prior_df = pd.read_csv(progress_path, dtype={"arxiv_id": str})
            if "arxiv_id" in prior_df.columns:
                prior_progress = {
                    str(row["arxiv_id"]): row
                    for row in prior_df.to_dict("records")
                }
        except Exception as exc:
            print(f"Existing Citation progress snapshot could not be read; "
                  f"per-paper Summary Cache remains authoritative: {exc}")

    def merged_progress_rows(results):
        merged = dict(prior_progress)
        merged.update({str(aid): row for aid, row in results.items()})
        return [merged[aid] for aid in input_ids if aid in merged]

    def save_queue_progress(results, edges_by_id, attempts, deferred_events=0,
                            queue_name="main", **_):
        # Heartbeat every 100 queue attempts; this is intentionally independent
        # of Drive checkpoint writes so the notebook remains visibly active.
        if attempts % 100 == 0:
            print(
                f"S2 citation queue: {len(results)}/{len(input_records)} "
                f"papers visited; attempts={attempts}; "
                f"deferred_429={deferred_events}; queue={queue_name}"
            )

        # A full CSV rewrite every 10 papers creates quadratic Drive I/O.
        # Per-paper Summary Cache writes are the durable checkpoint; the CSV is
        # only a coarse human-readable snapshot.
        if attempts % 500 == 0:
            atomic_csv(
                pd.DataFrame(merged_progress_rows(results)),
                progress_path,
                index=False
            )

    # 429 采用延迟队列：单篇失败不阻塞主队列；连续 429 时由 HTTPClient
    # 设置全局冷却，调度器暂停主队列并优先探测待重试论文。
    queue_result = _run_deferred_queue(
        input_records,
        fetch_s2_citation_metrics_one,
        http,
        max_deferred_retries=10,
        on_progress=save_queue_progress
    )

    result_by_id = queue_result["results"]
    edges_by_id = queue_result["edges_by_id"]
    citation_rows = merged_progress_rows(result_by_id)
    cache_hits = queue_result["summary_cache_hits"]

    # 最后一次尝试覆盖同一篇论文的部分结果，按输入顺序汇总，避免重复 citation edges。
    # 进度快照保留上一轮尚未在本轮触及的记录；本轮结果优先。
    edge_rows = [
        edge
        for aid in input_ids
        for edge in edges_by_id.get(aid, [])
    ]
    edge_df = pd.DataFrame(edge_rows, columns=edge_columns)
    atomic_csv(edge_df, edge_tmp_path, index=False)
    edge_tmp_path.replace(edge_final_path)

    # 保存最终进度快照；即使有论文在本轮重试预算内仍未恢复，也明确保留错误状态。
    atomic_csv(pd.DataFrame(citation_rows), progress_path, index=False)

    print(
        "Semantic Scholar queue summary:",
        {
            "papers": len(input_records),
            "paper_attempts": queue_result["attempts"],
            "deferred_429_events": queue_result["deferred_events"],
            "summary_cache_hits": cache_hits,
            "retry_budget_exhausted": len(queue_result["exhausted_ids"]),
            "raw_cache_hits": http.cache.hits
        }
    )

    # Final label table.
    s2_citation_3y_df = pd.DataFrame(citation_rows)

    atomic_csv(s2_citation_3y_df, 
        S2_OUTPUT_DIR / "semantic_scholar_citation_3y.csv",
        index=False
    )

    # Backward-compatible filename and variable.
    s2_ic3_df = s2_citation_3y_df.copy()

    atomic_csv(s2_ic3_df, 
        S2_OUTPUT_DIR
        / "semantic_scholar_influential_citation_3y.csv",
        index=False
    )

    # ------------------------------------------------------------
    # 9. BUILD COMPATIBLE S2 MASTER INPUT
    # ------------------------------------------------------------

    s2_all_df = s2_metadata_df.rename(
        columns={

            "title": "s2_title",

            "doi": "s2_doi",

            "citation_count": "s2_citation_count_current",

            "influential_citation_count":
                "s2_influential_citation_count_current",

            "reference_count": "s2_reference_count_current",

            "publication_date": "s2_publication_date",

            "year": "s2_year",

            "authors_json": "s2_authors_json"
        }
    ).merge(

        s2_citation_3y_df[
            [
                "arxiv_id",

                "s2_citation_count_3y",

                "s2_influential_citation_count_3y",

                "s2_c3_status",

                "s2_ic3_status"
            ]
        ],

        on="arxiv_id",

        how="left",

        validate="one_to_one"
    )

    # Preserve existing downstream Cell compatibility:
    # s2_df = matched papers only.
    s2_df = s2_all_df[
        s2_all_df["s2_paper_id"].notna()
    ].copy()

    atomic_csv(s2_df, 
        Path(PROCESSED) / "semantic_scholar_papers.csv",
        index=False
    )


    # ------------------------------------------------------------
    # 10. SAVE REQUEST LOG
    # ------------------------------------------------------------

    (S2_OUTPUT_DIR / "request_log.json").write_text(
        json.dumps(
            S2_REQUEST_LOG,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


    # ------------------------------------------------------------
    # 11. FINAL SUMMARY
    # ------------------------------------------------------------

    print("\n" + "=" * 65)
    print("SEMANTIC SCHOLAR: C3 + IC3 COLLECTION SUMMARY")
    print("=" * 65)

    print("Input papers:", len(s2_input))

    print(
        "Matched S2 papers:",
        s2_df["s2_paper_id"].notna().sum()
    )

    print("\nS2 matching status:")
    print(
        s2_audit_df["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nC3 status distribution:")
    print(
        s2_citation_3y_df["s2_c3_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nIC3 status distribution:")
    print(
        s2_citation_3y_df["s2_ic3_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        "\nValid C3 labels:",
        s2_citation_3y_df[
            "s2_citation_count_3y"
        ].notna().sum()
    )

    print(
        "Valid IC3 labels:",
        s2_citation_3y_df[
            "s2_influential_citation_count_3y"
        ].notna().sum()
    )

    print(
        "C3 complete:",
        (
            s2_citation_3y_df["s2_c3_status"]
            == "COMPLETE"
        ).sum()
    )

    print(
        "IC3 complete:",
        (
            s2_citation_3y_df["s2_ic3_status"]
            == "COMPLETE"
        ).sum()
    )

    print("\nSummary cache hits:", cache_hits)

    print(
        "Network requests this run:",
        len(S2_REQUEST_LOG)
    )

    print("\nSaved files:")

    print(
        S2_OUTPUT_DIR
        / "semantic_scholar_citation_3y.csv"
    )

    print(
        S2_OUTPUT_DIR
        / "semantic_scholar_influential_citation_3y.csv"
    )

    print(edge_final_path)

    print(
        Path(PROCESSED)
        / "semantic_scholar_papers.csv"
    )

    print("\nPreview:")
    log_preview(s2_citation_3y_df.head(10))

    print("\nSEMANTIC SCHOLAR C3 + IC3 COLLECTION FINISHED.")

    return {"s2_all_df": s2_all_df, "s2_audit_df": s2_audit_df, "s2_citation_3y_df": s2_citation_3y_df}


def run(config: Config, http: HTTPClient, papers_df=None, seed_df=None):
    """返回统一 SourceResult；异常由中控记录，绝不构造成功或零值。"""
    from fetch.control.state import source_result
    tables = collect(config, http, papers_df=papers_df, seed_df=seed_df)
    count = len(papers_df) if papers_df is not None else len(seed_df) if seed_df is not None else 0
    return source_result('semantic_scholar', tables, http, count)

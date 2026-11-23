"""显式调用的 Notebook 迁移模块。输入输出与主键见 README 和 DATA_DICTIONARY。"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fetch.control.config import Config
    from fetch.control.tools import HTTPClient
    import pandas as pd
from fetch.control.tools import atomic_csv, log_preview, log_print as print

def collect(config: Config, http: HTTPClient, papers_df=None, seed_df=None) -> dict[str, pd.DataFrame]:
    ROOT, RAW, PROCESSED = config.paths
    # ============================================================
    # DBLP AUTHOR FEATURE COLLECTION — VERSION 3.1.1
    #
    # Input:
    #   papers_df
    #   arxiv_id, title_v1, authors_v1_json,
    #   submission_date_v1, doi_v1
    #
    # Output:
    #   Author Publication Count before t0
    #   Academic Age before t0
    #
    # Database:
    #   Local DBLP XML dump
    #
    # Improvements:
    #   1. Persistent DBLP author IDs
    #   2. Author aliases and merged-profile handling
    #   3. Duplicate publication-version grouping
    #   4. Conservative author identity resolution with source-target conflict checks
    #   5. Two-pass streaming XML parsing with persistent intermediate cache
    #   6. Strict temporal cutoff and focal-work self-exclusion
    #   7. Complete matching and feature audit
    # ============================================================

    import os
    import re
    import sys
    import json
    import gzip
    import time
    import shutil
    import hashlib
    import pickle
    import subprocess
    import unicodedata
    import requests
    import pandas as pd
    import gc

    def log_memory(stage):
        """Log process RSS and system-available RAM without making psutil mandatory."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            rss_gb = process.memory_info().rss / (1024 ** 3)
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            print(
                f"[RAM] {stage}: process RSS={rss_gb:.2f} GB; "
                f"system available={available_gb:.2f} GB"
            )
        except Exception:
            pass

    from pathlib import Path
    from collections import defaultdict, Counter
    from urllib.parse import urlsplit
    from fetch.control.tools import quiet_tqdm as tqdm


    # DBLP record elements needed by the two streaming passes.
    # The parser scans author homepage records (www) plus publication records.
    DBLP_PUBLICATION_TAGS = {
        "article",
        "inproceedings",
        "proceedings",
        "book",
        "incollection",
        "phdthesis",
        "mastersthesis",
    }
    DBLP_RECORD_TAGS = tuple(sorted(DBLP_PUBLICATION_TAGS | {"www"}))


    # ============================================================
    # PART 0. ENVIRONMENT AND INPUT
    # ============================================================

    from lxml import etree

    DBLP_OUTPUT_DIR = PROCESSED / "dblp_author_features_v3_1_1"
    DBLP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Persistent intermediate cache lives on Google Drive, not in the ephemeral
    # Colab runtime. First run builds the two indexes; later runs load them.
    DBLP_CACHE_DIR = ROOT / "data/intermediate/dblp_v3_1_index"
    DBLP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Set True only when you intentionally want to rebuild both indexes.
    DBLP_FORCE_REBUILD_CACHE = config.refresh("dblp")
    # Bump CACHE_SCHEMA_VERSION when Pass 1 cache extraction logic or either
    # serialized payload structure changes. Bump IDENTITY_LOGIC_VERSION when
    # author identity rules change; bump HISTORY_LOGIC_VERSION when Pass 2
    # historical deduplication / indexing rules change.
    DBLP_CACHE_SCHEMA_VERSION = "3.1.0"  # Keep unchanged to reuse the existing Pass 1 cache.
    DBLP_IDENTITY_LOGIC_VERSION = "priority-full-name-over-signature-v2"
    DBLP_HISTORY_LOGIC_VERSION = "deduplicated-focal-work-index-v1"

    required_columns = {
        "arxiv_id",
        "title_v1",
        "authors_v1_json",
        "submission_date_v1"
    }

    missing_columns = required_columns - set(papers_df.columns)

    if missing_columns:
        raise ValueError(
            f"papers_df missing columns: {missing_columns}"
        )


    def is_missing(value):
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False


    papers_input = papers_df.copy()

    papers_input["arxiv_id"] = (
        papers_input["arxiv_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"v\d+$", "", regex=True)
    )

    papers_input = (
        papers_input
        .drop_duplicates("arxiv_id")
        .reset_index(drop=True)
    )
    input_paper_count = len(papers_input)

    papers_input["submission_date_v1"] = pd.to_datetime(
        papers_input["submission_date_v1"],
        errors="coerce",
        utc=True
    )

    if papers_input["submission_date_v1"].isna().any():
        raise ValueError(
            "Some papers have missing v1 submission dates."
        )

    print("=" * 65)
    print("DBLP AUTHOR FEATURE COLLECTION — V3.1.1")
    print("=" * 65)
    print("Input arXiv papers:", input_paper_count)


    # ============================================================
    # PART 1. TEXT NORMALIZATION
    # ============================================================

    def clean_text(element):
        if element is None:
            return ""
        return "".join(element.itertext()).strip()


    def normalize_name(name):
        """
        Normalize punctuation, accents, spaces and case.
        Used for matching only, not for defining author identity.
        """
        text = unicodedata.normalize(
            "NFKD", str(name or "")
        ).casefold()

        text = "".join(
            c for c in text
            if not unicodedata.combining(c)
        )

        return "".join(
            c for c in text
            if c.isalnum()
        )


    def name_signature(name):
        """
        First initial + surname.

        Examples:
            Ian Goodfellow
            Ian J. Goodfellow

        Both produce the same signature.

        This is only a secondary matching signal.
        It is never sufficient by itself to establish
        a persistent author identity.
        """
        text = str(name or "").strip()

        if "," in text:
            left, right = text.split(",", 1)
            text = right + " " + left

        text = unicodedata.normalize("NFKD", text).casefold()

        text = "".join(
            c for c in text
            if not unicodedata.combining(c)
        )

        tokens = re.findall(r"[^\W_]+", text, flags=re.UNICODE)

        if len(tokens) < 2:
            return None

        if tokens[-1] in {"jr", "sr", "ii", "iii", "iv"}:
            if len(tokens) >= 3:
                tokens.pop()

        # DBLP may append a four-digit disambiguation suffix.
        if re.fullmatch(r"\d{4}", tokens[-1]):
            tokens.pop()

        if len(tokens) < 2:
            return None

        return tokens[0][0] + "|" + tokens[-1]


    def normalize_title(title):
        text = unicodedata.normalize(
            "NFKD", str(title or "")
        ).casefold()

        text = "".join(
            c for c in text
            if not unicodedata.combining(c)
        )

        return " ".join(
            re.findall(r"[a-z0-9]+", text)
        )


    def normalize_doi(value):
        if not value:
            return None

        text = str(value).strip().lower()

        text = re.sub(
            r"^https?://(?:dx\.)?doi\.org/",
            "",
            text
        )

        text = re.sub(r"^doi:\s*", "", text)

        text = text.rstrip(".,;:)]}")

        return text or None


    def classify_identity_confidence(identity_method):
        """Map the identity evidence used by the matcher to an auditable tier."""
        if identity_method in {
            "RAW_EXACT_ALIAS",
            "EXACT_ALIAS",
            "NORMALIZED_FULL_NAME_ALIAS",
            "TARGET_PAPER_CONTEXT_PROFILE_ALIAS",
        }:
            return "HIGH"
        if identity_method == "UNIQUE_INITIAL_SURNAME_ALIAS":
            return "MEDIUM"
        if identity_method == "DBLP_EXACT_AUTHOR_STRING":
            return "LOW"
        return "UNRESOLVED"


    ARXIV_PATTERN = re.compile(
        r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)\s*"
        r"(\d{4}\.\d{4,5}|[a-z][a-z.-]*/\d{7})"
        r"(?:v\d+)?",
        re.I
    )

    DOI_PATTERN = re.compile(
        r"10\.\d{4,9}/[^\s<>\"'?#]+",
        re.I
    )


    def extract_arxiv_ids(text):
        results = set()

        for match in ARXIV_PATTERN.findall(str(text or "")):
            results.add(
                re.sub(r"v\d+$", "", match.lower())
            )

        return results


    def extract_dois(text):
        results = set()

        for match in DOI_PATTERN.findall(str(text or "")):
            doi = normalize_doi(match)

            if doi:
                results.add(doi)

        return results


    # ============================================================
    # PART 2. PREPARE INPUT PAPER AND AUTHOR INDEX
    # ============================================================

    paper_meta = {}
    source_author_rows = []

    title_to_aids = defaultdict(set)
    doi_to_aids = defaultdict(set)

    for row in papers_input.to_dict("records"):

        aid = row["arxiv_id"]

        raw_authors = row["authors_v1_json"]

        if isinstance(raw_authors, str):
            try:
                authors = json.loads(raw_authors)
            except (ValueError, TypeError):
                authors = []
        elif isinstance(raw_authors, list):
            authors = raw_authors
        else:
            authors = []

        if not isinstance(authors, list):
            authors = []

        author_names = []

        for author in authors:

            if isinstance(author, dict):
                name = str(author.get("name") or "").strip()
            else:
                name = str(author or "").strip()

            if name:
                author_names.append(name)

        t0 = row["submission_date_v1"]

        title = str(row.get("title_v1") or "").strip()
        title_key = normalize_title(title)

        doi = normalize_doi(row.get("doi_v1"))

        dois = {doi} if doi else set()

        paper_meta[aid] = {
            "arxiv_id": aid,
            "title": title,
            "title_key": title_key,
            "t0": t0,
            "t0_year": int(t0.year),
            "dois": dois,
            "authors": author_names
        }

        if title_key:
            title_to_aids[title_key].add(aid)

        for d in dois:
            doi_to_aids[d].add(aid)

        for position, name in enumerate(author_names, start=1):

            source_author_rows.append({
                "arxiv_id": aid,
                "author_position": position,
                "author_name_arxiv_v1": name,
                "profile_person_id": None,
                "profile_match_status": None,
                "profile_match_method": None,
                "identity_id": None,
                "identity_method": None,
                "identity_confidence": "UNRESOLVED",
                "identity_alias_raw": None
            })

    # The parsed metadata needed downstream is now held in paper_meta and
    # source_author_rows. Release the duplicate 30k-row DataFrame copy.
    del papers_input
    gc.collect()
    log_memory("after parsing arXiv input metadata")

    source_author_by_paper = defaultdict(list)

    for author in source_author_rows:
        source_author_by_paper[
            author["arxiv_id"]
        ].append(author)

    exact_input_lookup = defaultdict(set)
    signature_input_lookup = defaultdict(set)

    for author in source_author_rows:

        occ = (
            author["arxiv_id"],
            author["author_position"]
        )

        exact = normalize_name(
            author["author_name_arxiv_v1"]
        )

        signature = name_signature(
            author["author_name_arxiv_v1"]
        )

        if exact:
            exact_input_lookup[exact].add(occ)

        if signature:
            signature_input_lookup[signature].add(occ)


    print("Distinct author names:",
          len(exact_input_lookup))


    # ============================================================
    # PART 3. FIND OR DOWNLOAD LOCAL DBLP XML
    # ============================================================

    DBLP_XML_URL = "https://dblp.org/xml/dblp.xml.gz"


    def download_file(url, destination):

        destination = Path(destination)
        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temp_path = destination.with_suffix(
            destination.suffix + ".part"
        )

        print("Downloading:", url)

        with http.request(
            "GET", url,
            stream=True,
            timeout=(30, 300)
        ) as response:

            response.raise_for_status()

            total = int(
                response.headers.get("Content-Length", 0)
            )

            with open(temp_path, "wb") as f:

                progress = tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    desc="DBLP download"
                )

                for chunk in response.iter_content(
                    chunk_size=8 * 1024 * 1024
                ):

                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))

                progress.close()

        temp_path.replace(destination)


    def find_existing_dblp():

        explicit = config.dblp_xml_path

        if explicit:
            explicit = Path(explicit)

            if explicit.is_file():
                return explicit

        found = set()

        for base in [RAW, ROOT]:

            if not base.exists():
                continue

            for pattern in [
                "dblp*.xml.gz",
                "dblp.xml"
            ]:

                for path in base.rglob(pattern):

                    if (
                        path.is_file()
                        and path.stat().st_size > 50_000_000
                    ):
                        found.add(path)

        if not found:
            return None

        return max(
            found,
            key=lambda p: p.stat().st_mtime
        )


    DBLP_XML_PATH = find_existing_dblp()

    if DBLP_XML_PATH is None:

        DBLP_XML_PATH = RAW / "dblp" / "dblp.xml.gz"

        download_file(
            DBLP_XML_URL,
            DBLP_XML_PATH
        )

    else:

        print("Existing DBLP XML found.")
        print("Reusing:", DBLP_XML_PATH)

    print(
        "DBLP XML size (GB):",
        round(DBLP_XML_PATH.stat().st_size / 1024**3, 3)
    )


    # ============================================================
    # PART 4 + PART 5 — FIXED DBLP XML / DTD PARSER
    #
    # Fix 1: Always obtain a valid official DBLP DTD.
    # Fix 2: Validate the required %field; declaration.
    # Fix 3: Stream gzip directly without decompressing 1GB XML.
    # Fix 4: Resolve local DTD using the XML directory.
    # Fix 5: Keep memory usage low.
    # ============================================================

    import os
    import re
    import gzip
    import requests

    from pathlib import Path


    # ------------------------------------------------------------
    # 4.1 Locate the correct DTD
    # ------------------------------------------------------------

    DBLP_DTD_PATH = (
        DBLP_XML_PATH.parent / "dblp.dtd"
    )

    DBLP_DTD_URL = "https://dblp.org/xml/dblp.dtd"

    print("\nChecking official DBLP DTD...")


    # ------------------------------------------------------------
    # 4.2 Download a fresh official DTD
    # ------------------------------------------------------------

    if DBLP_DTD_PATH.exists() and not config.refresh("dblp"):
        dtd_content = DBLP_DTD_PATH.read_bytes()
    else:
        dtd_content = http.request("GET", DBLP_DTD_URL).content

    # ------------------------------------------------------------
    # 4.3 Validate DTD before saving
    # ------------------------------------------------------------

    if len(dtd_content) < 5000:

        raise RuntimeError(
            "Downloaded DBLP DTD is unexpectedly small."
        )

    if not re.search(
        rb'<!ENTITY\s+%\s*field\b',
        dtd_content,
        flags=re.IGNORECASE
    ):

        raise RuntimeError(
            "Invalid DBLP DTD: missing %field; declaration."
        )

    if not re.search(
        rb'<!ELEMENT\s+article\b',
        dtd_content,
        flags=re.IGNORECASE
    ):

        raise RuntimeError(
            "Invalid DBLP DTD: missing article declaration."
        )


    # Always replace the potentially invalid old DTD.

    temp_dtd = DBLP_DTD_PATH.with_suffix(
        ".dtd.tmp"
    )

    temp_dtd.write_bytes(
        dtd_content
    )

    temp_dtd.replace(
        DBLP_DTD_PATH
    )

    print("Official DTD downloaded and validated.")

    print("DTD size:", len(dtd_content), "bytes")

    print("DTD path:", DBLP_DTD_PATH)


    # ------------------------------------------------------------
    # 4.4 Validate DTD using lxml
    # ------------------------------------------------------------

    try:

        parsed_dtd = etree.DTD(
            str(DBLP_DTD_PATH)
        )

        print("DTD successfully parsed by lxml.")

    except Exception as e:

        raise RuntimeError(
            f"DBLP DTD validation failed: {e}"
        )


    # ------------------------------------------------------------
    # 5.1 Use original compressed XML
    # ------------------------------------------------------------

    DBLP_XML_PARSE_PATH = DBLP_XML_PATH

    if not DBLP_XML_PARSE_PATH.exists():

        raise FileNotFoundError(
            f"DBLP XML not found: {DBLP_XML_PARSE_PATH}"
        )

    print("\nDBLP XML:", DBLP_XML_PARSE_PATH)

    print(
        "XML size:",
        round(
            DBLP_XML_PARSE_PATH.stat().st_size / 1024**3,
            3
        ),
        "GB"
    )


    # ------------------------------------------------------------
    # 5.2 Streaming parser
    # ------------------------------------------------------------

    def make_dblp_iterator(path):

        path = Path(path).resolve()

        def parser_generator():

            original_cwd = os.getcwd()

            try:

                # The XML references dblp.dtd using a relative path.
                # Make the XML directory the working directory.

                os.chdir(
                    path.parent
                )

                if path.suffix.lower() == ".gz":

                    # Read gzip as a binary stream.
                    # No full XML decompression required.

                    with gzip.open(
                        path,
                        "rb"
                    ) as xml_stream:

                        yield from etree.iterparse(

                            xml_stream,

                            events=("end",),

                            tag=DBLP_RECORD_TAGS,

                            load_dtd=True,

                            no_network=True,

                            resolve_entities=True,

                            huge_tree=True,

                            recover=False

                        )

                else:

                    yield from etree.iterparse(

                        str(path),

                        events=("end",),

                        tag=DBLP_RECORD_TAGS,

                        load_dtd=True,

                        no_network=True,

                        resolve_entities=True,

                        huge_tree=True,

                        recover=False

                    )

            finally:

                os.chdir(
                    original_cwd
                )

        return parser_generator()


    # ------------------------------------------------------------
    # 5.3 Test the parser before the full scan
    # ------------------------------------------------------------

    print("\nTesting DBLP XML parser...")

    probe = make_dblp_iterator(
        DBLP_XML_PARSE_PATH
    )

    try:

        event, first_record = next(probe)

        print("XML parser successfully opened DBLP.")

        print("First record type:", first_record.tag)

        print("First record key:", first_record.get("key"))

    finally:

        probe.close()


    print("\n" + "=" * 65)

    print("DBLP XML / DTD PARSER READY")

    print("Original compressed XML preserved.")

    print("No additional full-size decompressed XML required.")

    print("=" * 65)


    # ============================================================
    # PART 6. STREAMING XML ITERATOR
    # ============================================================

    def iter_dblp_records():

        context = make_dblp_iterator(
            DBLP_XML_PARSE_PATH
        )

        for _, element in context:

            parent = element.getparent()

            if parent is None or parent.tag != "dblp":
                continue

            yield element

            # Release processed XML memory.
            element.clear()

            while element.getprevious() is not None:
                del parent[0]


    def child_texts(element, tag):

        return [
            clean_text(x)
            for x in element.findall(tag)
            if clean_text(x)
        ]


    def get_year(element):

        values = child_texts(element, "year")

        if not values:
            return None

        match = re.search(r"\b(18|19|20|21)\d{2}\b", values[0])

        return int(match.group(0)) if match else None


    def get_record_links(element):

        values = []

        for tag in ["ee", "url", "note", "eprint"]:

            for child in element.findall(tag):

                value = clean_text(child)

                if value:
                    values.append(value)

        return values


    def make_dblp_record_key(element, tag=None, title=None, year=None, authors=None):
        """Return a stable DBLP key, including a deterministic fallback."""
        tag = tag or element.tag
        if title is None:
            title_values = child_texts(element, "title")
            title = title_values[0] if title_values else ""
        if year is None:
            year = get_year(element)
        if authors is None:
            authors = child_texts(element, "author")

        key = element.get("key")
        if key:
            return key

        fingerprint = hashlib.md5(
            (tag + title + str(year) + "|".join(authors)).encode("utf-8")
        ).hexdigest()
        return "generated/" + fingerprint


    # ============================================================
    # PART 6B. PERSISTENT CACHE FINGERPRINTS AND HELPERS
    # ============================================================

    import datetime as _dt


    def quick_file_fingerprint(path, sample_bytes=1024 * 1024):
        """Cheap source fingerprint: stat metadata + first/last 1 MiB.

        This avoids rereading the entire ~1 GB XML on every cache hit. A full MD5
        is computed once on the first build and retained in the cache manifest.
        """
        path = Path(path)
        stat = path.stat()
        digest = hashlib.sha256()
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        with open(path, "rb") as handle:
            head = handle.read(sample_bytes)
            if stat.st_size > sample_bytes:
                handle.seek(max(0, stat.st_size - sample_bytes))
            tail = handle.read(sample_bytes)
        digest.update(head)
        digest.update(tail)
        return {
            "path": str(path.resolve()),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sample_sha256": digest.hexdigest(),
        }


    def atomic_pickle_gzip_dump(payload, path):
        # 保留 Notebook 的压缩格式和文件名，同时登记校验元数据。
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + '.tmp')
        with gzip.open(temp_path, 'wb', compresslevel=3) as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path.replace(path)
        http.cache.record(path, {'index': path.name})


    def load_pickle_gzip(path):
        # DBLP 索引可能很大，流式校验，避免把整个压缩索引再次复制到内存。
        from fetch.control.tools import checksum, fingerprint, SCHEMA_VERSION, ALGORITHM_VERSION
        meta = http.cache.meta(path)
        if meta.exists():
            record = json.loads(meta.read_text())
            if not (record.get('sha256') == checksum(path)
                    and record.get('request_fingerprint') == fingerprint({'index': Path(path).name})
                    and record.get('schema_version') == SCHEMA_VERSION
                    and record.get('algorithm_version') == ALGORITHM_VERSION
                    and record.get('source') == 'dblp' and record.get('complete') is True):
                raise ValueError('DBLP index integrity mismatch')
        with gzip.open(path, 'rb') as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or 'cache_key' not in payload:
            raise ValueError('Invalid DBLP index structure')
        if not meta.exists(): http.cache.record(path, {'index': Path(path).name})
        http.cache.hits += 1
        return payload


    def atomic_json_dump(payload, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)


    def sha256_json(payload):
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


    DBLP_XML_QUICK_FINGERPRINT = quick_file_fingerprint(DBLP_XML_PATH)

    paper_fingerprint_payload = [
        {
            "arxiv_id": aid,
            "title_key": meta["title_key"],
            "t0": meta["t0"].isoformat(),
            "dois": sorted(meta["dois"]),
            "authors": meta["authors"],
        }
        for aid, meta in sorted(paper_meta.items())
    ]
    DBLP_INPUT_FINGERPRINT = sha256_json(paper_fingerprint_payload)

    DBLP_CACHE_KEY = sha256_json({
        "cache_schema_version": DBLP_CACHE_SCHEMA_VERSION,
        "xml_quick_fingerprint": DBLP_XML_QUICK_FINGERPRINT,
        "input_fingerprint": DBLP_INPUT_FINGERPRINT,
    })

    DBLP_PASS1_CACHE_PATH = DBLP_CACHE_DIR / f"pass1_{DBLP_CACHE_KEY}.pkl.gz"
    DBLP_MANIFEST_PATH = DBLP_CACHE_DIR / f"manifest_{DBLP_CACHE_KEY}.json"

    try:
        DBLP_EXISTING_MANIFEST = (
            json.loads(DBLP_MANIFEST_PATH.read_text(encoding="utf-8"))
            if DBLP_MANIFEST_PATH.exists() else {}
        )
    except Exception:
        DBLP_EXISTING_MANIFEST = {}


    def update_dblp_cache_manifest(**updates):
        current = {}
        if DBLP_MANIFEST_PATH.exists():
            try:
                current = json.loads(DBLP_MANIFEST_PATH.read_text(encoding="utf-8"))
            except Exception:
                current = {}
        current.update({
            "cache_schema_version": DBLP_CACHE_SCHEMA_VERSION,
            "identity_logic_version": DBLP_IDENTITY_LOGIC_VERSION,
            "history_logic_version": DBLP_HISTORY_LOGIC_VERSION,
            "cache_key": DBLP_CACHE_KEY,
            "xml_quick_fingerprint": DBLP_XML_QUICK_FINGERPRINT,
            "input_fingerprint": DBLP_INPUT_FINGERPRINT,
            "cache_directory": str(DBLP_CACHE_DIR),
            "updated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        })
        current.update(updates)
        atomic_json_dump(current, DBLP_MANIFEST_PATH)


    print("DBLP cache directory:", DBLP_CACHE_DIR)
    print("DBLP source fingerprint:", DBLP_XML_QUICK_FINGERPRINT["sample_sha256"][:16])
    print("DBLP input fingerprint:", DBLP_INPUT_FINGERPRINT[:16])
    print("Force rebuild cache:", DBLP_FORCE_REBUILD_CACHE)


    # ============================================================
    # PART 7. FIRST PASS
    # AUTHOR IDENTITY + PAPER CANDIDATE DISCOVERY
    # ============================================================

    pass1_cache_hit = False
    pass1_scan_performed = False

    if (not DBLP_FORCE_REBUILD_CACHE) and DBLP_PASS1_CACHE_PATH.exists():
        try:
            print("\nPass 1 cache hit — loading paper candidates and author profiles...")
            _pass1_payload = load_pickle_gzip(DBLP_PASS1_CACHE_PATH)
            if _pass1_payload.get("cache_key") != DBLP_CACHE_KEY:
                raise ValueError("Pass 1 cache key mismatch")
            profile_evidence = _pass1_payload.get("profile_evidence")
            profile_alias_rows = _pass1_payload["profile_alias_rows"]
            profile_redirects = _pass1_payload["profile_redirects"]
            flagged_profile_occurrences = _pass1_payload["flagged_profile_occurrences"]
            candidates_by_paper = _pass1_payload["candidates_by_paper"]
            pass1_records = int(_pass1_payload["pass1_records"])
            pass1_cache_hit = True
            print("Loaded Pass 1 cache; XML scan skipped.")
        except Exception as _cache_error:
            print("Pass 1 cache could not be loaded; rebuilding:", repr(_cache_error))
            DBLP_PASS1_CACHE_PATH.replace(DBLP_PASS1_CACHE_PATH.with_suffix(".invalid.gz"))
            _pass1_payload = None
    else:
        _pass1_payload = None

    if _pass1_payload is None:
        pass1_scan_performed = True
        print("\nPass 1/2: matching DBLP papers and author identities...")
        profile_evidence = defaultdict(dict)

        profile_alias_rows = set()
        profile_redirects = {}

        flagged_profile_occurrences = set()

        candidates_by_paper = defaultdict(dict)

        pass1_records = 0


        for element in tqdm(
            iter_dblp_records(),
            desc="DBLP pass 1",
            unit="records"
        ):

            pass1_records += 1

            tag = element.tag
            record_key = element.get("key")

            # --------------------------------------------------------
            # 7A. AUTHOR HOME PAGE RECORDS
            # --------------------------------------------------------

            if tag == "www":

                if not record_key:
                    continue

                title_values = child_texts(
                    element, "title"
                )

                has_crossref = bool(
                    child_texts(element, "crossref")
                )

                is_homepage = (
                    record_key.startswith("homepages/")
                    and (
                        "Home Page" in title_values
                        or has_crossref
                    )
                )

                if not is_homepage:
                    continue

                aliases = child_texts(
                    element, "author"
                )

                if not aliases:
                    continue

                crossrefs = child_texts(
                    element, "crossref"
                )

                if crossrefs:
                    profile_redirects[
                        record_key
                    ] = crossrefs[0]

                hits = []

                for alias in aliases:

                    exact = normalize_name(alias)
                    signature = name_signature(alias)

                    for occ in exact_input_lookup.get(exact, set()):

                        hits.append(
                            (occ, 0, alias)
                        )

                    if signature:

                        for occ in signature_input_lookup.get(
                            signature, set()
                        ):

                            hits.append(
                                (occ, 1, alias)
                            )

                if not hits:
                    continue

                publtype = set(
                    (element.get("publtype") or "")
                    .lower()
                    .split()
                )

                # These profiles do not represent one verified person.
                invalid_profile = bool(
                    publtype.intersection({
                        "disambiguation",
                        "group",
                        "noshow"
                    })
                )

                if invalid_profile:

                    flagged_profile_occurrences.update(
                        hit[0] for hit in hits
                    )

                    continue

                # Preserve all aliases from a relevant profile.
                for alias in aliases:
                    profile_alias_rows.add(
                        (record_key, alias)
                    )

                # Store the best evidence for each input author.
                for occ, tier, alias in hits:

                    old = profile_evidence[occ].get(
                        record_key
                    )

                    if old is None or tier < old[0]:

                        profile_evidence[occ][record_key] = (
                            tier,
                            alias
                        )

                continue

            # --------------------------------------------------------
            # 7B. PUBLICATION RECORD CANDIDATES
            # --------------------------------------------------------

            if tag not in DBLP_PUBLICATION_TAGS:
                continue

            title_values = child_texts(
                element, "title"
            )

            title = title_values[0] if title_values else ""

            title_key = normalize_title(title)

            year = get_year(element)

            authors = child_texts(
                element, "author"
            )

            links = get_record_links(element)

            link_text = " ".join(links)

            record_dois = extract_dois(link_text)
            record_arxiv_ids = extract_arxiv_ids(link_text)

            target_aids = set()

            # Exact arXiv link.
            for arxiv_id in record_arxiv_ids:

                if arxiv_id in paper_meta:
                    target_aids.add(arxiv_id)

            # Exact DOI.
            for doi in record_dois:

                target_aids.update(
                    doi_to_aids.get(doi, set())
                )

            # Exact normalized title.
            if title_key:

                target_aids.update(
                    title_to_aids.get(title_key, set())
                )

            if not target_aids:
                continue

            key = make_dblp_record_key(element, tag, title, year, authors)

            candidate = {
                "key": key,
                "type": tag,
                "title": title,
                "title_key": title_key,
                "year": year,
                "authors": authors,
                "dois": record_dois,
                "arxiv_ids": record_arxiv_ids,
                "publtype": (
                    element.get("publtype") or ""
                ).lower()
            }

            for aid in target_aids:

                candidates_by_paper[aid][key] = candidate

        _pass1_payload = {
            "cache_key": DBLP_CACHE_KEY,
            "profile_evidence": profile_evidence,
            "profile_alias_rows": profile_alias_rows,
            "profile_redirects": profile_redirects,
            "flagged_profile_occurrences": flagged_profile_occurrences,
            "candidates_by_paper": candidates_by_paper,
            "pass1_records": pass1_records,
        }
        atomic_pickle_gzip_dump(_pass1_payload, DBLP_PASS1_CACHE_PATH)
        update_dblp_cache_manifest(
            pass1_cache_file=DBLP_PASS1_CACHE_PATH.name,
            pass1_records_scanned=pass1_records,
            pass1_cache_built_at_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        )
        print("Saved Pass 1 cache:", DBLP_PASS1_CACHE_PATH)

    # profile_evidence is included in legacy Pass 1 caches but is not read by
    # any downstream stage. Remove it from RAM and compact the persistent cache
    # once, while retaining every field actually used for identity resolution.
    if isinstance(_pass1_payload, dict) and _pass1_payload.get("profile_evidence") is not None:
        # Keep the key as a None sentinel for compatibility with older readers.
        _pass1_payload["profile_evidence"] = None
        profile_evidence = None
        atomic_pickle_gzip_dump(_pass1_payload, DBLP_PASS1_CACHE_PATH)
        print("Compacted Pass 1 cache: cleared unused profile_evidence")
    else:
        profile_evidence = None
    if isinstance(_pass1_payload, dict):
        del _pass1_payload
    gc.collect()
    log_memory("after Pass 1 cache compaction")

    # ============================================================
    # PART 8. RESOLVE AUTHOR PROFILE IDENTITIES
    # ============================================================

    def canonical_person_id(person_id):

        seen = set()
        current = person_id

        while current in profile_redirects:

            if current in seen:
                return min(seen)

            seen.add(current)

            current = profile_redirects[current]

        return current


    # Reconstruct alias candidate indexes from the cached Pass 1 profile aliases.
    # This does not require another XML scan: profile_alias_rows is already in the
    # persistent Pass 1 cache.
    profile_alias_raw_index = defaultdict(set)
    profile_alias_norm_index = defaultdict(set)
    profile_alias_signature_index = defaultdict(set)

    for profile_key, alias in profile_alias_rows:
        pid = canonical_person_id(profile_key)
        raw_alias = str(alias or "").strip()
        if not raw_alias:
            continue
        profile_alias_raw_index[raw_alias].add(pid)
        profile_alias_norm_index[normalize_name(raw_alias)].add(pid)
        signature = name_signature(raw_alias)
        if signature:
            profile_alias_signature_index[signature].add(pid)


    for author in source_author_rows:

        occ = (
            author["arxiv_id"],
            author["author_position"]
        )
        source_name = str(author["author_name_arxiv_v1"] or "").strip()
        normalized_name = normalize_name(source_name)
        signature = name_signature(source_name)

        raw_exact_candidates = set(
            profile_alias_raw_index.get(source_name, set())
        )
        normalized_candidates = set(
            profile_alias_norm_index.get(normalized_name, set())
        )
        signature_candidates = (
            set(profile_alias_signature_index.get(signature, set()))
            if signature else set()
        )

        # Priority: raw exact alias > normalized full name > first-initial + surname.
        # A weaker-tier collision cannot invalidate a unique stronger-tier match.
        if raw_exact_candidates:
            considered = raw_exact_candidates
            selected_method = "RAW_EXACT_ALIAS"
            selected_tier = 0
        elif normalized_candidates:
            considered = normalized_candidates
            selected_method = "NORMALIZED_FULL_NAME_ALIAS"
            selected_tier = 0
        else:
            considered = signature_candidates
            selected_method = "UNIQUE_INITIAL_SURNAME_ALIAS"
            selected_tier = 1

        all_candidate_ids = sorted(
            raw_exact_candidates | normalized_candidates | signature_candidates
        )
        selected_candidate_ids = set(considered)
        weaker_candidate_ids = sorted(
            set(all_candidate_ids) - selected_candidate_ids
        )
        author["profile_all_candidate_person_ids"] = all_candidate_ids
        author["profile_candidate_person_ids"] = sorted(considered)
        author["profile_weaker_candidate_person_ids"] = weaker_candidate_ids
        author["profile_priority_override"] = bool(weaker_candidate_ids)
        author["profile_candidate_count"] = len(considered)
        author.setdefault("identity_conflict_status", None)

        if len(considered) == 1:
            author["profile_person_id"] = next(iter(considered))
            author["profile_match_status"] = "MATCHED_DBLP_PROFILE"
            author["profile_match_method"] = selected_method

        elif len(considered) > 1:
            author["profile_match_status"] = "AMBIGUOUS_DBLP_PERSON_MATCH"
            if selected_method == "RAW_EXACT_ALIAS":
                conflict_status = "AMBIGUOUS_RAW_EXACT_ALIAS_COLLISION"
            elif selected_method == "NORMALIZED_FULL_NAME_ALIAS":
                conflict_status = "AMBIGUOUS_NORMALIZED_FULL_NAME_COLLISION"
            else:
                conflict_status = "AMBIGUOUS_INITIAL_SURNAME_ALIAS_COLLISION"
            author["identity_conflict_status"] = conflict_status

        elif occ in flagged_profile_occurrences:
            author["profile_match_status"] = (
                "DBLP_GROUP_OR_DISAMBIGUATION_PROFILE"
            )

        else:
            author["profile_match_status"] = "NO_DBLP_PROFILE_MATCH"


    # Build alias-to-person mapping only for relevant profiles.
    alias_to_person_raw = defaultdict(set)
    alias_to_person_norm = defaultdict(set)

    resolved_profile_ids = {
        a["profile_person_id"]
        for a in source_author_rows
        if a["profile_person_id"]
    }

    # The history map is restricted to source profiles that were resolved
    # unambiguously. This prevents ambiguous aliases from contaminating counts.
    for key, alias in profile_alias_rows:

        pid = canonical_person_id(key)
        # Keep competing profiles in this map even if one of them was not
        # selected for a source author. Otherwise a conflicted alias could look
        # falsely unique during historical-record counting.
        alias_to_person_raw[alias].add(pid)
        alias_to_person_norm[normalize_name(alias)].add(pid)

    # A separate map is used to independently resolve author names on the
    # selected target publication. It includes all relevant candidate profiles
    # so source-profile and target-publication evidence can be compared.
    candidate_alias_raw = defaultdict(set)
    candidate_alias_norm = defaultdict(set)
    candidate_alias_signature = defaultdict(set)

    for key, alias in profile_alias_rows:
        pid = canonical_person_id(key)
        candidate_alias_raw[alias].add(pid)
        candidate_alias_norm[normalize_name(alias)].add(pid)
        signature = name_signature(alias)
        if signature:
            candidate_alias_signature[signature].add(pid)


    def resolve_alias_person_details(name):
        """Return person resolution plus all weaker-tier alternatives.

        Evidence priority:
          1. Unique raw exact alias.
          2. Unique normalized full-name alias.
          3. Unique first-initial + surname signature.

        A weaker-tier collision does not invalidate a unique stronger match, but
        it is retained as an ambiguity warning and downgrades confidence to
        MEDIUM. A collision within the strongest available tier remains unresolved.
        """
        raw = str(name or "").strip()
        exact_ids = set(candidate_alias_raw.get(raw, set()))
        normalized_ids = set(candidate_alias_norm.get(normalize_name(raw), set()))
        signature = name_signature(raw)
        signature_ids = (
            set(candidate_alias_signature.get(signature, set()))
            if signature else set()
        )

        all_ids = exact_ids | normalized_ids | signature_ids

        if exact_ids:
            selected_ids = exact_ids
            method = "RAW_EXACT_ALIAS"
        elif normalized_ids:
            selected_ids = normalized_ids
            method = "NORMALIZED_FULL_NAME_ALIAS"
        else:
            selected_ids = signature_ids
            method = "UNIQUE_INITIAL_SURNAME_ALIAS"

        person_id = next(iter(selected_ids)) if len(selected_ids) == 1 else None
        weaker_ids = sorted(all_ids - selected_ids)

        return {
            "person_id": person_id,
            "match_method": method,
            "selected_person_ids": sorted(selected_ids),
            "all_person_ids": sorted(all_ids),
            "weaker_person_ids": weaker_ids,
        }


    def resolve_alias_person(name):
        return resolve_alias_person_details(name)["person_id"]


    # Deterministic unit checks for the V3.1.1 alias-priority rule.
    _live_alias_maps = (
        candidate_alias_raw,
        candidate_alias_norm,
        candidate_alias_signature,
    )
    try:
        # 1. Raw exact alias wins over a weaker signature collision.
        candidate_alias_raw = {"Alex Kim": {"P1"}}
        candidate_alias_norm = {normalize_name("Alex Kim"): {"P1"}}
        candidate_alias_signature = {name_signature("Alex Kim"): {"P1", "P2"}}
        assert resolve_alias_person("Alex Kim") == "P1"
        assert resolve_alias_person_details("Alex Kim")["weaker_person_ids"] == ["P2"]

        # 2. Unique normalized full name wins over a weaker signature collision.
        candidate_alias_raw = {}
        candidate_alias_norm = {normalize_name("Alex Kim"): {"P1"}}
        candidate_alias_signature = {name_signature("Alex Kim"): {"P1", "P2"}}
        assert resolve_alias_person("Alex Kim") == "P1"
        assert resolve_alias_person_details("Alex Kim")["weaker_person_ids"] == ["P2"]

        # 3. Raw exact alias collision remains unresolved.
        candidate_alias_raw = {"Alex Kim": {"P1", "P2"}}
        candidate_alias_norm = {normalize_name("Alex Kim"): {"P1", "P2"}}
        candidate_alias_signature = {name_signature("Alex Kim"): {"P1", "P2"}}
        assert resolve_alias_person("Alex Kim") is None

        # 4. Normalized full-name collision remains unresolved.
        candidate_alias_raw = {}
        candidate_alias_norm = {normalize_name("Alex Kim"): {"P1", "P2"}}
        candidate_alias_signature = {name_signature("Alex Kim"): {"P1", "P2"}}
        assert resolve_alias_person("Alex Kim") is None

        # 5. Unique signature can be used only as a fallback.
        candidate_alias_raw = {}
        candidate_alias_norm = {}
        candidate_alias_signature = {name_signature("Alex Kim"): {"P3"}}
        assert resolve_alias_person("Alex Kim") == "P3"

        # 6. Ambiguous signature fallback remains unresolved.
        candidate_alias_signature = {name_signature("Alex Kim"): {"P3", "P4"}}
        assert resolve_alias_person("Alex Kim") is None
    finally:
        (candidate_alias_raw,
         candidate_alias_norm,
         candidate_alias_signature) = _live_alias_maps

    print("V3.1.1 identity-priority self-tests: 6/6 passed.")


    def candidate_author_signature(candidate):

        signature = []

        for name in candidate["authors"]:

            pid = resolve_alias_person(name)

            if pid:
                signature.append("P:" + pid)
            else:
                signature.append(
                    "R:" + str(name).strip()
                )

        return tuple(sorted(signature))


    # ============================================================
    # PART 9. AUTHOR LIST COMPARISON
    # ============================================================

    def match_author_lists(source_authors, candidate_names):

        candidate_info = []

        for name in candidate_names:

            alias_resolution = resolve_alias_person_details(name)
            candidate_info.append({
                "name": name,
                "person_id": alias_resolution["person_id"],
                "person_id_match_method": alias_resolution["match_method"],
                "person_id_selected_candidates": alias_resolution["selected_person_ids"],
                "person_id_all_candidates": alias_resolution["all_person_ids"],
                "person_id_weaker_candidates": alias_resolution["weaker_person_ids"],
                "name_key": normalize_name(name),
                "signature": name_signature(name)
            })

        pairs = []
        identity_conflicts = []
        used_source = set()
        used_candidate = set()

        def unique_phase(source_key_fn, candidate_key_fn):

            source_groups = defaultdict(list)
            candidate_groups = defaultdict(list)

            for i, author in enumerate(source_authors):

                if i in used_source:
                    continue

                key = source_key_fn(author)

                if key is not None:
                    source_groups[key].append(i)

            for j, author in enumerate(candidate_info):

                if j in used_candidate:
                    continue

                key = candidate_key_fn(author)

                if key is not None:
                    candidate_groups[key].append(j)

            for key in set(source_groups).intersection(
                candidate_groups
            ):

                si = source_groups[key]
                ci = candidate_groups[key]

                # Only unique one-to-one evidence is accepted.
                if len(si) != 1 or len(ci) != 1:
                    continue

                i, j = si[0], ci[0]

                spid = source_authors[i].get(
                    "profile_person_id"
                )

                cpid = candidate_info[j].get(
                    "person_id"
                )

                if spid and cpid and spid != cpid:
                    identity_conflicts.append({
                        "source_index": i,
                        "candidate_index": j,
                        "source_person_id": spid,
                        "candidate_person_id": cpid,
                        "source_name": source_authors[i]["author_name_arxiv_v1"],
                        "candidate_name": candidate_info[j]["name"],
                        "match_evidence": "NORMALIZED_NAME_WITH_PERSON_ID_MISMATCH",
                    })
                    used_source.add(i)
                    used_candidate.add(j)
                    continue

                pairs.append((i, j))

                used_source.add(i)
                used_candidate.add(j)

        # Highest priority: persistent person ID.
        unique_phase(
            lambda x: x.get("profile_person_id"),
            lambda x: x.get("person_id")
        )

        # Second priority: normalized full name.
        unique_phase(
            lambda x: normalize_name(
                x["author_name_arxiv_v1"]
            ),
            lambda x: x["name_key"]
        )

        # Third priority: unique first-initial + surname.
        unique_phase(
            lambda x: name_signature(
                x["author_name_arxiv_v1"]
            ),
            lambda x: x["signature"]
        )

        return pairs, candidate_info, identity_conflicts

    def make_unmatched_author_rows(source_authors):
        return [
            {
                **author,
                "dblp_author_matched_to_paper": False,
                "dblp_record_key": None
            }
            for author in source_authors
        ]

    # ============================================================
    # PART 10. CANDIDATE GROUPING AND PAPER MATCHING
    # ============================================================

    def candidate_preference(candidate):

        type_rank = {
            "inproceedings": 6,
            "article": 5,
            "incollection": 4,
            "book": 3,
            "phdthesis": 2,
            "mastersthesis": 1
        }

        flags = set(
            candidate["publtype"].split()
        )

        return (
            int("withdrawn" not in flags),
            int("informal" not in flags),
            type_rank.get(candidate["type"], 0),
            int(bool(candidate["dois"])),
            int(candidate["year"] is not None)
        )


    paper_results = {}
    candidate_audit_rows = []

    for aid in tqdm(
        list(paper_meta),
        desc="DBLP paper matching"
    ):

        meta = paper_meta[aid]

        src_authors = source_author_by_paper[aid]

        candidate_dict = candidates_by_paper.get(
            aid, {}
        )

        candidate_metrics = []

        for candidate in candidate_dict.values():

            pairs, candidate_info, identity_conflicts = match_author_lists(
                src_authors,
                candidate["authors"]
            )

            source_n = len(src_authors)
            candidate_n = len(candidate["authors"])

            overlap = len(pairs)

            source_coverage = (
                overlap / source_n
                if source_n else 0
            )

            candidate_coverage = (
                overlap / candidate_n
                if candidate_n else 0
            )

            sources = []

            if aid in candidate["arxiv_ids"]:
                sources.append("ARXIV_ID")

            if meta["dois"].intersection(
                candidate["dois"]
            ):
                sources.append("DOI")

            if candidate["title_key"] == meta["title_key"]:
                sources.append("TITLE")

            identifier_rank = (
                2 if "ARXIV_ID" in sources
                else 1 if "DOI" in sources
                else 0
            )

            score = (
                identifier_rank,
                overlap,
                round(source_coverage, 6),
                round(candidate_coverage, 6)
            )

            signature = candidate_author_signature(
                candidate
            )

            group_id = hashlib.sha1(
                repr(signature).encode("utf-8")
            ).hexdigest()[:12]

            candidate_metrics.append({
                "candidate": candidate,
                "pairs": pairs,
                "candidate_info": candidate_info,
                "identity_conflicts": identity_conflicts,
                "score": score,
                "identifier_rank": identifier_rank,
                "sources": sources,
                "overlap": overlap,
                "source_coverage": source_coverage,
                "candidate_coverage": candidate_coverage,
                "group_id": group_id,
                "signature": signature
            })

        # No candidate records.
        if not candidate_metrics:

            paper_results[aid] = {
                "status": (
                    "PAPER_NOT_IN_DBLP_OR_NO_VERIFIED_MATCH"
                ),
                "method": None,
                "selected": None,
                "author_rows": make_unmatched_author_rows(src_authors),
                "candidate_count": 0,
                "candidate_group_size": 0,
                "author_match_count": 0,
                "initial_overlap": 0
            }

            continue

        # --------------------------------------------------------
        # Group records with identical author identities.
        # This handles preprint/conference duplicate records.
        # --------------------------------------------------------

        groups = defaultdict(list)

        for metric in candidate_metrics:

            groups[
                metric["signature"]
            ].append(metric)

        group_scores = {}

        for signature, items in groups.items():

            group_scores[signature] = max(
                item["score"]
                for item in items
            )

        best_score = max(
            group_scores.values()
        )

        best_groups = [
            signature
            for signature, score in group_scores.items()
            if score == best_score
        ]

        # Different author sets with identical best evidence:
        # do not force an arbitrary choice.
        if len(best_groups) > 1:

            for metric in candidate_metrics:

                candidate_audit_rows.append({
                    "arxiv_id": aid,
                    "dblp_record_key": metric["candidate"]["key"],
                    "candidate_group_id": metric["group_id"],
                    "match_sources": ";".join(metric["sources"]),
                    "author_overlap": metric["overlap"],
                    "source_author_coverage": metric["source_coverage"],
                    "candidate_author_coverage": metric["candidate_coverage"],
                    "selected": False,
                    "candidate_status": "AMBIGUOUS_GROUP"
                })

            paper_results[aid] = {
                "status": "AMBIGUOUS_DBLP_PAPER_MATCH",
                "method": None,
                "selected": None,
                "author_rows": make_unmatched_author_rows(src_authors),
                "candidate_count": len(candidate_metrics),
                "candidate_group_size": 0,
                "author_match_count": 0,
                "initial_overlap": 0
            }

            continue

        selected_signature = best_groups[0]

        # Reject title-only matches with insufficient author overlap.
        if (
            best_score[0] == 0
            and (
                best_score[1] == 0
                or best_score[2] < 0.5
                or best_score[3] < 0.5
            )
        ):
            for metric in candidate_metrics:
                candidate_audit_rows.append({
                    "arxiv_id": aid,
                    "dblp_record_key": metric["candidate"]["key"],
                    "candidate_group_id": metric["group_id"],
                    "match_sources": ";".join(metric["sources"]),
                    "author_overlap": metric["overlap"],
                    "source_author_coverage": metric["source_coverage"],
                    "candidate_author_coverage": metric["candidate_coverage"],
                    "selected": False,
                    "candidate_status": "LOW_AUTHOR_OVERLAP"
                })

            paper_results[aid] = {
                "status": "PAPER_NOT_IN_DBLP_OR_NO_VERIFIED_MATCH",
                "method": None,
                "selected": None,
                "author_rows": make_unmatched_author_rows(src_authors),
                "candidate_count": len(candidate_metrics),
                "candidate_group_size": 0,
                "author_match_count": 0,
                "initial_overlap": best_score[1]
            }

            continue

        selected_group = groups[
            selected_signature
        ]

        # Choose the best formal record within the group.
        selected_metric = max(
            selected_group,
            key=lambda x: (
                x["score"],
                candidate_preference(x["candidate"])
            )
        )

        selected = selected_metric["candidate"]

        pairs = selected_metric["pairs"]

        candidate_info = selected_metric[
            "candidate_info"
        ]

        pair_by_source = {
            si: ci for si, ci in pairs
        }

        # --------------------------------------------------------
        # Record source-vs-target persistent ID conflicts.
        # Conflicted authors remain unresolved and are excluded from history.
        # --------------------------------------------------------

        for conflict in selected_metric.get("identity_conflicts", []):
            source_author = src_authors[conflict["source_index"]]
            source_author["identity_id"] = None
            source_author["identity_method"] = "SOURCE_TARGET_PROFILE_ID_CONFLICT"
            source_author["identity_confidence"] = "UNRESOLVED"
            source_author["identity_conflict_status"] = "SOURCE_TARGET_PERSON_ID_MISMATCH"
            source_author["identity_conflict_source_person_id"] = conflict["source_person_id"]
            source_author["identity_conflict_target_person_id"] = conflict["candidate_person_id"]
            source_author["identity_conflict_target_author_name"] = conflict["candidate_name"]

        # If a source author has several plausible profile IDs, a target-paper
        # profile outside that candidate set is an explicit identity conflict.
        for si, ci in pairs:
            source_author = src_authors[si]
            candidate_author = candidate_info[ci]
            candidate_pid = candidate_author.get("person_id")
            source_candidates = set(source_author.get("profile_candidate_person_ids") or [])
            if (
                source_author.get("profile_match_status") == "AMBIGUOUS_DBLP_PERSON_MATCH"
                and candidate_pid
                and source_candidates
                and candidate_pid not in source_candidates
            ):
                source_author["identity_conflict_status"] = "TARGET_PROFILE_OUTSIDE_SOURCE_CANDIDATES"
                source_author["identity_conflict_target_person_id"] = candidate_pid
                source_author["identity_conflict_target_author_name"] = candidate_author["name"]

        # --------------------------------------------------------
        # Assign verified identities to matched source authors.
        # --------------------------------------------------------

        for si, ci in pairs:

            source_author = src_authors[si]

            candidate_author = candidate_info[ci]

            source_pid = source_author[
                "profile_person_id"
            ]

            candidate_pid = candidate_author[
                "person_id"
            ]

            source_status = source_author[
                "profile_match_status"
            ]

            # If the target publication's author alias itself has multiple IDs at
            # its strongest evidence tier, do not inherit the source profile ID.
            # That would silently resolve an ambiguous target-side identity.
            if (
                source_pid
                and candidate_pid is None
                and len(candidate_author.get("person_id_selected_candidates") or []) > 1
            ):
                source_author["identity_id"] = None
                source_author["identity_method"] = "TARGET_PUBLICATION_ALIAS_AMBIGUOUS"
                source_author["identity_confidence"] = "UNRESOLVED"
                source_author["identity_conflict_status"] = "TARGET_PUBLICATION_ALIAS_AMBIGUOUS"
                source_author["identity_conflict_source_person_id"] = source_pid
                source_author["identity_conflict_target_person_id"] = ";".join(
                    candidate_author.get("person_id_selected_candidates") or []
                )
                source_author["identity_conflict_target_author_name"] = candidate_author["name"]
                continue

            if source_pid and candidate_pid and source_pid != candidate_pid:
                source_author["identity_id"] = None
                source_author["identity_method"] = "SOURCE_TARGET_PROFILE_ID_CONFLICT"
                source_author["identity_confidence"] = "UNRESOLVED"
                source_author["identity_conflict_status"] = "SOURCE_TARGET_PERSON_ID_MISMATCH"
                source_author["identity_conflict_source_person_id"] = source_pid
                source_author["identity_conflict_target_person_id"] = candidate_pid
                source_author["identity_conflict_target_author_name"] = candidate_author["name"]
                continue

            if source_pid:

                source_author["identity_id"] = source_pid

                source_author["identity_method"] = (
                    source_author["profile_match_method"]
                )

                # If the selected publication uses an alias that
                # is absent from the stored profile aliases,
                # the paper-level identity match supplies evidence.
                if (
                    candidate_pid is None
                    and normalize_name(
                        source_author["author_name_arxiv_v1"]
                    ) == candidate_author["name_key"]
                ):

                    alias_to_person_raw[
                        candidate_author["name"]
                    ].add(source_pid)

            elif source_status == "NO_DBLP_PROFILE_MATCH":

                # Contextual profile matching is allowed only
                # when normalized full names agree.
                if (
                    candidate_pid
                    and normalize_name(
                        source_author["author_name_arxiv_v1"]
                    ) == candidate_author["name_key"]
                ):

                    source_author["identity_id"] = candidate_pid

                    source_author["identity_method"] = (
                        "TARGET_PAPER_CONTEXT_PROFILE_ALIAS"
                    )

                # DBLP documentation states that the exact
                # case-sensitive author string is an author entity.
                elif (
                    source_author["author_name_arxiv_v1"]
                    == candidate_author["name"]
                    and not candidate_pid
                ):

                    source_author["identity_id"] = (
                        "DBLP_ALIAS::" + candidate_author["name"]
                    )

                    source_author["identity_method"] = (
                        "DBLP_EXACT_AUTHOR_STRING"
                    )

                    source_author["identity_alias_raw"] = (
                        candidate_author["name"]
                    )

            # Keep confidence separate from the binary identity match.
            # Exact DBLP profile aliases are strongest; initial+surname is
            # medium confidence; raw exact-name fallback remains low confidence.
            if source_author.get("identity_id"):
                source_author["identity_confidence"] = (
                    classify_identity_confidence(
                        source_author.get("identity_method")
                    )
                )

                # If a stronger full-name alias was selected while other weaker
                # signature candidates also exist, retain the selected identity
                # but downgrade it to MEDIUM. This protects the strict HIGH-only
                # sample from possible homonym/alias collisions.
                weaker_ids = set(
                    source_author.get("profile_weaker_candidate_person_ids") or []
                )
                weaker_ids.update(
                    candidate_author.get("person_id_weaker_candidates") or []
                )
                if weaker_ids:
                    source_author["identity_confidence"] = "MEDIUM"
                    source_author["identity_priority_override"] = True
                    source_author["identity_priority_alternative_person_ids"] = sorted(weaker_ids)
                else:
                    source_author["identity_priority_override"] = False
                    source_author["identity_priority_alternative_person_ids"] = []

            # Ambiguous profile matches are never force-resolved.

        effective_pairs = [
            (si, ci)
            for si, ci in pairs
            if src_authors[si].get("identity_id")
        ]

        matched_source_indices = {
            si for si, _ in effective_pairs
        }

        matched_ids = [
            src_authors[si]["identity_id"]
            for si, _ in effective_pairs
        ]

        identities_unique = (
            len(matched_ids) == len(set(matched_ids))
        )

        # The feature target is the author team recorded on arXiv v1.
        # A DBLP record may contain additional names (e.g. a later journal
        # version); this does not make the original arXiv author set incomplete.
        # Completeness therefore requires every arXiv v1 author to be matched
        # and every resolved identity to be unique.
        full_match = (
            len(src_authors) > 0
            and len(effective_pairs) == len(src_authors)
            and identities_unique
        )

        if full_match:

            status = "MATCHED_FULL_AUTHOR_SET"

        elif effective_pairs:

            status = "MATCHED_PARTIAL_AUTHOR_SET"

        elif pairs or selected_metric["identifier_rank"] > 0:

            status = (
                "MATCHED_PAPER_AUTHOR_IDENTITY_INCOMPLETE"
            )

        else:

            status = (
                "PAPER_NOT_IN_DBLP_OR_NO_VERIFIED_MATCH"
            )

        if "ARXIV_ID" in selected_metric["sources"]:
            method = "arxiv_id_link_verified"

        elif "DOI" in selected_metric["sources"]:
            method = "doi_exact"

        else:
            method = "title_exact_author_confirmed"

        author_rows = []

        for si, source_author in enumerate(src_authors):

            is_matched = (
                si in matched_source_indices
            )

            author_rows.append({
                **source_author,
                "dblp_author_matched_to_paper": is_matched,
                "dblp_record_key": selected["key"]
            })

        for metric in candidate_metrics:

            candidate_audit_rows.append({
                "arxiv_id": aid,
                "dblp_record_key": metric["candidate"]["key"],
                "candidate_group_id": metric["group_id"],
                "match_sources": ";".join(metric["sources"]),
                "author_overlap": metric["overlap"],
                "source_author_coverage": metric["source_coverage"],
                "candidate_author_coverage": metric["candidate_coverage"],
                "selected": (
                    metric["candidate"]["key"] == selected["key"]
                ),
                "candidate_status": (
                    "SELECTED"
                    if metric["candidate"]["key"] == selected["key"]
                    else "ALTERNATIVE"
                )
            })

        paper_results[aid] = {
            "status": status,
            "method": method,
            "selected": selected,
            "author_rows": author_rows,
            "candidate_count": len(candidate_metrics),
            "candidate_group_size": len(selected_group),
            "author_match_count": len(effective_pairs),
            "initial_overlap": len(pairs)
        }


    log_memory("after paper candidate matching")

    # ============================================================
    # PART 11. PREPARE RELEVANT AUTHOR HISTORY INDEX
    # ============================================================

    matched_person_ids = set()

    alias_only_rows = []

    for aid, result in paper_results.items():

        for author in result["author_rows"]:

            if not author.get("dblp_author_matched_to_paper"):
                continue

            pid = author.get("identity_id")

            if not pid:
                continue

            matched_person_ids.add(pid)

            if pid.startswith("DBLP_ALIAS::"):

                alias_only_rows.append({
                    "alias": author["identity_alias_raw"],
                    "person_id": pid
                })


    history_alias_to_ids = defaultdict(set)

    # Historical DBLP author elements are matched against their raw alias text.
    # A raw alias is therefore usable when it maps to exactly one canonical
    # person ID. A different full name sharing only the same first-initial +
    # surname signature must not invalidate this unique raw alias. Only an actual
    # raw-alias collision across multiple person IDs is excluded.
    ambiguous_profile_aliases = set()

    for alias, ids in alias_to_person_raw.items():
        if len(ids) > 1:
            ambiguous_profile_aliases.add(alias)
            continue

        pid = next(iter(ids))
        if pid in matched_person_ids:
            history_alias_to_ids[alias].add(pid)

    # Exact author-string fallback is allowed only when no profile alias evidence
    # exists for that raw string. This prevents a fallback from bypassing a
    # conflicting or unresolved DBLP profile.
    for row in alias_only_rows:
        alias = row["alias"]
        if alias not in alias_to_person_raw and alias not in ambiguous_profile_aliases:
            history_alias_to_ids[alias].add(row["person_id"])

    # Only unique aliases are usable in historical counting.
    history_alias_to_person = {
        alias: next(iter(ids))
        for alias, ids in history_alias_to_ids.items()
        if len(ids) == 1
    }

    ambiguous_alias_count = (
        len(ambiguous_profile_aliases)
        + sum(1 for ids in history_alias_to_ids.values() if len(ids) > 1)
    )

    print(
        "\nRelevant author identities:",
        len(matched_person_ids)
    )

    print(
        "Relevant author aliases:",
        len(history_alias_to_person)
    )

    print(
        "Ambiguous aliases:",
        ambiguous_alias_count
    )


    # ============================================================
    # PART 12. SECOND PASS
    # COUNT DISTINCT HISTORICAL WORKS (CONSERVATIVE DEDUPLICATION)
    # ============================================================

    # Cache the history index against the resolved identity map. If identity
    # resolution changes, only Pass 2 is invalidated; Pass 1 remains reusable.
    identity_fingerprint_payload = {
        "identity_logic_version": DBLP_IDENTITY_LOGIC_VERSION,
        "history_logic_version": DBLP_HISTORY_LOGIC_VERSION,
        "matched_person_ids": sorted(matched_person_ids),
        "history_alias_to_person": sorted(history_alias_to_person.items()),
    }
    DBLP_IDENTITY_FINGERPRINT = sha256_json(identity_fingerprint_payload)
    DBLP_PASS2_CACHE_PATH = DBLP_CACHE_DIR / (
        f"pass2_{DBLP_CACHE_KEY}_{DBLP_IDENTITY_FINGERPRINT}.pkl.gz"
    )

    pass2_cache_hit = False
    pass2_scan_performed = False
    _pass2_payload = None
    if (not DBLP_FORCE_REBUILD_CACHE) and DBLP_PASS2_CACHE_PATH.exists():
        try:
            print("\nPass 2 cache hit — loading historical author index...")
            _pass2_payload = load_pickle_gzip(DBLP_PASS2_CACHE_PATH)
            if (
                _pass2_payload.get("cache_key") != DBLP_CACHE_KEY
                or _pass2_payload.get("identity_fingerprint") != DBLP_IDENTITY_FINGERPRINT
            ):
                raise ValueError("Pass 2 cache fingerprint mismatch")
            author_year_counts = _pass2_payload["author_year_counts"]
            unique_historical_work_count = int(_pass2_payload["unique_historical_work_count"])
            duplicate_history_records_collapsed = int(_pass2_payload["duplicate_history_records_collapsed"])
            ambiguous_history_fallback_key_count = int(_pass2_payload["ambiguous_history_fallback_key_count"])
            relevant_history_records = int(_pass2_payload["relevant_history_records"])
            work_author_years = _pass2_payload["work_author_years"]
            record_key_to_work = _pass2_payload["record_key_to_work"]
            pass2_cache_hit = True
            print("Loaded Pass 2 cache; XML scan skipped.")
        except Exception as _cache_error:
            print("Pass 2 cache could not be loaded; rebuilding:", repr(_cache_error))
            DBLP_PASS2_CACHE_PATH.replace(DBLP_PASS2_CACHE_PATH.with_suffix(".invalid.gz"))
            _pass2_payload = None

    if _pass2_payload is None:
        pass2_scan_performed = True
        print(
            "\nPass 2/2: counting deduplicated historical publications..."
        )


        class HistoricalWorkDeduplicator:
            """Streaming union-find for duplicate DBLP records.

            Strong identifiers (DOI and arXiv ID) are used to join versions.
            A normalized title + complete normalized author list is a fallback
            for records without strong identifiers. An identifier-free record can
            attach to one unambiguous existing fallback group, but two conflicting
            strong-ID groups are never merged solely because title/author text agrees.
            """

            def __init__(self):
                self.parent = {}
                self.rank = {}
                self.has_strong_id = {}
                self.work_author_years = defaultdict(dict)
                self.strong_key_to_work = {}
                self.fallback_key_to_works = defaultdict(set)
                self.record_key_to_work = {}
                self.next_work_id = 0
                self.relevant_records = 0

            def _new_work(self):
                work_id = self.next_work_id
                self.next_work_id += 1
                self.parent[work_id] = work_id
                self.rank[work_id] = 0
                self.has_strong_id[work_id] = False
                return work_id

            def find(self, work_id):
                parent = self.parent[work_id]
                if parent != work_id:
                    self.parent[work_id] = self.find(parent)
                return self.parent[work_id]

            def union(self, left, right):
                left = self.find(left)
                right = self.find(right)
                if left == right:
                    return left

                if self.rank[left] < self.rank[right]:
                    left, right = right, left

                self.parent[right] = left
                if self.rank[left] == self.rank[right]:
                    self.rank[left] += 1

                self.has_strong_id[left] = (
                    self.has_strong_id[left] or self.has_strong_id[right]
                )

                # Merge the author-specific earliest year for this work.
                for person_id, year in self.work_author_years.pop(right, {}).items():
                    old_year = self.work_author_years[left].get(person_id)
                    self.work_author_years[left][person_id] = (
                        year if old_year is None else min(old_year, year)
                    )

                return left

            def add_record(self, strong_keys, fallback_key, person_ids, year, record_key=None):
                strong_keys = sorted(set(strong_keys))

                strong_roots = {
                    self.find(self.strong_key_to_work[key])
                    for key in strong_keys
                    if key in self.strong_key_to_work
                }

                fallback_roots = set()
                if fallback_key:
                    fallback_roots = {
                        self.find(work_id)
                        for work_id in self.fallback_key_to_works.get(
                            fallback_key, set()
                        )
                    }

                if strong_roots:
                    root = next(iter(strong_roots))
                    for other in strong_roots:
                        root = self.union(root, other)

                    # A single identifier-free fallback group can be linked to a
                    # newly observed strong-ID record. Do not merge conflicting
                    # groups that already have their own DOI/arXiv identifiers.
                    if len(fallback_roots) == 1:
                        fallback_root = next(iter(fallback_roots))
                        root = self.find(root)
                        fallback_root = self.find(fallback_root)
                        if (
                            fallback_root != root
                            and not self.has_strong_id[fallback_root]
                        ):
                            root = self.union(root, fallback_root)

                elif strong_keys:
                    # No strong identifier has been seen before. Reuse a single
                    # identifier-free title/author group if one exists; otherwise
                    # start a separate work to avoid merging conflicting DOI records.
                    if len(fallback_roots) == 1:
                        fallback_root = next(iter(fallback_roots))
                        if not self.has_strong_id[fallback_root]:
                            root = fallback_root
                        else:
                            root = self._new_work()
                    else:
                        root = self._new_work()

                else:
                    # Without DOI/arXiv IDs, only an unambiguous exact title+author
                    # fingerprint is accepted as a duplicate signal.
                    if len(fallback_roots) == 1:
                        root = next(iter(fallback_roots))
                    else:
                        root = self._new_work()

                root = self.find(root)
                if strong_keys:
                    self.has_strong_id[root] = True
                    for key in strong_keys:
                        self.strong_key_to_work[key] = root

                if fallback_key:
                    self.fallback_key_to_works[fallback_key].add(root)

                # One author contributes at most once to a work. If versions have
                # different years, retain that author's earliest observed version.
                for person_id in set(person_ids):
                    old_year = self.work_author_years[root].get(person_id)
                    self.work_author_years[root][person_id] = (
                        year if old_year is None else min(old_year, year)
                    )

                if record_key:
                    self.record_key_to_work[record_key] = root

                self.relevant_records += 1
                return root

            def finalize(self):
                author_year_counts = defaultdict(Counter)
                unique_work_count = 0

                for work_id in list(self.parent):
                    root = self.find(work_id)
                    if root != work_id:
                        continue

                    author_years = self.work_author_years.get(root, {})
                    if not author_years:
                        continue

                    unique_work_count += 1
                    for person_id, year in author_years.items():
                        author_year_counts[person_id][year] += 1

                ambiguous_fallback_key_count = 0
                for work_ids in self.fallback_key_to_works.values():
                    roots = {self.find(work_id) for work_id in work_ids}
                    if len(roots) > 1:
                        ambiguous_fallback_key_count += 1

                duplicate_records_collapsed = max(
                    0, self.relevant_records - unique_work_count
                )

                final_work_author_years = {}
                for work_id in list(self.parent):
                    root = self.find(work_id)
                    if root == work_id and self.work_author_years.get(root):
                        final_work_author_years[root] = dict(self.work_author_years[root])

                final_record_key_to_work = {
                    key: self.find(work_id)
                    for key, work_id in self.record_key_to_work.items()
                }

                return (
                    author_year_counts,
                    unique_work_count,
                    duplicate_records_collapsed,
                    ambiguous_fallback_key_count,
                    final_work_author_years,
                    final_record_key_to_work,
                )


        def make_history_fallback_key(title, authors):
            """Exact normalized title + complete author-list fingerprint."""
            title_key = normalize_title(title)
            normalized_authors = [normalize_name(name) for name in authors]

            if not title_key or not normalized_authors:
                return None
            if any(not name for name in normalized_authors):
                return None

            payload = json.dumps(
                {
                    "title": title_key,
                    "authors": sorted(normalized_authors),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
            return "title_author:" + digest


        history_deduplicator = HistoricalWorkDeduplicator()

        EXCLUDED_PUBLICATION_TYPES = {
            "withdrawn",
            "edited",
            "data",
            "software",
            "encyclopedia"
        }

        for element in tqdm(
            iter_dblp_records(),
            desc="DBLP pass 2",
            unit="records"
        ):

            if element.tag not in DBLP_PUBLICATION_TAGS:
                continue

            publtype = set(
                (element.get("publtype") or "")
                .lower()
                .split()
            )

            if publtype.intersection(
                EXCLUDED_PUBLICATION_TYPES
            ):
                continue

            year = get_year(element)
            if year is None:
                continue

            authors = child_texts(element, "author")
            if not authors:
                continue

            record_person_ids = set()
            for alias in authors:
                person_id = history_alias_to_person.get(alias)
                if person_id and person_id in matched_person_ids:
                    record_person_ids.add(person_id)

            if not record_person_ids:
                continue

            # Only structured electronic links/eprint fields are treated as strong
            # work identifiers. Free-form notes are deliberately excluded because
            # they may mention unrelated DOI strings.
            identifier_values = []
            for identifier_tag in ("ee", "url", "eprint"):
                identifier_values.extend(child_texts(element, identifier_tag))
            link_text = " ".join(identifier_values)
            record_dois = extract_dois(link_text)
            record_arxiv_ids = extract_arxiv_ids(link_text)

            strong_keys = (
                {"doi:" + doi for doi in record_dois}
                | {"arxiv:" + arxiv_id for arxiv_id in record_arxiv_ids}
            )

            title_values = child_texts(element, "title")
            title = title_values[0] if title_values else ""
            fallback_key = make_history_fallback_key(title, authors)

            record_key = make_dblp_record_key(
                element, element.tag, title, year, authors
            )

            history_deduplicator.add_record(
                strong_keys=strong_keys,
                fallback_key=fallback_key,
                person_ids=record_person_ids,
                year=year,
                record_key=record_key,
            )

        (
            author_year_counts,
            unique_historical_work_count,
            duplicate_history_records_collapsed,
            ambiguous_history_fallback_key_count,
            work_author_years,
            record_key_to_work,
        ) = history_deduplicator.finalize()

        relevant_history_records = history_deduplicator.relevant_records

        print(
            "Relevant historical DBLP records scanned:",
            relevant_history_records
        )
        print(
            "Unique historical works after deduplication:",
            unique_historical_work_count
        )
        print(
            "Duplicate DBLP records collapsed:",
            duplicate_history_records_collapsed
        )
        print(
            "Ambiguous title-author fingerprints:",
            ambiguous_history_fallback_key_count
        )

        _pass2_payload = {
            "cache_key": DBLP_CACHE_KEY,
            "identity_fingerprint": DBLP_IDENTITY_FINGERPRINT,
            "author_year_counts": author_year_counts,
            "unique_historical_work_count": unique_historical_work_count,
            "duplicate_history_records_collapsed": duplicate_history_records_collapsed,
            "ambiguous_history_fallback_key_count": ambiguous_history_fallback_key_count,
            "relevant_history_records": relevant_history_records,
            "work_author_years": work_author_years,
            "record_key_to_work": record_key_to_work,
        }
        atomic_pickle_gzip_dump(_pass2_payload, DBLP_PASS2_CACHE_PATH)
        update_dblp_cache_manifest(
            pass2_cache_file=DBLP_PASS2_CACHE_PATH.name,
            pass2_identity_fingerprint=DBLP_IDENTITY_FINGERPRINT,
            pass2_records_relevant=relevant_history_records,
            unique_historical_work_count=unique_historical_work_count,
            pass2_cache_built_at_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        )
        print("Saved Pass 2 cache:", DBLP_PASS2_CACHE_PATH)

    # Downstream feature calculation only looks up the selected DBLP record
    # for each of the target arXiv papers. The legacy Pass 2 cache stores
    # record_key_to_work and work_author_years for every relevant DBLP work;
    # retain only target-paper focal records to avoid carrying the entire
    # historical work index through feature generation.
    needed_record_keys = {
        result["selected"]["key"]
        for result in paper_results.values()
        if result.get("selected") and result["selected"].get("key")
    }
    record_key_to_work = {
        key: record_key_to_work[key]
        for key in needed_record_keys
        if key in record_key_to_work
    }
    needed_work_ids = set(record_key_to_work.values())
    work_author_years = {
        work_id: work_author_years[work_id]
        for work_id in needed_work_ids
        if work_id in work_author_years
    }

    if isinstance(_pass2_payload, dict):
        _pass2_payload["record_key_to_work"] = record_key_to_work
        _pass2_payload["work_author_years"] = work_author_years
        atomic_pickle_gzip_dump(_pass2_payload, DBLP_PASS2_CACHE_PATH)
        del _pass2_payload
        print(
            "Compacted Pass 2 cache: retained",
            len(record_key_to_work), "target record keys and",
            len(work_author_years), "target work clusters"
        )
    del needed_record_keys, needed_work_ids
    gc.collect()
    log_memory("after Pass 2 cache compaction")

    # ============================================================
    # PART 13. CALCULATE AUTHOR FEATURES AT t0
    # ============================================================

    def calculate_author_features(person_id, t0_year, focal_work_id=None):

        # Use a private Counter so excluding the focal work never mutates cache.
        counts = Counter(author_year_counts.get(person_id, Counter()))

        # Exclude the target paper's deduplicated work cluster from its own
        # authors' historical productivity. This also removes earlier preprint /
        # conference / journal versions of the same work when DBLP links them.
        focal_author_year = (
            work_author_years.get(focal_work_id, {}).get(person_id)
            if focal_work_id is not None else None
        )
        focal_work_excluded = False
        if focal_author_year is not None and focal_author_year < t0_year:
            if counts.get(focal_author_year, 0) > 0:
                counts[focal_author_year] -= 1
                if counts[focal_author_year] <= 0:
                    counts.pop(focal_author_year, None)
                focal_work_excluded = True

        # DBLP provides publication years, not exact publication dates. Keep the
        # conservative calendar-year cutoff: only year < t0_year is included.
        prior_years = [
            year for year, count in counts.items()
            if year < t0_year and count > 0
        ]

        publication_count = sum(
            count for year, count in counts.items()
            if year < t0_year and count > 0
        )

        if prior_years:

            first_year = min(prior_years)

            academic_age = (
                t0_year - first_year
            )

        else:

            first_year = None
            academic_age = None

        return {
            "publication_count": int(publication_count),
            "first_publication_year": first_year,
            "academic_age": academic_age,
            "focal_work_excluded": bool(focal_work_excluded),
            "focal_author_year": focal_author_year,
        }


    def get_focal_work_info(paper_result, t0_year):
        selected = paper_result.get("selected")
        if not selected:
            return {
                "focal_work_id": None,
                "focal_work_earliest_year": None,
                "temporal_status": "NO_VERIFIED_DBLP_MATCH",
                "previously_published": False,
            }

        record_key = selected.get("key")
        focal_work_id = record_key_to_work.get(record_key)
        work_years = work_author_years.get(focal_work_id, {}) if focal_work_id is not None else {}
        observed_years = [int(y) for y in work_years.values() if y is not None]
        selected_year = selected.get("year")

        year_candidates = observed_years[:]
        if selected_year is not None:
            year_candidates.append(int(selected_year))
        earliest_year = min(year_candidates) if year_candidates else None

        if earliest_year is None:
            temporal_status = "PUBLICATION_YEAR_UNKNOWN"
        elif earliest_year < t0_year:
            temporal_status = "PREVIOUSLY_PUBLISHED_BEFORE_ARXIV_V1"
        elif earliest_year == t0_year:
            temporal_status = "SAME_CALENDAR_YEAR_AS_ARXIV_V1"
        else:
            temporal_status = "PUBLISHED_AFTER_ARXIV_V1"

        return {
            "focal_work_id": focal_work_id,
            "focal_work_earliest_year": earliest_year,
            "temporal_status": temporal_status,
            "previously_published": bool(
                earliest_year is not None and earliest_year < t0_year
            ),
        }


    # ============================================================
    # PART 14. BUILD PAPER-LEVEL AND AUTHOR-LEVEL TABLES
    # ============================================================

    paper_feature_rows = []
    by_author_rows = []

    # Pop each paper's matching objects as soon as its feature rows are built,
    # so the large nested matching structures shrink progressively.
    for paper_index, aid in enumerate(list(paper_meta), start=1):

        meta = paper_meta.pop(aid)
        result = paper_results.pop(aid)

        t0_year = meta["t0_year"]
        focal_info = get_focal_work_info(result, t0_year)
        focal_work_id = focal_info["focal_work_id"]

        paper_author_values = []
        matched_identity_confidences = []

        for author in result["author_rows"]:

            matched = author.get(
                "dblp_author_matched_to_paper",
                False
            )

            pid = author.get("identity_id")

            if matched and pid:

                values = calculate_author_features(
                    pid,
                    t0_year,
                    focal_work_id=focal_work_id
                )

            else:

                values = {
                    "publication_count": None,
                    "first_publication_year": None,
                    "academic_age": None
                }

            author_row = {
                "arxiv_id": aid,
                "author_position": author["author_position"],
                "arxiv_author_name_v1": author["author_name_arxiv_v1"],
                "dblp_person_id": pid,
                "dblp_identity_method": author.get("identity_method"),
                "dblp_identity_confidence": author.get("identity_confidence", "UNRESOLVED"),
                "dblp_profile_status": author.get("profile_match_status"),
                "dblp_profile_candidate_person_ids": ";".join(author.get("profile_candidate_person_ids") or []),
                "dblp_profile_all_candidate_person_ids": ";".join(author.get("profile_all_candidate_person_ids") or []),
                "dblp_profile_weaker_candidate_person_ids": ";".join(author.get("profile_weaker_candidate_person_ids") or []),
                "dblp_profile_candidate_count": author.get("profile_candidate_count", 0),
                "dblp_identity_priority_override": bool(author.get("identity_priority_override", False)),
                "dblp_identity_priority_alternative_person_ids": ";".join(author.get("identity_priority_alternative_person_ids") or []),
                "dblp_identity_conflict_status": author.get("identity_conflict_status"),
                "dblp_identity_conflict_source_person_id": author.get("identity_conflict_source_person_id"),
                "dblp_identity_conflict_target_person_id": author.get("identity_conflict_target_person_id"),
                "dblp_identity_conflict_target_author_name": author.get("identity_conflict_target_author_name"),
                "dblp_author_matched_to_paper": matched,
                "t0_year": t0_year,
                "dblp_focal_work_id": focal_work_id,
                "dblp_focal_work_excluded_from_history": values.get("focal_work_excluded", False),
                "dblp_author_publication_count_pre_t0": values["publication_count"],
                "dblp_author_first_publication_year_pre_t0": values["first_publication_year"],
                "dblp_author_academic_age_pre_t0": values["academic_age"],
                "dblp_record_key": result["selected"]["key"]
                if result["selected"] else None
            }

            by_author_rows.append(author_row)

            if matched and pid:
                paper_author_values.append(values)
                matched_identity_confidences.append(
                    author.get("identity_confidence", "UNRESOLVED")
                )

        full = (
            result["status"] == "MATCHED_FULL_AUTHOR_SET"
        )

        if full:

            total_publications = sum(
                x["publication_count"]
                for x in paper_author_values
            )

            missing_age_count = sum(
                x["academic_age"] is None
                for x in paper_author_values
            )

            if missing_age_count == 0:

                total_academic_age = sum(
                    x["academic_age"]
                    for x in paper_author_values
                )

            else:

                total_academic_age = None

        else:

            total_publications = None
            total_academic_age = None
            missing_age_count = None

        focal_work_excluded_author_count = sum(
            bool(x.get("focal_work_excluded", False))
            for x in paper_author_values
        )

        confidence_counts = Counter(matched_identity_confidences)
        high_confidence_count = confidence_counts.get("HIGH", 0)
        medium_confidence_count = confidence_counts.get("MEDIUM", 0)
        low_confidence_count = confidence_counts.get("LOW", 0)
        unresolved_identity_count = sum(
            1
            for author in result["author_rows"]
            if author.get("identity_confidence", "UNRESOLVED") == "UNRESOLVED"
        )

        all_high_confidence = (
            full
            and high_confidence_count == len(meta["authors"])
        )
        all_high_or_medium_confidence = (
            full
            and high_confidence_count + medium_confidence_count
            == len(meta["authors"])
        )

        selected = result["selected"]

        paper_feature_rows.append({

            "arxiv_id": aid,

            "title_v1": meta["title"],

            "t0_date": meta["t0"].isoformat(),

            "t0_year": t0_year,

            "dblp_match_status": result["status"],

            "dblp_match_method": result["method"],

            "dblp_record_key": (
                selected["key"] if selected else None
            ),

            "dblp_record_type": (
                selected["type"] if selected else None
            ),

            "dblp_record_year": (
                selected["year"] if selected else None
            ),

            "dblp_focal_work_id": focal_work_id,
            "dblp_focal_work_earliest_year": focal_info["focal_work_earliest_year"],
            "dblp_temporal_status": focal_info["temporal_status"],
            "dblp_previously_published_before_arxiv_v1": focal_info["previously_published"],
            "dblp_focal_work_excluded_from_author_history": bool(focal_work_excluded_author_count > 0),
            "dblp_focal_work_excluded_author_count": int(focal_work_excluded_author_count),

            "dblp_candidate_record_count": result["candidate_count"],

            "dblp_duplicate_version_group_size": result["candidate_group_size"],

            "arxiv_author_count": len(meta["authors"]),

            "dblp_author_match_count": result["author_match_count"],

            "dblp_author_identity_complete": full,

            "dblp_high_confidence_author_count": high_confidence_count,

            "dblp_medium_confidence_author_count": medium_confidence_count,

            "dblp_low_confidence_author_count": low_confidence_count,

            "dblp_unresolved_author_count": unresolved_identity_count,
            "dblp_identity_conflict_author_count": sum(
                bool(author.get("identity_conflict_status"))
                for author in result["author_rows"]
            ),

            "dblp_all_authors_high_confidence": all_high_confidence,

            "dblp_all_authors_high_or_medium_confidence": all_high_or_medium_confidence,

            "dblp_total_author_publication_count_pre_t0": total_publications,

            "dblp_total_academic_age_pre_t0": total_academic_age,

            "dblp_publication_count_available": (
                total_publications is not None
            ),

            "dblp_academic_age_available": (
                total_academic_age is not None
            ),

            "dblp_academic_age_missing_author_count": missing_age_count

        })

        if paper_index % 5000 == 0:
            print(f"DBLP feature generation: {paper_index}/{input_paper_count} papers")
            log_memory(f"after {paper_index} paper feature rows")

    # Save scalar counts needed for metadata, then release the large lookup
    # structures before constructing pandas output DataFrames.
    matched_person_id_count = len(matched_person_ids)
    relevant_author_alias_count = len(history_alias_to_person)
    target_record_key_count = len(record_key_to_work)

    del paper_results, paper_meta
    del author_year_counts, work_author_years, record_key_to_work
    del matched_person_ids, history_alias_to_person
    gc.collect()
    log_memory("after releasing DBLP history indexes")

    # ============================================================
    # PART 15. SAVE OUTPUT TABLES
    # ============================================================

    log_memory("before output DataFrame construction")
    dblp_author_features_df = pd.DataFrame(
        paper_feature_rows
    )
    log_memory("after paper-level DataFrame construction")

    dblp_author_features_by_author_df = pd.DataFrame(
        by_author_rows
    )
    log_memory("after author-level DataFrame construction")

    # The audit CSV has the same columns/rows; reuse the same DataFrame.
    dblp_author_match_audit_df = dblp_author_features_df

    dblp_candidate_match_audit_df = pd.DataFrame(
        candidate_audit_rows
    )

    # Dedicated author identity conflict / ambiguity audit.
    identity_conflict_rows = []
    for row in by_author_rows:
        conflict_status = row.get("dblp_identity_conflict_status")
        if conflict_status:
            identity_conflict_rows.append({
                "arxiv_id": row["arxiv_id"],
                "author_position": row["author_position"],
                "arxiv_author_name_v1": row["arxiv_author_name_v1"],
                "dblp_record_key": row.get("dblp_record_key"),
                "profile_status": row.get("dblp_profile_status"),
                "profile_candidate_person_ids": row.get("dblp_profile_candidate_person_ids"),
                "profile_all_candidate_person_ids": row.get("dblp_profile_all_candidate_person_ids"),
                "profile_weaker_candidate_person_ids": row.get("dblp_profile_weaker_candidate_person_ids"),
                "profile_candidate_count": row.get("dblp_profile_candidate_count"),
                "identity_conflict_status": conflict_status,
                "source_person_id": row.get("dblp_identity_conflict_source_person_id"),
                "target_person_id": row.get("dblp_identity_conflict_target_person_id"),
                "target_author_name": row.get("dblp_identity_conflict_target_author_name"),
            })
    dblp_identity_conflicts_df = pd.DataFrame(identity_conflict_rows)

    # Audit every author resolved by a stronger alias while weaker signature
    # candidates were also present. These identities are MEDIUM confidence.
    identity_priority_audit_rows = []
    for row in by_author_rows:
        alternatives = row.get("dblp_identity_priority_alternative_person_ids")
        if row.get("dblp_identity_priority_override") and alternatives:
            identity_priority_audit_rows.append({
                "arxiv_id": row["arxiv_id"],
                "author_position": row["author_position"],
                "arxiv_author_name_v1": row["arxiv_author_name_v1"],
                "selected_person_id": row.get("dblp_person_id"),
                "identity_method": row.get("dblp_identity_method"),
                "identity_confidence": row.get("dblp_identity_confidence"),
                "strongest_tier_candidates": row.get("dblp_profile_candidate_person_ids"),
                "all_profile_candidates": row.get("dblp_profile_all_candidate_person_ids"),
                "weaker_alternative_person_ids": alternatives,
                "dblp_record_key": row.get("dblp_record_key"),
            })
    dblp_identity_priority_audit_df = pd.DataFrame(identity_priority_audit_rows)

    # Dedicated audit for papers whose DBLP work predates the arXiv v1 t0,
    # plus records whose focal work could not be linked into the history index.
    temporal_audit_rows = []
    for row in paper_feature_rows:
        if (
            row.get("dblp_previously_published_before_arxiv_v1")
            or (
                row.get("dblp_record_key")
                and row.get("dblp_focal_work_id") is None
            )
        ):
            temporal_audit_rows.append({
                "arxiv_id": row["arxiv_id"],
                "t0_date": row["t0_date"],
                "t0_year": row["t0_year"],
                "dblp_record_key": row.get("dblp_record_key"),
                "dblp_record_year": row.get("dblp_record_year"),
                "dblp_focal_work_id": row.get("dblp_focal_work_id"),
                "dblp_focal_work_earliest_year": row.get("dblp_focal_work_earliest_year"),
                "dblp_temporal_status": row.get("dblp_temporal_status"),
                "dblp_previously_published_before_arxiv_v1": row.get("dblp_previously_published_before_arxiv_v1"),
                "dblp_focal_work_excluded_from_author_history": row.get("dblp_focal_work_excluded_from_author_history"),
                "dblp_focal_work_excluded_author_count": row.get("dblp_focal_work_excluded_author_count"),
            })
    dblp_temporal_exclusion_audit_df = pd.DataFrame(temporal_audit_rows)


    feature_path = (
        DBLP_OUTPUT_DIR / "dblp_author_features.csv"
    )

    author_path = (
        DBLP_OUTPUT_DIR / "dblp_author_features_by_author.csv"
    )

    audit_path = (
        DBLP_OUTPUT_DIR / "dblp_author_match_audit.csv"
    )

    candidate_path = (
        DBLP_OUTPUT_DIR / "dblp_candidate_match_audit.csv"
    )
    identity_conflict_path = (
        DBLP_OUTPUT_DIR / "dblp_identity_conflicts.csv"
    )
    identity_priority_audit_path = (
        DBLP_OUTPUT_DIR / "dblp_identity_priority_audit.csv"
    )
    temporal_audit_path = (
        DBLP_OUTPUT_DIR / "dblp_temporal_exclusion_audit.csv"
    )

    atomic_csv(dblp_author_features_df, 
        feature_path,
        index=False
    )

    atomic_csv(dblp_author_features_by_author_df, 
        author_path,
        index=False
    )

    atomic_csv(dblp_author_match_audit_df, 
        audit_path,
        index=False
    )

    atomic_csv(dblp_candidate_match_audit_df, 
        candidate_path,
        index=False
    )

    atomic_csv(dblp_identity_conflicts_df, 
        identity_conflict_path,
        index=False
    )

    atomic_csv(dblp_identity_priority_audit_df, 
        identity_priority_audit_path,
        index=False
    )

    atomic_csv(dblp_temporal_exclusion_audit_df, 
        temporal_audit_path,
        index=False
    )


    # ============================================================
    # PART 16. SOURCE METADATA AND CHECKSUM
    # ============================================================

    def calculate_md5(path):

        digest = hashlib.md5()

        with open(path, "rb") as f:

            for chunk in iter(
                lambda: f.read(16 * 1024 * 1024),
                b""
            ):
                digest.update(chunk)

        return digest.hexdigest()


    # Reuse the full XML MD5 from the cache manifest on cache hits. The quick
    # fingerprint above is enough to validate the local source without rereading
    # the entire compressed dump on every run.
    source_md5 = DBLP_EXISTING_MANIFEST.get("xml_md5")
    if not source_md5:
        print("\nCalculating DBLP XML checksum (first build only)...")
        source_md5 = calculate_md5(DBLP_XML_PATH)

    source_metadata = {

        "database": "DBLP",
        "pipeline_version": "3.1.1",

        "source_url": DBLP_XML_URL,

        "xml_path": str(DBLP_XML_PATH),

        "xml_parse_path": str(DBLP_XML_PARSE_PATH),

        "dtd_path": str(DBLP_DTD_PATH),

        "xml_size_bytes": DBLP_XML_PATH.stat().st_size,

        "xml_md5": source_md5,
        "xml_quick_fingerprint": DBLP_XML_QUICK_FINGERPRINT,
        "input_fingerprint": DBLP_INPUT_FINGERPRINT,
        "cache_key": DBLP_CACHE_KEY,
        "cache_schema_version": DBLP_CACHE_SCHEMA_VERSION,
        "identity_logic_version": DBLP_IDENTITY_LOGIC_VERSION,
        "history_logic_version": DBLP_HISTORY_LOGIC_VERSION,
        "cache_directory": str(DBLP_CACHE_DIR),
        "pass1_cache_file": DBLP_PASS1_CACHE_PATH.name,
        "pass2_cache_file": DBLP_PASS2_CACHE_PATH.name,
        "cache_force_rebuild": DBLP_FORCE_REBUILD_CACHE,

        "input_papers": input_paper_count,

        "pass1_records_in_index": pass1_records,
        "pass1_scan_performed_this_run": pass1_scan_performed,
        "pass1_cache_hit_this_run": pass1_cache_hit,
        "pass2_scan_performed_this_run": pass2_scan_performed,
        "pass2_cache_hit_this_run": pass2_cache_hit,

        "matched_person_id_count": matched_person_id_count,

        "relevant_author_alias_count": relevant_author_alias_count,

        "ambiguous_alias_count": ambiguous_alias_count,

        "relevant_history_records": relevant_history_records,

        "unique_historical_work_count_after_deduplication": unique_historical_work_count,

        "duplicate_history_records_collapsed": duplicate_history_records_collapsed,

        "ambiguous_history_fallback_key_count": ambiguous_history_fallback_key_count,

        "historical_work_deduplication": {
            "strong_identifiers": ["DOI", "arXiv ID"],
            "fallback": "exact normalized title + complete normalized author list",
            "conflicting_strong_identifier_groups_merged_by_title_author": False,
            "author_counting_rule": "At most one count per author per deduplicated work; earliest observed year for that author-work pair is retained"
        },

        "publication_types_included": list(DBLP_PUBLICATION_TAGS),

        "same_year_publications_excluded": True,

        "temporal_cutoff": "publication_year < t0_year",

        "temporal_cutoff_precision": "calendar_year",

        "temporal_cutoff_note": (
            "DBLP publication year is used instead of an exact publication date. "
            "Only publication_year < t0_year is counted. All records in the same "
            "calendar year as arXiv v1 are conservatively excluded. The focal paper's "
            "deduplicated work cluster is excluded from its own authors' history. "
            "Works whose earliest observed DBLP year predates arXiv v1 are flagged "
            "as PREVIOUSLY_PUBLISHED_BEFORE_ARXIV_V1; they remain in the output cohort "
            "for downstream main-sample / robustness decisions."
        ),

        "identity_confidence_levels": {
            "HIGH": "A unique raw exact or normalized full-name DBLP profile alias with no weaker-tier competing profile and no source-target profile ID conflict",
            "MEDIUM": "Unique first-initial + surname fallback, or a stronger full-name alias selected while weaker signature alternatives are also present",
            "LOW": "Exact raw author-string fallback without a persistent DBLP person ID",
            "UNRESOLVED": "No usable author identity"
        },

        "feature_definitions": {

            "author_publication_count":
                "Sum across arXiv v1 authors of distinct DBLP works with publication_year < t0_year, excluding the focal paper's deduplicated work cluster; unresolved/conflicted author identities do not contribute",

            "academic_age":
                "Sum across arXiv v1 authors of (t0_year - earliest observed DBLP work year strictly before t0_year), after excluding the focal paper's deduplicated work cluster",

            "missing_academic_age":
                "At least one arXiv v1 author has no observed DBLP work before t0_year"

        },

        "identity_conflict_policy": {
            "exact_alias_priority": "A unique raw exact alias has first priority, followed by a unique normalized full-name alias. A weaker first-initial + surname collision does not invalidate a unique full-name match.",
            "same_tier_collision_policy": "If multiple canonical DBLP person IDs share the strongest available raw/full-name alias, the identity remains UNRESOLVED. Signature-only resolution also requires exactly one candidate.",
            "weaker_tier_alternative_policy": "A unique stronger alias is selected over weaker signature candidates, but identity confidence is downgraded to MEDIUM and alternatives are written to dblp_identity_priority_audit.csv.",
            "source_target_person_id_mismatch": "If the resolved source profile ID and selected target-publication author ID disagree, mark IDENTITY_CONFLICT and exclude the author from historical feature aggregation.",
            "historical_alias_policy": "A raw author alias is usable when it maps to exactly one canonical person ID. Only raw aliases shared by multiple person IDs are excluded; first-initial + surname collisions alone do not invalidate a unique raw alias."
        },

        "temporal_alignment": {
            "focal_work_exclusion": "Exclude the selected target paper's deduplicated DBLP work cluster from each target author's historical count and academic age.",
            "pre_arxiv_publication_flag": "PREVIOUSLY_PUBLISHED_BEFORE_ARXIV_V1 when the earliest observed year among the selected focal-work cluster and its mapped authors is < t0_year.",
            "cohort_policy": "Keep flagged papers in the exported cohort; downstream analysis should decide whether to exclude them from the main specification and retain them for robustness analysis."
        },

        "cache": {
            "persistent": True,
            "pass1_cache_hit_skips_xml_scan": True,
            "pass2_cache_hit_skips_xml_scan": True,
            "source_validation": "File size + mtime_ns + first/last 1 MiB SHA-256 sample; full XML MD5 stored on first build and reused.",
            "force_rebuild_switch": "DBLP_FORCE_REBUILD_CACHE"
        },

        "output_directory": str(DBLP_OUTPUT_DIR)

    }

    (DBLP_OUTPUT_DIR / "dblp_source_metadata.json").write_text(
        json.dumps(
            source_metadata,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    update_dblp_cache_manifest(
        xml_md5=source_md5,
        pass1_cache_file=DBLP_PASS1_CACHE_PATH.name,
        pass2_cache_file=DBLP_PASS2_CACHE_PATH.name,
        pass1_records_in_index=pass1_records,
        pass1_scan_performed_this_run=pass1_scan_performed,
        pass1_cache_hit_this_run=pass1_cache_hit,
        pass2_scan_performed_this_run=pass2_scan_performed,
        pass2_cache_hit_this_run=pass2_cache_hit,
        relevant_history_records=relevant_history_records,
        unique_historical_work_count=unique_historical_work_count,
        completed_at_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
    )


    # ============================================================
    # PART 17. FINAL SUMMARY
    # ============================================================

    status_counts = (
        dblp_author_features_df[
            "dblp_match_status"
        ]
        .value_counts()
    )

    full_count = int(
        (
            dblp_author_features_df["dblp_match_status"]
            == "MATCHED_FULL_AUTHOR_SET"
        ).sum()
    )

    partial_count = int(
        (
            dblp_author_features_df["dblp_match_status"]
            == "MATCHED_PARTIAL_AUTHOR_SET"
        ).sum()
    )

    incomplete_count = int(
        (
            dblp_author_features_df["dblp_match_status"]
            == "MATCHED_PAPER_AUTHOR_IDENTITY_INCOMPLETE"
        ).sum()
    )

    publication_available = int(
        dblp_author_features_df[
            "dblp_publication_count_available"
        ].sum()
    )

    academic_age_available = int(
        dblp_author_features_df[
            "dblp_academic_age_available"
        ].sum()
    )

    print("\n" + "=" * 65)
    print("DBLP AUTHOR FEATURE COLLECTION V3.1.1 SUMMARY")
    print("=" * 65)

    print("Input papers:", input_paper_count)

    print("Matched full author sets:", full_count)

    print("Matched partial author sets:", partial_count)

    print("Paper matched but author identity incomplete:", incomplete_count)

    print(
        "Ambiguous paper matches:",
        int(
            (
                dblp_author_features_df["dblp_match_status"]
                == "AMBIGUOUS_DBLP_PAPER_MATCH"
            ).sum()
        )
    )

    print(
        "Publication-count feature available:",
        publication_available
    )

    print(
        "Academic-age feature available:",
        academic_age_available
    )

    print("Relevant historical records scanned:", relevant_history_records)
    print("Unique historical works after deduplication:", unique_historical_work_count)
    print("Duplicate DBLP records collapsed:", duplicate_history_records_collapsed)
    print("Ambiguous title-author fingerprints:", ambiguous_history_fallback_key_count)
    print("Identity conflict / ambiguity audit rows:", len(dblp_identity_conflicts_df))
    print("Identity priority override audit rows:", len(dblp_identity_priority_audit_df))
    print("Previously published before arXiv v1:", int(dblp_author_features_df["dblp_previously_published_before_arxiv_v1"].sum()))
    print("Target-paper focal record keys retained:", target_record_key_count)
    print("Pass 1 cache hit this run:", pass1_cache_hit)
    print("Pass 1 XML scan performed this run:", pass1_scan_performed)
    print("Pass 2 cache hit this run:", pass2_cache_hit)
    print("Pass 2 XML scan performed this run:", pass2_scan_performed)

    print("\nMatch status:")
    print(status_counts.to_string())

    print("\nAuthor identity confidence totals:")
    if not dblp_author_features_by_author_df.empty:
        print(
            dblp_author_features_by_author_df[
                "dblp_identity_confidence"
            ].value_counts(dropna=False).to_string()
        )

    print("\nOutput dimensions:")

    print(
        "Paper-level records:",
        len(dblp_author_features_df)
    )

    print(
        "Author-level records:",
        len(dblp_author_features_by_author_df)
    )

    print(
        "Candidate audit records:",
        len(dblp_candidate_match_audit_df)
    )

    print("\nDBLP XML MD5:", source_md5)

    print("\nSaved files:")

    for path in [
        feature_path,
        author_path,
        audit_path,
        candidate_path,
        identity_conflict_path,
        identity_priority_audit_path,
        temporal_audit_path,
        DBLP_OUTPUT_DIR / "dblp_source_metadata.json"
    ]:
        print(path)

    print("\nDBLP AUTHOR FEATURE COLLECTION V3.1.1 COMPLETED!")

    log_preview(
        dblp_author_features_df.head(10)
    )

    return {"dblp_author_features_df": dblp_author_features_df, "dblp_author_features_by_author_df": dblp_author_features_by_author_df}


def run(config: Config, http: HTTPClient, papers_df=None, seed_df=None):
    """返回统一 SourceResult；异常由中控记录，绝不构造成功或零值。"""
    from fetch.control.state import source_result
    tables = collect(config, http, papers_df=papers_df, seed_df=seed_df)
    count = len(papers_df) if papers_df is not None else len(seed_df) if seed_df is not None else 0
    return source_result('dblp', tables, http, count)

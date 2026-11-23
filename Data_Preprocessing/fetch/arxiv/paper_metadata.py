"""显式调用的 Notebook 迁移模块。输入输出与主键见 README 和 DATA_DICTIONARY。"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fetch.control.config import Config
    from fetch.control.tools import HTTPClient
    import pandas as pd
from fetch.control.tools import atomic_csv, log_preview, log_print as print

def collect(config: Config, http: HTTPClient, papers_df=None, seed_df=None) -> dict[str, pd.DataFrame]:
    import calendar, gzip, re, time, random, requests
    import xml.etree.ElementTree as ET
    from pathlib import Path
    import pandas as pd
    from fetch.control.tools import quiet_tqdm as tqdm
    ROOT, RAW, PROCESSED = config.paths
    START_YEAR, END_YEAR = config.start_year, config.end_year
    TARGET_PER_YEAR, RANDOM_SEED = config.target_per_year, config.random_seed
    ARXIV_PAGE_SIZE = config.arxiv_page_size
    NS = {"atom":"http://www.w3.org/2005/Atom", "arxiv":"http://arxiv.org/schemas/atom", "opensearch":"http://a9.com/-/spec/opensearch/1.1/"}
    import json, hashlib
    from fetch.control.tools import chunks, read_jsonl, write_jsonl, clean_arxiv_id, parse_arxiv_feed
    arxiv_request = http.arxiv

    required_columns = [
        "arxiv_id",
        "seed_query_year"
    ]

    if not all(c in seed_df.columns for c in required_columns):

        raise ValueError(
            "seed_df must contain arxiv_id and seed_query_year."
        )

    if seed_df.empty:

        raise ValueError(
            "seed_df is empty. Please run the arXiv seed collection first."
        )

    # Remove duplicate seed IDs
    seed_df = (
        seed_df[required_columns]
        .drop_duplicates("arxiv_id")
        .copy()
    )

    print("Seed IDs available:", seed_df["arxiv_id"].nunique())


    # ============================================================
    # PART 3. PARSE ONE ARXIV V1 ENTRY
    # ============================================================

    def parse_arxiv_v1_entry(entry, seed_year):

        # 1. Paper ID
        id_node = entry.find("atom:id", NS)

        if id_node is None:
            return None

        aid = clean_arxiv_id(
            id_node.text.split("/abs/")[-1]
        )

        # 2. Basic paper information
        title_node = entry.find("atom:title", NS)
        abstract_node = entry.find("atom:summary", NS)

        published_node = entry.find("atom:published", NS)
        updated_node = entry.find("atom:updated", NS)

        title = (
            " ".join((title_node.text or "").split())
            if title_node is not None
            else None
        )

        abstract = (
            " ".join((abstract_node.text or "").split())
            if abstract_node is not None
            else None
        )

        # 3. Authors and raw affiliations
        authors = []

        for author in entry.findall("atom:author", NS):

            name_node = author.find("atom:name", NS)
            aff_node = author.find("arxiv:affiliation", NS)

            authors.append({

                "name": (
                    name_node.text.strip()
                    if name_node is not None and name_node.text
                    else None
                ),

                "affiliation_raw": (
                    aff_node.text.strip()
                    if aff_node is not None and aff_node.text
                    else None
                )
            })

        # 4. All arXiv categories
        categories = []

        for cat in entry.findall("atom:category", NS):

            term = cat.attrib.get("term")

            if term:
                categories.append(term)

        # 5. Primary category
        primary_node = entry.find(
            "arxiv:primary_category",
            NS
        )

        primary = (
            primary_node.attrib.get("term")
            if primary_node is not None
            else None
        )

        if primary and primary not in categories:
            categories.insert(0, primary)

        # 6. Keep Computer Science categories
        cs_categories = sorted({
            c for c in categories
            if c.startswith("cs.")
        })

        # Exclude papers without a CS category in v1
        if not cs_categories:
            return None

        # 7. Additional raw metadata
        comment_node = entry.find("arxiv:comment", NS)
        journal_node = entry.find("arxiv:journal_ref", NS)
        doi_node = entry.find("arxiv:doi", NS)

        # 8. Return one structured paper record
        return {

            "paper_id": aid,
            "arxiv_id": aid,
            "arxiv_version": "v1",

            "submission_date_v1": (
                published_node.text
                if published_node is not None
                else None
            ),

            "v1_updated_date": (
                updated_node.text
                if updated_node is not None
                else None
            ),

            "title_v1": title,
            "abstract_v1": abstract,

            "primary_category_v1": primary,

            "categories_v1_json": json.dumps(
                categories,
                ensure_ascii=False
            ),

            "cs_categories_v1_json": json.dumps(
                cs_categories,
                ensure_ascii=False
            ),

            "authors_v1_json": json.dumps(
                authors,
                ensure_ascii=False
            ),

            "author_count_v1": len(authors),

            "comment_v1": (
                comment_node.text.strip()
                if comment_node is not None and comment_node.text
                else None
            ),

            "journal_ref_v1": (
                journal_node.text.strip()
                if journal_node is not None and journal_node.text
                else None
            ),

            "doi_v1": (
                doi_node.text.strip()
                if doi_node is not None and doi_node.text
                else None
            ),

            "seed_query_year": int(seed_year)
        }


    # ============================================================
    # PART 4. FETCH V1 METADATA IN BATCHES
    # ============================================================

    def fetch_arxiv_v1_metadata(seed_df, batch_size=None):

        batch_size = batch_size or config.arxiv_metadata_batch_size
        seed_rows = seed_df.to_dict("records")

        all_records = []
        excluded_rows = []

        api_returned_total = 0
        cached_batches = 0
        downloaded_batches = 0

        # Divide 600 papers into batches of 100
        batches = list(
            chunks(seed_rows, batch_size)
        )

        print("\nNumber of batches:", len(batches))

        for batch_no, batch in enumerate(
            tqdm(batches, desc="Fetch arXiv v1")
        ):

            # Normalize IDs
            ids = [
                clean_arxiv_id(r["arxiv_id"]) + "v1"
                for r in batch
            ]

            requested_ids = [
                clean_arxiv_id(r["arxiv_id"])
                for r in batch
            ]

            # Create a fingerprint for this exact batch.
            # Different paper samples will not reuse old caches.

            batch_signature = hashlib.sha1(
                ",".join(ids).encode("utf-8")
            ).hexdigest()[:10]

            batch_name = (
                f"v1_batch_{batch_no:04d}_{batch_signature}"
            )

            parsed_path = (
                RAW / "arxiv" / "v1_jsonl" /
                f"{batch_name}.jsonl"
            )

            xml_path = (
                RAW / "arxiv" / "v1_xml" /
                f"{batch_name}.xml.gz"
            )

            meta_path = (
                RAW / "arxiv" / "v1_jsonl" /
                f"{batch_name}.meta.json"
            )

            # ----------------------------------------------------
            # Step 1. Reuse a valid completed batch if available
            # ----------------------------------------------------

            # ----------------------------------------------------
            # Step 2. Request exact v1 records from arXiv
            # ----------------------------------------------------

            xml_bytes = arxiv_request(

                {
                    "id_list": ",".join(ids),

                    # IMPORTANT: explicitly request all 100 records
                    "max_results": len(ids)
                },

                xml_path
            )

            _, entries = parse_arxiv_feed(xml_bytes)

            # ----------------------------------------------------
            # Step 3. Match returned entries by arXiv ID
            # ----------------------------------------------------

            returned_map = {}

            for entry in entries:

                id_node = entry.find("atom:id", NS)

                if id_node is None:
                    continue

                aid = clean_arxiv_id(
                    id_node.text.split("/abs/")[-1]
                )

                if aid in requested_ids:
                    returned_map[aid] = entry

            missing_ids = [
                aid for aid in requested_ids
                if aid not in returned_map
            ]

            # Never silently accept incomplete API responses
            if missing_ids:

                raise RuntimeError(
                    f"Batch {batch_no} is incomplete. "
                    f"Requested: {len(requested_ids)}, "
                    f"received: {len(returned_map)}. "
                    f"Missing IDs: {missing_ids[:10]}"
                )

            api_returned_total += len(returned_map)

            # ----------------------------------------------------
            # Step 4. Parse each paper
            # ----------------------------------------------------

            seed_year_map = {
                clean_arxiv_id(r["arxiv_id"]):
                int(r["seed_query_year"])
                for r in batch
            }

            batch_records = []
            batch_excluded = []

            # Preserve original batch order
            for aid in requested_ids:

                entry = returned_map[aid]

                record = parse_arxiv_v1_entry(
                    entry,
                    seed_year_map[aid]
                )

                if record:

                    batch_records.append(record)

                else:

                    batch_excluded.append({

                        "arxiv_id": aid,

                        "seed_query_year": seed_year_map[aid],

                        "exclusion_reason": (
                            "No cs.* category found in v1 metadata"
                        )
                    })

            # ----------------------------------------------------
            # Step 5. Save parsed batch as JSONL
            # ----------------------------------------------------

            write_jsonl(
                parsed_path,
                batch_records
            )

            # Save batch manifest for safe resuming
            meta = {

                "requested_ids": requested_ids,

                "returned_ids": list(returned_map.keys()),

                "retained_papers": len(batch_records),

                "excluded_non_cs": batch_excluded
            }

            with meta_path.open(
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    meta,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            all_records.extend(batch_records)

            excluded_rows.extend(batch_excluded)

            downloaded_batches += 1

            print(
                f"Batch {batch_no + 1}/{len(batches)}: "
                f"Retrieved {len(returned_map)}, "
                f"retained {len(batch_records)}, "
                f"excluded {len(batch_excluded)}"
            )

        # ========================================================
        # PART 5. BUILD PAPER MASTER TABLE
        # ========================================================

        if not all_records:
            raise ValueError("arXiv v1 没有保留任何 CS 论文；请检查 seed 清单或采样范围。")
        papers = pd.DataFrame(all_records)

        if papers.empty:

            raise RuntimeError(
                "No CS papers were retained from arXiv v1."
            )

        papers = papers.drop_duplicates(
            subset=["arxiv_id"]
        ).copy()

        # Convert submission date into datetime
        papers["submission_date_v1"] = pd.to_datetime(
            papers["submission_date_v1"],
            errors="coerce",
            utc=True
        )

        # Missing submission dates need investigation
        missing_dates = papers[
            papers["submission_date_v1"].isna()
        ]

        if not missing_dates.empty:

            raise RuntimeError(
                f"{len(missing_dates)} papers have missing "
                "v1 submission dates. Please inspect the records."
            )

        # Extract actual submission year
        papers["submission_year"] = (
            papers["submission_date_v1"].dt.year
        )

        # Check year consistency
        year_mismatch = papers[
            papers["submission_year"]
            != papers["seed_query_year"]
        ].copy()

        if not year_mismatch.empty:

            atomic_csv(year_mismatch, 
                PROCESSED / "arxiv_v1_year_mismatch.csv",
                index=False
            )

            print(
                "WARNING: Year mismatch records:",
                len(year_mismatch)
            )

        # Keep the requested time range
        papers = papers[
            papers["submission_year"].between(
                START_YEAR,
                END_YEAR
            )
        ].copy()

        # Save paper master table
        atomic_csv(papers, 
            PROCESSED / "papers_arxiv_v1.csv",
            index=False
        )

        # ========================================================
        # PART 6. BUILD AUTHOR TABLE
        # ========================================================

        author_rows = []

        for row in papers.to_dict("records"):

            paper_authors = json.loads(
                row.get("authors_v1_json") or "[]"
            )

            for pos, author in enumerate(
                paper_authors,
                start=1
            ):

                author_rows.append({

                    "arxiv_id": row["arxiv_id"],

                    "author_position": pos,

                    "author_name_raw": author.get("name"),

                    "affiliation_raw_arxiv_v1": (
                        author.get("affiliation_raw")
                    )
                })

        authors_df = pd.DataFrame(
            author_rows,

            columns=[
                "arxiv_id",
                "author_position",
                "author_name_raw",
                "affiliation_raw_arxiv_v1"
            ]
        )

        atomic_csv(authors_df, 
            PROCESSED / "authors_arxiv_v1.csv",
            index=False
        )

        # Save records excluded by v1 CS category filter
        excluded_df = pd.DataFrame(
            excluded_rows,
            columns=[
                "arxiv_id",
                "seed_query_year",
                "exclusion_reason"
            ]
        )

        atomic_csv(excluded_df, 
            PROCESSED / "arxiv_v1_excluded_non_cs.csv",
            index=False
        )

        # ========================================================
        # PART 7. FINAL COLLECTION REPORT
        # ========================================================

        print("\n")
        print("=" * 55)
        print("ARXIV V1 COLLECTION SUMMARY")
        print("=" * 55)

        print(
            "Selected seed IDs:",
            seed_df["arxiv_id"].nunique()
        )

        print(
            "API records successfully retrieved:",
            api_returned_total
        )

        print(
            "Retained v1 CS papers:",
            papers["arxiv_id"].nunique()
        )

        print(
            "Excluded non-CS v1 papers:",
            len(excluded_df)
        )

        print(
            "Missing abstracts:",
            papers["abstract_v1"].isna().sum()
        )

        print(
            "Missing titles:",
            papers["title_v1"].isna().sum()
        )

        print(
            "Author-paper rows:",
            len(authors_df)
        )

        print(
            "New batches downloaded:",
            downloaded_batches
        )

        print(
            "Batches reused from cache:",
            cached_batches
        )

        print("\nPapers by year:")

        print(
            papers.groupby("submission_year")
            .size()
        )

        print("\nSaved files:")

        print(
            PROCESSED / "papers_arxiv_v1.csv"
        )

        print(
            PROCESSED / "authors_arxiv_v1.csv"
        )

        print(
            PROCESSED / "arxiv_v1_excluded_non_cs.csv"
        )

        print("\nARXIV V1 COLLECTION COMPLETED!")

        return papers, authors_df


    # ============================================================
    # PART 8. RUN COLLECTION
    # ============================================================


    papers_df, authors_df = fetch_arxiv_v1_metadata(seed_df)
    return {"papers_df": papers_df, "arxiv_authors_df": authors_df}


def run(config: Config, http: HTTPClient, papers_df=None, seed_df=None):
    """返回统一 SourceResult；异常由中控记录，绝不构造成功或零值。"""
    from fetch.control.state import source_result
    tables = collect(config, http, papers_df=papers_df, seed_df=seed_df)
    count = len(papers_df) if papers_df is not None else len(seed_df) if seed_df is not None else 0
    return source_result('arxiv_metadata', tables, http, count)

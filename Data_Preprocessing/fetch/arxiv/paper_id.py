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

    def clean_arxiv_id(value):

        value = str(value).strip()

        value = re.sub(
            r"^https?://(?:export\.)?arxiv\.org/abs/",
            "",
            value
        )

        value = re.sub(
            r"v\d+$",
            "",
            value
        )

        return value.strip("/")


    # ------------------------------------------------------------
    # 5. arXiv API request with caching
    # ------------------------------------------------------------

    def arxiv_request(params, cache_path):
        return http.arxiv(params, cache_path)


    # ------------------------------------------------------------
    # 6. Parse arXiv XML response
    # ------------------------------------------------------------

    def parse_arxiv_feed(xml_bytes):

        root = ET.fromstring(xml_bytes)

        total_node = root.find(
            "opensearch:totalResults",
            NS
        )

        total = (
            int(total_node.text)
            if total_node is not None
            and total_node.text
            else 0
        )

        entries = root.findall(
            "atom:entry",
            NS
        )

        return total, entries


    # ------------------------------------------------------------
    # 7. Main collection function
    # ------------------------------------------------------------

    def collect_arxiv_seed_ids():

        all_seed_rows = []

        monthly_audit_rows = []

        year_candidate_counts = {}

        # Loop through 2017-2022
        for year in range(
            START_YEAR,
            END_YEAR + 1
        ):

            print("\n" + "=" * 55)
            print(f"Processing year: {year}")
            print("=" * 55)

            year_rows = []

            year_candidate_total = 0

            # ----------------------------------------------------
            # Retrieve data month by month
            # ----------------------------------------------------

            for month in range(1, 13):

                last_day = calendar.monthrange(
                    year,
                    month
                )[1]

                start_date = (
                    f"{year}{month:02d}010000"
                )

                end_date = (
                    f"{year}{month:02d}"
                    f"{last_day:02d}2359"
                )

                # CS category + monthly submission date
                query = (
                    f"cat:cs.* AND submittedDate:"
                    f"[{start_date} TO {end_date}]"
                )

                print(
                    f"\nChecking {year}-{month:02d}..."
                )

                # ------------------------------------------------
                # Step 1: Get monthly candidate count
                # ------------------------------------------------

                count_xml = arxiv_request(
                    {
                        "search_query": query,
                        "start": 0,
                        "max_results": 1,
                        "sortBy": "submittedDate",
                        "sortOrder": "ascending",
                    },

                    RAW / "arxiv" / "seed_xml" /
                    f"count_{year}_{month:02d}.xml.gz"
                )

                month_total, _ = parse_arxiv_feed(
                    count_xml
                )

                print(
                    f"{year}-{month:02d}: "
                    f"{month_total:,} candidate records"
                )

                if month_total == 0:

                    monthly_audit_rows.append({
                        "year": year,
                        "month": month,
                        "candidate_count": 0,
                        "retrieved_count": 0,
                        "status": "EMPTY"
                    })

                    continue

                # Avoid silently exceeding the API query limit
                if month_total >= 30000:

                    raise RuntimeError(
                        f"{year}-{month:02d} has "
                        f"{month_total} records. "
                        "This month needs smaller date partitions."
                    )

                year_candidate_total += month_total

                month_rows = []

                # ------------------------------------------------
                # Step 2: Retrieve ALL records for this month
                # ------------------------------------------------

                page_starts = range(
                    0,
                    month_total,
                    ARXIV_PAGE_SIZE
                )

                for start in tqdm(
                    page_starts,
                    desc=f"{year}-{month:02d}"
                ):

                    expected_count = min(
                        ARXIV_PAGE_SIZE,
                        month_total - start
                    )

                    # Unique monthly cache filename
                    cache_path = (
                        RAW / "arxiv" / "seed_xml" /
                        f"seed_{year}_{month:02d}_"
                        f"{start:08d}_p{ARXIV_PAGE_SIZE}.xml.gz"
                    )

                    xml_bytes = arxiv_request(
                        {
                            "search_query": query,
                            "start": int(start),
                            "max_results": ARXIV_PAGE_SIZE,
                            "sortBy": "submittedDate",
                            "sortOrder": "ascending",
                        },
                        cache_path
                    )

                    _, entries = parse_arxiv_feed(
                        xml_bytes
                    )

                    # Verify that the page is complete
                    if len(entries) != expected_count:

                        raise RuntimeError(
                            f"Incomplete page: "
                            f"{year}-{month:02d}, "
                            f"start={start}, "
                            f"expected={expected_count}, "
                            f"received={len(entries)}. "
                            "Please inspect the API response/cache."
                        )

                    # Extract paper IDs
                    for entry in entries:

                        id_node = entry.find(
                            "atom:id",
                            NS
                        )

                        pub_node = entry.find(
                            "atom:published",
                            NS
                        )

                        if id_node is None:
                            continue

                        aid = clean_arxiv_id(
                            id_node.text.split("/abs/")[-1]
                        )

                        month_rows.append({
                            "arxiv_id": aid,
                            "seed_query_year": year,
                            "seed_query_month": month,
                            "seed_published": (
                                pub_node.text
                                if pub_node is not None
                                else None
                            ),
                        })

                # ------------------------------------------------
                # Step 3: Remove monthly duplicates
                # ------------------------------------------------

                unique_month = {
                    row["arxiv_id"]: row
                    for row in month_rows
                }

                month_rows = list(
                    unique_month.values()
                )

                # Verify monthly completeness
                if len(month_rows) != month_total:

                    raise RuntimeError(
                        f"Monthly count mismatch for "
                        f"{year}-{month:02d}: "
                        f"API reports {month_total}, "
                        f"but retrieved {len(month_rows)} unique IDs."
                    )

                year_rows.extend(month_rows)

                monthly_audit_rows.append({
                    "year": year,
                    "month": month,
                    "candidate_count": month_total,
                    "retrieved_count": len(month_rows),
                    "status": "SUCCESS"
                })

                print(
                    f"Retrieved {len(month_rows):,} "
                    f"unique IDs for {year}-{month:02d}"
                )

            # ----------------------------------------------------
            # Step 4: Combine all 12 months
            # ----------------------------------------------------

            unique_year = {
                row["arxiv_id"]: row
                for row in year_rows
            }

            year_rows = list(
                unique_year.values()
            )

            # Verify the complete yearly candidate set
            if len(year_rows) != year_candidate_total:

                raise RuntimeError(
                    f"Yearly count mismatch for {year}: "
                    f"expected {year_candidate_total}, "
                    f"retrieved {len(year_rows)}."
                )

            year_candidate_counts[year] = len(
                year_rows
            )

            print("\nYearly candidate total:", len(year_rows))

            # ----------------------------------------------------
            # Step 5: Random sampling from the FULL year
            # ----------------------------------------------------

            target = min(
                TARGET_PER_YEAR,
                len(year_rows)
            )

            rng = random.Random(
                RANDOM_SEED + year
            )

            selected_rows = rng.sample(
                year_rows,
                target
            )

            all_seed_rows.extend(
                selected_rows
            )

            print(
                f"{year}: selected "
                f"{len(selected_rows)} unique papers"
            )

        # --------------------------------------------------------
        # 8. Combine six years
        # --------------------------------------------------------

        if not all_seed_rows:
            raise ValueError("指定年份没有可用 arXiv CS 候选论文。")
        seed_df = pd.DataFrame(
            all_seed_rows
        )

        if seed_df.empty:
            raise RuntimeError(
                "No arXiv papers were collected."
            )

        seed_df = seed_df.drop_duplicates(
            subset=["arxiv_id"]
        )

        # --------------------------------------------------------
        # 9. Verify final sample size
        # --------------------------------------------------------

        expected_total = sum(
            min(TARGET_PER_YEAR, year_candidate_counts[y])
            for y in range(START_YEAR, END_YEAR + 1)
        )

        if len(seed_df) != expected_total:

            raise RuntimeError(
                f"Final sample mismatch: "
                f"expected {expected_total}, "
                f"received {len(seed_df)}."
            )

        year_counts = (
            seed_df.groupby("seed_query_year")["arxiv_id"]
            .nunique()
            .reindex(
                range(START_YEAR, END_YEAR + 1),
                fill_value=0
            )
        )

        # --------------------------------------------------------
        # 10. Save data
        # --------------------------------------------------------

        seed_path = (
            PROCESSED / "arxiv_seed_ids.csv"
        )

        atomic_csv(seed_df, 
            seed_path,
            index=False
        )

        audit_df = pd.DataFrame(
            monthly_audit_rows
        )

        atomic_csv(audit_df, 
            PROCESSED / "arxiv_monthly_candidate_counts.csv",
            index=False
        )

        # --------------------------------------------------------
        # 11. Final summary
        # --------------------------------------------------------

        print("\n")
        print("=" * 55)
        print("ARXIV COLLECTION SUMMARY")
        print("=" * 55)

        print("\nCandidate records by year:")

        for year, count in year_candidate_counts.items():

            print(
                f"{year}: {count:,} candidates"
            )

        print("\nSelected papers by year:")
        print(year_counts)

        print("\nTotal unique candidate IDs:")
        print(seed_df["arxiv_id"].nunique())

        print("\nSaved files:")
        print(seed_path)

        print(
            PROCESSED /
            "arxiv_monthly_candidate_counts.csv"
        )

        print("\nCollection completed successfully!")

        return seed_df


    # ------------------------------------------------------------
    # 12. Run collection
    # ------------------------------------------------------------


    return {"seed_df": collect_arxiv_seed_ids()}


def run(config: Config, http: HTTPClient, papers_df=None, seed_df=None):
    """返回统一 SourceResult；异常由中控记录，绝不构造成功或零值。"""
    from fetch.control.state import source_result
    tables = collect(config, http, papers_df=papers_df, seed_df=seed_df)
    count = len(papers_df) if papers_df is not None else len(seed_df) if seed_df is not None else 0
    return source_result('arxiv_seed', tables, http, count)

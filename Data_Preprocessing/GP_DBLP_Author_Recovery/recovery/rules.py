"""Pure rules copied from user-supplied Fetch 3.1.1; no Fetch runtime dependency.
Normalization and historical union-find logic intentionally unchanged.
"""
import re, unicodedata, hashlib, json
from collections import defaultdict, Counter
ARXIV_PATTERN = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)\s*(\d{4}\.\d{4,5}|[a-z][a-z.-]*/\d{7})(?:v\d+)?", re.I)
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s<>\"'?#]+", re.I)
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

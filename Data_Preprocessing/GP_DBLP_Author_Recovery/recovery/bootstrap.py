"""Memory-bounded, read-only bootstrap from Fetch and prior DBLP evidence.

Large optional DBLP CSVs are streamed in chunks. Only columns/rows required by
identity revalidation and focal-work exclusion are retained in memory. Original
source files are still fully SHA256-hashed and never modified.
"""
import json, logging, re
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd

from common import sha256
from .io import write_csv, write_json
from .names import compatible
from .rules import normalize_name

OPTIONAL_CSVS = [
    'dblp_author_features.csv',
    'dblp_author_features_by_author.csv',
    'dblp_candidate_match_audit.csv',
    'dblp_identity_conflicts.csv',
    'dblp_identity_priority_audit.csv',
    'dblp_author_match_audit.csv',
    'dblp_temporal_exclusion_audit.csv',
]
DETAIL_COLUMNS = {
    'arxiv_id', 'author_position', 't0_year', 'dblp_author_matched_to_paper',
    'dblp_person_id', 'dblp_profile_status',
    'dblp_identity_conflict_source_person_id',
    'dblp_identity_conflict_target_person_id',
    'dblp_identity_conflict_status', 'dblp_identity_confidence',
    'arxiv_author_name_v1', 'dblp_record_key',
}
FOCAL_COLUMNS = ['arxiv_id', 'dblp_record_key']
CHUNK_SIZE = 25_000


def truth(x):
    return str(x).strip().lower() in ('true', '1', '1.0')


def clean_pid(pid):
    pid = str(pid or '').strip()
    for prefix in ('https://dblp.org/pid/', 'http://dblp.org/pid/', 'https://dblp.org/pers/hd/'):
        if pid.startswith(prefix):
            pid = pid[len(prefix):].removesuffix('.html').removesuffix('.xml')
    return pid if not pid or pid.startswith('homepages/') else 'homepages/' + pid


def _header(path):
    return pd.read_csv(path, nrows=0, dtype=str, keep_default_na=False).columns.tolist()


def _iter_csv(path, usecols=None):
    return pd.read_csv(
        path, dtype=str, keep_default_na=False, usecols=usecols,
        chunksize=CHUNK_SIZE, low_memory=False,
    )


def _schema_entry(path, columns, rows, present=True):
    return dict(file=str(path), present=present, rows=int(rows), columns=list(columns))


def _count_csv_rows(path, columns):
    """Count rows with a one-column chunked scan when no focal projection exists."""
    if not columns:
        return 0
    count = 0
    for chunk in _iter_csv(path, usecols=[columns[0]]):
        count += len(chunk)
    return count


def _stream_focal_keys(path, columns, years, focal, focal_audit, source_name):
    """Read only the two focal columns from an optional audit CSV."""
    if not set(FOCAL_COLUMNS).issubset(columns):
        return _count_csv_rows(path, columns)
    count = 0
    for chunk in _iter_csv(path, usecols=FOCAL_COLUMNS):
        count += len(chunk)
        for aid, key in chunk[FOCAL_COLUMNS].itertuples(index=False, name=None):
            if aid in years and key:
                focal[aid].add(key)
                focal_audit.append(dict(
                    arxiv_id=aid, record_key=key, source=source_name,
                    reason='FETCH_TARGET_OR_CANDIDATE_RECORD_NOT_IDENTITY_PROOF',
                ))
    return count


def load(cfg):
    root = Path(cfg.FETCH_DATA_ROOT)
    dbdir = root / cfg.dblp_directory
    sources, schema = {}, []

    def read_required(path):
        if not path.is_file():
            raise FileNotFoundError('Required Fetch file missing: ' + str(path))
        sources[str(path.resolve())] = sha256(path)
        df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
        schema.append(_schema_entry(path, df.columns, len(df)))
        return df

    master = read_required(root / cfg.master_file)
    authors = read_required(root / cfg.authors_file)

    if not {'arxiv_id', 'title_v1', 'submission_date_v1'} <= set(master):
        raise ValueError('Fetch master lacks arxiv_id/title_v1/submission_date_v1')
    if not {'arxiv_id', 'author_position', 'author_name_raw'} <= set(authors):
        raise ValueError('Fetch authors lack occurrence key or author_name_raw')
    if master.arxiv_id.duplicated().any():
        raise ValueError('Duplicate Fetch master IDs')

    # Canonical roster is small enough to hold in memory; it bounds all large
    # optional evidence reads to author occurrences in this target dataset.
    authors['author_position'] = pd.to_numeric(authors.author_position, errors='raise')
    if (authors.author_position < 1).any() or (authors.author_position % 1 != 0).any():
        raise ValueError('Invalid author positions')
    authors['author_position'] = authors.author_position.astype(int)
    if authors.duplicated(['arxiv_id', 'author_position']).any():
        raise ValueError('Duplicate Fetch occurrence keys')

    if set(authors.arxiv_id) - set(master.arxiv_id):
        raise ValueError('Author references paper missing from Fetch master')
    if set(master.arxiv_id) - set(authors.arxiv_id):
        raise ValueError('Fetch paper has no author records; fix source export')

    dates = pd.to_datetime(master.submission_date_v1, utc=True, errors='raise')
    if dates.isna().any():
        raise ValueError('Missing arXiv v1 submission timestamp')
    years = dict(zip(master.arxiv_id, dates.dt.year))
    paper_ids = set(years)
    author_keys = set(zip(authors.arxiv_id, authors.author_position))

    focal = defaultdict(set)
    focal_audit = []

    def add(aid, key, source):
        if aid in years and key:
            focal[aid].add(key)
            focal_audit.append(dict(
                arxiv_id=aid, record_key=key, source=source,
                reason='FETCH_TARGET_OR_CANDIDATE_RECORD_NOT_IDENTITY_PROOF',
            ))

    # Include candidate/focal record keys already present on the paper master.
    if 'dblp_record_key' in master.columns:
        for aid, key in master[['arxiv_id', 'dblp_record_key']].itertuples(index=False, name=None):
            add(aid, key, 'papers_master_temp.csv')

    # The large author-detail file is read with a narrow projection and chunks.
    # Keep only rows that can map to the canonical author roster.
    detail_path = dbdir / 'dblp_author_features_by_author.csv'
    detailmap = {}
    detail_columns = []
    detail_rows_seen = 0
    if detail_path.is_file():
        sources[str(detail_path.resolve())] = sha256(detail_path)
        all_columns = _header(detail_path)
        detail_columns = [c for c in all_columns if c in DETAIL_COLUMNS]
        if not {'arxiv_id', 'author_position'} <= set(detail_columns):
            raise ValueError('DBLP author-detail file lacks arxiv_id/author_position')
        for chunk in _iter_csv(detail_path, usecols=detail_columns):
            detail_rows_seen += len(chunk)
            chunk['author_position'] = pd.to_numeric(chunk.author_position, errors='raise')
            if (chunk.author_position < 1).any() or (chunk.author_position % 1 != 0).any():
                raise ValueError('Invalid DBLP author-detail positions')
            chunk['author_position'] = chunk.author_position.astype(int)
            if 'dblp_record_key' in chunk.columns:
                for aid, key in chunk[['arxiv_id', 'dblp_record_key']].itertuples(index=False, name=None):
                    add(aid, key, detail_path.name)
            for record in chunk.to_dict('records'):
                key = (record['arxiv_id'], int(record['author_position']))
                if key not in author_keys:
                    continue
                if key in detailmap:
                    raise ValueError('Duplicate DBLP author-detail occurrence key: ' + repr(key))
                detailmap[key] = record
        schema.append(_schema_entry(detail_path, all_columns, detail_rows_seen))
        if len(detailmap) < detail_rows_seen:
            logging.warning(
                'DBLP author-detail rows outside canonical arXiv author roster: %s',
                detail_rows_seen - len(detailmap),
            )
    else:
        logging.warning('Optional Fetch file absent: %s', detail_path)
        schema.append(dict(file=str(detail_path), present=False))

    # Conflicts are streamed as well. Only conflicts for this target paper and
    # author roster are retained; all columns for those relevant audit rows are
    # preserved in fetch_conflict_audit_json.
    conflict_path = dbdir / 'dblp_identity_conflicts.csv'
    auditconf = defaultdict(list)
    if conflict_path.is_file():
        sources[str(conflict_path.resolve())] = sha256(conflict_path)
        conflict_columns = _header(conflict_path)
        if {'arxiv_id', 'author_position'} <= set(conflict_columns):
            conflict_rows_seen = 0
            for chunk in _iter_csv(conflict_path):
                conflict_rows_seen += len(chunk)
                if 'dblp_record_key' in chunk.columns:
                    for aid, key in chunk[['arxiv_id', 'dblp_record_key']].itertuples(index=False, name=None):
                        add(aid, key, conflict_path.name)
                paper_mask = chunk.arxiv_id.isin(paper_ids)
                pos_values = pd.to_numeric(chunk.author_position, errors='coerce')
                roster_mask = [
                    bool(is_paper) and pd.notna(pos) and (aid, int(pos)) in author_keys
                    for aid, pos, is_paper in zip(chunk.arxiv_id, pos_values, paper_mask)
                ]
                relevant = chunk.loc[roster_mask]
                for record in relevant.to_dict('records'):
                    key = (record['arxiv_id'], int(float(record['author_position'])))
                    auditconf[key].append(record)
            schema.append(_schema_entry(conflict_path, conflict_columns, conflict_rows_seen))
        else:
            schema.append(_schema_entry(conflict_path, conflict_columns, 0))
    else:
        logging.warning('Optional Fetch file absent: %s', conflict_path)
        schema.append(dict(file=str(conflict_path), present=False))

    # Other optional audit files are needed only for focal record-key hints.
    # Read only those columns, never their wide evidence payloads.
    skip = {'dblp_author_features_by_author.csv', 'dblp_identity_conflicts.csv'}
    for name in OPTIONAL_CSVS:
        if name in skip:
            continue
        path = dbdir / name
        if not path.is_file():
            logging.warning('Optional Fetch file absent: %s', path)
            schema.append(dict(file=str(path), present=False))
            continue
        sources[str(path.resolve())] = sha256(path)
        columns = _header(path)
        if set(FOCAL_COLUMNS).issubset(columns):
            nrows = _stream_focal_keys(path, columns, years, focal, focal_audit, name)
        else:
            # The file is source-tracked but does not contribute focal keys.
            nrows = 0
        schema.append(_schema_entry(path, columns, nrows))

    metadata_path = dbdir / 'dblp_source_metadata.json'
    if metadata_path.is_file():
        sources[str(metadata_path.resolve())] = sha256(metadata_path)
        json.loads(metadata_path.read_text(encoding='utf-8'))
        schema.append(dict(file=str(metadata_path), present=True, type='json'))
    else:
        logging.warning('Optional Fetch file absent: %s', metadata_path)
        schema.append(dict(file=str(metadata_path), present=False))

    # Build compact in-memory records. Large history/evidence columns are not
    # copied into Python objects; the original full CSV remains hash-tracked.
    papers = master.to_dict('records')
    counts = authors.groupby('arxiv_id').size()
    expected = {}
    from .rules import normalize_title
    for paper in papers:
        aid = paper['arxiv_id']
        paper['t0_year'] = int(years[aid])
        paper['_normalized_title'] = normalize_title(paper['title_v1'])
        raw = paper.get('author_count_v1') or paper.get('arxiv_author_count')
        expected[aid] = int(float(raw)) if raw else int(counts[aid])
        paper['expected_author_count'] = expected[aid]

    rows, diagnosis = [], []
    for author in authors.to_dict('records'):
        aid = author['arxiv_id']
        pos = int(author['author_position'])
        old = detailmap.get((aid, pos), {})
        name = author['author_name_raw']
        oid = f'{aid}:{pos}'
        if old.get('t0_year') and int(float(old['t0_year'])) != years[aid]:
            raise ValueError('Fetch author t0 differs from arXiv v1 timestamp')
        claimed = truth(old.get('dblp_author_matched_to_paper')) and bool(old.get('dblp_person_id'))
        flags = []
        if re.search(r'[a-z][A-Z]', name): flags.append('MISSING_SPACE_OR_MERGE')
        if re.search(r'\b[A-Z]\s+[a-z]{2,}\b', name): flags.append('BROKEN_NAME_TOKEN')
        if re.search(r'[A-Z]{3,}', name): flags.append('UPPERCASE_RUN')
        if any(x in name for x in ('\ufffd', 'Ã', 'Â', ';', '|', '&')): flags.append('PUNCTUATION_OR_ENCODING')
        if old.get('arxiv_author_name_v1') and normalize_name(name) != normalize_name(old['arxiv_author_name_v1']):
            flags.append('FETCH_NAME_DISAGREEMENT')
        record = dict(
            arxiv_id=aid, author_position=pos, original_name=name, repaired_name=name,
            occurrence_id=oid, source_occurrences_json=json.dumps([oid]),
            source_positions_json=json.dumps([pos]), t0_year=int(years[aid]),
            identity_status='UNRESOLVED', dblp_person_id='', dblp_identity_method='',
            dblp_identity_confidence='UNRESOLVED', dblp_author_matched_to_paper=False,
            source_identity_status='FETCH_MATCHED' if claimed else old.get('dblp_profile_status', 'UNRESOLVED'),
            source_pid=old.get('dblp_person_id') or old.get('dblp_identity_conflict_source_person_id', ''),
            fetch_claimed_verified=claimed,
            fetch_identity_confidence=old.get('dblp_identity_confidence', ''),
            fetch_identity_conflict=old.get('dblp_identity_conflict_status', ''),
            fetch_source_evidence_json=json.dumps(old, ensure_ascii=False),
            fetch_conflict_audit_json=json.dumps(auditconf[aid, pos], ensure_ascii=False),
            focal_record_keys_json=json.dumps(sorted(focal[aid])), evidence_json='{}',
            expected_author_count=expected[aid], name_diagnosis=';'.join(flags),
        )
        rows.append(record)
        if flags:
            diagnosis.append(record)

    write_json(cfg, 'report/fetch_schema_audit.json', schema)
    write_csv(cfg, 'diagnosis/suspicious_author_names.csv', diagnosis, list(rows[0]))
    write_csv(cfg, 'report/bootstrap_focal_candidates.csv', focal_audit,
              ['arxiv_id', 'record_key', 'source', 'reason'])
    write_csv(cfg, 'matching/bootstrap_author_occurrences.csv', rows)
    compact_detail = pd.DataFrame(list(detailmap.values()), columns=detail_columns)
    return papers, rows, master, compact_detail, sources


def revalidate(cfg, index, rows):
    audit = []
    for row in rows:
        old = row['source_pid']
        canonical = (index.canonical(old) or index.canonical(clean_pid(old))) if old else None
        ids, method, _ = index.tier(row['original_name'])
        relation = False
        if canonical:
            relation = any(compatible(row['original_name'], alias) for alias in index.aliases(canonical))
        claimed = row['fetch_claimed_verified']
        conflict = row['fetch_identity_conflict']
        evidence = json.loads(row['fetch_source_evidence_json'])
        conflict_clear = False
        if conflict and canonical:
            prior = [evidence.get('dblp_identity_conflict_source_person_id'), evidence.get('dblp_identity_conflict_target_person_id')]
            conflict_clear = all(x and index.canonical(clean_pid(x)) == canonical for x in prior)
        supported = canonical and ((canonical in ids and method != 'UNIQUE_INITIAL_SURNAME_ALIAS') or relation)
        reliable_prior = claimed and row['fetch_identity_confidence'] in ('HIGH', 'MEDIUM') and supported and (not conflict or conflict_clear)
        if reliable_prior:
            row.update(identity_status='VERIFIED', dblp_person_id=canonical,
                       dblp_author_matched_to_paper=True, dblp_identity_method='FETCH_MATCH_PROFILE_REVALIDATED',
                       dblp_identity_confidence=row['fetch_identity_confidence'])
            reason = 'CANONICAL_PID_AND_ALIAS_SUPPORT_EXISTING_FETCH_MATCH'
        elif claimed:
            reason = ('INVALID_PROFILE' if not canonical else
                      'EXPLICIT_FETCH_IDENTITY_CONFLICT' if conflict and not conflict_clear else
                      'PROFILE_NAME_CONTRADICTION' if not supported else 'INSUFFICIENT_FETCH_CONFIDENCE')
        else:
            reason = 'TARGET_PAPER_OR_HISTORICAL_EVIDENCE_REQUIRED'
        row['revalidation_status'] = reason
        row['evidence_json'] = json.dumps(dict(
            rule=reason, initial_pid=old, canonical_pid=canonical,
            profile_method=method, source_candidate_count=len(ids),
            compatible_alias=relation,
            fetch_target_record=evidence.get('dblp_record_key', ''),
            redirect_resolved=bool(old and canonical and old != canonical),
        ))
        audit.append(dict(
            arxiv_id=row['arxiv_id'], author_position=row['author_position'],
            original_name=row['original_name'], initial_pid=old, canonical_pid=canonical or '',
            fetch_claimed_verified=claimed, final_status=row['identity_status'],
            reason=reason, evidence_json=row['evidence_json'],
        ))
    duplicates = Counter((r['arxiv_id'], r['dblp_person_id']) for r in rows if r['identity_status'] == 'VERIFIED')
    for row, item in zip(rows, audit):
        if row['identity_status'] == 'VERIFIED' and duplicates[row['arxiv_id'], row['dblp_person_id']] > 1:
            row.update(identity_status='UNRESOLVED', dblp_person_id='', dblp_author_matched_to_paper=False,
                       dblp_identity_confidence='UNRESOLVED', revalidation_status='DUPLICATE_PID_IN_FETCH_TEAM')
            item.update(final_status='UNRESOLVED', reason=row['revalidation_status'])
    write_csv(cfg, 'matching/fetch_identity_revalidation.csv', audit)
    return rows

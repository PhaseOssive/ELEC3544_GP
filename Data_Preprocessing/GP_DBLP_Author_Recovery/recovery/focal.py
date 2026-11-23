"""Independent P0 audit: target exclusion never depends on identity confidence."""
def audit(store):
    db=store.db
    missing=db.execute('SELECT COUNT(*) FROM targets t LEFT JOIN focal f ON f.aid=t.aid AND f.key=t.key WHERE f.key IS NULL').fetchone()[0]
    leakage=db.execute('''SELECT COUNT(*) FROM candidate_coauthors c JOIN occurrences o ON o.oid=c.oid
    WHERE EXISTS (SELECT 1 FROM focal f LEFT JOIN record_works r ON r.key=f.key WHERE f.aid=o.aid AND (f.key=c.key OR r.work=c.work))''').fetchone()[0]
    future=db.execute('SELECT COUNT(*) FROM candidate_coauthors c JOIN occurrences o ON o.oid=c.oid WHERE c.year>=o.t0').fetchone()[0]
    report=dict(target_candidates_missing_exclusion=missing,coauthor_focal_leakage_count=leakage,coauthor_same_or_future_year_count=future,unreliable_target_candidates_excluded=db.execute('SELECT COUNT(*) FROM targets t JOIN focal f ON f.aid=t.aid AND f.key=t.key WHERE t.reliable=0').fetchone()[0],identity_reliability_independent_of_focal_exclusion=True)
    if missing or leakage or future:raise AssertionError('P0/temporal exclusion invariant failed: '+str(report))
    return report

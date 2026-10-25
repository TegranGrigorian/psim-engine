import pandas as pd

from psim_engine.cache import QueryCache, normalize_keys
from psim_engine.parallel import balanced_chunks
from psim_engine.queries import QUERY_KEY
from psim_engine.run import choose_sample


def keys(*rows) -> pd.DataFrame:
    return pd.DataFrame(
        [dict(date="2026-11-10", depart_time=t, origin_lat=o, origin_lon=-83.7,
              dest_lat=d, dest_lon=-83.6) for t, o, d in rows]
    )


def totals_rows(k: pd.DataFrame, minutes: float) -> pd.DataFrame:
    return k.assign(completed=True, walk_only=False, total_minutes=minutes)


def test_empty_cache_misses_everything_and_dedupes(tmp_path):
    cache = QueryCache(tmp_path)
    k = keys(("08:00", 42.1, 42.2), ("08:00", 42.1, 42.2), ("08:05", 42.1, 42.2))
    assert len(cache.missing_totals(k)) == 2


def test_cached_queries_are_not_routed_again_and_survive_a_save(tmp_path):
    cache = QueryCache(tmp_path)
    first = keys(("08:00", 42.1, 42.2))
    cache.add_totals(totals_rows(first, 30.0))
    cache.save()
    reopened = QueryCache(tmp_path)
    both = keys(("08:00", 42.1, 42.2), ("09:00", 42.1, 42.2))
    missing = reopened.missing_totals(both)
    assert list(missing.depart_time) == ["09:00"]


def test_float_noise_below_six_decimals_does_not_cause_a_miss(tmp_path):
    cache = QueryCache(tmp_path)
    cache.add_totals(totals_rows(keys(("08:00", 42.2765, 42.2)), 12.0))
    again = keys(("08:00", 42.27650000001, 42.2))
    assert len(cache.missing_totals(again)) == 0


def test_totals_for_returns_each_runs_query_ids(tmp_path):
    cache = QueryCache(tmp_path)
    k = keys(("08:00", 42.1, 42.2), ("09:00", 42.1, 42.2))
    cache.add_totals(totals_rows(k, 20.0).assign(total_minutes=[20.0, 25.0]))
    run = k.assign(query_id=["q1", "q2"])
    got = cache.totals_for(run).set_index("query_id")
    assert got.loc["q1", "total_minutes"] == 20.0 and got.loc["q2", "total_minutes"] == 25.0


def test_breakdown_for_only_returns_queries_that_have_one(tmp_path):
    cache = QueryCache(tmp_path)
    k = keys(("08:00", 42.1, 42.2), ("09:00", 42.1, 42.2))
    cache.add_breakdown(k.iloc[:1].assign(
        breakdown_completed=True, breakdown_total_minutes=21.0, wait_minutes=5.0,
        in_vehicle_minutes=10.0, walk_minutes=6.0, transfers=0))
    got = cache.breakdown_for(k.assign(query_id=["q1", "q2"]))
    assert list(got.query_id) == ["q1"]


def test_balanced_chunks_keep_departure_groups_together_and_lose_nothing():
    rows = []
    for t, n in [("08:00", 9), ("08:05", 5), ("08:10", 4), ("08:15", 3), ("08:20", 1)]:
        rows += [dict(date="2026-11-10", depart_time=t, i=i) for i in range(n)]
    df = pd.DataFrame(rows)
    chunks = balanced_chunks(df, 3)
    assert sum(len(c) for c in chunks) == len(df)
    times = [set(c.depart_time) for c in chunks]
    assert all(a.isdisjoint(b) for i, a in enumerate(times) for b in times[i + 1:])
    sizes = sorted(len(c) for c in chunks)
    assert sizes[-1] <= 9 and len(chunks) == 3


def test_sample_is_seeded_sorted_by_content_and_ignores_query_numbering():
    k = keys(*[(f"08:{m:02d}", 42.1, 42.2) for m in range(0, 50, 5)])
    a = k.assign(query_id=[f"q{i}" for i in range(len(k))])
    b = a.sample(frac=1, random_state=1).assign(query_id=lambda d: [f"z{i}" for i in range(len(d))])
    totals = lambda q: q[["query_id"]].assign(completed=True)
    s1 = choose_sample(a, totals(a), 4, seed=42)
    s2 = choose_sample(b, totals(b), 4, seed=42)
    assert list(s1.depart_time) == list(s2.depart_time) and len(s1) == 4
    s3 = choose_sample(a, totals(a), 4, seed=43)
    assert list(s3.depart_time) != list(s1.depart_time)


def test_sample_only_draws_completed_queries():
    k = keys(("08:00", 42.1, 42.2), ("08:05", 42.1, 42.2), ("08:10", 42.1, 42.2))
    q = k.assign(query_id=["a", "b", "c"])
    totals = pd.DataFrame({"query_id": ["a", "b", "c"], "completed": [True, False, True]})
    s = choose_sample(q, totals, 10, seed=1)
    assert set(s.depart_time) == {"08:00", "08:10"}

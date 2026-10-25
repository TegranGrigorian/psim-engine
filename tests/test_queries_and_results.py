import datetime as dt
from pathlib import Path

import pandas as pd

from psim_engine.config import CityConfig
from psim_engine.queries import build_queries
from psim_engine.run import RESULT_COLUMNS, assemble_results


def cfg() -> CityConfig:
    return CityConfig(
        city_dir=Path("/x"), city_slug="x", timezone="America/Detroit",
        service_date=dt.date(2026, 11, 10), weekend_service_date=dt.date(2026, 11, 14),
        departure_window_minutes=30, max_walk_minutes=20, max_trip_minutes=120,
        osm=Path("/x/osm.pbf"), gtfs={"baseline": []},
    )


def trips() -> pd.DataFrame:
    row = dict(persona_id="p1", depart_time="08:00", origin_lat=42.0, origin_lon=-83.0, dest_lat=42.1, dest_lon=-83.1)
    return pd.DataFrame([
        {**row, "trip_id": "t1", "day": "mon"},
        {**row, "trip_id": "t2", "day": "tue"},   # same commute, same service date
        {**row, "trip_id": "t3", "day": "sat"},   # weekend date: a different query
        {**row, "trip_id": "t4", "day": "mon", "depart_time": "08:05"},
    ])


def test_weekday_repeats_collapse_and_saturday_is_separate():
    queries, trip_query = build_queries(trips(), cfg())
    assert len(queries) == 3
    q = list(trip_query)  # t1, t2, t3, t4
    assert q[0] == q[1], "mon and tue commutes share the weekday service date"
    assert len({q[0], q[2], q[3]}) == 3, "saturday and a different time are separate queries"
    assert set(queries.date) == {"2026-11-10", "2026-11-14"}
    assert list(queries.query_id) == ["q000000", "q000001", "q000002"]


def test_results_copy_outcomes_to_every_trip_and_flag_failures():
    t = trips()
    queries, trip_query = build_queries(t, cfg())
    # queries in first-appearance order: q0 = weekday 08:00, q1 = saturday 08:00, q2 = weekday 08:05
    totals = pd.DataFrame({
        "query_id": queries.query_id,
        "completed": [True, False, True],
        "walk_only": [False, False, True],
        "total_minutes": [30.0, None, 9.0],
    })
    breakdown = pd.DataFrame({
        "query_id": [queries.query_id[0]], "breakdown_completed": [True],
        "breakdown_total_minutes": [31.0], "wait_minutes": [6.0], "in_vehicle_minutes": [14.0],
        "walk_minutes": [10.0], "transfers": [1],
    })
    res = assemble_results("baseline_s0042", t, trip_query, totals, breakdown)
    assert list(res.columns) == RESULT_COLUMNS
    r = res.set_index("trip_id")

    # t1 and t2 share a query, so both carry its totals and its breakdown
    for trip in ("t1", "t2"):
        assert r.loc[trip, "completed"] and r.loc[trip, "total_minutes"] == 30.0
        assert r.loc[trip, "breakdown_sampled"] and r.loc[trip, "wait_minutes"] == 6.0
        assert r.loc[trip, "transfers"] == 1 and r.loc[trip, "failure_reason"] == "none"

    # t3 failed: no times, no breakdown, reason is no_route
    assert not r.loc["t3", "completed"] and pd.isna(r.loc["t3", "total_minutes"])
    assert r.loc["t3", "failure_reason"] == "no_route" and not r.loc["t3", "breakdown_sampled"]

    # t4 was walked, and not sampled for the breakdown
    assert r.loc["t4", "walk_only"] and r.loc["t4", "total_minutes"] == 9.0
    assert pd.isna(r.loc["t4", "wait_minutes"]) and not r.loc["t4", "breakdown_sampled"]
    assert (res.routing_percentile == 50).all()

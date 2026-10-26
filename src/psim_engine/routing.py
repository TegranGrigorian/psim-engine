"""Routing with r5py.

Two passes, because r5py's detailed itineraries are slow (about 1.3 s per
origin-destination pair; the time is spent in r5py's Python layer) while its
travel time matrix is fast:

1. Every unique query: total door-to-door minutes at the 50th percentile over
   the departure window, from the travel time matrix. Run twice, transit+walk
   and walk-only, so we can tell a walked trip from a transit trip.
2. A seeded random sample of queries: full breakdown (wait, in-vehicle, walk,
   transfers) from detailed itineraries.
"""

import datetime as dt
import json
import zlib

import geopandas as gpd
import numpy as np
import pandas as pd
import r5py
from shapely.geometry import Point

from .config import CityConfig
from .planning import latest_departure
from .reliability import Leg, Schedule, simulate_delays, summarize_delays
from .summarize import query_outcomes

TRANSIT_MODES = [r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK]
WALK_MODES = [r5py.TransportMode.WALK]
MAX_GROUP = 150  # bounds the all-to-all matrix (group^2 pairs) per call


def _points(lat: pd.Series, lon: pd.Series, ids: list[str]) -> gpd.GeoDataFrame:
    geometry = [Point(x, y) for x, y in zip(lon.astype(float), lat.astype(float))]
    return gpd.GeoDataFrame({"id": ids}, geometry=geometry, crs="EPSG:4326")


def _departure(date: str, depart_time: str) -> dt.datetime:
    hours, minutes = depart_time.split(":")
    d = dt.date.fromisoformat(date)
    return dt.datetime(d.year, d.month, d.day, int(hours), int(minutes))


def _limits(cfg: CityConfig) -> dict:
    return dict(
        departure_time_window=dt.timedelta(minutes=cfg.departure_window_minutes),
        max_time=dt.timedelta(minutes=cfg.max_trip_minutes),
        max_time_walking=dt.timedelta(minutes=cfg.max_walk_minutes),
    )


def _pairwise_minutes(net, group: pd.DataFrame, departure: dt.datetime, modes, cfg) -> pd.Series:
    """50th-percentile minutes for each row's origin -> destination (NaN if none)."""
    ids = list(group.query_id)
    origins = _points(group.origin_lat, group.origin_lon, [f"o{q}" for q in ids])
    dests = _points(group.dest_lat, group.dest_lon, [f"d{q}" for q in ids])
    matrix = r5py.TravelTimeMatrix(
        net, origins=origins, destinations=dests, departure=departure,
        percentiles=[50], transport_modes=modes, **_limits(cfg),
    )
    col = "travel_time" if "travel_time" in matrix.columns else matrix.columns[-1]
    lookup = {(f, t): v for f, t, v in zip(matrix.from_id, matrix.to_id, matrix[col])}
    return pd.Series([lookup.get((f"o{q}", f"d{q}")) for q in ids], index=group.index, dtype="float64")


def route_totals(net, queries: pd.DataFrame, cfg: CityConfig, log=print) -> pd.DataFrame:
    """Pass 1. One row per query: completed, walk_only, total_minutes."""
    parts = []
    groups = list(queries.groupby(["date", "depart_time"], sort=True))
    for n, ((date, depart_time), group) in enumerate(groups, start=1):
        departure = _departure(date, depart_time)
        for start in range(0, len(group), MAX_GROUP):
            chunk = group.iloc[start:start + MAX_GROUP]
            transit = _pairwise_minutes(net, chunk, departure, TRANSIT_MODES, cfg)
            walk = _pairwise_minutes(net, chunk, departure, WALK_MODES, cfg)
            best = pd.concat([transit, walk], axis=1).min(axis=1)  # NaN-skipping
            parts.append(pd.DataFrame({
                "query_id": chunk.query_id,
                "completed": best.notna(),
                "walk_only": walk.notna() & (transit.isna() | (walk <= transit)),
                "total_minutes": best,
            }))
        if n % 20 == 0 or n == len(groups):
            log(f"  routed {n}/{len(groups)} departure times")
    out = pd.concat(parts, ignore_index=True)
    out["walk_only"] = out["walk_only"] & out["completed"]
    return out


PLAN_COLUMNS = ["planned_departure", "scheduled_minutes", "p_on_time", "delay_p95_minutes"]

_SCHEDULES: dict[tuple[str, str], Schedule] = {}


def _schedule(cfg: CityConfig, feed: str, date: str) -> Schedule:
    """Timetable of one feed on one date, built once per worker (cleaned feed zips)."""
    key = (feed, date)
    if key not in _SCHEDULES:
        zip_path = cfg.city_dir / "raw" / "gtfs_clean" / f"{feed}.zip"
        _SCHEDULES[key] = Schedule.from_gtfs(zip_path, dt.date.fromisoformat(date))
    return _SCHEDULES[key]


def _detail_plain(net, todo: pd.DataFrame, cfg: CityConfig) -> pd.DataFrame:
    """Detailed itinerary closest to each query's matrix total, leaving at its nominal time."""
    frames = []
    for (date, depart_time), group in todo.groupby(["date", "depart_time"], sort=True):
        origins = _points(group.origin_lat, group.origin_lon, list(group.query_id))
        dests = _points(group.dest_lat, group.dest_lon, list(group.query_id))
        itineraries = r5py.DetailedItineraries(
            net, origins=origins, destinations=dests,
            departure=_departure(date, depart_time), transport_modes=TRANSIT_MODES, **_limits(cfg),
        )
        frames.append(pd.DataFrame(itineraries.drop(columns="geometry")))
    segments = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["from_id", "option", "segment", "transport_mode", "departure_time", "travel_time"])
    targets = pd.DataFrame({
        "query_id": todo.query_id.values,
        "departure": [_departure(d, t) for d, t in zip(todo.date, todo.depart_time)],
        "target_minutes": todo.target_minutes.values,
    }).sort_values("query_id")
    out = query_outcomes(segments, targets)
    out = out.rename(columns={"completed": "breakdown_completed", "total_minutes": "breakdown_total_minutes"})
    for c in PLAN_COLUMNS:
        out[c] = None
    return out


def _hhmm(minutes: float) -> str:
    return f"{int(minutes // 60) % 24:02d}:{int(minutes % 60):02d}"


def _pair_minutes(net, row, departure: dt.datetime, cfg: CityConfig) -> float | None:
    """Door-to-door minutes when leaving exactly at `departure` (a one-minute window)."""
    matrix = r5py.TravelTimeMatrix(
        net,
        origins=_points(pd.Series([row.origin_lat]), pd.Series([row.origin_lon]), ["o"]),
        destinations=_points(pd.Series([row.dest_lat]), pd.Series([row.dest_lon]), ["d"]),
        departure=departure, percentiles=[50], transport_modes=TRANSIT_MODES,
        **{**_limits(cfg), "departure_time_window": dt.timedelta(minutes=1)},
    )
    col = "travel_time" if "travel_time" in matrix.columns else matrix.columns[-1]
    value = matrix[col].iloc[0]
    return None if pd.isna(value) else float(value)


def _failed_plan(query_id: str, planned: str | None = None) -> dict:
    return {"query_id": query_id, "breakdown_completed": False, "breakdown_total_minutes": None,
            "wait_minutes": None, "in_vehicle_minutes": None, "walk_minutes": None, "transfers": None,
            "planned_departure": planned, "scheduled_minutes": None,
            # a trip that cannot be planned to arrive in time is not on time
            "p_on_time": 0.0, "delay_p95_minutes": None}


def _plan_one(net, row, cfg: CityConfig) -> dict:
    """Plan arrive-by, route the planned departure in detail, and simulate delays."""
    rel = cfg.reliability
    date = dt.date.fromisoformat(row.date)
    midnight = dt.datetime(date.year, date.month, date.day)
    h, m = row.due_time.split(":")
    due = int(h) * 60 + int(m)

    plan = latest_departure(
        lambda d: _pair_minutes(net, row, midnight + dt.timedelta(minutes=d), cfg),
        due_minute=due, margin=rel.arrive_margin_minutes,
    )
    if plan is None:
        return _failed_plan(row.query_id)
    depart, scheduled = plan
    departure = midnight + dt.timedelta(minutes=depart)

    itineraries = r5py.DetailedItineraries(
        net,
        origins=_points(pd.Series([row.origin_lat]), pd.Series([row.origin_lon]), [row.query_id]),
        destinations=_points(pd.Series([row.dest_lat]), pd.Series([row.dest_lon]), [row.query_id]),
        departure=departure, transport_modes=TRANSIT_MODES,
        **{**_limits(cfg), "departure_time_window": dt.timedelta(minutes=1)},
    )
    segments = pd.DataFrame(itineraries.drop(columns="geometry"))
    targets = pd.DataFrame({"query_id": [row.query_id], "departure": [pd.Timestamp(departure)],
                            "target_minutes": [scheduled]})
    out = query_outcomes(segments, targets, with_legs=True).iloc[0]
    if not out.completed:
        return _failed_plan(row.query_id, _hhmm(depart))

    legs = [Leg(**leg) for leg in json.loads(out.legs_json)]
    schedules = {leg.feed: _schedule(cfg, leg.feed, row.date) for leg in legs}
    arrival = depart + out.total_minutes  # minutes after midnight, scheduled
    slack = max(0.0, (due - rel.arrive_margin_minutes) - arrival)
    seed = zlib.crc32(f"{row.date}|{row.due_time}|{row.origin_lat}|{row.origin_lon}|{row.dest_lat}|{row.dest_lon}".encode())
    delays = simulate_delays(legs, schedules, rel, np.random.default_rng(seed))
    p_on_time, p95 = summarize_delays(delays, slack, rel)
    return {
        "query_id": row.query_id, "breakdown_completed": True, "breakdown_total_minutes": out.total_minutes,
        "wait_minutes": out.wait_minutes, "in_vehicle_minutes": out.in_vehicle_minutes,
        "walk_minutes": out.walk_minutes, "transfers": out.transfers,
        "planned_departure": _hhmm(depart), "scheduled_minutes": scheduled,
        "p_on_time": p_on_time, "delay_p95_minutes": p95,
    }


def route_detail(net, todo: pd.DataFrame, cfg: CityConfig) -> pd.DataFrame:
    """Pass 2 for already-chosen queries.

    `todo`: query_id, date, depart_time, coordinates, due_time, target_minutes (the matrix
    total). Queries with a start time (`due_time`) are planned: the rider leaves as late as
    the timetable allows and still arrives on time, and the delay model (when the config has
    one) says how often that plan fails. The others leave at their nominal time and get the
    single detailed option closest to the matrix total (see summarize.py). Returns one row per
    query_id, including queries r5py found no option for (breakdown_completed False).
    """
    planned_mask = (todo.due_time != "") & (cfg.reliability is not None)
    parts = []
    if (~planned_mask).any():
        parts.append(_detail_plain(net, todo[~planned_mask], cfg))
    for row in todo[planned_mask].itertuples(index=False):
        parts.append(pd.DataFrame([_plan_one(net, row, cfg)]))
    out = pd.concat(parts, ignore_index=True)
    out["transfers"] = out["transfers"].astype("Int64")
    return out

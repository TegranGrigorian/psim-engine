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
import random

import geopandas as gpd
import pandas as pd
import r5py
from shapely.geometry import Point

from .config import CityConfig
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


def route_breakdown(net, queries: pd.DataFrame, totals: pd.DataFrame, cfg: CityConfig,
                    sample: int, seed: int, log=print) -> pd.DataFrame:
    """Pass 2. Detailed itineraries for a seeded random sample of completed queries.

    Each sampled query gets the single option closest to its matrix total (see
    summarize.py), plus `breakdown_total_minutes`'s gap to that total.
    """
    done_ids = sorted(totals[totals.completed].query_id)
    chosen = set(random.Random(seed).sample(done_ids, min(sample, len(done_ids))))
    todo = queries[queries.query_id.isin(chosen)]
    frames = []
    done = 0
    for (date, depart_time), group in todo.groupby(["date", "depart_time"], sort=True):
        origins = _points(group.origin_lat, group.origin_lon, list(group.query_id))
        dests = _points(group.dest_lat, group.dest_lon, list(group.query_id))
        itineraries = r5py.DetailedItineraries(
            net, origins=origins, destinations=dests,
            departure=_departure(date, depart_time), transport_modes=TRANSIT_MODES, **_limits(cfg),
        )
        frames.append(pd.DataFrame(itineraries.drop(columns="geometry")))
        done += len(group)
        if done % 25 < len(group):
            log(f"  breakdown {done}/{len(todo)} queries")
    segments = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["from_id", "option", "segment", "transport_mode", "departure_time", "travel_time"])
    targets = todo.merge(totals[["query_id", "total_minutes"]], on="query_id")
    targets = pd.DataFrame({
        "query_id": targets.query_id,
        "departure": [_departure(d, t) for d, t in zip(targets.date, targets.depart_time)],
        "target_minutes": targets.total_minutes,
    }).sort_values("query_id")
    out = query_outcomes(segments, targets)
    return out.rename(columns={"completed": "breakdown_completed",
                               "total_minutes": "breakdown_total_minutes"})

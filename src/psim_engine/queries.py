"""Turns trips.csv rows into unique routing queries.

Weekday trips (mon..fri) all route on the same service date, so a worker's five
identical commutes collapse into one query. A query is (date, departure time,
origin, destination); results are copied back onto every trip that shares it.
"""

import pandas as pd

from .config import CityConfig

QUERY_KEY = ["date", "depart_time", "origin_lat", "origin_lon", "dest_lat", "dest_lon", "due_time"]


def build_queries(trips: pd.DataFrame, cfg: CityConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (unique queries with a `query_id`, per-trip `query_id` Series).

    The Series is indexed like `trips`, so `trips.assign(query_id=series)` works.
    """
    keyed = trips[["depart_time", "origin_lat", "origin_lon", "dest_lat", "dest_lon"]].copy()
    # start time of the class or shift, "" for trips without one (older trips.csv have no column)
    keyed["due_time"] = trips["due_time"].fillna("").astype(str) if "due_time" in trips else ""
    keyed["date"] = trips["day"].map(lambda d: cfg.date_for_day(d).isoformat())
    # Row order of first appearance, so ids are stable for a given trips.csv.
    unique = keyed.drop_duplicates(QUERY_KEY).reset_index(drop=True)
    unique["query_id"] = [f"q{i:06d}" for i in range(len(unique))]
    merged = keyed.merge(unique, on=QUERY_KEY, how="left", validate="many_to_one")
    return unique[["query_id", *QUERY_KEY]], pd.Series(merged["query_id"].values, index=trips.index)

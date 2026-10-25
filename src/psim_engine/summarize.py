"""Turns r5py DetailedItineraries output into a trip breakdown.

r5py returns many *options* per origin-destination pair, with repeats, and some
of them are paths a traveler would never take (on a sparse evening query, a
cluster of options took 92 min while the travel time matrix's 90th percentile
was 53 min). So detailed options are not a distribution, and we do not average
them. The travel time matrix gives the authoritative total; here we pick the
single option whose elapsed time is closest to it and report that option's
components, so the parts add up to a total close to the real one.

For each option:

- total_minutes: arrival minus the *requested* departure, so it includes the
  wait at the origin. A direct-walk option has no schedule; its total is the
  walking time.
- walk_minutes / in_vehicle_minutes: summed segment travel times by mode.
- wait_minutes: total - walk - in_vehicle (initial wait, transfer waits and
  boarding slack). r5py's own per-segment `wait_time` does not add up to
  elapsed time, so we derive it.
- transfers: transit segments - 1.
"""

import pandas as pd

OUTCOME_COLUMNS = [
    "query_id",
    "completed",
    "total_minutes",
    "wait_minutes",
    "in_vehicle_minutes",
    "walk_minutes",
    "transfers",
]


def _minutes(td: pd.Series) -> pd.Series:
    return td.dt.total_seconds() / 60.0


def option_metrics(segments: pd.DataFrame, departures: pd.Series) -> pd.DataFrame:
    """One row per (query, option).

    `segments`: from_id, option, segment, transport_mode, departure_time, travel_time.
    `departures`: requested departure (Timestamp), indexed by query id.
    """
    seg = segments.copy()
    seg["is_walk"] = seg["transport_mode"].astype(str).str.upper().str.endswith("WALK")
    seg["minutes"] = _minutes(seg["travel_time"])
    seg["end_time"] = seg["departure_time"] + seg["travel_time"]

    keys = ["from_id", "option"]
    grouped = seg.groupby(keys, sort=True)
    index = grouped.size().index
    out = pd.DataFrame(
        {
            "walk_minutes": seg[seg.is_walk].groupby(keys)["minutes"].sum(),
            "in_vehicle_minutes": seg[~seg.is_walk].groupby(keys)["minutes"].sum(),
            "n_transit": seg[~seg.is_walk].groupby(keys)["minutes"].count(),
            "start": grouped["departure_time"].min(),
            "end": grouped["end_time"].max(),
            "sum_minutes": grouped["minutes"].sum(),
        }
    ).reindex(index)
    out["walk_minutes"] = out["walk_minutes"].fillna(0.0)
    out["in_vehicle_minutes"] = out["in_vehicle_minutes"].fillna(0.0)
    out["transfers"] = (out["n_transit"].fillna(0) - 1).clip(lower=0).astype(int)
    out = out.reset_index()
    requested = out["from_id"].map(departures)
    scheduled = _minutes(out["end"] - requested)
    out["total_minutes"] = scheduled.where(out["start"].notna(), out["sum_minutes"])
    out["wait_minutes"] = (
        out["total_minutes"] - out["walk_minutes"] - out["in_vehicle_minutes"]
    ).clip(lower=0.0)
    return out.drop(columns=["n_transit", "start", "end", "sum_minutes"])


def query_outcomes(
    segments: pd.DataFrame, targets: pd.DataFrame
) -> pd.DataFrame:
    """The representative option for each query in `targets`.

    `targets`: query_id, departure (Timestamp), target_minutes (the matrix total).
    Queries with no option come back with completed=False.
    """
    cols = ["total_minutes", "wait_minutes", "in_vehicle_minutes", "walk_minutes", "transfers"]
    chosen: dict[str, dict] = {}
    if len(segments):
        departures = targets.set_index("query_id")["departure"]
        goal = targets.set_index("query_id")["target_minutes"]
        opts = option_metrics(segments, departures)
        opts["gap"] = (opts["total_minutes"] - opts["from_id"].map(goal)).abs()
        # lowest gap, then lowest option number, so the choice is deterministic
        best = opts.sort_values(["from_id", "gap", "option"]).groupby("from_id", sort=False).head(1)
        for row in best.itertuples(index=False):
            chosen[str(row.from_id)] = {c: float(getattr(row, c)) for c in cols}
    rows = []
    for qid in targets["query_id"]:
        r = chosen.get(qid)
        rows.append({"query_id": qid, "completed": False} if r is None
                    else {"query_id": qid, "completed": True, **r})
    df = pd.DataFrame(rows)
    for c in OUTCOME_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df["transfers"] = df["transfers"].astype("Int64")
    return df[OUTCOME_COLUMNS]

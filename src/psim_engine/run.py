"""`psim-engine run`: route one generated run and write trip_results.parquet."""

import json
import time
from pathlib import Path

import pandas as pd

from .config import CityConfig
from .network import build_network
from .queries import build_queries
from .routing import route_breakdown, route_totals

RESULT_COLUMNS = [
    "run_id", "trip_id", "persona_id", "completed", "walk_only", "total_minutes",
    "wait_minutes", "in_vehicle_minutes", "walk_minutes", "transfers",
    "breakdown_total_minutes", "breakdown_sampled", "routing_percentile", "failure_reason",
]


def assemble_results(run_id: str, trips: pd.DataFrame, trip_query: pd.Series,
                     totals: pd.DataFrame, breakdown: pd.DataFrame) -> pd.DataFrame:
    """Copies per-query outcomes back onto every trip (pure pandas, no JVM)."""
    df = trips[["trip_id", "persona_id"]].assign(query_id=trip_query)
    df = df.merge(totals, on="query_id", how="left", validate="many_to_one")
    cols = ["breakdown_total_minutes", "wait_minutes", "in_vehicle_minutes", "walk_minutes", "transfers"]
    detail = breakdown[breakdown.breakdown_completed.fillna(False).astype(bool)][["query_id", *cols]]
    df = df.merge(detail, on="query_id", how="left", validate="many_to_one")
    df["breakdown_sampled"] = df.query_id.isin(set(detail.query_id))
    df["run_id"] = run_id
    df["routing_percentile"] = 50
    # r5py does not say *why* nothing was found, so every failure is `no_route`.
    df["failure_reason"] = df.completed.map({True: "none", False: "no_route"})
    df["transfers"] = df["transfers"].astype("Int64")
    return df[RESULT_COLUMNS]


def run(data_dir: Path, city: str, run_id: str, breakdown_sample: int, limit: int | None,
        log=print) -> Path:
    cfg = CityConfig.load(data_dir, city)
    run_dir = cfg.city_dir / "generated" / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    scenario, seed = manifest["scenario"], int(manifest["seed"])

    trips = pd.read_csv(run_dir / "trips.csv", dtype={"depart_time": str})
    if limit:
        trips = trips.head(limit)
    queries, trip_query = build_queries(trips, cfg)
    log(f"{run_id}: {len(trips)} trips -> {len(queries)} unique queries")

    t0 = time.time()
    net = build_network(cfg, scenario)
    log(f"network built in {time.time() - t0:.1f}s")

    t0 = time.time()
    totals = route_totals(net, queries, cfg, log)
    log(f"totals in {time.time() - t0:.1f}s")

    t0 = time.time()
    breakdown = route_breakdown(net, queries, totals, cfg, breakdown_sample, seed, log)
    log(f"breakdown in {time.time() - t0:.1f}s")

    results = assemble_results(run_id, trips, trip_query, totals, breakdown)
    out_dir = cfg.city_dir / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "trip_results.parquet"
    results.to_parquet(out, compression="snappy", index=False)
    sampled = results[results.breakdown_sampled].drop_duplicates("breakdown_total_minutes")
    if len(sampled):
        gap = (sampled.breakdown_total_minutes - sampled.total_minutes).abs()
        log(f"breakdown vs matrix total, {len(sampled)} sampled queries: "
            f"median gap {gap.median():.1f} min, max {gap.max():.1f} min")
    log(f"wrote {out} ({len(results)} rows, {results.completed.mean():.1%} completed)")
    return out

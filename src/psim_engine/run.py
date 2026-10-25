"""`psim-engine run`: route one generated run and write trip_results.parquet."""

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .cache import QueryCache, normalize_keys
from .config import CityConfig
from .parallel import Pool, _detail_task, _totals_task
from .queries import QUERY_KEY, build_queries

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


@dataclass
class LoadedRun:
    run_id: str
    scenario: str
    seed: int
    trips: pd.DataFrame
    queries: pd.DataFrame  # unique queries: query_id + QUERY_KEY
    trip_query: pd.Series  # query_id per trip


def find_runs(cfg: CityConfig, scenario: str) -> list[str]:
    """Every generated run folder of a scenario, e.g. baseline_s0042, baseline_s0043."""
    root = cfg.city_dir / "generated"
    return sorted(d.name for d in root.glob(f"{scenario}_s*") if (d / "manifest.json").exists())


def _load_run(cfg: CityConfig, run_id: str, limit: int | None) -> LoadedRun:
    run_dir = cfg.city_dir / "generated" / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    trips = pd.read_csv(run_dir / "trips.csv", dtype={"depart_time": str})
    if limit:
        trips = trips.head(limit)
    queries, trip_query = build_queries(trips, cfg)
    return LoadedRun(run_id, manifest["scenario"], int(manifest["seed"]), trips, queries, trip_query)


def choose_sample(queries: pd.DataFrame, totals: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Seeded random sample of a run's completed queries for the detailed breakdown.

    Sorted by query key first, so the sample depends only on the run's content
    and seed, never on query numbering or on what happens to be cached.
    """
    done = normalize_keys(queries).merge(totals[["query_id", "completed"]], on="query_id")
    done = done[done.completed.astype(bool)].sort_values(QUERY_KEY).reset_index(drop=True)
    picks = sorted(random.Random(seed).sample(range(len(done)), min(size, len(done))))
    return done.iloc[picks].drop(columns="completed").reset_index(drop=True)


def _route_scenario(cfg, scenario, runs, breakdown_sample, workers, memory, log) -> dict[str, pd.DataFrame]:
    cache = QueryCache.open(cfg, scenario)
    all_queries = pd.concat([r.queries[QUERY_KEY] for r in runs], ignore_index=True)
    pool: Pool | None = None

    def get_pool() -> Pool:
        nonlocal pool
        if pool is None:
            t0 = time.time()
            pool = Pool(cfg, scenario, workers, memory)
            log(f"  started {workers} routing workers in {time.time() - t0:.1f}s")
        return pool

    try:
        # Pass 1: totals for every query nobody has routed yet.
        unique = normalize_keys(all_queries).drop_duplicates(QUERY_KEY)
        missing = cache.missing_totals(all_queries)
        log(f"{scenario}: {len(unique)} unique queries, {len(unique) - len(missing)} cached, "
            f"{len(missing)} to route")
        if len(missing):
            t0 = time.time()
            missing["query_id"] = [f"m{i:07d}" for i in range(len(missing))]
            routed = get_pool().map(_totals_task, missing, "totals", log)
            cache.add_totals(missing.merge(routed, on="query_id")[QUERY_KEY + ["completed", "walk_only", "total_minutes"]])
            cache.save()
            log(f"  totals in {time.time() - t0:.1f}s")

        # Pass 2: detailed breakdown for each run's seeded sample, minus what is cached.
        run_totals = {r.run_id: cache.totals_for(r.queries) for r in runs}
        samples = {r.run_id: choose_sample(r.queries, run_totals[r.run_id], breakdown_sample, r.seed)
                   for r in runs}
        wanted = pd.concat([s[QUERY_KEY] for s in samples.values()], ignore_index=True)
        todo = cache.missing_breakdown(wanted)
        log(f"{scenario}: {len(normalize_keys(wanted).drop_duplicates(QUERY_KEY))} sampled queries, "
            f"{len(todo)} need a detailed route")
        if len(todo):
            t0 = time.time()
            todo = todo.merge(normalize_keys(cache.totals)[QUERY_KEY + ["total_minutes"]], on=QUERY_KEY)
            todo = todo.rename(columns={"total_minutes": "target_minutes"})
            todo["query_id"] = [f"d{i:07d}" for i in range(len(todo))]
            detail = get_pool().map(_detail_task, todo, "detail", log)
            cache.add_breakdown(todo.merge(detail, on="query_id")[QUERY_KEY + [
                "breakdown_completed", "breakdown_total_minutes", "wait_minutes",
                "in_vehicle_minutes", "walk_minutes", "transfers"]])
            cache.save()
            log(f"  detail in {time.time() - t0:.1f}s")
    finally:
        if pool is not None:
            pool.close()

    return {r.run_id: (run_totals[r.run_id], cache.breakdown_for(samples[r.run_id])) for r in runs}


def run_many(data_dir: Path, city: str, run_ids: list[str], breakdown_sample: int, workers: int,
             memory: str, limit: int | None, log=print) -> list[Path]:
    cfg = CityConfig.load(data_dir, city)
    runs = [_load_run(cfg, run_id, limit) for run_id in run_ids]
    by_scenario: dict[str, list[LoadedRun]] = {}
    for r in runs:
        by_scenario.setdefault(r.scenario, []).append(r)

    written = []
    for scenario, scenario_runs in by_scenario.items():
        log(f"{scenario}: {len(scenario_runs)} run(s), {sum(len(r.trips) for r in scenario_runs)} trips")
        routed = _route_scenario(cfg, scenario, scenario_runs, breakdown_sample, workers, memory, log)
        for r in scenario_runs:
            totals, breakdown = routed[r.run_id]
            results = assemble_results(r.run_id, r.trips, r.trip_query, totals, breakdown)
            out_dir = cfg.city_dir / "results" / r.run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / "trip_results.parquet"
            results.to_parquet(out, compression="snappy", index=False)
            written.append(out)
        log(f"{scenario}: wrote {len(scenario_runs)} result file(s) under {cfg.city_dir / 'results'}")
    return written

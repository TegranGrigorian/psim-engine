import argparse
import os
from pathlib import Path

from .config import CityConfig
from .run import find_runs, run_many


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="psim-engine", description="Route psim-tools trips with r5py")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="route generated runs and write trip_results.parquet")
    r.add_argument("--city", required=True)
    r.add_argument("--run-id", nargs="+", default=[],
                   help="one or more runs, e.g. baseline_s0042 (folders under generated/)")
    r.add_argument("--scenario", help="route every generated run of this scenario")
    r.add_argument("--data-dir", type=Path, default=Path("data"))
    r.add_argument("--breakdown-sample", type=int, default=100,
                   help="queries per run routed in detail (wait/in-vehicle/walk/transfers); slow, ~1.3 s each")
    r.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                   help="routing worker processes, each with its own JVM")
    r.add_argument("--worker-memory", default="4G", help="JVM memory limit per worker, e.g. 4G")
    r.add_argument("--limit", type=int, default=None, help="only the first N trips of each run (smoke tests)")
    args = p.parse_args(argv)

    run_ids = list(args.run_id)
    if args.scenario:
        run_ids += [x for x in find_runs(CityConfig.load(args.data_dir, args.city), args.scenario)
                    if x not in run_ids]
    if not run_ids:
        p.error("give --run-id and/or --scenario (and run `psim-tools generate` first)")
    run_many(args.data_dir, args.city, run_ids, args.breakdown_sample, args.workers,
             args.worker_memory, args.limit)


if __name__ == "__main__":
    main()

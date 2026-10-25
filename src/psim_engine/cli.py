import argparse
from pathlib import Path

from .run import run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="psim-engine", description="Route psim-tools trips with r5py")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="route one generated run")
    r.add_argument("--city", required=True)
    r.add_argument("--run-id", required=True, help="e.g. baseline_s0042 (folder under generated/)")
    r.add_argument("--data-dir", type=Path, default=Path("data"))
    r.add_argument("--breakdown-sample", type=int, default=100,
                   help="queries to route in detail (wait/in-vehicle/walk/transfers); slow, ~1.3 s each")
    r.add_argument("--limit", type=int, default=None, help="only the first N trips (smoke tests)")
    args = p.parse_args(argv)
    run(args.data_dir, args.city, args.run_id, args.breakdown_sample, args.limit)


if __name__ == "__main__":
    main()

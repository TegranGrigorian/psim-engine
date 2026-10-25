# psim-engine

Routes the trips written by `psim-tools generate` with [r5py](https://r5py.readthedocs.io) and writes `trip_results.parquet` (columns: `docs/SCHEMA.md` section 4.1).

Needs Java 21 or newer and Python 3.12 or newer. From the repo root:

```
cd psim-engine && uv sync                      # once
data/ann_arbor/fetch.sh                        # GTFS + OSM (needs osmium)
psim-tools/target/release/psim-tools generate --city ann_arbor --base-seed 42
psim-engine/.venv/bin/psim-engine run --city ann_arbor --run-id baseline_s0042 --data-dir data
cd psim-engine && .venv/bin/python -m pytest
```

How it works (details in SCHEMA.md 4.1): queries are deduplicated, every query gets a total travel time from r5py's fast `TravelTimeMatrix`, and a seeded random sample (`--breakdown-sample`) gets the slower `DetailedItineraries` for the wait / in-vehicle / walk / transfer breakdown.

Raw GTFS is never modified. Cleaned copies (empty optional tables and macOS junk removed, which R5 rejects) go in `data/<city>/raw/gtfs_clean/`.

Written with AI assistance (Claude), see `docs/Ai-policy.md`. Marked with the git tag `ai-engine-v1` so it can be found and redone by hand.

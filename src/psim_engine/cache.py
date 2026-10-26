"""Cache of routing results, shared by every run of a scenario.

The same origin-destination-time query comes up in many replicates (weekday
repeats, and personas who share a home and workplace), so results are stored by
query and reused. The cache lives in a folder named by a hash of everything
that could change an answer (OSM, GTFS, routing settings), so a data refresh
starts a fresh cache instead of serving stale numbers.
"""

import hashlib
import importlib.metadata
from pathlib import Path

import pandas as pd

from .config import CityConfig
from .queries import QUERY_KEY

# Bump when the routing method changes in a way that changes results.
METHOD_VERSION = "2"

TOTALS_COLUMNS = ["completed", "walk_only", "total_minutes"]
BREAKDOWN_COLUMNS = [
    "breakdown_completed", "breakdown_total_minutes", "wait_minutes",
    "in_vehicle_minutes", "walk_minutes", "transfers",
    # planned required trips only: when the rider leaves, the timetable travel time, and the delay model's output
    "planned_departure", "scheduled_minutes", "p_on_time", "delay_p95_minutes",
]


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Same key spelling everywhere: 6-decimal coordinates, text date and time."""
    out = df.copy()
    for col in ("origin_lat", "origin_lon", "dest_lat", "dest_lon"):
        out[col] = out[col].astype(float).round(6)
    out["date"] = out["date"].astype(str)
    out["depart_time"] = out["depart_time"].astype(str)
    out["due_time"] = out["due_time"].fillna("").astype(str)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def network_key(cfg: CityConfig, scenario: str) -> str:
    """Hash of every input that can change a routing result."""
    parts = [
        f"method={METHOD_VERSION}",
        f"r5py={importlib.metadata.version('r5py')}",
        f"window={cfg.departure_window_minutes}",
        f"max_walk={cfg.max_walk_minutes}",
        f"max_trip={cfg.max_trip_minutes}",
        f"osm={_sha256(cfg.osm)}",
    ]
    for folder in cfg.gtfs[scenario]:
        for zip_path in sorted(folder.glob("*.zip")):
            parts.append(f"gtfs:{folder.name}={_sha256(zip_path)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


class QueryCache:
    def __init__(self, directory: Path):
        self.dir = directory
        self.totals = self._read("totals.parquet", QUERY_KEY + TOTALS_COLUMNS)
        self.breakdown = self._read("breakdown.parquet", QUERY_KEY + BREAKDOWN_COLUMNS)

    @classmethod
    def open(cls, cfg: CityConfig, scenario: str) -> "QueryCache":
        return cls(cfg.city_dir / "results" / "_cache" / f"{scenario}_{network_key(cfg, scenario)}")

    def _read(self, name: str, columns: list[str]) -> pd.DataFrame:
        path = self.dir / name
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for name, df in (("totals.parquet", self.totals), ("breakdown.parquet", self.breakdown)):
            tmp = self.dir / (name + ".part")
            df.to_parquet(tmp, index=False)
            tmp.replace(self.dir / name)

    @staticmethod
    def _missing(keys: pd.DataFrame, have: pd.DataFrame) -> pd.DataFrame:
        keys = normalize_keys(keys).drop_duplicates(QUERY_KEY)
        if not len(have):
            return keys.reset_index(drop=True)
        seen = normalize_keys(have)[QUERY_KEY].assign(_seen=True)
        merged = keys.merge(seen, on=QUERY_KEY, how="left")
        return keys[merged["_seen"].isna().values].reset_index(drop=True)

    def missing_totals(self, keys: pd.DataFrame) -> pd.DataFrame:
        return self._missing(keys, self.totals)

    def missing_breakdown(self, keys: pd.DataFrame) -> pd.DataFrame:
        return self._missing(keys, self.breakdown)

    def add_totals(self, rows: pd.DataFrame) -> None:
        self.totals = pd.concat([self.totals, normalize_keys(rows)[QUERY_KEY + TOTALS_COLUMNS]],
                                ignore_index=True)

    def add_breakdown(self, rows: pd.DataFrame) -> None:
        self.breakdown = pd.concat(
            [self.breakdown, normalize_keys(rows)[QUERY_KEY + BREAKDOWN_COLUMNS]], ignore_index=True)

    def totals_for(self, queries: pd.DataFrame) -> pd.DataFrame:
        """`query_id` plus the cached totals for each of a run's queries."""
        q = normalize_keys(queries)
        return q.merge(normalize_keys(self.totals), on=QUERY_KEY, how="left",
                       validate="one_to_one")[["query_id", *TOTALS_COLUMNS]]

    def breakdown_for(self, queries: pd.DataFrame) -> pd.DataFrame:
        """`query_id` plus cached breakdowns, for the queries that have one."""
        q = normalize_keys(queries)
        return q.merge(normalize_keys(self.breakdown), on=QUERY_KEY, how="inner",
                       validate="one_to_one")[["query_id", *BREAKDOWN_COLUMNS]]

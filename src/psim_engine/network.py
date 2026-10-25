"""Builds the r5py transport network for a scenario."""

from pathlib import Path

import r5py

from .config import CityConfig
from .gtfs import prepare_feeds


def build_network(cfg: CityConfig, scenario: str) -> r5py.TransportNetwork:
    if scenario not in cfg.gtfs:
        raise KeyError(f"scenario `{scenario}` not in config gtfs: {sorted(cfg.gtfs)}")
    if not cfg.osm.exists():
        raise FileNotFoundError(f"{cfg.osm} missing; run data/<city>/fetch.sh (needs osmium)")
    clean_root: Path = cfg.city_dir / "raw" / "gtfs_clean"
    feeds = prepare_feeds(cfg.gtfs[scenario], clean_root)
    return r5py.TransportNetwork(str(cfg.osm), [str(f) for f in feeds])

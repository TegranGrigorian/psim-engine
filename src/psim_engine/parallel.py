"""Routing in several worker processes.

r5py's detailed itineraries are bound by Python, not by R5, so threads do not
help; separate processes, each with its own JVM, do. Workers are started with
`spawn` (a JVM cannot be forked) and each builds the network once. This module
must not import r5py in the main process: that would start a JVM there, too.
"""

import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from .config import CityConfig

_NET = None
_CFG = None


def _init_worker(cfg: CityConfig, scenario: str, memory: str) -> None:
    global _NET, _CFG
    # r5py reads --max-memory from the command line when it is first imported.
    sys.argv = [sys.argv[0], "--max-memory", memory]
    from .network import build_network

    _CFG = cfg
    _NET = build_network(cfg, scenario)


def _warm() -> bool:
    """Lets one worker build and cache the network before the others start."""
    return _NET is not None


def _totals_task(queries: pd.DataFrame) -> pd.DataFrame:
    from .routing import route_totals

    return route_totals(_NET, queries, _CFG, log=lambda *_: None)


def _detail_task(todo: pd.DataFrame) -> pd.DataFrame:
    from .routing import route_detail

    return route_detail(_NET, todo, _CFG)


def balanced_chunks(df: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    """Splits rows into at most `n` chunks of similar size, never splitting a
    (date, depart_time) group, so each chunk shares departure times."""
    groups = sorted((g for _, g in df.groupby(["date", "depart_time"], sort=True)),
                    key=len, reverse=True)
    bins: list[list[pd.DataFrame]] = [[] for _ in range(max(1, min(n, len(groups))))]
    sizes = [0] * len(bins)
    for g in groups:  # largest first, into the currently smallest bin
        i = sizes.index(min(sizes))
        bins[i].append(g)
        sizes[i] += len(g)
    return [pd.concat(b, ignore_index=True) for b in bins if b]


class Pool:
    """A pool of routing workers for one scenario."""

    def __init__(self, cfg: CityConfig, scenario: str, workers: int, memory: str):
        ctx = mp.get_context("spawn")
        args = dict(mp_context=ctx, initializer=_init_worker, initargs=(cfg, scenario, memory))
        # Build the network once, alone, so the workers don't race to write r5py's cache.
        with ProcessPoolExecutor(max_workers=1, **args) as warm:
            warm.submit(_warm).result()
        self.workers = workers
        self._pool = ProcessPoolExecutor(max_workers=workers, **args)

    def map(self, fn, df: pd.DataFrame, label: str, log=print) -> pd.DataFrame:
        """Runs `fn` over balanced chunks of `df`, concatenating the results."""
        if not len(df):
            return pd.DataFrame()
        chunks = balanced_chunks(df, self.workers * 3)
        futures = [self._pool.submit(fn, c) for c in chunks]
        parts, done = [], 0
        for f in as_completed(futures):
            parts.append(f.result())
            done += 1
            log(f"  {label}: {done}/{len(futures)} chunks")
        return pd.concat(parts, ignore_index=True)

    def close(self) -> None:
        self._pool.shutdown()

    def __enter__(self) -> "Pool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

"""How much do bus delays cost a rider who plans by the timetable?

The simulation routes on scheduled service, so by itself it has no delays. This module
adds them, with a model that is deliberately simple and whose one parameter comes from
published data:

- Each bus is late by L minutes (never early), L ~ Exponential(rate). The rate is set so
  that a share `on_time_share` of departures is within `on_time_window_minutes` of the
  schedule, which is how the agency reports on-time performance (TheRide: 82% in Q1 FY2026).
  The agency figure fixes one point of the distribution. The shape below that point and the
  tail above it are assumptions, so results should be read as "given this delay model".
- A rider on the planned itinerary boards each bus at its actual time. If the rider arrives
  at a transfer after the planned bus has actually left, they wait for the next scheduled
  departure of that route (plus a fresh lateness draw). That is where unreliability piles up.
- Early departures are ignored (they would make riders miss buses too), and the same
  lateness distribution is used for every route and hour.

Output per planned trip: the probability of arriving by the deadline, and the 95th
percentile of the delay (the extra minutes to allow so that 95% of trips are on time).
"""

import math
import zipfile
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Reliability:
    on_time_share: float
    on_time_window_minutes: float
    arrive_margin_minutes: float
    target_reliability: float
    draws: int
    source: str

    @property
    def rate(self) -> float:
        """Exponential rate (per minute) with P(late > window) = 1 - on_time_share."""
        return -math.log(1.0 - self.on_time_share) / self.on_time_window_minutes

    @classmethod
    def from_json(cls, raw: dict) -> "Reliability":
        return cls(
            on_time_share=float(raw["on_time_share"]),
            on_time_window_minutes=float(raw.get("on_time_window_minutes", 5)),
            arrive_margin_minutes=float(raw.get("arrive_margin_minutes", 5)),
            target_reliability=float(raw.get("target_reliability", 0.95)),
            draws=int(raw.get("draws", 1000)),
            source=str(raw.get("source", "")),
        )


@dataclass(frozen=True)
class Leg:
    """One bus ride of the planned itinerary, in minutes after midnight."""
    feed: str
    route_id: str
    stop_id: str  # boarding stop
    reach_stop: float  # when the rider planned to be at the stop
    board: float  # scheduled departure
    alight: float  # scheduled arrival at the end of the ride

    @property
    def slack(self) -> float:
        """Planned wait at the stop, the cushion against a late arrival."""
        return max(0.0, self.board - self.reach_stop)


class Schedule:
    """Scheduled departures per (route, stop) on one date, for missed-transfer waits."""

    def __init__(self, departures: dict[tuple[str, str], np.ndarray]):
        self._d = departures

    def next_departure(self, route_id: str, stop_id: str, after: float) -> float | None:
        """First scheduled departure strictly after `after` (minutes), or None."""
        times = self._d.get((route_id, stop_id))
        if times is None:
            return None
        i = int(np.searchsorted(times, after + 1e-6, side="right"))
        return float(times[i]) if i < len(times) else None

    @classmethod
    def from_gtfs(cls, zip_path: Path, service_date: Date) -> "Schedule":
        """Reads stop times of the trips running on `service_date`."""
        ymd = service_date.strftime("%Y%m%d")
        weekday = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][service_date.weekday()]
        with zipfile.ZipFile(zip_path) as z:
            def table(name, **kw):
                if name not in z.namelist():
                    return pd.DataFrame()
                with z.open(name) as f:
                    return pd.read_csv(f, dtype=str, encoding="utf-8-sig", **kw)

            active: set[str] = set()
            cal = table("calendar.txt")
            if len(cal):
                ok = (cal["start_date"] <= ymd) & (cal["end_date"] >= ymd) & (cal[weekday] == "1")
                active |= set(cal.loc[ok, "service_id"])
            exc = table("calendar_dates.txt")
            if len(exc):
                today = exc[exc["date"] == ymd]
                active |= set(today.loc[today["exception_type"] == "1", "service_id"])
                active -= set(today.loc[today["exception_type"] == "2", "service_id"])
            trips = table("trips.txt", usecols=["route_id", "service_id", "trip_id"])
            trips = trips[trips["service_id"].isin(active)]
            route_of = dict(zip(trips["trip_id"], trips["route_id"]))
            keep = set(route_of)
            parts = []
            with z.open("stop_times.txt") as f:
                for chunk in pd.read_csv(f, dtype=str, encoding="utf-8-sig", chunksize=500_000,
                                         usecols=["trip_id", "stop_id", "departure_time"]):
                    parts.append(chunk[chunk["trip_id"].isin(keep)])
        st = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not len(st):
            return cls({})  # no service that day
        st = st[st["departure_time"].notna() & (st["departure_time"] != "")]
        hms = st["departure_time"].str.split(":", expand=True).astype(float)
        st["minute"] = hms[0] * 60 + hms[1] + hms[2] / 60
        st["route_id"] = st["trip_id"].map(route_of)
        out = {
            key: np.sort(g["minute"].to_numpy())
            for key, g in st.groupby(["route_id", "stop_id"], sort=False)
        }
        return cls(out)


def simulate_delays(legs: list[Leg], schedules: dict[str, Schedule], rel: Reliability,
                    rng: np.random.Generator) -> np.ndarray:
    """Minutes by which the rider arrives later than planned, one value per draw."""
    n = rel.draws
    carried = np.zeros(n)  # lateness the rider carries into the next leg
    for leg in legs:
        late = rng.exponential(1.0 / rel.rate, n)
        boards_planned_bus = carried <= leg.slack + late
        nxt = schedules[leg.feed].next_departure(leg.route_id, leg.stop_id, leg.board) if leg.feed in schedules else None
        if nxt is None:
            missed_delay = np.full(n, 180.0)  # no later bus that day: effectively a failure
        else:
            missed_delay = (nxt - leg.board) + rng.exponential(1.0 / rel.rate, n)
        carried = np.where(boards_planned_bus, late, missed_delay)
    return carried


def summarize_delays(delays: np.ndarray, slack_minutes: float, rel: Reliability) -> tuple[float, float]:
    """(probability of arriving by the deadline, the delay that `target_reliability` of trips stay under)."""
    return float((delays <= slack_minutes + 1e-9).mean()), float(np.quantile(delays, rel.target_reliability))

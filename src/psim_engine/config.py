"""Reads the same `data/<city>/config.json` that psim-tools validates."""

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from .reliability import Reliability


@dataclass(frozen=True)
class CityConfig:
    city_dir: Path
    city_slug: str
    timezone: str
    service_date: dt.date  # routes mon..fri trips
    weekend_service_date: dt.date  # routes sat trips
    departure_window_minutes: int
    max_walk_minutes: int
    max_trip_minutes: int
    osm: Path
    gtfs: dict[str, list[Path]]  # scenario -> feed folders
    reliability: Reliability | None = None  # delay model; None = timetable only

    @classmethod
    def load(cls, data_dir: Path, city: str) -> "CityConfig":
        city_dir = data_dir / city
        raw = json.loads((city_dir / "config.json").read_text())
        return cls(
            city_dir=city_dir,
            city_slug=raw["city_slug"],
            timezone=raw["timezone"],
            service_date=dt.date.fromisoformat(raw["service_date"]),
            weekend_service_date=dt.date.fromisoformat(raw["weekend_service_date"]),
            departure_window_minutes=raw["departure_window_minutes"],
            max_walk_minutes=raw["max_walk_minutes"],
            max_trip_minutes=raw["max_trip_minutes"],
            osm=city_dir / raw["osm"],
            gtfs={name: [city_dir / f for f in folders] for name, folders in raw["gtfs"].items()},
            reliability=Reliability.from_json(raw["reliability"]) if raw.get("reliability") else None,
        )

    def date_for_day(self, day: str) -> dt.date:
        """Calendar date a `mon`..`sat` trip is routed on (SCHEMA section 1)."""
        if day in ("mon", "tue", "wed", "thu", "fri"):
            return self.service_date
        if day == "sat":
            return self.weekend_service_date
        raise ValueError(f"no service date for day `{day}` (v1 generates mon..sat)")

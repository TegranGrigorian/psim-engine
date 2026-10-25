"""GTFS preparation.

r5py (R5) rejects optional tables that are present but have no rows, and some
feeds ship macOS `__MACOSX/` resource-fork entries. We never modify the raw
download; we write a cleaned copy and route on that.
"""

import zipfile
from pathlib import Path

# Optional tables: safe to drop when they have a header but no rows.
OPTIONAL_TABLES = {
    "fare_attributes.txt",
    "fare_rules.txt",
    "frequencies.txt",
    "transfers.txt",
    "shapes.txt",
    "feed_info.txt",
    "calendar_dates.txt",
    "pathways.txt",
    "levels.txt",
    "translations.txt",
    "attributions.txt",
}


def _has_rows(data: bytes) -> bool:
    lines = [line for line in data.splitlines() if line.strip()]
    return len(lines) > 1


def clean_gtfs(src: Path, dst: Path) -> list[str]:
    """Writes a cleaned copy of `src` to `dst`. Returns the names it dropped."""
    dropped: list[str] = []
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            name = info.filename
            base = name.rsplit("/", 1)[-1]
            if info.is_dir() or name.startswith("__MACOSX/") or base.startswith("._"):
                dropped.append(name)
                continue
            data = zin.read(name)
            if base in OPTIONAL_TABLES and not _has_rows(data):
                dropped.append(name)
                continue
            zout.writestr(name, data)
    tmp.replace(dst)
    return dropped


def find_feed_zips(folder: Path) -> list[Path]:
    """The raw GTFS zip(s) in a config `gtfs` folder."""
    zips = sorted(folder.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"no GTFS zip in {folder}; run data/<city>/fetch.sh")
    return zips


def prepare_feeds(folders: list[Path], clean_root: Path) -> list[Path]:
    """Cleaned zips for every feed folder, cached under `clean_root`.

    Re-cleans when the raw zip is newer than the cleaned copy.
    File names look like `gtfs__theride__google_transit.zip`.
    """
    out: list[Path] = []
    for folder in folders:
        for raw in find_feed_zips(folder):
            # r5py links each GTFS file into its cache by *basename*, so names must be
            # unique across agencies and scenarios (raw feeds are all google_transit.zip).
            dst = clean_root / f"{folder.parent.name}__{folder.name}__{raw.stem}.zip"
            if not dst.exists() or dst.stat().st_mtime < raw.stat().st_mtime:
                clean_gtfs(raw, dst)
            out.append(dst)
    return out

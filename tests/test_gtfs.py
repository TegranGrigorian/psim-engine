import zipfile
from pathlib import Path

from psim_engine.gtfs import clean_gtfs, prepare_feeds


def make_zip(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return path


def names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as z:
        return set(z.namelist())


def test_drops_empty_optional_tables_and_macos_junk(tmp_path):
    src = make_zip(
        tmp_path / "in.zip",
        {
            "agency.txt": "agency_id,agency_name\n1,A\n",
            "fare_rules.txt": "fare_id,route_id\n",
            "frequencies.txt": "trip_id,start_time\n",
            "fare_attributes.txt": "fare_id,price\n1,2.0\n",
            "__MACOSX/._agency.txt": "junk",
        },
    )
    dropped = clean_gtfs(src, tmp_path / "out.zip")
    assert names(tmp_path / "out.zip") == {"agency.txt", "fare_attributes.txt"}
    assert set(dropped) == {"fare_rules.txt", "frequencies.txt", "__MACOSX/._agency.txt"}


def test_never_drops_required_tables_even_if_empty(tmp_path):
    src = make_zip(tmp_path / "in.zip", {"stops.txt": "stop_id\n", "trips.txt": "trip_id\n"})
    clean_gtfs(src, tmp_path / "out.zip")
    assert names(tmp_path / "out.zip") == {"stops.txt", "trips.txt"}


def test_cleaned_names_are_unique_across_agencies_and_scenarios(tmp_path):
    # r5py links GTFS files into its cache by basename, so they must not collide.
    raw = tmp_path / "raw"
    folders = []
    for parent in ("gtfs", "gtfs_redesign_v1"):
        for agency in ("theride", "umich"):
            d = raw / parent / agency
            d.mkdir(parents=True)
            make_zip(d / "google_transit.zip", {"agency.txt": "agency_id\n1\n"})
            folders.append(d)
    out = prepare_feeds(folders, tmp_path / "clean")
    assert len({p.name for p in out}) == 4
    assert all(p.exists() for p in out)

"""Download NYC TLC Yellow Taxi trip records to data/raw/.

Default: yellow_tripdata_2015-01.parquet (~12 M rows, lat/lon included).
The 2015 dataset is the newest year that still embeds pickup/dropoff
coordinates directly in the trip records (2016+ switched to zone IDs only).

Usage
-----
python scripts/download_data.py                   # downloads 2015-01
python scripts/download_data.py 2015-01 2015-02   # specific months
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
DEFAULT_MONTHS = ["2015-01"]


def download_month(year_month: str, dest_dir: Path) -> None:
    """Download one month of TLC yellow cab parquet to *dest_dir*.

    Parameters
    ----------
    year_month : str
        ``YYYY-MM`` string, e.g. ``"2015-01"``.
    dest_dir : Path
        Directory in which to save the downloaded file.
    """
    filename = f"yellow_tripdata_{year_month}.parquet"
    dest = dest_dir / filename

    if dest.exists():
        mb = dest.stat().st_size / 1_000_000
        print(f"  skip  {filename}  ({mb:.0f} MB, already present)")
        return

    url = f"{BASE_URL}/{filename}"
    print(f"  GET   {url}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=8 << 20):
                fh.write(chunk)

    mb = dest.stat().st_size / 1_000_000
    print(f"  saved {filename}  ({mb:.0f} MB)")


if __name__ == "__main__":
    months = sys.argv[1:] or DEFAULT_MONTHS
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(months)} file(s) → {RAW_DIR}")
    for m in months:
        download_month(m, RAW_DIR)
    print("Done.")

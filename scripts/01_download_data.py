"""
Pull FMCSA datasets from data.transportation.gov, filter to IN/IL, save CSVs.
Source fields are all typed as text, so numeric and date filters happen
client-side after parsing.
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

BASE = "https://data.transportation.gov/resource"
PAGE = 50_000
APP_TOKEN = os.environ.get("SODA_APP_TOKEN")


def soda_get(dataset_id: str, where: str) -> pd.DataFrame:
    url = f"{BASE}/{dataset_id}.csv"
    headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        params = {
            "$where": where,
            "$limit": PAGE,
            "$offset": offset,
            "$order": ":id",
        }
        resp = requests.get(url, params=params, headers=headers, timeout=180)
        if resp.status_code != 200:
            print(f"  ! HTTP {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
            resp.raise_for_status()
        chunk = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        if chunk.empty:
            break
        frames.append(chunk)
        total = sum(len(f) for f in frames)
        print(f"  fetched {len(chunk):,} (offset={offset:,}, total={total:,})")
        if len(chunk) < PAGE:
            break
        offset += PAGE
        time.sleep(0.2)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    print("[1/3] Motor Carrier Census  (kjg3-diqy)")
    census = soda_get("kjg3-diqy", "phy_state in ('IN','IL')")
    census["nbr_power_unit_num"] = pd.to_numeric(census["nbr_power_unit"], errors="coerce").fillna(0)
    before = len(census)
    census = census[census["nbr_power_unit_num"] > 0].copy()
    print(f"  filtered to active power units: {len(census):,} (dropped {before - len(census):,})")
    census.to_csv(RAW / "census_in_il.csv", index=False)

    print("[2/3] SMS Inspections  (rbkj-cgst)")
    insp = soda_get("rbkj-cgst", "report_state in ('IN','IL')")
    insp["insp_date_parsed"] = pd.to_datetime(insp["insp_date"], errors="coerce")
    before = len(insp)
    insp = insp[insp["insp_date_parsed"] >= "2024-03-01"].copy()
    print(f"  filtered to >=2024-03-01: {len(insp):,} (dropped {before - len(insp):,})")
    insp.to_csv(RAW / "inspections_in_il.csv", index=False)

    print("[3/3] Crash Data  (4wxs-vbns)")
    crashes = soda_get("4wxs-vbns", "report_state in ('IN','IL')")
    crashes["report_date_parsed"] = pd.to_datetime(crashes["report_date"], errors="coerce")
    before = len(crashes)
    crashes = crashes[crashes["report_date_parsed"] >= "2024-03-01"].copy()
    print(f"  filtered to >=2024-03-01: {len(crashes):,} (dropped {before - len(crashes):,})")
    crashes.to_csv(RAW / "crashes_in_il.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()

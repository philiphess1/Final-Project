"""
Build the carrier-level analysis table.

Inputs:  data/raw/{census,inspections,crashes}_in_il.csv
Outputs:
  data/processed/inspected_cohort.csv
  data/processed/all_registered.csv
  data/processed/summary_stats.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def fleet_bin(n: float) -> str:
    if pd.isna(n) or n <= 0:
        return "unknown"
    if n == 1:
        return "1"
    if n <= 5:
        return "2-5"
    if n <= 15:
        return "6-15"
    if n <= 50:
        return "16-50"
    if n <= 500:
        return "51-500"
    return "500+"


FLEET_ORDER = ["1", "2-5", "6-15", "16-50", "51-500", "500+"]


def main() -> None:
    print("Loading raw data...")
    census = pd.read_csv(RAW / "census_in_il.csv", low_memory=False)
    inspections = pd.read_csv(RAW / "inspections_in_il.csv", low_memory=False)
    crashes = pd.read_csv(RAW / "crashes_in_il.csv", low_memory=False)

    print(f"  census:      {len(census):,}")
    print(f"  inspections: {len(inspections):,}")
    print(f"  crashes:     {len(crashes):,}")

    census = census.rename(columns={"dot_number": "DOT_NUMBER"})
    census["DOT_NUMBER"] = pd.to_numeric(census["DOT_NUMBER"], errors="coerce")
    census = census.dropna(subset=["DOT_NUMBER"]).copy()
    census["DOT_NUMBER"] = census["DOT_NUMBER"].astype("int64")

    # A=interstate, B=intrastate hazmat, C=intrastate non-hazmat
    op_map = {"A": "Interstate", "B": "Intrastate", "C": "Intrastate"}
    census["operation_type"] = census["carrier_operation"].map(op_map).fillna("Unknown")

    def _bool(col: str) -> pd.Series:
        s = census[col]
        if s.dtype == bool:
            return s
        return s.astype(str).str.upper().isin({"TRUE", "Y", "1"})

    for_hire = _bool("authorized_for_hire") | _bool("exempt_for_hire")
    private_only = _bool("private_only")
    census["carrier_type"] = np.where(for_hire, "For-Hire", np.where(private_only, "Private", "Other"))

    census["nbr_power_unit"] = pd.to_numeric(census["nbr_power_unit"], errors="coerce").fillna(0)
    census["driver_total"] = pd.to_numeric(census["driver_total"], errors="coerce").fillna(0)

    census["fleet_bin"] = census["nbr_power_unit"].apply(fleet_bin)
    census["phy_state"] = census["phy_state"].fillna("UNKNOWN")
    census["phy_city"] = census["phy_city"].fillna("UNKNOWN").str.upper().str.strip()

    # ---- Inspections aggregation -------------------------------------------
    inspections = inspections.rename(columns={"dot_number": "DOT_NUMBER"})
    inspections["DOT_NUMBER"] = pd.to_numeric(inspections["DOT_NUMBER"], errors="coerce")
    inspections = inspections.dropna(subset=["DOT_NUMBER"]).copy()
    inspections["DOT_NUMBER"] = inspections["DOT_NUMBER"].astype("int64")

    insp_cols = [
        "unsafe_viol", "fatigued_viol", "dr_fitness_viol",
        "subt_alcohol_viol", "vh_maint_viol", "hm_viol", "basic_viol",
        "driver_oos_total", "vehicle_oos_total", "oos_total",
    ]
    for col in insp_cols:
        if col in inspections.columns:
            inspections[col] = pd.to_numeric(inspections[col], errors="coerce").fillna(0)
        else:
            inspections[col] = 0

    inspections["inspected_in_in"] = (inspections["report_state"] == "IN").astype(int)
    inspections["inspected_in_il"] = (inspections["report_state"] == "IL").astype(int)

    agg = inspections.groupby("DOT_NUMBER").agg(
        n_inspections=("DOT_NUMBER", "size"),
        unsafe_viol=("unsafe_viol", "sum"),
        fatigued_viol=("fatigued_viol", "sum"),
        dr_fitness_viol=("dr_fitness_viol", "sum"),
        subt_alcohol_viol=("subt_alcohol_viol", "sum"),
        vh_maint_viol=("vh_maint_viol", "sum"),
        hm_viol=("hm_viol", "sum"),
        basic_viol=("basic_viol", "sum"),
        driver_oos_total=("driver_oos_total", "sum"),
        vehicle_oos_total=("vehicle_oos_total", "sum"),
        oos_total=("oos_total", "sum"),
        n_insp_in=("inspected_in_in", "sum"),
        n_insp_il=("inspected_in_il", "sum"),
    ).reset_index()

    agg["any_vh_maint"] = (agg["vh_maint_viol"] > 0).astype(int)
    agg["any_unsafe"] = (agg["unsafe_viol"] > 0).astype(int)
    # fatigued_viol is the Hours of Service BASIC
    agg["any_hos"] = (agg["fatigued_viol"] > 0).astype(int)
    agg["any_substance"] = (agg["subt_alcohol_viol"] > 0).astype(int)

    crashes = crashes.rename(columns={"dot_number": "DOT_NUMBER"})
    crashes["DOT_NUMBER"] = pd.to_numeric(crashes["DOT_NUMBER"], errors="coerce")
    crashes = crashes.dropna(subset=["DOT_NUMBER"]).copy()
    crashes["DOT_NUMBER"] = crashes["DOT_NUMBER"].astype("int64")
    for col in ("fatalities", "injuries"):
        if col in crashes.columns:
            crashes[col] = pd.to_numeric(crashes[col], errors="coerce").fillna(0)
        else:
            crashes[col] = 0
    crashes["tow_away_flag"] = (
        crashes.get("tow_away", "N").astype(str).str.upper().eq("Y").astype(int)
    )

    crash_agg = crashes.groupby("DOT_NUMBER").agg(
        n_crashes=("DOT_NUMBER", "size"),
        n_fatalities=("fatalities", "sum"),
        n_injuries=("injuries", "sum"),
        n_tow_away=("tow_away_flag", "sum"),
    ).reset_index()

    base_cols = [
        "DOT_NUMBER", "legal_name", "phy_state", "phy_city",
        "nbr_power_unit", "driver_total", "fleet_bin",
        "carrier_type", "operation_type",
    ]
    base = census[base_cols].drop_duplicates(subset=["DOT_NUMBER"], keep="first")

    full = base.merge(agg, on="DOT_NUMBER", how="left").merge(crash_agg, on="DOT_NUMBER", how="left")
    fill_zero = [
        "n_inspections", "unsafe_viol", "fatigued_viol", "dr_fitness_viol",
        "subt_alcohol_viol", "vh_maint_viol", "hm_viol", "basic_viol",
        "driver_oos_total", "vehicle_oos_total", "oos_total",
        "n_insp_in", "n_insp_il",
        "any_vh_maint", "any_unsafe", "any_hos", "any_substance",
        "n_crashes", "n_fatalities", "n_injuries", "n_tow_away",
    ]
    for col in fill_zero:
        if col in full.columns:
            full[col] = full[col].fillna(0).astype(int)

    full["was_inspected"] = (full["n_inspections"] > 0).astype(int)
    full["fleet_bin"] = pd.Categorical(full["fleet_bin"], categories=FLEET_ORDER + ["unknown"], ordered=True)

    inspected = full.loc[full["was_inspected"] == 1].copy()

    inspected["vh_per_insp"] = inspected["vh_maint_viol"] / inspected["n_inspections"]
    inspected["all_viol_per_insp"] = inspected["basic_viol"] / inspected["n_inspections"]
    inspected["oos_rate"] = inspected["oos_total"] / inspected["n_inspections"]

    full.to_csv(PROC / "all_registered.csv", index=False)
    inspected.to_csv(PROC / "inspected_cohort.csv", index=False)

    summary = {
        "registered_total": int(len(full)),
        "registered_in": int((full["phy_state"] == "IN").sum()),
        "registered_il": int((full["phy_state"] == "IL").sum()),
        "inspections_total": int(len(inspections)),
        "inspections_in": int((inspections["report_state"] == "IN").sum()),
        "inspections_il": int((inspections["report_state"] == "IL").sum()),
        "crashes_total": int(len(crashes)),
        "inspected_carriers": int(len(inspected)),
        "inspected_pct": round(100 * len(inspected) / max(len(full), 1), 2),
        "non_inspected_carriers": int(len(full) - len(inspected)),
        "non_inspected_pct": round(100 * (len(full) - len(inspected)) / max(len(full), 1), 2),
        "vh_maint_rate_overall": round(100 * inspected["any_vh_maint"].mean(), 2),
        "vh_maint_rate_in": round(
            100 * inspected.loc[inspected["phy_state"] == "IN", "any_vh_maint"].mean(), 2
        ),
        "vh_maint_rate_il": round(
            100 * inspected.loc[inspected["phy_state"] == "IL", "any_vh_maint"].mean(), 2
        ),
        "vh_maint_violations_total": int(inspected["vh_maint_viol"].sum()),
        "unsafe_viol_total": int(inspected["unsafe_viol"].sum()),
        "fatigued_viol_total": int(inspected["fatigued_viol"].sum()),
        "subt_alcohol_viol_total": int(inspected["subt_alcohol_viol"].sum()),
        "vh_maint_rate_by_fleet": (
            inspected.groupby("fleet_bin", observed=True)["any_vh_maint"]
            .agg(["mean", "size"])
            .round(4)
            .to_dict(orient="index")
        ),
        "vh_maint_rate_by_carrier_type": (
            inspected.groupby("carrier_type", observed=True)["any_vh_maint"]
            .agg(["mean", "size"])
            .round(4)
            .to_dict(orient="index")
        ),
        "vh_maint_rate_by_operation": (
            inspected.groupby("operation_type", observed=True)["any_vh_maint"]
            .agg(["mean", "size"])
            .round(4)
            .to_dict(orient="index")
        ),
    }

    with open(PROC / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nSummary:")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk}: {sv}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

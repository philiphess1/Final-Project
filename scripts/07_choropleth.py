"""
Indiana county choropleth of vehicle-maintenance violation rate.

ZIP-to-county mapping from pgeocode; county geometry from the public Census
2018 county GeoJSON. Smoothed county-level violation rate is computed with
empirical-Bayes shrinkage so small-n counties don't dominate the map.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pgeocode

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

GEOJSON_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/"
    "geojson-counties-fips.json"
)
GEOJSON_CACHE = RAW / "us_counties_fips.geojson"

# IN state FIPS = 18
IN_FIPS = "18"


def load_county_geometry() -> gpd.GeoDataFrame:
    if not GEOJSON_CACHE.exists():
        print(f"Fetching {GEOJSON_URL} ...")
        urllib.request.urlretrieve(GEOJSON_URL, GEOJSON_CACHE)
    gdf = gpd.read_file(GEOJSON_CACHE)
    gdf["state_fips"] = gdf["id"].str[:2]
    in_gdf = gdf[gdf["state_fips"] == IN_FIPS].copy()
    in_gdf["county_fips"] = in_gdf["id"].astype(str)
    return in_gdf[["county_fips", "NAME", "geometry"]]


def main() -> None:
    cohort = pd.read_csv(PROC / "inspected_cohort.csv")
    cohort = cohort[cohort["phy_state"] == "IN"].copy()

    census = pd.read_csv(RAW / "census_in_il.csv", low_memory=False,
                         usecols=["dot_number", "phy_state", "phy_zip"])
    census = census[census["phy_state"] == "IN"].copy()
    census["DOT_NUMBER"] = pd.to_numeric(census["dot_number"], errors="coerce")
    census = census.dropna(subset=["DOT_NUMBER"]).copy()
    census["DOT_NUMBER"] = census["DOT_NUMBER"].astype("int64")
    census["zip5"] = census["phy_zip"].astype(str).str[:5].str.zfill(5)

    cohort = cohort.merge(census[["DOT_NUMBER", "zip5"]], on="DOT_NUMBER", how="left")

    # ZIP to county via pgeocode
    print("Geocoding ZIPs to counties...")
    nomi = pgeocode.Nominatim("us")
    unique_zips = cohort["zip5"].dropna().unique().tolist()
    geo = nomi.query_postal_code(unique_zips).set_index("postal_code")
    cohort["county_name"] = cohort["zip5"].map(geo["county_name"])
    cohort = cohort.dropna(subset=["county_name"]).copy()
    print(f"  Resolved {len(cohort):,} of {len(cohort) + cohort['county_name'].isna().sum():,} carriers")

    # Aggregate by county with empirical-Bayes shrinkage toward state mean
    state_mean = cohort["any_vh_maint"].mean()
    M = 30  # prior strength (carriers worth of state-mean prior)
    grouped = cohort.groupby("county_name").agg(
        n=("DOT_NUMBER", "size"),
        rate=("any_vh_maint", "mean"),
    )
    grouped["smoothed_rate"] = (
        (grouped["rate"] * grouped["n"] + state_mean * M) / (grouped["n"] + M)
    )

    # Match to county geometry
    in_gdf = load_county_geometry()
    in_gdf["NAME_upper"] = in_gdf["NAME"].str.upper()
    grouped = grouped.reset_index()
    grouped["NAME_upper"] = grouped["county_name"].str.upper()
    plot_gdf = in_gdf.merge(grouped, on="NAME_upper", how="left")
    plot_gdf["smoothed_rate"] = plot_gdf["smoothed_rate"].fillna(state_mean)
    plot_gdf["n"] = plot_gdf["n"].fillna(0).astype(int)

    print(f"\nState mean any-VH-maint rate: {state_mean:.3f}")
    print(f"Counties with carriers: {(plot_gdf['n'] > 0).sum()} of {len(plot_gdf)}")
    print("Top 5 counties by smoothed rate:")
    print(plot_gdf.sort_values("smoothed_rate", ascending=False)[
        ["NAME", "n", "rate", "smoothed_rate"]
    ].head().to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 9))
    plot_gdf.plot(
        column="smoothed_rate",
        ax=ax,
        cmap="RdYlBu_r",
        edgecolor="white",
        linewidth=0.4,
        legend=True,
        legend_kwds={"label": "Smoothed VH Maint violation rate",
                     "orientation": "horizontal", "shrink": 0.6, "pad": 0.02},
    )
    # Label counties with the highest n
    top_n = plot_gdf.sort_values("n", ascending=False).head(8)
    for _, row in top_n.iterrows():
        c = row.geometry.centroid
        ax.annotate(f"{row['NAME']}\n(n={row['n']})", xy=(c.x, c.y),
                    ha="center", va="center", fontsize=7, color="#222")
    ax.set_title(
        "Indiana Carrier Vehicle Maintenance Violation Rate, by Home County\n"
        f"(Empirical-Bayes shrinkage toward state mean = {state_mean:.1%}; M = {M})",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "fig_in_county_choropleth.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIG / 'fig_in_county_choropleth.png'}")

    plot_gdf.drop(columns=["geometry"]).to_csv(
        PROC / "county_violation_rates.csv", index=False
    )


if __name__ == "__main__":
    main()

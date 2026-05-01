"""
EDA figures used in the final report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FLEET_ORDER = ["1", "2-5", "6-15", "16-50", "51-500", "500+"]


def main() -> None:
    df = pd.read_csv(PROC / "inspected_cohort.csv")
    df["fleet_bin"] = df["fleet_bin"].astype(str)

    # Fig 1: total violations by BASIC category
    totals = pd.Series({
        "Vehicle Maint.": df["vh_maint_viol"].sum(),
        "Unsafe Driving": df["unsafe_viol"].sum(),
        "Hours of Service": df["fatigued_viol"].sum(),
        "Controlled Substances": df["subt_alcohol_viol"].sum(),
    }).sort_values()
    colors = ["#264653", "#E9C46A", "#3D5A80", "#E63946"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(totals.index, totals.values, color=colors)
    ax.set_xlabel("Total Violations (Last 24 Months)")
    ax.set_title("Total Violations by BASIC Category\nIN + IL Carriers, 24-Month SMS Window")
    for b, v in zip(bars, totals.values):
        ax.text(v + max(totals.values) * 0.005, b.get_y() + b.get_height() / 2,
                f"{int(v):,}", va="center")
    ax.grid(True, axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_violations_by_basic.png", dpi=150)
    plt.close(fig)

    # Fig 2: violation rate by fleet size bin
    fleet = (
        df.groupby("fleet_bin").agg(rate=("any_vh_maint", "mean"), n=("DOT_NUMBER", "size"))
        .reindex([b for b in FLEET_ORDER if b in df["fleet_bin"].unique()])
        .dropna()
    )
    overall = df["any_vh_maint"].mean()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = ["#3D5A80" if r < overall else "#E63946" for r in fleet["rate"]]
    bars = ax.bar(fleet.index, fleet["rate"] * 100, color=cmap)
    ax.axhline(overall * 100, ls="--", color="grey",
               label=f"Overall avg ({overall * 100:.1f}%)")
    for b, r, n in zip(bars, fleet["rate"], fleet["n"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{r * 100:.1f}%\n(n={n:,})", ha="center", fontsize=9)
    ax.set_ylabel("% of Carriers with VH Maint Violation")
    ax.set_xlabel("Fleet Size (Number of Power Units)")
    ax.set_title("Vehicle Maintenance Violation Rate by Fleet Size\nIN + IL Carriers, Last 24 Months")
    ax.set_ylim(0, max(fleet["rate"]) * 110)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig2_rate_by_fleet.png", dpi=150)
    plt.close(fig)

    # Fig 3: carrier type + operation type
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, col, palette, title in [
        (axes[0], "carrier_type", ["#3D5A80", "#E9C46A"], "By Carrier Type"),
        (axes[1], "operation_type", ["#264653", "#A5BACE"], "By Operation Type"),
    ]:
        agg = df.groupby(col).agg(rate=("any_vh_maint", "mean"), n=("DOT_NUMBER", "size"))
        agg = agg[agg["n"] >= 100]
        cs = palette * 5
        bars = ax.bar(agg.index, agg["rate"] * 100, color=cs[: len(agg)])
        for b, r, n in zip(bars, agg["rate"], agg["n"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f"{r * 100:.1f}%\n(n={n:,})", ha="center", fontsize=9)
        ax.set_ylim(0, max(agg["rate"]) * 110)
        ax.set_title(title)
        ax.set_ylabel("% with VH Maint Violation")
    fig.suptitle("Vehicle Maintenance Violation Rate by Carrier Characteristics\nIN + IL Carriers")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_carrier_and_op_type.png", dpi=150)
    plt.close(fig)

    # Fig 4: top IN cities by violation rate
    df_in = df[df["phy_state"] == "IN"].copy()
    by_city = df_in.groupby("phy_city").agg(
        rate=("any_vh_maint", "mean"),
        n=("DOT_NUMBER", "size"),
    )
    by_city = by_city[by_city["n"] >= 20].sort_values("rate", ascending=False).head(12)
    by_city = by_city.iloc[::-1]
    in_avg = df_in["any_vh_maint"].mean()
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(by_city.index, by_city["rate"] * 100, color="#E63946")
    for b, r in zip(bars, by_city["rate"]):
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2,
                f"{r * 100:.1f}%", va="center", fontsize=9)
    ax.axvline(in_avg * 100, ls=":", color="grey", label="IN avg")
    ax.set_xlabel("% of Carriers with VH Maint Violation")
    ax.set_title("Vehicle Maintenance Violation Rate by Indiana City\n(Cities with 20+ Inspected Carriers)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_top_in_cities.png", dpi=150)
    plt.close(fig)

    # Fig 5: IN vs IL side by side
    state_rates = df.groupby("phy_state")[
        ["any_vh_maint", "any_unsafe", "any_hos", "any_substance"]
    ].mean().rename(columns={
        "any_vh_maint": "Vehicle Maint",
        "any_unsafe": "Unsafe Driving",
        "any_hos": "Hours of Service",
        "any_substance": "Controlled Substances",
    })
    fig, ax = plt.subplots(figsize=(9, 5))
    state_rates.T.plot(kind="bar", ax=ax, color=["#E63946", "#3D5A80"])
    ax.set_ylabel("Share of Inspected Carriers")
    ax.set_title("Share of Inspected Carriers with at Least One Violation, by State")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title="State")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_state_compare.png", dpi=150)
    plt.close(fig)

    # Fig 6: inspection exposure vs fleet size
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sub = df[df["nbr_power_unit"] > 0].copy()
    sub["log_pu"] = np.log10(sub["nbr_power_unit"])
    bins = np.linspace(0, sub["log_pu"].max(), 25)
    grp = sub.groupby(pd.cut(sub["log_pu"], bins, include_lowest=True), observed=True).agg(
        mean_insp=("n_inspections", "mean"),
        median_insp=("n_inspections", "median"),
        n=("DOT_NUMBER", "size"),
    ).reset_index()
    grp["bin_center"] = grp["log_pu"].apply(lambda x: 10 ** ((x.left + x.right) / 2))
    ax.plot(grp["bin_center"], grp["mean_insp"], "o-", color="#3D5A80", label="Mean inspections / carrier")
    ax.plot(grp["bin_center"], grp["median_insp"], "s--", color="#E63946", label="Median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fleet size (power units, log scale)")
    ax.set_ylabel("Roadside inspections per carrier (log)")
    ax.set_title("Inspection exposure scales with fleet size")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_inspection_exposure.png", dpi=150)
    plt.close(fig)

    print("Wrote EDA figures to", FIG)


if __name__ == "__main__":
    main()

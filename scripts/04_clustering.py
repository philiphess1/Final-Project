"""
K-means clustering on per-carrier BASIC violation rates.
Per-inspection rates are used (not raw counts) so fleet-size exposure
doesn't dominate the cluster geometry.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CLUSTER_COLS = [
    "vh_per_insp",
    "unsafe_per_insp",
    "fatigued_per_insp",
    "subt_alcohol_per_insp",
    "oos_rate",
]


def main() -> None:
    df = pd.read_csv(PROC / "inspected_cohort.csv")
    print(f"Loaded {len(df):,} inspected carriers")

    df["unsafe_per_insp"] = df["unsafe_viol"] / df["n_inspections"]
    df["fatigued_per_insp"] = df["fatigued_viol"] / df["n_inspections"]
    df["subt_alcohol_per_insp"] = df["subt_alcohol_viol"] / df["n_inspections"]

    # need >= 2 inspections so per-insp rates aren't just 0/1
    cluster_df = df[df["n_inspections"] >= 2].copy().reset_index(drop=True)
    print(f"Clustering on {len(cluster_df):,} carriers with >=2 inspections")

    X = cluster_df[CLUSTER_COLS].fillna(0).values
    Xs = StandardScaler().fit_transform(X)

    inertias, silhouettes = [], []
    ks = list(range(2, 8))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(Xs, labels, sample_size=min(5000, len(Xs)), random_state=42))
        print(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={silhouettes[-1]:.4f}")

    # silhouette favors k=2: a high-risk segment vs everyone else.
    # Earlier we tried k=3 to match the prelim's low/medium/high framing,
    # but the third cluster turned out to be a 59-carrier outlier group
    # with no clear interpretation, so we stay with k=2.
    best_k = ks[int(np.argmax(silhouettes))]
    print(f"\nUsing k = {best_k} (silhouette best)")

    km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    cluster_df["cluster"] = km.fit_predict(Xs)

    profile = cluster_df.groupby("cluster").agg(
        n=("DOT_NUMBER", "size"),
        vh_per_insp=("vh_per_insp", "mean"),
        unsafe_per_insp=("unsafe_per_insp", "mean"),
        fatigued_per_insp=("fatigued_per_insp", "mean"),
        oos_rate=("oos_rate", "mean"),
        any_vh_maint=("any_vh_maint", "mean"),
        median_fleet=("nbr_power_unit", "median"),
        mean_fleet=("nbr_power_unit", "mean"),
        share_intrastate=("operation_type", lambda s: (s == "Intrastate").mean()),
        share_private=("carrier_type", lambda s: (s == "Private").mean()),
        share_in=("phy_state", lambda s: (s == "IN").mean()),
    ).round(4)
    print("\nCluster profile:")
    print(profile)
    profile.to_csv(PROC / "cluster_profiles.csv")

    # rank clusters by total per-insp violation rate; lowest = Low-Risk
    risk_score = profile[["vh_per_insp", "unsafe_per_insp", "fatigued_per_insp", "oos_rate"]].sum(axis=1)
    rank = risk_score.rank(method="dense", ascending=True).astype(int)
    n_clusters = len(profile)
    label_map = {
        cid: ("Low-Risk" if r == 1 else "High-Risk" if r == n_clusters else f"Tier-{r}")
        for cid, r in rank.items()
    }
    cluster_df["cluster_label"] = cluster_df["cluster"].map(label_map)
    profile["label"] = profile.index.map(label_map)
    profile.to_csv(PROC / "cluster_profiles.csv")

    out = cluster_df[[
        "DOT_NUMBER", "phy_state", "phy_city", "fleet_bin", "carrier_type",
        "operation_type", "n_inspections", "vh_maint_viol", "any_vh_maint",
        "cluster", "cluster_label",
    ]]
    out.to_csv(PROC / "cluster_assignments.csv", index=False)

    metrics = ["vh_per_insp", "unsafe_per_insp", "fatigued_per_insp", "oos_rate"]
    plot_df = profile[metrics].rename(columns={
        "vh_per_insp": "Vehicle Maint",
        "unsafe_per_insp": "Unsafe Driving",
        "fatigued_per_insp": "Hours of Service",
        "oos_rate": "Out-of-Service",
    })
    plot_df.index = [f"{label_map[c]}\n(n={profile.loc[c, 'n']:,.0f})" for c in profile.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind="bar", ax=ax, color=["#E63946", "#F4A261", "#E9C46A", "#264653"])
    ax.set_ylabel("Violations per inspection")
    ax.set_title(f"K-means cluster profile (k={best_k})")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="BASIC category", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "cluster_profile.png", dpi=150)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ks, inertias, "o-", color="#264653")
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia"); ax1.set_title("K-means inertia")
    ax2.plot(ks, silhouettes, "o-", color="#E63946")
    ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette"); ax2.set_title("Silhouette score")
    fig.tight_layout()
    fig.savefig(FIG / "cluster_selection.png", dpi=150)
    plt.close(fig)

    with open(PROC / "cluster_meta.json", "w") as f:
        json.dump({
            "best_k": int(best_k),
            "ks": ks,
            "inertias": [float(x) for x in inertias],
            "silhouettes": [float(x) for x in silhouettes],
            "label_map": {int(k): v for k, v in label_map.items()},
        }, f, indent=2)


if __name__ == "__main__":
    main()

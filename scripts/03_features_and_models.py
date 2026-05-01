"""
Features + classifiers for the vehicle-maintenance-violation outcome.
LogReg / RandomForest / XGBoost evaluated with stratified 5-fold CV.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def smoothed_target_encode(series: pd.Series, target: pd.Series, m: float = 50.0) -> pd.Series:
    global_mean = target.mean()
    grouped = target.groupby(series).agg(["mean", "size"])
    smoothed = (grouped["mean"] * grouped["size"] + global_mean * m) / (grouped["size"] + m)
    return series.map(smoothed).fillna(global_mean)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["log_power_units"] = np.log1p(df["nbr_power_unit"].clip(lower=0))
    df["log_drivers"] = np.log1p(df["driver_total"].clip(lower=0))
    df["log_inspections"] = np.log1p(df["n_inspections"].clip(lower=0))
    df["fleet_per_driver"] = df["nbr_power_unit"] / df["driver_total"].replace(0, np.nan)
    df["fleet_per_driver"] = df["fleet_per_driver"].fillna(df["nbr_power_unit"])
    df["pct_inspections_in_in"] = df["n_insp_in"] / df["n_inspections"].replace(0, np.nan)
    df["pct_inspections_in_in"] = df["pct_inspections_in_in"].fillna(0)

    df["any_crash"] = (df["n_crashes"] > 0).astype(int)
    df["log_crashes"] = np.log1p(df["n_crashes"].clip(lower=0))

    df["city_state"] = df["phy_city"].astype(str) + "_" + df["phy_state"].astype(str)

    y = df["any_vh_maint"].astype(int)
    return df, y


NUMERIC_FEATURES = [
    "log_power_units", "log_drivers", "log_inspections",
    "fleet_per_driver", "pct_inspections_in_in",
    "log_crashes", "any_crash",
]
CATEGORICAL_FEATURES = ["carrier_type", "operation_type", "phy_state", "fleet_bin"]
TE_FEATURE = "city_state"


def make_logreg_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES + ["city_te"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear"))])


def make_rf_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES + ["city_te"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre),
                     ("clf", RandomForestClassifier(
                         n_estimators=400, max_depth=None, min_samples_leaf=20,
                         n_jobs=-1, random_state=42))])


def make_xgb_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES + ["city_te"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre),
                     ("clf", xgb.XGBClassifier(
                         n_estimators=600, max_depth=5, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.9,
                         eval_metric="logloss", tree_method="hist",
                         random_state=42, n_jobs=-1))])


def cv_evaluate(name: str, build_fn, X: pd.DataFrame, y: pd.Series,
                splits: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    fold_metrics = []
    oof = np.zeros(len(y))
    for fold, (tr, te) in enumerate(splits):
        X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        # fit the city encoding on the train fold only
        city_te_map = (
            y_tr.groupby(X_tr["city_state"]).agg(["mean", "size"])
            .pipe(lambda g: (g["mean"] * g["size"] + y_tr.mean() * 50.0) / (g["size"] + 50.0))
        )
        global_mean = y_tr.mean()
        X_tr["city_te"] = X_tr["city_state"].map(city_te_map).fillna(global_mean)
        X_te["city_te"] = X_te["city_state"].map(city_te_map).fillna(global_mean)
        pipe = build_fn()
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        oof[te] = proba
        m = {
            "fold": fold,
            "auc_roc": roc_auc_score(y_te, proba),
            "auc_pr": average_precision_score(y_te, proba),
        }
        fold_metrics.append(m)
        print(f"  [{name}] fold {fold}: AUC-ROC={m['auc_roc']:.4f}  AUC-PR={m['auc_pr']:.4f}")
    return {
        "name": name,
        "auc_roc_mean": float(np.mean([m["auc_roc"] for m in fold_metrics])),
        "auc_roc_std": float(np.std([m["auc_roc"] for m in fold_metrics])),
        "auc_pr_mean": float(np.mean([m["auc_pr"] for m in fold_metrics])),
        "auc_pr_std": float(np.std([m["auc_pr"] for m in fold_metrics])),
        "folds": fold_metrics,
        "oof": oof,
    }


def plot_curves(results: dict[str, dict], y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in results.items():
        proba = res["oof"]
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, lw=2,
                label=f"{name} (AUC = {res['auc_roc_mean']:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (5-fold OOF predictions)\nVehicle Maintenance Violation")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in results.items():
        proba = res["oof"]
        prec, rec, _ = precision_recall_curve(y, proba)
        ax.plot(rec, prec, lw=2,
                label=f"{name} (AUC-PR = {res['auc_pr_mean']:.3f})")
    base = y.mean()
    ax.axhline(base, color="grey", lw=1, ls="--", label=f"Base rate = {base:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (5-fold OOF)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "pr_curves.png", dpi=150)
    plt.close(fig)


def plot_xgb_importance(model: xgb.XGBClassifier, feat_names: list[str], top_n: int = 15) -> pd.DataFrame:
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    # booster keys are f0, f1, ... so map them back to the original column names
    rows = []
    for k, v in score.items():
        idx = int(k[1:])
        rows.append({"feature": feat_names[idx], "gain": v})
    imp = pd.DataFrame(rows).sort_values("gain", ascending=False)
    top = imp.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["gain"], color="#3D5A80")
    ax.set_xlabel("Gain")
    ax.set_title(f"XGBoost Feature Importance (top {top_n})")
    fig.tight_layout()
    fig.savefig(FIG / "xgb_importance.png", dpi=150)
    plt.close(fig)
    return imp


def main() -> None:
    df = pd.read_csv(PROC / "inspected_cohort.csv")
    print(f"Loaded inspected cohort: {len(df):,} carriers")
    df["fleet_bin"] = df["fleet_bin"].astype(str)
    df, y = build_features(df)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TE_FEATURE]
    X = df[feature_cols].copy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))

    results: dict[str, dict] = {}
    print("\n=== Logistic Regression ===")
    results["Logistic Regression"] = cv_evaluate("LogReg", make_logreg_pipeline, X, y, splits)
    print("\n=== Random Forest ===")
    results["Random Forest"] = cv_evaluate("RF", make_rf_pipeline, X, y, splits)
    print("\n=== XGBoost ===")
    results["XGBoost"] = cv_evaluate("XGB", make_xgb_pipeline, X, y, splits)

    plot_curves(results, y.values)

    print("\nRefitting XGBoost on full data for importance plot...")
    full_X = X.copy()
    city_te_map = (
        y.groupby(full_X["city_state"]).agg(["mean", "size"])
        .pipe(lambda g: (g["mean"] * g["size"] + y.mean() * 50.0) / (g["size"] + 50.0))
    )
    full_X["city_te"] = full_X["city_state"].map(city_te_map).fillna(y.mean())
    final_pipe = make_xgb_pipeline()
    final_pipe.fit(full_X, y)
    pre = final_pipe.named_steps["pre"]
    feat_names = (
        NUMERIC_FEATURES + ["city_te"]
        + list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    )
    imp = plot_xgb_importance(final_pipe.named_steps["clf"], feat_names)
    imp.to_csv(PROC / "feature_importance.csv", index=False)

    summary = {
        name: {k: v for k, v in res.items() if k != "oof"}
        for name, res in results.items()
    }
    summary["base_rate"] = float(y.mean())
    summary["n"] = int(len(y))
    with open(PROC / "model_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    oof_df = pd.DataFrame({"DOT_NUMBER": df["DOT_NUMBER"].values, "y": y.values})
    for name, res in results.items():
        oof_df[f"proba_{name.replace(' ', '_')}"] = res["oof"]
    oof_df.to_csv(PROC / "cv_predictions.csv", index=False)

    print("\nFinal CV metrics:")
    for name, res in results.items():
        print(f"  {name}: AUC-ROC = {res['auc_roc_mean']:.4f} ± {res['auc_roc_std']:.4f} | "
              f"AUC-PR = {res['auc_pr_mean']:.4f} ± {res['auc_pr_std']:.4f}")


if __name__ == "__main__":
    main()

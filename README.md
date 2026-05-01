# Operational Visibility Gaps in Midwest Commercial Trucking

I513 Usable AI, Spring 2026 final project.
Philip Hess, Nick Frische, Blake Marcotte.

This repo pulls public FMCSA datasets from `data.transportation.gov`, builds a
carrier-level analysis table for Indiana and Illinois, trains three classifiers
to predict whether a carrier incurs a vehicle maintenance violation, runs
k-means clustering on per-inspection violation rates to find risk archetypes,
and renders a final PDF report.

## Layout

```
scripts/                    Pipeline (run in order)
  01_download_data.py       Pull census, inspections, crashes from Socrata API
  02_build_analysis_table.py  Aggregate + merge into a carrier-level table
  03_features_and_models.py   LogReg / RandomForest / XGBoost, 5-fold CV
  04_clustering.py            K-means on BASIC violation profile
  05_eda_figures.py           Rebuild EDA figures
  06_build_report.py          Render report HTML to PDF (headless Chrome)
data/
  raw/                      Cached API pulls (CSV)
  processed/                Carrier-level tables, JSON summaries, model outputs
figures/                    PNGs referenced by the report
report/
  final_report.html         Source for the PDF
  final_report.pdf          The report submission
```

## Run

```bash
pip install -r requirements.txt
python3 scripts/01_download_data.py        # ~1-2 min over public Socrata API
python3 scripts/02_build_analysis_table.py
python3 scripts/03_features_and_models.py  # ~30s on a laptop
python3 scripts/04_clustering.py
python3 scripts/05_eda_figures.py
python3 scripts/06_build_report.py         # needs Google Chrome installed
```

The PDF render step uses headless Chrome (no Pango/Cairo dependency). On macOS
the script auto-discovers `/Applications/Google Chrome.app`.

Optional: set `SODA_APP_TOKEN` to a Socrata app token for faster API pulls
(unauthenticated requests are throttled).

## Datasets

| Dataset | Socrata ID | Records (IN+IL) |
|--|--|--|
| Motor Carrier Census | `kjg3-diqy` | 90,128 |
| SMS Inspections | `rbkj-cgst` | 242,548 |
| Crash Data | `4wxs-vbns` | 18,359 |

Filtered to IN/IL events from 2024-03-01 onward (the FMCSA SMS 24-month window).

## Headline result

Cross-validated AUC-ROC of **0.764** for logistic regression, beating both
random forest (0.756) and XGBoost (0.742). Clustering identifies a
**1,215-carrier high-risk segment** (mostly mid-size, private, intrastate,
Indiana-based) with ~8x the per-inspection vehicle maintenance violation rate
of the low-risk cluster. See `report/final_report.pdf` for the full
discussion.

## Limitations

The biggest one is inspection selection bias: 76.5% of registered IN/IL
carriers had no roadside inspections in the SMS window, and the model is
trained only on the inspected cohort. Section 4.3 of the report and the
"two-stage selection model" item in §5 cover this.

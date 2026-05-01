"""
Render report/final_report.html to PDF using headless Chrome.
Chrome is used instead of WeasyPrint to avoid the Pango/Cairo
native-library dependency on macOS.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REPORT = ROOT / "report"
REPORT.mkdir(parents=True, exist_ok=True)

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if c and Path(c).exists():
            return c
    raise RuntimeError("Could not find Chrome / Chromium for headless rendering")


def main() -> None:
    chrome = find_chrome()
    html_path = (REPORT / "final_report.html").resolve()
    pdf_path = (REPORT / "final_report.pdf").resolve()
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        f"file://{html_path}",
    ]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print("STDERR:", res.stderr[-1000:])
        raise SystemExit(res.returncode)
    print(f"Wrote {pdf_path}")

    summary = json.loads((PROC / "summary_stats.json").read_text())
    models = json.loads((PROC / "model_results.json").read_text())
    cluster_meta = json.loads((PROC / "cluster_meta.json").read_text())
    key = {
        "registered_total": summary["registered_total"],
        "registered_in": summary["registered_in"],
        "registered_il": summary["registered_il"],
        "inspections_total": summary["inspections_total"],
        "crashes_total": summary["crashes_total"],
        "inspected_carriers": summary["inspected_carriers"],
        "inspected_pct": summary["inspected_pct"],
        "vh_maint_rate_overall": summary["vh_maint_rate_overall"],
        "vh_maint_rate_in": summary["vh_maint_rate_in"],
        "vh_maint_rate_il": summary["vh_maint_rate_il"],
        "logreg_auc": models["Logistic Regression"]["auc_roc_mean"],
        "rf_auc": models["Random Forest"]["auc_roc_mean"],
        "xgb_auc": models["XGBoost"]["auc_roc_mean"],
        "best_k": cluster_meta["best_k"],
    }
    (PROC / "key_numbers.json").write_text(json.dumps(key, indent=2))


if __name__ == "__main__":
    main()

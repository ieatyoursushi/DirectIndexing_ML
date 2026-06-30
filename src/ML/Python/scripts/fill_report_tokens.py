"""Replace the «TOKENS» in final_report.ipynb with numbers from the fresh artifacts.

Run from src/ML/Python AFTER `dotnet run mlnet-all`, BEFORE `dotnet run report`:
  uv run python -m scripts.fill_report_tokens

Keeps the report's prose numbers in lockstep with the rendered tables/plots: the
builder writes placeholder tokens, this fills them from data/lots.csv + the
artifact JSONs, then `dotnet run report` executes the notebook.
"""
from pathlib import Path

import json

import pandas as pd

from scripts import report_helpers as rh


def main() -> int:
    root = rh.repo_root()
    art = root / "data" / "artifacts-mlnet"

    cols = ["Y_Oracle", "Y_Soft_BT", "G_YTD", "Symbol"]
    lots = pd.read_csv(root / "data" / "lots.csv", usecols=cols)

    gbt_soft = rh.load_artifact(art, "gbt_soft_bt_metrics.json")
    gbt_orc  = rh.load_artifact(art, "gbt_oracle_metrics.json")
    rf_soft  = rh.load_artifact(art, "rf_soft_bt_metrics.json")
    lb_orc   = rh.load_artifact(art, "oracle_cv_leaderboard.json")
    logit_orc = next(m for m in lb_orc if m["modelName"] == "logistic")

    tokens = {
        "«N_ROWS»":            f"{len(lots):,}",
        "«N_CONSTIT»":         f"{lots['Symbol'].nunique()}",
        "«ORACLE_RATE»":       f"{lots['Y_Oracle'].mean():.2%}",
        "«SOFT_RATE»":         f"{(lots['Y_Soft_BT'].dropna() > 0).mean():.2%}",
        "«GYTD_OPEN»":         f"{(lots['G_YTD'] > 0).mean():.1%}",
        "«GBT_SOFT_CV»":       f"{gbt_soft['cvBestMeanPrAuc']:.3f}",
        "«GBT_SOFT_TEST»":     f"{gbt_soft['testPrAuc']:.3f}",
        "«GBT_ORACLE_TESTPR»": f"{gbt_orc['testPrAuc']:.3f}",
        "«LOGIT_ORACLE_CV»":   f"{logit_orc['meanCvPrAuc']:.3f}",
        "«RF_FF_BEST»":        f"{rf_soft['featureFraction']:g}",
    }

    nb_path = Path("notebooks/final_report.ipynb")
    nb = json.loads(nb_path.read_text())
    n_replaced = 0
    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for tok, val in tokens.items():
            if tok in src:
                n_replaced += src.count(tok)
                src = src.replace(tok, val)
        cell["source"] = src.splitlines(keepends=True)

    leftover = [t for t in tokens if any(
        t in ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
        for c in nb["cells"])]
    nb_path.write_text(json.dumps(nb, indent=1))
    print(f"[fill] replaced {n_replaced} token occurrences")
    print("[fill] values:", {k.strip('«»'): v for k, v in tokens.items()})
    if leftover:
        print(f"[fill] WARNING leftover tokens: {leftover}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""EDA plots from data/lots.csv. Pure rendering — no ML logic.

Inputs : path to lots.csv
Outputs: class_balance.png, corr_heatmap.png, feature_dist.png, summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUMERIC = [
    "L", "H", "S", "B", "W", "K",
    "G_YTD", "Sigma_TE", "WashClock",
    "R_t", "SigmaRange", "DeltaMA50", "DeltaMA200",
    "TaxAlpha", "DaysToYE",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── class balance ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    pos_oracle = float(df["Y_Oracle"].mean())
    pos_softbt = float((df["Y_Soft_BT"] > 0).mean()) if df["Y_Soft_BT"].notna().any() else 0.0
    ax.bar(["Y_Oracle = 1", "Y_Soft_BT > 0"], [pos_oracle, pos_softbt], color=["#3a7", "#37a"])
    ax.set_ylabel("positive rate")
    ax.set_title("Class balance")
    for i, v in enumerate([pos_oracle, pos_softbt]):
        ax.text(i, v, f"{v:.3%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=120)
    plt.close(fig)

    # ── correlation heatmap ─────────────────────────────────────────────────
    corr = df[NUMERIC].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(NUMERIC))); ax.set_xticklabels(NUMERIC, rotation=45, ha="right")
    ax.set_yticks(range(len(NUMERIC))); ax.set_yticklabels(NUMERIC)
    ax.set_title("Pearson correlation (numeric features)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / "corr_heatmap.png", dpi=120)
    plt.close(fig)

    # ── feature distributions (5 × 3 grid) ──────────────────────────────────
    fig, axes = plt.subplots(5, 3, figsize=(12, 14))
    for i, col in enumerate(NUMERIC):
        ax = axes[i // 3, i % 3]
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(series, bins=50, color="#357", alpha=0.85)
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=8)
    fig.suptitle("Feature distributions")
    fig.tight_layout()
    fig.savefig(out / "feature_dist.png", dpi=110)
    plt.close(fig)

    # ── summary.json ────────────────────────────────────────────────────────
    summary = {
        "n_rows": int(len(df)),
        "n_symbols": int(df["Symbol"].nunique()),
        "missing_per_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "positive_rate": {
            "Y_Oracle": pos_oracle,
            "Y_Soft_BT > 0": pos_softbt,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[eda] wrote class_balance.png, corr_heatmap.png, feature_dist.png, summary.json → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Exploratory Data Analysis — plots and summaries.

Produces the EDA artifacts required by the PSTAT 131/231 rubric:
  - class_balance.png       — visualises Y_Soft_BT > 0 (~31%) vs Y_Oracle (~0.8%)
  - corr_heatmap.png        — feature correlation matrix (motivates PCA)
  - feature_dist.png        — histograms per numeric feature
  - missing_data.json       — NaN counts per column
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import LABEL_COLS, NUMERIC_FEATURES, save_json


def class_balance(df: pd.DataFrame, out_dir: Path) -> dict:
    """Compare positive rates of Y_Oracle and Y_Soft_BT > 0."""
    y_oracle = df["Y_Oracle"].astype(int)
    softbt_mask = df["Y_Soft_BT"].notna()
    y_softbt = (df.loc[softbt_mask, "Y_Soft_BT"] > 0).astype(int)

    stats = {
        "Y_Oracle": {
            "n": int(len(y_oracle)),
            "positives": int(y_oracle.sum()),
            "positive_rate": float(y_oracle.mean()),
        },
        "Y_Soft_BT_binary": {
            "n": int(len(y_softbt)),
            "positives": int(y_softbt.sum()),
            "positive_rate": float(y_softbt.mean()),
            "n_NaN_dropped": int((~softbt_mask).sum()),
        },
    }

    # Plot side-by-side bars
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, s) in zip(axes, stats.items()):
        rate = s["positive_rate"]
        ax.bar(["negative", "positive"], [1 - rate, rate],
               color=["#888888", "#e74c3c"])
        ax.set_title(f"{name}\nn={s['n']:,}  pos={s['positives']:,} ({rate:.2%})")
        ax.set_ylabel("fraction")
        ax.set_ylim(0, 1)
        for i, v in enumerate([1 - rate, rate]):
            ax.text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=9)
    fig.suptitle("Class Balance — Y_Oracle vs Y_Soft_BT(>0)")
    fig.tight_layout()
    fig.savefig(out_dir / "class_balance.png", dpi=120)
    plt.close(fig)
    print(f"[eda] class_balance.png  →  Y_Oracle pos={stats['Y_Oracle']['positive_rate']:.2%}, "
          f"Y_Soft_BT_binary pos={stats['Y_Soft_BT_binary']['positive_rate']:.2%}")
    return stats


def correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> dict:
    """Pearson correlation matrix on numeric features. Motivates PCA."""
    X = df[NUMERIC_FEATURES].dropna()
    corr = X.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(NUMERIC_FEATURES)))
    ax.set_xticklabels(NUMERIC_FEATURES, rotation=45, ha="right")
    ax.set_yticks(range(len(NUMERIC_FEATURES)))
    ax.set_yticklabels(NUMERIC_FEATURES)
    # Overlay correlation values
    for i in range(len(NUMERIC_FEATURES)):
        for j in range(len(NUMERIC_FEATURES)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Numeric feature correlation matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "corr_heatmap.png", dpi=120)
    plt.close(fig)

    # Flag highly correlated pairs (|r| > 0.7, off-diagonal)
    high = []
    for i, a in enumerate(NUMERIC_FEATURES):
        for j, b in enumerate(NUMERIC_FEATURES):
            if i < j and abs(corr.values[i, j]) > 0.7:
                high.append({"a": a, "b": b, "r": float(corr.values[i, j])})
    print(f"[eda] corr_heatmap.png  →  {len(high)} pairs with |r|>0.7")
    for p in high:
        print(f"       {p['a']:<12s} ↔ {p['b']:<12s}  r = {p['r']:+.3f}")
    return {"high_correlation_pairs": high}


def feature_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram per numeric feature in a 4x4 grid."""
    n = len(NUMERIC_FEATURES)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()
    for ax, feat in zip(axes, NUMERIC_FEATURES):
        vals = df[feat].dropna().values
        # Clip extreme outliers for visualisation only (display ±3 IQR)
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        clipped = vals[(vals >= lo) & (vals <= hi)]
        ax.hist(clipped, bins=40, color="#3498db", alpha=0.8)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Numeric feature distributions  (outliers beyond ±3 IQR clipped for display)")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_dist.png", dpi=120)
    plt.close(fig)
    print(f"[eda] feature_dist.png  ({n} features)")


def missing_data_summary(df: pd.DataFrame) -> dict:
    """Per-column NaN counts. Rubric requires explicit discussion of missing data."""
    counts = df.isna().sum().sort_values(ascending=False)
    summary = {
        "total_rows": int(len(df)),
        "by_column": {col: int(n) for col, n in counts.items() if n > 0},
    }
    if not summary["by_column"]:
        summary["note"] = "No missing data in any column."
        print("[eda] missing_data: none")
    else:
        print("[eda] missing_data:")
        for col, n in summary["by_column"].items():
            pct = n / len(df)
            print(f"       {col:<12s}  {n:>8,}  ({pct:.2%})")
    return summary


def run_all(df: pd.DataFrame, out_dir: Path) -> None:
    """One-shot EDA — produces all plots and writes summary.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "class_balance": class_balance(df, out_dir),
        "correlation": correlation_heatmap(df, out_dir),
        "missing_data": missing_data_summary(df),
    }
    feature_distributions(df, out_dir)
    save_json(summary, out_dir / "summary.json")

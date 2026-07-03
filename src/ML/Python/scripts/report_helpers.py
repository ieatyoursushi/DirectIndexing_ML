"""Shared loaders + plot helpers for the final-report notebook.

The notebook (notebooks/final_report.ipynb) imports from here so its code
cells stay short and readable. Pure presentation — all ML numbers come from
the C# JSON artifacts in data/artifacts-mlnet/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#: Display names for model keys used in artifact filenames / leaderboards.
MODEL_NAMES = {
    "gbt":      "Gradient Boosted Trees",
    "rf":       "Random Forest",
    "logistic": "Logistic Regression (L2)",
    "elnet":    "Elastic Net",
    "linreg":   "Linear Regression",
}

# Schema v3 (v0.25): d = 17. G_YTD → three TaxLedger columns; TaxAlpha → TaxValue.
NUMERIC_FEATURES = [
    "L", "H", "S", "B", "W", "K",
    "RealizedGainsYTD", "LossCarryforward", "OrdinaryOffsetBudget",
    "Sigma_TE", "WashClock",
    "R_t", "SigmaRange", "DeltaMA50", "DeltaMA200",
    "TaxValue", "DaysToYE",
]


def repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default cwd) to the directory containing
    DirectIndexing.sln — mirrors PythonRunner.LocateRepoRoot, so the notebook
    runs identically under `dotnet run report` and interactively."""
    d = (start or Path.cwd()).resolve()
    for candidate in (d, *d.parents):
        if (candidate / "DirectIndexing.sln").exists():
            return candidate
    raise FileNotFoundError("DirectIndexing.sln not found above " + str(d))


def set_style() -> None:
    """Shared matplotlib look for every figure in the report."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })


def load_artifact(artifacts_dir: Path, name: str) -> dict | list:
    return json.loads((Path(artifacts_dir) / name).read_text())


def leaderboard_df(leaderboard: list[dict]) -> pd.DataFrame:
    """CV leaderboard JSON → tidy frame sorted by mean CV PR-AUC."""
    rows = []
    for m in leaderboard:
        folds = m["perFoldScores"]
        rows.append({
            "model": MODEL_NAMES.get(m["modelName"], m["modelName"]),
            "type": m["modelType"],
            "mean CV PR-AUC": m["meanCvPrAuc"],
            "fold std": float(np.std(folds, ddof=1)),
            "configs tried": len(m["allConfigs"]),
            "champion": "★" if m.get("isChampion") else "",
        })
    return (pd.DataFrame(rows)
              .sort_values("mean CV PR-AUC", ascending=False)
              .reset_index(drop=True))


def tuning_table(entry: dict) -> pd.DataFrame:
    """One leaderboard entry's allConfigs → grid-search results frame."""
    rows = [{**c["params"], "mean CV PR-AUC": c["meanPrAuc"]} for c in entry["allConfigs"]]
    return (pd.DataFrame(rows)
              .sort_values("mean CV PR-AUC", ascending=False)
              .reset_index(drop=True))


def lb_entry(leaderboard: list[dict], model_name: str) -> dict:
    return next(m for m in leaderboard if m["modelName"] == model_name)


def downsample_curve(curve: list[dict], max_points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """ROC/PR curves from C# carry ~30k threshold points; thin them for plotting."""
    xs = np.array([p["x"] for p in curve])
    ys = np.array([p["y"] for p in curve])
    if len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs, ys = xs[idx], ys[idx]
    return xs, ys


def plot_roc_pr_overlay(metrics_by_model: dict[str, dict], target: str) -> plt.Figure:
    """Overlay ROC and PR curves for several models on one target."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    for key, m in metrics_by_model.items():
        label = MODEL_NAMES.get(key, key)
        rx, ry = downsample_curve(m["rocCurve"])
        ax_roc.plot(rx, ry, lw=2, label=f"{label} (AUC {m['testRocAuc']:.3f})")
        px, py = downsample_curve(m["prCurve"])
        ax_pr.plot(px, py, lw=2, label=f"{label} (AP {m['testPrAuc']:.3f})")

    ax_roc.plot([0, 1], [0, 1], color="#aaa", lw=1, ls="--")
    ax_roc.set_xlabel("False positive rate"); ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title(f"ROC — test set, target = {target}")
    ax_roc.legend(loc="lower right", fontsize=8)

    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_ylim(0, 1.02)
    ax_pr.set_title(f"Precision–Recall — test set, target = {target}")
    ax_pr.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def confusion_df(metrics: dict, which: str = "confusionAt05") -> pd.DataFrame:
    """Confusion-matrix dict → 2×2 labeled frame."""
    c = metrics[which]
    return pd.DataFrame(
        [[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
        index=pd.Index(["actual 0", "actual 1"]),
        columns=pd.Index(["predicted 0", "predicted 1"]),
    )


def test_metrics_table(metrics_by_model: dict[str, dict]) -> pd.DataFrame:
    """Champion test-set metrics side by side."""
    rows = []
    for key, m in metrics_by_model.items():
        rows.append({
            "model": MODEL_NAMES.get(key, key),
            "test ROC-AUC": m["testRocAuc"],
            "test PR-AUC": m["testPrAuc"],
            "F1 @ 0.5": m["f1At05"],
            "F1 @ best threshold": m["f1AtBest"],
            "best threshold": m["bestThreshold"],
        })
    return pd.DataFrame(rows).set_index("model")


def plot_cv_folds(leaderboard: list[dict], target: str) -> plt.Figure:
    """Per-fold CV PR-AUC for every model — spread shows tuning stability."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    entries = sorted(leaderboard, key=lambda m: m["meanCvPrAuc"], reverse=True)
    for i, m in enumerate(entries):
        folds = m["perFoldScores"]
        ax.scatter([i] * len(folds), folds, alpha=0.7, s=40, zorder=3)
        ax.scatter([i], [m["meanCvPrAuc"]], color="black", marker="_", s=600, zorder=4)
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels(
        [MODEL_NAMES.get(m["modelName"], m["modelName"]) + (" ★" if m.get("isChampion") else "")
         for m in entries],
        rotation=15, ha="right")
    ax.set_ylabel("PR-AUC per CV fold")
    ax.set_title(f"5-fold cross-validation scores — target = {target} (bar = fold mean)")
    fig.tight_layout()
    return fig

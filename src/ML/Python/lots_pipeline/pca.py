"""Principal Component Analysis on the numeric feature space.

Standardises features first (StandardScaler), then fits a full PCA. Outputs:
  - pca_components.json   — variance explained per component, n_kept for 95% cumulative
  - pca_loadings.csv      — component × original-feature loadings matrix
  - pca_scree.png         — variance explained + cumulative curve

Decision rule: keep top-k components for ≥95% cumulative variance.
Standardisation is mandatory before PCA — otherwise high-variance raw features
(e.g. G_YTD in dollars) would dominate the principal axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .io import NUMERIC_FEATURES, save_csv, save_json, save_model


def fit_pca(df: pd.DataFrame, variance_threshold: float = 0.95) -> dict:
    """Fit PCA on standardised numeric features.

    Returns a dict with the fitted scaler+pca, variance ratios, and n_kept.
    """
    X = df[NUMERIC_FEATURES].dropna().values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=len(NUMERIC_FEATURES), svd_solver="full", random_state=42)
    pca.fit(X_std)

    cum = np.cumsum(pca.explained_variance_ratio_)
    n_kept = int(np.searchsorted(cum, variance_threshold) + 1)

    print(f"[pca] fitted on {len(X_std):,} rows × {len(NUMERIC_FEATURES)} features")
    print(f"[pca] variance retained: top-{n_kept} components → {cum[n_kept - 1]:.2%} cumulative")
    print(f"[pca] per-component variance (first 8): "
          + ", ".join(f"{v:.3f}" for v in pca.explained_variance_ratio_[:8]))

    return {
        "scaler": scaler,
        "pca": pca,
        "n_kept": n_kept,
        "variance_threshold": variance_threshold,
    }


def write_artifacts(fit: dict, out_dir: Path, results_dir: Path) -> None:
    """Write all PCA outputs to disk."""
    out_dir = Path(out_dir);     out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)

    pca: PCA = fit["pca"]
    n_kept = fit["n_kept"]

    # pca_components.json — top-level summary
    components_summary = {
        "n_features": len(NUMERIC_FEATURES),
        "variance_threshold": fit["variance_threshold"],
        "n_kept": n_kept,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "feature_names": NUMERIC_FEATURES,
    }
    save_json(components_summary, out_dir / "pca_components.json")

    # pca_loadings.csv — component × feature loadings matrix
    loadings = pd.DataFrame(
        pca.components_,
        index=[f"PC{i + 1}" for i in range(pca.n_components_)],
        columns=NUMERIC_FEATURES,
    )
    save_csv(loadings, out_dir / "pca_loadings.csv")

    # Scree plot
    _scree_plot(pca, n_kept, fit["variance_threshold"], results_dir / "pca_scree.png")

    # Fitted scaler+pca for reuse (not strictly needed by C# but useful for Python replay)
    save_model({"scaler": fit["scaler"], "pca": pca, "n_kept": n_kept},
               out_dir / "pca_model.joblib")


def _scree_plot(pca: PCA, n_kept: int, threshold: float, out_path: Path) -> None:
    ratios = pca.explained_variance_ratio_
    cum    = np.cumsum(ratios)
    xs     = np.arange(1, len(ratios) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(xs, ratios, color="#3498db", alpha=0.8, label="per-component")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Variance explained (per component)", color="#3498db")
    ax1.tick_params(axis="y", labelcolor="#3498db")
    ax1.set_xticks(xs)

    ax2 = ax1.twinx()
    ax2.plot(xs, cum, marker="o", color="#e74c3c", label="cumulative")
    ax2.axhline(threshold, ls="--", color="#888888", alpha=0.7,
                label=f"{threshold:.0%} threshold")
    ax2.axvline(n_kept, ls="--", color="#888888", alpha=0.7)
    ax2.set_ylabel("Cumulative variance explained", color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")
    ax2.set_ylim(0, 1.05)

    ax1.set_title(f"PCA scree plot  —  top-{n_kept} components → {cum[n_kept - 1]:.1%} cumulative variance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[pca] scree → {out_path}")

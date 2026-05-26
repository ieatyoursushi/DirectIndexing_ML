"""K-means clustering for the 'dimension basket of lots' substitute-stock concept.

Operates on **per-symbol** aggregates (not per-row snapshots). For each S&P 500 ticker,
compute the mean of asset-level features (R_t, SigmaRange, DeltaMA50, DeltaMA200)
plus its Sector one-hot, then cluster the resulting ~503-row matrix.

The resulting Symbol → cluster_id map enables a v0.3 simulation enhancement: at
harvest time, instead of waiting 30 days (wash-sale), buy a substitute stock from
the same cluster to maintain factor exposure and reduce tracking error during the
wash-sale window.

For v0.1: train + write artifacts only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .io import save_csv, save_json, save_model


# Asset-level features used for clustering (lot/portfolio-level features excluded —
# they vary day-to-day; we want a stable per-symbol fingerprint)
ASSET_FEATURES = ["R_t", "SigmaRange", "DeltaMA50", "DeltaMA200"]


def aggregate_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """One row per Symbol with mean asset features + Sector."""
    grouped = df.groupby("Symbol").agg(
        {feat: "mean" for feat in ASSET_FEATURES} | {"Sector": "first"}
    ).reset_index()
    grouped = grouped.dropna(subset=ASSET_FEATURES)
    grouped["Sector"] = grouped["Sector"].fillna("Unknown").astype(str)
    print(f"[kmeans] aggregated to {len(grouped)} symbols × {len(ASSET_FEATURES)} features + Sector "
          f"(sectors: {grouped['Sector'].nunique()} unique)")
    return grouped


def _build_design_matrix(per_symbol: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Scale numerics + one-hot Sector. Returns (X, feature_names)."""
    scaler = StandardScaler()
    X_num  = scaler.fit_transform(per_symbol[ASSET_FEATURES].values)

    ohe = OneHotEncoder(drop="first", sparse_output=False)
    X_cat = ohe.fit_transform(per_symbol[["Sector"]].values)
    cat_names = [f"Sector={s}" for s in ohe.categories_[0][1:]]

    X = np.hstack([X_num, X_cat])
    names = ASSET_FEATURES + cat_names
    return X, names


def tune_k(per_symbol: pd.DataFrame, k_range: list[int]) -> dict:
    """Elbow + silhouette across candidate k. Returns best k and the curves."""
    X, _ = _build_design_matrix(per_symbol)

    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(X, labels)) if k > 1 else float("nan")
        silhouettes.append(sil)
        print(f"[kmeans] k={k:<3d}  inertia={km.inertia_:>10.2f}  silhouette={sil:.3f}")

    # Simple "elbow" heuristic: pick k where marginal inertia drop is small
    # AND silhouette score is reasonable. For a clean run pick max-silhouette k.
    best_k = k_range[int(np.argmax(silhouettes))]
    print(f"[kmeans] selected k = {best_k}  (max silhouette = {max(silhouettes):.3f})")

    return {
        "k_range": k_range,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": best_k,
    }


def fit_final(per_symbol: pd.DataFrame, k: int) -> dict:
    """Fit the final K-means at the chosen k."""
    X, feature_names = _build_design_matrix(per_symbol)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    return {
        "model": km,
        "labels": labels,
        "feature_names": feature_names,
        "symbols": per_symbol["Symbol"].tolist(),
        "sectors": per_symbol["Sector"].tolist(),
    }


def write_artifacts(tune_result: dict, fit_result: dict, out_dir: Path, results_dir: Path) -> None:
    out_dir = Path(out_dir);     out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)

    # cluster_assignments.json  — Symbol → cluster_id
    assignments = {sym: int(lab) for sym, lab in zip(fit_result["symbols"], fit_result["labels"])}
    save_json(assignments, out_dir / "cluster_assignments.json")

    # kmeans_centers.json — centroids in standardised feature space
    centers = fit_result["model"].cluster_centers_
    centers_obj = {
        "k": int(fit_result["model"].n_clusters),
        "feature_names": fit_result["feature_names"],
        "centers": centers.tolist(),
        "inertia": float(fit_result["model"].inertia_),
    }
    save_json(centers_obj, out_dir / "kmeans_centers.json")

    # Diagnostic table — Symbol, Sector, cluster_id (useful for sanity-checking)
    diag = pd.DataFrame({
        "Symbol":     fit_result["symbols"],
        "Sector":     fit_result["sectors"],
        "cluster_id": fit_result["labels"],
    })
    save_csv(diag.sort_values(["cluster_id", "Sector", "Symbol"]),
             out_dir / "cluster_assignments.csv")

    # Elbow plot
    _elbow_plot(tune_result, results_dir / "kmeans_elbow.png")

    # Fitted model + tuning curve for replay
    save_model({"model": fit_result["model"],
                "tune": tune_result,
                "feature_names": fit_result["feature_names"]},
               out_dir / "kmeans_model.joblib")


def _elbow_plot(tune_result: dict, out_path: Path) -> None:
    ks    = tune_result["k_range"]
    iner  = tune_result["inertias"]
    sil   = tune_result["silhouettes"]
    best  = tune_result["best_k"]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ks, iner, marker="o", color="#3498db", label="inertia (WCSS)")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (WCSS)", color="#3498db")
    ax1.tick_params(axis="y", labelcolor="#3498db")
    ax1.set_xticks(ks)

    ax2 = ax1.twinx()
    ax2.plot(ks, sil, marker="s", color="#e74c3c", label="silhouette")
    ax2.axvline(best, ls="--", color="#888888", alpha=0.7, label=f"best k = {best}")
    ax2.set_ylabel("Silhouette score", color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")

    ax1.set_title("K-means tuning: inertia (elbow) + silhouette across k")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[kmeans] elbow → {out_path}")

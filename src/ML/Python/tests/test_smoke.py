"""End-to-end smoke test on a tiny synthetic frame.

Verifies the pipeline plumbing without requiring the real lots.csv.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lots_pipeline import eda, io, kmeans, logistic, pca, targets


# ── Synthetic data factory ───────────────────────────────────────────────────

def _make_synthetic_lots(n_rows: int = 2000, n_symbols: int = 20, seed: int = 42) -> pd.DataFrame:
    """Mimic the LotStateVector schema with random data."""
    rng = np.random.default_rng(seed)
    sectors = ["Tech", "Financials", "Energy", "Health", "Consumer"]
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]

    df = pd.DataFrame({
        "L":          rng.normal(-0.01, 0.03, n_rows).astype("float32"),
        "H":          rng.integers(1, 300, n_rows).astype("int32"),
        "S":          rng.integers(0, 2, n_rows).astype("int32"),
        "B":          rng.uniform(50, 500, n_rows).astype("float32"),
        "W":          rng.uniform(0.001, 0.01, n_rows).astype("float32"),
        "K":          rng.integers(1, 5, n_rows).astype("int32"),
        "G_YTD":      rng.normal(50000, 200000, n_rows).astype("float32"),
        "Sigma_TE":   rng.uniform(0.01, 0.06, n_rows).astype("float32"),
        "WashClock":  rng.integers(0, 50, n_rows).astype("int32"),
        "R_t":        rng.normal(0, 0.02, n_rows).astype("float32"),
        "SigmaRange": rng.uniform(0.005, 0.03, n_rows).astype("float32"),
        "DeltaMA50":  rng.normal(0, 0.05, n_rows).astype("float32"),
        "DeltaMA200": rng.normal(0, 0.10, n_rows).astype("float32"),
        "TaxAlpha":   rng.uniform(0, 1000, n_rows).astype("float32"),
        "DaysToYE":   rng.integers(1, 365, n_rows).astype("int32"),
        "Y_Oracle":   rng.choice([0, 1], n_rows, p=[0.99, 0.01]).astype("int32"),
        "Y_Soft_GBM": rng.uniform(0, 1, n_rows).astype("float32"),
        "Y_Soft_BT":  rng.uniform(0, 0.5, n_rows).astype("float32"),
        "Symbol":     rng.choice(symbols, n_rows),
        "Sector":     rng.choice(sectors, n_rows),
        "Timestep":   rng.integers(200, 500, n_rows).astype("int32"),
    })
    # Inject some NaN at end-of-window for Y_Soft_BT (mirrors real data)
    df.loc[df.index[-30:], "Y_Soft_BT"] = np.nan
    return df


@pytest.fixture
def tmp_dirs():
    """Temp out_dir + results_dir, cleaned up automatically."""
    base = Path(tempfile.mkdtemp(prefix="lots_pipeline_test_"))
    out_dir     = base / "artifacts"
    results_dir = base / "results"
    out_dir.mkdir()
    results_dir.mkdir()
    yield out_dir, results_dir
    shutil.rmtree(base)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_io_save_json(tmp_dirs):
    out_dir, _ = tmp_dirs
    io.save_json({"a": 1, "b": 2}, out_dir / "test.json")
    assert (out_dir / "test.json").exists()


def test_targets_softbt_drops_nan():
    df = _make_synthetic_lots()
    df_clean, y = targets.get_target(df, "soft_bt")
    assert df_clean["Y_Soft_BT"].notna().all()
    assert len(y) == len(df_clean)
    assert set(y.unique()).issubset({0, 1})


def test_targets_oracle_passthrough():
    df = _make_synthetic_lots()
    df_clean, y = targets.get_target(df, "oracle")
    assert set(y.unique()).issubset({0, 1})


def test_eda_runs(tmp_dirs):
    out_dir, results_dir = tmp_dirs
    df = _make_synthetic_lots()
    eda.run_all(df, results_dir)
    assert (results_dir / "class_balance.png").exists()
    assert (results_dir / "corr_heatmap.png").exists()
    assert (results_dir / "feature_dist.png").exists()
    assert (results_dir / "summary.json").exists()


def test_pca_pipeline(tmp_dirs):
    out_dir, results_dir = tmp_dirs
    df = _make_synthetic_lots()
    fit = pca.fit_pca(df, variance_threshold=0.95)
    pca.write_artifacts(fit, out_dir, results_dir)
    assert (out_dir / "pca_components.json").exists()
    assert (out_dir / "pca_loadings.csv").exists()
    assert (results_dir / "pca_scree.png").exists()


def test_kmeans_pipeline(tmp_dirs):
    out_dir, results_dir = tmp_dirs
    df = _make_synthetic_lots()
    per_symbol = kmeans.aggregate_by_symbol(df)
    tune       = kmeans.tune_k(per_symbol, k_range=[2, 3, 5])
    fit        = kmeans.fit_final(per_symbol, tune["best_k"])
    kmeans.write_artifacts(tune, fit, out_dir, results_dir)
    assert (out_dir / "cluster_assignments.json").exists()
    assert (out_dir / "kmeans_centers.json").exists()
    assert (results_dir / "kmeans_elbow.png").exists()


def test_logistic_oracle(tmp_dirs):
    """Tiny logistic run on synthetic data — checks pipeline plumbing."""
    out_dir, results_dir = tmp_dirs
    df = _make_synthetic_lots(n_rows=3000)
    df_clean, y = targets.get_target(df, "oracle")
    # Force enough positives for stratification
    if y.sum() < 5:
        idx = np.random.default_rng(0).choice(len(y), 20, replace=False)
        y = y.copy(); y.loc[idx] = 1
    metrics = logistic.train_and_eval(df_clean, y, "oracle", out_dir, results_dir)
    assert (out_dir / "logistic_oracle_model.joblib").exists()
    assert (out_dir / "logistic_oracle_metrics.json").exists()
    assert (out_dir / "logistic_oracle_coefficients.csv").exists()
    assert "roc_auc" in metrics["test"]

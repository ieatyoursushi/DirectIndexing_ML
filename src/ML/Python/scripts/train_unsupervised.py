"""Unsupervised training entry point — PCA + K-means.

Usage (from src/ML/Python/):
    uv run python -m scripts.train_unsupervised \\
        --in ../../../data/lots.csv --out ../../../data/artifacts/ \\
        --results ../../Export/eda/
"""

import argparse
from pathlib import Path

from lots_pipeline import io, kmeans, pca


def main() -> int:
    ap = argparse.ArgumentParser(description="Train PCA + K-means on lots.csv")
    ap.add_argument("--in",      dest="input_path",  default="../../../data/lots.csv")
    ap.add_argument("--out",     dest="output_dir",  default="../../../data/artifacts/")
    ap.add_argument("--results", dest="results_dir", default="../../Export/eda/")
    ap.add_argument("--variance-threshold", type=float, default=0.95)
    ap.add_argument("--k-range", type=int, nargs="+", default=[5, 10, 15, 20, 25])
    args = ap.parse_args()

    df = io.load_lots(args.input_path)
    out_dir     = Path(args.output_dir)
    results_dir = Path(args.results_dir)

    # ── PCA ─────────────────────────────────────────────────────────────────
    print("\n=== PCA ===")
    pca_fit = pca.fit_pca(df, variance_threshold=args.variance_threshold)
    pca.write_artifacts(pca_fit, out_dir, results_dir)

    # ── K-means ────────────────────────────────────────────────────────────
    print("\n=== K-means ===")
    per_symbol = kmeans.aggregate_by_symbol(df)
    tune       = kmeans.tune_k(per_symbol, args.k_range)
    fit        = kmeans.fit_final(per_symbol, tune["best_k"])
    kmeans.write_artifacts(tune, fit, out_dir, results_dir)

    print("\n[train_unsupervised] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

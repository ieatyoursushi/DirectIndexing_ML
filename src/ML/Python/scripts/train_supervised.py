"""Supervised training entry point — logistic regression with CV.

Usage (from src/ML/Python/):
    uv run python -m scripts.train_supervised \\
        --in ../../../data/lots.csv --out ../../../data/artifacts/ \\
        --results ../../Export/models/ --target soft_bt
    uv run python -m scripts.train_supervised \\
        --in ../../../data/lots.csv --out ../../../data/artifacts/ \\
        --results ../../Export/models/ --target oracle
"""

import argparse
from pathlib import Path

from lots_pipeline import io, logistic, targets


def main() -> int:
    ap = argparse.ArgumentParser(description="Train logistic regression on lots.csv")
    ap.add_argument("--in",      dest="input_path",  default="../../../data/lots.csv")
    ap.add_argument("--out",     dest="output_dir",  default="../../../data/artifacts/")
    ap.add_argument("--results", dest="results_dir", default="../../Export/models/")
    ap.add_argument("--target",  dest="target",      default="soft_bt",
                    choices=["soft_bt", "oracle"])
    args = ap.parse_args()

    df = io.load_lots(args.input_path)
    df_clean, y = targets.get_target(df, args.target)

    metrics = logistic.train_and_eval(
        df_clean, y,
        target_name=args.target,
        out_dir=Path(args.output_dir),
        results_dir=Path(args.results_dir),
    )

    # One-line summary for the orchestrator log
    print(f"\n[train_supervised] target={args.target}  "
          f"ROC-AUC={metrics['test']['roc_auc']:.4f}  "
          f"PR-AUC={metrics['test']['pr_auc']:.4f}  "
          f"F1@best={metrics['test']['f1_tuned']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

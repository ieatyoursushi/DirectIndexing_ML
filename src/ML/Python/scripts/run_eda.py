"""EDA entry point — produces all exploratory plots and summary.

Usage (from src/ML/Python/):
    uv run python -m scripts.run_eda \\
        --in ../../../data/lots.csv --out ../../Export/eda/
"""

import argparse
from pathlib import Path

from lots_pipeline import eda, io


def main() -> int:
    ap = argparse.ArgumentParser(description="Run EDA on lots.csv")
    ap.add_argument("--in",  dest="input_path",  default="../../../data/lots.csv")
    ap.add_argument("--out", dest="output_dir",  default="../../Export/eda/")
    args = ap.parse_args()

    df = io.load_lots(args.input_path)
    eda.run_all(df, Path(args.output_dir))
    print("[run_eda] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

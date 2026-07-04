"""Report layer orchestrator — the final pipeline stage.

Invoked by `dotnet run report`:
  1. preflight: verify lots.csv + every ML.NET artifact the notebook reads,
  2. generate the codebook (scripts.codebook),
  3. execute notebooks/final_report.ipynb with nbclient,
  4. export the executed notebook to HTML with embedded figures.

Does NOT train models. If artifacts are missing it exits 2 with the list and
points at `dotnet run mlnet-all` / `dotnet run report-all`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from scripts import codebook

REQUIRED_ARTIFACTS = [
    "soft_bt_cv_leaderboard.json",
    "oracle_cv_leaderboard.json",
    "gbt_soft_bt_metrics.json",
    "gbt_oracle_metrics.json",
    "rf_soft_bt_metrics.json",
    "rf_oracle_metrics.json",
    "linreg_soft_bt_metrics.json",
    "linreg_soft_bt_coefficients.csv",
    "pca_scree.json",
    "kmeans_elbow.json",
    "tax_value_regression_metrics.json",   # v0.25: Y_TaxValue regression section
]

# v0.25 ablation inputs: the gated arm's leaderboards (in artifacts-mlnet-gated/,
# a sibling of --artifacts) and the gated dataset for the diagnosis chapter.
REQUIRED_GATED_ARTIFACTS = [
    "oracle_cv_leaderboard.json",
    "soft_bt_cv_leaderboard.json",
]


def preflight(lots: Path, artifacts: Path, notebook: Path) -> None:
    lots_gated = lots.parent / "lots_gated.csv"
    art_gated  = artifacts.parent / "artifacts-mlnet-gated"

    missing = [str(p) for p in [lots, lots_gated, notebook] if not p.exists()]
    missing += [str(artifacts / a) for a in REQUIRED_ARTIFACTS if not (artifacts / a).exists()]
    missing += [str(art_gated / a) for a in REQUIRED_GATED_ARTIFACTS
                if not (art_gated / a).exists()]
    if missing:
        print("[report] missing required inputs:", file=sys.stderr)
        for m in missing:
            print(f"[report]   {m}", file=sys.stderr)
        print("[report] run `dotnet run mlnet-all` + `mlnet-tax` on the canonical arm, "
              "`simulate --oracle=gated` + retrains for the ablation arm "
              "(keep them in data/artifacts-mlnet-gated/), then retry.",
              file=sys.stderr)
        sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lots",      required=True, help="path to data/lots.csv")
    ap.add_argument("--artifacts", required=True, help="path to data/artifacts-mlnet/")
    ap.add_argument("--notebook",  required=True, help="authored notebook to execute")
    ap.add_argument("--out",       required=True, help="output directory")
    args = ap.parse_args()

    lots      = Path(args.lots)
    artifacts = Path(args.artifacts)
    notebook  = Path(args.notebook)
    out_dir   = Path(args.out)

    preflight(lots, artifacts, notebook)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path, html_cb = codebook.generate(lots, out_dir)
    print(f"[report] codebook: {md_path}, {html_cb}")

    print(f"[report] executing {notebook} ...")
    nb = nbformat.read(notebook, as_version=4)
    client = NotebookClient(
        nb,
        kernel_name="python3",
        timeout=600,
        # Run cells with cwd = src/ML/Python so `scripts` imports and the
        # repo-root walk behave the same as every other pipeline stage.
        resources={"metadata": {"path": str(Path.cwd())}},
    )
    client.execute()

    executed = out_dir / "final_report.ipynb"
    nbformat.write(nb, str(executed))

    exporter = HTMLExporter()
    exporter.embed_images = True
    body, _ = exporter.from_notebook_node(nb)
    html_path = out_dir / "final_report.html"
    html_path.write_text(body)

    print(f"[report] wrote {executed}")
    print(f"[report] wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

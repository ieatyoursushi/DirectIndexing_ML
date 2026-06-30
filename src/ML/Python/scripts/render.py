"""Render PNG plots + HTML report from C# JSON artifacts.

Reads from data/artifacts-mlnet/, writes PNGs to src/Export/{eda,models}-mlnet/
and an index.html under src/Export/models-mlnet/. Pure rendering — no ML logic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Template


def load(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text())


def plot_roc_pr(metrics: dict, out: Path, target: str) -> None:
    roc = metrics["rocCurve"]
    pr  = metrics["prCurve"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    rx = [p["x"] for p in roc]; ry = [p["y"] for p in roc]
    axes[0].plot(rx, ry, color="#357", lw=2)
    axes[0].plot([0, 1], [0, 1], color="#aaa", lw=1, ls="--")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC  (AUC = {metrics['testRocAuc']:.3f})")

    px = [p["x"] for p in pr]; py = [p["y"] for p in pr]
    axes[1].plot(px, py, color="#3a7", lw=2)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR  (AP = {metrics['testPrAuc']:.3f})")

    fig.suptitle(f"logistic — target = {target}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"logistic_{target}_curves.png", dpi=120)
    plt.close(fig)


def plot_scree(scree: dict, out: Path) -> None:
    ks = scree["k"]
    ev = scree["explainedVariance"]
    cv = scree["cumulativeVariance"]
    n_kept = scree["nKept"]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(ks, ev, color="#357", alpha=0.7, label="explained variance")
    ax1.set_xlabel("principal component")
    ax1.set_ylabel("explained variance", color="#357")
    ax2 = ax1.twinx()
    ax2.plot(ks, cv, color="#a37", marker="o", lw=2, label="cumulative")
    ax2.axhline(scree.get("threshold", 0.95), color="#aaa", ls="--", lw=1)
    ax2.set_ylabel("cumulative variance", color="#a37")
    ax2.set_ylim(0, 1.02)
    ax1.set_title(f"PCA scree  (keep top {n_kept} for ≥{scree.get('threshold', 0.95):.0%} variance)")
    fig.tight_layout()
    fig.savefig(out / "pca_scree.png", dpi=120)
    plt.close(fig)


def plot_elbow(elbow: dict, out: Path) -> None:
    ks  = elbow["ks"]
    sil = elbow["silhouette"]
    inr = elbow["inertia"]
    bestK = elbow["bestK"]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(ks, inr, color="#357", marker="o", lw=2, label="inertia")
    ax1.set_xlabel("k"); ax1.set_ylabel("inertia", color="#357")
    ax2 = ax1.twinx()
    ax2.plot(ks, sil, color="#a37", marker="s", lw=2, label="silhouette")
    ax2.set_ylabel("silhouette", color="#a37")
    ax1.axvline(bestK, color="#aaa", ls="--", lw=1)
    ax1.set_title(f"K-means elbow + silhouette  (best k = {bestK})")
    fig.tight_layout()
    fig.savefig(out / "kmeans_elbow.png", dpi=120)
    plt.close(fig)


HTML = Template("""
<!doctype html>
<html><head>
<meta charset="utf-8">
<title>ML.NET pipeline — v0.1 report</title>
<style>
 body { font-family: -apple-system, system-ui, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }
 h1 { border-bottom: 2px solid #357; padding-bottom: .3em; }
 h2 { color: #357; margin-top: 2em; }
 table { border-collapse: collapse; margin: 1em 0; }
 th, td { border: 1px solid #ddd; padding: 6px 12px; text-align: right; }
 th { background: #f5f5f5; }
 img { max-width: 100%; border: 1px solid #eee; }
 .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }
 .note { color: #777; font-size: .9em; }
</style></head><body>

<h1>ML.NET pipeline — v0.1 report</h1>
<p class="note">Schema-first, typed pipeline. C# emits JSON; this page renders it.</p>

<h2>Class balance &amp; data summary</h2>
<div class="grid">
  <img src="../eda-mlnet/class_balance.png" alt="class balance">
  <img src="../eda-mlnet/corr_heatmap.png"  alt="correlation heatmap">
</div>
<img src="../eda-mlnet/feature_dist.png" alt="feature distributions">

<h2>Unsupervised — PCA &amp; K-means</h2>
<div class="grid">
  <img src="../eda-mlnet/pca_scree.png"    alt="PCA scree">
  <img src="../eda-mlnet/kmeans_elbow.png" alt="K-means elbow">
</div>

{% for target, m in metrics.items() %}
<h2>Supervised — logistic (target = {{ target }})</h2>
<img src="logistic_{{ target }}_curves.png" alt="ROC + PR">
<table>
  <tr><th>metric</th><th>value</th></tr>
  <tr><td>best C</td><td>{{ "%.4f"|format(m.bestC) }}</td></tr>
  <tr><td>L2 used (1/C)</td><td>{{ "%.4f"|format(m.l2Used) }}</td></tr>
  <tr><td>CV PR-AUC (mean over 5 folds)</td><td>{{ "%.4f"|format(m.cvBestMeanPrAuc) }}</td></tr>
  <tr><td>test ROC-AUC</td><td>{{ "%.4f"|format(m.testRocAuc) }}</td></tr>
  <tr><td>test PR-AUC</td><td>{{ "%.4f"|format(m.testPrAuc) }}</td></tr>
  <tr><td>F1 at threshold 0.5</td><td>{{ "%.4f"|format(m.f1At05) }}</td></tr>
  <tr><td>F1 at best threshold ({{ "%.3f"|format(m.bestThreshold) }})</td><td>{{ "%.4f"|format(m.f1AtBest) }}</td></tr>
</table>
<p class="note">Confusion at 0.5 — TP={{ m.confusionAt05.tp }} FP={{ m.confusionAt05.fp }} TN={{ m.confusionAt05.tn }} FN={{ m.confusionAt05.fn }}</p>
<p class="note">Confusion at best — TP={{ m.confusionAtBest.tp }} FP={{ m.confusionAtBest.fp }} TN={{ m.confusionAtBest.tn }} FN={{ m.confusionAtBest.fn }}</p>
{% endfor %}

</body></html>
""")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts",   required=True)
    ap.add_argument("--eda-out",     required=True)
    ap.add_argument("--models-out",  required=True)
    args = ap.parse_args()

    art = Path(args.artifacts)
    eda = Path(args.eda_out); eda.mkdir(parents=True, exist_ok=True)
    mdl = Path(args.models_out); mdl.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict] = {}
    for target in ("oracle", "soft_bt"):
        m = load(art / f"logistic_{target}_metrics.json")
        if m is not None:
            plot_roc_pr(m, mdl, target)
            metrics[target] = m

    scree = load(art / "pca_scree.json")
    if scree is not None:
        plot_scree(scree, eda)

    elbow = load(art / "kmeans_elbow.json")
    if elbow is not None:
        plot_elbow(elbow, eda)

    (mdl / "index.html").write_text(HTML.render(metrics=metrics))
    print(f"[render] wrote PNGs to {eda}, {mdl} + index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render PNG plots + HTML summary from C# JSON artifacts.

Reads from data/artifacts-mlnet*/, writes PNGs to src/Export/models-mlnet/ and an
index.html beside them. Covers the two supervised models kept after the pre-v0.3
downsizing (GBT = champion, logistic = linear control). Pure rendering — no ML logic.
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


MODELS = ("gbt", "logistic")
TARGETS = ("oracle", "soft_bt")


def plot_roc_pr(metrics: dict, out: Path, model: str, target: str) -> None:
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

    fig.suptitle(f"{model} — target = {target}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"{model}_{target}_curves.png", dpi=120)
    plt.close(fig)


HTML = Template("""
<!doctype html>
<html><head>
<meta charset="utf-8">
<title>ML.NET pipeline — GBT vs logistic</title>
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

<h1>ML.NET pipeline — GBT (champion) vs logistic (linear control)</h1>
<p class="note">Schema-first, typed pipeline. C# emits JSON; this page renders it.
Report ROC-AUC, PR-AUC and the test positive rate together (standing rule 5).</p>

<h2>Class balance &amp; data summary</h2>
<div class="grid">
  <img src="../eda-mlnet/class_balance.png" alt="class balance">
  <img src="../eda-mlnet/corr_heatmap.png"  alt="correlation heatmap">
</div>
<img src="../eda-mlnet/feature_dist.png" alt="feature distributions">

{% for (model, target), m in metrics.items() %}
<h2>{{ model }} (target = {{ target }})</h2>
<img src="{{ model }}_{{ target }}_curves.png" alt="ROC + PR">
<table>
  <tr><th>metric</th><th>value</th></tr>
  <tr><td>rows train / test</td><td>{{ m.rowsTrain }} / {{ m.rowsTest }}</td></tr>
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

    metrics: dict[tuple[str, str], dict] = {}
    for model in MODELS:
        for target in TARGETS:
            m = load(art / f"{model}_{target}_metrics.json")
            if m is not None:
                plot_roc_pr(m, mdl, model, target)
                metrics[(model, target)] = m

    (mdl / "index.html").write_text(HTML.render(metrics=metrics))
    print(f"[render] wrote {len(metrics)} model/target plots + index.html to {mdl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

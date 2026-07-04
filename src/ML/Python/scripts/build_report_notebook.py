"""Regenerate notebooks/final_report.ipynb (the v0.3 structural rewrite).

Run from src/ML/Python:  uv run python -m scripts.build_report_notebook

The report scales the PSTAT 231 proof-of-concept (2 years, smooth bull, ~170k
rows) to a 20-year, ~1.85M-row, multi-regime stress test as preparation for a
live tax-alpha system. Code cells read the fresh artifacts at execution time;
markdown prose cites a handful of numbers via «TOKENS» that
scripts.fill_report_tokens replaces from the artifacts after a retrain, so the
narration can never drift from the rendered tables.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}

cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip("\n")))

# ──────────────────────────────────────────────────────────────────────────
md(r"""
# Learning the Tax-Loss-Harvest Decision at Scale

### From a two-year proof of concept to a twenty-year stress test — and a redesigned, tax-law-faithful harvest oracle

**Gabriel Kung** · DirectIndexing v0.25 · continuation of the PSTAT 231 final project
University of California, Santa Barbara

Code: [github.com/ieatyoursushi/DirectIndexing_ML](https://github.com/ieatyoursushi/DirectIndexing_ML)
— a .NET 8 pipeline (data download → portfolio simulation → ML.NET training) with this
report layer rendered in Python. This document extends the original two-year study to roughly
two decades of real S&P 500 history — **«N_ROWS» lot-day observations** spanning the 2008
financial crisis, the 2020 COVID crash, and the 2022 selloff — and follows that stress test to
its consequence: the twenty-year evidence exposed one of the harvest rule's four gates as an
accounting artifact, and the rule itself was **redesigned into the scalarized, tax-law-faithful
objective production direct-indexing engines actually use**. The analysis below runs on the
redesigned oracle, keeps the original as a measured ablation baseline, and reorients the
project from "can we recover a toy rule" toward "what would a *live* tax-alpha system need."
""")

md(r"""
## Table of Contents

1. [Introduction](#Introduction)
2. [Data Source and the Two Regimes](#Data-Source-and-the-Two-Regimes)
3. [Loading the Data](#Loading-the-Data)
4. [Codebook](#Codebook)
5. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
   - [The outcomes, one variable at a time](#The-outcomes,-one-variable-at-a-time)
   - [The decision boundary is a level set](#The-decision-boundary-is-a-level-set)
   - [The tax ledger through twenty years](#The-tax-ledger-through-twenty-years)
   - [Cost-basis aging: why the harvest signal thins over twenty years](#Cost-basis-aging:-why-the-harvest-signal-thins-over-twenty-years)
   - [Diagnosis: why the gains gate had to go](#Diagnosis:-why-the-gains-gate-had-to-go)
   - [Missing data](#Missing-data)
   - [Outcome versus predictors](#Outcome-versus-predictors)
   - [Correlation structure](#Correlation-structure)
6. [Data Splitting and Cross-Validation](#Data-Splitting-and-Cross-Validation)
7. [The Recipe: Feature Handling](#The-Recipe:-Feature-Handling)
8. [Model Fitting and Tuning](#Model-Fitting-and-Tuning)
9. [Model Selection and Test-Set Performance](#Model-Selection-and-Test-Set-Performance)
10. [The Headline: Gated versus Scalarized — the Oracle Ablation](#The-Headline:-Gated-versus-Scalarized-—-the-Oracle-Ablation)
11. [Recovering the Value Function](#Recovering-the-Value-Function)
12. [Unsupervised Structure](#Unsupervised-Structure)
13. [Toward a Live Tax-Alpha System](#Toward-a-Live-Tax-Alpha-System)
14. [Conclusion](#Conclusion)
15. [References and Reproducibility](#References-and-Reproducibility)
""")

md(r"""
## Introduction

**Direct indexing** holds an index's constituent stocks directly rather than through a single
fund, so that individual positions can be sold at a loss to **harvest** a tax deduction while
the portfolio as a whole continues to track the benchmark. The unit of bookkeeping is the
**lot** — a parcel of shares of one stock bought on one day at one price (its **cost basis**) —
and every harvesting decision is made lot by lot, subject to two hard legal/threshold
constraints — the IRS **wash-sale rule** (a 30-day repurchase lockout) and a minimum **loss
depth** worth acting on — plus an economic trade-off between the harvest's **tax value** and
the **tracking error** it inflicts on index replication.

In the language of the **Capital Asset Pricing Model**, write a portfolio's excess return as
$R_p - r_f = \alpha_p + \beta_p (R_B - r_f) + \varepsilon_p$, with
$\beta_p = \operatorname{Cov}(R_p, R_B)/\operatorname{Var}(R_B)$. An index fund is the
degenerate case $\beta = 1, \alpha = 0$. A direct-indexing portfolio is better thought of as
an **estimator of the index**, $\hat\beta_{DI}$: built from the constituents, it holds
$\hat\beta_{DI} \approx 1$ while the harvesting engine adds an after-tax
$\alpha_{\text{tax}} > 0$ sourced from the tax code rather than from security selection. The
estimator degrades exactly when lots are harvested — each wash-sale absence nudges
$\hat\beta_{DI}$ off 1 and adds idiosyncratic variance — and the two effects sum to the
**tracking error** $\sigma_{TE}^2 = (\hat\beta_{DI}-1)^2\operatorname{Var}(R_B) +
\operatorname{Var}(\varepsilon_{DI})$. Direct indexing in one line: *hold
$\hat\beta_{DI}\approx 1$, harvest $\alpha$ from the tax code.*

The "correct" harvest decision is encoded as a deterministic **oracle** — this run's canonical
oracle (v0.25) mirrors the objective form published by production stock-level TLH engines:
hard gates survive only where they encode a genuine legal rule or threshold fact, and
everything economic collapses into one **scalarized objective** thresholded at zero,

$$
f^*(x) = \mathbb{1}[L \le -0.02]\cdot\mathbb{1}[\mathcal W \ge 30]\cdot
\mathbb{1}[\sigma_{TE}\le \theta_{\max}]\cdot\mathbb{1}[U(x) > 0],
\qquad
U(x) = \mathrm{TaxValue} - \lambda\,\sigma_{TE}^2 - c_{\mathrm{trade}},
$$

where $L$ is the lot's unrealized return, $\mathcal W$ the wash-sale clock,
$\theta_{\max}=0.15$ a *loose* tail-risk ceiling ($3\times$ the old operating cap — it binds on
zero rows in twenty years of history), $\lambda = 90{,}000$ the tracking-error price calibrated
from this dataset, $c_{\mathrm{trade}} = \$10$ flat round-trip friction, and $\mathrm{TaxValue}$
the **capacity-aware dollar value of the harvest** from a real Schedule D ledger (realized
gains, the \$3k/yr ordinary-income allowance, and indefinite loss carryforward — 26 USC
§1211(b)/§1212(b)). The decision boundary is the **level set** $\{U = 0\}$, not the corner of
an axis-aligned box. (Full mathematics: `DataMemo/PortfolioMath.md`, `DataMemo/SimulationMath.md`,
and the design record `DataMemo/GYTD_Redesign_Plan.md`.)

This is not the oracle the project started with. The v0.2 oracle was a four-gate AND whose
third gate, $G^{\mathrm{YTD}}>0$ ("only harvest if the book has realized gains this year"),
the twenty-year run exposed as an accounting artifact — economically backwards for
individual-investor direct indexing and absent from every production TLH methodology we could
find. The [diagnosis section](#Diagnosis:-why-the-gains-gate-had-to-go) tells that story in
full, because it is the project's cleanest empirical arc: the defect was *discovered by
scaling the data*, grounded in tax law and industry practice, redesigned (issue #23,
PRs #24–#26), and measured before/after — the gated original survives as this report's
**ablation baseline**.

**Why scale to twenty years.** The PSTAT 231 submission learned this decision from a single
**two-year window (2024–2026)** — a smooth bull market with only micro-drawdowns. That run
answered the academic question (yes, the boundary is learnable; gradient-boosted trees win)
but left a glaring external-validity question: *does any of it survive a real bear market?* The
**data window is now a parameter of the study**: the download layer accepts an arbitrary
`--from`/`--to` date range (issues #19/#20), bounded in practice at ~20 years by the price
API's ≈5,000-bars-per-ticker response cap, and the simulation adapts to whatever range the
cache holds. This report re-runs the entire pipeline on roughly **two decades of history
through three major crises** and asks three questions:

1. **Robustness.** Does the champion model's forward-harvest-propensity skill survive 2008,
   2020, and 2022, or was it an artifact of a calm market?
2. **Boundary geometry.** The two-year run found that the oracle's *learnability by linear
   models* depended on which gates bind — and the twenty-year run revealed that dependence to
   be an artifact of a defective gate. With the defect fixed, the question sharpens: **how much
   of the tree-versus-linear gap was real problem structure, and how much was the gate's box
   geometry?** The [ablation section](#The-Headline:-Gated-versus-Scalarized-—-the-Oracle-Ablation)
   answers with both oracles measured side by side.
3. **Live-system readiness.** What does the long horizon expose about the simulation's design
   choices — the fixed cost basis, the survivorship-biased universe — that a production
   tax-alpha engine would have to fix? This document is preparation for the live-simulation
   layer (v0.4–v0.5), not a deployment of it.

The predictive targets: the hard oracle label `Y_Oracle`; the forward-looking **soft label**
`Y_Soft_BT > 0` ("will the oracle fire on this lot within the next 30 trading days?"), which
depends on unobserved future prices and is therefore the genuine prediction problem; and — new
with the redesign — the **continuous regression target** `Y_TaxValue` (the ledger's
capacity-aware harvest value), with the raw objective `U(x)` exported as `Y_Utility` (it
doubles as the future reinforcement-learning reward).
""")

md(r"""
## Data Source and the Two Regimes

Two external sources feed the pipeline; everything else is derived. **Daily prices** are
end-of-day OHLCV bars for the S&P 500 constituents from the Financial Modeling Prep (FMP) API
(`/stable/historical-price-eod/full`), now pulled over a configurable date range via
`dotnet run download --from YYYY-MM-DD --to YYYY-MM-DD` (the original run hardcoded two years).
With no explicit dates the downloader falls back to a rolling `years`-long window; either way it
extends the fetch backward by ≈200 trading days so the moving-average features are defined on
the portfolio's first day, warns when a custom range is too short for that warmup, and
re-aggregates an existing cache incrementally instead of re-downloading it. The practical
ceiling is the API itself — ≈5,000 daily bars per ticker, which is what pins *this* run's
window to **2006-07 through 2026-06**. **Index membership** comes from State Street's SPDR
S&P 500 ETF (SPY) daily holdings file.

> Financial Modeling Prep. (2026). *Historical Price EOD API* [Data set].
> https://financialmodelingprep.com/
>
> State Street Global Advisors. (2026). *SPDR S&P 500 ETF Trust (SPY) Daily Holdings*
> [Data set]. https://www.ssga.com/

**A survivorship caveat, stated up front.** Membership and weights come from *today's* SPY
holdings, but a 20-year price window only exists for the **«N_CONSTIT» constituents** with two
decades of continuous history — down from the 503 in the two-year run. Names that were added,
dropped, merged, or delisted over the period are absent, so the universe is **survivorship-
biased toward long-lived large caps**. This inflates index returns and understates idiosyncratic
risk relative to a true point-in-time membership; a production system would need a point-in-time
constituent feed (flagged as future work). It does not bias the *harvest-decision* learning
problem, which is conditional on each lot's own state, but it does mean the absolute
tax-alpha and tracking-error magnitudes here are optimistic.

**Provenance.** Real prices drive a \$10M equal-dollar simulated portfolio opened after a
200-trading-day moving-average warmup; the backtesting engine marks every lot to market each
day, updates the tax ledger, evaluates the oracle, executes harvests (re-buying after the
30-day wash window), and writes one row per open lot per day. Because the **acting oracle
changes the trajectory itself** (which lots get harvested → wash clocks → ledger state → which
rows exist), the two oracles produce **two separate datasets** via
`dotnet run simulate --oracle=scalarized|gated`: the canonical `data/lots.csv` (scalarized)
that this report analyzes, and the ablation baseline `data/lots_gated.csv`. The canonical
result is **«N_ROWS» rows × 26 columns** across Timesteps 200–4999 (≈19 active years). Every
column is defined in the [Codebook](#Codebook) and the standalone `codebook.md`.

| | PSTAT 231 run | This run (scalarized oracle) |
|---|---|---|
| Window | 2024–2026 (≈2 yr) | 2006-07 – 2026-06 (≈19 active yr) |
| Market character | smooth bull, micro-drawdowns | 2008 GFC, 2020 COVID, 2022 bear |
| Oracle | 4-gate AND (incl. `G_YTD > 0`) | 3 hard gates · 𝟙[U > 0] |
| Constituents | 503 | «N_CONSTIT» |
| Rows | ~170,751 | «N_ROWS» |
| `Y_Oracle` positive rate | 1.6% | «ORACLE_RATE» |
| `Y_Soft_BT > 0` rate | 19.9% | «SOFT_RATE» |
""")

md(r"""
## Loading the Data

The notebook locates the repository root the same way the .NET orchestrator does (walking up
to `DirectIndexing.sln`), so it runs identically under `dotnet run report` and interactively.
""")

code(r"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_here = Path.cwd()
_ml_dir = _here if (_here / "scripts").exists() else _here.parent
if str(_ml_dir) not in sys.path:
    sys.path.insert(0, str(_ml_dir))

from scripts import report_helpers as rh

rh.set_style()
ROOT      = rh.repo_root()
ART       = ROOT / "data" / "artifacts-mlnet"          # scalarized (canonical) artifacts
ART_GATED = ROOT / "data" / "artifacts-mlnet-gated"    # gated ablation-arm artifacts

lots = pd.read_csv(ROOT / "data" / "lots.csv")         # canonical scalarized dataset
SIM_YEAR = (lots["Timestep"] - 200) / 252.0            # approx. years since portfolio open

# Gated ablation arm — only the columns the diagnosis section needs.
gated = pd.read_csv(ROOT / "data" / "lots_gated.csv",
                    usecols=["Timestep", "L", "Sigma_TE", "WashClock",
                             "RealizedGainsYTD", "Y_Oracle"])

print(f"scalarized: {len(lots):,} rows x {lots.shape[1]} columns | "
      f"Timesteps {lots['Timestep'].min()}-{lots['Timestep'].max()} "
      f"(~{SIM_YEAR.max():.1f} active years) | {lots['Symbol'].nunique()} tickers")
print(f"gated arm:  {len(gated):,} rows (ablation baseline)")
lots.head()
""")

md(r"""
## Codebook

One row is a **lot snapshot** — the state of one tax lot on one simulated trading day:
seventeen numeric features (lot-, portfolio-, asset-level, and derived — the portfolio-level
block is now the **TaxLedger triple** `RealizedGainsYTD` / `LossCarryforward` /
`OrdinaryOffsetBudget`, and the derived `TaxValue` supersedes the old `TaxAlpha`), one
categorical, three metadata columns dropped before modeling, and six labels (hard, two soft,
and the new continuous `Y_TaxValue` / `Y_Utility` / spectator `Y_Oracle_GatedSpec`). The table
below renders the same schema that generates the standalone codebook, so the two cannot drift.
""")

code(r"""
from scripts.codebook_schema import COLUMNS
pd.DataFrame(COLUMNS)[["name", "dtype", "units", "role", "encoding"]] \
    .style.hide(axis="index").set_properties(**{"text-align": "left"})
""")

md(r"""
## Exploratory Data Analysis

The twenty-year EDA is organized around two facts. First, **the harvest event got an order of
magnitude rarer** — the sections below establish that and diagnose its cause (cost-basis aging,
not market calm). Second, **the decision boundary changed shape**: the redesigned oracle's
harvest region is a smooth level set in (tracking error, tax value) space rather than the
corner of a box, and the new ledger and objective columns make that geometry directly visible.
The gated original gets a full past-tense autopsy, because the evidence against it is the
scale-up's central discovery.
""")

md(r"""
### The outcomes, one variable at a time
""")

code(r"""
fig, axes = plt.subplots(2, 3, figsize=(13, 8))

counts = lots["Y_Oracle"].value_counts().sort_index()
bars = axes[0, 0].bar(["0 (hold)", "1 (harvest)"], counts.values, color=["#357", "#a33"])
for b, v in zip(bars, counts.values):
    axes[0, 0].text(b.get_x() + b.get_width()/2, v, f"{v:,}\n({v/len(lots):.2%})",
                    ha="center", va="bottom", fontsize=9)
axes[0, 0].set_ylim(0, counts.max() * 1.18)
axes[0, 0].set_title("Y_Oracle — hard label"); axes[0, 0].set_ylabel("rows")

axes[0, 1].hist(lots["Y_Soft_GBM"], bins=40, color="#357"); axes[0, 1].set_yscale("log")
axes[0, 1].set_title("Y_Soft_GBM — P(fire in 30d), GBM paths"); axes[0, 1].set_xlabel("value")

axes[0, 2].hist(lots["Y_Soft_BT"].dropna(), bins=31, color="#3a7"); axes[0, 2].set_yscale("log")
axes[0, 2].set_title("Y_Soft_BT — realized fraction of next 30d"); axes[0, 2].set_xlabel("value")

tv_pos = lots.loc[lots["Y_TaxValue"] > 0, "Y_TaxValue"]
axes[1, 0].hist(tv_pos, bins=50, color="#a37"); axes[1, 0].set_yscale("log")
axes[1, 0].set_title(f"Y_TaxValue > 0 — harvest value ($), "
                     f"{len(tv_pos)/len(lots):.1%} of rows")
axes[1, 0].set_xlabel("dollars")

axes[1, 1].hist(lots["Y_Utility"], bins=80, color="#573"); axes[1, 1].set_yscale("log")
axes[1, 1].axvline(0, color="#a33", ls="--", lw=1.2, label="U = 0 (decision boundary)")
axes[1, 1].set_title("Y_Utility — U(x) = TaxValue − λσ²TE − c")
axes[1, 1].set_xlabel("dollars"); axes[1, 1].legend(fontsize=8)

agree = pd.crosstab(lots["Y_Oracle"], lots["Y_Oracle_GatedSpec"], normalize=True)
im = axes[1, 2].imshow(agree.values, cmap="Blues")
for i in range(agree.shape[0]):
    for j in range(agree.shape[1]):
        axes[1, 2].text(j, i, f"{agree.values[i, j]:.3%}", ha="center", va="center",
                        fontsize=9, color="black")
axes[1, 2].set_xticks([0, 1]); axes[1, 2].set_yticks([0, 1])
axes[1, 2].set_xlabel("v0.2 gated spectator says"); axes[1, 2].set_ylabel("acting oracle says")
axes[1, 2].set_title("Same-row oracle agreement"); axes[1, 2].grid(False)
fig.tight_layout(); plt.show()

print(f"Y_Oracle = 1:               {lots['Y_Oracle'].mean():.2%}")
print(f"Y_Soft_BT > 0 (of labeled): {(lots['Y_Soft_BT'].dropna() > 0).mean():.2%}")
print(f"U > 0:                      {(lots['Y_Utility'] > 0).mean():.2%}")
""")

md(r"""
The hard label fires on only **«ORACLE_RATE» of lot-days** — an order of magnitude *rarer*
than the 1.6% of the two-year run, and the soft target drops from 19.9% to **«SOFT_RATE»**.
This is the opposite of what a naïve intuition predicts: twenty years *including three crashes*
should offer **more** harvest opportunities than a calm two-year bull, not fewer. The
resolution is not about the market — it is about the **simulation's cost-basis dynamics**, which
the [aging section](#Cost-basis-aging:-why-the-harvest-signal-thins-over-twenty-years)
isolates. The methodological consequence is severe: at a «ORACLE_RATE» base rate, accuracy is
meaningless (predicting "never harvest" scores >99.7%), so model selection uses **PR-AUC** and
training uses balanced class weights and stratified splits — the same discipline as before,
but now load-bearing.

The bottom row is new with the redesign. `Y_TaxValue` is zero for the ~98% of lot-days with no
harvestable loss and heavy-tailed on the rest — a **zero-inflated continuous target** whose
kinks come from real tax mechanics (the offset-capacity split and the short/long-term rate
jump at one year). `Y_Utility` is the raw objective before thresholding: its bulk sits just
below zero (the λσ² penalty at typical tracking error, minus trade friction, with no loss to
harvest), and the harvest decision is literally the right tail crossing the dashed line. The
agreement matrix previews the ablation: the two oracles agree on the overwhelming majority of
rows but disagree exactly where the economics and the old accounting rule part ways.
""")

md(r"""
### The decision boundary is a level set

This is the picture of the philosophical change. The v0.2 oracle's harvest region was the
corner of an axis-aligned box — four independent pass/fail checks. The redesigned region lives
in the (σ_TE, TaxValue) plane above the parabola $\mathrm{TaxValue} = \lambda\sigma_{TE}^2 +
c_{\mathrm{trade}}$: a lot's tax value must *buy* its tracking-error cost, continuously, the
way production TLH engines state their objective (tax alpha minus λ·TE²).
""")

code(r"""
LAMBDA, C_TRADE, THETA_MAX = 90_000.0, 10.0, 0.15

elig = lots[(lots["L"] <= -0.02) & (lots["WashClock"] >= 30)]   # hard gates open
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

for y, color, lbl, ms in ((0, "#357", "hold (U ≤ 0)", 4), (1, "#a33", "harvest (U > 0)", 6)):
    sub = elig[elig["Y_Oracle"] == y]
    if len(sub) > 20_000:
        sub = sub.sample(20_000, random_state=42)
    axes[0].scatter(sub["Sigma_TE"], sub["TaxValue"], s=ms, alpha=0.25, c=color, label=lbl)

sig = np.linspace(0.0, max(0.06, float(elig["Sigma_TE"].max()) * 1.05), 200)
axes[0].plot(sig, LAMBDA * sig**2 + C_TRADE, color="k", lw=2,
             label=r"$U=0$:  TaxValue $= \lambda\sigma^2 + c$")
axes[0].set_xlabel(r"$\sigma_{TE}$ — annualized tracking error")
axes[0].set_ylabel("TaxValue ($)")
axes[0].set_yscale("symlog", linthresh=10)
axes[0].set_title("Harvest region above the level curve (hard-gate-eligible rows)")
axes[0].legend(fontsize=8, loc="upper left")

sig_all = lots["Sigma_TE"]
axes[1].hist(sig_all, bins=60, color="#357", alpha=0.8)
axes[1].axvline(0.05, color="#a37", ls=":",  lw=1.5, label="old θ₂ cap (0.05)")
axes[1].axvline(THETA_MAX, color="#a33", ls="--", lw=1.5, label="θ_max ceiling (0.15)")
axes[1].set_xlim(0, 0.16); axes[1].set_yscale("log")
axes[1].set_xlabel(r"$\sigma_{TE}$"); axes[1].set_title(
    "Realized σ_TE vs the old cap and the new tail-only ceiling")
axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()

print(f"θ_max binds on {(sig_all > THETA_MAX).mean():.4%} of rows "
      f"(max realized σ_TE = {sig_all.max():.4f})")
""")

md(r"""
Two structural facts to read off. **Left:** among rows where both hard gates are open, the
harvest/hold split follows the level curve — marginal tax values clear the boundary only at
calm tracking error, and the same value fails as σ_TE rises. The trade-off is *continuous*;
nothing in the v0.2 box could express it. **Right:** the old fine-grained TE cap (0.05) sat
*above the 99.9th percentile* of realized tracking error — a cap that almost never operated in
its intended fine-grained role — while its replacement, the loose θ_max = 0.15 ceiling, is a
pure tail-risk circuit breaker that **binds on zero rows in twenty years**. The marginal TE
trade-off now lives inside U(x), priced by λ, instead of pretending to be a hard constraint.
""")

md(r"""
### The tax ledger through twenty years

The redesign replaced a constant \$1M "gains seed" with a real Schedule D ledger. The
scalarized book is deliberately **loss-only** (no exogenous gains process yet), which is the
tax situation of a passive client with no outside capital-gains activity: every year's
harvested losses first consume the \$3,000 ordinary-income allowance and the remainder
**banks as indefinite carryforward** (26 USC §1212(b)) instead of evaporating at a year-end
reset.
""")

code(r"""
byT_led = lots.groupby("Timestep").agg(
    carry=("LossCarryforward", "first"),
    net  =("RealizedGainsYTD", "first"),
    budget=("OrdinaryOffsetBudget", "first"),
)
yr_led = (byT_led.index - 200) / 252.0

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
axes[0].plot(yr_led, byT_led["carry"] / 1e6, color="#357", lw=1.8)
axes[0].set_xlabel("simulation year"); axes[0].set_ylabel("LossCarryforward ($M)")
axes[0].set_title("Banked losses accumulate — the mechanic the old seed erased")

axes[1].plot(yr_led, byT_led["net"] / 1e6, color="#a33", lw=1)
axes[1].axhline(0, color="#aaa", lw=1, ls="--")
axes[1].set_xlabel("simulation year"); axes[1].set_ylabel("RealizedGainsYTD ($M)")
axes[1].set_title("Net realized P&L YTD — loss-only book, resets each January")
fig.tight_layout(); plt.show()

print(f"final LossCarryforward: ${byT_led['carry'].iloc[-1]:,.0f}")
print(f"OrdinaryOffsetBudget range: [{byT_led['budget'].min():.0f}, {byT_led['budget'].max():.0f}]")
""")

md(r"""
The staircase on the left is the point of the whole redesign in one chart: the GFC harvests
bank millions of dollars of carryforward that a real taxpayer would carry against future gains
for decades — value the v0.2 design *deleted every January* and then used the artificial seed
to paper over. One honest caveat carried from the design doc: with no outside gains, each
year's *immediately usable* offset capacity is just the \$3k allowance, so this book is the
**conservative floor persona** — a high-earner client with regular outside gains (RSU vests,
sales) would absorb the banked losses far faster, making the tax alpha here an understatement
of the target clientele's.
""")

md(r"""
### Cost-basis aging: why the harvest signal thins over twenty years
""")

code(r"""
lots["_atloss"] = (lots["L"] <= -0.02).astype(float)
byT = lots.groupby("Timestep").agg(
    frac_loss=("_atloss", "mean"),
    mean_age =("H", "mean"),
    harvests =("Y_Oracle", "sum"),
)
yr = (byT.index - 200) / 252.0

fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
axes[0].plot(yr, byT["frac_loss"].values, color="#a33", lw=1)
axes[0].set_xlabel("simulation year (≈ Timestep / 252)")
axes[0].set_ylabel("fraction of lots ≥ 2% below cost basis")
axes[0].set_title("Supply of harvestable losses (spikes = bear markets)")

axes[1].plot(yr, byT["mean_age"].values / 252.0, color="#357", lw=1.5)
axes[1].set_xlabel("simulation year")
axes[1].set_ylabel("mean lot age (years)")
axes[1].set_title("Lots age as the book holds cost basis")
fig.tight_layout(); plt.show()
""")

md(r"""
The left panel is the mechanism, and it is starker than "spikes at every bear market." There is
**one eruption** — the 2008–09 financial crisis, where at the March-2009 bottom roughly **90% of
the book sits ≥ 2% underwater** — and then the signal never truly returns. The 2020 COVID crash
(a 34% index drawdown) and the 2022 selloff barely register: their daily peaks put only ~1–2%
and well under 1% of lots underwater, respectively. Why? The right panel: lots are opened once
at the warmup day and **hold their cost basis indefinitely** (only a harvest-and-reopen resets
it), so mean lot age climbs steadily and the un-harvested lots accumulate years of appreciation.
A position carrying a 2007 basis is so deep in the money by 2020 that a one-third market crash
cannot push it 2% below cost. The portfolio **ages out of harvestability** — harvests become
rare, bursty, and (after the first decade) nearly extinct even in genuine crashes.

This is the single most important finding of the scale-up, and it is a *simulation-design*
finding, not a market finding: a real direct-indexing account receives ongoing contributions
and rebalances, continuously minting fresh lots at current prices that *can* dip. The
open-once / hold-forever design has no such mechanism, so over long horizons it understates
harvest opportunity. The fix — modeling contributions and rebalancing — is exactly what the
[live-system section](#Toward-a-Live-Tax-Alpha-System) returns to.
""")

md(r"""
### Diagnosis: why the gains gate had to go

Everything in this section runs on **`lots_gated.csv` — the v0.2 four-gate oracle acting on
the same twenty years of prices** — because this is the autopsy that produced the redesign.
The v0.2 oracle required `G_YTD > 0`: only harvest if the book has net realized gains this
calendar year, propped up by a constant \$1M January "seed" standing in for unmodeled outside
gains. The two charts below are the evidence that killed it.
""")

code(r"""
byT_g = gated.groupby("Timestep").agg(
    harvests=("Y_Oracle", "sum"),
    gytd    =("RealizedGainsYTD", "first"),   # == legacy G_YTD in the gated arm
)
yr_g = (byT_g.index - 200) / 252.0

fig, ax1 = plt.subplots(figsize=(11, 4))
ax1.bar(yr_g, byT_g["harvests"].values, color="#a33", width=(yr_g.max()/len(yr_g)),
        label="harvests/day")
ax1.set_xlabel("simulation year"); ax1.set_ylabel("oracle harvests per day", color="#a33")
ax2 = ax1.twinx()
ax2.plot(yr_g, byT_g["gytd"].values / 1e6, color="#357", lw=1.2, label="G_YTD ($M)")
ax2.axhline(0, color="#aaa", lw=1, ls="--"); ax2.grid(False)
ax2.set_ylabel("G_YTD ($ millions)", color="#357")
ax1.set_title("GATED ARM — harvest activity vs the gains budget over ~19 years")
fig.tight_layout(); plt.show()
""")

md(r"""
Harvesting is **regime-clustered**: the red bars are nearly silent in the long expansions and
fire in dense bursts during the drawdowns — precisely when a tax-alpha engine earns its keep.
Of the oracle harvests in the whole gated run, the overwhelming majority happen in the first
two simulation years (the GFC); no later year comes close.

The blue line is the gains budget `G_YTD`, re-seeded each January to \$1M (10% of the initial
book) and drawn down by realized losses — and it exposes a **self-strangling feedback loop**.
During 2008–09 the harvest bursts realize losses fast enough to burn the \$1M seed to zero
mid-year; the moment `G_YTD` hits the floor, gate 3 slams shut and **freezes all harvesting
until the January re-seed** — in the richest loss environment of the entire window, the engine
spends the majority of the crisis year forbidden to harvest by its own accounting rule. Outside
those two GFC years the same gate never comes close to closing (the seed dwarfs realized
losses). Both failure modes — shut exactly when losses are most abundant, vacuously open the
other ~17 years — belong to the same defect, dissected below.
""")

code(r"""
gates_g = {
    "L ≤ −0.02  (loss)":       (gated["L"] <= -0.02).mean(),
    "σ_TE ≤ 0.05  (tracking)": (gated["Sigma_TE"] <= 0.05).mean(),
    "G_YTD > 0  (gains)":      (gated["RealizedGainsYTD"] > 0).mean(),
    "WashClock ≥ 30  (wash)":  (gated["WashClock"] >= 30).mean(),
}
simyr_g = ((gated["Timestep"] - 200) // 252).astype(int)
gate_by_year = pd.DataFrame({
    "L ≤ −0.02  (loss)":       gated["L"] <= -0.02,
    "σ_TE ≤ 0.05  (tracking)": gated["Sigma_TE"] <= 0.05,
    "G_YTD > 0  (gains)":      gated["RealizedGainsYTD"] > 0,
    "WashClock ≥ 30  (wash)":  gated["WashClock"] >= 30,
}).groupby(simyr_g).mean()

fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.8),
                         gridspec_kw={"width_ratios": [1, 1.6]})
names = list(gates_g); vals = [gates_g[k] for k in names]
bars = axes[0].barh(names, vals, color=["#a33", "#357", "#3a7", "#a37"])
for b, v in zip(bars, vals):
    axes[0].text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.1%}", va="center", fontsize=9)
axes[0].set_xlim(0, 1.12); axes[0].set_xlabel("fraction of rows gate is open")
axes[0].set_title("Marginal gate pass rates (gated arm)")
axes[0].invert_yaxis()

for name, color in zip(gate_by_year.columns, ["#a33", "#357", "#3a7", "#a37"]):
    axes[1].plot(gate_by_year.index, gate_by_year[name], marker="o", ms=4, lw=1.6,
                 label=name, color=color)
axes[1].set_xlabel("simulation year (0 ≈ 2007)"); axes[1].set_ylabel("fraction open")
axes[1].set_ylim(-0.04, 1.09); axes[1].set_xticks(gate_by_year.index[::2])
axes[1].set_title("Per-year pass rates — binding was a regime property")
axes[1].legend(fontsize=7, loc="center right")
fig.tight_layout(); plt.show()
""")

md(r"""
The two panels together say the gated structure was **regime-dependent, and degenerately so**.
The gains gate was open on **«GYTD_OPEN» of rows** overall — near-vestigial — but the per-year
view shows *all* of its closure concentrated in simulation years 1–2 (the GFC and its
aftermath), where it was shut on roughly 60% of lot-days. Every other year it was open on
essentially 100% of rows. The binding constraint everywhere else was the **loss gate**: the
cost-basis aging above means very few lots are ever 2% underwater, so it was the loss
condition, not the availability of gains, that starved the harvest signal. As an *economic*
fact, the gains gate's behavior was exactly backwards: it blocked harvesting only during the
one regime where losses were abundant, and rubber-stamped it the other seventeen years.

> **The defect and the fix (issue #23, shipped as PRs #24–#26).** Two counts against the gate.
> **(1) The seed was a simulation artifact:** the constant \$1M January re-seed neither scaled
> with the growing book nor reflected any client's actual tax situation, so the gate's
> open/closed state was an accounting fiction. **(2) The gate was conceptually misaligned with
> beta-tracking direct indexing.** Production stock-level TLH engines (Wealthfront's published
> methodology states the objective directly as tax alpha minus λ·tracking-error²; Betterment's
> fires when benefit net of cost clears a threshold) hold $\hat\beta_{DI} \approx 1$ and
> capture losses when the trade-off clears; *whether* those losses offset gains this year is
> the client tax return's concern, because for an individual a harvested loss is never wasted —
> it offsets gains from anywhere on the 1040, then \$3,000/yr of ordinary income
> (26 USC §1211(b)), and the remainder **carries forward indefinitely** (§1212(b)). A hard
> "must have realized gains this year" trigger is therefore *stricter than the tax code* and
> refuses real tax alpha — most visibly in the 2008–09 self-strangulation above. The fix
> (`DataMemo/GYTD_Redesign_Plan.md` v2): the gains information moved from a gate into the
> **TaxLedger's capacity-aware `TaxValue`** inside the scalarized objective; the fine-grained
> TE cap was demoted into the continuous $\lambda\sigma^2$ term with a loose tail-only ceiling;
> a flat trade cost completes the net-benefit test. Where a G_YTD-style gate *would* be valid —
> C-corporations, whose losses offset only capital gains within a bounded carryback/forward
> window — is recorded as an explicit non-goal. All models in this report are trained on the
> corrected oracle, with the gated arm retained as the measured ablation baseline.
""")

md(r"""
### Missing data
""")

code(r"""
na = lots.isna().sum()
display(na[na > 0].to_frame("missing cells").assign(
    pct=lambda d: (100 * d["missing cells"] / len(lots)).round(3)))
""")

md(r"""
Missingness remains **structural, not accidental**, and at 1.85M rows it is now a vanishing
fraction. `Y_Soft_BT` is undefined for the final 30 simulation days (its forward window runs
off the data) — a known, `Timestep`-determined gap, so those rows are *excluded* from
soft-label training rather than imputed (imputing a label would manufacture ground truth).
`DeltaMA50`/`DeltaMA200` carry a handful of NaNs on tickers with short early histories, handled
by per-fold median imputation. `Sector` remains the degenerate column from the SPY scrape
(≈99% the placeholder `"-"` → "Unknown"); it is cleaned and one-hot encoded for schema honesty
but carries essentially no information — an unchanged data-quality gap.
""")

md(r"""
### Outcome versus predictors
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
data_l = [lots.loc[lots["Y_Oracle"] == 0, "L"], lots.loc[lots["Y_Oracle"] == 1, "L"]]
axes[0].violinplot(data_l, showmedians=True)
axes[0].axhline(-0.02, color="#a33", ls="--", lw=1, label="loss gate (−2%)")
axes[0].set_xticks([1, 2]); axes[0].set_xticklabels(["Y_Oracle = 0", "Y_Oracle = 1"])
axes[0].set_ylabel("L — unrealized return")
axes[0].set_title("Unrealized return by harvest decision"); axes[0].legend(fontsize=8)

tv0 = lots.loc[(lots["Y_Oracle"] == 0) & (lots["TaxValue"] > 0), "TaxValue"]
tv1 = lots.loc[lots["Y_Oracle"] == 1, "TaxValue"]
axes[1].violinplot([np.log10(tv0.clip(lower=1)), np.log10(tv1.clip(lower=1))],
                   showmedians=True)
axes[1].set_xticks([1, 2])
axes[1].set_xticklabels(["Y_Oracle = 0\n(TaxValue > 0)", "Y_Oracle = 1"])
axes[1].set_ylabel("log10 TaxValue ($)")
axes[1].set_title("Harvest value by decision — U(x) prices the split")

rbins = pd.cut(lots["R_t"].clip(-0.10, 0.10), bins=30)
rmean = lots.groupby(rbins, observed=True)["Y_Soft_GBM"].mean()
axes[2].plot([iv.mid for iv in rmean.index], rmean.values, marker="o", color="#3a7", lw=2)
axes[2].set_xlabel("R_t — today's return (clipped ±10%)")
axes[2].set_ylabel("mean Y_Soft_GBM")
axes[2].set_title("Forward harvest propensity vs today's return")
fig.tight_layout(); plt.show()
""")

md(r"""
The left violin confirms the loss gate as a hard separator — the harvested class sits entirely
below −2%, with a long tail of un-harvested deep-loss lots where another condition (wash-sale,
the utility threshold) blocked the trade. The middle panel is new: among lots *with* a
harvestable loss, the harvested class concentrates at higher tax value — the continuous
U(x) > 0 split, visible as a distributional shift rather than a clean threshold because the
λσ² penalty moves the effective cutoff row by row. The right panel shows the forward soft
label rising sharply as today's return turns negative: a stock falling today is far likelier
to cross the loss gate within 30 days. The relationship is monotone but soft — a probability
shift, not a determination — which is exactly the structure a gradient-boosted classifier
exploits and a hard threshold cannot.
""")

md(r"""
### Correlation structure
""")

code(r"""
feat = [f for f in rh.NUMERIC_FEATURES if lots[f].nunique() > 1]
corr = lots[feat].corr()
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(feat))); ax.set_xticklabels(feat, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(feat))); ax.set_yticklabels(feat, fontsize=8)
for i in range(len(feat)):
    for j in range(len(feat)):
        v = corr.iloc[i, j]
        if abs(v) > 0.45 and i != j:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.7 else "black")
fig.colorbar(im, shrink=0.8)
ax.set_title("Pearson correlation — numeric features (constant columns dropped)")
fig.tight_layout(); plt.show()
""")

md(r"""
The correlation structure is stable across the scale-up: mostly modest off-diagonals, with the
expected blocks — `H`/`S` (long-term flag is a step function of holding age), the moving-average
deviations co-moving with `L`, the new ledger columns (`RealizedGainsYTD`,
`OrdinaryOffsetBudget`) coupled through the shared tax calendar with `DaysToYE`, and `TaxValue`
tracking `L` (it is a kinked function of the loss). Nothing approaches the collinearity that
would force dropping a feature; the regularized linear models absorb this comfortably. As
before, `K` (lots per ticker) and the encoded `Sector` are effectively constant and contribute
a zero eigenvalue we meet again in [PCA](#Unsupervised-Structure).
""")

md(r"""
## Data Splitting and Cross-Validation

*(Written for a general audience.)* Before any model sees the data, **20% of rows are locked
away as a test set** and never touched during development; all learning and tuning happens on
the remaining 80%. Because the positive class is now extraordinarily rare (≈«ORACLE_RATE» for
the hard label), the split is **stratified** — each class is divided 80/20 *separately*
(`StratifiedSplit.cs`, seed 42) so both sides inherit the population's class proportion exactly.

Tuning uses **stratified 5-fold cross-validation** on the training set: for each hyperparameter
configuration, the model trains on four folds and is scored on the held-out fifth, rotating five
times, and the five scores are averaged. The score is **PR-AUC** (area under the precision–recall
curve), the right metric under extreme imbalance — ROC-AUC can look excellent while the few true
positives drown in false alarms, exactly what PR-AUC penalizes. Every preprocessing statistic
(imputation medians, class weights, normalization, one-hot vocabulary) is fit on the *training
portion of each fold only* — enforced structurally in the C# pipeline, where the fitting
functions only accept a training fold (`DataMemo/MLNetLeakageAudit.md`).
""")

code(r"""
gbt_soft = rh.load_artifact(ART, "gbt_soft_bt_metrics.json")
gbt_orc  = rh.load_artifact(ART, "gbt_oracle_metrics.json")
pd.DataFrame({
    "target": ["Y_Oracle (hard)", "Y_Soft_BT > 0 (soft)"],
    "usable rows": [gbt_orc["rowsTrain"] + gbt_orc["rowsTest"],
                    gbt_soft["rowsTrain"] + gbt_soft["rowsTest"]],
    "train (80%)": [gbt_orc["rowsTrain"], gbt_soft["rowsTrain"]],
    "test (20%)":  [gbt_orc["rowsTest"],  gbt_soft["rowsTest"]],
}).set_index("target")
""")

md(r"""
## The Recipe: Feature Handling

All models share one preprocessing recipe (an ML.NET `EstimatorChain`, the analogue of a
tidymodels recipe), fit on training data only: **median imputation** of the moving-average NaNs;
**one-hot encoding** of the cleaned `Sector`; **mean-variance normalization** of all seventeen
numeric features (load-bearing for the regularized linear models, where penalties are
scale-sensitive; harmless for trees); **balanced class weights** (≈500:1 at this base rate, so
the rare positives are not ignored); and **concatenation** into one feature vector, with `Symbol`
and `Timestep` held out as metadata. The outcome is then modeled as a binary classification of
`Y_Oracle` or `Y_Soft_BT > 0`, plus one deliberately mis-specified linear *regression* on the
continuous `Y_Soft_BT` retained as a cautionary comparison. One schema-discipline note: the
new regression target `Y_TaxValue` equals the `TaxValue` *feature* by construction, so the
[value-function regression](#Recovering-the-Value-Function) excludes that feature — the task
is recovering the ledger function from raw state, not copying a column.
""")

md(r"""
## Model Fitting and Tuning

Five model types were fit and tuned by grid search, each configuration scored by mean PR-AUC
across the stratified 5-fold CV above. **Note for this version:** the random forest now tunes
its **feature-sampling fraction** (the ML.NET analogue of `mtry`/`m`) over {0.3, 0.5, 0.7} — the
single most important RF hyperparameter and the decorrelation knob — in addition to tree count
and leaves, closing a tuning gap from the original submission.

| Model | What it does | Grid searched |
|---|---|---|
| **Logistic (L2)** | linear logit boundary | $C \in \{0.01, 0.1, 1, 10\}$ |
| **Elastic net** | linear with L1+L2 penalties | $\lambda_{L1} \times \lambda_{L2}$, 3×3 |
| **Random forest** | bagged decorrelated trees | trees ∈ {100,200} × leaves ∈ {20,31} × **featureFraction ∈ {0.3,0.5,0.7}** |
| **Gradient boosted trees** | sequential residual-fitting trees | trees ∈ {100,200} × learning rate ∈ {0.05,0.1} × leaves ∈ {20,31} |
| **Linear regression (ridge)** | least-squares on the continuous soft label — a poor-fit demonstration | $\lambda_{L2}$, 3 values |
""")

code(r"""
lb_soft = rh.load_artifact(ART, "soft_bt_cv_leaderboard.json")
lb_orc  = rh.load_artifact(ART, "oracle_cv_leaderboard.json")
print("Cross-validation leaderboard — target = Y_Soft_BT > 0")
display(rh.leaderboard_df(lb_soft).style.format(
    {"mean CV PR-AUC": "{:.4f}", "fold std": "{:.4f}"}).hide(axis="index"))
print("Cross-validation leaderboard — target = Y_Oracle")
display(rh.leaderboard_df(lb_orc).style.format(
    {"mean CV PR-AUC": "{:.4f}", "fold std": "{:.4f}"}).hide(axis="index"))
""")

code(r"""
rh.plot_cv_folds(lb_soft, "Y_Soft_BT > 0"); plt.show()
rh.plot_cv_folds(lb_orc, "Y_Oracle"); plt.show()
""")

md(r"""
On the soft target the structure is now **one big gap, in the right place: gradient boosting
far ahead, with random forest and the linear models bunched well behind.** The
interaction-driven question ("will this lot become harvestable within 30 days?") rewards
boosting's ability to combine a near-boundary loss *and* a slack utility margin *and*
wash-sale clearance over an unobserved future window — temporal structure no weighted sum of
today's features expresses. Notably, the forest no longer separates much from logistic here:
with the box-corner geometry gone from the label generator, bagging's advantage over a linear
score shrinks, while boosting's remains — a first hint of the attribution story the
[ablation section](#The-Headline:-Gated-versus-Scalarized-—-the-Oracle-Ablation) makes
precise. The fold-level view confirms the GBT gap dwarfs the spread within folds, so the
ranking is not a lucky-fold artifact.
""")

md(r"""
### Tuning results per model
""")

code(r"""
for key in ("gbt", "rf", "elnet", "logistic", "linreg"):
    print(f"--- {rh.MODEL_NAMES[key]} — grid, target = Y_Soft_BT > 0 ---")
    display(rh.tuning_table(rh.lb_entry(lb_soft, key)).style.format(
        {"mean CV PR-AUC": "{:.4f}"}).hide(axis="index"))
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
gbt_tbl = rh.tuning_table(rh.lb_entry(lb_soft, "gbt"))
for (lr, lv), grp in gbt_tbl.groupby(["learningRate", "numberOfLeaves"]):
    grp = grp.sort_values("numberOfTrees")
    axes[0].plot(grp["numberOfTrees"], grp["mean CV PR-AUC"], marker="o",
                 label=f"lr={lr}, leaves={lv}")
axes[0].set_xlabel("number of trees"); axes[0].set_ylabel("mean CV PR-AUC")
axes[0].set_title("GBT — capacity helps on every axis"); axes[0].legend(fontsize=8)

rf_tbl = rh.tuning_table(rh.lb_entry(lb_soft, "rf"))
for ff, grp in rf_tbl.groupby("featureFraction"):
    grp = grp.sort_values("numberOfTrees")
    axes[1].plot(grp["numberOfTrees"], grp["mean CV PR-AUC"], marker="s",
                 label=f"featureFraction={ff}")
axes[1].set_xlabel("number of trees"); axes[1].set_ylabel("mean CV PR-AUC")
axes[1].set_title("RF — the newly-tuned feature-sampling knob"); axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()
""")

md(r"""
GBT improves on every capacity axis — more trees, more leaves, the larger learning rate. The new
RF panel shows the **feature-fraction sweep**: a lower fraction forces the trees to disagree
(more decorrelation), and the chosen value of **«RF_FF_BEST»** is the bias–variance sweet spot for
this feature space — a knob that was silently frozen at 0.7 in the original submission and is now
searched, as a properly-tuned random forest requires.
""")

md(r"""
## Model Selection and Test-Set Performance

Only the two CV champions — **gradient boosted trees and random forest** — are evaluated on the
sealed test set; the linear models were eliminated by their CV scores without spending test data.
""")

code(r"""
rf_soft = rh.load_artifact(ART, "rf_soft_bt_metrics.json")
rf_orc  = rh.load_artifact(ART, "rf_oracle_metrics.json")
champs_soft = {"gbt": gbt_soft, "rf": rf_soft}
champs_orc  = {"gbt": gbt_orc,  "rf": rf_orc}

print("Test-set performance — target = Y_Soft_BT > 0")
display(rh.test_metrics_table(champs_soft).style.format("{:.4f}"))
print("Test-set performance — target = Y_Oracle")
display(rh.test_metrics_table(champs_orc).style.format("{:.4f}"))

rh.plot_roc_pr_overlay(champs_soft, "Y_Soft_BT > 0"); plt.show()
rh.plot_roc_pr_overlay(champs_orc, "Y_Oracle"); plt.show()
""")

code(r"""
print("GBT, soft target — threshold 0.5 (left) vs F1-optimal (right)")
display(pd.concat({"threshold = 0.5": rh.confusion_df(gbt_soft, "confusionAt05"),
                   "best threshold":  rh.confusion_df(gbt_soft, "confusionAtBest")}, axis=1))
""")

md(r"""
The robustness verdict is the good news of the scale-up: on the genuine prediction task — the
forward 30-day harvest propensity — **the gradient-boosted ensemble's skill survives two decades
and three crashes essentially intact** (test PR-AUC **«GBT_SOFT_TEST»** here vs 0.862 on the
two-year run), with the CV→test gap staying small. A model tuned on a smooth bull market
generalizes to 2008, 2020, and 2022 against an «SOFT_RATE» base rate. The PR curves show GBT
holding precision far deeper into the recall range than the forest; at equal recall the forest
raises substantially more false alarms (and its odd vote-scale "best threshold" reflects ML.NET's
uncalibrated random-forest scores, not a ranking defect).
""")

md(r"""
## The Headline: Gated versus Scalarized — the Oracle Ablation

The project's central experiment now has three acts.

**Act one — the trigger.** On the calm two-year window, logistic regression *nearly solved*
the gated oracle (CV PR-AUC 0.987): the gains gate was open on 100% of rows, collapsing the
four-way AND to essentially the half-space $L \le -0.02$, which a hyperplane represents
exactly. On the twenty-year window the same model **collapsed to ≈0.12** — the gains gate
finally bound (only in the GFC years), the box corner materialized, and no hyperplane could
carve it. That collapse was the empirical trigger for the diagnosis above: the non-linearity
that destroyed the linear models was *manufactured by a defective accounting rule*, not by
harvest economics.

**Act two — the fix.** The v0.25 oracle replaces the box with the level set $\{U(x)=0\}$.
Because the acting oracle changes the simulated trajectory itself, the comparison below is two
complete simulate→train pipelines — **gated vs scalarized, identical prices, identical d=17
feature schema** — so the only difference is the label-generating rule.

**Act three — the measurement:**
""")

code(r"""
lb_orc_g  = rh.load_artifact(ART_GATED, "oracle_cv_leaderboard.json")
lb_soft_g = rh.load_artifact(ART_GATED, "soft_bt_cv_leaderboard.json")

def _cv(lb, name):
    return rh.lb_entry(lb, name)["meanCvPrAuc"]

rows = []
for key in ("gbt", "rf", "logistic", "elnet", "linreg"):
    rows.append({
        "model": rh.MODEL_NAMES[key],
        "oracle — gated": _cv(lb_orc_g, key),
        "oracle — scalarized": _cv(lb_orc, key),
        "Δ oracle": _cv(lb_orc, key) - _cv(lb_orc_g, key),
        "soft — gated": _cv(lb_soft_g, key),
        "soft — scalarized": _cv(lb_soft, key),
        "Δ soft": _cv(lb_soft, key) - _cv(lb_soft_g, key),
    })
abl = pd.DataFrame(rows).set_index("model")
display(abl.style.format("{:+.4f}", subset=["Δ oracle", "Δ soft"])
           .format("{:.4f}", subset=[c for c in abl.columns if not c.startswith("Δ")])
           .background_gradient(subset=["Δ oracle"], cmap="RdYlGn", vmin=-0.2, vmax=0.75))

gap = lambda lb: _cv(lb, "gbt") - _cv(lb, "logistic")
print(f"GBT − logistic gap, oracle target:  gated {gap(lb_orc_g):+.4f}  →  "
      f"scalarized {gap(lb_orc):+.4f}")
print(f"GBT − logistic gap, soft target:    gated {gap(lb_soft_g):+.4f}  →  "
      f"scalarized {gap(lb_soft):+.4f}")
""")

md(r"""
Three findings, in decreasing order of drama.

**1. The attribution splits in two — and the byte-level discipline is what lets us split it.**
The historical logistic collapse (0.987 → 0.12) was measured under the old d=15 schema. The
gated *rerun* above, under the new d=17 schema with the identical gated oracle, already lifts
logistic to «LOGIT_ORACLE_GATED» — a pure **feature-set effect**: the capacity-aware
`TaxValue` is a far more informative coordinate than the old `TaxAlpha` (which valued winners'
gains as if they were losses and ignored offset capacity). The rest of the recovery — and the
leap of the *raw* linear models (linreg, elastic net) from ≈0.10 to ≈0.82 — is the
**oracle-swap effect**. It was the box-corner geometry, not linear-model capacity, that made
the gated oracle unlearnable.

**2. The oracle-target gap nearly closes — the honest result.** GBT still wins, but its edge
over logistic shrinks from ≈0.15 to ≈0.015. A smooth economic boundary is largely linearly
recoverable once `TaxValue` is visible, exactly as the design plan predicted ("a smooth
economic boundary partially closing the gap is the *more realistic* result, even if it softens
the v0.2 headline"). Real decision boundaries are smoother than toy AND-gates; the v0.2
tree-triumph on the oracle target was, in large part, an artifact of the defective gate.

**3. The headline relocates rather than dies.** On the *temporal* soft target — will a real
cost-benefit test clear within 30 unobserved days — the tree advantage is **oracle-invariant**:
GBT leads logistic by roughly the same margin in both arms. The genuinely hard, genuinely
non-linear problem was never the cross-sectional rule; it is the forward propensity, where
path-dependent interactions (how close is the loss to the boundary, how much utility margin,
when does the wash clock clear, does τ(h) flip at one year) compound over the window. That is
the project's durable scientific statement: **fixing the label generator's economics removed
the manufactured non-linearity and left the real one standing.**
""")

md(r"""
## Recovering the Value Function

The redesign's continuous target gives the tree-versus-linear question a cleaner regression
form: predict `Y_TaxValue` — the ledger's capacity-aware harvest value
$\tau(h)\cdot\min(\text{loss},\ \text{capacity}) + \tau_f\cdot\max(\text{loss}-\text{capacity},0)\cdot\delta$
— from raw features, with the `TaxValue` feature excluded (it *is* the target). The function
is zero-inflated, kinked at the capacity split, and jumps at the one-year holding boundary.
""")

code(r"""
taxreg = rh.load_artifact(ART, "tax_value_regression_metrics.json")
tr = pd.DataFrame(taxreg["models"]).set_index("modelName")
display(tr.style.format({"testRmse": "{:.2f}", "testMae": "{:.2f}",
                         "testR2": "{:.4f}", "testPosRmse": "{:.2f}"}))
print(f"features: {len(taxreg['features'])} (TaxValue excluded) | "
      f"positive rate: {taxreg['positiveRate']:.2%}")
""")

md(r"""
The gap that vanished from the classification problem reappears here, structurally: the linear
model manages **R² ≈ «TAXREG_LIN_R2»** while boosted regression trees reach
**R² ≈ «TAXREG_GBT_R2»** — because min/max kinks and a discrete rate jump are exactly what a
hyperplane cannot represent and axis-aligned splits can. This is the regression analogue of
the classification story, and it matters beyond pedagogy: this fitted value function is the
**warm start for the v0.4 reinforcement-learning layer**, whose per-decision reward is
precisely the exported `Y_Utility`.
""")

md(r"""
## Unsupervised Structure
""")

code(r"""
scree = rh.load_artifact(ART, "pca_scree.json")
elbow = rh.load_artifact(ART, "kmeans_elbow.json")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ks, ev, cv = scree["k"], scree["explainedVariance"], scree["cumulativeVariance"]
axes[0].bar(ks, ev, color="#357", alpha=0.75)
ax0b = axes[0].twinx(); ax0b.plot(ks, cv, color="#a37", marker="o", lw=2)
ax0b.axhline(scree["threshold"], color="#aaa", ls="--", lw=1); ax0b.set_ylim(0, 1.05); ax0b.grid(False)
ax0b.set_ylabel("cumulative", color="#a37")
axes[0].set_xlabel("principal component"); axes[0].set_ylabel("explained variance", color="#357")
axes[0].set_title(f"PCA scree — {scree['nKept']} PCs reach {scree['threshold']:.0%}")
axes[1].plot(elbow["ks"], elbow["inertia"], color="#357", marker="o", lw=2)
ax1b = axes[1].twinx(); ax1b.plot(elbow["ks"], elbow["silhouette"], color="#a37", marker="s", lw=2)
ax1b.set_ylabel("silhouette", color="#a37"); ax1b.grid(False)
axes[1].axvline(elbow["bestK"], color="#aaa", ls="--", lw=1)
axes[1].set_xlabel("k"); axes[1].set_ylabel("inertia", color="#357")
axes[1].set_title(f"K-means — best k = {elbow['bestK']}")
fig.tight_layout(); plt.show()
""")

md(r"""
The feature space stays **genuinely high-dimensional** at scale (now seventeen numeric
features including the ledger triple): the first component explains only a modest share of the
variance and it takes most of the components to clear 95% — no low-dimensional summary hides
inside, consistent with the modest correlation heatmap. K-means finds only weak clustering
(best $k$ small, silhouette low), as expected for a smooth panel of market states rather than
a mixture of distinct regimes — interestingly, even twenty years of *visibly* distinct regimes
do not separate into clean clusters in raw feature space, because the regimes live in the
*temporal* structure the unsupervised view discards.
""")

md(r"""
## Toward a Live Tax-Alpha System

The scale-up was framed as preparation for a live system (v0.4–v0.5). One backlog item has
already been retired by this version; the rest sharpen into a concrete sequence:

1. **~~The gains gate~~ — SHIPPED (v0.25, issue #23, PRs #24–#26).** The gate is out, the
   TaxLedger is in, and the ablation above measures exactly what the fix changed. What remains
   of the *tax* modeling is refinement, recorded as explicit non-goals until needed: split the
   blended ledger into typed short/long-term pools with cross-netting; replace the constant
   carryforward discount δ with the hazard-rate object it abbreviates
   (Pr(absorbed by a gain before death) × time value — carryforward dies with the taxpayer,
   Rev. Rul. 74-175); and let per-client outside-gain activity set δ. The corporate regime
   (bounded carryback/forward, where a gains-availability condition *is* economically real)
   stays out of scope.

2. **Cost-basis aging → contributions, rebalancing, and the sell-winner trim (v0.3).** The
   harvest signal thinning to «ORACLE_RATE» over twenty years is an artifact of the open-once /
   hold-forever design. A live account continuously mints fresh lots (contributions) and trims
   overweight winners back toward index weight (rebalancing) — which both *replenishes
   harvestable supply* and, crucially for the new ledger, **makes `RealizedGainsYTD`
   endogenous**: the trim transition is the first non-harvest action and pre-builds the action
   space the RL layer needs. This is now the highest-priority simulation change.

3. **Survivorship → point-in-time membership.** Restricting to «N_CONSTIT» names with full
   20-year history biases the universe toward long-lived large caps and makes the tax-alpha
   and tracking-error magnitudes optimistic. A production system needs a point-in-time
   constituent feed with additions, deletions, and corporate actions.

The endpoint is the **reinforcement-learning policy layer (v0.4)** — and the redesign built its
substrate. The per-decision reward is precisely the exported `Y_Utility`
$= \mathrm{TaxValue} - \lambda\sigma_{TE}^2 - c_{\mathrm{trade}}$, accumulated over an episode
on the deterministic ledger; the supervised models trained here are the value-function warm
start. The structural reason supervised learning cannot be the endpoint is already visible in
the data: offset capacity and the tracking-error budget are *shared, depletable resources*, so
lot-level decisions are conditionally dependent and the i.i.d.-row assumption breaks — a
policy, not a per-row classifier, is the right object. The oracle and soft labels were always
scaffolding; the twenty-year stress test plus the ablation tell us which scaffolding to keep
(the regime-robust soft-propensity target, the ledger) and which is already rebuilt (the gate).
""")

md(r"""
## Conclusion

Scaling the harvest-decision study from a two-year bull market to two decades — and then
following the evidence into a redesign of the harvest rule itself — produced four results.

**First, robustness:** gradient-boosted trees predicting the forward 30-day harvest propensity
hold ≈«GBT_SOFT_TEST» test PR-AUC against an «SOFT_RATE» base rate, stress-tested across 2008,
2020, and 2022. The genuine prediction problem survives the regime change — the single most
important thing a would-be live system needed to know.

**Second, the diagnosis:** the twenty-year window exposed the v0.2 gains gate as an accounting
artifact — vestigially open «GYTD_OPEN» of the time yet strangling harvesting mid-crisis — and
the historical logistic collapse (CV 0.987 on two years → ≈0.12 on twenty) turned out to be
that artifact's signature, not a deep fact about harvest economics.

**Third, the redesign and its measurement:** replacing the four-gate box with the
industry-faithful scalarized objective $U(x) = \mathrm{TaxValue} - \lambda\sigma_{TE}^2 -
c_{\mathrm{trade}}$ over a real Schedule D ledger, and rerunning both complete pipelines,
splits the old headline cleanly in two. On the cross-sectional oracle target the tree–linear
gap nearly closes (manufactured non-linearity, now gone); on the temporal soft target it
persists at full size in both arms (real non-linearity, still standing). The zero-inflated,
kinked value function itself is tree-recoverable (R² «TAXREG_GBT_R2») and linearly not
(R² «TAXREG_LIN_R2»). Honest science: the more realistic oracle *softened* the flashiest v0.2
result and replaced it with a better-founded one.

**Fourth, the long horizon still indicts the book's statics:** cost-basis aging all but
extinguishes the harvest signal after the first decade — the top of the v0.3 backlog
(contributions, rebalancing, and the sell-winner trim that makes realized gains endogenous).

The honest caveats are the roadmap: the universe is survivorship-biased toward long-lived
large caps; the labels are semi-synthetic (real prices, simulated portfolio); the loss-only
ledger models the conservative no-outside-gains persona, understating tax alpha for the
high-earner clients DI products target; the blended short/long-term pool and constant δ are
recorded simplifications; and a walk-forward temporal evaluation (train on the first decade,
test on the second) is the natural next robustness check beyond the stratified random split
used here. What carries forward is sharper than before: the harvest decision is learnable,
its forward propensity is predictable across regimes, the label generator now encodes real
tax law instead of a convenient fiction — and the exported objective `Y_Utility` is the
reward function the v0.4 policy layer will maximize.
""")

md(r"""
## References and Reproducibility

**Data**
- Financial Modeling Prep. (2026). *Historical Price EOD API* [Data set]. https://financialmodelingprep.com/
- State Street Global Advisors. (2026). *SPDR S&P 500 ETF Trust (SPY) Daily Holdings* [Data set]. https://www.ssga.com/

**Methodology**
- James, Witten, Hastie & Tibshirani (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.
- IRS Publication 550 — wash-sale rule and capital-loss carryforward. https://www.irs.gov/publications/p550
- 26 USC §1211(b) (capital-loss limitation, the \$3,000 ordinary-income allowance) and
  §1212(b) (indefinite individual carryforward) — the tax mechanics encoded in `TaxLedger`.
- Moussawi, Lo & Weisberger — Wealthfront Research, *Stock-Level Tax-Loss Harvesting*
  whitepaper (the production objective form — tax alpha minus λ·TE² — that the v0.25 oracle
  adopts). https://research.wealthfront.com/whitepapers/stock-level-tax-loss-harvesting/
- Betterment — *Tax Loss Harvesting+ Methodology* (the benefit-net-of-cost harvest test behind
  the $c_{\mathrm{trade}}$ term).
- Project memos: `DataMemo/SimulationMath.md`, `PortfolioMath.md`, `MLNetLeakageAudit.md`,
  `Lifecycle_v02.md`; the theory pair `DataMemo/data_memo_theory.md` (pre-plan) and
  `data_memo_theory_part2.md` (post-course reconciliation + v0.3–v0.4 program); and the
  ratified design record `DataMemo/GYTD_Redesign_Plan.md` (v2 — includes the measured
  ablation table in §6.1).

**Reproducing this report** (from `src/`; .NET 8 + [uv](https://docs.astral.sh/uv/)):

```text
export FMP_API_KEY=...
dotnet run download --from 2006-07-01 --to 2026-06-12   # custom window (≈20 yr = the API cap)
dotnet run simulate                                     # scalarized oracle → data/lots.csv
dotnet run simulate --oracle=gated                      # ablation arm → data/lots_gated.csv
                                                        # (--ctrade=0 for the frictionless arm)
dotnet run mlnet-all                                    # CV all models, test champions, artifacts
                                                        # (mlnet-soft / mlnet-oracle re-run one target;
                                                        #  repeat per arm, keeping artifacts-mlnet-gated/)
dotnet run mlnet-tax                                    # Y_TaxValue regression (TaxValue excluded)
dotnet run mlnet-unsupervised                           # PCA / K-means on the canonical arm
cd ML/Python && uv run python -m scripts.fill_report_tokens && cd ../..
dotnet run report                                       # codebook + execute this notebook + HTML
```

The notebook itself is generated by `scripts/build_report_notebook.py`; edit that script (not
the `.ipynb`) and regenerate, so the prose, the token-filling, and the executed report cannot
drift apart.
""")

code(r"""
import sys, matplotlib
print("software versions:")
print("  python    ", sys.version.split()[0])
for mod in (pd, np, matplotlib):
    print(f"  {mod.__name__:<10}", mod.__version__)
print("  ML.NET     3.0.1 (C# training pipeline)")
""")

nb["cells"] = cells
out = Path("notebooks/final_report.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print(f"wrote {out} with {len(cells)} cells")

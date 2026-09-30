# Direct Indexing ML — tax-loss-harvesting decisions in a simulated portfolio

A C# (.NET 8) + Python pipeline that **manufactures its own dataset and learns from it**:
it replays real S&P 500 prices through a simulated \$10M direct-indexing portfolio, labels
every tax lot on every day with a harvesting rulebook, and trains supervised models to predict
*harvest propensity* — "will it be worth harvesting this lot within the next 30 days?"

> **New here, or coming back after a while?** Read this README for the front-door tour (what it
> is, how to run it, the methodology). For the full project recap, the comparison to the original
> proposal, and the v0.3–v0.4 plan, see **[`PSTAT231_RECAP.md`](DataMemo/archive/PSTAT231_RECAP.md)**. For the
> first-principles math walk of the whole codebase, see **[`DataMemo/archive/Lifecycle_v02.md`](DataMemo/archive/Lifecycle_v02.md)**. (PSTAT 231 being the intro-ML grad project based course)

**Status:** v0.1 → **v0.26 complete**. v0.2 was the PSTAT 231 final submission; the pipeline was
then **scaled from a fixed 2-year window to a custom range of up to ~20 years**
(`download --from/--to`) and re-run on 2006–2026 (1.85M lot-day rows through the 2008, 2020, and
2022 drawdowns). That scale-up exposed two defects, and fixing them is what v0.25/v0.26 were:

- **v0.25 — oracle redesign (issue #23).** The `G_YTD` gains gate was economically invalid for an
  individual investor, so it was replaced by a `TaxLedger` and a **scalarized objective**
  `U = TaxValue − λσ_TE² − c_trade` behind three hard gates. Measured consequence: the
  oracle-target tree-over-linear gap collapsed `0.155 → 0.015` — much of the original "trees win"
  headline was *manufactured by the defective gate*.
- **v0.26 — validation hardening.** Purged chronological splits (`--split=temporal`). ROC-AUC held
  across time and the deterministic-oracle control stayed ~1.0, so **leakage is ruled out**; the
  PR-AUC drop is the cost-basis-aging **prevalence crash**.

The surviving headline is stronger than the original: the tree advantage on the **temporal**
(forward-propensity) target is real and oracle-invariant. Next milestone: **v0.3 — simulator
realism**, whose P0 is the cost-basis-aging fix that v0.26 diagnosed. See [Results](#results)
and [`ROADMAP.md`](ROADMAP.md).

---

## Table of contents

- [What this is, in one minute](#what-this-is-in-one-minute)
- [Why it's built this way (the core idea)](#why-its-built-this-way-the-core-idea)
- [Architecture & lifecycle](#architecture--lifecycle)
- [The data point — what one row means](#the-data-point--what-one-row-means)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Command reference](#command-reference)
- [Results](#results)
- [Methodology & philosophy](#methodology--philosophy)
- [Roadmap](#roadmap)
- [Further reading](#further-reading)

---

## What this is, in one minute

**Direct indexing** = holding the individual stocks of an index (instead of one index fund) so you
can sell *individual lots* at a loss to realize a tax benefit ("tax-loss harvesting"), while a fund
holder cannot. Selling a loser banks a **tax alpha** — `τ · |loss|` dollars of offset against
realized gains — at the cost of a little **tracking error** (drift from the index), which you can
neutralize by rebuying a correlated substitute or the same stock 30 days later (the wash-sale rule).

The decision *"should I harvest lot k today?"* has a correct answer computable from observable state —
a four-condition rulebook we call the **oracle**. That makes it a well-posed machine-learning problem.
But the *valuable* question isn't "does the rule fire today?" (you can just run the rule); it's
**"will the rule fire within the next 30 trading days?"** — harvest *propensity* — which depends on
unknown future prices and can only be *estimated* from today's state. That estimation is what the
models learn.

No public dataset records lot-level tax state under this rulebook. So the project **generates** the
dataset by simulating a portfolio over real historical prices.

## Why it's built this way (the core idea)

The whole codebase derives from the tax code in five steps:

1. **Tax asymmetry → why direct indexing exists.** A lot below cost basis holds an *option* worth
   `τ(h)·|qₖ(Pₜ − pₖ)|` dollars (`τ` = 37% short-term / 20% long-term). That option only exists at
   **lot granularity** — an index-fund holder owns no lots.
2. **The decision has a correct answer → the oracle.** Harvesting is constrained by a loss
   threshold, a wash-sale lockout, a tracking-error ceiling, and — the economic part — whether
   the harvest is *worth its cost*:

   ```
   f*(x) = 𝟙[L ≤ −2%] · 𝟙[WashClock ≥ 30] · 𝟙[σ_TE ≤ θ_max] · 𝟙[U(x) > 0]
   where  U(x) = TaxValue − λ·σ_TE² − c_trade
   ```

   The decision boundary is the **level set** `{U = 0}` — a smooth surface, not a box corner.

   > *This changed in v0.25.* The original rule was a four-way AND whose third gate required
   > `G_YTD > 0` ("the portfolio must have realized gains this year") — a convex **polytope**.
   > That gate turned out to be economically invalid for an individual investor (capital losses
   > carry forward indefinitely), so the gains information moved *inside* `TaxValue` as a
   > capacity-aware value term. Removing it collapsed the tree-over-linear gap on the oracle
   > target from 0.155 to 0.015 — the polytope corner, not problem structure, had been doing
   > the work. See [`DataMemo/decisions/GYTD_Redesign_Plan.md`](DataMemo/decisions/GYTD_Redesign_Plan.md).
3. **Why simulate → the rulebook is executable.** Replay real prices through a portfolio that obeys
   the rule and log every `(lot, day)` state. The simulator *is* the problem's physics (with real
   price risk), not an approximation of it.
4. **Why ML when `f*` is known code →** (a) a *recoverability sanity check* (a broken pipeline can't
   relearn a deterministic rule from 170K labels); (b) the real target is the **future**, which `f*`
   can't see; (c) the learned propensity model `η̂(x) ≈ E[future harvestability | x]` is a value-function
   surrogate — the stepping stone to a reinforcement-learning policy later.
5. **The invariant that governs everything:**

   > **Labels may peek at the future (at dataset-construction time). Features never may.**

   Soft labels are computed from prices at *t+1 … t+30* — legal, because labels are the answer key and
   exist only at training time. Every *feature* is computable from information available at day *t*.
   At deployment the future doesn't exist; the model bridges the gap. That is what supervised learning
   *is*, made explicit.

## Architecture & lifecycle

Four layers, each a **map between files on disk**. Layers communicate *only* through serialized
artifacts — so any layer can be rerun, replaced, or audited in isolation. That file-boundary design is
the architectural thesis of the project.

```mermaid
flowchart LR
    FMP[(FMP API + SSGA<br/>SPY holdings)] --> L1
    L1["**1 · download**<br/>DataCollection"] -->|"data/raw/*.json<br/>constituents.json"| L2
    L2["**2 · simulate**<br/>Core/Simulation"] -->|"data/lots.csv<br/>(the dataset)"| L3
    L3["**3 · mlnet-all**<br/>ML/CSharp/MLNet"] -->|"data/artifacts-mlnet/*<br/>(leaderboards, metrics,<br/>coeffs, model zips)"| L4
    L4["**4 · report**<br/>ML/Python"] -->|"src/Export/report/<br/>notebook + HTML"| OUT([deliverable])
    L2 -.->|"lots.csv (live EDA cells)"| L4
```

| Stage | Command | Input → Output | What happens |
|---|---|---|---|
| **1 · Download** | `download [--from YYYY-MM-DD --to YYYY-MM-DD]` | API → `data/raw/`, `constituents.json` | Fetch SPY constituents (SSGA holdings xlsx) + per-ticker EOD price history (FMP). The window is a **parameter**: pass explicit `--from`/`--to` dates (up to ~20 years — the API's ≈5,000-bar-per-ticker cap), or omit both for a rolling 2-year window. Either way the fetch extends ~200 trading days back so `MA_200` is defined on day one, and an existing cache is re-aggregated incrementally. |
| **2 · Simulate** | `simulate [--oracle=gated\|scalarized]` | prices → `data/lots.csv` | Open an equal-dollar \$10M portfolio, step day by day: value it, compute tracking error, update the `TaxLedger`, snapshot every open lot, apply the oracle, harvest + queue a 30-day rebuy when it fires. Since **v0.25** the default oracle is the **scalarized** `f* = 𝟙[ℓ≤−θ₁]·𝟙[𝒲≥30]·𝟙[σ_TE≤θ_max]·𝟙[U>0]`; `--oracle=gated` reruns the v0.2 four-gate rule as the ablation baseline → `data/lots_gated.csv`. Each row is one `(lot, day)` observation with features **and** labels. |
| **3 · Train** | `mlnet-all` | `lots.csv` → `data/artifacts-mlnet/` | Cross-validate 5 models on 2 targets, select champions, evaluate only the champions on the sealed test set, emit metrics/leaderboards. Plus PCA + K-means diagnostics. |
| **4 · Report** | `report` | artifacts → `src/Export/report/` | Generate a codebook (with a schema-drift assert against `lots.csv`), execute the analysis notebook, export self-contained HTML. |

A run is the composition `download → simulate → mlnet-all → report`. Each step is **idempotent** and
**replayable from its on-disk inputs**, so you can rerun any stage without redoing the ones before it
(as long as their outputs exist).

## The data point — what one row means

One row of `data/lots.csv` is an immutable "photograph" of one lot at one day — the type
[`LotStateVector`](src/Core/Portfolio/LotStateVector.cs), which is the load-bearing schema of the
whole codebase. Since **v0.25** it is **schema v3** (`d = 17` numeric features, 26 columns): the
`G_YTD` scalar became the three-field `TaxLedger` block, `TaxAlpha` became the capacity-aware
`TaxValue`, and three new labels were added.

| Group | Columns |
|---|---|
| **Lot-level** | `L` unrealized return · `H` holding days · `S` short/long flag · `B` cost basis · `W` lot weight · `K` open lots in same ticker |
| **Portfolio-level** (TaxLedger) | `RealizedGainsYTD` signed net realized P&L YTD (the pre-v0.25 `G_YTD`) · `LossCarryforward` banked losses, survives year-end (26 USC §1212(b)) · `OrdinaryOffsetBudget` remaining $3k/yr allowance · `Sigma_TE` tracking error · `WashClock` days since last harvest (999 = never) |
| **Asset-level** | `R_t` daily return · `SigmaRange` range-vol proxy · `DeltaMA50` · `DeltaMA200` |
| **Derived** | `TaxValue` = τ(h)·min(loss, capacity) + τ_f·max(loss−capacity, 0)·δ · `DaysToYE` |
| **Labels** | `Y_Oracle ∈ {0,1}` (hard, "fires today?") · `Y_Soft_BT ∈ [0,1]` (fraction of next 30 real days the rule fires; NaN near the window end) · `Y_Soft_GBM ∈ [0,1]` (same, over 200 simulated paths) · `Y_TaxValue ∈ ℝ≥0` (regression target = `TaxValue`) · `Y_Utility ∈ ℝ` (the scalarized objective `U(x)`, the future RL reward) · `Y_Oracle_GatedSpec ∈ {0,1}` (what the v0.2 gated oracle would say on this row — the ablation spectator) |
| **Metadata** | `Symbol` · `Sector` · `Timestep` *(dropped before modeling)* |

The soft labels are the project's real target: holding portfolio state frozen, *would the oracle fire
in the next 30 days?* — `Y_Soft_BT` averages that 0/1 predicate along the one real price path;
`Y_Soft_GBM` counts first-passage over 200 simulated paths. There is **no optimization and no ML inside
the labeler** — it evaluates a fixed rule forward and averages.

## Repository layout

```
.
├── PSTAT231_RECAP.md          ← start here for the full recap + v0.3–v0.4 roadmap
├── README.md                 ← you are here
├── DirectIndexing.sln
├── src/
│   ├── Program.cs            ← the orchestrator: one switch maps command → layer
│   ├── DataCollection/       ← Layer 1: MarketDataDownloader, Models
│   ├── Core/
│   │   ├── Portfolio/        ← Lot, PortfolioState, LotStateVector (the schema)
│   │   ├── Oracle/           ← OracleBoundary (the pure, stateless rulebook f*)
│   │   └── Simulation/       ← PriceLoader, SimulationEngine, SoftLabelBuilder,
│   │                            TrackingErrorProxy, GbmSimulator, MonteCarloEngine
│   ├── Export/               ← SimulationExporter + generated plots/report (gitignored)
│   ├── Tests/                ← state machine, oracle gates, TE invariants, leakage regressions
│   └── ML/
│       ├── CSharp/MLNet/     ← Layer 3: trainers, splits, preprocessing, tuning, metrics
│       └── Python/           ← Layer 4: report/codebook/eda/render scripts + notebook (uv-managed)
├── DataMemo/                 ← design + math docs (see Further reading)
└── data/                     ← raw cache, lots.csv, artifacts (all gitignored — re-derivable)
```

## Prerequisites

| Tool | Version | Needed for |
|---|---|---|
| [.NET SDK](https://dotnet.microsoft.com/download) | **8.0** | everything (the orchestrator + all C# layers) |
| [`uv`](https://docs.astral.sh/uv/) | recent | Layer 4 (report) — manages the Python ≥ 3.11 environment under `src/ML/Python/` |
| FMP API key | — | **only** `download`. Set `FMP_API_KEY` in your environment. Not needed if you already have `data/raw/` (it ships in the submission zip). |

The Python environment is created/used automatically by the `report` commands via the `PythonRunner`
subprocess seam — you don't normally invoke Python directly. To set it up manually:
`cd src/ML/Python && uv sync`.

## Quickstart

```bash
# 0. (first time only) point at your FMP key — skip if data/raw/ already exists
export FMP_API_KEY=your_key_here

# 1. fetch prices + constituents              → data/raw/, constituents.json
dotnet run --project src -- download                                  # rolling 2-year window
dotnet run --project src -- download --from 2006-07-01 --to 2026-06-12  # or any custom range (≤ ~20 yr)

# 2. simulate the portfolio + label every lot → data/lots.csv
dotnet run --project src -- simulate

# 3. cross-validate 5 models, test champions  → data/artifacts-mlnet/
dotnet run --project src -- mlnet-all

# 4. build the report (codebook + notebook + HTML) → src/Export/report/
dotnet run --project src -- report
```

Or run the whole training+report in one shot once `lots.csv` exists:

```bash
dotnet run --project src -- report-all     # = mlnet-all then report
```

Verify everything works:

```bash
dotnet run --project src -- test           # 30+ assertions incl. leakage regressions
```

> You can also `cd src && dotnet run <command>` instead of the `--project src --` form.

## Command reference

**Pipeline (the main path):**

| Command | Layer | What it does |
|---|---|---|
| `download [--from D --to D]` | 1 | Fetch constituents + EOD prices → `data/raw/`, `constituents.json` (needs `FMP_API_KEY`). Custom date range with both flags, rolling 2-year window with neither |
| `simulate` | 2 | Backtest the \$10M portfolio, label every lot-day → `data/lots.csv` |
| `simulate-mc` | 2 | Monte-Carlo variant on synthetic GBM prices → `data/lots-mc.csv` |
| `mlnet-all` | 3 | CV all 5 models × 2 targets, test the champions, render → `data/artifacts-mlnet/` |
| `report` | 4 | Codebook + execute `final_report.ipynb` + export HTML → `src/Export/report/` (exits 2 with a list if artifacts are missing) |
| `report-all` | 3+4 | `mlnet-all` then `report`, one command |
| `submission` | packaging | Assemble `submission.zip` at the repo root (deliverables at zip root; raw cache included so it reproduces with no API key; `--no-data` drops `lots.csv`) |

**Devtools & granular subcommands:**

| Command | What it does |
|---|---|
| `test` | Run the test suite (portfolio state machine, oracle gates, TE invariants, GBM stats, splits/imputation/weights/grid-search, **leakage regressions**) |
| `deps` | Regex-scan `src/**/*.cs` → dependency/coupling atlas (`src/Export/diagrams/Dependencies.md`: mermaid layer/class/inheritance graphs + fan-in/out tables) |
| `mlnet-eda` / `mlnet-unsupervised` / `mlnet-render` | Run individual ML.NET sub-stages (EDA stats, PCA+K-means, Python rendering) |
| `mlnet-gbt` / `mlnet-rf` / `mlnet-elnet` / `mlnet-linreg` / `mlnet-supervised` | Train a single model family in isolation |
| `mlnet-soft` / `mlnet-oracle` | Re-run champion selection + champion test eval for **one target** (finish a partial `mlnet-all`, or retrain after an oracle change) |
| `mlnet-tax` | **(v0.25)** Regress the continuous `Y_TaxValue` (the `TaxValue` feature is excluded — the task is recovering the ledger function from raw state); SDCA-linear vs FastTree, R² 0.10 vs 0.92 |
| `mlnet-compare` | Run the CV leaderboard / champion comparison without the full pipeline |

### Modes, splits & artifact layout (v0.25–v0.26)

Two families of flags reshape the pipeline without new commands. **Oracle flags** shape the
*label generator* (`simulate` / `simulate-mc` only — passing them to an `mlnet-*` mode prints a
warning, since those read whatever `lots.csv` is on disk):

| Flag | Default | Effect |
|---|---|---|
| `--oracle=gated\|scalarized` | `scalarized` | Which oracle acts. The acting oracle changes the trajectory itself (harvests → wash clocks → ledger → which rows exist), so the two are **separate runs**, not label columns: `lots.csv` vs `lots_gated.csv`. |
| `--ctrade=<dollars>` | `10` | The flat round-trip harvest friction in `U = TaxValue − λσ_TE² − c_trade` (scalarized only). `--ctrade=0` = the frictionless ablation arm. |

**Split flags** shape *evaluation* (any `mlnet-*` mode; **v0.26**). The soft labels look 30 days
forward, so adjacent rows share future context — a stratified *random* split can leak it. Temporal
mode is the honest alternative:

| Flag | Default | Effect |
|---|---|---|
| `--split=temporal` | stratified-random | Chronological purged split: train precedes test with a purge/embargo gap ≥ the label horizon, so no training row's forward window overlaps the test period. |
| `--embargo=<days>` | `30` | The purge gap width (= the `Y_Soft_BT` horizon). |
| `--testfrac=<0..1>` | `0.20` | Test fraction. `--testfrac=0.5` = the **decade walk-forward** (train ~2006–2016, test ~2016–2026). |

**Artifact directories** (all under `data/`, gitignored, re-derivable) keep the arms from
clobbering each other — the directory name records *how the numbers were produced*:

| Directory | Produced by | Holds |
|---|---|---|
| `artifacts-mlnet/` | `mlnet-*` on `lots.csv`, random split | **the canonical** scalarized-oracle results |
| `artifacts-mlnet-gated/` | same, after swapping `lots_gated.csv` → `lots.csv` | the v0.2 gated-oracle **ablation baseline** (the 2×2 comparison in the report) |
| `artifacts-mlnet-temporal/` | `mlnet-* --split=temporal` | the honest-split results (last written wins between the 80/20 and `--testfrac=0.5` decade runs) |
| `artifacts-mlnet-temporal-8020/` | a preserved copy of the 80/20 temporal run | kept aside so the decade run doesn't overwrite it — see [`DataMemo/decisions/ValidationHardening_v026.md`](DataMemo/decisions/ValidationHardening_v026.md) |

## Results

**Headline (PSTAT 231 v0.2 submission — 2-year window, 170,751 rows):**

| Model | soft target (base rate 19.9%) | oracle target (sanity check) |
|---|---|---|
| **Gradient-boosted trees ★** | CV **0.858** → test **0.862** PR-AUC | 0.9997 |
| **Random forest ★** | 0.694 | 0.9987 |
| Logistic (L2) | 0.650 | 0.987 |
| Elastic net | 0.635 | 0.442 |
| Linear regression (demo) | 0.559 (~10% of predictions escape `[0,1]`) | 0.372 |

What the numbers *mean* (the science, not the recitation):

- **GBT's CV→test gap is +0.004** — the honesty signature: tuned cleanly, nothing leaked or overfit to
  folds. At the F1-optimal threshold: ~76% precision / ~83% recall against a 19.9% base rate.
- **The linear tier (~0.65) is representation-limited, not regularization-limited** — every linear model
  chose its *weakest* penalty, so the ceiling is the functional form, not overfitting.
- **The GBT–RF gap (0.16)** — same trees, different composition: boosting *sharpens* the thin positive
  region that bagging *blurs*.
- **Trees recover the known oracle near-perfectly (≥ 0.999)** — the recoverability sanity check passes.

**The 20-year stress test (current live run — 2006–2026, 1,846,015 rows, 409 survivor tickers):**

| Metric | 2-yr submission | 20-yr run |
|---|---|---|
| `Y_Oracle` / soft base rate | 1.6% / 19.9% | **0.20% / 2.47%** (cost-basis aging) |
| GBT — soft target, CV → test PR-AUC | 0.858 → 0.862 | **0.850 → 0.844** ✅ robust |
| GBT — oracle target, test PR-AUC | 0.9997 | 0.975 |
| Logistic — oracle target, CV PR-AUC | 0.987 | **0.123** ← the flip |
| `G_YTD > 0` gate open | 100% of rows | 94.3% (all closure in 2008–09) |

Three things the scale-up established (full narrative in the rewritten
[`final_report.ipynb`](src/ML/Python/notebooks/final_report.ipynb)):

- **Robustness** — the genuine prediction task (forward 30-day harvest propensity) survives two
  decades and three crashes essentially intact, against a 12× rarer base rate.
- **The geometry flip** — logistic regression nearly solved the oracle on the calm 2-year window
  and collapses on the 20-year one: *a conjunction is only as non-linear as the gates the data
  makes bind*, and that is set by the regime, not the rule.
- **Two simulation defects exposed** — lots opened once age out of harvestability (the harvest
  signal is nearly extinct after the first decade), and the `G_YTD` gains gate is both vestigial
  (open 94% of the time) and **self-strangling** (the 2008–09 harvest bursts burn the $1M seed to
  zero and freeze harvesting mid-crisis). The latter is now **issue #23 / v0.25**: the gate is
  misaligned with beta-tracking direct indexing (losses carry forward under US tax law — see the
  Wealthfront stock-level TLH whitepaper), so it will be removed/redesigned and all models
  retrained on the corrected oracle.

> When quoting a number, know which window you mean — and note the oracle-target numbers sit on
> the 4-gate rule that v0.25 deprecates. Full divergence story:
> [`PSTAT231_RECAP.md` §6](DataMemo/archive/PSTAT231_RECAP.md).

## Methodology & philosophy

Five disciplines hold the project together. Breaking any of them silently corrupts the results — they
are the things to preserve as the project grows.

- **Labels may peek at the future; features never may.** The single invariant (see [core idea](#why-its-built-this-way-the-core-idea) #5). Every new feature must be computable from information available at day *t*; only labels may use *t+1 … t+30*.
- **Leakage invariants are types, not conventions.** Median imputation and class weights have *no overload* that accepts the full dataset — they only accept a training fold. Normalization and one-hot encoding live inside the fitted pipeline. The leakage mistake scikit-learn lets you make implicitly, the C# layer makes *unrepresentable*. (Details: [`DataMemo/spec/MLNetLeakageAudit.md`](DataMemo/spec/MLNetLeakageAudit.md).)
- **Champion selection is enforced by code shape.** Cross-validation ranks all five models; only the top-two classifiers ever reach the function that touches the sealed test set. "Only the best one or two models touch test" is a *structural* property, not a promise.
- **PR-AUC, not ROC-AUC, for rare positives — but report both since v0.26.** With a rare positive rate,
  ROC-AUC is blind to a flood of false positives; precision-recall AUC punishes drowning the true
  positives in alarms. But the two measure different things and v0.26 uses that: ROC-AUC is
  prevalence-*insensitive* (pure ranking → the leakage control), PR-AUC is prevalence-*bounded* (its
  no-skill floor is the base rate). Under honest temporal splits ROC held while PR-AUC fell — the drop
  was the aging prevalence crash, not lost skill. So the standing rule is now **ROC-AUC + PR-AUC +
  test-period prevalence, together**. (And PR-AUC is still *not* the deployment objective — a model with
  higher PR-AUC can produce *less* tax alpha if it fires on correlated lots at once; that economic
  evaluation is the next layer, see roadmap.)
- **Layers communicate only through files.** No hidden shared mutable state crosses a layer boundary;
  the entire input to any layer is inspectable on disk. This is what lets the Python report layer be
  swapped, the ML layer be rewritten (it was — from Python to ML.NET), or any stage be audited alone.

> **A note on scope.** This started as a PSTAT 131/231 course project. The professor's guidance was to
> simplify to "a fixed dataset + one binary outcome + multiple models." This project keeps that clean
> modeling core but deliberately keeps the **simulator** — because there *is* no fixed dataset to pull
> for this problem; the simulator is how the tabular dataset is manufactured, with real price risk. The
> trade-offs of that choice, and how the final project diverged from the original proposal, are written
> up in [`PSTAT231_RECAP.md` §5](DataMemo/archive/PSTAT231_RECAP.md).

## Roadmap

Conceptually, the long arc is: **supervised oracle approximation → continuous tax-value modeling → a
reinforcement-learning policy that needs no hand-coded rule at all.**

| Version | Timeline | Focus |
|---|---|---|
| **v0.1** | PSTAT 231 (Spring 2026) | ✅ Supervised baseline: hard + soft labels, ~15 features, 4+ models with k-fold CV |
| **v0.2** | PSTAT 231 (Spring 2026) | ✅ Champion selection, PCA/K-means, report layer + submission packaging |
| **20-yr scale-up** | June 2026 | ✅ Custom download range (`--from`/`--to`, issues #19/#20); 1.85M-row multi-regime re-run; report rewritten against it |
| **v0.25** | Summer 2026 | ✅ Oracle redesign (issue #23, merged): `G_YTD` gate removed → `TaxLedger` + scalarized objective `U = TaxValue − λσ_TE² − c_trade` behind 3 hard gates; `--oracle=gated\|scalarized` ablation arms + `Y_TaxValue`/`Y_Utility`/spectator labels; all models retrained. Measured: oracle-target GBT–LR gap 0.155→0.015 (box geometry, not model capacity, drove the v0.2 gap); soft-target tree advantage oracle-invariant |
| **v0.26** | Summer 2026 | ✅ Validation hardening (PR #30): `TemporalSplit`/`SplitPolicy`/`DataSplit` purged chronological splits (`--split=temporal`). Finding: ROC-AUC holds across time (ranking transfers, **no leakage** — the deterministic oracle target is the flat control), PR-AUC drop is the cost-basis-aging prevalence crash — which is exactly the v0.3 mandate |
| **v0.3** | Summer / Junior Fall | Cost-basis-aging fix is now **P0** (contributions/rebalancing + sell-winner trim → makes `RealizedGainsYTD` endogenous, v0.4 action-space scaffolding); volatility sub-model (design fully, build-gate on a decision-flip ablation); richer soft labels (#17); tax-alpha metric layer + 6-rung baseline ladder (#12/#22) |
| **v0.4** | Junior Year | **v0.4a** (new): constrained-optimizer execution baseline (GBT + subset + replacement optimizer under shared budgets) — RL must beat *this*, not just GBT. **v0.4b**: RL policy layer (#15) warm-started from the supervised η̂; reward = the shipped `U(x)` accumulated over episodes |
| **v0.5** | Senior Capstone | Full-system integration: live data, client-parameterized policies, real-history backtests |
| **v1.0** | Post-graduation | Production deployment; RIA-style direct-indexing service |

The eventual deployment goal is a live direct-indexing system (the author as its own first client), with
the open research bet being whether RL and neural methods recover meaningfully more tax alpha — at equal
or lower tracking error — than the supervised oracle-approximation baseline.

**The full version-by-version plan, the open-issue ledger, and the recommended critical path for v0.3
and v0.4 live in [`PSTAT231_RECAP.md` §7–§9](DataMemo/archive/PSTAT231_RECAP.md).**

## Further reading

| Document | What's in it |
|---|---|
| [**`ROADMAP.md`**](ROADMAP.md) | **The authoritative version planner** (v0.1 → v1.0): what each version delivers, its gate criteria, and the standing rules. When it and an older doc disagree, it wins. |
| [`PSTAT231_RECAP.md`](DataMemo/archive/PSTAT231_RECAP.md) | The orientation doc: full v0.1–v0.2 recap, current-state vs. frozen-submission divergence, and the comparison to the original proposal. (Roadmap sections here are superseded by `ROADMAP.md`.) |
| [`DataMemo/archive/Lifecycle_v02.md`](DataMemo/archive/Lifecycle_v02.md) | First-principles walk of the *entire* codebase: every layer's signature, the day-loop sequence, the lot lifecycle state machine, and the math each piece implements. |
| [`DataMemo/spec/SimulationMath.md`](DataMemo/spec/SimulationMath.md) · [`PortfolioMath.md`](DataMemo/spec/PortfolioMath.md) | The simulation and portfolio mathematics (ledger transitions, year-end roll, tracking-error derivation, the endogenous dataset-size identity). |
| [`DataMemo/spec/MLDerivations.md`](DataMemo/spec/MLDerivations.md) | **The ML mathematics, layered** — plain-language orientation (§0), the typed working body (§1–§8: feature space, oracle, label family, protocol, per-model objectives), and an every-symbol-pinned appendix. Current at schema v3 / scalarized oracle. |
| [`DataMemo/spec/MLNetLayer.md`](DataMemo/spec/MLNetLayer.md) | **The ML.NET layer** — why C# and not sklearn, the typed pipeline shape, and the complete sklearn ↔ ML.NET parameter/solver/metric reconciliation. |
| [`DataMemo/spec/MLNetLeakageAudit.md`](DataMemo/spec/MLNetLeakageAudit.md) | Where every fit happens and why none of them leak. |
| [`DataMemo/decisions/GYTD_Redesign_Plan.md`](DataMemo/decisions/GYTD_Redesign_Plan.md) | The gains-gate redesign v2 (shipped in v0.25): the scalarized oracle, the `TaxLedger`, and §6.1's **measured** gated-vs-scalarized ablation table. |
| [`DataMemo/decisions/ValidationHardening_v026.md`](DataMemo/decisions/ValidationHardening_v026.md) | **(v0.26)** Why the old random splits were suspect, and the ROC-AUC-vs-PR-AUC diagnosis that ruled out leakage and isolated the cost-basis-aging prevalence crash. |
| [`DataMemo/archive/architecture_thread/direct_indexing_concept_architecture_plan_contextualized.md`](DataMemo/archive/architecture_thread/direct_indexing_concept_architecture_plan_contextualized.md) | Parity assessment of a ChatGPT-5.5-Pro architecture thread (companion `..._plan.md`) against the actual repo, plus the synthesized v0.25→v1.0 version planner (the deep expansion of this Roadmap). Slated for promotion to `DataMemo/ArchitecturePlan.md`. |
| [`DataMemo/archive/data_memo_theory.md`](DataMemo/archive/data_memo_theory.md) · [`data_memo_theory_part2.md`](DataMemo/archive/data_memo_theory_part2.md) | The theory pair: the pre-implementation formal framework, and the post-course reconciliation (what converged/deviated/emerged) + the v0.3–v0.4 theoretical program (GARCH, covariance cleaning, tax ledger, RL MDP). |
| [`src/ML/Python/notebooks/final_report.ipynb`](src/ML/Python/notebooks/final_report.ipynb) | The rewritten 20-year analysis report (generated by `scripts/build_report_notebook.py`; the frozen 2-year submission version lives in `src/Export/report/`). |
| [`DataMemo/archive/DataMemo.ipynb`](DataMemo/archive/DataMemo.ipynb) | The original pre-implementation proposal (with the professor's feedback). |


## Project Images: 

<img width="1679" height="836" alt="Screenshot 2026-06-29 at 6 49 28 PM" src="https://github.com/user-attachments/assets/0ada718f-1cdb-43b0-a427-88d0c4615539" />


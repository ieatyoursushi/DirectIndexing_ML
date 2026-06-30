# Project Recap — Direct Indexing ML
### v0.1–v0.2 complete · orientation + roadmap into v0.3–v0.4

> **Why this file exists.** Post-v0.2 you're starting to drift from what's *fundamentally*
> implemented. This is the single document to re-read when that happens: what the project is,
> what each layer actually does and why, how far it diverged from the original DataMemo pre-plan,
> what the current (post-submission) repo state is vs. the frozen PSTAT 231 deliverable, and the
> concrete recommended path through v0.3 (this summer) and v0.4 (next year). It **subsumes the
> README** (commands + roadmap reproduced in §3 and §8) and points to the deeper synthesis docs
> in `DataMemo/` for the math.
>
> Companion docs (do not duplicate — read for depth):
> `DataMemo/Lifecycle_v02.md` (full first-principles codebase walk),
> `DataMemo/SimulationMath.md`, `DataMemo/PortfolioMath.md`,
> `DataMemo/ML_Derivations_Explicit_Rigorous_DollarMath.md`,
> `DataMemo/MLNetLeakageAudit.md`, `DataMemo/GYTD_Redesign_Plan.md` (the v0.3 design).

---

## 0. TL;DR — where the project stands (read this first)

You built a **four-layer C#/.NET 8 pipeline that manufactures its own dataset and learns from it**,
because no public dataset records lot-level tax state under a harvesting rulebook — but the rulebook
is *executable*, so you generate the data by replaying real S&P 500 prices through a simulated
$10M direct-indexing portfolio and logging every (lot, day) state.

```
download → simulate → mlnet-all → report → submission
(prices)   (lots.csv)  (5 models)  (notebook+HTML)  (course zip)
```

- **v0.1** = supervised baseline: oracle hard labels + soft labels + ~15 features + models with CV. First built in Python, then **rewritten in ML.NET** (issue #10 — "I just can't w python") because a typed language lets leakage and schema be *type errors*, not conventions.
- **v0.2** = the PSTAT 231 final submission: five models with **champion selection** (only top-2 touch the test set, enforced by code shape), PCA/K-means, and a **report layer** that executes a rubric-mapped notebook → HTML. **Headline finding: GBT champions the real (soft) target at 0.86 PR-AUC; the linear tier hits a representational ceiling (~0.65); trees recover the known oracle near-perfectly — a recoverability sanity check.**
- **Current HEAD ≠ the frozen submission.** After submitting, data was shifted from an arbitrary window from **2 years (170,751 rows) → up to ~20 years (1,846,016 rows)** (commit `c421c4f`, dynamic ranges). This *changed one of the headline findings* (see §6) and is the empirical trigger for the v0.3 G_YTD redesign. **This is the thing most worth re-anchoring on.** (dynamic-range dataset selection in the download section)

**Should it be redone? No.** The architecture is the asset — extend along the seams it was designed with (§7). A rewrite would throw away the leakage discipline and simulation physics that are the hard, load-bearing parts. Details and the recommended critical path in §8–§9.

---

## 1. The thesis, from first principles

Derive the whole codebase from the tax code:

1. **Tax asymmetry (why direct indexing exists).** Realized losses offset realized gains. A lot trading below basis holds an *option* worth `τ(h)·|qₖ(Pₜ−pₖ)|` (τ = 37%/20% short/long-term). That option only exists at **lot granularity** — an index-fund holder owns no lots; a direct-indexing holder owns hundreds and can harvest individually while still tracking the index.
2. **The decision has a correct answer (the oracle).** Harvesting is constrained (loss threshold, tracking-error budget, gains budget, wash-sale lockout), so "harvest lot k today?" is computable from observable state:

   `f*(x) = 𝟙[L ≤ −θ₁] · 𝟙[σ_TE ≤ θ₂] · 𝟙[G_YTD > 0] · 𝟙[W ≥ 30]`,  θ₁=2%, θ₂=5%, 30-day wash window.

   Geometrically: the intersection of four half-spaces — a convex polytope Ω, and `f* = 𝟙_Ω`.
3. **Why simulate.** No dataset records this. The rulebook is executable ⇒ **generate** the data: replay real prices through a portfolio obeying the rule, log every (lot, day). The simulator *is* the problem's physics, with real price risk — not an approximation.
4. **Why ML when f\* is known code.** Three reasons, increasing in importance:
   - *Recoverability sanity check* — if a model can't relearn a deterministic `f*` from 170K labels, the pipeline is broken. (Observed: trees ≈ perfect, linear models partial.)
   - *The real target is the future* — the valuable question isn't "does the rule fire today?" (just run it) but "**will it fire within 30 trading days?**" — harvest *propensity*, which depends on unrealized future prices and can only be *estimated* from current state. That's supervised learning.
   - *Distillation toward RL (v0.4)* — the soft-label model `η̂(x) ≈ E[future harvestability | x]` compresses a 30-day forward simulation into one function eval: a learned value-function surrogate, the stepping stone to the RL policy layer.
5. **The single most important invariant — the information asymmetry:**

   > **Labels may peek at the future (at dataset-construction time). Features never may.**

   `Y_Soft_BT` is computed from prices at t+1…t+30 (legal — labels are the answer key, exist only at training time). Every feature is computable from information available at day t. At deployment the future doesn't exist; the model bridges the gap. **Do not break this in v0.3+** (§7).

---

## 2. What the data point *is* (the schema — `LotStateVector`)

One row of `lots.csv` = one immutable "photograph" of (lot state, portfolio state, asset state) at time t. `LotStateVector` has fan-in **20 types / fan-out 0** — empirically the load-bearing schema of the whole codebase.

| Group | Features |
|---|---|
| **Lot-level** | `L` unrealized return ·  `H` holding days · `S` short/long flag · `B` cost basis · `W` lot weight · `K` # open lots same ticker |
| **Portfolio-level (shared state 𝒮ₜ)** | `G_YTD` net realized gain YTD · `Sigma_TE` tracking error · `WashClock` days since last harvest (999 = never) |
| **Asset-level (price series)** | `R_t` daily return · `SigmaRange` range-vol proxy · `DeltaMA50` · `DeltaMA200` |
| **Derived** | `TaxAlpha = τ(h)·\|G_lot\|·𝟙[G_YTD>0]` · `DaysToYE` |
| **Labels** | `Y_Oracle ∈ {0,1}` (hard) · `Y_Soft_GBM ∈ [0,1]` (200 GBM paths) · `Y_Soft_BT ∈ [0,1]` (real-path 30-day, NaN near window end) |
| **Metadata (drop before modeling)** | `Symbol`, `Sector`, `Timestep` |

The two soft labels answer the same question by different mechanisms — *holding portfolio state frozen, would the oracle fire in the next 30 days?* `Y_Soft_BT` averages the predicate along the **one real path** (`∈ {0, 1/30, …, 1}`); `Y_Soft_GBM` uses **first-passage over 200 simulated paths** (each counts once). No optimization, no ML, no extrema search inside the labeler — it evaluates a fixed predicate forward and averages.

---

## 3. The architecture as built (this is the README, expanded)

Four layers, each a **map between files on disk** — layers communicate only through serialized artifacts, so any layer can be rerun, replaced, or audited in isolation. That file-boundary design *is* the architectural thesis.

```
download : ∅ → 𝒥                 (raw price cache: data/raw/*.json + constituents.json)
simulate : 𝒥 → (𝒳×𝒴)ᴺ            (data/lots.csv — the manufactured dataset)
mlnet    : (𝒳×𝒴)ᴺ → Ĥ            (data/artifacts-mlnet/ — leaderboards, metrics, coeffs, model zips)
report   : Ĥ → paper             (src/Export/report/ — executed notebook + HTML + codebook)
```

| Command | Layer | What it does |
|---|---|---|
| `dotnet run download` | 1 — data | SSGA SPY holdings (issue #7, FMP constituent endpoint retired) + FMP EOD prices → `data/raw/`, `constituents.json`. Now **dynamic range** (issue #20, commit `c421c4f`) instead of fixed 2y. |
| `dotnet run simulate` | 2 — simulation | Backtest the $10M portfolio, label every lot-day → `data/lots.csv` |
| `dotnet run simulate-mc` | 2 — alt | Monte-Carlo (synthetic GBM prices) variant → `data/lots-mc.csv` |
| `dotnet run mlnet-all` | 3 — ML | CV all 5 models × 2 targets, test the champions, render → `data/artifacts-mlnet/` |
| `dotnet run report` | 4 — report | Codebook (header-drift assert vs lots.csv) + execute `final_report.ipynb` + export HTML |
| `dotnet run report-all` | 3+4 | train + report in one command |
| `dotnet run submission` | packaging | assemble `submission.zip` (raw price cache included → reproducible with no FMP key) |
| `dotnet run deps` | devtools | regex scan `src/**/*.cs` → dependency/coupling atlas (mermaid) |
| `dotnet run test` | tests | 30+ assertions: state machine, oracle gates, TE invariants, GBM stats, splits/imputation/weights/grid-search + **leakage regression tests** |

**Source map (what lives where):**

- **Layer 1** `src/DataCollection/` — `MarketDataDownloader.cs`, `Models.cs`
- **Layer 2** `src/Core/` — `Portfolio/{Lot, PortfolioState, LotStateVector}.cs`, `Oracle/OracleBoundary.cs`, `Simulation/{PriceLoader, SimulationEngine, SoftLabelBuilder, TrackingErrorProxy, GbmSimulator, MonteCarloEngine}.cs`, `Export/SimulationExporter.cs`
- **Layer 3** `src/ML/CSharp/MLNet/` — `MLnetPipeline.cs`, `Models/{Logistic, ElasticNet, RandomForest, GradientBoostedTrees, LinearRegression, Pca, KMeans}Trainer.cs`, `Splits/{StratifiedSplit, StratifiedKFold}.cs`, `Preprocessing/{MedianImputer, ClassWeights, …}.cs`, `Tuning/{GridSearchCV, CvResult}.cs`, `Metrics/{BinaryMetrics, SilhouetteScore}.cs`
- **Layer 4** `src/ML/Python/` (managed by `uv`, invoked via the `PythonRunner` subprocess seam) — `scripts/{report, codebook, codebook_schema, eda, render, package_submission, dependencies}.py`, `notebooks/final_report.ipynb`. **Python is confined to rendering** — the .NET side stays the single source of truth for everything the models touch.

---

## 4. What v0.1 and v0.2 delivered, layer by layer (and why)

**Layer 1 — Download.** Window = N years + 200 trading-day warmup so `MA_200` (→ feature `DeltaMA200`) is defined on the *first* portfolio day. A feature's data requirement propagates backward into the fetch contract — the first instance of schema-first design. Switched constituent source to the SSGA SPY holdings xlsx when FMP's endpoint retired (issue #7); incremental re-aggregation (issue #3).

**Layer 2 — Simulation (the heart).**
- `PriceLoader` aligns ragged per-ticker JSON onto one shared trading calendar → a dense matrix `P ∈ (ℝ>0 ∪ NaN)^{503×T}`; precomputes returns, range-vol, MA50/200. Read-only after `Load`. `CreateForTesting` injects synthetic returns with the I/O deleted (pure-math unit tests).
- `SimulationEngine` day loop: value portfolio → compute `σ_TE` **before** the lot loop (it's a *decision input*, gate 2, not a consequence) → for each open lot, **snapshot first** (records decision-time state), **then** harvest if `f*=1` (realize ΔG<0, zero the wash clock, queue a t+30 reopen at fresh basis) → drain reopen queue → AdvanceDay → year-end reset.
- `OracleBoundary.Label` is a **pure, stateless** function — referentially transparent — so the engine, the soft-labeler, and tests share it with zero coupling.
- `TrackingErrorProxy`: forward-looking `σ̂_TE = √(δwᵀ Σ̂ δw · 252)` where δw is the active-weight deviation from equal-weight (every wash-locked ticker contributes −1/N) and Σ̂ is the daily-return covariance built once at construction. **War story:** v0.0 computed portfolio returns from total dollar value, so every harvest looked like a −1/503 crash, inflating median TE to 33% and jamming gate 2 permanently shut (zero positives). Rebuilt twice (rolling scalar → quadratic form) to be *structurally invariant* to harvest/reopen. This Σ̂ is the plug-in point for RMT cleaning (issue #5).
- `SoftLabelBuilder` second pass: replaces each frozen snapshot with a `with`-copy carrying the two soft labels (deterministic per (Symbol, Timestep) seed, safe under `Parallel.For`).
- **Endogenous dataset size:** a lot emits rows only while open ⇒ harvested capital is invisible for exactly 30 days. So `N_rows = n₀·T − Σ min(30, T_end−t_h) − ε` — *more drawdowns ⇒ more harvests ⇒ fewer rows*. (At the v0.2 freeze: 251,500 − 80,426 − 323 = **170,751**, reconciles exactly.)

**Layer 3 — ML.NET.** Two binary targets: `y_oracle` (base rate 1.6%, the sanity check) and `y_soft = 𝟙[Y_Soft_BT>0]` (base rate 19.9%, the real problem). Five model families, each `RunCV()` first (stratified 5-fold, **scored by PR-AUC** — with 1.6% prevalence, ROC-AUC is blind to the false-positive flood). **Champion selection**: CV ranks all five, only the top-2 classifiers ever reach `RunSupervisedModel` → the held-out test set. The "only the best one or two touch test" rubric requirement is *implemented as code shape*, not convention. LinReg always runs as the **deliberate failure exhibit** (least squares doesn't know [0,1] exists — ~10% of predictions escape the unit interval). **Leakage invariants are types**: `MedianImputer.Fit` and `ClassWeights.AttachBalancedWeights` admit *no overload* taking the full dataset — only a train fold. The accident sklearn gets right implicitly, C# makes unrepresentable (`MLNetLeakageAudit.md`).

**Layer 4 — Report.** `dotnet run report` preflights artifacts (exit 2 + actionable list if missing), generates the codebook from a schema dict with a **header-drift assert against lots.csv**, executes the notebook via nbclient, exports self-contained HTML. **Hybrid data flow mirrors the layer boundary:** EDA statistics recompute live from `lots.csv` in notebook cells (rubric-visible code); model sections read the C# artifacts (so the report can never silently disagree with the pipeline that trained the models).

### Headline empirical results — the *frozen v0.2 submission* (2-year, 170,751 rows)

| | soft target (base rate 19.9%) | oracle target (sanity check) |
|---|---|---|
| **GBT ★** | CV **0.858** → test **0.862** PR-AUC | 0.9997 |
| **RF ★** | 0.694 | 0.9987 |
| Logistic | 0.650 | **0.987** ← see note |
| Elastic net | 0.635 | 0.442 |
| LinReg (demo) | 0.559 (≈10% preds outside [0,1]) | 0.372 |

Three results that *are* the project's science:
1. **GBT's CV→test gap is +0.004** — the honesty signature; nothing overfit to folds. At the F1-optimal threshold (0.685): 76% precision, 83% recall against a 19.9% base rate.
2. **The GBT–RF gap (0.16)** — same trees, different composition: boosting *sharpens* the thin positive region that bagging *blurs*.
3. **The linear tier (~0.64–0.65) is representation-limited, not regularization-limited** — every linear model chose its *weakest* penalty; capacity, not overfitting, is the ceiling.
4. **Logistic hits 0.987 on the oracle** because with the $1M gains seed the `G_YTD > 0` gate never closed (100% of rows), so the 4-way AND nearly collapsed to the single half-space `L ≤ −2%` — which *is* linearly separable. **A conjunction is only as non-linear as the number of gates that actually bind.** ← This finding *changed* on the 20-year run; see §6.

---

## 5. DataMemo (the pre-plan) vs. v0.2 (what shipped)

The original `DataMemo/DataMemo.ipynb` was your pre-implementation conceptual plan (professor-required). Here's how the final project diverged from it — what survived, what changed, and what the professor pushed back on.

### What the DataMemo got right and *survived intact*
- **The core problem framing** — lot-level harvest decision as a **binary classification** `y ∈ {0,1}`, treating each time-dependent lot as i.i.d. via an oracle rule (so no true time-series model needed). This is exactly what shipped. Your professor explicitly blessed the rule-based-label workaround.
- **The oracle/boundary intuition** — the DataMemo described an "oracle machine the ML roughly follows," with the model learning *within* the boundary's interior (the Stokes-theorem analogy: interior info not conserved to the boundary). That became `OracleBoundary` + the polytope Ω + the recoverability sanity check verbatim.
- **The informative predictors** you guessed — unrealized loss %, holding period, realized gains YTD, tracking error — are *literally* `L, H, G_YTD, Sigma_TE` in the schema.
- **Separation of concerns** — PCA/K-means dimensionality reduction as a stage *separate from and before* the i.i.d. classification. Shipped as `RunUnsupervised` (though see the caveat below).
- **The .NET-dominant, Python-for-write-up stack** — you predicted "C# for portfolio logic + Python as the functional/ML write-up layer." That's *exactly* the final shape (Python confined to the report layer behind `PythonRunner`).

### What the professor pushed back on — and how it actually resolved
The professor approved the memo but said: *"most of what you've written is a software-engineering project with a thin layer of ML… you do not need a custom DBMS, an MVC app, a multi-language pipeline, a full direct-indexing engine, or real-time API integration. Pull a fixed dataset once, fit multiple classifiers, evaluate with proper CV."*

How v0.2 resolved each:
- **Custom DBMS / MVC app / real-time API** — *dropped*, as advised. No DBMS, no app, no live API; `download` pulls a fixed cache once.
- **"Multi-language pipeline"** — *kept, but disciplined.* You didn't simplify to single-language; you kept C#+Python but **confined Python to rendering** behind one subprocess seam. The justification you wrote in your rebuttal (ML.NET ≈ Python ML with SWE scalability) held up.
- **"Full direct-indexing engine"** — *this is the one place you went bigger, not smaller.* The professor said you don't need a simulator; you built one anyway — because (correctly) **there is no fixed dataset to "pull once"** for this problem. The simulator *is* how the tabular dataset gets manufactured. This was the right call and the project's most distinctive contribution, but it's worth naming honestly: v0.2 is *more* engineering-heavy than the professor recommended, deliberately.
- **"Oracle approximation limits how interesting results are"** — the professor's sharpest point. He suggested a richer multi-condition rule or an ML-vs-baseline comparison. You did **both**: the oracle is a genuine 4-gate conjunction, *and* the soft-label target (`Y_Soft_BT`) reframes the problem from "reproduce a rule" to "predict whether the rule fires in the next 30 days" — which is no longer a rule you can just run, it requires the future. That directly answers his critique.

### What materially *changed or grew* from the pre-plan
| DataMemo idea | What shipped |
|---|---|
| "~10–20 predictors, 200 stocks × weekly × 5yr ≈ 52,000 obs" | ~15 features, **daily** (not weekly), 503 tickers; 170,751 rows (v0.2) → 1.85M rows (current 20-year) |
| Single hard binary label | **Two label families** — hard `Y_Oracle` *and* soft `Y_Soft_BT/GBM ∈ [0,1]`. The soft labels are the genuine intellectual addition that didn't exist in the pre-plan. |
| "missing data handled by DBMS / imputation, maybe time-series" | NaN-in-place alignment + median imputation **fit on train fold only**; missingness is *structural* (`Y_Soft_BT = NaN` when <30 days remain), a modeled feature not a defect |
| PCA "to select a subset of stocks that still tracks the index" | PCA/K-means shipped, **but as feature-space diagnostics, not yet portfolio construction.** The original "shave multicollinear stocks to reduce the tracked universe" goal is deferred to the portfolio layer (issue #6) — it needs the simulation/RL environment as a prerequisite. *This is the biggest gap between intent and delivery.* |
| TaxAlpha as a vague goal | a concrete *feature* `TaxAlpha = τ(h)·\|G_lot\|·𝟙[G_YTD>0]`, with portfolio-level tax-alpha *metrics* deferred to issue #12 |
| Python ↔ C# "functional calling layer" | resolved into a clean one-way **file-boundary** architecture, not function-call interop |

**Bottom line on the comparison:** the *conceptual core* of the DataMemo survived almost perfectly — you predicted the problem, the labels' informative predictors, the classification framing, and the stack. What changed is (a) you committed to **manufacturing** the dataset via a real simulator rather than "pulling one once" (going against the professor's simplification, correctly), (b) you added the **soft-label propensity reframing** that elevates the project above "reproduce a rule," and (c) the **PCA-for-portfolio-construction** idea got demoted to a diagnostic and deferred. The pre-plan's graduate-level ambition (measure-theoretic lot atoms, oracle boundary) is *more* realized in v0.2 than the professor's scoped-down version would have allowed.

---

## 6. ⚠️ Current repo state ≠ the frozen v0.2 submission (the thing you're forgetting)

This is the single most important re-anchor. **After** the PSTAT 231 submission (PR #11, `Lifecycle_v02.md`, and the report all describe the **2-year / 170,751-row** run), you made commit **`c421c4f` — "dynamic data-collection and simulation ranges"** and re-simulated over **~20 years (2004–2024)**. The repo *right now* has:

- `data/lots.csv` = **1,846,016 rows** (≈1.85M), dated June 20 — not 170,751.
- Current `soft_bt_cv_leaderboard.json`: **GBT mean CV PR-AUC ≈ 0.850** (still champion, still strong — the soft-target finding is robust across regimes).
- **One headline finding flipped.** On the 2-year run, logistic nearly solved the oracle (CV 0.987) because the `G_YTD > 0` gate never bound. On the 20-year run, `G_YTD > 0` holds on ~94% of rows but *does* bind across regimes, and **logistic on the oracle target collapses to CV ≈ 0.12** (per `GYTD_Redesign_Plan.md`). The conjunction stops degenerating to one half-space once real bull/bear regimes make multiple gates bind. The geometry claim from the report is now *window-dependent* — which is itself a richer result.

**Implication:** the numbers in the README/report/PR #11/`Lifecycle_v02.md` are the *PSTAT submission snapshot*; the live pipeline has moved on. Before quoting a number to anyone, know which window you mean. The 20-year switch already partially closes issues #19 (historical scope) and #20 (dynamic range) — but the report narrative hasn't been re-run against it (issue #18 is exactly "construct interpretations from the CV results," now doubly relevant because the results moved).

---

## 7. The full roadmap / issue ledger (so nothing gets forgotten)

Every open issue mapped to where it belongs. **PRs:** #1 (core domain) and #2 (simulation) **merged**; #11 (v0.2 ML.NET) and #8 (Python ML layer) **open**; #13/#14 (derivations) closed; #10/#7/#3 closed.

| Version | Theme | Issues / artifacts | Status |
|---|---|---|---|
| **v0.1** | supervised baseline | hard+soft labels, ~15 feats, models+CV (Python → ML.NET) | ✅ done |
| **v0.2** | champion selection + report | PR #11; 5 models, PCA/K-means, report layer, submission zip | ✅ submitted |
| **v0.2 cleanup** | interpret results | **#18** (interpretations from CV results — now needs re-run on 20yr), **#4** (formally define models — mostly done in derivations) | 🔄 open, low-effort |
| **already in progress** | data scope | **#19** (10+ yr history), **#20** (dynamic sim range) — *partially shipped in `c421c4f`* | 🔄 |
| **v0.3 (summer)** | volatility sub-model + tax ledger | **GYTD redesign** (`GYTD_Redesign_Plan.md`, Options B→C), GARCH/EWMA σ̂, Ledoit-Wolf shrinkage, **#17** (richer soft-label families), **#5** (RMT covariance cleaning) | ⬜ next |
| **v0.3→0.4 bridge** | economic evaluation | **#12** (portfolio tax-alpha metrics, opportunity cost), **#22** (custom deployment/portfolio score metrics beyond ROC/PR) | ⬜ prereq for RL reward |
| **v0.4 (next year)** | RL policy layer | **#15** (split SimulationEngine: training-sim vs RL portfolio-sim, `Oracle/Classifier/RL` policy interface), **#6** (unsupervised dim-reduction for wash-sale replacement / TE-min position selection), PPO/SAC agent | ⬜ |
| **v0.5–1.0** | distillation, scale, deploy | **#16** (knowledge distillation: big NN → soft labels → small model / RL warm-start), live data, client-parameterized policies, RIA-style SaaS | ⬜ long-horizon |
| **infra (any time)** | reproducibility | **#21** (Nix env for 1:1 dependency replication) | ⬜ |

### Load-bearing invariants — *do not break these in v0.3+*
When you extend the pipeline, these are the things that, if violated, silently corrupt the science:
1. **Labels may peek at the future; features never may** (§1.5). Any new soft label (#17) or tax-ledger feature (GYTD plan) must keep every *feature* σ(𝓕ₜ)-measurable.
2. **Leakage invariants are types** — new preprocessing must fit on the **train fold only**, with no full-dataset overload (`MedianImputer.Fit` pattern).
3. **File-boundary layers** — layers talk only through on-disk artifacts; don't introduce hidden shared mutable state across the layer edge.
4. **Oracle stays pure/stateless** — it's shared by engine, labeler, and tests; if it gains state, the soft labels become incomputable as a second pass.
5. **Snapshot precedes harvest** — rows record decision-time state, not post-harvest state.
6. **PR-AUC, not ROC-AUC, for rare positives** — and now (issue #12/#22) PR-AUC is *not* the deployment objective either: a higher-PR-AUC model can produce *less* tax alpha if it fires on correlated lots simultaneously (spiking σ_TE) or harvests too early in the year.

---

## 8. Should the project be re-done? — No. Here's why, and where to extend instead

**Verdict: do not rewrite. Extend along the seams the architecture was explicitly designed with.** The reasons:

- The **hard, load-bearing parts are done and correct**: the simulation physics (with the TE estimator rebuilt twice to be harvest-invariant), the typed leakage discipline, the schema-first contract, the endogenous-dataset identity that reconciles exactly. A rewrite re-pays all of that cost to gain nothing.
- The architecture *already anticipates* v0.3 and v0.4. The GYTD plan shows the supervised `η̂(x)` becomes the **RL value-function warm-start**; issue #15's policy interface (`Oracle/Classifier/RL`) is a clean extension of `SimulationEngine`, not a replacement; the TE estimator's Σ̂ is the documented plug-in point for RMT (#5). These are seams, not rewrites.
- The one *structural* change worth doing — splitting `SimulationEngine` into a **training-simulation** mode and an **RL portfolio-simulation engine** (#15) — is an extension with a shared core, not a redo.

The only thing that *would* justify a partial rebuild is the **PCA-for-portfolio-construction** gap (§5): the DataMemo's original "use unsupervised methods to shave multicollinear stocks and reduce the tracked universe" was demoted to a diagnostic. Realizing it (issue #6) requires the portfolio-construction/RL layer first — so it's correctly *deferred*, not *abandoned*.

---

## 9. Recommended path through v0.3 (summer) and v0.4 (next year)

A concrete critical path with dependencies called out. The ordering is chosen so each step unblocks the next and so the *economic* evaluation exists before you design an RL reward.

### Summer → v0.3 (do roughly in this order)

1. **Tie off v0.2 first (cheap, high value): issue #18 on the 20-year run.** The window already changed (§6); re-run the report narrative against the 1.85M-row data and write up *why* the logistic-oracle finding flipped (regime-dependent gate binding). This converts the accidental `c421c4f` change into a deliberate result and resolves the "I forget what's implemented" drift at the source. *Also resolves #19/#20 framing.*
2. **G_YTD redesign — ship Option B, target Option C** (`GYTD_Redesign_Plan.md`). Interim **B**: replace the constant $1M seed with `seed_t = κ·V_t` recomputed at year-end (one-line defect fix; gate starts binding again). Target **C** — the real v0.3 work: promote `G_YTD` from a binary gate to a **continuous tax ledger** (`realized_gains_YTD`, `loss_carryforward` across year-end, `$3k` ordinary-offset budget), add a "sell-winner" gains-realization transition to the simulator, and emit a **continuous `Y_TaxValue` regression target**. This simultaneously *fixes the economics* (carryforward + $3k offsetting, which the strict gate currently refuses — real tax alpha left on the table) and *resolves the "removing G_YTD makes it less interesting" worry by making the target continuous*. It's also the natural bridge to the RL reward.
3. **The volatility sub-model (the README's named v0.3 deliverable).** Replace constant-σ GBM in `SoftLabelBuilder.ComputeGBM` with **GARCH/EWMA** time-varying σ̂; this is the "volatility as a supervised sub-model with its own σ̂_t ∈ ℝ target" idea from the README/issues. Deliverable question: *does a better σ̂ improve harvest-urgency scoring?*
4. **Richer soft-label families (issue #17)** — multi-horizon (`y_30/y_90/y_180`), persistence-weighted (`y_persist = (1/30)Σ f*(x_{t+s})`), tax-alpha-weighted (`y_alpha`, `y_max`). Each gets its own PR/ROC leaderboard. **Guard:** keep this information out of the feature space (the i.i.d.-per-lot assumption).
5. **(Optional, fits the vol theme) RMT covariance cleaning (issue #5)** — Marchenko–Pastur eigenvalue cleaning of Σ̂ in `TrackingErrorProxy`, and **Ledoit-Wolf shrinkage** (README v0.3 item). Plug-in point already exists.

### Bridge work (do *before* RL): economic evaluation — issues #12 + #22
Build portfolio-level metrics — **total tax alpha** `α = Σ(P_cost − P_harvest)·shares·τ`, opportunity cost, performance vs the mechanistic oracle — via a backtest that runs the trained model's scores as harvest decisions on held-out years. **This is a hard prerequisite for v0.4**: the RL reward *is* "realized after-tax alpha minus a tracking-error penalty," so you need this measurement layer before you can train or even evaluate a policy. It also retires the "PR-AUC ≠ tax alpha" caveat with real numbers.

### Next year → v0.4 (RL policy layer)
1. **Split the simulator (issue #15)** into `trainingsimulation` and an agentic `portfolio_simulation_engine` with a **policy interface**: `OraclePolicy` (generates lots.csv, today's behavior), `ClassifierPolicy` (backtest/eval mode), `RLPolicy` (training mode with an unsupervised TE minimizer).
2. **Train a PPO/SAC agent** on that environment. Reward = continuous tax ledger (v0.3 Option C) − TE penalty; **warm-start the value function from the supervised `η̂`** (the distillation bridge). Deliverable question: *how much alpha does the RL policy recover over the supervised baseline — i.e., how much of the oracle gap is irreducible?*
3. **Wire in the unsupervised portfolio layer (issue #6)** — PCA/K-means on the *return covariance* (not feature space) to pick TE-minimizing wash-sale replacements: when a lot is harvested, either buy a (semi-)collinear substitute immediately or rebuy the same asset after 30 days. This finally realizes the DataMemo's original dimensionality-reduction-for-portfolio-construction intent and makes "different ML methods solve different subproblems" concrete.
4. **(v0.5+) Knowledge distillation (issue #16)** — a large NN labels `soft_bt`-style targets to train smaller supervised models / warm-start the RL agent. (Your flagged PhD-interest area.)

### One-line summary of the recommendation
> Don't rewrite. **Summer:** close out #18 on the 20-year data, then make `G_YTD` a continuous tax ledger (GYTD Option C) and add the volatility sub-model + richer soft labels — *all of which raise the difficulty/interest of the supervised target rather than removing structure*. **Build the tax-alpha metric layer (#12/#22) as the bridge.** **Next year:** split the simulator into a policy-driven RL environment whose reward is that tax ledger minus a TE penalty, warm-started from the v0.3 supervised model. Every step reuses the existing seams; nothing needs to be thrown away.

---

*Cross-references: `DataMemo/Lifecycle_v02.md` (§ numbering used above), `DataMemo/GYTD_Redesign_Plan.md` (the v0.3 design in full), `DataMemo/SimulationMath.md`, `DataMemo/PortfolioMath.md`, `DataMemo/MLNetLeakageAudit.md`. Frozen-submission numbers: PR #11. Original pre-plan: `DataMemo/DataMemo.ipynb`.*

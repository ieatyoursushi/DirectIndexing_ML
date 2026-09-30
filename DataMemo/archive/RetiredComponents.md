# Retired Components — the single archive of what was removed, and what it taught

> **Status: ARCHIVE (frozen).** This is the one place the findings of retired components live.
> The *machinery* was deleted in the pre-v0.3 downsizing; the *findings* are kept here.
> Every retired file is recoverable from the annotated tag **`archive/v0.3-pre-downsize`**
> (= commit `03f0c3e`, the last commit before the downsizing):
>
> ```bash
> git checkout archive/v0.3-pre-downsize -- <path>      # restore one file
> git worktree add ../di-archive archive/v0.3-pre-downsize   # or the whole pre-downsize tree
> ```

## Why the downsizing happened (the selection criterion)

A component stays only if it is **reachable from the forward objective** — something on the
v0.3 → v0.4a → v0.4b path consumes it (an RL input, a baseline/control, or an invariant
enforcer). A component that is not reachable keeps its **finding** here and loses its
**machinery**. Three concrete pressures made this worth doing before v0.3 feature work:

1. **Compute.** `--contrib` grows the dataset 1.85M → 5.18M rows; the five-model grid was
   36 configs × 5 folds × 2 targets = 360 fits. GBT + logistic is 12 configs.
2. **The code the RL environment wraps** (`SimulationEngine`, `OracleBoundary`,
   `PortfolioState`, `LotStateVector`) carried the legacy gated arm and a divergent twin engine.
3. **Cognitive load.** Every retained component is one more object whose mathematics must stay
   in sync with its implementation (see `DataMemo/spec/SymbolTable.md`).

What was **kept** in the supervised layer, and why: **GBT** (`FastTree`) — champion, RL state
input, fitted-Q / value-function substrate; **logistic** (`LbfgsLogisticRegression`) — the
linear control that measures how much non-linearity exists, and the calibrated differentiable
score (its RL analogue is a linear-function-approximation baseline); **tax-value regression**
(SDCA linear vs FastTree) — the FastTree ĝ is the v0.4 value-function warm start.

Number provenance: "v0.2" = 2-year window, 170,751 rows, d = 15, gated oracle (PSTAT 231
submission). "v0.25" = 20-year, d = 17, CV PR-AUC (`GYTD_Redesign_Plan.md` §6.1). "v0.26" =
20-year scalarized, stratified-random vs temporal-purged 80/20 CV PR-AUC
(`ValidationHardening_v026.md` §3). Quoted, not re-derived.

---

## 1. Random forest — `RandomForestTrainer.cs` (`FastForest`)

| measurement | oracle | soft |
|---|---|---|
| v0.2 test PR-AUC (2y) | 0.9987 | 0.694 |
| v0.25 CV, gated arm | 0.9710 | 0.6713 |
| v0.25 CV, scalarized | 0.9883 | 0.6332 |
| v0.26 CV, temporal | 0.9252 | 0.3911 |

**Finding.** Same base learner as GBT, different composition: *boosting sharpens the thin
positive region that bagging blurs*. The GBT–RF soft-target gap was 0.16 (2y) and 0.18
(20y scalarized) — a bias-dominated regime, where sequential residual fitting beats parallel
variance reduction. FastForest is uncalibrated (emits `Score`, no `Probability`), which is why
`BinaryMetrics` had a `Score` fallback. Its grid (T × J × κ = 12 configs) was the most
expensive in the pipeline.
**Why retired.** Never beat GBT on any target or split; its lesson is recorded.
**Revive if** a variance-reduction ensemble is needed as a regime-robust baseline.

## 2. Elastic-net logistic — `ElasticNetTrainer.cs` (`SdcaLogisticRegression`)

| measurement | oracle | soft |
|---|---|---|
| v0.2 test PR-AUC (2y) | 0.442 | 0.635 |
| v0.25 CV, gated arm | 0.0892 | — |
| v0.25 CV, scalarized | 0.8107 | 0.5874 |
| v0.26 CV, temporal | 0.8694 | 0.3663 |

**Finding.** Every linear model chose its **weakest** penalty (λ₁ = λ₂ = 0.001 at the grid
edge): the linear tier is *representation-limited, not regularization-limited*. The
gated → scalarized leap (0.09 → 0.81 on the oracle target) is the cleanest single piece of
evidence that the v0.2 box corner — not linear capacity — was destroying linear models.
**Why retired.** Same hypothesis class as L2 logistic, dominated by it on every target; the
sparsity story never materialized.

## 3. Linear-regression demonstrator — `LinearRegressionTrainer.cs`

| measurement | oracle | soft |
|---|---|---|
| v0.2 test PR-AUC (2y) | 0.372 | 0.559 (ROC-AUC 0.842) |
| v0.25 CV, gated arm | 0.1195 | — |
| v0.25 CV, scalarized | 0.8227 | — |
| v0.26 CV, temporal | 0.8942 | 0.2775 |

**Finding.** The deliberate negative control: an unbounded affine map fit to a {0,1} target has
codomain ℝ, so predictions escape [0,1]; and a Bernoulli outcome is heteroskedastic
(Var(Y|x) = η(1−η)), so OLS is not BLUE for this DGP. The ROC 0.842 vs PR 0.559 pair is the
standing illustration of why ROC-AUC alone misleads on rare positives.
**Recorded drift (unresolved, not re-measured).** The escape fraction is quoted as "~10%"
(README, `PSTAT231_RECAP.md`) and "≈24%" (`MLDerivations.md` §4.5); the synthetic unit-test
fixture measured 26.3%. The artifacts at the tag are the arbiter.
**Why retired.** Pedagogical; the point is made. The "linear cannot represent kinks" control
survives in the linear half of the tax-value regression (R² ≈ 0.10 vs trees ≈ 0.9).

## 4. Feature-space PCA — `PcaPipeline.cs`

**What it was.** Eigendecomposition of the standardized 17-feature covariance
C ∈ ℝ^{17×17} (the **lot-state** space 𝒳), keeping the smallest r with ≥ 95% cumulative
explained variance.
**Finding.** 𝒳 is genuinely high-dimensional: the first component explains a modest share and
most components are needed to clear 95%; near-constant coordinates (`K`; `LossCarryforward`
in gated runs) contribute ~zero eigenvalues by construction. No low-dimensional summary of lot
state exists to exploit.
**Successor role — a *different* mathematical object (issue #6).** PCA moves from the feature
space 𝒳 ⊂ ℝ¹⁷ to the **return covariance** Σ̂_t ∈ ℝ^{N×N} (asset co-movement), where it has
two jobs:

1. **Shrink the set of names/lots the agent harvests over** — the core+reserve universe:
   shave off multicollinear, similarly-clustered names while preserving the index's beta and
   factor exposures, with the reduced-away names forming the substitute pool for wash-sale
   replacement (the `PortfolioState.HarvestLot` comment and issue #6's analogy). Roadmap:
   v0.45 (after the RL layer).
2. **Supply the agent's factor summaries** — the projection U_kᵀ δw of the active-weight
   vector onto the top-k eigenvectors of Σ̂_t, which the v0.4b state vector calls for.

Both reuse the eigendecomposition built for point-in-time covariance cleaning in v0.3-4.

## 5. K-means + silhouette — `KMeansPipeline.cs`, `SilhouetteScore.cs`

**What it was.** Per-symbol aggregates of the four asset-level features (g_s ∈ ℝ⁴,
standardized), Lloyd's algorithm, k ∈ {5, 10, 15, 20, 25} chosen by maximum mean silhouette.
**Finding.** Only weak clustering (small best k, low silhouette): even twenty years of visibly
distinct regimes do not separate in raw feature space, because the regimes live in the
*temporal* structure the cross-sectional view discards.
**Successor.** Clustering names in return/factor space for representative selection in the
core+reserve construction (v0.45) — again a different object (loadings on Σ̂_t's eigenvectors,
not technical-indicator aggregates).

## 6. The gated oracle arm and its spectator label

**What it was.** `--oracle=gated`: the v0.2 four-gate conjunction
f*_gated = 𝟙[ℓ ≤ −θ₁]·𝟙[σ_TE ≤ θ₂]·𝟙[G_YTD > 0]·𝟙[𝒲 ≥ 30] (θ₂ = 0.05), with an external-gains
seed (10% of book in `SimulationEngine`, 5% in `MonteCarloEngine` — itself a drift), written
to `lots_gated.csv`; plus the `Y_Oracle_GatedSpec` spectator column (what the gated rule would
say on the scalarized run's rows, via counterfactual legacy-G_YTD bookkeeping).
**Findings** (full table: `DataMemo/decisions/GYTD_Redesign_Plan.md` §6.1):
- The `G_YTD > 0` gate was **economically invalid** for an individual investor (capital losses
  carry forward, 26 USC §1212(b)), open on 94.3% of rows (vestigial) and **self-strangling**
  (the 2008–09 harvest bursts burned the seed to zero and froze harvesting mid-crisis).
- **Manufactured vs real non-linearity:** removing the box corner collapsed the oracle-target
  GBT−logistic gap 0.155 → 0.015, while the soft-target tree advantage (~0.16 gated, ~0.19
  scalarized) is oracle-invariant — the temporal propensity problem is where real
  non-linearity lives.
- **Acting ≠ spectator:** the acting oracle changes the trajectory itself (harvests → wash
  clocks → ledger → which rows exist), so oracle ablations must be separate runs.
- Carryforward is load-bearing under the scalarized oracle ($4.3M over 20y); θ_max binds on 0 rows.
**Why retired.** A deprecated ablation baseline whose result is recorded; carrying a legacy
acting mode through the RL environment's types is pure cost. The v0.26 carry-over "re-validate
the gated arm under temporal splits" is closed as **won't do** (recoverable from the tag).
Schema consequence: `LotStateVector` v3 → **v4** (the spectator column is dropped).

## 7. The course report and submission layer

**What it was.** `report` / `report-all` / `submission` commands; `scripts/report.py`,
`build_report_notebook.py` (the authored-builder → notebook → token-fill → HTML chain),
`fill_report_tokens.py`, `package_submission.py`.
**Kept as historical deliverables (not regenerated):** `src/Export/report/` (the frozen
2-year PSTAT 231 submission) and `src/ML/Python/notebooks/final_report.ipynb` (the 20-year
rewrite). **Kept as live tooling:** the codebook generator and its header-drift assert
(`scripts/codebook.py`, `codebook_schema.py`) — now its own `codebook` command.
**Lesson.** Edit the builder, never the generated notebook; drift must fail loudly (the
codebook assert pattern, now generalized by `scripts/check_math_sync.py`).
**Successor.** The v0.3-6 economic ladder report (after-tax wealth, benefit decomposition,
TE/turnover/cost per rung) replaces the classification-metric narrative.

## 8. `MonteCarloEngine`'s duplicate day loop

**What it was.** A 669-line parallel copy of `SimulationEngine` (day loop, snapshot, harvest,
σ_TE buffer, initialisation) over independent GBM prices, writing `lots-mc.csv`.
**Finding — duplicated simulator logic drifts.** By the pre-v0.3 audit it had diverged in six
ways: no contributions; no reopen wash re-check; year-end every 252 steps instead of the
calendar; approximate `DaysToYE`; `K` hard-coded to 1; a 5% seed vs 10%; and a σ_TE covariance
taken from *real* returns while prices were independent synthetic GBMs (the true Σ of that
world is diagonal). `Y_Soft_BT` was NaN by design.
**Successor.** `PriceLoader.FromGbm(...)` builds a synthetic price world; `simulate-mc` runs
the one canonical `SimulationEngine` over it. One environment, two price sources — the shape
the v0.4b RL environment needs (real history for evaluation, synthetic paths for cheap
episodes and stress regimes).

## 9. Minor CLI removals

`mlnet-rf`, `mlnet-elnet`, `mlnet-linreg`, `mlnet-unsupervised`; `mlnet-supervised` /
`mlnet-baseline` (back-compat duplicates of logistic-only runs); the `train` stub
(`NotImplementedException`); `--oracle=gated`.

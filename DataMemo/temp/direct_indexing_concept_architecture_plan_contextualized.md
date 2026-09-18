# Direct Indexing Concept Architecture Plan — Contextualized Against the Repository

> **Status:** parity assessment + synthesized version planner. Companion to
> `direct_indexing_concept_architecture_plan.md` (the ChatGPT 5.5 Pro inquiry thread) in this
> same `temp/` directory. This document is the repository-aware response: how much the GPT
> thread's architecture matches what is actually built and actually wanted, and the corrected,
> fully-contextualized roadmap that results. Pending review, the planner section (§5) is the
> candidate for promotion to `DataMemo/ArchitecturePlan.md` with a pointer from the README
> Roadmap section — until then, treat the README/RECAP as canonical and this as the working
> expansion.
>
> **Provenance asymmetry, stated up front.** The GPT thread's only *supplied* context was the
> final-report introduction and the post-v0.25 `LotStateVector.cs`. But the inquiry prompts
> themselves (recorded in the companion's appendix) carried substantial project knowledge, and
> the post-v0.25 `LotStateVector` doc comments encode a surprising amount of design (the ledger
> triple, `TaxValue`'s capacity formula, `Y_Utility`, spectator ≠ acting). So the GPT plan is
> best read as *Gabriel's own architectural intuitions, formalized by a model that never saw
> the code, the 20-year EDA, the measured ablations, or the issue ledger*. That predicts
> exactly the failure pattern found below: philosophy nearly perfect, empirics stale, and the
> one finding that only the data could teach — cost-basis aging — completely absent.

---

## 1. TL;DR — the parity verdict

**Overall: high parity on philosophy and decomposition (~85% of the document is either
already built or genuinely the forward direction), with three material blind spots and about
six adoptable additions.**

**Where the thread nails what the project already is or ratified:**

- Its boxed core principle — *"supervised models estimate the world; deterministic
  optimization enforces feasibility; RL decides long-horizon tradeoffs"* — is nearly verbatim
  the identity ratified during the v0.25 redesign (oracle = industry-faithful mechanistic
  engine; `GYTD_Redesign_Plan.md` §8's RL bridge; the estimator-vs-bookkeeping criterion).
  The inquiries and GPT's synthesis are *convergent with the project's actual philosophy*,
  independently derived. That is strong evidence the philosophy is stable and communicable.
- Its one-step utility $U(x) = \mathrm{TaxValue} - \lambda\sigma_{TE}^2 - c_{\mathrm{trade}}$
  is exactly what shipped in PRs [#24](https://github.com/ieatyoursushi/DirectIndexing_ML/pull/24)–[#26](https://github.com/ieatyoursushi/DirectIndexing_ML/pull/26)
  (`OracleConfig.cs`: λ=90,000, θ_max=0.15, c_trade=$10).
- Its label taxonomy (§3.3) describes all six shipped labels correctly, including the
  spectator ≠ acting distinction and the `Y_TaxValue`-excludes-`TaxValue` leakage rule.
- Its "GBT is not a disposable precursor to RL" position matches the durable-baseline framing
  the repo already committed to (and extends naturally to issue #16's distillation plan).

**The three material blind spots:**

1. **Cost-basis aging is absent.** The single most important empirical finding of the
   20-year scale-up — the open-once/hold-forever book ages out of harvestability (oracle rate
   1.6% → 0.20%; one GFC eruption, then near-extinction even through 2020/2022) — appears
   nowhere in the GPT plan. Its Phase B jumps to universe compression while the simulator
   still cannot mint fresh lots. The repo's P0 (contributions/rebalancing/trim) must be
   inserted *before* the GPT's Phase B, not after.
2. **Empirics are stale.** "GBT ≈ 0.8–0.9 PR-AUC" predates the measured v0.25 ablation:
   per-target and per-arm, the numbers are now GBT 0.9956/0.8158 (oracle/soft, scalarized),
   the oracle-target GBT−logistic gap collapsed 0.155 → 0.015 after the redesign, and the
   soft-target gap is oracle-invariant (~0.16–0.19). The "manufactured vs. real
   non-linearity" split (measured in `GYTD_Redesign_Plan.md` §6.1) postdates and sharpens
   everything the thread says about the supervised layer's meaning.
3. **No knowledge of the committed structure.** The thread's Phases A–E don't map to the
   v0.25→v1.0 version spine, the issue ledger (#5, #6, #12, #15, #16, #17, #21, #22, #23),
   or what v0.25 already closed. §5 below does that mapping.

**The six adoptions worth taking (ranked, detailed in §4):** purged/embargoed chronological
validation (→ v0.26, the immediate gate); the six-rung baseline ladder for economic
evaluation (→ #12/#22); the core+reserve universe formalization (→ #6, v0.35); the
eligibility-first `SubScore` replacement module (→ v0.35); **the constrained-optimizer
execution baseline as its own phase (→ v0.4a — the most important structural addition: RL
must beat GBT + optimizer, not just GBT)**; and the turnover-penalty term + guardrails list.

---

## 2. Section-by-section parity scorecard

Verdict legend: **BUILT** (already exists in repo), **ALIGNED** (matches the committed
forward direction), **ALIGNED*** (aligned with corrections), **ADOPT** (new, take it),
**ADAPT** (take with modification), **MISSING** (repo knows something the thread doesn't).


| GPT § | Claim / proposal | Repo reality | Verdict |
|---|---|---|---|
| §1 Executive summary | Hybrid system; supervised layer done; the question is now system-level after-tax improvement | Matches the post-v0.25 identity exactly; the "no longer only classification" reframe is precisely the v0.25→v0.4 arc | **ALIGNED** |
| §2.1 Objective | Max after-tax wealth − λ_TE·TE − λ_turn·turnover − λ_cost·costs − λ_risk·risk | Per-decision form shipped as `U(x)`; TE and trade-cost terms exist (`OracleConfig`). Turnover and risk penalties are **not yet terms** — future U/reward extension. Bounded-alpha realism matches the loss-only-persona floor framing | **ALIGNED*** |
| §2.2 Success metrics + 6 baselines | Economic metrics over PR-AUC; ladder from no-DI → RL | Exactly issues #12 (portfolio metrics) + #22 (deployment metrics), which predate the thread but were never this concrete. The ladder is the missing spec | **ADOPT** |
| §3.1 Supervised layer status | Built; ~0.8–0.9 PR-AUC; don't equate with tax alpha | Built, but numbers superseded by the measured ablation (`GYTD_Redesign_Plan.md` §6.1); the PR-AUC ≠ tax-alpha guardrail is already report prose | **ALIGNED*** |
| §3.2 State representation | Lot/tax/market/derived groups incl. q (shares) | Matches schema v3 (d=17). One correction: `Shares` (q) is **in-memory plumbing, never exported** — it exists for soft-label re-dollarization, not as a feature | **BUILT** |
| §3.3 Labels (all six) | Hard, Soft_BT, Soft_GBM, TaxValue, Utility, spectator; leakage rule; spectator ≠ acting | All shipped (PRs #24–#25); descriptions accurate. The `Y_TaxValue` exclusion rule is enforced in `FeatureLists.NumericFeaturesTaxValueRegression` and measured (linreg R² 0.10 vs FastTree 0.92) | **BUILT** |
| §3.4 GBT's five durable roles | Baseline, ranker, sensor, interpretability, ablation anchor | Matches repo framing; add a sixth the thread missed: **distillation teacher** (issue #16) | **ALIGNED*** |
| §4.1 Prediction / decision / control | Three task types; RL justified only for sequential control | Matches `GYTD_Redesign_Plan.md` §8's structural argument: offset capacity and TE budget are shared depletable resources → lot decisions conditionally dependent → i.i.d. breaks → policy object needed | **ALIGNED** |
| §4.2 Information vs economic limit; non-additive gains | Two ceilings; ablate before assuming additivity | Matches the project's attribution discipline (one boundary-shaping change per PR; measured, not asserted) | **ALIGNED** |
| §5 Nine-layer decomposition | Data→universe→risk→opportunity→screen→policy→replacement→execution→evaluation | Crosswalk: layers 0 (download+`TaxLedger`), 2 (PCA/K-means, diagnostics only), 3 (MLNet), 7 (`SimulationEngine`), and half of 8 (report) exist. Layers 1, 4, 5, 6 and economic-8 **do not exist yet** — they *are* the forward roadmap, slotted in §5 below | **ALIGNED** |
| §6 Universe compression (core+reserve) | 20–30% core targeting the *benchmark*, reserve pool as substitutes; PCA for risk, clustering for representatives; deterministic offline optimizer | Concretizes issue #6 far beyond its current one-line form. The "core targets the benchmark, not the full-DI portfolio" point and the "compression may destroy harvestable dispersion" caution are both right and worth preserving verbatim. Interaction to flag: survivorship/point-in-time membership (report caveat) compounds under compression | **ADOPT** (→ v0.35) |
| §7 Replacement selection | Eligibility-first (compliance ≠ correlation), SubScore, fallback policy | Formalizes intent already recorded in `PortfolioState.HarvestLot`'s doc comment (reopen vs. colinear-substitute routes) + issue #5 (RMT covariance). Currently the simulator only reopens the same ticker after 30d — the substitute path is unbuilt | **ADOPT** (→ v0.35) |
| §8.1 GBT scores held lots only | Candidate set ⊂ holdings ⊂ universe | Correct and clarifying; matches how the engine works today | **ALIGNED** |
| §8.2 Label-environment dependence | Don't train on full-universe labels, deploy in hybrid without testing transfer | The repo **proved the mechanism** the thread hypothesizes: the acting oracle changes the trajectory itself, which is why v0.25 ablations are separate simulate runs, not label columns. Same logic extends to universe changes — the transfer test in v0.35 is mandatory, not optional | **ALIGNED** (repo stronger) |
| §9 RL: optional, high-level, hybrid | Earn-its-complexity rule; low-dim action `a_t = (τ_t, TE budget, turnover budget, m, regime)`; supervised inputs; anti-leakage for fed predictions | Matches #15/v0.4 intent; the **low-dimensional action space is a genuine upgrade** over the implicit per-lot action space in earlier repo notes. The anti-leakage rule for predictions-as-features is new discipline to encode | **ADOPT** (→ v0.4b) |
| §10 Component taxonomy | Empirical / mechanistic / supervised / RL | Identical to the ledger design doc's estimator-vs-bookkeeping criterion, generalized. `TaxLedger` is its §10.2 poster child | **ALIGNED** |
| §11 Phase A | Freeze/document schema; validate under strict time-series methodology; purge/embargo ≥ label horizon | Freeze/document: **done** (codebook + header-drift assert, `MLNetLeakageAudit.md`, byte-identity discipline). The purge/embargo point is **the single most valuable technical addition in the whole thread** — current splits are stratified *random*, and 30-day forward labels overlap across adjacent rows | **ADOPT** (→ v0.26) |
| §11 Phases B–E | Universe → constrained baseline → RL decision → end-to-end comparison | Right order *after* inserting simulator realism (aging fix) ahead of Phase B. Phase C (constrained non-RL execution baseline) is missing from the repo roadmap and is adopted as v0.4a | **ADAPT** (§5) |
| §12 Research questions | Supervised / construction / replacement / RL questions | Merged into §7 register below with repo-specific additions | **ADOPT** |
| §13 Guardrails | 8 non-negotiables | Adopt nearly wholesale. Note #4 (don't train in one environment, deploy in another) is already *operationalized* by the separate-runs ablation design; #5 (no future leakage) is the existing σ(𝓕ₜ) rule + the new purged-split work; #7 (auditable tax module) is `TaxLedger` | **ADOPT** |
| §14–§15 Blueprint + position | Pipeline diagram; development order | Correct with two amendments: insert simulator-realism stage before the basket optimizer; add distillation/deployment tail. The "most defensible development order" boxed at the end is, with those inserts, exactly §5 below | **ADAPT** |
| Appendix | Gabriel's prompts: motivation ($10M→$75M thought experiment, 20%→17% ≡ 15% tax-bill cut), oracle-ceiling worry, vol-model skepticism, GBT-vs-RL, synthesizer framing, compression, replacement, deterministic-vs-trained | This is the project's actual intent, well recorded. Two prompts deserve explicit answers from the repo (below): the oracle-ceiling question (answered by the ablation) and the vol-model question (answered by ratified decision: design fully, gate the build) | **ALIGNED** |

**Appendix follow-ups the repo can now answer:**

- *"Does the oracle already contain most of the economically relevant logic, making RL
  redundant?"* — Partially answered by measurement: the scalarized oracle's cross-sectional
  boundary is largely linearly recoverable (gap 0.015), i.e., the *one-step* rule is not where
  the hard problem lives. The temporal problem (soft target, gap ~0.19) and the shared-resource
  coupling are where sequential value can hide. That is exactly what v0.4a is designed to
  quantify before RL is attempted.
- *"Is the propensity target closer to policy learning than classification?"* — The repo's
  answer stands: still supervised prediction, but `Y_Soft_BT` under the scalarized oracle is
  now "how often would this lot clear a *real cost-benefit test* in the next 30 days" — a
  better-defined quantity than under the vestigial gate, and the natural value-function
  warm-start.

---

## 3. What the GPT thread missed (the repo-only knowledge)

These are absent from the companion document because they live in the 20-year data, the
code, or the issue ledger — none of which the thread saw. Any future agent reading the
companion doc alone would mis-plan without them.

1. **Cost-basis aging (the P0).** The open-once/hold-forever book ages out of
   harvestability: a 2007 basis is so deep in the money by 2020 that a 34% crash can't push
   it 2% underwater. Harvest signal: one GFC eruption, then near-extinction. Consequence:
   **contributions, rebalancing, and the sell-winner trim come before universe compression**
   — there is little point optimizing a core basket for harvestable dispersion while the
   simulator structurally extinguishes dispersion in year 3. The GPT roadmap has no stage
   for this.
2. **The trim process is dual-purpose.** Beyond fixing aging, trimming winners makes
   `RealizedGainsYTD` endogenous (activating the ledger's carryforward-netting path that the
   loss-only book never exercises) and is the **first non-harvest action** — pre-building the
   action space v0.4b needs. It is v0.4 scaffolding disguised as simulator realism.
3. **The measured ablation and its methodology.** Manufactured vs. real non-linearity
   (`GYTD_Redesign_Plan.md` §6.1): the oracle-target tree advantage was mostly gate-geometry
   artifact; the temporal advantage is real. And the *method* matters as much as the result —
   PR #24 landed the schema change with byte-identical labels, which is the only reason the
   feature-set effect (0.12→0.83) could be separated from the oracle-swap effect (0.83→0.98).
   That attribution discipline (one boundary-shaping change per PR, spectator labels,
   separate acting-oracle runs) is a repo asset the plan should carry forward as a rule.
4. **`MonteCarloEngine`.** A second, fully synthetic environment (calibrated GBM, same
   schema) already exists — relevant to v0.4b (cheap policy-training episodes, stress
   regimes) and never mentioned by the thread.
5. **Knowledge distillation (issue #16, v0.5–v1.0).** The GBT's sixth role: teacher for a
   deployable student model. Absent from the thread's five roles.
6. **Survivorship / point-in-time membership.** The universe is 409 survivors of ~503; the
   report flags this as biasing tax-alpha magnitudes optimistically. Universe compression
   (§6 of the thread) *interacts* with this: a core selected on survivor covariance inherits
   the bias. Point-in-time membership is a v1.0 data-layer prerequisite.
7. **Operational discipline the thread can't see:** the codebook header-drift assert, the
   report/token regeneration chain (edit the builder, never the notebook), `MLNetLeakageAudit.md`'s
   fit-on-train-fold-only invariants, the Nix environment issue (#21), and the ~90-second
   full-20y simulate loop that makes "re-simulate and re-measure" a cheap default.

---

## 4. What to adopt from the GPT thread (ranked)

1. **Purged, embargoed, chronological validation (→ v0.26).** The thread's sharpest
   technically-new point: `Y_Soft_BT` has a 30-trading-day forward window, so adjacent rows
   share future context; stratified *random* splits can leak that context across the
   train/test boundary and flatter every soft-target number in the repo (hard-oracle numbers
   are cross-sectional and less exposed, but the same hygiene should apply). Remedy:
   chronological splits with a purge/embargo ≥ the label horizon around every boundary.
   This gates everything else — adopt first.
2. **The six-rung baseline ladder (→ #12/#22 harness spec).** (1) benchmark/no-DI, (2) naive
   loss-threshold harvesting, (3) deterministic oracle, (4) GBT + fixed thresholds, (5) GBT +
   constrained optimizer, (6) + policy/RL. This is the evaluation spine the metric issues
   have been missing; every future version's "gate criteria" in §5 reference rungs of it.
3. **Core+reserve universe formalization (→ v0.35, upgrades #6).** Including the two
   specific insights worth preserving verbatim: the core must target the *benchmark* (not a
   hypothetical full-DI book), and compression that destroys idiosyncratic dispersion
   destroys the harvest opportunity it was meant to serve — so `DispersionPotential` belongs
   in the basket objective, not just TE.
4. **Eligibility-first replacement + `SubScore` (→ v0.35, upgrades #5 + the HarvestLot
   comment).** Compliance is rules, not correlation; scoring is deterministic over empirical
   estimates (ΔTE, factor distance, sector, liquidity, concentration, future harvestability);
   fallback behavior is an explicit configured policy.
5. **The constrained-optimizer execution baseline as its own stage (→ v0.4a; GPT Phase C).**
   The most important *structural* adoption: the repo roadmap currently jumps from
   supervised models to RL. Inserting "GBT scores + deterministic subset optimizer +
   replacement optimizer under shared TE/turnover budgets" as a distinct, measured stage
   does two things: it may capture most of the cross-lot value by itself (the shared-resource
   coupling is a *feasibility* problem before it is a *timing* problem), and it makes v0.4b's
   go/no-go decision empirical: RL proceeds only if a material sequential gap remains.
6. **Smaller adoptions:** turnover as an explicit future penalty term in U/reward (alongside
   the recorded non-goal of lot-varying c_trade); the anti-leakage rule for
   predictions-fed-to-policies (train on past → predict forward block → feed policy; never
   in-sample scores); the eight guardrails as a standing section in the promoted plan; the
   PR-AUC ≠ tax alpha framing (already report prose, now a named guardrail).

---

## 5. The synthesized version planner

The version spine is the repo's (README Roadmap / `PSTAT231_RECAP.md` §7–§9), with the GPT
thread's Phases A–E interleaved where they belong. Format per version: **goal → workstreams
(with repo seams + issues) → gate criteria → crosswalk**. Standing rules across all versions:
one boundary-shaping change per PR; ablation columns/runs for anything that changes labels or
trajectories; schema changes take a version bump through the codebook assert; economic claims
cite the baseline ladder, not classification metrics alone.

---

### v0.25 — Oracle redesign ✅ (complete, July 2026)

Shipped as PRs #24–#27: `TaxLedger` (schema v3, d=17, byte-identical landing), scalarized
oracle `f* = 𝟙[ℓ≤−θ₁]·𝟙[𝒲≥30]·𝟙[σ_TE≤θ_max]·𝟙[U>0]` with `U = TaxValue − λσ_TE² − c_trade`,
`--oracle=gated|scalarized` ablation arms + spectator label, c_trade=$10, report + theory
docs rewritten. Measured: oracle-target GBT−logistic gap 0.155→0.015; soft-target gap
oracle-invariant; value-function regression R² 0.10 (linear) vs 0.92 (trees); θ_max binds on
0 rows; carryforward load-bearing ($4.3M/20y). *Crosswalk: GPT Phase A items 1–2, 4–5 done
here; the thread's "current milestone" section describes this state minus the measurements.*

### v0.26 — Validation hardening (NEW; GPT Phase A item 3 + critical note)

**Goal:** re-establish every headline number under leak-proof temporal methodology before
building anything new on top.

**Workstreams:**
- `TemporalSplit` in `src/ML/CSharp/MLNet/Splits/` (alongside `StratifiedSplit`):
  chronological train/test with a configurable purge/embargo window (default ≥ 30 trading
  days — the `Y_Soft_BT` horizon) excised around every boundary; same for the CV folds
  (forward-chaining or purged k-fold).
- Re-validate the v0.25 champions and the full 2×2 gated/scalarized ablation table under
  purged chronological splits; add the walk-forward decade split the report already names
  (train 2006–2016, test 2016–2026).
- Re-check class weights / prevalence handling per split scheme (positive rate drifts across
  regimes, so chronological folds are more imbalanced than stratified ones).
- Report appendix: random-split vs purged-split deltas, so the honest-evaluation story is
  itself measured.

**Gate:** if soft-target numbers hold within a few points, proceed with confidence; if they
drop materially, diagnose (likely overlap leakage) *before* v0.3 — better now than under a
more complex simulator. **Cheap:** no simulation changes; retraining only (~minutes per arm).

### v0.3 — Simulator realism + metric layer (RECAP v0.3 row ∪ prerequisites GPT skipped)

**Goal:** make the book behave like an account a live client would recognize, and make
evaluation economic.

**Workstreams:**
1. **Sell-winner trim process** (`SimulationEngine` extension mirroring `HarvestLot`):
   periodic trims of overweight winners toward index weight. Dual purpose: realizes gains →
   `RealizedGainsYTD` endogenous → carryforward netting exercised; and it is the first
   non-harvest action (v0.4b scaffolding). Ship as its own PR with an on/off flag (the ledger
   is already correct with it off).
2. **Contributions / fresh lots** — the cost-basis-aging fix (the scale-up's P0): periodic
   cash inflows minting lots at current prices; optionally dividend reinvestment. Measure the
   harvest-signal recovery (oracle rate, dark-window count, N_rows) as its own ablation.
3. **Tax-alpha & deployment metric layer (#12/#22)** built as the GPT §2.2 harness: the
   six-rung baseline ladder, after-tax wealth, realized/deferred benefit decomposition
   (immediate-use vs banked, straight off the ledger), TE/turnover/cost accounting per run.
   From here on, version gates cite ladder rungs.
4. **Richer soft-label families (#17):** `y_alpha`-style targets now fall out of `U` and
   `Y_TaxValue`; add horizon variants and max-vs-mean aggregations as label columns
   (labels may peek forward; features never).
5. **Volatility/uncertainty sub-model — DESIGN IN FULL, BUILD BEHIND A GATE** (ratified):
   - *Design (unconditional):* a supervised estimator `history → (σ̂_i, Σ̂)` (EWMA/GARCH
     per-name; Ledoit-Wolf or RMT-cleaned covariance, issue #5). Roles, spelled out so the
     design survives even if the build waits: (a) **opportunity creation** — volatility is
     the generator of harvestable dispersion, so σ̂ is the leading indicator of *future*
     TLH opportunity, not just a risk number; (b) `Y_Soft_GBM` calibration (replaces the
     trailing-21d/0.20-fallback σ); (c) Σ̂ into `TrackingErrorProxy` (the v0.3 theory-memo
     program); (d) **the uncertainty axis that distinguishes otherwise-identical lot/portfolio
     states** — two lots with the same (ℓ, h, TaxValue, σ_TE) but different forward σ̂ have
     different option value to a policy, which is precisely information an agent can act on
     and the current state cannot express.
   - *Build gate (decision-flip ablation):* swap σ̂/Σ̂ into the pipeline and measure the
     fraction of oracle/U decisions and Y_Soft_GBM labels that flip. Material flip rate →
     build the full sub-model layer; negligible → record the negative result (it is itself a
     §4.2-style overlap finding) and revisit at v0.4b, where the *policy* consumes uncertainty
     even if the *oracle* doesn't.
6. **Optional:** typed ST/LT ledger pools (law-exact Schedule D), recorded extension.

**Gate:** aging fixed (harvest signal persists across decades), ledger fully exercised
(carryforward absorbs endogenous gains), metric harness runs rungs 1–3 of the ladder.
*Crosswalk: RECAP v0.3 row; GPT has no equivalent stage — this is the inserted prerequisite
for its Phase B.*

### v0.35 — Universe & replacement layer (GPT Phase B; upgrades #5/#6)

**Goal:** core+reserve construction and the substitute path — the two missing layers (1 and
6) of the GPT §5 decomposition.

**Workstreams:**
1. Covariance cleaning productionized (RMT #5 / Ledoit-Wolf from v0.3) feeding both basket
   optimization and TE.
2. **Offline core-basket optimizer** (deterministic, slow cadence): min TE + complexity
   penalty − dispersion potential, s.t. budget/cardinality/sector/liquidity/concentration —
   GPT §6.4's formulation, targeting the *benchmark*. PCA for risk compression, clustering
   for representative selection (their §6.3 division of labor). Sweep |C| (e.g. 75–200 names)
   → the "how small before TE or dispersion degrades" curve (GPT §12.2).
3. **Replacement module:** eligibility rules first (wash/compliance/liquidity/concentration —
   rules, not correlations), then deterministic `SubScore` (ΔTE, factor distance, sector,
   liquidity, concentration, future-harvestability estimate), then explicit fallback policy
   (proxy / cash / timed re-entry). Replaces the current same-ticker-reopen-only behavior;
   keep reopen as the ablation baseline.
4. **Hybrid-environment re-simulation + transfer test:** regenerate labels in the
   core+reserve environment and test whether the full-universe GBT transfers or must be
   retrained — mandatory, because the v0.25 ablations already proved trajectories are
   oracle- and environment-dependent (GPT §8.2, elevated from hypothesis to rule).
5. Flag survivorship interaction: basket selection on survivor covariance inherits the bias;
   note point-in-time membership as the v1.0 fix.

**Gate:** hybrid book tracks benchmark within budget at materially lower name count while
preserving measured harvestable dispersion (ladder rungs 2–4 in the hybrid environment).

### v0.4a — Constrained execution baseline (GPT Phase C — NEW stage, adopted)

**Goal:** capture cross-lot, budget-coupled value *without* RL, and measure what remains.

**Workstreams:** daily loop = GBT/soft scores rank held-lot candidates → deterministic subset
optimizer selects the feasible action set under shared TE/turnover/capacity budgets →
replacement optimizer executes substitutes → ledger/TE state transitions. All predictions
consumed by the optimizer are produced walk-forward (train-on-past → score forward block),
per the anti-leakage rule for fed predictions.

**Gate (the RL go/no-go):** ladder rung 5 vs rung 4. If the optimizer captures nearly all
the gap to the oracle-driven ceiling, v0.4b shrinks or waits; the residual — timing/deferral
value the one-step optimizer can't see — is RL's earnable surplus, now *quantified*.
*Crosswalk: GPT Phase C verbatim; absent from the previous repo roadmap.*

### v0.4b — RL policy layer (RECAP v0.4 ∪ GPT Phase D; issue #15)

**Goal:** high-level sequential control over the deterministic execution stack.

**Workstreams:** `IHarvestPolicy` (#15) with the oracle-as-greedy-baseline; **low-dimensional
action space** (adopted from GPT §9.2): `a_t = (τ_t threshold, TE-budget spend, turnover
budget, top-m candidates, regime)` — not per-lot binaries; state = ledger + TE + factor
summaries + candidate score summaries + wash/budget summaries (+ σ̂ uncertainty if the v0.3
gate passed, and even if it didn't — re-test here, since option value is a policy concern);
reward = accumulated `Y_Utility` (already exported; the ledger is the reward-accounting
substrate, deterministic under the policy); warm start from the supervised η̂ and the
value-function regression; PPO/SAC per RECAP; `MonteCarloEngine` as cheap episode generator
and stress-regime source before real-history fine-tuning.

**Gate:** ladder rung 6 vs rung 5 under walk-forward evaluation, stability across regimes,
alpha per unit TE/turnover — GPT §12.4's questions verbatim.

### v0.5 — End-to-end evaluation + integration (RECAP v0.5 ∪ GPT Phase E)

Walk-forward economic comparison across the entire ladder (the Phase E study, now with all
rungs built); live/recent data; client parameterization (hazard-rate δ replacing the
constant, outside-gains personas, goal-conditioned discounts — the `GYTD_Redesign_Plan.md`
§1.5/§3.3 hooks); knowledge distillation (#16) of the champion stack into a deployable
student; environment reproducibility (#21).

### v1.0 — Deployment (RECAP v1.0)

RIA-style service, author as first client. Point-in-time constituent feed (kills the
survivorship caveat), auditable tax/compliance module boundary (guardrail #7 — `TaxLedger`
already is one), monitoring, and the standing guardrails (§4.6) as operating policy.

---

## 6. Corrections ledger (stale or imprecise GPT claims → current facts)

| GPT claim | Correction |
|---|---|
| "GBT ≈ 0.8–0.9 PR-AUC" | Per-target/per-arm (CV PR-AUC, d=17, 20y): oracle — GBT 0.9956 / logistic 0.9804 (scalarized), 0.9865 / 0.8320 (gated); soft — GBT 0.8158 / logistic 0.6233 (scalarized), 0.8503 / 0.6904 (gated). The interesting quantity is the *gap structure*, not the level |
| "Do not assume … RL, universe compression, replacement optimization implemented" | Correct then and now — and v0.25 (ledger, scalarized oracle, ablation infra, tax/utility labels) *is* implemented, which the thread's status line predates |
| Lot state includes q (shares) as a feature | `Shares` is non-exported plumbing for soft-label re-dollarization; the feature schema is d=17 without it |
| Objective includes λ_turn·Turnover | Not yet a term; U has TE² and flat c_trade. Turnover enters as a v0.4a budget constraint first, possibly a reward term later |
| "Oracle currently incorporates … trading costs, TE penalties" (appendix, forward-tense) | Now true (PRs #25–#26); was aspirational when written |
| Oracle-ceiling worry: "does the oracle leave room for RL?" | Measured answer: the one-step boundary is nearly linear-recoverable (gap 0.015) — the ceiling concern is real for *cross-sectional* prediction; the surviving room is temporal (soft gap ~0.19) + shared-resource coupling, exactly what v0.4a/b are staged to quantify |
| Phase A: "freeze and document the LotStateVector and label semantics" | Done continuously: codebook + header-drift assert + `MLNetLeakageAudit.md`; schema changes are version bumps, not drift |

---

## 7. Open-questions register (merged: GPT §12 + appendix + repo)

Preserved as the durable research backlog; versions in §5 reference these.

**Supervised / validation:** Do the v0.25 results survive purged chronological splits and
the decade walk-forward (v0.26)? Which feature groups carry unique signal (the ledger triple
vs TaxValue collapse)? Does the full-universe GBT transfer to the hybrid environment (v0.35
test)?

**Economics:** How much of the oracle-driven ceiling does GBT+optimizer capture (v0.4a)?
How much tax alpha survives realistic frictions and the loss-only vs high-outside-activity
personas (δ hazard modeling)? What does the $10M→$75M/20y thought experiment yield under the
ladder — is the 20%→17% effective-rate intuition (≡ 15% tax-bill cut) achievable?

**Construction:** Optimal |core| tradeoff curve; does compression destroy harvestable
dispersion; which replacement features predict realized tracking + future harvestability;
how often does the eligibility system dead-end into fallback?

**Volatility:** Does σ̂/Σ̂ flip decisions (v0.3 gate)? If not for the oracle, does
uncertainty still earn its place as policy state (v0.4b re-test)? Does Σ̂ cleaning move
σ_TE enough to matter at θ_max scale?

**Sequential:** Does high-level RL beat the constrained baseline; does it learn genuine
deferral (sacrificing current U for later opportunity); is it stable across regimes; does
MC-pretraining transfer to real history?

**System:** Do the gains from vol model + GBT + optimizer + RL overlap or add (the §4.2
non-additivity warning — every layer gets an ablation column)? Does total complexity clear
the operational/model-risk bar for a solo-RIA deployment (guardrail territory)?

---

## 8. Bottom line

The inquiry thread was a good investment: with almost no supplied context, it converged on
the project's ratified philosophy, correctly formalized several intentions that existed only
as code comments and issue one-liners (core+reserve, SubScore, high-level RL actions), and
contributed one genuinely load-bearing correction (purged temporal validation) plus one
structural improvement to the roadmap (the constrained-optimizer stage). What it could not
supply — and what this document adds — is everything the data taught (cost-basis aging, the
measured ablation, the manufactured-vs-real non-linearity split) and everything the repo
already encodes (the version spine, the issue ledger, the attribution discipline, the
ledger-as-reward substrate). The synthesis in §5 is the roadmap the README table abbreviates;
after review, promote it to `DataMemo/ArchitecturePlan.md` and point the README at it.

*Cross-references: `GYTD_Redesign_Plan.md` v2 (§6.1 measured ablation, §8 RL bridge);
`PortfolioMath.md` §2.3/§4 (ledger + composite oracle); `SimulationMath.md` §2/§4;
`Lifecycle_v02.md` (+ v0.25 callouts); `PSTAT231_RECAP.md` §7–§9; README Roadmap; issues
#5, #6, #12, #15, #16, #17, #21, #22, #23; PRs #24–#27; companion:
`direct_indexing_concept_architecture_plan.md` (this directory).*

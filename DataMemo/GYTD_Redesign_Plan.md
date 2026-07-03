# G_YTD Redesign Plan v2 — from binary gains gate to scalarized tax-aware objective

> **Status:** ratified design, v0.2-capstone → v0.25/v0.3 target (issue #23). Supersedes the
> original `GYTD_Redesign_Plan.md` (v1, "Options A/B/C") — v1's text survives in git history.
>
> **Implementation status:**
> - [x] **Step 1 shipped** (`feature/v025-pr1-taxledger`): `TaxLedger` (loss-only) replaces the
>   `G_YTD` scalar; schema v3 (d = 17: `RealizedGainsYTD`/`LossCarryforward`/`OrdinaryOffsetBudget`
>   replace `G_YTD`, capacity-aware `TaxValue` replaces `TaxAlpha`); new `Y_TaxValue` regression
>   label. Verified: labels byte-identical to the v0.2 baseline over all 1.85M rows.
> - [ ] Step 2 (oracle swap: gates·𝟙[U>0], TE continuization, `--oracle` ablation flag)
> - [ ] Step 3 (`c_trade`)
>
> **Ratified decisions (2026-07-02).** (1) Oracle identity: *industry-faithful engine* — the
> oracle IS a mechanistic retail DI engine, 3 hard gates · 𝟙[U(x)>0]; ablations are measurement
> discipline, not the project's identity. (2) v0.25 scope: full scalarized oracle (steps 1–3),
> trim process deferred. (3) Ablations: separate runs via `--oracle=gated|scalarized` plus a
> spectator gated label inside the scalarized run. (4) Schema: expose the raw ledger triple
> plus `TaxValue` (d = 17), one version bump.
>
> **What changed since v1.** v1 framed the problem as *calibration* (the gate binds on ~6% of
> rows over the 20-year window; the seed doesn't scale) and proposed three fixes of increasing
> ambition. Subsequent analysis against published institutional TLH methodology (Wealthfront
> stock-level TLH whitepaper; Betterment TLH methodology) established a stronger claim: the
> gate's **functional form is wrong for the individual-investor DI use case, not merely its
> constant**. No production retail TLH engine conditions the harvest decision on current-year
> realized gains; the correct home for gains/ledger information is a **continuous value term
> inside a scalarized objective**, not any gate — binary or thresholded-continuous. This
> document records that reframing, the resulting composite oracle, the tax-law grounding, the
> feature-space consequences, and the predicted model impacts, so a future instance can
> implement without re-deriving the thread.

---

## 0. TL;DR — the target oracle

```
f*(x) = 𝟙[ℓ ≤ −θ₁]                     hard gate: loss depth        (KEEP — real threshold rule)
      · 𝟙[𝒲 ≥ 30]                      hard gate: wash-sale clock   (KEEP — IRS §1091)
      · 𝟙[σ_TE ≤ θ_max]                hard gate: LOOSE TE ceiling  (KEEP, re-tuned — tail-risk circuit breaker)
      · 𝟙[ U(x) > 0 ]                  scalarized objective         (NEW — absorbs G_YTD and fine-grained TE)

U(x) = τ(h)·taxValueₖ(ledgerₜ)  −  λ·σ_TE²  −  c_trade
```

- `G_YTD` is **removed as a gate** and reborn as `taxValueₖ`, a continuous ℝ≥0-valued function
  of a new `TaxLedger` state object (§3), sitting in the *reward* position of `U(x)`.
- The old fine-grained TE cap θ₂ is **demoted**: the marginal TE tradeoff moves into the
  continuous `−λσ_TE²` term (matching Wealthfront's published objective, tax alpha minus a
  tracking-error-squared penalty); a **loose** ceiling `θ_max ≫ θ₂` survives as a hard gate
  purely to bound pathological/tail regimes (§2.3).
- `c_trade` is a new additive transaction-cost term (Betterment's "benefit net of cost" test).
- `f*` remains an indicator `𝒳 → {0,1}` — harvest is a discrete action — but the boundary is
  now the **level set** `{x : U(x) = 0}` rather than the corner of an axis-aligned box.

---

## 1. Why the gate is structurally invalid (not just miscalibrated)

### 1.1 Empirical trigger (carried over from v1)

20-year backtest (1.85M rows, 2004–2024): `G_YTD > 0` holds on ~94% of rows. A gate open 94%
of the time is noise in the conjunction — but note the flip side, measured in
`data_memo_theory_part2.md` §A.5: logistic regression's oracle-target PR-AUC swings from
**0.987 (2-year run, gate binds on 0% of rows)** to **~0.12 (20-year run, binds on ~6%)**.
Even a 6%-binding fourth gate is the dominant source of the oracle's non-linearity for linear
models. This number anchors every prediction in §6.

### 1.2 Institutional evidence — the gate has no counterpart in real systems

- **Wealthfront** (stock-level TLH whitepaper): harvests losses **based on a threshold on the
  loss itself**, and states its objective as maximizing a function of **tax alpha minus
  λ·(tracking error)²**. No current-year-gains precondition anywhere in the methodology.
- **Betterment**: harvest fires when **benefit net of cost exceeds a threshold**, where cost
  includes trading friction and the opportunity cost of waiting for a better harvest — again,
  never "does the client currently have gains."

### 1.3 Tax-code grounding — why no rational engine would gate on YTD gains

For an **individual** (this project's domain), a qualifying realized loss is *never wasted* by
harvesting it now: it offsets realized gains from **anywhere on the 1040** (not just this
portfolio), then up to **$3,000/yr of ordinary income**, and the remainder **carries forward
indefinitely** with preserved short/long-term character (26 USC §1212; Schedule D netting).
"Do I have gains today" therefore never vetoes the harvest decision — it only shifts *how much
of the benefit is immediate vs. deferred*. That is magnitude/timing information, i.e. it
belongs in a **value function**, not a gate.

### 1.4 Strategy-philosophy grounding — the gate is anti-correlated with DI itself

DI's engineering objective is second-moment index replication (minimize
`σ̂_TE = √(δwᵀ Σ̂ δw · 252)`, which subsumes β-matching), under the passive buy-and-hold
thesis: turnover is the enemy, and the strategy's terminal state is *never selling winners*
(hold to death → §1014 basis step-up extinguishes the embedded gain). A gate requiring the
portfolio to have *realized gains this year* as a precondition to harvest demands the one
behavior the strategy is built to avoid (opportunistic gain realization) before permitting the
one behavior it is built to do (loss harvesting). A well-run DI book posting large YTD gains
most years would itself signal excess unforced turnover. `G_YTD`'s implicit assumption
(reliable annual exogenous gains) is the assumption structure of an *active/rebalancing*
strategy — or of the corporate regime in §1.5 — not retail long-horizon indexing.

### 1.5 Where a G_YTD-style gate WOULD be valid (documented non-goals)

- **C-corporations** (26 USC §1212(a)): capital losses offset **capital gains only** (no
  ordinary-income floor), carry **back 3 / forward 5** years, then **expire**. Bounded horizon
  + no floor ⇒ "will gains exist to absorb this within the window" is a *real* constraint —
  the precise condition under which a gains-availability gate is economically legitimate.
  **Out of scope**; recorded so nobody generalizes the individual ledger to entities by accident.
- **Goal-conditioned clients** (known large gain this year: RSU vest, business sale): handled
  as a *parameter*, not a gate — lower the effective discount δ (§3.3) for that client so
  current-year harvests are valued nearer face value. Same object, different role.

---

## 2. The composite oracle, gate by gate

### 2.1 Gates that survive unchanged — and the criterion for why

**Hard gates remain only where they encode a genuine legal rule or threshold fact:**

| Gate | Grounding | Status |
|---|---|---|
| `ℓ ≤ −θ₁` (loss depth) | Wealthfront's own trigger is a loss threshold | keep |
| `𝒲 ≥ 30` (wash clock) | IRS §1091, not a design choice | keep |

### 2.2 The scalarized objective `U(x)`

`U(x) = τ(h)·taxValueₖ(ledgerₜ) − λσ_TE² − c_trade`, harvest-relevant iff `U(x) > 0`.

Two implementation shapes were considered for the ledger term and **shape (b) is chosen**:

- (a) *continuous-threshold gate*: keep `taxValueₖ > θ₄` as its own AND-ed factor. Rejected —
  still structurally a gate; discards the magnitude information that makes a ledger useful,
  and forces `taxValueₖ` to compete with TE as an independent pass/fail rather than trade off
  against it.
- **(b) *fully merged scalar*: `taxValueₖ` sits in the reward position of one `U(x)`,
  thresholded once.** Matches the institutional objective form and composes directly with the
  v0.4 RL reward (which is exactly this scalar minus the TE penalty).

**Type discipline.** `U : 𝒳 → ℝ` is a genuine intermediate object, not just plumbing: it is
exportable as a **continuous regression target** in its own right (issue #17's `y_alpha`
family gets this for free once `U` exists). `f* = 𝟙[U > 0]` keeps the codomain `{0,1}`.

### 2.3 TE: continuous term PLUS a loose hard ceiling (revision vs. earlier thread drafts)

A purely additive objective has **no upper bound on tolerated TE**: a catastrophically large
loss makes `τ(h)|ℓ|` dominate `λσ_TE²`, mathematically "justifying" arbitrary benchmark
deviation. Fine in the typical operating range; broken in tails (bear-market regimes with many
simultaneous deep losses; and cf. the v0.0 war story where a TE-computation artifact spiked
median TE to 33%). Standard risk-system practice: **soft objective for marginal tradeoffs,
hard ceiling as circuit breaker, operating at different scales.** So:

- `−λσ_TE²` inside `U(x)` does the fine-grained shaping (was gate 2's job);
- `𝟙[σ_TE ≤ θ_max]` survives as a hard gate with `θ_max` **much looser** than the old θ₂ —
  it should essentially never bind in normal regimes, only in pathologies.

Net structure: **three hard gates + one scalarized term**, not "two gates + objective."

### 2.4 `c_trade`

Additive cost term (Betterment's net-benefit test). v0.2/v0.3 scope: a **flat constant** —
which means it lives as a **config parameter inside `OracleBoundary`, NOT a feature** (a
constant cannot discriminate rows; something needed to *compute* a label is not automatically
something the model must *see*). It becomes a lot-level feature only if later modeled as
varying (proportional to notional / bid-ask), at which point it is lot-level, not shared state.

---

## 3. The TaxLedger — the tax-law object replacing the seed

### 3.1 Mechanics being encoded (real Schedule D logic, individual regime)

1. Losses exist for tax purposes only when **realized** (the premise of TLH).
2. Every realized gain/loss has **character**: short-term (≤1y, ordinary rates — why
   `ComputeTaxAlpha` used `holdingDays >= 365 ? 0.20 : 0.37`) vs long-term (>1y, preferential).
3. **Netting order**: ST nets against ST; LT against LT; opposite-signed nets then net against
   each other → one annual net number.
4. Net loss → up to **$3,000/yr** offsets ordinary income (use-it-or-lose-it annually).
5. Remainder **carries forward indefinitely**, character preserved, re-entering next year's
   netting — the mechanic that makes "harvest now, use later" always weakly correct for
   individuals, and the direct refutation of the gate.
6. **Terminal boundary: death, and it is forfeiture, not usage** (Rev. Rul. 74-175). Unused
   carryforward can be applied on the final Form 1040 (still under the $3k ordinary-income
   cap) and then **evaporates** — it does not transfer to heirs or the estate. Meanwhile the
   *positions* (including harvest substitutes) receive the §1014 step-up. Consequence: the
   TLH *mechanism* is fully compatible with hold-to-death (each substitute gets its own
   step-up), but the **marginal dollar of carryforward beyond what will plausibly be absorbed
   before death has expected value → 0** (negative, net of the c_trade spent creating it).

### 3.2 State object (as shipped in step 1)

```csharp
// portfolio-level shared state 𝒮_t — replaces the seeded G_YTD scalar
// (src/Core/Portfolio/TaxLedger.cs; mutable class matching PortfolioState's style)
public sealed class TaxLedger
{
    public decimal RealizedGainsYTD { get; }   // signed NET realized P&L YTD (≡ old G_YTD);
                                               // resets to 0 at year-end
    public decimal LossCarryforward { get; }   // SURVIVES year-end — the field the
                                               // old constant re-seed got wrong
    public decimal OrdinaryOffsetBudget =>     // DERIVED: max(0, $3k − net loss so far);
        ...;                                   // resets implicitly at year-end
    public decimal OffsetCapacity =>           // max(net, 0) + OrdinaryOffsetBudget
        ...;
}
```

Implementation note vs. the original sketch: `RealizedGainsYTD` stores the **signed net**
(not a gains-only gross), which is exactly the old G_YTD semantics — required for the
byte-identity guarantee in gated mode — and `OrdinaryOffsetBudget` is **derived** from it
rather than stored, eliminating a redundant-state consistency hazard.

Transitions tie to the existing `AdvanceDay` / `ResetForNewYear` seams:

- Harvests (`RecordRealized`) accrue signed P&L; the legacy seed (`RecordExternalGains`)
  deliberately does NOT net against carryforward, preserving gated-mode bit-identity.
- `RollYearEnd` (called by `ResetForNewYear`): net loss beyond the $3k allowance banks into
  `LossCarryforward`; `RealizedGainsYTD → 0`; budget resets implicitly.

This is **deterministic bookkeeping over already-simulated events** — a state-transition
function `𝒮_t × decision_t → 𝒮_{t+1}`, structurally identical to `WashClock` evolution. It is
categorically NOT a sub-ML problem: unlike the v0.3 volatility sub-model (an estimator
`history → Σ̂` of an unobserved *future* second moment), the ledger has no latent parameter —
everything it needs is computed by the engine in the same pass. The estimator-vs-bookkeeping
distinction is the criterion. Learning enters only at v0.4, and what is learned there is the
*policy*; the ledger stays deterministic underneath it as the reward-accounting substrate.

Empirical note from the step-1 rerun (gated mode, 20y): `LossCarryforward ≡ 0` throughout —
the gains gate is self-limiting (harvesting halts once net ≤ 0), so year-end net never fell
below the $3k allowance. The carryforward becomes load-bearing exactly when the gate is
removed (scalarized mode, seed off) — every harvested dollar beyond $3k/yr banks.

### 3.3 The value function

```
offsetCapacityₜ = max(RealizedGainsYTDₜ, 0) + OrdinaryOffsetBudgetₜ
taxValueₖ = τ(hₖ) · min(|ℓₖ|, offsetCapacityₜ)                    // used THIS year, full rate
          + τ_future · max(|ℓₖ| − offsetCapacityₜ, 0) · δ         // banked, discounted
```

`LossCarryforward` does not appear as an input — it is the state that *accumulates the
output* of this formula across years.

Shipped constants (step 1): `τ_short = 0.37`, `τ_long = 0.20`, `τ_future = 0.20`, `δ = 0.5`.

**δ is a hazard-rate object, not a flat time-value discount:**

```
δ ≈ Pr(loss absorbed by a gain before death) × time-value discount to absorption
```

High for clients with frequent outside gain events (δ → face value); shrinking toward
"share of the $3k/yr trickle × remaining years" for a purely passive low-outside-activity
client, per §3.1(6). v0.25 implements δ as a constant, but this framing must be
recorded so "carryforward = eventually worth face value" is never silently assumed.

**Usage-pattern reality (client-type dependence, stated explicitly):** absorption is
(i) a guaranteed $3k/yr floor plus (ii) **event-driven lumps whenever the client realizes a
gain anywhere on their 1040** — RSU vests, business/property sales, outside portfolio trims —
not a scheduled annual flow and not necessarily from the DI book at all. The relevant client
attribute is *outside capital-gains activity*. The loss-only simulation
(`RealizedGainsYTD ≡ 0` in scalarized mode, §4 deferred) therefore models the
**low-outside-activity persona**: a conservative floor on tax alpha, understated for the
high-earner persona DI products actually target. One sentence saying so belongs in the report.

**Known simplification (state it, don't hide it):** v0.25/v0.3 collapses ST/LT into one
blended pool with a single `τ(hₖ)` applied at harvest. Real Schedule D runs two typed pools
with cross-netting and character-preserving carryforward. Splitting into two typed pools is a
bounded future extension for law-exact behavior.

**Consistency obligation (RESOLVED in step 1):** the old `ComputeTaxAlpha` returned `τ(h)·|P&L|`
unconditionally whenever the gate was open — no `min()` against capacity, and it counted
winners' |gains| as if they were harvestable losses. Both defects are folded into the new
`TaxValue = TaxLedger.ComputeTaxValue(lossDollars, h)`; the old feature no longer exists.

---

## 4. Gains-realization process (§3 of v1, carried forward — DEFERRED past v0.25)

The simulator only ever realizes losses; that absence is *why* a seed existed. Making
`RealizedGainsYTD` endogenous requires a **"sell winner" transition** mirroring `HarvestLot`:
periodic trims of overweight winners back toward index weight (also the natural counterpart
to harvest-and-reopen for the TE story). This is a `SimulationEngine` extension, deliberately
scoped as **dual-purpose infrastructure**: it is the first non-trivial *action* beyond
harvesting, pre-building the action space the v0.4 `IHarvestPolicy` layer requires.

Note the orthogonality: **the ledger is correct even with this process set to zero** — it
collapses to `offsetCapacityₜ = $3,000/yr`, which is tax-code-accurate for a pure-harvest,
never-trim book. Ship order is therefore ledger first (done), trim process later.

---

## 5. Pipeline consequences

### 5.1 Labeling — `Y_Soft_BT` is invariant by construction

`SoftLabelBuilder` consumes `f*` as a black box `𝒳 → {0,1}` and Cesàro-averages
`ỹ_BT(x_t) = (1/30)·Σₛ f*(x_{t+s})`. Swapping the oracle's internals (AND → gates·𝟙[U>0])
requires **zero changes** there — the payoff of `OracleBoundary` being stateless/referentially
transparent (`Lifecycle_v02.md` §3.4). Keep the two softness axes distinct:

- `U(x)`: **cross-sectional** — value of this lot at this timestep;
- `Y_Soft_BT`: **temporal** — firing frequency of the (still-binary) oracle over 30 future days.

They compose; they do not collide. The propensity label over the revamped oracle is in fact
*better-defined*: "how often would this lot clear a real cost-benefit test" vs "…clear an AND
containing a vestigial gate." New exports: `Y_TaxValue` (regression on `taxValueₖ` — SHIPPED,
step 1) and optionally raw `U(x)` as its own target (issue #17 family — step 2).

Level-set note (ties to `data_memo_theory_part2.md` §5.A): the new decision boundary is
exactly `∂Ω = {x : U(x) = 0}` — the same mathematical object as the fitted model's level
curves `L_c = {x : η̂(x) = c}`, appearing at the label-generating stage instead. Consistency,
not coincidence.

**Ablation mechanics (ratified decision 3).** The acting oracle changes the trajectory
(harvests → wash clocks → ledger → 30-day dark windows → which rows exist), so gated and
scalarized labels CANNOT coexist as two columns of one honest run. Ablations are therefore
**separate simulation runs** behind `--oracle=gated|scalarized`. Additionally, the scalarized
run records a **spectator gated label**: old-G_YTD is deterministic bookkeeping over the
realized trajectory (`spectatorG_YTD = seed + Σ realized P&L of this run's harvests`, re-seeded
at year-end), so the old 4-gate predicate can be evaluated pointwise on identical rows —
boundary-geometry comparison for free, clearly labeled as spectator ≠ acting.

### 5.2 Feature space (`LotStateVector`) — SHIPPED in step 1 (schema v3, d = 17)

```csharp
// ── portfolio-level (shared state 𝒮_t) ── REPLACED the single G_YTD scalar
public float RealizedGainsYTD     { get; init; }
public float LossCarryforward     { get; init; }
public float OrdinaryOffsetBudget { get; init; }
// ── lot-level derived (joins ledger with h_k, ℓ_k) ── replaced TaxAlpha
public float TaxValue { get; init; }   // = g(ledger_t, h_k, ℓ_k)
// unchanged
public float Sigma_TE  { get; init; }
public int   WashClock { get; init; }
// new label
public float Y_TaxValue { get; init; } // ≡ TaxValue in v0.25 → regressions on it
                                       // must EXCLUDE the TaxValue feature
```

- **Decision (ratified): expose the raw three ledger fields, not only the collapsed `TaxValue`.**
  Rationale: downstream consumers (v0.4 RL reward, issue #12 metrics) need the decomposition
  of "immediately usable" vs "banked"; collapsing prematurely hides it. The schema bump
  (15 → 17 features, 21 → 24 columns) landed as one version bump across `SimulationExporter`,
  `LotStateVectorCsvReader`, `FeatureLists`, `MLReadyRow`/`MedianImputer`/`ClassWeights`/
  `PcaPipeline`, `codebook_schema.py`, `report_helpers.py`, `eda.py`, and the codebook —
  enforced by the header-drift assert in `scripts/codebook.py` and `tests/test_codebook_schema.py`.
- Gate-removal ≠ dimension-removal: dropping `G_YTD` as a *gate* (label construction, step 2)
  is independent of the ledger/`TaxValue` columns as *features* (schema decision, step 1 —
  done). The two decisions stayed explicit and separate.
- `c_trade`: config constant, not a feature (§2.4).
- Leakage rule unchanged: every feature stays σ(𝓕ₜ)-measurable; only labels may peek forward.

### 5.3 Call-site coupling (the one non-black-box edit — step 2)

`SimulationEngine.Run()` currently extracts four named scalars into
`OracleBoundary.Label(unrealizedReturn, sigmaTE, gYtd, washClock)`. `U(x)` additionally needs
`τ(h)`, `offsetCapacityₜ`, and λ. Bounded edit: absorb the churn in the existing
`Label(LotStateVector snapshot)` convenience overload — after step 1 the snapshot already
carries the ledger fields and `H`, so only that overload's body changes; all callers routed
through it stay untouched. Plus an `OracleConfig` record (θ₁, θ_max, λ, c_trade, τ rates, δ,
mode, seed setting) replacing the compile-time consts.

---

## 6. Predicted model impacts (grounded in measured numbers)

Anchor: LR oracle PR-AUC 0.987 (2y, gate never binds) → ~0.12 (20y, binds ~6%);
GBT 0.9997 / RF 0.9987 on oracle; GBT 0.858–0.862 / LR ~0.65 on soft (20y).

**Ablation I — drop the gate, nothing else (Option A of v1, kept as ablation only).**
Confident prediction: the 20-year finding *reverses toward the 2-year one*. The 3-gate
conjunction collapses back toward the `ℓ ≤ −θ₁` half-space; LR oracle PR-AUC climbs back
toward ~0.98; GBT/RF stay at ceiling; **the GBT–LR gap on the oracle target nearly vanishes**
— the disqualifying con already flagged in v1, now with before/after numbers in-repo.

**Ablation II — full composite (`oracle_scalarized`).** Genuinely uncertain; do not
oversell. `U(x)` reintroduces non-linearity as a *quadratic* level curve in `(ℓ, σ_TE)`
rather than a box corner. LR should not collapse as in Ablation I, but its recovery depends
on (a) λ relative to the empirically realized σ_TE range (a mild curve over a narrow
operating band is quasi-linear in practice) and (b) whether `Sigma_TE²` is offered as an
engineered feature. Empirical question → its own ablation column, not an armchair claim.
Honest framing for the report: a smooth economic boundary partially closing the GBT–LR gap
is the *more realistic* result, even if it softens the v0.2 headline.

**Dataset-size effects (endogenous N).** `N_rows = n₀T − Σ_harvests min(30, T_end − t_h) − ε`:
looser gating ⇒ more harvests ⇒ more 30-day dark windows ⇒ **fewer rows**. Magnitude caveat:
the old gate failing on 6% of rows converts to new harvests only where the *other* gates
simultaneously hold — a joint condition, much rarer than 6%; measure by rerun, don't assert.
Separate two efficiencies: fewer rows = cheaper compute (real), but **not** free statistical
efficiency — the minority class shrinks too, eroding exactly the resolution that let boosting
sharpen the thin positive region RF blurs. A tradeoff to measure. (Step-1 datum: the full
20y simulate + soft-label pass runs in ~84s, so rerun cost is a non-issue.)

**Positive-rate drift**: which lots clear changes ⇒ re-validate stratified CV and class
weights against the new prevalence, not just re-run.

---

## 7. Sequencing — ship as independent ablations

Attribution discipline: never change two boundary-shaping components in one PR.

1. **Ledger (loss-only) + `Y_TaxValue`** — ✅ SHIPPED (`feature/v025-pr1-taxledger`).
   Zero new stochastic surface; `ComputeTaxAlpha` folded into `taxValueₖ`; labels verified
   byte-identical to the v0.2 baseline (1.85M rows).
2. **Oracle swap + TE continuization** (`--oracle=gated|scalarized`, gates·𝟙[U>0],
   `−λσ_TE²` in, θ₂ → loose θ_max, spectator gated label). Own ablation runs
   (`oracle_gated` vs `oracle_scalarized`). Re-measure.
3. **`c_trade`** — additive to the now-continuous `U(x)`; low-risk layer-on.
4. **§4 trim process** — makes `RealizedGainsYTD` endogenous; dual-purpose v0.4 scaffolding.
   DEFERRED past v0.25.
5. **Feature-exposure decision (§5.2)** — ✅ folded into step 1 with the schema version bump.

Non-goals recorded: corporate/bounded-carryforward ledger variants (§1.5), ST/LT typed pools
(§3.3 simplification note), lot-size-dependent `c_trade` (§2.4) — each a bounded future
extension, none required for the individual-DI oracle this project models.

---

## 8. Bridge to v0.4 (unchanged in spirit from v1 §4, sharpened)

Once `U(x)` exists, the supervised `η̂(x)` is the RL **value-function warm-start**; the RL
reward is precisely `τ(h)·taxValue(ledger) − λσ_TE²` accumulated over the episode — i.e. the
ledger built here IS the reward-accounting substrate, deterministic under the learned policy.
The reason supervised learning cannot be the endpoint is already structural: offset capacity
and the TE budget are **shared, depletable resources**, making lot decisions conditionally
dependent and breaking the i.i.d.-row assumption — the FQI/`IHarvestPolicy` motivation.

---

*Cross-references: `SimulationMath.md` §2 (seeding + year-end reset — to be rewritten at
step 2), `PortfolioMath.md` §2.3 (G_YTD sign convention — superseded by TaxLedger),
`Lifecycle_v02.md` §3.3 (endogenous N_rows), §3.4 (oracle statelessness), §4.4 (tier table),
§8.IV; `data_memo_theory_part2.md` §5.A (level sets), §A.5 (0.987→0.12); `MLNetLeakageAudit.md`
(schema invariants); `PSTAT231_RECAP.md` (roadmap rows v0.25/v0.3/v0.4); issues #12 (tax-alpha
metrics), #15 (policy interface), #17 (soft-label families), #22 (deployment metrics),
#23 (this redesign).*

*External grounding: Wealthfront stock-level TLH whitepaper
(research.wealthfront.com/whitepapers/stock-level-tax-loss-harvesting/); Betterment TLH
methodology; 26 USC §1091, §1211(b), §1212; IRC §1014; Rev. Rul. 74-175.*

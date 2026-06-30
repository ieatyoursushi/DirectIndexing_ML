# G_YTD Redesign Plan — promoting the gains gate from a binary guard to a tax ledger

> **Status:** design plan for a *future* implementation (target ≥ v0.3). Nothing here is
> implemented yet. The current simulation keeps the binary `G_YTD > 0` gate with a constant
> 10% ($1M) annual seed; the v0.2 report (`final_report.ipynb`) discusses why that gate has
> become near-vestigial over the 20-year window and points here for the fix.
>
> **Why this document exists.** The 20-year backtest (1.85M rows, 2004–2024) showed
> `G_YTD > 0` holding on ~94% of rows — the 4th oracle gate almost never binds. That is the
> empirical trigger for revisiting the gate. This plan records the motivation and three
> concrete redesign options so a future instance can implement the change and re-simulate
> without re-deriving the reasoning.

---

## 1. What the gate encodes today, and why it is now weak

**Current gate (`OracleBoundary.Label`):** harvest requires `G_YTD > 0`, i.e. there are net
realized gains *this calendar year* to offset the harvested loss against. `G_YTD` is seeded
at `0.10 · V₀ = $1,000,000` at simulation start and re-seeded to the same constant after each
year-end reset (`SimulationEngine.InitializePortfolio` / the year-end branch).

Three problems, in increasing order of severity:

1. **It is near-vestigial over long horizons.** With a constant positive re-seed and a
   simulation that only ever realizes *losses* (harvests) — never gains — `G_YTD` sits at
   `+$1M` minus accumulated harvest losses, which over a mostly-bull 20-year window stays
   positive ~94% of the time. A gate that is open 94% of the time is not shaping the decision
   boundary; it is noise in the conjunction.

2. **The constant seed does not scale or respond to regime.** `$1M/yr` is fixed while the
   portfolio value `Vₜ` grows several-fold over 20 years, so the seed becomes a shrinking
   fraction of the book. It also does not move with the market — a real client realizes *more*
   gains in bull years (rebalancing, distributions, trims) and *fewer* in bear years, exactly
   when harvest opportunities are richest. A constant seed inverts that coupling.

3. **It is stricter than the actual tax code, which costs tax alpha.** US tax lets capital
   losses (a) offset realized capital gains, (b) offset up to **$3,000/yr of ordinary income**,
   and (c) **carry forward indefinitely**. So a realized loss is *bankable even with zero
   current-year gains* — it simply waits for a future gain (or the annual $3k). The strict
   `G_YTD > 0` gate makes the engine **refuse valuable losses** whenever the current year's
   gains are spent, even though those losses would carry forward. In a live system that is tax
   alpha left on the table — a business cost, not just a modeling simplification.

**Counter-point (why it was still a good v0.2 choice):** the gate made the oracle a genuine
4-way conjunction, which (i) produced the project's cleanest finding — the decision geometry
*collapses toward a single half-space when a gate stops binding*, demonstrated by logistic
regression nearly solving the oracle on the 2-year run (CV 0.987) and *failing* on the 20-year
run (CV ~0.12) — and (ii) created a self-limiting harvest dynamic. The redesign should
**preserve the modeling richness while fixing the economics.**

---

## 2. Three redesign options

### Option A — Remove the gate (3-gate oracle)

Drop `G_YTD > 0` entirely; harvest any lot clearing the loss / tracking-error / wash-sale
gates. Closest to "losses always carry forward."

- **Pros:** simplest; matches the carryforward intuition; maximizes harvest count.
- **Cons:** makes the supervised target *more* predictable (the user's stated worry — fewer
  interacting gates ⇒ a more axis-aligned boundary ⇒ linear models recover more of it), and
  throws away the realized-gains accounting entirely, which a real tax engine needs. **Not
  recommended as the endpoint**, though useful as an ablation to quantify how much the gate
  was contributing.

### Option B — Dynamic seed (keep the gate, fix the number)

Keep the binary gate but replace the constant `$1M` with a seed that scales and responds:

- scale with the book: `seed_t = κ · V_t` (e.g. κ = 0.10) recomputed at each year-end so the
  gains budget tracks portfolio size;
- optionally couple to regime: a larger realized-gains assumption after up-years, smaller
  after down-years (e.g. proportional to trailing benchmark return, floored at 0).

- **Pros:** small, local change to `SimulationEngine`; removes the "shrinking fraction"
  defect; keeps the conjunction intact.
- **Cons:** still a *modeled* external-gains process rather than the portfolio's own accounting;
  still a binary gate, so still stricter than carryforward reality. A reasonable interim step.

### Option C — Continuous tax ledger (recommended)

**Promote `G_YTD` from a binary gate to a continuous tax-accounting state**, and make the
*value of harvesting* — not a 0/1 condition — the modeling target.

Maintain a running ledger per simulated year and across years:

```
realized_gains_YTD      (from a modeled gains-realization process, see §3)
loss_carryforward       (cumulative unused harvested losses, carried across year-end)
ordinary_offset_budget  ($3,000/yr capacity)
```

The **economic value** of harvesting lot k today becomes a continuous quantity:

  taxValueₖ = τ(hₖ) · min(|lossₖ|, offsetCapacityₜ)
              + τ_future · max(|lossₖ| − offsetCapacityₜ, 0) · discount

where `offsetCapacityₜ = realized_gains_YTD + ordinary_offset_budget`, `τ(h)` is the
short/long-term rate (already in `ComputeTaxAlpha`), and the second term prices the
carryforward portion at a discounted future rate. The oracle's binary 4th gate is replaced by
"`taxValueₖ > threshold`", and — more importantly — this same `taxValueₖ` becomes a richer
**continuous regression target** alongside the existing soft labels.

- **Pros:** simultaneously *more correct* (matches carryforward + $3k + offsetting) and *more
  interesting* (predicting continuous tax alpha is a harder, more informative ML problem than
  predicting a binary gate — it **resolves the "removing G_YTD makes it less interesting"
  worry by making the target continuous rather than removing it**). Reuses the existing
  `TaxAlpha` feature and aligns with issue #12 (portfolio tax-alpha metrics).
- **Cons:** largest change; requires modeling a gains-realization process (§3) and a new
  label family in `SoftLabelBuilder`.

**Recommendation:** ship **B as an interim** (cheap, removes the worst defect) and treat **C
as the v0.3 target** (the real fix, and the natural bridge to the RL layer).

---

## 3. The simulation change every option except A implies

The deeper issue: **the simulator only realizes losses (harvests); it never sells winners.**
That is *why* a seed is needed at all — without an external gains source, `G_YTD` could only
ever go negative. Any "full portfolio-history gain tracking" therefore requires adding a
**gains-realization process**:

- periodic rebalancing / trimming of overweight winners back toward index weight (this also
  improves the tracking-error story — it is the natural counterpart to harvest-and-reopen);
- optional dividend / distribution modeling.

This is a `SimulationEngine` extension (a "sell winner" transition mirroring `HarvestLot`,
feeding `realized_gains_YTD`), not just an `OracleBoundary` tweak. It is the prerequisite for
Option C's ledger to be endogenous rather than seeded.

---

## 4. Beyond v0.3 — soft boundaries and reinforcement learning

The author's note (negative reinforcement / soft boundaries instead of a hard gate) lands
here:

- A **hard gate** is a thresholded indicator; a **soft boundary** replaces it with a smooth
  penalty (e.g. a logistic ramp on remaining offset capacity, or a cost term that grows as the
  carryforward balance deepens). This makes the decision differentiable and removes the brittle
  0/1 cliff — useful for any gradient-based policy.
- The natural endpoint is the **RL policy layer (v0.4–v0.5)**: an agent needs *no* hand-coded
  gate at all. The tax ledger of §2.C and the tracking-error budget become the **reward**
  (realized after-tax alpha minus a tracking-error penalty), and the agent learns the harvest
  *policy* directly. Negative reinforcement = penalize harvests that breach the TE budget or
  bank losses with no foreseeable offset. The oracle / soft labels were always scaffolding to
  manufacture supervised targets *before* this layer exists; once the ledger is continuous
  (Option C), the supervised model `η̂(x)` becomes a value-function warm-start for the RL agent.

---

## 5. Implementation checklist (for a future instance)

Interim (Option B):
- [ ] `SimulationEngine`: replace `_seedAmount = totalValue * 0.10m` (fixed) with a year-end
      recompute `seed = κ · currentPortfolioValue`; thread `κ` as a parameter.
- [ ] Re-run `simulate` → `mlnet-all`; confirm `G_YTD > 0` prevalence drops below ~94% and the
      gate starts binding again; re-narrate the report's G_YTD section.

Target (Option C):
- [ ] Add a `TaxLedger` type (realized_gains_YTD, loss_carryforward, ordinary_offset_budget)
      to `Core/Portfolio`, evolving across `AdvanceDay` / `ResetForNewYear` (carryforward must
      survive year-end; only `realized_gains_YTD` and the $3k budget reset).
- [ ] Add a gains-realization transition to `SimulationEngine` (§3) feeding `realized_gains_YTD`.
- [ ] Replace the binary 4th gate in `OracleBoundary` with `taxValueₖ > θ₄`, and add a
      continuous `Y_TaxValue` label to `SoftLabelBuilder` + the `LotStateVector` schema +
      `SimulationExporter` header + `codebook_schema.py` (header-drift assert will force this).
- [ ] Decide whether `TaxLedger` state (offset capacity, carryforward) becomes *features*
      (recommended — keeps the conditional-independence argument intact, like `G_YTD` today).
- [ ] Re-run the pipeline; the report gains a continuous-tax-alpha regression alongside the
      binary champions.

Ablation (Option A, optional):
- [ ] A `--no-gains-gate` flag on the oracle to quantify the gate's contribution to the
      decision geometry (expected: linear models recover more of the oracle when removed).

---

*Cross-references: `SimulationMath.md` §2 (seeding + year-end reset), `PortfolioMath.md` §2.3
(G_YTD sign convention), `Lifecycle_v02.md` §3.4/§8.IV, issue #12 (portfolio tax-alpha
metrics), issue #15 (RL simulation layer).*

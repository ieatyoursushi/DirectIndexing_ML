# Roadmap — Direct Indexing ML

> **The authoritative version planner.** This file supersedes the roadmap sections scattered
> across `README.md` §Roadmap, `PSTAT231_RECAP.md` §7–§9, and the working plans in
> `DataMemo/archive/architecture_thread/` (the GPT-thread architecture review and its contextualized parity
> assessment, from which the v0.25→v1.0 spine below was synthesized). When this file and an
> older doc disagree, this file wins. Deep rationale per stage lives in the linked memos.
>
> Last updated: **2026-09-17** (v0.26 merged; v0.3 P0 shipped — contributions + reopen wash-sale fix).

---

## Status at a glance

| Version | Name | Status | Where |
|---|---|---|---|
| v0.1 | Supervised baseline (ML.NET rewrite) | ✅ complete | issue #10 |
| v0.2 | PSTAT 231 submission — champions, PCA/K-means, report layer | ✅ complete | PR #11 |
| — | 20-year scale-up (`download --from/--to`, 1.85M rows, report rewrite) | ✅ complete | commit `c421c4f`, issues #19/#20 |
| v0.25 | Oracle redesign — `TaxLedger`, scalarized `f*`, ablation arms | ✅ complete | PRs #24–#27, issue #23, `DataMemo/decisions/GYTD_Redesign_Plan.md` |
| v0.26 | Validation hardening — purged chronological splits | ✅ complete | PR #30, `DataMemo/decisions/ValidationHardening_v026.md` |
| **v0.3** | **Simulator realism + economic metric layer** | **⏭ next (P0: cost-basis-aging fix)** | issues #12/#17/#22 |
| v0.35 | Universe & replacement layer (core+reserve, SubScore) | planned | issues #5/#6 |
| v0.4a | Constrained execution baseline (optimizer, no RL) | planned | — |
| v0.4b | RL policy layer | planned | issue #15 |
| v0.5 | End-to-end evaluation + integration | planned | issues #16/#21 |
| v1.0 | Deployment (RIA-style service, author as first client) | planned | — |

**Headline state after v0.26:** champion ranking skill is real and transfers across time
(ROC-AUC held under purged temporal splits and a decade walk-forward; leakage ruled out via
the deterministic-oracle control). The PR-AUC drop under temporal splits (soft 0.81 → 0.46)
is the **cost-basis-aging prevalence crash** (soft+ 3.93% → 0.22% in the aged tail), which is
exactly the v0.3 P0.

## Standing rules (apply to every version)

1. **One boundary-shaping change per PR.** Mechanical renames ride along; semantics don't.
2. **Ablation arms for anything that changes labels or trajectories** (the v0.25
   gated/scalarized pattern: separate runs + spectator label).
3. **Schema changes take a version bump** through the codebook assert (`LotStateVector` is v3).
4. **Economic claims cite the baseline ladder** (once built, v0.3 item 3), not classification
   metrics alone.
5. **Since v0.26:** report **ROC-AUC + PR-AUC + test-period positive rate together** — never
   PR-AUC alone — and prefer `--split=temporal` for any forward-deployment claim. Random
   splits remain default only to reproduce historical v0.25 numbers.
6. **Labels may peek at the future; features never.** (The founding invariant.)
7. Don't rewrite the theory docs — extend along the seams.

---

## ✅ v0.25 — Oracle redesign (complete, July 2026)

`TaxLedger` replaced the `G_YTD` scalar (schema v3, d=17: `RealizedGainsYTD`,
`LossCarryforward`, `OrdinaryOffsetBudget`; capacity-aware `TaxValue`; labels byte-identical
landing). Scalarized oracle `f* = 𝟙[ℓ≤−θ₁]·𝟙[𝒲≥30]·𝟙[σ_TE≤θ_max]·𝟙[U>0]` with
`U = TaxValue − λσ_TE² − c_trade` is canonical; `--oracle=gated` reruns the v0.2 rule as the
ablation baseline. Measured (design doc §6.1): oracle-target GBT−logistic gap 0.155 → 0.015;
soft-target tree advantage oracle-invariant; value-regression R² 0.10 linear vs 0.92 trees;
carryforward load-bearing ($4.3M/20y). → `DataMemo/decisions/GYTD_Redesign_Plan.md`

## ✅ v0.26 — Validation hardening (complete, July 2026, PR #30)

**Built:** `Splits/TemporalSplit.cs` (chronological `TrainTest` with purge/embargo ≥ the
30-day label horizon; `PurgedFolds` purged k-fold), `SplitPolicy.cs`
(`--split=temporal --embargo=N --testfrac=F`; default stays stratified-random →
reproduces v0.25 bit-for-bit; temporal artifacts → `data/artifacts-mlnet-temporal/`),
`DataSplit.cs` facade (partition semantics in one place, 14 call sites), 5 tests.

**Found:** ROC-AUC held/rose under temporal splits while PR-AUC collapsed; the oracle target
(deterministic in current features) is the leakage control and stayed ~1.0 across all splits
→ **no material ranking leakage in v0.1–v0.25**. The decade walk-forward (train 2006–16,
test 2016–26) isolated mild honest drift (soft ROC 0.997 → 0.961). In lift over no-skill the
temporal model is *better* (~25× → ~210×). Gate: **pass, with a mandate** (standing rule 5).
→ `DataMemo/decisions/ValidationHardening_v026.md`

**Carried forward (planned in the v0.26 spec, consciously deferred):**
- [ ] Gated-arm re-validation under temporal splits (the full 2×2 was cut to the canonical
      scalarized arm; the gated oracle is a deprecated ablation baseline, so re-run it
      temporally only if a v0.3+ claim ever leans on gated-arm numbers).
- [ ] Explicit class-weight / imbalance-handling re-check per split scheme (prevalence was
      *measured and diagnosed*, but no re-weighting experiment was run; fold-imbalance
      handling matters more once v0.3 restores prevalence).
- [ ] Fold the random-vs-temporal delta table into the executed report notebook as an
      appendix (currently lives only in the DataMemo memo).

## ⏭ v0.3 — Simulator realism + economic metric layer (next)

**Goal:** make the book behave like an account a live client would recognize, and make
evaluation economic.

1. **P0 — cost-basis-aging fix:** ✅ **shipped** (`ContributionPolicy`, `--contrib`). Periodic
   cash inflows mint fresh lots at current prices in the most *underweight* wash-sale-eligible
   names (bounded lot growth; new cash corrects drift, as a real manager does). Off by default
   — an unflagged run reproduces v0.26 byte-for-byte.
   **Measured** (20y, scalarized, temporal test region): temporal-test `oracle+`
   0.0119% → 0.1039% (**8.7×**), `soft+` 0.2205% → 4.0051% (**18.2×**); final-year mean lot age
   15.10y → 9.20y; final-year ≥2%-underwater 0.062% → 2.413% (**39×**). Rows 1.85M → 5.18M.
   *Also fixed a latent wash-sale bug it exposed:* reopens were scheduled for `harvest_day+30`
   and never re-checked the clock — safe only while tickers held one lot at a time. With
   multi-lot tickers, a same-day re-harvest reset the clock and the reopen bought into a fresh
   window (17.4% of run-opened lots). Reopens now defer until the window truly clears → 0
   genuine violations; inert on single-lot books.
   **Remaining for the gate:** retrain under `--split=temporal` on `lots_contrib.csv` to
   quantify the PR-AUC recovery (ROC-AUC was never the problem).
2. **Sell-winner trim process** (`SimulationEngine` extension mirroring `HarvestLot`), own PR
   behind a flag: realizes gains → `RealizedGainsYTD` endogenous → carryforward netting
   exercised; also the first non-harvest action (v0.4b scaffolding).
3. **Tax-alpha & deployment metric layer (#12/#22):** the six-rung baseline ladder, after-tax
   wealth, realized/deferred benefit decomposition off the ledger, TE/turnover/cost per run.
   From here on, version gates cite ladder rungs.
4. **Richer soft-label families (#17):** horizon variants and max-vs-mean aggregations of
   `U`/`Y_TaxValue` as label columns.
5. **Volatility/uncertainty sub-model — design in full, build behind a gate:** design the
   `history → (σ̂, Σ̂)` estimator (EWMA/GARCH + Ledoit-Wolf/RMT, #5) unconditionally; build
   only if the decision-flip ablation (fraction of oracle/U/Y_Soft_GBM outcomes that flip)
   is material; otherwise record the negative result and revisit at v0.4b.
6. Optional: typed ST/LT ledger pools (law-exact Schedule D).

**Gate:** aging fixed (harvest signal persists across decades — temporal-split PR-AUC
recovers), ledger fully exercised (carryforward absorbs endogenous gains), metric harness
runs ladder rungs 1–3. **Validate under `--split=temporal` per standing rule 5.**

## v0.35 — Universe & replacement layer

Core+reserve construction and the substitute path. Covariance cleaning productionized (#5);
offline core-basket optimizer (min TE + complexity − dispersion potential, s.t.
budget/cardinality/sector/liquidity; sweep |C| ≈ 75–200); replacement module (eligibility
rules → deterministic `SubScore` → explicit fallback policy; keep same-ticker reopen as the
ablation baseline); hybrid-environment re-simulation + champion transfer test (mandatory —
v0.25 proved trajectories are oracle- and environment-dependent); flag the survivorship
interaction (point-in-time membership is the v1.0 fix).
**Gate:** hybrid book tracks benchmark within budget at materially lower name count while
preserving harvestable dispersion (ladder rungs 2–4 in the hybrid environment).

## v0.4a — Constrained execution baseline (no RL)

Daily loop: champion soft scores rank candidates → deterministic subset optimizer selects the
feasible action set under shared TE/turnover/capacity budgets → replacement optimizer
executes substitutes → ledger/TE transitions. All fed predictions produced walk-forward.
**Gate (the RL go/no-go):** ladder rung 5 vs rung 4 — if the optimizer captures nearly all
the gap to the oracle-driven ceiling, v0.4b shrinks or waits; the residual is RL's earnable
surplus, now quantified.

## v0.4b — RL policy layer (#15)

`IHarvestPolicy` with the oracle as greedy baseline; **low-dimensional action space**
(thresholds/budgets/top-m, not per-lot binaries); state = ledger + TE + factor/candidate/wash
summaries (+ σ̂ if available — re-test the uncertainty axis here regardless of the v0.3
gate); reward = accumulated `Y_Utility` (already exported); warm start from the supervised
η̂ and the value regression; PPO/SAC; `MonteCarloEngine` as episode generator before
real-history fine-tuning.
**Gate:** ladder rung 6 vs rung 5 under walk-forward evaluation, stability across regimes,
alpha per unit TE/turnover.

## v0.5 — End-to-end evaluation + integration

Walk-forward economic comparison across the full ladder; live/recent data; client
parameterization (hazard-rate δ, outside-gains personas, goal-conditioned discounts — the
`GYTD_Redesign_Plan.md` §1.5/§3.3 hooks); knowledge distillation of the champion stack into
a deployable student (#16); environment reproducibility (#21).

## v1.0 — Deployment

RIA-style service, author as first client. Point-in-time constituent feed (kills the
survivorship caveat), auditable tax/compliance module boundary (`TaxLedger` already is one),
monitoring, standing guardrails as operating policy.

---

## Document map

| Doc | Role |
|---|---|
| `ROADMAP.md` (this file) | Authoritative version planner + status |
| `PSTAT231_RECAP.md` | Orientation: what the project is, layer by layer, and why |
| `README.md` | Commands, schema, results snapshot |
| `DataMemo/decisions/GYTD_Redesign_Plan.md` | v0.25 design + measured ablations |
| `DataMemo/decisions/ValidationHardening_v026.md` | v0.26 method + leakage-vs-regime diagnosis |
| `DataMemo/archive/Lifecycle_v02.md` | First-principles codebase walk (frozen at v0.2) |
| `DataMemo/archive/data_memo_theory.md` / `_part2.md` | Pre-plan theory + post-course reconciliation |
| `DataMemo/archive/architecture_thread/…contextualized.md` | Working parity review this roadmap was distilled from |

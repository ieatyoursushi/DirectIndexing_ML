# Roadmap — Direct Indexing ML

> **The authoritative version planner.** It supersedes the roadmap sections of `README.md`
> and of the archived `DataMemo/archive/PSTAT231_RECAP.md` and architecture thread. When this
> file and an older doc disagree, this file wins. The mathematics lives in
> [`DataMemo/spec/`](DataMemo/spec/SymbolTable.md) (live, machine-checked), and the rationale
> behind each version in [`DataMemo/decisions/`](DataMemo/README.md).
>
> Last updated: **2026-09-30**. The pre-v0.3 downsizing is complete, and **v0.3 is in
> progress**: P0 and v0.3-1 (the two-sided §1091 wash window) have shipped, and v0.3-2 (the
> §1222 calendar holding period) is next.

---

## Status at a glance

| Version | Name | Status | Where |
|---|---|---|---|
| v0.1 | Supervised baseline (ML.NET rewrite) | ✅ complete | issue #10 |
| v0.2 | PSTAT 231 submission: champions, PCA/K-means, report layer | ✅ complete (frozen) | PR #11 |
| — | 20-year scale-up (`download --from/--to`, 1.85M rows) | ✅ complete | commit `c421c4f`, issues #19/#20 |
| v0.25 | Oracle redesign: `TaxLedger`, scalarized `f*` | ✅ complete | PRs #24–#27, [`decisions/GYTD_Redesign_Plan.md`](DataMemo/decisions/GYTD_Redesign_Plan.md) |
| v0.26 | Validation hardening: purged chronological splits | ✅ complete | PR #30, [`decisions/ValidationHardening_v026.md`](DataMemo/decisions/ValidationHardening_v026.md) |
| — | **Pre-v0.3 downsizing + the math↔code spine** | ✅ complete | PR #31, [`archive/RetiredComponents.md`](DataMemo/archive/RetiredComponents.md), [`spec/SymbolTable.md`](DataMemo/spec/SymbolTable.md) |
| **v0.3** | **Simulator correctness + volatility sub-model + metric ladder** | **⏭ in progress** (P0 ✅, v0.3-1 ✅; v0.3-2 next) | issues #5/#12/#17/#22 |
| v0.4a | Constrained execution baseline (optimizer, no RL) | planned | — |
| v0.4b | RL policy layer | planned | issue #15 |
| v0.45 | Universe & replacement layer (core+reserve, `SubScore`, PCA on $\hat\Sigma_t$) | planned; **moved after RL** | issues #5/#6 |
| v0.5 | End-to-end evaluation + integration | planned | issues #16/#21 |
| v1.0 | Deployment (RIA-style service, author as first client) | planned | — |

**Critical path to the RL layer:** v0.3 → v0.4a → v0.4b. The universe layer (formerly v0.35)
is renumbered **v0.45** and follows RL, for two reasons. The agent's low-dimensional action
space does not grow with universe size. And both the v0.4a go/no-go gate and the v0.4b agent
run fine on the current full-universe book with same-ticker reopen. v0.45 is then evaluated
with the full ladder in hand.

## Audit findings (2026-09-30) — what the pre-v0.3 audit found, and where each is fixed

| # | Finding | Why it matters | Fixed in |
|---|---|---|---|
| **F1** | `TrackingErrorProxy` estimates $\hat\Sigma$ once from the **full** price history | $\hat\sigma_{\mathrm{TE}}$ is a feature and an input to $U$, so a 2008 row sees 2009–2026 returns. This breaks standing rule 6. The deterministic-oracle leakage control cannot see it: it tests splits, not feature construction. A point-in-time window with $L<N$ is also rank-deficient, which makes shrinkage *necessary* | ✅ **v0.3-6** (`--cov=fullsample` kept as the measurement arm) |
| **F2** | `MonteCarloEngine` duplicated the simulator's day loop and had drifted six ways | every simulator fix had to be made twice; the RL environment needs one engine | ✅ downsizing (one engine, two price sources) |
| **F3** | three inconsistent RL-reward definitions across the docs; the increment form **telescopes**, and a summed per-lot $U$ miscounts shared TE | an optimizer exploits exactly these | ✅ pinned in [`SymbolTable.md` §I](DataMemo/spec/SymbolTable.md) (running-cost form, derived) |
| **F4** | symbol overloading (λ, δ, W, τ, σ, H/L, S, U, V, D, γ, q, θ, "GBM", "days") | the main source of the cognitive load, and the root cause of F6/F7 | ✅ notation contract, [`SymbolTable.md` §J](DataMemo/spec/SymbolTable.md) |
| **F5** | docs had drifted from code and from each other (e.g. a cited test that never existed; two R² values) | stale specs are worse than none | ✅ `docs-check` + recorded-drift table |
| **F6** | holding period counts **trading** days but is compared with **365**, so "long-term" ≈ 1.45 calendar years (§1222) | lots held 1.0–1.45y got τ = 0.37 instead of 0.20; the `S` feature was wrong | ✅ **v0.3-2**: `TaxLedger.IsLongTerm` (calendar). Measured (60-name GBM world, 1260 d, seed 7): `S` flips on 4,913 / 48,551 rows (10.1%), yet **0** TaxValue/label changes — offset capacity is 0 on 98.3% of rows in a loss-only book, and τ only multiplies the capacity slice. The fix becomes economically visible once gains exist (v0.3-3 ledger, v0.3-6 trim) |
| **F7** | §1091 enforced one-sided: no buy *after* a loss sale, but nothing blocks a **harvest after a recent buy**; the window is 30 **trading** days | contributions buy the most-underweight (fallen) names, exactly the ones about to be harvested. Measured by an independent audit on a weekday-calendar synthetic world: **24.3% of the contrib arm's loss sales were wash sales** | ✅ **v0.3-1**: 0 violations after the fix |
| **F8** | `LossCarryforward` was banked but **never consumed**, and the $3k ordinary line was valued at full rate even when carryforward already claimed it; ST and LT shared one blended pool, so a loss earned its own τ instead of the rate of what it offset | new-loss value overstated in years with carryforward; character netting absent | ✅ **v0.3-3** (Schedule D ledger) |

## Standing rules (apply to every version)

1. **One boundary-shaping change per PR.** Mechanical renames ride along; semantics don't.
2. **Ablation arms for anything that changes labels or trajectories**: separate simulate runs
   (the acting policy changes which rows exist), plus a *spectator* re-evaluation on the
   baseline's rows for row-aligned flip rates. `--lots=` plus the derived artifact directories
   keep arms apart.
3. **Schema changes take a version bump** through the codebook assert (`LotStateVector` is
   **v4**: 25 columns, d = 17).
4. **Economic claims cite the baseline ladder** (from v0.3-11), not classification metrics
   alone.
5. **Report ROC-AUC + PR-AUC + test-period positive rate together**, never PR-AUC alone. Prefer
   `--split=temporal` for any forward-deployment claim.
6. **Labels may peek at the future; features never.** (The founding invariant. F1 is a live
   violation until v0.3-6.)
7. Don't rewrite theory docs; extend along the seams. Archive docs get banners, never edits.
8. **The spine stays in sync:** a PR that changes an `[math:*]`-anchored member updates its row
   in [`DataMemo/spec/SymbolTable.md`](DataMemo/spec/SymbolTable.md) in the same PR, and
   `dotnet run --project src -- docs-check` passes. Every type states its units (**d_trd** vs
   **d_cal**, $, annualized).

---

## ✅ v0.25 — Oracle redesign (complete, July 2026)

`TaxLedger` replaced the `G_YTD` scalar (schema v3, d = 17). The scalarized oracle
$f^*=\mathbf 1[\ell\le-\theta_1]\,\mathbf 1[\mathcal W\ge30]\,\mathbf 1[\sigma_{\mathrm{TE}}\le\theta_{\max}]\,\mathbf 1[U>0]$,
with $U=\mathrm{TaxValue}-\lambda_{\mathrm{TE}}\sigma_{\mathrm{TE}}^2-c_{\mathrm{trade}}$, is
canonical.

Measured (design doc §6.1):
- the oracle-target GBT−logistic gap fell 0.155 → 0.015;
- the soft-target tree advantage is oracle-invariant;
- value-regression R² was 0.10 linear vs ≈0.9 trees;
- carryforward is load-bearing ($4.3M over 20y).

The gated ablation arm was later retired (downsizing).

## ✅ v0.26 — Validation hardening (complete, July 2026, PR #30)

**Built:** `TemporalSplit` (chronological split with purge/embargo ≥ the 30-day label horizon;
purged k-fold), `SplitPolicy`, and the `DataSplit` facade.

**Found:** ROC-AUC held under temporal splits while PR-AUC collapsed. The deterministic oracle
target stayed ≈1.0, so there is **no split-level ranking leakage** (F1 is a *feature-level*
leak, a different object). The PR-AUC drop was the cost-basis-aging prevalence crash.

**Baseline the v0.3 gate compares against** (GBT, test set, soft target, 80/20 temporal):
ROC-AUC **0.9970**, PR-AUC **0.4585**, test prevalence **0.221%**.

**Carry-overs, resolved by the downsizing:**
- gated-arm temporal re-validation: *won't do* (arm retired; recoverable from the tag);
- class-weight re-check per split: folded into v0.3-3;
- random-vs-temporal delta table in the report: *closed* (the report is frozen; the table
  lives in the decision record).

## ✅ Pre-v0.3 downsizing + the math↔code spine (complete, 2026-09-30, PR #31)

**Criterion:** keep a component only if something on the v0.3 → v0.4b path consumes it. Its
*finding* is kept regardless, in [`archive/RetiredComponents.md`](DataMemo/archive/RetiredComponents.md);
its *machinery* is deleted and recoverable from the tag `archive/v0.3-pre-downsize`
(= `03f0c3e`).

**Retired:**
- RF, elastic net, the linreg demonstrator;
- feature-space PCA/K-means;
- the gated oracle arm, taking the schema to v4;
- the course report/submission layer;
- `MonteCarloEngine`'s duplicate loop, now `PriceLoader.FromGbm` plus the one engine.

The supervised layer is now **GBT** (champion, RL state input, fitted-Q substrate) and
**logistic** (the linear control that measures non-linearity). The grid shrank from 36 to 12
configs per target.

**Added:**
- `--lots=` with derived per-arm artifact directories;
- repo-root-anchored paths (`dotnet run --project src` from the root now works);
- `codebook` and `docs-check` commands;
- the three-tier DataMemo (`spec/` live, `decisions/` frozen, `archive/` history);
- [`SymbolTable.md`](DataMemo/spec/SymbolTable.md) with ≈40 `[math:id]` code anchors;
- exact-value PR-AUC/ROC tests;
- cross-language schema tests.

**Latent bugs fixed on the way:**
- `eda.py` had crashed since schema v3;
- render read the wrong directory under `--split=temporal`;
- `.gitignore` missed `lots_contrib.csv` and the arm artifact directories.

**Verified byte-identical** to `03f0c3e` on every remaining column, on a fixed-seed synthetic world.

---

## ⏭ v0.3 — Simulator correctness + volatility sub-model + metric ladder (in progress)

**Goal:**
- the book obeys the tax code it models (F6, F7) and its features are 𝓕_t-measurable (F1);
- the volatility sub-model is designed in full and built behind its gate;
- evaluation becomes economic;
- the policy seam the RL layer plugs into exists.

**Ordering principle:** anything that changes the simulator's physics lands before anything
that measures it.

### P0 — cost-basis-aging fix: ✅ shipped (`ContributionPolicy`, `--contrib`)

Periodic inflows mint fresh lots in the most underweight wash-eligible names. Off by default.
It also fixed a reopen wash-sale bug that multi-lot tickers exposed.

Measured (20y, temporal test region):
- `oracle+` 0.0119% → 0.1039% (8.7×);
- `soft+` 0.2205% → 4.0051% (18.2×);
- final-year ≥2%-underwater 0.062% → 2.413%.

⚠ **These figures predate the §1091 fix (F7)** and must be re-measured on the 20-year data. On
the fixed-seed weekday synthetic world, the fix lowered the contrib arm's `oracle+` rate by 18%
(0.890% → 0.728%): the removed harvests were the wash sales. Expect the real 8.7× to shrink
somewhat.

### The PR sequence

| PR | Content | Gate / measurement |
|---|---|---|
| ✅ **v0.3-1** §1091 two-sided window (F7) | calendar-dated state: a lot-level clock $\mathcal W_k$ = min(days since the ticker's last loss sale, days since a *different* open lot was acquired), with a strict gate $\mathcal W>30$ (the window is inclusive); `CanBuy` gates every buy; the reopen lands at sale + 31 calendar days; contributions skip names holding a harvestable lot (`--contrib-allow-harvestable` for the ablation). SymbolTable `wash_clock`, `can_buy`, `wash_audit` | **done:** the independent audit reads **0 violations** on worlds that had 24.3% (weekday, contrib) and 97.7% (daily calendar, reopen on day 30). Weekday world: baseline loss sales +15% (calendar window ≈ 21 trading days, not 30); contrib `oracle+` −18%. **Pending on your machine:** re-measure the P0 on the 20-year data |
| ✅ **v0.3-2** §1222 calendar holding period (F6) | `TaxLedger.IsLongTerm = date(t) > date(s_k) + 1yr`, one function used by `Lot`, the snapshot and the soft-label forward steps; `H` stays a trading-day feature | **done:** `S` flips on 10.1% of rows of the 60-name world, 0 TaxValue/label changes — capacity is 0 on 98.3% of rows of a loss-only book (visible once gains exist) |
| ✅ **v0.3-2b** re-harvest guard → ablation (Q1) | the "no second harvest of a ticker within 30 d of its own loss sale" term of $\mathcal W$ is stricter than §1091 (a loss sale is disallowed only by a replacement **acquisition**). `--no-reharvest-guard` drops it. Removing it exposed a real gap the guard had masked: a lot bought **and sold** inside the window is still a replacement for another lot's loss sale (Reg. 1.1091-1), so the before-side now also scans closed lots acquired within 30 d | **done:** guard-off audit = **0 violations**. 60-name world + `--contrib`: oracle+ 0.703% → 1.013%, harvests 1295 → 1884, rows with $\mathcal W\le30$ 62% → 41%; soft labels fall (BT 0.137 → 0.107) because more losses are realised. Without contributions (one lot per ticker) the arms are byte-identical. Default stays guarded until the P0 re-run decides |
| ✅ **v0.3-3** ledger completion (F8), schema v5 | `TaxLedger` becomes Schedule D: pools $(G^{ST},G^{LT})$ by §1222 character, carryforward $(C^{ST},C^{LT})$ by character, the year-end netting map $\mathcal S$ (`LedgerState.Close`: carryforward enters as a loss of its character → cross-netting → $3k ordinary deduction ST-first → character carryforward). TaxValue is the **counterfactual difference** $\Delta T+\tau_f\delta\,\Delta C'$, so a loss earns the rate of what it displaces. Schema v5: `NetST, NetLT, CarryST, CarryLT, OrdinaryOffsetBudget` replace `RealizedGainsYTD, LossCarryforward, OrdinaryOffsetBudget` (d 17 → 19) | **done:** hand-computed Schedule D tests (ST-first deduction, character carryforward, consumption, cross-netting, displaced-rate valuation, F8 crowd-out). 60-name `--contrib` world: 142 of 39,936 loss rows were overvalued by up to $810 (mean $280) in early-January rows with carryforward on the books; 0 oracle flips. Small in a loss-only book; the trim (v0.3-4) is what gives the pools gains to net |
| ✅ **v0.3-4** sell-winner trim | `TrimPolicy` behind `--trim` (`--trim-interval=`, `--trim-band=`): every 63 d, the ≤10 names above 1.5× equal weight sell whole **gain** lots, highest basis first, within their excess; proceeds reinvest through the contribution buy path (shared `BuyUnderweight`, contributions byte-identical) | **done:** test: only gain lots on trim days, 0 §1091 violations, carryforward consumed. 60-name `--contrib --trim` world: 127 gain lots, $359k realized; final carryforward $2.78M → $2.54M; oracle+ 0.703% → 0.669%. **Finding:** a pure-TLH book with no outside gains is *carryforward-saturated* (the $3k line is free on 0.04% of rows), so trimmed gains are absorbed by carryforward and the marginal harvest is valued at the banked rate $\tau_f\delta=0.10$ — $\delta$ is the dominant economic parameter of this client, which is the case for the v0.5 outside-gains personas |
| **v0.3-5** P0 close-out (your machine) | re-simulate baseline/contrib after v0.3-1…4; retrain GBT + logistic under `--split=temporal`; decide the `--contrib` and re-harvest-guard defaults | ROC + PR + test prevalence vs the v0.26 baseline (0.9970 / 0.4585 / 0.221%) |
| ✅ **v0.3-6** point-in-time $\hat\Sigma_t$ (F1) + dollar δw (Q2) | memo [`decisions/VolatilityModel_v03.md`](DataMemo/decisions/VolatilityModel_v03.md) written **first**. `ICovarianceEstimator`: `FullSampleCovariance` (legacy look-ahead arm), `PointInTimeCovariance` (window 252, refit 21 d) raw or Ledoit–Wolf (constant-correlation target, closed form). TE on **dollar** active weights. `RiskModel` default = `pit-lw` + dollars; `--cov=fullsample\|pit\|pit-lw`, `--te-weights=names\|dollars` (the legacy pair is byte-identical to pre-v0.3-6) | **done:** tests cover S singular → LW PD; ρ* → 1 for a true target, falling with T for a misspecified one; Σ̂_t unchanged by $r_{>t}$; dollar δw sees position size. 60-name `--contrib` GBM world, 2×2 arms: look-ahead alone ≈ 0 (median TE 0.0074 → 0.0073, stationary world); **dollar δw dominates**: median TE 2.5× (→ 0.018), U>0 among ℓ≤−2% rows 99% → 78%, oracle+ 0.703% → 0.619%. **Pending on your machine:** the F1 arm on 20-year data, where it should matter |
| **v0.3-7** σ̂ estimators + forecast evaluation | `IVolEstimator`: `Trailing21`, `Ewma(0.94)`, `Garch11` with walk-forward `Fit(window)` only; per-name $\hat\sigma_{i,t}$ and market $\hat\sigma_{m,t}$; `vol-eval` → QLIKE vs baselines, by regime | QLIKE of EWMA/GARCH beats trailing-21 |
| **v0.3-8** role 1: σ̂ as a feature (schema v6) | `SigmaHat`, `SigmaMkt`, `ZBarrier` $=d/(\hat\sigma\sqrt h)$, `PBarrier` $=2\Phi(-z)$ | ablations under `--split=temporal` only: with/without; logistic-closes-the-gap test; metrics stratified by $\hat\sigma_m$ tercile |
| **v0.3-9** role 2: σ̂ in labels | `Y_Soft_GBM` by filtered historical simulation (`--soft-gbm=gbm\|fhs`) | label shift; bias vs `Y_Soft_BT` within vol buckets |
| **v0.3-10** role 3: σ̂ in the environment | `PriceLoader.FromFhs` — date-block residual bootstrap along simulated σ paths (`simulate-mc --world=gbm\|fhs`) | clustering (ACF of \|r\| > 0), excess kurtosis, cross-correlation preserved, seed-deterministic |
| **v0.3-11** label families (#17) + policy seam + ladder | `Y_Soft_BT_90/180`, `Y_Persist`, `Y_TaxWeighted` (`--target=`); `IHarvestPolicy` (Oracle default, `NeverHarvest`, `ThresholdHarvest`), `RunMetrics` from the ledger + trade log, `ladder` command | rungs 1–3 run from one command; the RL seam |
| **v0.3-12** exit: RL readiness | SymbolTable §I finalized (state incl. σ̂ summaries, ledger pools, factor exposures; the pinned reward with TE from $\hat\Sigma_t$); `decisions/PolicyLayer_v04.md` | the v0.3 gate |

**v0.3 gate:**
- zero §1091 violations; holding periods on the calendar; Schedule D tests green;
- $\hat\Sigma_t$ is 𝓕_t-measurable, with its effect recorded; σ̂ QLIKE beats trailing-21;
- all three σ̂ roles measured as separate ablations;
- the trim exercises carryforward; ladder rungs 1–3 run;
- `docs-check` is green.

### Commands for the steps that need the real 20-year data (run on your machine)

```bash
# the cache: data/raw/ from `download --from 2006-07-01 --to 2026-06-12`
dotnet run --project src -- simulate                   # baseline arm   → data/lots.csv
dotnet run --project src -- simulate --contrib         # P0 arm         → data/lots_contrib.csv
# v0.3-1/2 effect sizes: rerun the two lines above after each fix and diff prevalence
dotnet run --project src -- mlnet-all --lots=data/lots_contrib.csv --split=temporal
#   → data/artifacts-mlnet_contrib-temporal/  (cite ROC-AUC, PR-AUC, test prevalence)
dotnet run --project src -- codebook --lots=data/lots_contrib.csv   # schema-drift assert
dotnet run --project src -- docs-check && dotnet run --project src -- test
```

## v0.4a — Constrained execution baseline (no RL)

Daily loop:
1. walk-forward GBT scores rank the gate-passing held lots;
2. a deterministic subset optimizer picks the feasible set under shared TE/turnover/capacity
   budgets;
3. same-ticker reopen stays the replacement rule until v0.45;
4. ledger and TE transition.

Predictions fed to the optimizer follow the fed-predictions rule (SymbolTable `fed_predictions`).

**Gate (the RL go/no-go):** ladder rung 5 vs rung 4. If the optimizer captures nearly all the
gap to the oracle-driven ceiling, v0.4b shrinks or waits. The residual (timing and deferral
value a one-step optimizer can't see) is RL's earnable surplus, now quantified.

## v0.4b — RL policy layer (#15)

The MDP is fully typed in [`SymbolTable.md` §I](DataMemo/spec/SymbolTable.md).
- **State:** a flat record of the ledger, $\hat\sigma_{\mathrm{TE}}$, $V_t$, wash and candidate-score
  summaries, and factor exposures $\Phi_k^\top\delta w$ (PCA on $\hat\Sigma_t$), plus $\hat\sigma$
  summaries, re-tested here regardless of the v0.3-7 outcome.
- **Action:** low-dimensional $(\vartheta,m,b)$ driving a deterministic executor. The oracle is a
  fixed action, so rung 3 lies inside the policy class.
- **Transition:** `ProcessDay`.
- **Reward:** the pinned running-cost form
  $r_t=B_t-\tfrac{\kappa_r}{2}\tfrac{V_t}{252}\hat\sigma^2_{\mathrm{TE}}(s_{t+1})-c_{\mathrm{trade}}\lvert A_t\rvert$
  (the increment form is rejected: it telescopes).
- **Warm starts:** the GBT $\hat\eta$ and the $\hat g_{\mathrm{tax}}$ regression.
- **Episodes:** from `PriceLoader.FromGbm` before real-history fine-tuning.
- **Runtime:** C#-native vs Python (gymnasium/SB3) is decided at the start of v0.4b; the v0.3-11
  records are serializable for either.

**Gate:** rung 6 vs rung 5 under walk-forward evaluation, stability across regimes, and alpha
per unit of TE/turnover.

## v0.45 — Universe & replacement layer (moved after RL)

- **Core+reserve construction.** PCA's successor role (issue #6): the eigendecomposition of the
  point-in-time $\hat\Sigma_t$ from v0.3-6 shrinks the set of names the agent harvests over. It
  drops multicollinear, similarly-clustered names while preserving index beta and factor
  exposures; the reduced-away names form the substitute pool.
- **Offline core-basket optimizer:** min TE + complexity − dispersion potential, sweeping
  |C| ≈ 75–200.
- **Replacement module:** eligibility rules → `SubScore` → explicit fallback; same-ticker reopen
  stays as the ablation baseline.
- **Mandatory hybrid-environment transfer test** for the GBT and the policy.
- **Survivorship flag:** point-in-time membership is the v1.0 fix.

**Gate:** the hybrid book tracks the benchmark within budget at a materially lower name count
while preserving harvestable dispersion (ladder rungs 2–6 in the hybrid environment).

## v0.5 — End-to-end evaluation + integration

- walk-forward economic comparison across the full ladder, on live/recent data;
- client parameterization (hazard-rate $\delta_{\mathrm{cf}}$, outside-gains personas via
  `TaxLedger.RecordExternalGains`, $\kappa_r$);
- knowledge distillation (#16);
- environment reproducibility (#21).

## v1.0 — Deployment

- RIA-style service, with the author as first client;
- a point-in-time constituent feed;
- an auditable tax/compliance module boundary (`TaxLedger` already is one);
- monitoring, with the standing rules as operating policy.

---

## Open questions and backlog (decisions, not tasks; each names its deciding version)

| Question | Decide at | Current lean |
|---|---|---|
| RL runtime: C#-native (derivative-free search over $(\vartheta,m,b)$, TorchSharp later) vs Python gymnasium/SB3 over a step bridge | v0.4b start | undecided; v0.3-11 records are runtime-agnostic |
| ~~Materiality thresholds for the $\hat\sigma$ decision-flip gate~~ | — | **resolved:** σ̂ is built unconditionally (it is structural: TE, Markov state, simulator physics); its three roles are measured as separate ablations (v0.3-8/9/10) instead of gating the build |
| Does `--contrib` become the default simulation? | v0.3-3 | yes, if the corrected recovery holds |
| §1091 exactness: block the harvest (practice) vs partial disallowance with basis adjustment (law-exact) | v0.3-1 | block; exact treatment recorded as an extension |
| $\kappa_r$ (relative active-risk aversion) calibration | v0.3-12 / v0.4b | client parameter; ≈\$12/day at $V$=\$10M, σ_TE=2.5%, κ_r=1 |
| Leakage regression test (fit on train vs train + test, assert the production path matches train) | any | backlog; the audit memo records that it never existed |
| Tax-value regression R²: 0.92 vs 0.88 in different docs | next canonical run | re-measure; the artifact is the arbiter |
| Richer soft-label families (#17); typed ST/LT pools | v0.4a+ | deferred |

## Document map

| Doc | Role |
|---|---|
| `ROADMAP.md` (this file) | authoritative planner, status, findings, rules |
| `README.md` | front door: what it is, how to run it, results snapshot |
| [`DataMemo/README.md`](DataMemo/README.md) | the three documentation tiers |
| [`DataMemo/spec/SymbolTable.md`](DataMemo/spec/SymbolTable.md) | **start here for the math**: every object, typed, with its code member and test |
| `DataMemo/spec/*` | live mathematics (checked by `docs-check`) |
| `DataMemo/decisions/*` | why each version is the way it is (frozen) |
| [`DataMemo/archive/RetiredComponents.md`](DataMemo/archive/RetiredComponents.md) | everything the downsizing removed, and what it taught |
| `DataMemo/archive/*` | history: theory memos, the v0.2 lifecycle walk, the course recap, the architecture thread |

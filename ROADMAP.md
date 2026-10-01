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
| **F1** | `TrackingErrorProxy` estimates $\hat\Sigma$ once from the **full** price history | $\hat\sigma_{\mathrm{TE}}$ is a feature and an input to $U$, so a 2008 row sees 2009–2026 returns. This breaks standing rule 6. The deterministic-oracle leakage control cannot see it: it tests splits, not feature construction. A point-in-time window with $L<N$ is also rank-deficient, which makes shrinkage *necessary* | **v0.3-4** (no gate: correctness) |
| **F2** | `MonteCarloEngine` duplicated the simulator's day loop and had drifted six ways | every simulator fix had to be made twice; the RL environment needs one engine | ✅ downsizing (one engine, two price sources) |
| **F3** | three inconsistent RL-reward definitions across the docs; the increment form **telescopes**, and a summed per-lot $U$ miscounts shared TE | an optimizer exploits exactly these | ✅ pinned in [`SymbolTable.md` §I](DataMemo/spec/SymbolTable.md) (running-cost form, derived) |
| **F4** | symbol overloading (λ, δ, W, τ, σ, H/L, S, U, V, D, γ, q, θ, "GBM", "days") | the main source of the cognitive load, and the root cause of F6/F7 | ✅ notation contract, [`SymbolTable.md` §J](DataMemo/spec/SymbolTable.md) |
| **F5** | docs had drifted from code and from each other (e.g. a cited test that never existed; two R² values) | stale specs are worse than none | ✅ `docs-check` + recorded-drift table |
| **F6** | holding period counts **trading** days but is compared with **365**, so "long-term" ≈ 1.45 calendar years (§1222) | lots held 1.0–1.45y got τ = 0.37 instead of 0.20; the `S` feature was wrong | ✅ **v0.3-2**: `TaxLedger.IsLongTerm` (calendar). Measured (60-name GBM world, 1260 d, seed 7): `S` flips on 4,913 / 48,551 rows (10.1%), yet **0** TaxValue/label changes — offset capacity is 0 on 98.3% of rows in a loss-only book, and τ only multiplies the capacity slice. The fix becomes economically visible once gains exist (v0.3-3 ledger, v0.3-4 trim) |
| **F7** | §1091 enforced one-sided: no buy *after* a loss sale, but nothing blocks a **harvest after a recent buy**; the window is 30 **trading** days | contributions buy the most-underweight (fallen) names, exactly the ones about to be harvested. Measured by an independent audit on a weekday-calendar synthetic world: **24.3% of the contrib arm's loss sales were wash sales** | ✅ **v0.3-1**: 0 violations after the fix |

## Standing rules (apply to every version)

1. **One boundary-shaping change per PR.** Mechanical renames ride along; semantics don't.
2. **Ablation arms for anything that changes labels or trajectories**: separate simulate runs
   (the acting policy changes which rows exist), plus a *spectator* re-evaluation on the
   baseline's rows for row-aligned flip rates. `--lots=` plus the derived artifact directories
   keep arms apart.
3. **Schema changes take a version bump** through the codebook assert (`LotStateVector` is
   **v4**: 25 columns, d = 17).
4. **Economic claims cite the baseline ladder** (from v0.3-6), not classification metrics
   alone.
5. **Report ROC-AUC + PR-AUC + test-period positive rate together**, never PR-AUC alone. Prefer
   `--split=temporal` for any forward-deployment claim.
6. **Labels may peek at the future; features never.** (The founding invariant. F1 is a live
   violation until v0.3-4.)
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
| **v0.3-3** P0 close-out | retrain GBT + logistic under `--split=temporal` on the corrected `lots_contrib.csv`; class-weight re-check; decide whether `--contrib` becomes the default | ROC + PR + test prevalence vs the v0.26 baseline above |
| **v0.3-4** point-in-time $\hat\Sigma_t$ (F1) | memo `decisions/VolatilityModel_v03.md` written **first** (design in full). `ICovarianceEstimator`: `FullSample` (legacy arm, to measure the look-ahead), `PitSample`, `PitLedoitWolf` (constant-correlation target, closed form), optional MP clipping; the eigendecomposition is exposed for the PCA successor role; `--cov=` arms | **no gate** (correctness). Measure the σ_TE shift, the spectator flip rate of $f^*$/$U$, label deltas, temporal metrics |
| **v0.3-5** per-name $\hat\sigma_{i,t}$ | `IVolEstimator`: `Trailing21` (current), `Ewma(0.94)`, `Garch11` (Gaussian MLE, α+β<1, 𝓕_{t−1} data only); seam `SoftLabelBuilder.EstimateVol` | **decision-flip gate**, thresholds pre-registered in the memo. Material → v0.3-5b adds $\hat\sigma$ as a feature (schema v5). Otherwise record the negative result and re-test $\hat\sigma$ as RL state at v0.4b |
| **v0.3-6** policy seam + metric ladder | `IHarvestPolicy` (Oracle = default, byte-identical), `NeverHarvest` (rung 1), `ThresholdHarvest` (rung 2), Oracle (rung 3). `RunMetrics` from the ledger: after-tax wealth, used-now vs banked benefit, ex-ante vs **realized** TE, turnover, cost → `data/runs/<tag>/metrics.json`. A `ladder` command. Flat, serializable observation/action records | rungs 1–3 run from one command. This seam is where rung 6 (RL) plugs in |
| **v0.3-7** sell-winner trim | a second action type through the seam, behind a flag | carryforward netting exercised; `RealizedGainsYTD` endogenous |
| **v0.3-8** exit: RL readiness | SymbolTable §I finalized: reward calibration ($\kappa_r$), observation vector, executor, fed-predictions rule | the v0.3 gate |

**Deferred out of v0.3:** richer soft-label families (#17) → v0.4a backlog (the RL reward uses
the ledger, not the soft-label family); typed ST/LT ledger pools → backlog.

**v0.3 gate:**
- zero §1091 violations, and holding periods on the calendar;
- temporal PR-AUC recovers on the corrected contrib data, with ROC and prevalence reported alongside;
- $\hat\sigma_{\mathrm{TE}}$ is 𝓕_t-measurable, with its effect recorded;
- the $\hat\sigma$ build decision is recorded with numbers;
- ladder rungs 1–3 run;
- the trim exercises carryforward;
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
  summaries, re-tested here regardless of the v0.3-5 outcome.
- **Action:** low-dimensional $(\vartheta,m,b)$ driving a deterministic executor. The oracle is a
  fixed action, so rung 3 lies inside the policy class.
- **Transition:** `ProcessDay`.
- **Reward:** the pinned running-cost form
  $r_t=B_t-\tfrac{\kappa_r}{2}\tfrac{V_t}{252}\hat\sigma^2_{\mathrm{TE}}(s_{t+1})-c_{\mathrm{trade}}\lvert A_t\rvert$
  (the increment form is rejected: it telescopes).
- **Warm starts:** the GBT $\hat\eta$ and the $\hat g_{\mathrm{tax}}$ regression.
- **Episodes:** from `PriceLoader.FromGbm` before real-history fine-tuning.
- **Runtime:** C#-native vs Python (gymnasium/SB3) is decided at the start of v0.4b; the v0.3-6
  records are serializable for either.

**Gate:** rung 6 vs rung 5 under walk-forward evaluation, stability across regimes, and alpha
per unit of TE/turnover.

## v0.45 — Universe & replacement layer (moved after RL)

- **Core+reserve construction.** PCA's successor role (issue #6): the eigendecomposition of the
  point-in-time $\hat\Sigma_t$ from v0.3-4 shrinks the set of names the agent harvests over. It
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
| RL runtime: C#-native (derivative-free search over $(\vartheta,m,b)$, TorchSharp later) vs Python gymnasium/SB3 over a step bridge | v0.4b start | undecided; v0.3-6 records are runtime-agnostic |
| Materiality thresholds for the $\hat\sigma$ decision-flip gate | v0.3-4 memo | propose: spectator flip rate > 1% of oracle positives, or temporal ΔPR-AUC > 0.01 (ratify) |
| Does `--contrib` become the default simulation? | v0.3-3 | yes, if the corrected recovery holds |
| §1091 exactness: block the harvest (practice) vs partial disallowance with basis adjustment (law-exact) | v0.3-1 | block; exact treatment recorded as an extension |
| $\kappa_r$ (relative active-risk aversion) calibration | v0.3-8 / v0.4b | client parameter; ≈\$12/day at $V$=\$10M, σ_TE=2.5%, κ_r=1 |
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

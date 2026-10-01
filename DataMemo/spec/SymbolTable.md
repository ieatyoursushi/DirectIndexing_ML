# Symbol Table — every mathematical object ↔ its code ↔ its test

> **Status: LIVE SPEC — the index of the spec tier.** One row per mathematical object, typed
> like a statically-typed language declares a variable. Each row gives the domain → codomain,
> the space the object lives in, its units, a one-line definition, the code member that
> implements it, the test that pins it, and where it is derived.
> `dotnet run --project src -- docs-check` (→ `scripts/check_math_sync.py`) **fails** if:
> - a `[math:id]` tag in `src/**/*.cs` has no row here;
> - a live row has no tag in code;
> - a named code or test member no longer exists;
> - a constant in §K differs from its declaration;
> - §B's coordinate list drifts from the schema;
> - any relative markdown link is broken.
>
> **The rule this table exists for:** you only need to hold `DataMemo/spec/` in your head.
> Design records (`decisions/`) are frozen, and archive docs (`archive/`) are history. Standing
> rule 8: a PR that changes an `[math:*]`-anchored member updates its row here in the same PR.

**Reading a row.**
- *Type* uses the conventions below.
- *Code* is `Class.Member`, and several are separated by `;`.
- *Test* is `TestClass.Method`, a runnable pin of the definition.
- *Since* is the version the current definition dates from.

Derivation links point into the spec docs. [MLD] = [`MLDerivations.md`](MLDerivations.md),
[SIM] = [`SimulationMath.md`](SimulationMath.md), [PM] = [`PortfolioMath.md`](PortfolioMath.md),
[AUD] = [`MLNetLeakageAudit.md`](MLNetLeakageAudit.md).

**Type conventions.**
- `t ∈ ℤ≥0` is the **simulation day index**, i.e. trading days on the loaded calendar.
- `date(t)` is its calendar date.
- *Units* are part of the type: `$` (US dollars), `d_trd` (trading days), `d_cal` (calendar days), `ann.` (annualized). A value is dimensionless unless a unit is stated.

Known bugs are **type errors in exactly this sense**. F7 (the one-sided, trading-day wash
window) was fixed in v0.3-1; F6 (the trading-day holding period vs a calendar threshold)
was fixed in v0.3-2.

---

## A. Primitives and portfolio state

| id | symbol | type | definition | code | test | derivation | since |
|---|---|---|---|---|---|---|---|
| `lot` | $\mathrm{Lot}_k \equiv q_k\,\delta_{(p_k,s_k)}$ | atom of a measure on $\mathbb R_{>0}\times\mathbb Z_{\ge0}$: shares $q_k\in\mathbb Z_{>0}$, basis $p_k\in\mathbb R_{>0}$ ($/sh), purchase day $s_k\in\mathbb Z_{\ge0}$ (d_trd) and purchase **date** $\mathrm{date}(s_k)$ (the unit §1091/§1222 are measured in) | one tax lot as a Dirac atom of the per-asset lot measure $\mu_t^{A_i}$ | `Lot.UnrealizedReturn; Lot.PurchaseDate` | `PortfolioStateTests.Test_HarvestLoss_DecreasesRealizedGains` | [PM] §1 | v0.3-1 |
| `holding_period` | $h_k=t-s_k$ | $\mathbb Z_{\ge0}$, **d_trd** | lot age counted on the trading calendar — a feature coordinate only; the tax character uses `lt_flag` (calendar) | `Lot.HoldingPeriod` | — | [MLD] §1.2 | v0.1 |
| `lt_flag` | $\mathbf 1_{\mathrm{LT}}(t,k)=\mathbf 1[\mathrm{date}(t)>\mathrm{date}(s_k)+1\,\mathrm{yr}]$ | $\{0,1\}$ | §1222 "held more than one year" on the **calendar** (holding period starts the day after acquisition, so the anniversary is still short-term; a Feb-29 lot's anniversary is Feb 28). v0.3-2 fixed F6 (was $\mathbf 1[h_k\ge365]$ with $h_k$ in d_trd ≈ 1.45 yr) | `TaxLedger.IsLongTerm`, `Lot.IsLongTerm` | `TaxLedgerTests.Test_IsLongTerm_CalendarEdges` | [MLD] §1.2 | v0.1 (calendar v0.3-2) |
| `state` | $\mathcal S_t=(\mu_t,\ \mathrm{ledger}_t,\ \mathcal W_t)$ | product of the lot measure, the ledger ($\mathbb R\times\mathbb R_{\ge0}$), and the clock map $\mathcal W_t:\mathcal A\to\mathbb Z_{\ge0}\cup\{999\}$ | the portfolio state triple; mutable, owned by the engine | `PortfolioState.OpenLots` | `TaxLedgerTests.Test_PortfolioState_RoutesThroughLedger` | [PM] §2.1 | v0.25 |
| `ledger` | $\mathrm{ledger}_t=(G^{\mathrm{ST}}_t,G^{\mathrm{LT}}_t,C^{\mathrm{ST}}_t,C^{\mathrm{LT}}_t)$ | $\mathbb R^2\times\mathbb R^2_{\ge0}$, $ | signed net realized P&L year-to-date **by §1222 character** (a sale enters the pool of `lt_flag` at its sale date), and prior-year loss carryforward by character (§1212(b) keeps character). $G^{\mathrm{net}}=G^{\mathrm{ST}}+G^{\mathrm{LT}}$, $C=C^{\mathrm{ST}}+C^{\mathrm{LT}}$ are derived. Character pools since v0.3-3 (F8) | `TaxLedger.RecordRealized; TaxLedger.State` | `TaxLedgerTests.Test_LedgerNet_AccumulatesSignedRealized` | [PM] §2.3 | v0.25 (pools v0.3-3) |
| `schedule_d` | $\mathcal S(\mathrm{ledger})=(T,\ \mathrm{ded},\ C'^{\mathrm{ST}},C'^{\mathrm{LT}})$: $n_S=G^{\mathrm{ST}}-C^{\mathrm{ST}}$, $n_L=G^{\mathrm{LT}}-C^{\mathrm{LT}}$; opposite signs cross-net toward 0; $\mathrm{ded}=\min(O_{\max},n_S^-+n_L^-)$ taken from $n_S^-$ first; $C'^{\mathrm{ST}}=n_S^- -\mathrm{ded}_S$, $C'^{\mathrm{LT}}=n_L^- -\mathrm{ded}_L$; $T=\tau_{\mathrm{ST}}n_S^++\tau_{\mathrm{LT}}n_L^+-\tau_{\mathrm{ord}}\,\mathrm{ded}$ | ledger → $\mathbb R\times[0,O_{\max}]\times\mathbb R^2_{\ge0}$, pure | Schedule D + carryover worksheet as if the year closed now. Carryforward enters step 1, so it is **consumed** by the year's gains and claims the \$3k line before any new loss (fixes F8) | `LedgerState.Close` | `TaxLedgerTests.Test_RollYearEnd_BanksExcessLoss` | [PM] §2.5 | v0.3-3 |
| `offset_budget` | $O_t=O_{\max}-\mathrm{ded}(\mathcal S(\mathrm{ledger}_t))$ | $[0,3000]$, $ | §1211(b) ordinary-income offset not yet claimed — carryforward claims it first | `TaxLedger.OrdinaryOffsetBudget` | `TaxLedgerTests.Test_OffsetBudget_And_Capacity_DrawDown` | [MLD] §1.3 | v0.25 |
| `offset_capacity` | $\mathrm{cap}_t=n_S^{+}+n_L^{+}+O_t$ (post-netting gains of $\mathcal S$) | $\mathbb R_{\ge0}$, $ | dollars of a *new* harvested loss (either character) usable this tax year, net of carryforward | `TaxLedger.OffsetCapacity` | `TaxLedgerTests.Test_OffsetBudget_And_Capacity_DrawDown` | [MLD] §1.3 | v0.25 |
| `year_end_roll` | $(C^{\mathrm{ST}},C^{\mathrm{LT}})\leftarrow(C'^{\mathrm{ST}},C'^{\mathrm{LT}})$ of $\mathcal S$; $G^{\mathrm{ST}},G^{\mathrm{LT}}\leftarrow0$ | ledger → ledger, applied when `date(t+1).year ≠ date(t).year` | the Schedule D year boundary (`schedule_d`); wash clocks persist | `TaxLedger.RollYearEnd` | `TaxLedgerTests.Test_RollYearEnd_BanksExcessLoss` | [PM] §2.5 | v0.25 |
| `wash_clock` | $\mathcal W_{k,t}=\min\bigl(999,\ g\cdot(d-\mathrm{lastLoss}(A)),\ \min_{j\in\mathrm{acq}_{30}(A),\,j\ne k}(d-\mathrm{date}(s_j))\bigr)$, $d=\mathrm{date}(t)$; $g\in\{1,\infty\}$ the re-harvest guard; $\mathrm{acq}_{30}(A)$ = open lots of $A$ plus closed lots acquired within 30 d | $\mathbb Z_{\ge0}\cup\{999\}$, **d_cal**, **lot-level** | §1091 **before-side**: calendar distance to the lot's nearest wash-relevant event — the acquisition of a **different** lot (a replacement; a lot bought and already sold inside the window still counts, Reg. 1.1091-1) and, under the guard (default), the ticker's last *loss* sale (stricter than the law; `--no-reharvest-guard` drops it, v0.3-2b). Clean iff $\mathcal W>30$ (the window is inclusive). Fixed F7 in v0.3-1 | `PortfolioState.WashClock; PortfolioState.DaysSinceLossSale; PortfolioState.ReharvestGuard` | `WashSaleTests.Test_State_BeforeSide_LotLevelClock; WashSaleTests.Test_ReharvestGuard_Off_IsStillLawful` | [PM] §2.4 | v0.3-1 |
| `can_buy` | $\mathrm{CanBuy}_t(A)=\mathbf 1[d-\mathrm{lastLoss}(A)>30\ \mathrm{d_{cal}}]$ | ticker → $\{0,1\}$ | §1091 **after-side**: a buy (reopen or contribution) may not land within 30 calendar days after a loss sale of the ticker; `EarliestBuyDate` = sale + 31 | `PortfolioState.CanBuy; PortfolioState.EarliestBuyDate` | `PortfolioStateTests.Test_WashSaleClock_StartsAtZeroAfterHarvest` | [PM] §2.4 | v0.3-1 |
| `harvest` | $\mu\mathrel{-}=q_k\delta_{(p_k,s_k)};\ G^{\mathrm{net}}\mathrel{+}=q_k(P_t-p_k);\ \mathrm{lastLoss}(A)\leftarrow d$ if a loss | state → state | sell a lot at $P_t$: realize P&L into the ledger, remove the atom, date-stamp a loss sale (gain sales open no §1091 window) | `PortfolioState.HarvestLot` | `PortfolioStateTests.Test_HarvestLoss_DecreasesRealizedGains` | [SIM] §2.3 | v0.3-1 |
| `reopen` | buy $\lfloor q_kP_{t_h}/P_t\rfloor$ shares of $A$ at the first $t$ with $\mathrm{date}(t)\ge\mathrm{date}(t_h)+31$ | state → state | same-ticker rebuy just outside the §1091 window; re-checked with `can_buy` and re-deferred **by date** if the ticker was loss-sold again meanwhile | `SimulationEngine.ProcessDay; PriceLoader.FirstIndexOnOrAfter` | `WashSaleTests.Test_Engine_ZeroViolations_OnWorldsThatHadThem` | [SIM] §2.2 | v0.3-1 |
| `contribution` | every $\Delta_c$ d_trd, deposit $V_0\,r_c\,\Delta_c/252$ into the $n_c$ most underweight names with `can_buy` and (default) no harvestable lot | state → state (exogenous cash flow) | the cost-basis-aging fix: mints fresh lots at current prices, steering new cash away from names about to be harvested (buying them would make the fresh lot a §1091 replacement) | `SimulationEngine.ProcessContribution; ContributionPolicy.AmountPer; ContributionPolicy.SkipHarvestableNames` | `ContributionPolicyTests.Test_AmountProRatedOverInterval` | [SIM] §2 | v0.3-1 |
| `trim` | every $\Delta_{tr}$ d_trd, for the $n_{tr}$ names with $w_A>(1+b)\,\bar w$ ($\bar w=1/N$): sell whole lots with $P_t\ge p_k$, highest $p_k$ first, while $q_kP_t\le$ the remaining excess $(w_A-\bar w)V_t$; reinvest via the `contribution` buy path | state → state | sell-winner trim (v0.3-4, `--trim`): makes realized gains endogenous so `schedule_d` has gains to net. Gain sales enter the pool of their `lt_flag` and open no §1091 window. Measured: in a loss-harvesting book with no outside gains the carryforward dwarfs the trimmed gains, so they are consumed by it and the marginal harvest stays at the banked rate $\tau_{\mathrm{fut}}\delta_{\mathrm{cf}}$ | `SimulationEngine.ProcessTrim; TrimPolicy.Band` | `TrimTests.Test_Trim_SellsOnlyGains_ConsumesCarryforward` | [SIM] §2 | v0.3-4 |
| `wash_audit` | $\mathrm{viol}(S)=\mathbf 1[\exists\,b:\ A_b=A_S,\ \mathrm{lot}_b\ne\mathrm{lot}_S,\ \lvert\mathrm{date}(b)-\mathrm{date}(S)\rvert\le30\ \mathrm{d_{cal}}]$ for each loss sale $S$ | trade log → set of (sale, buy) pairs | an **independent** restatement of §1091 over the engine's trade log (not the engine's own gating), used to audit the engine: a loss sale is a wash sale iff a *different* lot of the ticker was acquired within ±30 calendar days | `WashSaleAudit.Violations; SimulationEngine.Trades` | `WashSaleTests.Test_Audit_WindowEdges_SameLot_AndGains` | ROADMAP F7 | v0.3-1 |
| `portfolio_value` | $V_t=\sum_{k\in\mathcal K_t}q_kP_t^{(A_k)}$ | $\mathbb R_{>0}$, $ | mark-to-market value of open lots with a valid close | `SimulationEngine.ProcessDay` | — | [SIM] §2.2 | v0.1 |
| `price_world` | $\{P_t^{(i)}\}_{i\le N,\,t<T}$ | $\mathbb R_{>0}^{N\times T}$ on a trading calendar | the price source: real history (`Load`) or a synthetic GBM world (`FromGbm`); one engine consumes both | `PriceLoader.Load; PriceLoader.FromGbm` | `SyntheticWorldTests.Test_RealisedVol_MatchesSigma` | [SIM] §1, §6 | v0.3 |

## B. The feature vector $x\in\mathcal X\subset\mathbb R^{19}$ (schema v5)

$\phi_{\mathrm{lot}}:(\mathrm{Lot}_k,\mathcal S_t,P_t)\mapsto x_{k,t}$ is realized by
`SimulationEngine.ExtractSnapshot` (one `LotStateVector` per open lot per day). Rows are in
**schema order**, which `docs-check` asserts equals `FeatureLists.NumericFeatures` (C#) and
`NUMERIC_FEATURES` (Python). Per-column prose (units, encoding, missingness) lives once, in
`src/ML/Python/scripts/codebook_schema.py`.

| id | symbol | type | definition | code | test | derivation | since |
|---|---|---|---|---|---|---|---|
| `x.L` | $\ell_k$ | $(-1,\infty)$ | $(P_t-p_k)/p_k$ | `Lot.UnrealizedReturn` | — | [MLD] §1.2 | v0.1 |
| `x.H` | $h_k$ | $\mathbb Z_{\ge0}$, d_trd | see `holding_period` | `Lot.HoldingPeriod` | — | [MLD] §1.2 | v0.1 |
| `x.S` | $\mathbf 1_{\mathrm{LT}}$ | $\{0,1\}$ | see `lt_flag` (calendar since v0.3-2) | `Lot.IsLongTerm` | — | [MLD] §1.2 | v0.1 |
| `x.B` | $p_k$ | $\mathbb R_{>0}$, $/sh | cost basis | `Lot.CostBasis` | — | [MLD] §1.2 | v0.1 |
| `x.W` | $w_k=q_kP_t/V_t$ | $(0,1)$ | lot weight | `SimulationEngine.ExtractSnapshot` | — | [MLD] §1.2 | v0.1 |
| `x.K` | $K^{A_i}_t$ | $\mathbb Z_{>0}$ | open lots in the same ticker | `SimulationEngine.ExtractSnapshot` | — | [MLD] §1.2 | v0.1 |
| `x.NetST` | $G^{\mathrm{ST}}_t$ | $\mathbb R$, $ | see `ledger` | `TaxLedger.NetShortTerm` | — | [MLD] §1.3 | v0.3-3 |
| `x.NetLT` | $G^{\mathrm{LT}}_t$ | $\mathbb R$, $ | see `ledger` | `TaxLedger.NetLongTerm` | — | [MLD] §1.3 | v0.3-3 |
| `x.CarryST` | $C^{\mathrm{ST}}_t$ | $\mathbb R_{\ge0}$, $ | see `ledger` | `TaxLedger.CarryShortTerm` | — | [MLD] §1.3 | v0.3-3 |
| `x.CarryLT` | $C^{\mathrm{LT}}_t$ | $\mathbb R_{\ge0}$, $ | see `ledger` | `TaxLedger.CarryLongTerm` | — | [MLD] §1.3 | v0.3-3 |
| `x.OrdinaryOffsetBudget` | $O_t$ | $[0,3000]$, $ | see `offset_budget` | `TaxLedger.OrdinaryOffsetBudget` | — | [MLD] §1.3 | v0.25 |
| `x.Sigma_TE` | $\hat\sigma_{\mathrm{TE},t}$ | $\mathbb R_{\ge0}$, ann. | see `sigma_te` (⚠ F1) | `TrackingErrorProxy.Update` | — | [SIM] §5 | v0.2 |
| `x.WashClock` | $\mathcal W_{k,t}$ | $\mathbb Z_{\ge0}\cup\{999\}$, d_cal | see `wash_clock` | `PortfolioState.WashClock` | — | [PM] §2.4 | v0.3-1 |
| `x.R_t` | $r_t^{(i)}$ | $\mathbb R$ | $(P_t-P_{t-1})/P_{t-1}$ | `PriceLoader.DailyReturn` | — | [MLD] §1.2 | v0.1 |
| `x.SigmaRange` | $(P^{\mathrm{hi}}_t-P^{\mathrm{lo}}_t)/P_{t-1}$ | $\mathbb R_{\ge0}$ | intraday range (synthetic world: $\sigma_{\mathrm{d}}\sqrt{4/\pi}$) | `PriceLoader.RangeVol` | — | [SIM] §6.3 | v0.1 |
| `x.DeltaMA50` | $(P_t-\mathrm{MA}_{50})/\mathrm{MA}_{50}$ | $\mathbb R$ | 50-day MA deviation | `PriceLoader.DeviationFromMA` | — | [MLD] §1.2 | v0.1 |
| `x.DeltaMA200` | $(P_t-\mathrm{MA}_{200})/\mathrm{MA}_{200}$ | $\mathbb R$ | 200-day MA deviation | `PriceLoader.DeviationFromMA` | — | [MLD] §1.2 | v0.1 |
| `x.TaxValue` | $g_{\mathrm{tax}}(\mathrm{ledger}_t,h_k,D_k)$ | $\mathbb R_{\ge0}$, $ | see `g_tax` | `TaxLedger.ComputeTaxValue` | — | [MLD] §1.3 | v0.25 |
| `x.DaysToYE` | $\mathrm{date}(\text{Dec }31)-\mathrm{date}(t)$ | $\mathbb Z_{\ge0}$, d_cal | calendar days to year-end | `SimulationEngine.ExtractSnapshot` | — | [MLD] §1.2 | v0.1 |

The categorical $z=\texttt{Sector}\in\mathcal Z$ enters through `phi_pre` (§E). $q_k$ (`Shares`)
rides the in-memory snapshot for soft-label re-dollarization and is **never** exported.

## C. The tax value, the objective, and the oracle

| id | symbol | type | definition | code | test | derivation | since |
|---|---|---|---|---|---|---|---|
| `g_tax` | $g_{\mathrm{tax}}(\mathrm{ledger},c,D)=\bigl[T(\mathrm{ledger})-T(\mathrm{ledger}\oplus_c(-D))\bigr]+\tau_{\mathrm{fut}}\delta_{\mathrm{cf}}\bigl[C'(\mathrm{ledger}\oplus_c(-D))-C'(\mathrm{ledger})\bigr]$ | $\mathcal L\times\{0,1\}\times\mathbb R_{\ge0}\to\mathbb R_{\ge0}$, $ | value of harvesting a loss of $D_k=\max(0,(p_k-P_t)q_k)$ dollars now, as a **counterfactual difference of `schedule_d`**: this year's tax saved at the rate of what the loss *displaces* (ST gain $\tau_{\mathrm{ST}}$, LT gain $\tau_{\mathrm{LT}}$, the ordinary line $\tau_{\mathrm{ord}}$; nothing if carryforward already took them) plus newly banked carryforward, discounted. The lot's character $c$=`lt_flag` only picks the pool. With no carryforward and one gain character it reduces to the v0.25 $\tau\min(D,\mathrm{cap})+\tau_{\mathrm{fut}}\delta_{\mathrm{cf}}(D-\mathrm{cap})^+$ | `TaxLedger.ComputeTaxValue` | `TaxLedgerTests.Test_ComputeTaxValue_CapacitySplit_And_Rates` | [MLD] §1.3 | v0.25 (counterfactual v0.3-3) |
| `cov_hat` | $\hat\Sigma$ | $\mathbb R^{N\times N}$, symmetric PSD, daily units | pairwise available-case sample covariance of daily returns. ⚠ **F1:** estimated once from the **full** history, so it is not $\mathcal F_t$-measurable; replaced by `cov_hat_pit` in v0.3-6 | `TrackingErrorProxy.ComputeCovariance` | `TrackingErrorProxyTests.Test_SigmaTE_Positive_ForAntiCorrelatedUniverse` | [SIM] §5.2 | v0.2 |
| `sigma_te` | $\hat\sigma_{\mathrm{TE},t}=\sqrt{252\,\delta w_t^\top\hat\Sigma\,\delta w_t}$, $\delta w_{t,i}=\mathbf 1[i\in\mathcal H_t]/\lvert\mathcal H_t\rvert-1/N$ | $\mathbb R_{\ge0}$, ann. | ex-ante tracking error of the **equal-weight held set** $\mathcal H_t$ vs an equal-weight benchmark. It depends on *which tickers* are held, not on dollar weights. Computed once per day, **before** that day's harvests | `TrackingErrorProxy.Update` | `TrackingErrorProxyTests.Test_SigmaTE_StaysBounded_AfterStructuralLotRemoval` | [SIM] §5.1 | v0.2 |
| `U` | $U(x)=g_{\mathrm{tax}}-\lambda_{\mathrm{TE}}\hat\sigma_{\mathrm{TE}}^2-c_{\mathrm{trade}}$ | $\mathcal X\to\mathbb R$, $ | the per-lot, one-step net-benefit score. **Not** a sequential reward: summing it over a day's harvests charges the shared $\hat\sigma_{\mathrm{TE}}$ once per lot (F3, §I) | `OracleBoundary.Utility` | `OracleScalarizedTests.Test_Utility_Arithmetic_And_CTrade` | [MLD] §2.1 | v0.25 |
| `f_star` | $f^*(x)=\mathbf 1[\ell\le-\theta_1]\,\mathbf 1[\mathcal W>\theta_3]\,\mathbf 1[\hat\sigma_{\mathrm{TE}}\le\theta_{\max}]\,\mathbf 1[U>0]$ | $\mathcal X\to\{0,1\}$, deterministic, measurable in the current features | the mechanistic oracle; its boundary is the level set $\{U=0\}$ cut by three halfspaces | `OracleBoundary.Label` | `OracleScalarizedTests.Test_TradeOff_TaxValueVsTrackingError` | [MLD] §2.1 | v0.25 |

## D. The label family and the training targets

| id | symbol | type | definition | code | test | derivation | since |
|---|---|---|---|---|---|---|---|
| `y_oracle` | $Y_{\mathrm{Oracle}}=f^*(x)$ | $\{0,1\}$ | hard label from the **acting** oracle; the leakage control (deterministic in the current features) | `SimulationEngine.ExtractSnapshot` | `OracleScalarizedTests.Test_SnapshotOverload_MatchesScalarForm` | [MLD] §2.2 | v0.1 |
| `soft_step` | $\varphi(P,s)$ | $\mathbb R_{>0}\times\{1..T_{\mathrm{fwd}}\}\to\{0,1\}$ | $f^*$ at forward step $s$ under **frozen** portfolio state ($\mathrm{cap}_t$, $\hat\sigma_{\mathrm{TE},t}$ fixed); only $P$, $\mathbf 1_{\mathrm{LT}}$ at $\mathrm{date}(t)+\Delta_{\mathrm{cal}}(t,t{+}s)$ and $\mathcal W+\Delta_{\mathrm{cal}}(t,t{+}s)$ (d_cal, real calendar; 7/5 extrapolation past its end) move | `SoftLabelBuilder.StepLabel` | — | [SIM] §4 | v0.25 |
| `y_soft_bt` | $\tilde y_{\mathrm{BT}}=\frac1{T_{\mathrm{fwd}}}\sum_{s=1}^{T_{\mathrm{fwd}}}\varphi(P_{t+s},s)$ | $[0,1]\cup\{\mathrm{NaN}\}$ | **occupation fraction** of the next 30 days on the one realized path (NaN if $t+T_{\mathrm{fwd}}\ge T$). Labels may peek forward; features never | `SoftLabelBuilder.ComputeBT` | `SyntheticWorldTests.Test_CanonicalEngine_RunsOnSyntheticWorld` | [MLD] §2.3 | v0.1 |
| `y_soft_gbm` | $\tilde y_{\mathrm{GBM}}=\frac1M\sum_{m=1}^{M}\mathbf 1[\exists s\le T_{\mathrm{fwd}}:\varphi(P^{(m)}_s,s)=1]$ | $[0,1]$ | **first-passage** probability over $M=200$ simulated GBM paths started at $P_t$ with $\hat\sigma=$ `sigma_hat_trailing`. A different functional from $\tilde y_{\mathrm{BT}}$ | `SoftLabelBuilder.ComputeGBM; GbmSimulator.FractionFiring` | `GbmSimulatorTests.Test_FractionFiring_InRange_ForRealisticPredicate` | [MLD] §2.6 | v0.1 |
| `sigma_hat_trailing` | $\hat\sigma^{(i)}_t=\sqrt{252}\cdot\mathrm{sd}(r^{(i)}_{t-21..t-1})$, fallback 0.20 | $\mathbb R_{>0}$, ann., $\mathcal F_{t-1}$-measurable | the only volatility estimate in the pipeline today; its seam is where v0.3-7's `IVolEstimator` plugs in | `SoftLabelBuilder.EstimateVol` | — | [SIM] §4.1 | v0.1 |
| `y_taxvalue` | $Y_{\mathrm{TaxValue}}=g_{\mathrm{tax}}(\cdot)$ | $\mathbb R_{\ge0}$, $ | regression target; equals the `TaxValue` feature, so regressions on it **must** drop that feature | `SimulationEngine.ExtractSnapshot` | — | [MLD] §2.5 | v0.25 |
| `y_utility` | $Y_{\mathrm{Utility}}=U(x)$ | $\mathbb R$, $ | raw objective before thresholding (per-lot; see `U`) | `SimulationEngine.ExtractSnapshot` | — | [MLD] §2.5 | v0.25 |
| `y_target_bin` | $y=\mathbf 1[\tilde y_{\mathrm{BT}}>0]$ (target `soft_bt`) · $y=Y_{\mathrm{Oracle}}$ (target `oracle`) | $\{0,1\}$ | the binary targets the classifiers actually fit ("fires at least once in 30 days"); NaN rows are dropped for `soft_bt` | `GradientBoostedTreesTrainer.SelectTarget; LogisticTrainer.SelectTarget` | — | [MLD] §2.5 | v0.1 |

## E. Estimators — model class, objective, estimator, prediction rule, estimand

Five distinct objects, which the course-level word "model" blurs together:
- **Model class** $\mathcal H$: a set of functions.
- **Objective**: a functional on $\mathcal H$.
- **Estimator**: the data → $\mathcal H$ map, i.e. the (approximate) argmin plus CV tuning.
- **Prediction rule**: a score thresholded at $\vartheta$.
- **Estimand**: the unknown population quantity being approximated.

| id | symbol | type | definition | code | test | derivation | since |
|---|---|---|---|---|---|---|---|
| `eta` | $\eta_y(x)=\mathbb P(y=1\mid X=x)$ | $\mathcal X\to[0,1]$ — the **estimand**; unknown truth, no code | the conditional harvest propensity for target $y$ (`y_target_bin`). For the `oracle` target it is $\{0,1\}$-valued (deterministic); for `soft_bt` it is genuinely stochastic | — (estimand) | — | [MLD] §8 | v0.1 |
| `phi_pre` | $\phi(x,z)=[\mathrm{Norm}(x)\,\Vert\,\mathrm{OneHot}(\mathrm{Clean}(z))]$ | $\mathcal X\times\mathcal Z\to\mathbb R^{17+m}$ | preprocessing; $(\mu_j,\sigma_j)$ and the sector vocabulary are fit on the training fold only | `PreprocessingPipeline.Build` | — | [MLD] §3.3, [AUD] | v0.1 |
| `imputer` | $\mathrm{med}_{\mathcal D_{\mathrm{tr}}}(x_j)$ | per-feature scalar | median imputation fit on the training fold only (signature-enforced) | `MedianImputer.Fit` | `PreprocessingTests.Test_MedianImputerReplacesNaNs` | [AUD] §2 | v0.1 |
| `class_weights` | $w_c=n_{\mathrm{tr}}/(2\,n_c)$ | $c\in\{0,1\}\to\mathbb R_{>0}$ | balanced weights from the training fold only | `ClassWeights.AttachBalancedWeights` | `PreprocessingTests.Test_ClassWeightsBalanced` | [AUD] §1 | v0.1 |
| `eta_hat_gbt` | $\hat\eta_{\mathrm{GBT}}(x)=\mathrm{sgm}(a\,F_M(\phi)+b)$, $F_M=\sum_{m\le M}\nu f_m$ | **class** $\mathcal H_{\mathrm{GBT}}$: Platt-calibrated sums of $J$-leaf trees. **objective:** class-weighted logistic loss via functional gradient. **estimand:** $\eta_y$ | **the champion**, RL state input, and fitted-Q substrate. Grid $M\in\{100,200\},\nu\in\{.05,.1\},J\in\{20,31\}$ | `GradientBoostedTreesTrainer.RunCV; GradientBoostedTreesTrainer.Run` | `GbtTrainerTests.Test_GbtCv_SeparableData` | [MLD] §4.2 | v0.2 |
| `eta_hat_lr` | $\hat\eta_{\mathrm{LR}}(x)=\mathrm{sgm}(w^\top\phi+b)$ | **class** $\mathcal H_{\mathrm{lin}}$. **objective:** class-weighted cross-entropy $+\lambda_{L2}\lVert w\rVert^2$, $\lambda_{L2}=1/C$. **estimand under misspecification:** the KL projection of $\eta_y$ onto $\mathcal H_{\mathrm{lin}}$ | **the linear control**: the GBT−LR gap measures the target's non-linearity (≈0.015 oracle, ≈0.19 soft) | `LogisticTrainer.RunCV; LogisticTrainer.Run` | `ChampionSelectionTests.Test_GbtBeatsLogisticOnNonLinearTarget` | [MLD] §4.1, §4.3 | v0.1 |
| `g_hat_tax` | $\hat g:\mathbb R^{16}\to\mathbb R$ | squared-error regression (SDCA linear vs FastTree) of $Y_{\mathrm{TaxValue}}$ on $x$ minus `TaxValue` | **function recovery**: the estimand $\mathbb E[Y_{\mathrm{TaxValue}}\mid x_{-\mathrm{TV}}]=g_{\mathrm{tax}}$ is deterministic; trees recover the kinks (R² ≈ 0.9 vs 0.10 linear). v0.4 value-function warm start | `TaxValueRegressionPipeline.Run` | — | [MLD] §4.4 | v0.25 |
| `champion` | $\mathcal M^\star=\arg\max_{\mathcal M}\widehat{\mathrm{PRAUC}}_{\mathrm{CV}}(\mathcal M)$ | list of CV results → model name | a pure function of CV results; the test set is touched only after the leaderboard is written | `MLnetPipeline.SelectChampion` | `ChampionSelectionTests.Test_SelectChampion_IsArgmaxOfCv` | [MLD] §7 | v0.3 |

## F. Evaluation functionals

| id | symbol | type | definition | code | test | derivation | since |
|---|---|---|---|---|---|---|---|
| `pr_auc` | $\mathrm{AP}=\sum_k(\mathrm{Rec}_k-\mathrm{Rec}_{k-1})\mathrm{Prec}_k$ | scored sample $\to[0,1]$ | step-function average precision. **No-skill floor = prevalence $p$**: report it with ROC-AUC and $p$ (rule 5) | `BinaryMetrics.Compute` | `BinaryMetricsTests.Test_AveragePrecision_And_Roc_HandComputed; BinaryMetricsTests.Test_NoSkill_Floors_RocHalf_PrEqualsPrevalence` | [MLD] §5 | v0.1 |
| `roc_auc` | $\int_0^1\mathrm{TPR}\,d\,\mathrm{FPR}$ | scored sample $\to[0,1]$ | ranking quality; no-skill = ½ at any prevalence (the leakage-control lens) | `BinaryMetrics.Compute` | `BinaryMetricsTests.Test_PerfectRanker_IsOne_AtAnyPrevalence` | [MLD] §5.2 | v0.1 |
| `f1` | $F_1(\vartheta)=2\,\mathrm{Prec}\,\mathrm{Rec}/(\mathrm{Prec}+\mathrm{Rec})$ | $[0,1]$ | at $\vartheta=0.5$ and at the maximizing $\vartheta^\star$ | `BinaryMetrics.Compute` | — | [MLD] §5.1 | v0.1 |

## G. Partitions

| id | symbol | type | definition | code | test | derivation | since |
|---|---|---|---|---|---|---|---|
| `split_policy` | $\mathcal D=\mathcal D_{\mathrm{tr}}\sqcup\mathcal D_{\mathrm{te}}$ | row set → pair of row sets | dispatches on `SplitPolicy` (random default; temporal on `--split=temporal`) | `DataSplit.TrainTest; DataSplit.Folds` | `TemporalSplitTests.Test_DataSplit_PolicyDispatch` | [MLD] §3.2 | v0.26 |
| `split_random` | stratified random 80/20 | row set → pair | preserves prevalence; inherits the full-sample regime mix | `StratifiedSplit.Split` | `StratifiedSplitTests.Test_PreservesClassProportionWithin1Percent` | [MLD] §3.2 | v0.1 |
| `split_temporal` | $\mathcal D_{\mathrm{te}}=\{t\ge T^\star\},\ \mathcal D_{\mathrm{tr}}=\{t\le T^\star-E-1\}$, $E\ge T_{\mathrm{fwd}}$ | row set → pair | chronological split with a purge, so no training label window $(t,t+T_{\mathrm{fwd}}]$ reaches the test period | `TemporalSplit.TrainTest` | `TemporalSplitTests.Test_TrainTest_BoundaryAndEmbargo` | [MLD] §3.2 | v0.26 |
| `purged_folds` | purged $k$-fold | row set → $k$ (train, val) pairs | the same purge on both sides of every interior validation block | `TemporalSplit.PurgedFolds` | `TemporalSplitTests.Test_PurgedFolds_EmbargoBothSides` | [MLD] §3.2 | v0.26 |

---

## H. Planned — v0.3 objects (no code yet; defined so the implementation has a target)

| id | symbol | type | definition | code | test | derivation | since |
|---|---|---|---|---|---|---|---|
| `cov_hat_pit` | $\hat\Sigma_t=(1-\rho_{\mathrm{LW}})\,S_t+\rho_{\mathrm{LW}}\,F_t$, with $S_t$ the sample covariance of $r_{t-L..t-1}$ | $\mathbb R^{N\times N}$ PSD, $\mathcal F_{t-1}$-measurable | point-in-time Ledoit–Wolf shrinkage toward a constant-correlation target $F_t$, closed-form $\rho_{\mathrm{LW}}$. **Required, not optional:** $\mathrm{rank}(S_t)\le L-1<N$ when $L\le N$ ($L=252$, $N\approx409$). Optional MP clipping at $\lambda_+=\bar\sigma^2(1+\sqrt{\varrho})^2$, $\varrho=N/L$. Fixes F1 | — (v0.3-6) | — | [archive part II §C.2.3](../archive/data_memo_theory_part2.md) | planned |
| `sigma_hat_ewma` | $\hat\sigma^2_t=\lambda_{\mathrm{EW}}\hat\sigma^2_{t-1}+(1-\lambda_{\mathrm{EW}})r^2_{t-1}$ | $\mathbb R_{>0}$, $\mathcal F_{t-1}$-measurable | RiskMetrics EWMA, $\lambda_{\mathrm{EW}}=0.94$ (IGARCH special case) | — (v0.3-7) | — | [archive part II §C.2.1](../archive/data_memo_theory_part2.md) | planned |
| `sigma_hat_garch` | $\hat\sigma^2_t=\omega+\alpha r^2_{t-1}+\beta\hat\sigma^2_{t-1}$, $\alpha+\beta<1$ | $\mathbb R_{>0}$, $\mathcal F_{t-1}$-measurable | per-name GARCH(1,1), Gaussian MLE fit only on data $\le t-1$ | — (v0.3-7) | — | [archive part II §C.2.2](../archive/data_memo_theory_part2.md) | planned |
| `flip_rate` | $\varphi_{\mathrm{flip}}=\dfrac{\#\{i: f^*_{\mathrm{new}}(x_i)\ne f^*_{\mathrm{old}}(x_i)\}}{\#\{i: f^*_{\mathrm{old}}(x_i)=1\}}$ on the **baseline run's rows** | $\mathbb R_{\ge0}$ | the v0.3-7 build gate. It is a **spectator** quantity: re-evaluate the new oracle on the *same* states, because separate acting runs are not row-aligned. Trajectory-level effects are measured separately, on the ladder | — (v0.3-6/5) | — | ROADMAP v0.3-7 | planned |
| `ladder` | rungs $\mathcal R_1..\mathcal R_6$ | policies → metric vectors | (1) no harvesting · (2) naive loss-threshold · (3) the oracle · (4) GBT + fixed thresholds · (5) GBT + constrained optimizer · (6) RL | — (v0.3-11) | — | ROADMAP | planned |
| `run_metrics` | $(W^{\mathrm{AT}}_T,\ B^{\mathrm{use}},\ B^{\mathrm{bank}},\ \mathrm{TE}^{\mathrm{ex\,ante}},\ \mathrm{TE}^{\mathrm{realized}},\ \mathrm{turnover},\ \mathrm{cost})$ | run → $\mathbb R^7$ | after-tax terminal wealth; benefit split into used-now vs banked straight off the ledger; realized TE = $\sqrt{252}\,\mathrm{sd}(r^P_t-r^B_t)$ (distinct from ex-ante $\hat\sigma_{\mathrm{TE}}$) | — (v0.3-11) | — | ROADMAP v0.3-11 | planned |

## I. Planned — the v0.4 MDP, with the reward **pinned** (finding F3)

| id | symbol | type | definition | code | test | derivation | since |
|---|---|---|---|---|---|---|---|
| `mdp_state` | $s_t=(\mathrm{ledger}_t,\ \hat\sigma_{\mathrm{TE},t},\ V_t,\ \mathrm{DaysToYE}_t,\ \omega_t,\ \xi_t,\ \Phi_k^\top\delta w_t\ [,\ \hat\sigma\text{-summary}])$ | a flat record in $\mathbb R^{d_s}$, $\mathcal F_t$-measurable | ledger, TE, and value; $\omega_t$ = wash-window summary; $\xi_t$ = quantiles of $\hat\eta$ over gate-passing held lots; $\Phi_k^\top\delta w_t$ = active-weight exposure to the top-$k$ eigenvectors of `cov_hat_pit` (PCA's successor role) | — (v0.3-11 / v0.4b) | — | ROADMAP v0.4b | planned |
| `mdp_action` | $a_t=(\vartheta_t,\ m_t,\ b_t)$ | $[0,1]\times\mathbb Z_{\ge0}\times\mathbb R_{\ge0}$ — **low-dimensional** | score threshold, max harvests, TE budget. A deterministic **executor** $E(s_t,a_t)\mapsto A_t\subseteq\mathcal K_t$ keeps gate-passing lots with score ≥ $\vartheta_t$, top $m_t$, subject to projected $\hat\sigma^2_{\mathrm{TE}}\le b_t$. The oracle is the fixed action $(\text{score}=U,\ \vartheta{=}0,\ m{=}\infty,\ b{=}\theta_{\max}^2)$, so rung 3 lies **inside** the policy class and "RL beats the oracle" is well-posed | — (v0.3-11) | — | ROADMAP v0.4b | planned |
| `mdp_transition` | $s_{t+1}\sim\mathcal T(\cdot\mid s_t,a_t)$ | kernel $\mathcal S\times\mathcal A\to\Delta(\mathcal S)$ | exactly `SimulationEngine.ProcessDay` with the harvest set $A_t$; randomness only through $P_{t+1}$ (real history or `FromGbm`) | — (v0.3-11) | — | [SIM] §2.2 | planned |
| `reward` | $r_t=B_t-\lambda^{\mathrm{run}}_t\,\hat\sigma^2_{\mathrm{TE}}(s_{t+1})-c_{\mathrm{trade}}\,\lvert A_t\rvert$ | $\mathcal S\times\mathcal A\to\mathbb R$, $/day | **the pinned reward**, derived below. $B_t=\sum_{j}g_{\mathrm{tax}}(\mathrm{ledger}^{(j-1)}_t,h_{k_j},D_{k_j})$ over the day's harvests in execution order, which equals $\sum Y_{\mathrm{TaxValue}}$ over the day's harvested rows, since the engine updates the ledger sequentially. $\lambda^{\mathrm{run}}_t=\kappa_r V_t/(2\cdot252)$ | — (v0.4b) | — | this section | planned |
| `return` | $G=\sum_{t\ge0}\gamma^t r_t$, $\gamma\in(0,1)$ | $\mathbb R$, $ | discounted return; $V^\pi(s)=\mathbb E_\pi[G\mid s_0=s]$ | — (v0.4b) | — | [archive part II §C.3.1](../archive/data_memo_theory_part2.md) | planned |
| `fed_predictions` | $\hat\eta^{(\le t-E)}$ | model trained on rows $\le t-E$ | **anti-leakage rule:** any prediction fed into $s_t$ must come from a model trained walk-forward on data ending at least $E$ days before $t$ | — (v0.4a) | — | ROADMAP v0.4a | planned |

**Why the reward takes this form.** It is a derivation, not a style choice. Three candidate rewards
appear in older docs. Two of them are wrong in ways an optimizer will exploit.

1. **The increment form telescopes** (`data_memo_theory_part2 §C.3.1`:
   $r_t=\ldots-\lambda\,\Delta\hat\sigma_{\mathrm{TE}}$). Write $x_t=\hat\sigma^2_{\mathrm{TE}}(s_t)$
   and sum by parts:
   $$\sum_{t=0}^{T-1}\gamma^t(x_{t+1}-x_t)=\gamma^{T-1}x_T-x_0+(1-\gamma)\sum_{t=1}^{T-1}\gamma^{t-1}x_t .$$
   The interior path is weighted by $(1-\gamma)$, which is 1% at $\gamma=0.99$. As $\gamma\to1$
   only the endpoints are penalized. An agent can therefore run large tracking error
   mid-episode almost for free, provided it returns to the benchmark by the end.
2. **Summed per-lot utility miscounts.**
   $\sum_{k\in A_t}U_k=B_t-\lvert A_t\rvert(\lambda_{\mathrm{TE}}\hat\sigma^2_{\mathrm{TE},t}+c_{\mathrm{trade}})$.
   The TE charge scales with the *number of harvests*, not the TE incurred, and it uses the
   *pre-action* $\hat\sigma_{\mathrm{TE},t}$. $U$ is a good decision **threshold** (a shadow
   price calibrated in `OracleConfig`) but a wrong accounting identity.
3. **Running cost is the certainty-equivalent charge.** The daily active P&L
   $\Delta A_t=V_t(r^P_t-r^B_t)$ has variance $\approx V_t^2\hat\sigma^2_{\mathrm{TE}}/252$. A
   mean–variance agent with absolute risk aversion $\alpha$ therefore pays
   $\tfrac\alpha2\mathrm{Var}(\Delta A_t)$ per day. Writing relative active-risk aversion as
   $\kappa_r=\alpha V_t$ (dimensionless) gives
   $\lambda^{\mathrm{run}}_t\hat\sigma^2=\tfrac{\kappa_r}{2}V_t\hat\sigma^2/252$ dollars per day.
   Charged on the **post-action** state $s_{t+1}$, this penalizes every day spent away from
   the benchmark and scales with book size.
   *Magnitude check:* at $V=\$10$M, $\hat\sigma_{\mathrm{TE}}=0.025$, $\kappa_r=1$ it is about
   \$12/day, roughly \$3.1k/yr.
   $\kappa_r$ is a **client parameter**, calibrated at v0.4b; it is not $\lambda_{\mathrm{TE}}$.

## J. Notation contract — collisions and canonical names (use in all *new* text)

The overloads below are real. Several occur **inside a single spec document**, and two known
bugs (F6, F7) are the same failure: an unpinned type or unit. Archive docs keep their original
notation.

| symbol as found | meanings found in the docs | canonical from now on |
|---|---|---|
| $\lambda$ | TE price in $U$ (90,000); logistic L2 strength $1/C$; EWMA decay 0.94; eigenvalues | $\lambda_{\mathrm{TE}}$ · $\lambda_{L2}$ · $\lambda_{\mathrm{EW}}$ · $\lambda_j$ (eigenvalues only) |
| $\delta$ | carryforward discount 0.5; Ledoit–Wolf intensity; active weights $\delta w$ | $\delta_{\mathrm{cf}}$ · $\rho_{\mathrm{LW}}$ · $\delta w$ (vector only) |
| $W$ | lot weight; the 30-day label window; Brownian motion $W_t$; wash clock $\mathcal W$ | $w_k$ · $T_{\mathrm{fwd}}$ · $W_t$ (Brownian only) · $\mathcal W$ |
| $\tau$ | tax rate $\tau(h)$; classification threshold; stopping time | $\tau(h)$ (tax only) · $\vartheta$ · $\nu$ |
| $\sigma$ | logistic sigmoid; volatility; σ-algebra $\sigma(X)$ | $\mathrm{sgm}(z)$ · $\sigma_i,\ \sigma_{\mathrm{TE}}$ · $\mathcal F_t$, $\mathcal F^X$ |
| $H$, $L$ | holding days / unrealized return (features) vs daily high / low price | $h_k$, $\ell_k$ · $P^{\mathrm{hi}}_t$, $P^{\mathrm{lo}}_t$ |
| $S$ | long-term flag; GBM price $S_t$; state $\mathcal S_t$ | $\mathbf 1_{\mathrm{LT}}$ · $P_t$ · $\mathcal S_t$ / $s_t$ |
| $U$ | net-benefit score $U(x)$; eigenvector matrix $U_k$ | $U(x)$ · $\Phi_k$ |
| $V$ | portfolio value $V_t$; value function $V^\pi$ | $V_t$ (time subscript) · $V^\pi$ (policy superscript) |
| $D$ | dollar loss $D_k$; dataset $D$ | $D_k$ · $\mathcal D$ |
| $\gamma$ | discount factor; risk aversion | $\gamma$ · $\kappa_r$ |
| $q$ | shares $q_k$; Marchenko–Pastur aspect ratio | $q_k$ · $\varrho=N/L$ |
| $\theta$ | oracle thresholds; policy parameters $\pi_\theta$ | $\theta_1,\theta_3,\theta_{\max}$ · $\pi_\psi$ |
| "GBM" | geometric Brownian motion (code: `GbmSimulator`, `Y_Soft_GBM`) vs gradient boosting (speech) | **GBM = Brownian only**; the tree model is **GBT** |
| "days" | trading-day indices vs calendar days, silently mixed (F6; F7 fixed in v0.3-1) | always state **d_trd** or **d_cal** |

## K. Constants (checked against their declarations by `docs-check`)

| id | symbol | value | units | code | note |
|---|---|---|---|---|---|
| `theta_1` | $\theta_1$ | 0.02 | — | `OracleConfig.LossThreshold; OracleBoundary.LossThreshold` | loss-depth gate (both declarations must agree) |
| `theta_3` | $\theta_3$ | 30 | d_cal (clean iff $\mathcal W>30$: the window is inclusive) | `OracleConfig.WashSaleDays; OracleBoundary.WashSaleDays; PortfolioState.WashWindowDays; WashSaleAudit.WindowCalendarDays` | §1091 half-width (all four declarations must agree) |
| `theta_max` | $\theta_{\max}$ | 0.15 | ann. | `OracleConfig.TrackingErrorCeiling` | tail-only TE circuit breaker; binds on 0 rows in 20y |
| `lambda_te` | $\lambda_{\mathrm{TE}}$ | 90000 | $ per unit $\sigma^2$ | `OracleConfig.Lambda` | decision-threshold shadow price (not $\kappa_r$) |
| `c_trade` | $c_{\mathrm{trade}}$ | 10 | $ per harvest | `OracleConfig.CTrade` | flat round-trip friction |
| `tau_st` | $\tau_{\mathrm{ST}}$ | 0.37 | — | `TaxLedger.TauShortTerm` | ordinary marginal rate |
| `tau_lt` | $\tau_{\mathrm{LT}}$ | 0.20 | — | `TaxLedger.TauLongTerm` | long-term rate |
| `tau_ord` | $\tau_{\mathrm{ord}}$ | 0.37 | — | — (`TaxLedger.TauOrdinary` = `TauShortTerm`) | rate of the §1211(b) ordinary deduction |
| `tau_fut` | $\tau_{\mathrm{fut}}$ | 0.20 | — | `TaxLedger.TauFuture` | rate on banked losses |
| `delta_cf` | $\delta_{\mathrm{cf}}$ | 0.5 | — | `TaxLedger.CarryforwardDiscount` | stand-in for a hazard-rate discount |
| `o_max` | $O_{\max}$ | 3000 | $ / yr | `TaxLedger.AnnualOrdinaryOffsetCap` | §1211(b) |
| `t_fwd` | $T_{\mathrm{fwd}}$ | 30 | d_trd | `SoftLabelBuilder.Window` | label horizon |
| `w_vol` | — | 21 | d_trd | `SoftLabelBuilder.VolWindow` | trailing σ window for $\tilde y_{\mathrm{GBM}}$ |
| `e_embargo` | $E$ | 30 | d_trd | `SplitPolicy.EmbargoDays` | must satisfy $E\ge T_{\mathrm{fwd}}$ |
| `t_warmup` | — | 200 | d_trd | `PriceLoader.WarmupDays` | MA-200 warmup |
| `trading_days` | — | 252 | d_trd / yr | `GbmSimulator.TradingDays` | annualization |
| `contrib_interval` | $\Delta_c$ | 63 | d_trd | `ContributionPolicy.IntervalDays` | quarterly |
| `contrib_rate` | $r_c$ | 0.10 | fraction of $V_0$ / yr | `ContributionPolicy.AnnualRate` | |
| `contrib_names` | $n_c$ | 20 | names | `ContributionPolicy.NamesPerContribution` | most-underweight names bought |
| `h_lt` | — | 1 | calendar yr | — (`AddYears(1)` inside `TaxLedger.IsLongTerm`; no day-count literal since v0.3-2) | §1222 |

## Known drift, recorded rather than silently resolved

| where | discrepancy | resolution path |
|---|---|---|
| tax-value regression R² (trees) | 0.92 (MLDerivations §4.4, README) vs 0.88 (`decisions/GYTD_Redesign_Plan.md` §6.1) | re-measure on the next canonical run; the artifact is the arbiter |
| linreg escape fraction (retired) | "~10%" vs "≈24%" | archived; see `archive/RetiredComponents.md` §3 |

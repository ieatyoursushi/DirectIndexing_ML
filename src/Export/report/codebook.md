# Codebook — `data/lots.csv`

**Project:** DirectIndexing — learning the tax-loss-harvest decision boundary
**Dataset:** 1837354 rows × 26 columns. One row = one *lot snapshot*:
the state of a single tax lot on a single simulated trading day.

Rows are produced by the backtesting simulation (`dotnet run simulate`) over a
$10M equal-dollar S&P 500 portfolio driven by real daily prices (Financial
Modeling Prep API). Features describe the lot, the asset, and the shared
portfolio state; labels record whether/with what propensity the deterministic
harvesting oracle fires. Metadata columns are dropped before modeling.

Missing values are written as empty cells in the CSV and read as NaN.


## 1. `L`

| | |
|---|---|
| **Type** | float |
| **Units** | unitless (fractional return) |
| **Role** | feature (lot-level) |
| **Values / encoding** | Continuous in (−1, ∞). Negative = loss position. |
| **Missing** | None. |
| **Source** | `Lot.UnrealizedReturn(P_t)` |

Unrealized return of the lot: L = (P_t − p_k) / p_k, where P_t is today's close and p_k is the lot's cost basis per share. Negative values are paper losses; L ≤ −0.02 is the oracle's loss gate.

## 2. `H`

| | |
|---|---|
| **Type** | int |
| **Units** | trading days |
| **Role** | feature (lot-level) |
| **Values / encoding** | Non-negative integer. |
| **Missing** | None. |
| **Source** | `Lot.HoldingPeriod(t)` |

Holding period of the lot: H = t − s_k, the number of simulation days since the lot was purchased.

## 3. `S`

| | |
|---|---|
| **Type** | int (binary) |
| **Units** | — |
| **Role** | feature (lot-level) |
| **Values / encoding** | 0 = short-term (< 365 days), 1 = long-term (≥ 365 days). |
| **Missing** | None. |
| **Source** | `Lot.IsLongTerm(t)` |

Long-term holding flag: S = 1[H ≥ 365]. Determines whether the long-term or (higher) short-term capital-gains tax rate applies in the tax-alpha formula.

## 4. `B`

| | |
|---|---|
| **Type** | float |
| **Units** | US dollars per share |
| **Role** | feature (lot-level) |
| **Values / encoding** | Positive continuous. |
| **Missing** | None. |
| **Source** | `Lot.CostBasis` |

Cost basis per share p_k — the price the lot was purchased at.

## 5. `W`

| | |
|---|---|
| **Type** | float |
| **Units** | unitless (portfolio fraction) |
| **Role** | feature (lot-level) |
| **Values / encoding** | Continuous in (0, 1). |
| **Missing** | None. |
| **Source** | `derived in SimulationEngine` |

Lot weight in the portfolio: W = q_k · P_t / V_t, the lot's share of total portfolio market value on day t.

## 6. `K`

| | |
|---|---|
| **Type** | int |
| **Units** | count |
| **Role** | feature (lot-level) |
| **Values / encoding** | Positive integer. |
| **Missing** | None. |
| **Source** | `counted from PortfolioState.OpenLots` |

Number of open lots in the same ticker as this lot (including itself). Stays 1 in v0.1 unless harvested lots are re-opened.

## 7. `RealizedGainsYTD`

| | |
|---|---|
| **Type** | float |
| **Units** | US dollars |
| **Role** | feature (portfolio-level, TaxLedger) |
| **Values / encoding** | Signed continuous. Positive = net realized gains. |
| **Missing** | None. |
| **Source** | `TaxLedger.RealizedGainsYTD (via PortfolioState.Ledger)` |

Signed net realized gain/loss for the calendar year to date (the pre-v0.25 G_YTD), shared by every lot at the same timestep. In gated-oracle runs it is seeded with external gains (+$1,000,000 = 10% of the $10M portfolio) at simulation start and after each year-end reset; harvesting a loss pushes it down. Resets to 0 at year-end (net loss beyond the $3k ordinary allowance rolls into LossCarryforward instead of vanishing).

## 8. `LossCarryforward`

| | |
|---|---|
| **Type** | float |
| **Units** | US dollars |
| **Role** | feature (portfolio-level, TaxLedger) |
| **Values / encoding** | Non-negative continuous, monotone non-decreasing within a run. |
| **Missing** | None. |
| **Source** | `TaxLedger.LossCarryforward (updated at year-end roll)` |

Accumulated net capital losses beyond each year's $3,000 ordinary-income allowance (26 USC §1212(b)). Carries forward indefinitely — SURVIVES the year-end reset — which is the tax-law mechanic making 'harvest now, use later' always weakly correct for individual investors.

## 9. `OrdinaryOffsetBudget`

| | |
|---|---|
| **Type** | float |
| **Units** | US dollars |
| **Role** | feature (portfolio-level, TaxLedger) |
| **Values / encoding** | Continuous in [0, 3000]. Resets to 3000 at year-end. |
| **Missing** | None. |
| **Source** | `TaxLedger.OrdinaryOffsetBudget (derived)` |

Remaining ordinary-income offset allowance for the year: max(0, $3,000 − net loss realized so far) per 26 USC §1211(b). Together with max(RealizedGainsYTD, 0) it forms offsetCapacity, the dollars of a new harvested loss usable this tax year.

## 10. `Sigma_TE`

| | |
|---|---|
| **Type** | float |
| **Units** | annualized volatility (fraction) |
| **Role** | feature (portfolio-level) |
| **Values / encoding** | Positive continuous; 0.05 = 5% annualized TE. |
| **Missing** | None. |
| **Source** | `TrackingErrorProxy.Update` |

Forward-looking tracking-error estimate vs the equal-weight benchmark: σ_TE = sqrt(δwᵀ Σ̂ δw · 252), where Σ̂ is the daily return covariance matrix and δw the active-weight deviation from holding every constituent. Shared by every lot at the same timestep. The oracle requires σ_TE ≤ 0.05 (5% budget).

## 11. `WashClock`

| | |
|---|---|
| **Type** | int |
| **Units** | calendar days |
| **Role** | feature (portfolio-level) |
| **Values / encoding** | Non-negative integer. Sentinel 999 = ticker never harvested. |
| **Missing** | None (sentinel encodes 'never'). |
| **Source** | `PortfolioState.GetWashClock` |

Days since the last harvest of this lot's ticker. The IRS wash-sale rule blocks re-claiming a loss within 30 days, so the oracle requires WashClock ≥ 30. Clocks persist across year-end.

## 12. `R_t`

| | |
|---|---|
| **Type** | float |
| **Units** | unitless (daily return) |
| **Role** | feature (asset-level) |
| **Values / encoding** | Signed continuous. |
| **Missing** | None. — observed: 2 of 1837354 rows (0.00%) |
| **Source** | `PriceLoader close series` |

One-day simple return of the ticker: R_t = (P_t − P_{t−1}) / P_{t−1}.

## 13. `SigmaRange`

| | |
|---|---|
| **Type** | float |
| **Units** | unitless (fraction of price) |
| **Role** | feature (asset-level) |
| **Values / encoding** | Positive continuous. |
| **Missing** | None. — observed: 2 of 1837354 rows (0.00%) |
| **Source** | `PriceLoader OHLC` |

Range-based intraday volatility proxy: (High_t − Low_t) / P_{t−1}.

## 14. `DeltaMA50`

| | |
|---|---|
| **Type** | float |
| **Units** | unitless (fractional deviation) |
| **Role** | feature (asset-level) |
| **Values / encoding** | Signed continuous. |
| **Missing** | NaN (empty cell) when fewer than 50 prior closes exist for the ticker; median-imputed inside each training fold. — observed: 93 of 1837354 rows (0.01%) |
| **Source** | `computed in SimulationEngine` |

Price deviation from the 50-day moving average: (P_t − MA_50) / MA_50. Momentum / mean-reversion signal.

## 15. `DeltaMA200`

| | |
|---|---|
| **Type** | float |
| **Units** | unitless (fractional deviation) |
| **Role** | feature (asset-level) |
| **Values / encoding** | Signed continuous. |
| **Missing** | NaN (empty cell) when fewer than 200 prior closes exist (sparse price history); median-imputed inside each training fold. — observed: 626 of 1837354 rows (0.03%) |
| **Source** | `computed in SimulationEngine` |

Price deviation from the 200-day moving average: (P_t − MA_200) / MA_200. The 200-day warmup window exists so this is defined from the first active simulation day.

## 16. `TaxValue`

| | |
|---|---|
| **Type** | float |
| **Units** | US dollars |
| **Role** | feature (derived, lot-level × TaxLedger) |
| **Values / encoding** | Non-negative continuous; 0 when the lot is not at a loss. |
| **Missing** | None. |
| **Source** | `TaxLedger.ComputeTaxValue(lossDollars, H)` |

Capacity-aware dollar value of harvesting this lot today: TaxValue = τ(H)·min(loss, offsetCapacity) + τ_future·max(loss − offsetCapacity, 0)·δ, where offsetCapacity = max(RealizedGainsYTD, 0) + OrdinaryOffsetBudget, τ(H) is the short/long-term rate (0.37/0.20), τ_future = 0.20 and δ = 0.5 discounts the banked (carried-forward) slice. Supersedes the v0.2 TaxAlpha, which valued every loss dollar at the full current-year rate and counted winners' |gains| as harvestable.

## 17. `DaysToYE`

| | |
|---|---|
| **Type** | int |
| **Units** | calendar days |
| **Role** | feature (derived) |
| **Values / encoding** | Integer in [0, 365]. |
| **Missing** | None. |
| **Source** | `calendar arithmetic in SimulationEngine` |

Calendar days remaining until December 31 of the simulated tax year. Year-end is when the ledger's annual accumulators reset (and net losses roll into LossCarryforward), so harvest urgency varies with this clock.

## 18. `Y_Oracle`

| | |
|---|---|
| **Type** | int (binary) |
| **Units** | — |
| **Role** | label (hard) |
| **Values / encoding** | 0 = do not harvest, 1 = harvest. Positive rate ≈ 1.6%. |
| **Missing** | None. |
| **Source** | `OracleBoundary.Label` |

Deterministic oracle harvest decision — in gated (v0.2-legacy) runs, the conjunction of four gates: 1[L ≤ −0.02] · 1[Sigma_TE ≤ 0.05] · 1[RealizedGainsYTD > 0] · 1[WashClock ≥ 30]. The gains gate is a tracked defect (issue #23); the v0.25 scalarized oracle replaces it with a utility threshold. This is the decision boundary the supervised models try to learn. Never used as a model input.

## 19. `Y_Soft_GBM`

| | |
|---|---|
| **Type** | float |
| **Units** | probability |
| **Role** | label (soft, stochastic) |
| **Values / encoding** | Continuous in [0, 1] in increments of 1/200. |
| **Missing** | None. |
| **Source** | `SoftLabelBuilder + GbmSimulator.FractionFiring` |

Probability the oracle fires within the next 30 trading days, estimated as the fraction of 200 geometric-Brownian-motion forward price paths (per-stock σ calibrated from trailing 21-day realized volatility) on which the oracle predicate is hit, with portfolio state frozen at the snapshot. First-passage semantics: each path counts at most once.

## 20. `Y_Soft_BT`

| | |
|---|---|
| **Type** | float |
| **Units** | fraction of days |
| **Role** | label (soft, deterministic) |
| **Values / encoding** | Continuous in [0, 1] in increments of 1/30. |
| **Missing** | NaN (empty cell) when fewer than 30 forward days remain in the data window — structurally missing for the final 30 timesteps (670–699). These rows are excluded from soft-label training. — observed: 12186 of 1837354 rows (0.66%) |
| **Source** | `SoftLabelBuilder (real forward window)` |

Fraction of the next 30 actual trading days on which the oracle would fire, computed from the real forward price series with portfolio state frozen at the snapshot. The primary supervised training target (binarized as Y_Soft_BT > 0 for classification).

## 21. `Y_TaxValue`

| | |
|---|---|
| **Type** | float |
| **Units** | US dollars |
| **Role** | label (continuous regression target) |
| **Values / encoding** | Non-negative continuous dollars. |
| **Missing** | None. |
| **Source** | `TaxLedger.ComputeTaxValue at snapshot time` |

Cross-sectional regression target: taxValue_k of this lot at this timestep — the capacity-aware harvest value from the TaxLedger. Numerically identical to the TaxValue feature by construction in v0.25, so regressions on this target MUST exclude TaxValue from the feature set (the task is recovering g(ledger, H, L) from raw features). First member of the issue #17 richer-label family.

## 22. `Y_Utility`

| | |
|---|---|
| **Type** | float |
| **Units** | US dollars |
| **Role** | label (diagnostic / RL reward) |
| **Values / encoding** | Signed continuous dollars. |
| **Missing** | None. |
| **Source** | `OracleBoundary.Utility(TaxValue, Sigma_TE, config)` |

Raw scalarized objective before thresholding: U(x) = TaxValue − λ·Sigma_TE² − c_trade (λ = 90,000, c_trade = 0 until v0.25 PR 3). The scalarized oracle fires iff U > 0 (plus the hard gates), so the decision boundary is the level set {U = 0}. Computed under the run's OracleConfig in both gated and scalarized runs. Never a feature — 𝟙[U > 0] is the oracle's own boundary; exported as the issue-#17 continuous target and the v0.4 RL per-decision reward.

## 23. `Y_Oracle_GatedSpec`

| | |
|---|---|
| **Type** | int (binary) |
| **Units** | — |
| **Role** | label (ablation spectator) |
| **Values / encoding** | 0 = legacy oracle would not harvest, 1 = would harvest. |
| **Missing** | None. |
| **Source** | `OracleBoundary legacy overload over spectator G_YTD` |

What the v0.2 four-gate oracle would decide on THIS row, with legacy G_YTD bookkeeping (seed + realized P&L of this run's harvests, re-seeded each year-end) carried counterfactually alongside the acting oracle. Equals Y_Oracle in gated runs; in scalarized runs it enables same-row boundary-geometry comparison. Spectator ≠ acting: the trajectory (which rows exist, wash clocks, ledger state) was produced by the acting oracle.

## 24. `Symbol`

| | |
|---|---|
| **Type** | string |
| **Units** | — |
| **Role** | metadata (dropped before modeling) |
| **Values / encoding** | Uppercase ticker string. |
| **Missing** | None. |
| **Source** | `SPY holdings (constituents.json)` |

Ticker symbol of the lot's asset, e.g. 'AAPL'. S&P 500 constituent.

## 25. `Sector`

| | |
|---|---|
| **Type** | string (categorical) |
| **Units** | — |
| **Role** | categorical feature |
| **Values / encoding** | '-' or empty → 'Unknown' before one-hot encoding (vocabulary fit on the training fold only). |
| **Missing** | '-' placeholder ≈99.5% of rows; empty ≈0.5%. |
| **Source** | `SPY holdings (constituents.json)` |

GICS-style sector of the ticker from the SPY holdings file. In the v0.1 data this column is degenerate: ≈99.5% of rows carry the placeholder '-' and the rest are empty, so after cleaning it is effectively a single 'Unknown' category.

## 26. `Timestep`

| | |
|---|---|
| **Type** | int |
| **Units** | trading-day index |
| **Role** | metadata (dropped before modeling) |
| **Values / encoding** | Integer in [200, 699]. |
| **Missing** | None. |
| **Source** | `SimulationEngine day loop` |

Simulation day index t. Days 0–199 are the moving-average warmup (no rows emitted); active rows span t = 200–699, roughly two calendar years of real price history.


---
*Generated by `scripts/codebook.py` from `scripts/codebook_schema.py`; the
generator asserts the schema matches the CSV header, so this document always
reflects the shipped dataset.*
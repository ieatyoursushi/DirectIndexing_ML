# Simulation Layer — Mathematical Reference

This memo documents the mathematical foundations and design decisions for the simulation
layer that produces `data/lots.csv` (the training dataset) and `data/lots-mc.csv` (the
Monte Carlo augmentation dataset).

See `PortfolioMath.md` for the portfolio domain model (Lot, PortfolioState, LotSnapshot).

---

## §1  Architecture Overview

```
PriceLoader  ─────────────────────────────────┐
(real historical prices)                       │
                                               ▼
SimulationEngine ──► SoftLabelBuilder ──► SimulationExporter
(backtesting)         (Y_Soft_GBM via              │
 Y_Oracle hard label)  GbmSimulator;               ▼
                        Y_Soft_BT via         data/lots.csv
                        real forward window)

GbmSimulator ◄──── used by both SoftLabelBuilder and MonteCarloEngine
(standalone GBM engine)

MonteCarloEngine ────────────────────────────► SimulationExporter
(alternate: synthetic GBM prices,                   │
 calibrated σ from PriceLoader)                     ▼
                                              data/lots-mc.csv
```

**Two simulation modes produce the same `LotStateVector` schema:**

| Mode | Prices | Y_Oracle | Y_Soft_GBM | Y_Soft_BT |
|------|--------|----------|------------|-----------|
| `simulate` (backtesting) | Real historical (FMP) | Deterministic | GbmSimulator | Real forward window |
| `simulate-mc` (Monte Carlo) | Synthetic GBM | Deterministic on sim prices | GbmSimulator | NaN |

---

## §2  Backtesting Engine (`SimulationEngine`)

### 2.1  Initialisation

Portfolio is opened on day `t₀ = WarmupDays = 200` (the first day with valid MA₂₀₀ data).
Equal-dollar allocation:

$$
\text{perLot} = V_0 / N, \qquad k_i = \lfloor \text{perLot} / P_{t_0}^{(i)} \rfloor
$$

where $V_0 = \$10{,}000{,}000$ (default) and $N$ = number of tickers with valid prices.

**Tax state (v0.25, issue #23):** the engine's tax bookkeeping is the `TaxLedger`
(`Core/Portfolio/TaxLedger.cs`), replacing the bare `G_YTD` scalar. Stored state:
`RealizedGainsYTD` (signed net realized P&L this calendar year — identical semantics to the
old G_YTD) and `LossCarryforward` (net losses beyond each year's ordinary allowance;
survives year-end, 26 USC §1212(b)). Derived: `OrdinaryOffsetBudget`
$= \max(0,\ \$3{,}000 - \max(0, -\text{net}))$ (§1211(b)) and
$\text{offsetCapacity} = \max(\text{net}, 0) + \text{OrdinaryOffsetBudget}$.

Seeding is **mode-dependent** (`OracleConfig.SeedExternalGains`):

- **Gated mode** (v0.2-legacy ablation arm): external gains are seeded at start and after
  each year-end reset, $\leftarrow 0.10 \cdot V_0 = \$1{,}000{,}000$, simulating prior-year
  or outside gains at roughly the S&P 500's long-run annual pace. Without the seed the
  legacy gate $G^{\text{YTD}} > 0$ is permanently closed. The seed deliberately does NOT
  net against carryforward, so gated-mode label trajectories are bit-identical to the
  pre-ledger engine.
- **Scalarized mode** (canonical): **no seed.** The book is honestly loss-only —
  offsetCapacity collapses to the \$3k/yr ordinary allowance, the tax-code-accurate floor
  for a client with no outside capital-gains activity (the conservative persona; stated in
  the report).

In both modes the engine also tracks a **spectator legacy-G_YTD** (seed + Σ realized P&L of
*this run's* harvests, reset+reseeded at year-end) so the v0.2 four-gate predicate stays
evaluable pointwise on every row (`Y_Oracle_GatedSpec`).

### 2.2  Day Loop  ($t = t_0, \ldots, T-1$)

For each trading day $t$:

1. **Price lookup** — retrieve close prices $\{P_t^{(i)}\}$ from PriceLoader.
2. **Portfolio value** — $V_t = \sum_{k \in \mathcal{K}_t} q_k P_t^{(A_k)}$  
   (sum only over lots with valid close prices at $t$).
3. **Tracking error update** — call `TrackingErrorProxy.Update` with open-lot symbols  
   to get $\hat{\sigma}_{\text{TE},t}$ (see §5).
4. **Feature extraction + oracle** — for each open lot $k$:
   - Compute $\ell_k = (P_t^{(A_k)} - p_k) / p_k$, loss dollars
     $D_k = \max(0,\ (p_k - P_t) q_k)$, and the ledger valuation
     $\text{taxValue}_k = \tau(h_k)\min(D_k, \text{capacity}) + \tau_f \max(D_k - \text{capacity}, 0)\,\delta$
     with $\tau(h) = 0.37/0.20$ (short/long at $h = 365$), $\tau_f = 0.20$, $\delta = 0.5$.
   - Evaluate the mode's oracle (`OracleBoundary.Label(snapshot, config)`):
     - **Scalarized (canonical):**
       $f^*(\mathbf{x}_k) = \mathbf{1}[\ell_k \le -\theta_1] \cdot \mathbf{1}[\mathcal{W}^{(A_k)} \ge 30] \cdot \mathbf{1}[\hat\sigma_{\text{TE}} \le \theta_{\max}] \cdot \mathbf{1}[U(\mathbf{x}_k) > 0]$,
       where $U = \text{taxValue}_k - \lambda \hat\sigma_{\text{TE}}^2 - c_{\text{trade}}$
       ($\theta_{\max} = 0.15$, $\lambda = 90{,}000$, $c_{\text{trade}} = \$10$; calibration
       provenance in `OracleConfig.cs` / `GYTD_Redesign_Plan.md` v2).
     - **Gated (legacy ablation):**
       $f^* = \mathbf{1}[\ell_k \le -\theta_1] \cdot \mathbf{1}[\hat\sigma_{\text{TE}} \le \theta_2] \cdot \mathbf{1}[G^{\text{YTD}} > 0] \cdot \mathbf{1}[\mathcal{W}^{(A_k)} \ge 30]$.
   - Record snapshot $(k, t)$ as a `LotStateVector` row, including labels
     `Y_TaxValue` $= \text{taxValue}_k$, `Y_Utility` $= U$, and the spectator
     `Y_Oracle_GatedSpec` (Y_Soft labels are 0 placeholders).
   - If $f^* = 1$: harvest the lot (see §2.3)
5. **Reopen queue** — lots that cleared the 30-day wash-sale window exactly on day $t$ are
   reopened at the current price with the same dollar amount.
6. **Advance clocks** — `state.AdvanceDay()` increments every wash-sale clock by 1.
7. **Year-end reset** — if $\text{date}(t+1).\text{year} \ne \text{date}(t).\text{year}$:
   - `Ledger.RollYearEnd()`: carryforward $\mathrel{+}= \max(0,\ \text{netLoss} - \$3{,}000)$,
     then net $\leftarrow 0$ (budget resets implicitly since it is derived)
   - Gated mode only: re-seed net $\leftarrow \$1{,}000{,}000$
   - Wash clocks **persist** (IRS wash-sale window crosses Dec 31)

### 2.3  Harvest Transition

When the oracle fires on lot $k$ at price $P_t$:

$$
\Delta = q_k (P_t - p_k) \qquad \text{(negative for a loss)} \qquad
\text{Ledger.RecordRealized}(\Delta): \text{RealizedGainsYTD} \mathrel{+}= \Delta
$$

The lot is removed from $\mu_t$, the wash clock resets $\mathcal{W}_t^{(A_k)} \leftarrow 0$,
the spectator legacy-G_YTD also accrues $\Delta$, and a reopen entry is queued for day
$t + 30$ with dollar amount $q_k P_t$.

---

## §3  Geometric Brownian Motion (`GbmSimulator`)

### 3.1  Model

Log-normal price dynamics (Itô SDE):

$$
dS = \mu\, S\, dt + \sigma\, S\, dW_t
$$

Discrete Euler–Maruyama step with $\Delta = 1/252$ (one trading day):

$$
S_{t+\Delta} = S_t \cdot \exp\!\Bigl((\mu - \tfrac{\sigma^2}{2})\Delta + \sigma\sqrt{\Delta}\, Z\Bigr), \qquad Z \sim \mathcal{N}(0,1)
$$

The $-\sigma^2/2$ term is the **Itô drift correction**: it ensures the *median* of the
log-price process tracks $\mu$, not the mean.  Without it, paths would have an
upward bias of $\sigma^2/2$ per unit time.

### 3.2  Parameters

| Symbol | Name | Default | Notes |
|--------|------|---------|-------|
| $\mu$ | Annual drift | 0 | Risk-neutral; positive for bullish scenarios |
| $\sigma$ | Annual volatility | Calibrated per-stock | Fallback 20% if calibration fails |
| $\Delta$ | Time step | $1/252$ | One trading day in annual units |
| $N$ | Number of paths | 200 | For Y_Soft_GBM; 1 for MonteCarloEngine price gen |
| $H$ | Horizon | 30 | Forward window in trading days |

### 3.3  Box-Muller N(0,1) Generator

`GbmSimulator.NextGaussian` uses the Box-Muller transform:

$$
Z = \sqrt{-2 \ln U_1} \cdot \cos(2\pi U_2), \qquad U_1 \sim \text{Uniform}(0,1],\; U_2 \sim \text{Uniform}[0,1)
$$

$U_1$ is drawn from $(0, 1]$ (i.e. `1 - NextDouble()`) to guarantee $\ln U_1$ is finite.
Only one of the two orthogonal variates produced per draw is returned to keep the method
stateless.

### 3.4  First-Passage Probability (`FractionFiring`)

For each of $N$ paths:
1. Simulate $H$ steps.
2. Evaluate a predicate $\phi(S_s, s)$ at each step.
3. If $\phi$ returns true, increment the fire counter and **stop** (first-passage semantics).

$$
\hat{p}_{\text{fire}} = \frac{1}{N} \sum_{j=1}^{N} \mathbf{1}\bigl[\exists\, s \in [1,H] : \phi(S_s^{(j)}, s) = 1\bigr]
$$

This is an unbiased Monte Carlo estimate of
$P(\exists\, s \in [1,H] : \phi(S_s, s) = 1)$.  The first-passage construction is correct
for tax-loss harvesting: once a lot is harvested on a given path, subsequent price moves
are counterfactual and should not be counted again.

---

## §4  Soft Labels (`SoftLabelBuilder`)

Both strategies run in a **second pass** after the backtesting day loop, filling
`Y_Soft_GBM` and `Y_Soft_BT` on each snapshot in place.  Portfolio state is **frozen**
at the snapshot's timestep: the ledger scalars (net realized, offsetCapacity),
$\sigma_{\text{TE}}$ do not evolve during the forward window, while the wash clock and the
holding period advance with the step ($\mathcal{W} + s$, $h + s$ — so $\tau(h)$ can flip
short→long inside the window). For the scalarized objective the loss is **re-dollarized**
at each forward price, $D_k(P) = \max(0, (p_k - P)q_k)$, using the frozen share count $q_k$
carried on the snapshot (in-memory only, not exported). The oracle itself stays a black box
$\mathcal{X} \to \{0,1\}$, so the Cesàro machinery below is invariant to the v0.25 swap of
its internals.

### 4.1  Y_Soft_GBM — Stochastic Forward Window

For snapshot $(k, t)$:

1. Estimate trailing 21-day annualised vol:
   $$\hat\sigma_k = \sqrt{252} \cdot \hat s_{r,21} \qquad \text{(daily return std, trailing 21 days)}$$
   Fallback $\hat\sigma_k = 0.20$ if fewer than 5 valid returns.

2. Call `GbmSimulator.FractionFiring` with the mode's oracle predicate as closure —
   scalarized (canonical):
   $$
   \phi(P, s) = \mathbf{1}\!\Bigl[\frac{P - p_k}{p_k} \le -0.02\Bigr]
                \cdot \mathbf{1}[\mathcal{W}^{(A_k)} + s \ge 30]
                \cdot \mathbf{1}[\hat\sigma_{\text{TE}} \le \theta_{\max}]
                \cdot \mathbf{1}\!\bigl[\text{taxValue}(D_k(P),\, h_k + s,\, \text{cap})
                      - \lambda\hat\sigma_{\text{TE}}^2 - c_{\text{trade}} > 0\bigr]
   $$
   (gated mode substitutes the legacy four-gate AND with frozen $G^{\text{YTD}}$).

3. $\tilde{y}_{\text{GBM}} = \hat{p}_{\text{fire}} \in [0, 1]$

**Interpretation:** probability that the oracle would fire at some point in the
next 30 trading days if prices follow the calibrated GBM and portfolio state stays frozen.

### 4.2  Y_Soft_BT — Deterministic Forward Window

For snapshot $(k, t)$:

$$
\tilde{y}_{\text{BT}} = \frac{1}{30} \sum_{s=1}^{30} f^*\!\left(\ell_k^{(t+s)},\, \hat\sigma_{\text{TE}},\, \text{ledger}_t,\, h_k + s,\, \mathcal{W}^{(A_k)} + s\right)
$$

where $\ell_k^{(t+s)} = (P_{t+s}^{(A_k)} - p_k) / p_k$ is computed from the **real** price series.

$\tilde{y}_{\text{BT}} = \text{NaN}$ when fewer than 30 forward days remain in the data window
(approximately the last 30 rows per lot).

**Interpretation:** fraction of the next 30 actual trading days where the oracle would fire,
treating portfolio state as frozen.  This is deterministic and does not depend on the
simulation order of harvest decisions.

### 4.3  Why Two Soft Labels?

| | Y_Soft_GBM | Y_Soft_BT |
|--|-----------|-----------|
| **Source** | Stochastic (GBM) | Deterministic (real data) |
| **Always available** | Yes (no NaN except bad price) | No (NaN near end of window) |
| **Distribution** | Calibrated to local σ; may not match fat tails | Matches actual market dynamics |
| **Use case** | Richer soft-label signal; probabilistic training target | Empirically grounded validation |

---

## §5  Tracking Error (`TrackingErrorProxy`)

### 5.1  Definition — Quadratic Form (v0.2)

$$
\hat\sigma_{\text{TE},t} = \sqrt{\delta w_t^\top \hat\Sigma\, \delta w_t \cdot 252}
$$

where $\hat\Sigma$ is the $N \times N$ daily return covariance matrix pre-computed once
from the full price history at load time, and $\delta w_t$ is the active weight deviation:

$$
\delta w_i = \begin{cases}
  \dfrac{1}{n_t^{\text{open}}} - \dfrac{1}{N} & \text{if lot } i \text{ is open at day } t \\[6pt]
  -\dfrac{1}{N}                                & \text{otherwise (in wash-sale / not held)}
\end{cases}
$$

Both portfolio and benchmark use **equal weights**, consistent with the equal-dollar
lot initialisation.

### 5.2  Covariance Estimation

$\hat\Sigma$ is estimated once at construction from the full available return history
(up to 504 trading days).  Pairwise available-case sample covariance with Bessel's correction:

$$
\hat\Sigma_{ij} = \frac{1}{T_{ij}-1}
  \sum_{\substack{t=0 \\ r_t^{(i)},\, r_t^{(j)} \ne \text{NaN}}}^{T-1}
  \bigl(r_t^{(i)} - \bar{r}^{(i)}\bigr)\bigl(r_t^{(j)} - \bar{r}^{(j)}\bigr)
$$

$\hat\Sigma$ is symmetric by construction; diagonal entries $\hat\Sigma_{ii}$ are the
per-stock daily return variances.  A guard `max(variance, 0)` before the square root
prevents numerical underflow near zero.

### 5.3  Why the Quadratic Form (Not Rolling Scalar Std)

The v0.1 estimator `std(r_port − r_bench, window=30) × √252` estimates the same quantity
from the *realised* scalar series.  The quadratic form:

- Is **forward-looking** — given today's weights and the historical covariance structure,
  it reports the *expected* TE going forward rather than what recently happened.
- Exposes **cross-stock correlations** explicitly — two highly correlated stocks in the
  portfolio produce less incremental TE than two uncorrelated ones with the same weight deviation.
- Is the natural extension point for **Random Matrix Theory**: replace $\hat\Sigma$ with a
  Marchenko-Pastur-cleaned $\hat\Sigma_{\text{clean}}$ (zero noise eigenvalues, rescale trace)
  to separate signal from noise in the $N=503, T=504$ regime where $q = N/T \approx 1$.

### 5.4  Historical Note — Dollar-Value Bug (v0.0)

The original implementation computed $r_t^{\text{port}} = V_t / V_{t-1} - 1$ from total
portfolio dollar value.  When a lot was harvested, $V_t$ dropped by the harvested value,
producing a structural one-day "return" of $\approx -1/503 \approx -0.2\%$ per harvest
even with unchanged prices.  Over 100+ harvest events:

$$
\hat\sigma_{\text{TE}} \approx 0.002 \times \sqrt{252} \times \sqrt{\text{harvests}} \gg 5\%
$$

This permanently blocked the oracle's $\sigma_{\text{TE}} \le 5\%$ gate.  Observed in
the real run: $\hat\sigma_{\text{TE}}$ median $33\%$, max $116\%$.

The v0.1 fix (equal-weight ticker returns) eliminated the structural jump.  v0.2 (quadratic
form) retains the structural invariance while adding the covariance-based forward-looking estimate.

### 5.5  Computational Cost

| Step | Complexity | Frequency |
|---|---|---|
| Compute $\hat\Sigma$ | $O(N^2 T) \approx 127\text{M ops}$ | Once at construction |
| Build $\delta w$ | $O(N)$ | Each day |
| $v = \hat\Sigma\,\delta w$ + $\delta w^\top v$ | $O(N^2) \approx 253\text{K ops}$ | Each day |

For $N=503$, $T=504$: construction $< 1\text{s}$; per-day $\approx 253\text{K} \times 300 \approx 76\text{M}$ total ops.

---

## §6  Monte Carlo Engine (`MonteCarloEngine`)

### 6.1  Price Generation

For each ticker $i$, generate a GBM price series of length $T$ starting from $S_0^{(i)} = 100$:

$$
S_{t+1}^{(i)} = S_t^{(i)} \cdot \exp\!\Bigl((\mu - \tfrac{\sigma_i^2}{2})\Delta + \sigma_i\sqrt{\Delta}\, Z_t^{(i)}\Bigr)
$$

Tickers are simulated **independently** (no cross-correlation).  This is a simplification;
in the real data, cross-sectional return correlations exist within sectors.  Correlated GBM
via a Cholesky factored covariance matrix is a natural v0.2 extension.

### 6.2  Volatility Calibration

When constructed from a `PriceLoader`:

$$
\hat\sigma_i = \sqrt{252} \cdot \hat s_i, \qquad
\hat s_i = \sqrt{\frac{1}{T_{\text{cal}}} \sum_{t=t_{\text{last}}-60}^{t_{\text{last}}} r_t^{(i)2} - \bar r_i^2}
$$

using the trailing 60 calendar-day return window.  Falls back to $\hat\sigma_i = 0.20$ if
fewer than 5 valid returns are available.

### 6.3  Range Volatility Proxy

The real simulation uses $(H_t - L_t) / P_{t-1}$ (true intraday range).  The MC engine
does not simulate intraday paths; it uses the expected absolute value of a one-step
$\mathcal{N}(0, \sigma_{\text{daily}})$ draw as a proxy for the normalised daily range:

$$
\widehat{\text{RangeVol}}_t^{(i)} = \sigma_{\text{daily}}^{(i)} \cdot \sqrt{\frac{4}{\pi}} \approx 1.128\, \sigma_{\text{daily}}^{(i)}
$$

The factor $\sqrt{4/\pi}$ comes from $2 \cdot E[|Z|] = 2 \cdot \sqrt{2/\pi}$ (twice the
expected absolute half-range under a Brownian bridge approximation).

### 6.4  When to Use Monte Carlo vs Backtesting

| Criterion | Backtesting | Monte Carlo |
|-----------|-------------|-------------|
| Empirically correct feature distributions | ✓ | ✗ (i.i.d. GBM) |
| Reproduces fat tails, momentum, sector correlation | ✓ | ✗ |
| Y_Soft_BT available | ✓ | ✗ |
| Runnable without FMP data files | ✗ | ✓ |
| Controllable market regime (drift, vol) | ✗ | ✓ |
| Scalable to many years / scenarios | Limited by data | ✓ |
| Primary training data | ✓ | Supplement / augmentation |

---

## §7  Dataset Characteristics

### 7.1  Size — endogenous, not fixed

The dataset size is **not** the static product $252 \times \text{years} \times N$. A lot
emits one row per day *only while open*; every harvest removes its lot from the dataset for
exactly the 30-day wash-sale window (reopened on day $t+30$, first snapshot $t+31$). The
exact identity:

$$
N_{\text{rows}} \;=\; \underbrace{n_0 \cdot T}_{\text{ceiling}}
\;-\; \sum_{\text{harvests } h} \min\!\bigl(30,\; T_{\text{end}} - t_h\bigr)
\;-\; \varepsilon
$$

where $n_0$ = lots opened on the warmup day, $T$ = active simulation days, and
$\varepsilon$ = small leakage from NaN-close days and zero-share reopens.

Current run (10% G_YTD seed): $503 \times 500 = 251{,}500$ ceiling, $2{,}730$ harvests
costing $80{,}426$ lot-days, $\varepsilon = 323$ (0.13%) → $N_{\text{rows}} = 170{,}751$.

**Consequence:** row count is endogenous to the market path. A window with more drawdowns
fires more harvests and produces a *smaller* dataset; raising the G_YTD seed from 5% to 10%
roughly doubled harvests and shrank the dataset from 201,407 to 170,751 rows. Any shift in
the underlying price data moves $N_{\text{rows}}$ through the harvest channel.

### 7.2  Class Balance

Oracle fires when all four gates pass simultaneously.  In the 2024–2026 backtesting window
(predominantly bullish market):

| Gate | Pass rate |
|------|-----------|
| $\ell \le -2\%$ | ~13% |
| $\sigma_{\text{TE}} \le 5\%$ | ~84% |
| $G^{\text{YTD}} > 0$ | ~65% |
| $\mathcal{W}^{(A_k)} \ge 30$ | 100% (structural) |

The gates are **anti-correlated**: conditions that create harvest opportunities (market
decline → losses) also deplete $G^{\text{YTD}}$ (harvesting losses reduces it).  Empirically,
with the \$1M (10%) G_YTD seed:

$$
\text{Y\_Oracle=1 rate} \approx 1.6\%, \qquad N_{\text{positive}} \approx 2{,}730
\qquad (N_{\text{rows}} = 170{,}751)
$$

Handling in ML: use class-weighted loss (balanced weights $\approx 30$:$1$ for the positive
class) or focus on the soft labels — `Y_Soft_BT` > 0 on ~20% of labeled rows,
`Y_Soft_GBM` > 0 on ~65% of rows — as the primary training targets.

> **v0.25 update (20-year window, per arm).** The table above is the historical 2-year gated
> run. On the 20-year data: gated arm `Y_Oracle` ≈ 0.20% positive (gains gate open ~94% of
> rows, closing only in 2008–09); scalarized arm ≈ 0.20% with `Y_Soft_BT > 0` ≈ 3.2%, θ_max
> binding on 0 rows, and the endogenous-N channel running both directions (removing the gate
> added harvests/dark windows; adding $c_{\text{trade}}$ pruned marginal harvests and *grew*
> the dataset to 1,849,022 rows). Measured ablation table: `GYTD_Redesign_Plan.md` §6.1.

### 7.3  Conditional Independence

The raw panel is correlated: (a) cross-sectionally (shared ledger state and $\sigma_{\text{TE}}$),
(b) temporally (same lot across consecutive days).

However, **given the feature vector** $\mathbf{x}_k = (L, H, S, B, W, K, G_{\text{YTD}}, \sigma_{\text{TE}}, \ldots)$,
the label $Y_k$ is conditionally independent of the labels of other lots:

- The oracle is a deterministic function of $\mathbf{x}_k$ alone (no cross-lot interaction).
- The portfolio-level state shared across lots ($G_{\text{YTD}}$, $\sigma_{\text{TE}}$) is
  **encoded as features**, not hidden confounders.

Formally: $Y_k \perp Y_j \mid X_k = x_k, X_j = x_j$ for $k \ne j$.

This is the standard supervised-learning i.i.d. assumption applied to the conditional model
$p(Y \mid X)$, and it is satisfied by construction of the oracle boundary.  The exception is
`Y_Soft_BT`, whose forward window may partially overlap between consecutive snapshots of the
same lot — see §4.2.

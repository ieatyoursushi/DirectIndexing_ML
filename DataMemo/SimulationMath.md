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

where $V_0 = \$1{,}000{,}000$ (default) and $N$ = number of tickers with valid prices.

G_YTD is seeded at simulation start and after each year-end reset:

$$
G_{t_0}^{\text{YTD}} \leftarrow 0.05 \cdot V_0 = \$50{,}000
$$

This simulates prior-year or external gains the client has realised elsewhere (dividends
reinvested, other account sales).  Without seeding, the third oracle gate $G^{\text{YTD}} > 0$
is permanently closed and no harvests fire.

### 2.2  Day Loop  ($t = t_0, \ldots, T-1$)

For each trading day $t$:

1. **Price lookup** — retrieve close prices $\{P_t^{(i)}\}$ from PriceLoader.
2. **Portfolio value** — $V_t = \sum_{k \in \mathcal{K}_t} q_k P_t^{(A_k)}$  
   (sum only over lots with valid close prices at $t$).
3. **Tracking error update** — call `TrackingErrorProxy.Update` with open-lot symbols  
   to get $\hat{\sigma}_{\text{TE},t}$ (see §5).
4. **Feature extraction + oracle** — for each open lot $k$:
   - Compute $\ell_k = (P_t^{(A_k)} - p_k) / p_k$
   - Evaluate $f^*(\mathbf{x}_k) = \mathbf{1}[\ell_k \le -\theta_1] \cdot \mathbf{1}[\hat\sigma_{\text{TE}} \le \theta_2] \cdot \mathbf{1}[G^{\text{YTD}} > 0] \cdot \mathbf{1}[\mathcal{W}^{(A_k)} \ge 30]$
   - Record snapshot $(k, t)$ as a `LotStateVector` row (Y_Soft labels are 0 placeholders)
   - If $f^* = 1$: harvest the lot (see §2.3)
5. **Reopen queue** — lots that cleared the 30-day wash-sale window exactly on day $t$ are
   reopened at the current price with the same dollar amount.
6. **Advance clocks** — `state.AdvanceDay()` increments every wash-sale clock by 1.
7. **Year-end reset** — if $\text{date}(t+1).\text{year} \ne \text{date}(t).\text{year}$:
   - $G^{\text{YTD}} \leftarrow 0$, then $G^{\text{YTD}} \leftarrow G^{\text{YTD}} + 50{,}000$
   - Wash clocks **persist** (IRS wash-sale window crosses Dec 31)

### 2.3  Harvest Transition

When the oracle fires on lot $k$ at price $P_t$:

$$
\Delta G^{\text{YTD}} = q_k (P_t - p_k) \qquad \text{(negative for a loss)}
$$

The lot is removed from $\mu_t$, the wash clock resets $\mathcal{W}_t^{(A_k)} \leftarrow 0$,
and a reopen entry is queued for day $t + 30$ with dollar amount $q_k P_t$.

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
at the snapshot's timestep: $G^{\text{YTD}}$, $\sigma_{\text{TE}}$, and $\mathcal{W}^{(A_k)}$
do not evolve during the forward window.

### 4.1  Y_Soft_GBM — Stochastic Forward Window

For snapshot $(k, t)$:

1. Estimate trailing 21-day annualised vol:
   $$\hat\sigma_k = \sqrt{252} \cdot \hat s_{r,21} \qquad \text{(daily return std, trailing 21 days)}$$
   Fallback $\hat\sigma_k = 0.20$ if fewer than 5 valid returns.

2. Call `GbmSimulator.FractionFiring` with the oracle predicate as closure:
   $$
   \phi(P, s) = \mathbf{1}\!\Bigl[\frac{P - p_k}{p_k} \le -0.02\Bigr]
                \cdot \mathbf{1}[\hat\sigma_{\text{TE}} \le 0.05]
                \cdot \mathbf{1}[G^{\text{YTD}} > 0]
                \cdot \mathbf{1}[\mathcal{W}^{(A_k)} + s \ge 30]
   $$

3. $\tilde{y}_{\text{GBM}} = \hat{p}_{\text{fire}} \in [0, 1]$

**Interpretation:** probability that the oracle would fire at some point in the
next 30 trading days if prices follow the calibrated GBM and portfolio state stays frozen.

### 4.2  Y_Soft_BT — Deterministic Forward Window

For snapshot $(k, t)$:

$$
\tilde{y}_{\text{BT}} = \frac{1}{30} \sum_{s=1}^{30} f^*\!\left(\ell_k^{(t+s)},\, \hat\sigma_{\text{TE}},\, G^{\text{YTD}},\, \mathcal{W}^{(A_k)} + s\right)
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

## §5  Tracking Error Proxy (`TrackingErrorProxy`)

### 5.1  Definition

$$
\hat\sigma_{\text{TE},t} = \sqrt{252} \cdot \hat s_{30}\!\left(r^{\text{port}} - r^{\text{bench}}\right)
$$

where $\hat s_{30}$ is the 30-day rolling sample standard deviation, and:

$$
r_t^{\text{port}}  = \frac{1}{|\mathcal{K}_t|} \sum_{k \in \mathcal{K}_t} r_t^{(A_k)}, \qquad
r_t^{\text{bench}} = \frac{1}{N} \sum_{i=1}^{N} r_t^{(i)}
$$

Both returns are **equal-weighted averages** of daily log-returns.

### 5.2  Why Equal-Weight Returns (Not Dollar-Value Changes)

The previous implementation computed $r_t^{\text{port}} = V_t / V_{t-1} - 1$.

**Bug:** when a lot is harvested, $V_t$ drops discontinuously by the harvested lot's value.
For a 503-stock equal-weight portfolio, this produces a one-day "return" of $\approx -1/503$
from the structural removal alone — even if all prices are unchanged.  Over time:

- Each harvest event contributes a $\approx -0.2\%$ structural jump.
- 100+ harvest events produce spurious variance in the portfolio return series.
- Annualised: $\Delta V / V \approx -0.002$, and $0.002 \times \sqrt{252} \approx 0.032$.
  Accumulated over many harvests: $\hat\sigma_{\text{TE}} \gg 5\%$, blocking the oracle permanently.

**Fix:** use equal-weighted average of *individual ticker returns* instead of dollar-value ratio.
Removing a ticker from the portfolio does not change the equal-weight return of the
remaining tickers, so structural composition changes are invisible to the estimator.

The residual tracking error ($\hat\sigma_{\text{TE}} > 0$) correctly reflects periods when
the portfolio composition diverges systematically from the benchmark — e.g., when many tickers
are simultaneously in the wash-sale window and their benchmark exposure is missing.

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

### 7.1  Size

With 503 tickers, 300 trading days post-warmup, and oracle harvests reducing the portfolio:

$$
N_{\text{rows}} \approx 503 \times 300 \times \text{(fraction of days a lot is open)} \approx 120{,}000\text{–}140{,}000
$$

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
decline → losses) also deplete $G^{\text{YTD}}$ (harvesting losses reduces it).  Empirically:

$$
\text{Y\_Oracle=1 rate} \approx 0.8\%, \qquad N_{\text{positive}} \approx 986
$$

Handling in ML: use class-weighted loss (positive class weight $\approx 120$) or focus on
the soft label regressions `Y_Soft_GBM` (31% non-zero) as the primary training target.

### 7.3  Conditional Independence

The raw panel is correlated: (a) cross-sectionally (shared $G^{\text{YTD}}$ and $\sigma_{\text{TE}}$),
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

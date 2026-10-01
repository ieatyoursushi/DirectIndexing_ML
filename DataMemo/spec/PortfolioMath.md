# Portfolio Domain Model — Formal Specification

> **Status: LIVE SPEC** — must match the code; checked by `dotnet run --project src -- docs-check`.
> Index of every symbol ↔ code member ↔ test: [`SymbolTable.md`](SymbolTable.md).
### Gabriel Kung · Co-authored with Claude Sonnet

> Companion to `data_memo_theory.md`. This document formalises the **implementation objects** in `Core/Portfolio/` and maps each C# class precisely to its mathematical definition. The goal is to make the bridge between the ML theory (§1–9 of the theory memo) and the simulation code unambiguous for future model development.

---

## 1. The Lot as a Dirac Atom

### 1.1 Measure-Valued Portfolio Representation

Fix a probability space and a universe of assets $\mathcal{S} = \{A_1, \ldots, A_n\}$ (the S&P 500 constituents). At each simulation day $t$, the holdings in asset $A_i$ form a **finite atomic measure** on the space of (price, time) pairs:

$$\mu_t^{A_i} = \sum_{k : \text{lot } k \text{ open}} q_k \, \delta_{(p_k,\, s_k)}$$

where:

| Symbol | C# field | Meaning |
|--------|----------|---------|
| $q_k \in \mathbb{Z}_{>0}$ | `Lot.Shares` | Quantity — the *mass* of the atom |
| $p_k \in \mathbb{R}_{>0}$ | `Lot.CostBasis` | Purchase price per share — the *basis* of the atom |
| $s_k \in \mathbb{Z}_{\geq 0}$ | `Lot.PurchaseDayIndex` | Purchase day index — the *time support point* |
| $\delta_{(p_k, s_k)}$ | the `Lot` object itself | Unit point mass at $(p_k, s_k)$ |

A `Lot` object is precisely one Dirac atom. A `List<Lot>` is the full measure $\mu_t^{A_i}$ (or, since `OpenLots` spans all tickers, the aggregate measure $\mu_t = \sum_{i} \mu_t^{A_i}$).

### 1.2 Derived Quantities from the Atom

Given the current price $P_t$:

**Unrealised return** (normalised displacement from the atom's basis):
$$\ell_k = \frac{P_t - p_k}{p_k} \in (-1, \infty)$$
```csharp
lot.UnrealizedReturn(currentPrice)   // = (currentPrice - CostBasis) / CostBasis
```
$\ell_k < 0$ for a loss position — a necessary condition for harvesting.

**Holding period** (distance in time from the support point):
$$h_k = t - s_k \in \mathbb{Z}_{\geq 0}$$
```csharp
lot.HoldingPeriod(currentDay)        // = currentDay - PurchaseDayIndex
```

**Short/long-term flag** (§1222, on the calendar — since v0.3-2; $h_k$ above is a trading-day
feature and is *not* the legal holding period):
$$s = \mathbb{1}[\mathrm{date}(t) > \mathrm{date}(s_k) + 1\,\mathrm{yr}] \in \{0, 1\}$$
```csharp
lot.IsLongTerm(today)                // = TaxLedger.IsLongTerm(PurchaseDate, today)
```
This determines the applicable tax rate $\tau(h)$ in the tax-alpha formula (§4 of theory memo):
$$\tau(s) = \begin{cases} \tau_{\text{ST}} & s = 0 \\ \tau_{\text{LT}} & s = 1 \end{cases}, \quad \tau_{\text{ST}} > \tau_{\text{LT}}$$

---

## 2. PortfolioState as the State Triple

### 2.1 Formal Definition

The complete portfolio state at day $t$ is the triple (v0.25: the tax component grew from a
scalar into the `TaxLedger`):

$$\mathcal{S}_t = \left(\mu_t,\ \text{ledger}_t,\ \mathcal{W}_t\right)$$

| Component | C# member | Type | Meaning |
|-----------|-----------|------|---------|
| $\mu_t$ | `OpenLots` | `List<Lot>` | Full lot measure across all assets |
| $\text{ledger}_t$ | `Ledger` | `TaxLedger` | Schedule D bookkeeping by §1222 character (v0.3-3): `NetShortTerm`, `NetLongTerm` $\in \mathbb{R}$ (signed net this year), `CarryShortTerm`, `CarryLongTerm` $\in \mathbb{R}_{\ge 0}$ (prior-year carryforward, survives year-end); derived `RealizedGainsYTD` (their sum), `OrdinaryOffsetBudget` $\in [0, 3000]$ and `OffsetCapacity`, both computed from the netting map `LedgerState.Close` |
| $\mathcal{W}_t$ | `_lastLossSale` + open-lot `PurchaseDate`s, via `WashClock(lot)` / `CanBuy(ticker)` | `Dictionary<string,DateOnly>` + `Lot.PurchaseDate` | §1091 state on the **calendar** (v0.3-1): last loss-sale date per ticker and the acquisition dates of open lots |

The pre-v0.25 name `G_YTD` survives only as a reading aid for $G^{\mathrm{net}}=G^{\mathrm{ST}}+G^{\mathrm{LT}}$.

### 2.2 Time Evolution

**SetDate(date(t))** is the only way time enters §1091 state: every wash clock is a
*calendar-date difference* computed on demand (§2.4), never an incremented counter.

**HarvestLot()** implements the state transition on lot $k$ of asset $A_i$:
1. Realise P&L into the pool of the lot's §1222 character at the sale date (`Ledger.RecordRealized(ΔG, lot.IsLongTerm(date))`):
$$\Delta G = q_k (P_t - p_k) \qquad \text{(negative for a loss)}$$
$$G^{c}_{t+1} = G^{c}_t + \Delta G,\qquad c=\mathbf 1_{\mathrm{LT}}(t,k)$$

2. Remove atom from the measure:
$$\mu_{t+1}^{A_i} = \mu_t^{A_i} - q_k\,\delta_{(p_k, s_k)}$$

3. If the sale realized a **loss**, date-stamp it: $\mathrm{lastLossSale}(A_i) \leftarrow \mathrm{date}(t)$.
   Gain sales (the v0.3-4 trim) never open a §1091 window.

### 2.3 The TaxLedger — Sign Convention and Tax-Law Semantics (supersedes the G_YTD gate detail)

$G^{\mathrm{ST}}, G^{\mathrm{LT}}$ (and their sum $G^{\mathrm{net}}$) are **signed scalars** tracking net realised P&L for the year:

- **Positive**: net realised gains dominate (realized/external gains exceed harvested losses)
- **Negative**: net realised losses dominate (TLH has offset or exceeded gains)

Harvesting a losing lot ($P_t < p_k$) makes $\Delta G < 0$, pushing the net **more negative**.
This is correct and intentional — TLH is the act of deliberately realising losses.

**What changed in v0.25 (issue #23).** The old oracle condition $\mathbb{1}[G > 0]$ read
this scalar as a *gate* and produced a "self-limiting dynamic": harvesting stopped once
losses exhausted the year's gains. The 20-year run showed that dynamic to be economically
backwards for individual investors — under 26 USC §1211(b)/§1212(b) a harvested loss is
never wasted (it offsets gains from anywhere on the 1040, then \$3k/yr of ordinary income,
and the rest carries forward indefinitely), so "do I have gains this year" is magnitude and
timing information, not a veto. The ledger therefore encodes it as **value, not permission**:

Since v0.3-3 (ROADMAP F8) the ledger is the law's own netting rather than a blended pool.
The year-end netting map $\mathcal S$ (`LedgerState.Close`; full statement in
[MLDerivations §1.3](MLDerivations.md) and `SymbolTable.md` `schedule_d`):

1. carryforward enters as a loss of its character;
2. ST and LT cross-net;
3. up to \$3,000 of net loss is deducted from ordinary income, ST first;
4. the remainder carries forward by character.

It yields the year's tax $T$, the deduction, and next year's carryforward $C'$. Then

$$\text{OrdinaryOffsetBudget}_t = 3000-\mathrm{ded}_t,\qquad \text{OffsetCapacity}_t = n_S^+ + n_L^+ + \text{OrdinaryOffsetBudget}_t$$

$$\text{taxValue}_k = \bigl[T(\mathrm{ledger}_t)-T(\mathrm{ledger}_t\oplus_c(-D_k))\bigr]
+ \tau_f\,\delta\,\bigl[\textstyle\sum C'(\mathrm{ledger}_t\oplus_c(-D_k))-\sum C'(\mathrm{ledger}_t)\bigr]$$

with $D_k$ the loss in dollars, $c$ the lot's §1222 character (calendar anniversary), $\tau_f = 0.20$,
and $\delta = 0.5$ a constant stand-in for a hazard-rate discount on banked losses. The current-year
slice earns the rate of what it displaces ($\tau_{\mathrm{ST}}=0.37$ against a ST gain or the ordinary
line, $\tau_{\mathrm{LT}}=0.20$ against a LT gain), and nothing when carryforward already absorbs those.
At year-end, `RollYearEnd()` keeps only $C'$ (which **survives**) and zeroes both annual pools. The
legacy gate was retired with the gated oracle arm (`archive/RetiredComponents.md` §6).

### 2.4 The Wash-Sale Clock $\mathcal{W}$ — §1091 on both sides, in calendar days (v0.3-1)

**The law.** 26 USC §1091 disallows a loss if substantially identical stock is acquired
within 30 **calendar** days **before or after** the sale. That is an inclusive window of 61
days centered on the sale date.

**The simulator enforces both sides** (ROADMAP finding F7, fixed in v0.3-1):

- **Before-side (the harvest gate).** For lot $k$ of ticker $A$ on date $d$,
  $$\mathcal{W}_{k} = \min\Bigl(999,\ g\cdot(d-\mathrm{lastLossSale}(A)),\ \min_{j\in\mathrm{acq}_{30}(A),\,j\ne k}\bigl(d-\mathrm{date}(s_j)\bigr)\Bigr)\quad[\mathrm{d_{cal}}],$$
  so a **different** lot bought within 30 days is a replacement and blocks the harvest. It counts
  whether still open or already sold ($\mathrm{acq}_{30}$, v0.3-2b; Reg. 1.1091-1). The lot being
  sold is never its own replacement. The oracle gate is
  $$f^*(x) \supseteq \mathbb{1}[\mathcal{W}_{k} > 30],$$
  **strict**, because day 30 is still inside the inclusive window.
  (The $d-\mathrm{lastLossSale}$ term, $g=1$, is the pre-existing conservative rule of not
  re-harvesting a ticker within 30 days of its own harvest. It is **stricter than §1091**, since
  selling acquires nothing. `--no-reharvest-guard` sets $g=\infty$; the audit stays at 0
  violations either way.)
- **After-side (every buy).** $\mathrm{CanBuy}(A)=\mathbb 1[d-\mathrm{lastLossSale}(A)>30]$.
  The same-ticker reopen is scheduled for the first trading day on or after
  $\mathrm{sale}+31$ calendar days, and is re-deferred by date if the ticker is loss-sold again
  meanwhile. Contributions buy only tickers that pass CanBuy and, by default, hold no currently
  harvestable lot.

**Audited independently.** `WashSaleAudit` restates the law over the engine's trade log (not
its gating code). On fixed-seed synthetic worlds it found 24.3% of the contribution arm's loss
sales to be wash sales before the fix, and 0 after.

### 2.5 Year-End Reset

On January 1 of each simulated year the ledger rolls through the netting map $\mathcal S$: the
character-split remainder beyond the \$3k ordinary deduction becomes `CarryShortTerm`/`CarryLongTerm`
(which persist and are consumed by later gains), and both annual pools reset to 0. Wash-sale clocks intentionally **do not
reset** — the IRS 30-day window crosses year-end boundaries.

```csharp
portfolioState.ResetForNewYear();  // Ledger.RollYearEnd(); wash clocks untouched
```

---

## 3. LotSnapshot as the Feature Extraction Map — Graph of $\mathcal{X} \times \mathcal{Y}$

### 3.1 Formal Definition

The feature extraction map is:
$$g : \mathcal{S}_t \times \mathbf{P}_t \to \mathcal{X}^{|\mathcal{K}_t|}$$

where $\mathbf{P}_t$ is the price vector at time $t$ and $\mathcal{K}_t$ is the set of open lots. Applied to a single lot $k$, it yields one observation in the product space:

$$\text{LotSnapshot}_{k,t} \cong (x_{k,t},\, \tilde{y}_{k,t}) \in \mathcal{X} \times \mathcal{Y}$$

where $\mathcal{X} \subset \mathbb{R}^d$ is the feature space and $\mathcal{Y} = \{0,1\} \times [0,1]$ carries both label types. The full record therefore lives in $\mathbb{R}^{d+2}$ — $d$ feature coordinates plus two label coordinates (`Y_Oracle`, `Y_Soft`).

The **dimensionality partition** of the $d$ feature coordinates:

```
LotSnapshot ∈ ℝ^d × 𝒴
├── x ∈ 𝒳 ⊂ ℝ^d  (features — model inputs)
│   ├── L, H, S, B, W, K              ← 𝒳_lot     ⊂ ℝ^6   (lot-level)
│   ├── NetST, NetLT, CarryST, CarryLT, OrdinaryOffsetBudget,
│   │   Sigma_TE, WashClock           ← 𝒳_portfolio ⊂ ℝ^7  (portfolio-level: ledger + risk)
│   ├── R_t, SigmaRange, DeltaMA50, DeltaMA200,
│   │   SigmaHat, SigmaMkt            ← 𝒳_asset ⊂ ℝ^6  (asset-level + market σ̂)
│   └── TaxValue, DaysToYE, ZBarrier, PBarrier  ← 𝒳_derived ⊂ ℝ^4  (composite)
│
└── y ∈ 𝒴  (labels — model targets, never inputs)
    ├── Y_Oracle ∈ {0,1}              ← hard label  f*(x)
    ├── Y_Soft   ∈ [0,1]              ← soft labels ỹ(x)  (GBM + BT)
    ├── Y_TaxValue ∈ ℝ≥0              ← continuous regression target (≡ TaxValue feature;
    │                                    regressions on it exclude that feature)
    └── Y_Utility ∈ ℝ                 ← raw U(x)  (per-lot diagnostic; see MLDerivations §2.5)
```

So $d = 23$ before one-hot encoding of `Sector` (schema v6, v0.3-8: + the σ̂ feature role; 19 in v5, 17 in v3/v4, 15 pre-v0.25). The ML model learns $\hat{\eta} : \mathbb{R}^d \to [0,1]$ using the $d$ feature columns as input and `Y_Soft` as the training target (or `Y_Oracle` for hard-label classifiers).

**Schema-first timing:** `LotSnapshot` is defined now as the **interface contract** before the simulation exists. Every downstream component — `PriceLoader`, `OracleGate`, `SoftLabelBuilder`, `SimulationExporter` — is built against this schema. Defining it late would mean those components implicitly define the schema through whatever they happen to produce, which is riskier in a typed system.

`LotSnapshot` is:
- **Immutable** (`record` type in C#, value semantics) — unlike `Lot` and `PortfolioState` which are mutable `class` types.
- **One row** in `lots.csv` and **one observation** $(x, y) \in \mathcal{X} \times \mathcal{Y}$ for the ML model.
- **`float`-typed** for most numeric fields for ML.NET `IDataView` compatibility.

### 3.2 Feature-to-Math Correspondence

#### Lot-level (from `Lot` object)

| Field | Formula | Source | Domain |
|-------|---------|--------|--------|
| `L` | $\ell_k = (P_t - p_k)/p_k$ | `lot.UnrealizedReturn()` | $(-1, \infty)$ |
| `H` | $h_k = t - s_k$ | `lot.HoldingPeriod()` | $\mathbb{Z}_{\geq 0}$ |
| `S` | $s = \mathbb{1}[\mathrm{date}(t) > \mathrm{date}(s_k)+1\,\mathrm{yr}]$ | `lot.IsLongTerm(date)` | $\{0,1\}$ |
| `B` | $p_k$ | `lot.CostBasis` | $\mathbb{R}_{>0}$ |
| `W` | $w_k = q_k P_t / V_t$ | derived | $(0,1)$ |
| `K` | lot count for ticker $A_i$ | counted from `OpenLots` | $\mathbb{Z}_{>0}$ |

#### Portfolio-level (from `PortfolioState` / `TaxLedger`)

| Field | Formula | Source |
|-------|---------|--------|
| `NetST`, `NetLT` | $G^{\mathrm{ST}}, G^{\mathrm{LT}}$: signed net realised P&L YTD by §1222 character | `Ledger.NetShortTerm`, `Ledger.NetLongTerm` |
| `CarryST`, `CarryLT` | $C^{\mathrm{ST}}, C^{\mathrm{LT}}$: prior-year carryforward by character (output of the netting map $\mathcal S$), survives year-end, consumed by later gains | `Ledger.CarryShortTerm`, `Ledger.CarryLongTerm` |
| `OrdinaryOffsetBudget` | $\max(0, \$3k - \max(0, -\text{net}))$ | `Ledger.OrdinaryOffsetBudget` (derived) |
| `Sigma_TE` | $\sigma_{\text{TE}} = \sqrt{\delta w^\top \Sigma\, \delta w}$ | computed in simulation |
| `WashClock` | $\mathcal{W}_{k} \in \mathbb{Z}_{\geq 0}\cup\{999\}$, calendar days | `PortfolioState.WashClock(lot)` |

#### Asset-level (from price series, computed in `Simulation/`)

| Field | Formula |
|-------|---------|
| `R_t` | $r_t = (P_t - P_{t-1})/P_{t-1}$ |
| `SigmaRange` | $(H_t - L_t)/P_{t-1}$ — range volatility proxy |
| `DeltaMA50` | $(P_t - \text{MA}_{50})/\text{MA}_{50}$ |
| `DeltaMA200` | $(P_t - \text{MA}_{200})/\text{MA}_{200}$ |

#### Derived / composite

| Field | Formula |
|-------|---------|
| `TaxValue` | $\tau(h_k)\min(D_k, \text{cap}) + \tau_f \max(D_k - \text{cap}, 0)\,\delta$ — capacity-aware harvest value (supersedes v0.2 `TaxAlpha`, which valued winners' \|gains\| as losses and ignored capacity) |
| `DaysToYE` | calendar days remaining in simulated tax year |

#### Labels

| Field | Type | Meaning |
|-------|------|---------|
| `Y_Oracle` | $f^*(x) \in \{0,1\}$ | Hard label from the acting oracle |
| `Y_Soft` | $\tilde{y}(x) \in [0,1]$ | Soft labels from forward windows (GBM + BT) |
| `Y_TaxValue` | $\in \mathbb{R}_{\ge 0}$ | Continuous target ≡ `TaxValue` (exclude that feature when regressing) |
| `Y_Utility` | $U(x) \in \mathbb{R}$ | Raw scalarized objective (per-lot, one-step) |

#### Metadata (drop before modelling)

`Symbol`, `Sector`, `Timestep` — for EDA, stratified splitting, and interpretability. Not features.

### 3.3 Mutable vs Immutable — The Architectural Invariant

```
Lot              mutable class    — evolves each timestep (price changes, IsOpen toggled)
PortfolioState   mutable class    — evolves each timestep (G_YTD accumulates, clocks tick)
LotSnapshot      immutable record — frozen at extraction time, never changes
```

This mirrors the distinction in §2.2 of the theory memo between the **stochastic process** $\{(X_{i,t}, Y_{i,t})\}$ (evolving) and the **i.i.d. training sample** $S = \{(x_i, y_i)\}_{i=1}^N$ (frozen). Once a `LotSnapshot` is written to `lots.csv` it is a fixed realisation — an element of the empirical distribution $\hat{\mathcal{D}}$. The ML model never sees the mutable simulation objects.

### 3.4 The Dataset as a Graph — Cardinality Estimate

The full training dataset is the graph of the empirical map over all $(k,t)$ pairs:

$$S = \bigl\{(x_{k,t},\, \tilde{y}_{k,t})\bigr\}_{k \in \mathcal{K}_t,\; t = 1,\ldots,T}$$

This is a finite set of points in $\mathcal{X} \times \mathcal{Y}$ — the empirical distribution $\hat{\mathcal{D}}_N$ supported on $N$ atoms.

At steady state, with roughly 150 open lots per day across 252 simulated trading days per year:

$$N = |\mathcal{K}| \times T \approx 150 \times 252 \approx 37{,}800 \text{ rows per simulated year}$$

Over 2 simulated years: $N \approx 75{,}600$. Each row is one `LotSnapshot` — one point in $\mathbb{R}^{d+2}$.

### 3.5 Conditional Independence — The i.i.d. Justification

The raw panel $\{(X_{k,t}, Y_{k,t})\}$ is **not** i.i.d. — it has two sources of correlation that must be resolved before treating observations as independent training examples.

#### Source 1: Cross-sectional dependence (lots at the same $t$)

At any fixed $t$, all open lots share the **same portfolio-level state**:
$$G_t^{\text{YTD}},\; \sigma_{\text{TE},t} \in \text{PortfolioState}_t$$

So `LotSnapshot(AAPL, t=50)` and `LotSnapshot(MSFT, t=50)` share the same ledger (`NetST` … `OrdinaryOffsetBudget`) and `Sigma_TE` coordinates — they are correlated through the common $\mathcal{S}_t$.

#### Source 2: Temporal dependence (same lot at consecutive days)

`LotSnapshot(AAPL_lot1, t=50)` and `LotSnapshot(AAPL_lot1, t=51)` — the features `L`, `H`, `SigmaRange`, `DeltaMA50` all evolve smoothly day-to-day, making consecutive snapshots of the same lot nearly collinear.

#### Why neither source breaks the i.i.d. assumption for the ML model

The label $\tilde{y}_{k,t} = f^*(x_{k,t})$ is a **deterministic function of $x_{k,t}$ alone** (oracle) or a deterministic function of the forward price path (soft label). Once you condition on the full feature vector, no other snapshot provides additional information about this lot's label:

$$\tilde{y}_{k,t} \perp \tilde{y}_{j,s} \mid x_{k,t} \quad \forall (j,s) \neq (k,t)$$

The shared portfolio state is not hidden — it is **explicitly encoded** as columns in every snapshot. `NetST`, `NetLT`, `CarryST`, `CarryLT`, `OrdinaryOffsetBudget`, and `Sigma_TE` appear as coordinates in $x_{k,t}$. The cross-sectional correlation is absorbed into the feature representation rather than lurking as latent confounding.

This is the **ergodic collapse** described in §2.2 of the theory memo:

```
Raw panel:     correlated across lots and time
                    ↓  condition on features
Feature space: conditionally independent observations
                    ↓  justified by Markov property of 𝒮_t
ML training:   treat as i.i.d. draws from 𝒟
```

Formally, $Y_{k,t}$ is $\sigma(X_{k,t})$-measurable under the oracle — it is a measurable function of the current feature vector, not of past or future states or other lots. Conditioning on $X_{k,t}$ screens off all temporal and cross-sectional dependence.

#### The one exception — soft label forward-window leakage

$\tilde{y}_{k,t}$ (soft label) is computed over a 30-day forward simulation window. Near the end of the simulation ($t$ close to $T$), the forward windows of different lots overlap — they all observe the same future price paths. This introduces a mild **label correlation** that the conditional independence argument does not dissolve, because the dependence operates through the future, not the current feature vector.

**Practical fix:** time-based train/test split.
- Train: $t = 1, \ldots, 200$ (windows land entirely within the simulation)
- Test: $t = 201, \ldots, 252$ (potential overlap contained within the test period, never contaminates training labels)

This is not a flaw in the model — it is a known, bounded, and handled boundary condition of the soft labelling scheme.

---

## 4. Oracle Conditions — Unified View

**Canonical (v0.25 scalarized):** hard gates survive only where they encode a genuine legal
rule or threshold fact; everything economic is one scalarized objective thresholded at zero:

$$f^*(x) = \underbrace{\mathbb{1}[\ell \leq -\theta_1]}_{\text{loss deep enough}} \cdot \underbrace{\mathbb{1}[\mathcal{W}_{k} > 30]}_{\text{wash-sale clear}} \cdot \underbrace{\mathbb{1}[\sigma_{\text{TE}} \leq \theta_{\max}]}_{\text{tail-risk ceiling}} \cdot \underbrace{\mathbb{1}[U(x) > 0]}_{\text{net benefit}}$$

$$U(x) = \text{taxValue}_k(\text{ledger}_t, h_k, \ell_k) - \lambda\,\sigma_{\text{TE}}^2 - c_{\text{trade}}$$

| Condition | Source field(s) | Value | Grounding |
|------|-------------|-----------|-----------|
| Loss sufficient | `L` | $\theta_1 = 0.02$ | threshold-on-the-loss trigger (industry standard) |
| Wash-sale clear | `WashClock` | $> 30$ calendar days | IRS §1091, both sides (v0.3-1) |
| TE ceiling | `Sigma_TE` | $\theta_{\max} = 0.15$ | tail-only circuit breaker; binds on 0 rows in 20y |
| Net benefit | `TaxValue`, `Sigma_TE` | $U > 0$; $\lambda = 90{,}000$, $c_{\text{trade}} = \$10$ | Wealthfront objective form / Betterment net-benefit test |

The decision boundary is the **level set** $\partial\Omega = \{x : U(x) = 0\}$ — a smooth
curve in $(\sigma_{\text{TE}}, \text{taxValue})$ space — rather than the corner of an
axis-aligned box. The fine-grained TE trade-off lives inside $U$ (priced by $\lambda$);
the marginal TE cap of v0.2 ($\theta_2 = 0.05$) is demoted.

**Retired (v0.2 gated oracle, removed in the pre-v0.3 downsizing):** the conjunction of four
halfspace indicators — the harvest region a convex polytope, per §3.1 of the theory memo:

$$f^*_{\text{gated}}(x) = \mathbb{1}[\ell \leq -\theta_1] \cdot \mathbb{1}[\sigma_{\text{TE}} \leq \theta_2] \cdot \mathbb{1}[G_t^{\text{YTD}} > 0] \cdot \mathbb{1}[\mathcal{W}_t^{A_i} \geq 30]$$

The gains gate was removed because its information is magnitude/timing, not permission
(§2.3); its box-corner geometry — not linear-model capacity — is what made the gated oracle
linearly unrecoverable at scale (measured: `GYTD_Redesign_Plan.md` §6.1). Findings:
`archive/RetiredComponents.md` §6.

---

## 5. Build Dependency Order

The bottom-up dependency graph of `Core/Portfolio/`:

```
LotSnapshot        ← defines what data the ML model needs (feature schema)
      ↑
PortfolioState     ← tracks G_YTD, WashClock, OpenLots (shared mutable state)
      ↑
Lot                ← the atomic unit (one Dirac atom in the measure)
```

Every component in `Simulation/`, `Oracle/`, and `Export/` takes `Lot` and `PortfolioState` as inputs and produces `LotSnapshot` rows as output. Building these three first ensures all downstream method signatures have concrete types to reference.
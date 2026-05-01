# Portfolio Domain Model — Formal Specification
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

**Short/long-term flag** (step function of holding period):
$$s = \mathbb{1}[h_k \geq 365] \in \{0, 1\}$$
```csharp
lot.IsLongTerm(currentDay)           // = HoldingPeriod >= 365
```
This determines the applicable tax rate $\tau(h)$ in the tax-alpha formula (§4 of theory memo):
$$\tau(h) = \begin{cases} \tau_{\text{ST}} & h < 365 \\ \tau_{\text{LT}} & h \geq 365 \end{cases}, \quad \tau_{\text{ST}} > \tau_{\text{LT}}$$

---

## 2. PortfolioState as the State Triple

### 2.1 Formal Definition

The complete portfolio state at day $t$ is the triple:

$$\mathcal{S}_t = \left(\mu_t,\ G_t^{\text{YTD}},\ \mathcal{W}_t\right)$$

| Component | C# member | Type | Meaning |
|-----------|-----------|------|---------|
| $\mu_t$ | `OpenLots` | `List<Lot>` | Full lot measure across all assets |
| $G_t^{\text{YTD}} \in \mathbb{R}$ | `G_YTD` | `decimal` | Net realised gain/loss, calendar year to date |
| $\mathcal{W}_t : \mathcal{S} \to \mathbb{Z}_{\geq 0}$ | `_washClocks` | `Dictionary<string,int>` | Days since last harvest per ticker |

### 2.2 Time Evolution

**AdvanceDay()** implements the daily increment of $\mathcal{W}_t$:
$$\mathcal{W}_{t+1}^{A_i} = \mathcal{W}_t^{A_i} + 1 \quad \forall i \in \mathcal{S}$$

**HarvestLot()** implements the state transition on lot $k$ of asset $A_i$:
1. Realise P&L:
$$\Delta G = q_k (P_t - p_k) \qquad \text{(negative for a loss)}$$
$$G_{t+1}^{\text{YTD}} = G_t^{\text{YTD}} + \Delta G$$

2. Remove atom from the measure:
$$\mu_{t+1}^{A_i} = \mu_t^{A_i} - q_k\,\delta_{(p_k, s_k)}$$

3. Reset the wash-sale clock:
$$\mathcal{W}_{t+1}^{A_i} = 0$$

### 2.3 Sign Convention for $G_t^{\text{YTD}}$ — Critical Detail

$G_t^{\text{YTD}}$ is a **signed scalar** tracking net realised P&L for the year:

- **Positive**: net realised gains dominate (sold positions for more than cost basis in aggregate)
- **Negative**: net realised losses dominate (TLH has offset or exceeded gains)

Harvesting a losing lot ($P_t < p_k$) makes $\Delta G < 0$, pushing $G_t^{\text{YTD}}$ **more negative**. This is correct and intentional — TLH is the act of deliberately realising losses.

The oracle condition $\mathbb{1}[G_t^{\text{YTD}} > 0]$ creates a **self-limiting dynamic**: once you have harvested enough losses to offset all gains, the oracle stops firing. This prevents harvesting losses into negative $G_t^{\text{YTD}}$ territory beyond what the $\$3{,}000$ ordinary income deduction limit can absorb.

| Situation | $G_t^{\text{YTD}}$ sign | Oracle condition $\mathbb{1}[G > 0]$ | Harvest enabled? |
|-----------|--------------------------|---------------------------------------|-----------------|
| Gains banked, no losses yet harvested | $+$ | 1 | ✓ |
| After several loss harvests | $-$ | 0 | ✗ |
| After new gains realised | $+$ again | 1 | ✓ again |

### 2.4 The Wash-Sale Clock $\mathcal{W}_t$

The IRS wash-sale rule prohibits claiming a loss on an asset if a substantially identical asset is purchased within 30 calendar days before or after the sale.

In the simulation:
$$\text{IsWashSaleBlocked}(A_i) = \mathbb{1}\!\left[\mathcal{W}_t^{A_i} < 30\right]$$

This is embedded in the oracle as an additional gate:
$$f^*(x) = \mathbb{1}[\ell \leq -\theta_1] \cdot \mathbb{1}[\sigma_{\text{TE}} \leq \theta_2] \cdot \mathbb{1}[G_t^{\text{YTD}} > 0] \cdot \mathbb{1}[\mathcal{W}_t^{A_i} \geq 30]$$

After harvest, $\mathcal{W}_t^{A_i} \leftarrow 0$ and the clock counts up through `AdvanceDay()` until it reaches 30, at which point the asset becomes harvestable again.

### 2.5 Year-End Reset

$G_t^{\text{YTD}}$ resets on January 1 of each simulated year. Wash-sale clocks intentionally **do not reset** — the IRS 30-day window crosses year-end boundaries.

```csharp
portfolioState.ResetForNewYear();  // G_YTD ← 0m; wash clocks untouched
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
│   ├── G_YTD, Sigma_TE, WashClock    ← 𝒳_portfolio ⊂ ℝ^3  (portfolio-level)
│   ├── R_t, SigmaRange, DeltaMA50, DeltaMA200  ← 𝒳_asset ⊂ ℝ^4  (asset-level)
│   └── AlphaTax, DaysToYE            ← 𝒳_derived  ⊂ ℝ^2  (composite)
│
└── y ∈ 𝒴  (labels — model targets, never inputs)
    ├── Y_Oracle ∈ {0,1}              ← hard label  f*(x)
    └── Y_Soft   ∈ [0,1]              ← soft label  ỹ(x)
```

So $d \approx 15$ before one-hot encoding of `Sector`. The ML model learns $\hat{\eta} : \mathbb{R}^d \to [0,1]$ using the $d$ feature columns as input and `Y_Soft` as the training target (or `Y_Oracle` for hard-label classifiers).

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
| `S` | $s = \mathbb{1}[h \geq 365]$ | `lot.IsLongTerm()` | $\{0,1\}$ |
| `B` | $p_k$ | `lot.CostBasis` | $\mathbb{R}_{>0}$ |
| `W` | $w_k = q_k P_t / V_t$ | derived | $(0,1)$ |
| `K` | lot count for ticker $A_i$ | counted from `OpenLots` | $\mathbb{Z}_{>0}$ |

#### Portfolio-level (from `PortfolioState`)

| Field | Formula | Source |
|-------|---------|--------|
| `G_YTD` | $G_t^{\text{YTD}} \in \mathbb{R}$ | `portfolioState.G_YTD` |
| `Sigma_TE` | $\sigma_{\text{TE}} = \sqrt{\delta w^\top \Sigma\, \delta w}$ | computed in simulation |
| `WashClock` | $\mathcal{W}_t^{A_i} \in \mathbb{Z}_{\geq 0}$ | `portfolioState.GetWashClock()` |

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
| `AlphaTax` | $\alpha_{\text{tax}} = \tau(h)\cdot \lvert G_{\text{lot}} \rvert \cdot \mathbb{1}[G_t^{\text{YTD}} > 0]$ |
| `DaysToYE` | calendar days remaining in simulated tax year |

#### Labels

| Field | Type | Meaning |
|-------|------|---------|
| `Y_Oracle` | $f^*(x) \in \{0,1\}$ | Hard label from oracle gate |
| `Y_Soft` | $\tilde{y}(x) \in [0,1]$ | Soft label from forward simulation window |

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

So `LotSnapshot(AAPL, t=50)` and `LotSnapshot(MSFT, t=50)` share the same `G_YTD` and `Sigma_TE` coordinates — they are correlated through the common $\mathcal{S}_t$.

#### Source 2: Temporal dependence (same lot at consecutive days)

`LotSnapshot(AAPL_lot1, t=50)` and `LotSnapshot(AAPL_lot1, t=51)` — the features `L`, `H`, `SigmaRange`, `DeltaMA50` all evolve smoothly day-to-day, making consecutive snapshots of the same lot nearly collinear.

#### Why neither source breaks the i.i.d. assumption for the ML model

The label $\tilde{y}_{k,t} = f^*(x_{k,t})$ is a **deterministic function of $x_{k,t}$ alone** (oracle) or a deterministic function of the forward price path (soft label). Once you condition on the full feature vector, no other snapshot provides additional information about this lot's label:

$$\tilde{y}_{k,t} \perp \tilde{y}_{j,s} \mid x_{k,t} \quad \forall (j,s) \neq (k,t)$$

The shared portfolio state is not hidden — it is **explicitly encoded** as columns in every snapshot. `G_YTD` and `Sigma_TE` appear as coordinates in $x_{k,t}$. The cross-sectional correlation is absorbed into the feature representation rather than lurking as latent confounding.

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

The full oracle $f^* : \mathcal{X} \to \{0,1\}$ fires when **all four gates** are simultaneously open:

$$f^*(x) = \underbrace{\mathbb{1}[\ell \leq -\theta_1]}_{\text{loss deep enough}} \cdot \underbrace{\mathbb{1}[\sigma_{\text{TE}} \leq \theta_2]}_{\text{tracking error budget}} \cdot \underbrace{\mathbb{1}[G_t^{\text{YTD}} > 0]}_{\text{gains to offset}} \cdot \underbrace{\mathbb{1}[\mathcal{W}_t^{A_i} \geq 30]}_{\text{wash-sale clear}}$$

| Gate | Source field | Threshold |
|------|-------------|-----------|
| Loss sufficient | `L` | $\theta_1 > 0$ (e.g. 2%) |
| TE budget | `Sigma_TE` | $\theta_2 > 0$ |
| Gains available | `G_YTD` | $> 0$ |
| Wash-sale clear | `WashClock` | $\geq 30$ |

This is the conjunction of four halfspace indicators — the harvest region $\Omega$ is a convex polytope in $\mathcal{X}$ as described in §3.1 of the theory memo.

The **sign asymmetry** in the first two fields:
- `L` must be **negative** (loss) and sufficiently below $-\theta_1$
- `G_YTD` must be **positive** (net gains exist to offset against)

These are opposite-sign requirements on features that come from different parts of the state triple — `L` from $\mu_t$ (lot-level), `G_YTD` from the scalar component of $\mathcal{S}_t$ (portfolio-level).

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
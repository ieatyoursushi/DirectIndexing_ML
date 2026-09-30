# ML Mathematical Derivations — Lot Vector Space, Oracle, and Model Family

> **Status: LIVE SPEC** — must match the code; checked by `dotnet run --project src -- docs-check`.
> Index of every symbol ↔ code member ↔ test: [`SymbolTable.md`](SymbolTable.md).

### Gabriel Kung, co-authored with Claude

> **Purpose.** This memo makes explicit, with full domain/codomain typing for every object,
> the mathematics of (1) the `LotStateVector` feature space, (2) the oracle $f^*$ and the
> label family it generates, and (3) each supervised and unsupervised model implemented in
> the ML.NET pillar.
>
> **Schema version: v4 (d = 17), oracle: scalarized (the only mode since the pre-v0.3
> downsizing), validation: purged temporal splits available (v0.26), models: GBT + logistic.**
> Retired objects (the v0.2 gated oracle, RF, elastic net, the linreg demonstrator,
> feature-space PCA/K-means) keep their findings in `archive/RetiredComponents.md`.
>
> **How to read this document — it is layered, by design.** It replaces three parallel memos
> (`ML_Derivations_Simplified.md`, `MLDerivations.md`, `ML_Derivations_Explicit_Rigorous_DollarMath.md`)
> that carried the same content at three rigor levels and consequently drifted out of date
> together. The three audiences are now three *sections* of one document:
>
> | If you want… | Read |
> |---|---|
> | the pipeline in plain language, no measure theory | **§0 Orientation** |
> | the working mathematics with types | **§1–§8** (the body) |
> | every symbol pinned, one object per equation | **Appendix A** |
>
> The level of rigor targeted in the body is that of a stochastic-calculus derivation: every
> variable introduced with its type as a map between sets, every random object with its
> measurable structure — the convention used to write geometric Brownian motion as
> $$dS_t = \mu\,S_t\,dt + \sigma\,S_t\,dW_t,\qquad S_t:\Omega\to\mathbb{R}_{>0},\quad W_t:\Omega\times[0,T]\to\mathbb{R},\quad dW_t\sim\mathcal N(0,dt).$$
>
> **Cross-references.** `data_memo_theory.md` (probability space, boundary geometry),
> `PortfolioMath.md` (lot accounting, the TaxLedger), `SimulationMath.md` (price process, the
> day loop), `GYTD_Redesign_Plan.md` (why the oracle is scalarized), `ValidationHardening_v026.md`
> (why splits are temporal), `MLNetLayer.md` (implementation + sklearn parameter map),
> `MLNetLeakageAudit.md` (training-fold-only invariants).

---

# §0. Orientation — the pipeline without notation

The whole system is one chain:

```text
Portfolio state (lots, tax ledger, tracking error)
      ↓   feature extraction
 Lot features X  — one row = one lot, on one day (17 numbers)
      ↓   the oracle rule f*
 Hard label Y_Oracle ∈ {0,1}  — "harvest this lot today?"
      ↓   run the rule forward 30 days
 Soft labels ∈ [0,1]  — "how often would it fire soon?"
      ↓   supervised learning
 Predicted harvest propensity  η̂(x) ∈ [0,1]
```

**One row is one lot on one day.** It carries 17 numeric features in four groups:

| Group | Features | What they describe |
|---|---|---|
| **Lot** (6) | unrealized return, holding days, long-term flag, cost basis, portfolio weight, lot count | the position itself |
| **Portfolio** (5) | realized gains YTD, loss carryforward, ordinary-offset budget, tracking error, wash-sale clock | shared state — same for every lot that day |
| **Asset** (4) | daily return, intraday range, MA50 deviation, MA200 deviation | what the stock is doing |
| **Derived** (2) | tax value, days to year-end | composites of the above |

**The oracle** is the hand-written harvesting rule — it plays the role of "ground truth." Since
v0.25 it is a *cost-benefit test behind three hard gates*, not a four-way AND:

```text
Harvest = 1  if   loss is deep enough          (≥ 2% below cost)
            AND  wash-sale window has cleared  (≥ 30 days)
            AND  tracking error isn't extreme  (≤ 15% — a circuit breaker)
            AND  the harvest is worth it       (U > 0)

where      U = tax value of harvesting − λ·(tracking error)² − trade cost
```

The fourth condition is the economic one: a harvest must *pay for* the benchmark drift and the
trading friction it causes. The *tax value* is capacity-aware — a dollar of loss you can use
against gains this year is worth more than a dollar you must bank for later.

> **What changed in v0.25, in one sentence.** The old rule had a fourth *gate*, "the portfolio
> must have realized gains this year," which was economically invalid for an individual
> investor (losses carry forward indefinitely), so it became a continuous *value term* inside
> $U$ instead of a pass/fail condition. See `GYTD_Redesign_Plan.md`.

**The soft labels** exist because the hard oracle only answers "today." Freeze the portfolio
state, roll the prices forward 30 days, and ask how often the rule *would* fire. That fraction
is the real prediction target — it depends on unobserved future prices, so it is a genuine
forecasting problem rather than a lookup.

**The models** then learn to predict that propensity from today's features alone. Everything
below formalizes each arrow.

---

# §1. The Lot Vector Space $\mathcal X$

## 1.1 The feature-extraction map

A single observation is produced by the simulation engine through the feature-extraction map
applied to one open lot at one day:

$$
g:\ \mathcal S_t\times P_t\ \longrightarrow\ \mathcal X^{\,|\mathcal K_t|},
\qquad
g(\mathcal S_t, P_t) = \bigl(x_{k,t}\bigr)_{k\in\mathcal K_t},
$$

where $\mathcal S_t$ is the portfolio state at day $t$, $P_t\in\mathbb R_{>0}^{N}$ the price
cross-section, and $\mathcal K_t$ the set of open lots. The $k$-th component
$x_{k,t}\in\mathcal X$ is exactly one `LotStateVector` (minus its labels and metadata).
Projecting onto a single lot,

$$
\phi_{\mathrm{lot}}:(\mathrm{Lot}_k,\mathcal S_t,P_t)\ \longmapsto\ x_{k,t}\in\mathbb R^{17}.
$$

> **Schema note (v0.25).** $d$ moved $15\to17$: the single portfolio coordinate
> $G^{\mathrm{YTD}}$ became the three-field **TaxLedger** block, and the derived coordinate
> $\alpha_{\mathrm{tax}}$ was replaced by the capacity-aware $\mathrm{TaxValue}$.

## 1.2 Coordinates with explicit types

The 17 numeric coordinates (`FeatureLists.NumericFeatures`, in schema order) are the following
maps. For lot $k$ with shares $q_k\in\mathbb Z_{>0}$, cost basis $p_k\in\mathbb R_{>0}$,
purchase day $s_k\in\mathbb Z_{\ge0}$, current price $P_t\in\mathbb R_{>0}$, portfolio value
$V_t\in\mathbb R_{>0}$:

$$
\begin{aligned}
L=\ell_k &= \frac{P_t-p_k}{p_k}\in(-1,\infty)
   &&\text{normalized unrealized return}\\
H=h_k &= t-s_k\in\mathbb Z_{\ge0}
   &&\text{holding period (days)}\\
S &= \mathbf 1[h_k\ge365]\in\{0,1\}
   &&\text{long-term flag}\\
B &= p_k\in\mathbb R_{>0}
   &&\text{cost basis}\\
W=w_k &= \frac{q_kP_t}{V_t}\in(0,1)
   &&\text{portfolio weight}\\
K &= \#\{\text{open lots with ticker }A_i\}\in\mathbb Z_{>0}
   &&\text{lot count}\\[4pt]
G^{\mathrm{net}}_t &\in\mathbb R
   &&\text{RealizedGainsYTD — signed net realized P\&L}\\
C^{\mathrm{fwd}}_t &\in\mathbb R_{\ge0}
   &&\text{LossCarryforward — banked losses}\\
O_t &\in[0,3000]
   &&\text{OrdinaryOffsetBudget — §1211(b) allowance left}\\
\sigma_{\mathrm{TE}} &\in\mathbb R_{\ge0}
   &&\text{annualized tracking error (shared state)}\\
\mathcal W^{A_i}_t &\in\mathbb Z_{\ge0}\cup\{999\}
   &&\text{days since last harvest of }A_i\\[4pt]
R_t &= \frac{P_t-P_{t-1}}{P_{t-1}}\in\mathbb R
   &&\text{daily return}\\
\Sigma\mathrm{Range} &= \frac{H_t-L_t}{P_{t-1}}\in\mathbb R_{\ge0}
   &&\text{intraday range vol proxy}\\
\Delta\mathrm{MA}_{50} &= \frac{P_t-\mathrm{MA}_{50}}{\mathrm{MA}_{50}}\in\mathbb R
   &&\text{50-day MA deviation}\\
\Delta\mathrm{MA}_{200} &= \frac{P_t-\mathrm{MA}_{200}}{\mathrm{MA}_{200}}\in\mathbb R
   &&\text{200-day MA deviation}\\[4pt]
\mathrm{TaxValue} &= g_{\mathrm{tax}}(\mathrm{ledger}_t,h_k,\ell_k)\in\mathbb R_{\ge0}
   &&\text{capacity-aware harvest value (§1.3)}\\
\mathrm{DaysToYE} &\in\mathbb Z_{\ge0}
   &&\text{days to year-end}
\end{aligned}
$$

> ⚠ **Known unit error (ROADMAP finding F6, fixed in v0.3-2).** $h_k=t-s_k$ counts
> **trading** days, but the long-term threshold 365 is a **calendar** quantity, so
> $\mathbf 1[h\ge365]$ means ≈1.45 calendar years. §1222's "more than one year" is
> $\mathrm{date}(t)>\mathrm{date}(s_k)+1\,\mathrm{yr}$ (`SymbolTable.md` `lt_flag_cal`).

where $H_t,L_t$ are the day-$t$ high/low and
$\tau:\mathbb Z_{\ge0}\to\{\tau_{\mathrm{ST}},\tau_{\mathrm{LT}}\}$,
$\tau(h)=\tau_{\mathrm{ST}}\mathbf 1[h<365]+\tau_{\mathrm{LT}}\mathbf 1[h\ge365]$, with
$\tau_{\mathrm{ST}}=0.37>\tau_{\mathrm{LT}}=0.20$.

## 1.3 The tax-value map $g_{\mathrm{tax}}$ (the v0.25 object)

The TaxLedger (`Core/Portfolio/TaxLedger.cs`) is the deterministic Schedule D state
$\mathrm{ledger}_t=(G^{\mathrm{net}}_t,\,C^{\mathrm{fwd}}_t)$ with the derived allowance
$O_t=\max\{0,\ 3000-\max(0,-G^{\mathrm{net}}_t)\}$ and **offset capacity**

$$
\mathrm{cap}_t \;=\; \max\{G^{\mathrm{net}}_t,\,0\} \;+\; O_t \;\in\mathbb R_{\ge0}.
$$

For a lot whose loss in dollars is $D_k=\max\{0,\ (p_k-P_t)q_k\}$ (zero for a winner),

$$
\boxed{\;
g_{\mathrm{tax}}(\mathrm{ledger}_t,h_k,\ell_k)
=\underbrace{\tau(h_k)\cdot\min\{D_k,\ \mathrm{cap}_t\}}_{\text{usable this year, full rate}}
\;+\;\underbrace{\tau_{\mathrm{fut}}\cdot\max\{D_k-\mathrm{cap}_t,\,0\}\cdot\delta}_{\text{banked, discounted}}
\;}
$$

with $\tau_{\mathrm{fut}}=0.20$ and $\delta=0.5$. Economically: $\delta$ is a **hazard-rate
object** — $\delta\approx\Pr(\text{loss absorbed by a gain before death})\times$ (time-value
discount), not a flat rate; carryforward is extinguished at death (Rev. Rul. 74-175), so the
marginal banked dollar is worth strictly less than face value.

> **Legacy (v0.2).** The superseded coordinate was
> $\alpha_{\mathrm{tax}}=\tau(h_k)\cdot|G_{\mathrm{lot}}|\cdot\mathbf 1[G^{\mathrm{YTD}}_t>0]$,
> which had two defects: it applied no capacity $\min$, and it counted a *winner's* $|G|$ as
> if it were a harvestable loss. Both are fixed above. Artifacts written before v0.25 carry
> the old column and are not comparable coordinate-for-coordinate.

## 1.4 Direct-sum decomposition

The coordinates partition by *origin of state* into four blocks:

$$
\mathcal X \;=\; \underbrace{\mathcal X_{\mathrm{lot}}}_{\mathbb R^6}\ \oplus\
\underbrace{\mathcal X_{\mathrm{port}}}_{\mathbb R^5}\ \oplus\
\underbrace{\mathcal X_{\mathrm{asset}}}_{\mathbb R^4}\ \oplus\
\underbrace{\mathcal X_{\mathrm{derived}}}_{\mathbb R^2},
\qquad \dim\mathcal X = 6+5+4+2 = 17.
$$

- $\mathcal X_{\mathrm{lot}}=(L,H,S,B,W,K)$ — intrinsic to the lot.
- $\mathcal X_{\mathrm{port}}=(G^{\mathrm{net}}_t,C^{\mathrm{fwd}}_t,O_t,\sigma_{\mathrm{TE}},\mathcal W^{A_i}_t)$ — shared state $\mathcal S_t$, identical across every lot on day $t$.
- $\mathcal X_{\mathrm{asset}}=(R_t,\Sigma\mathrm{Range},\Delta\mathrm{MA}_{50},\Delta\mathrm{MA}_{200})$ — from the price series.
- $\mathcal X_{\mathrm{derived}}=(\mathrm{TaxValue},\mathrm{DaysToYE})$ — composites.

The categorical field $z=\texttt{Sector}\in\mathcal Z$ is appended separately (§3.3). One
in-memory field, $q_k$ (`Shares`), is carried on the snapshot but **never exported** — the
soft-label builders need it to re-dollarize $D_k$ along forward price paths (§2.4); it is not
a feature.

## 1.5 The observation pair

Each row of `lots.csv` is a point $(x_{k,t},\,y_{k,t})\in\mathcal X\times\mathcal Y$ where the
label space is a five-fold product (§2.5):

$$
\mathcal Y=\underbrace{\{0,1\}}_{Y_{\mathrm{Oracle}}}\times\underbrace{[0,1]^2}_{Y_{\mathrm{Soft\,BT}},\,Y_{\mathrm{Soft\,GBM}}}\times\underbrace{\mathbb R_{\ge0}}_{Y_{\mathrm{TaxValue}}}\times\underbrace{\mathbb R}_{Y_{\mathrm{Utility}}}.
$$

The **panel index** is $(i,t)$ with $i$ the lot/stock and $t$ the day. Under the oracle,
$Y_{i,t}$ is $\sigma(X_{i,t})$-measurable, so observations are treated as conditionally
independent given features. **This assumption is known to be imperfect and is load-bearing
only for the supervised stage**: offset capacity $\mathrm{cap}_t$ and the TE budget are
*shared depletable resources*, which couples lot decisions within a day — the structural
reason a policy (v0.4), not a per-row classifier, is the eventual object
(`GYTD_Redesign_Plan.md` §8).

---

# §2. The Targets: the Oracle and its Label Family

## 2.1 The oracle map $f^*$ (scalarized, v0.25)

`OracleBoundary.Label` realizes the deterministic measurable function

$$
f^*:\mathcal X\to\{0,1\},\qquad
f^*(x)=\underbrace{\mathbf 1[\ell\le-\theta_1]}_{\text{loss depth}}\cdot
\underbrace{\mathbf 1[\mathcal W^{A_i}_t\ge\theta_3]}_{\text{IRS §1091}}\cdot
\underbrace{\mathbf 1[\sigma_{\mathrm{TE}}\le\theta_{\max}]}_{\text{tail circuit breaker}}\cdot
\underbrace{\mathbf 1[U(x)>0]}_{\text{net-benefit test}},
$$

with the **scalarized objective**

$$
\boxed{\ U(x)\;=\;\mathrm{TaxValue}(x)\;-\;\lambda\,\sigma_{\mathrm{TE}}^2\;-\;c_{\mathrm{trade}}\ \in\mathbb R\ }
$$

and project constants $\theta_1=0.02$, $\theta_3=30$, $\theta_{\max}=0.15$,
$\lambda=90{,}000$, $c_{\mathrm{trade}}=\$10$ (`OracleConfig`). Note $\mathrm{TaxValue}$
already carries the rates $\tau(h),\tau_{\mathrm{fut}}$ internally (§1.3), so $U$ applies no
further rate factor.

**Geometry — the essential change.** The v0.2 oracle was an intersection of four halfspaces,
$\Omega=H_1\cap H_2\cap H_3\cap H_4$: a convex **polytope**, whose corner no hyperplane can
represent. The v0.25 oracle's decision boundary is the **level set**

$$
\partial\Omega=\{x\in\mathcal X:\ U(x)=0\}
=\Bigl\{x:\ \mathrm{TaxValue}(x)=\lambda\sigma_{\mathrm{TE}}^2+c_{\mathrm{trade}}\Bigr\},
$$

a smooth (parabolic in $(\sigma_{\mathrm{TE}},\mathrm{TaxValue})$) hypersurface intersected
with three halfspaces. **This is why the measured tree-over-linear advantage on the oracle
target collapsed** from a 0.155 to a 0.015 PR-AUC gap after v0.25: much of it was the box
corner, not irreducible problem structure (`GYTD_Redesign_Plan.md` §6.1).

> **Retired (v0.2 gated oracle).** The ablation baseline
> $f^*_{\mathrm{gated}}=\mathbf 1[\ell\le-\theta_1]\cdot\mathbf 1[\sigma_{\mathrm{TE}}\le0.05]\cdot\mathbf 1[G^{\mathrm{YTD}}_t>0]\cdot\mathbf 1[\mathcal W\ge\theta_3]$
> was removed in the pre-v0.3 downsizing (schema v4); its measured findings are in
> `archive/RetiredComponents.md` §6. The methodological lesson survives: the *acting* oracle
> changes the trajectory (which lots are harvested → wash clocks → ledger → which rows
> exist), so any future oracle ablation is a **separate simulation run**, not a second label
> column of one run.

## 2.2 The hard label

$$
Y_{\mathrm{Oracle}}=f^*(x)\in\{0,1\}.
$$

Because $f^*$ is a deterministic function of the current-timestep features, it is *perfectly
recoverable in principle* — which makes it the project's **leakage control**: any split
scheme under which a model fails to rank it near-perfectly is a split scheme with a problem
(§3.2, and `ValidationHardening_v026.md`).

## 2.3 The `soft_bt` label as a path functional

The soft backtest label upgrades the *pointwise* gate $f^*$ to a *forward-looking frequency*.
Fix a snapshot lot at day $t_0$ with **frozen portfolio state**

$$
\bigl(\mathrm{ledger}_{t_0}\ (\text{hence }\mathrm{cap}_{t_0}),\ \sigma_{\mathrm{TE}},\ \mathcal W_0:=\mathcal W^{A_i}_{t_0},\ p_k,\ q_k\bigr),
$$

held constant (`SoftLabelBuilder.ComputeBT`), and let $\{P_{t_0+s}\}_{s=1}^{W}$ be the
**actual realized** closing prices over the forward window $W=30$ trading days. Three things
advance across the window: the price $P_{t_0+s}$, the wash-sale clock $\mathcal W_0+s$, and —
new in v0.25 — the holding period $h_k+s$, so $\tau(h)$ can flip short→long *inside* the
window.

Define the per-step forward return, the re-dollarized loss, and the per-step firing:

$$
\ell_s := \frac{P_{t_0+s}-p_k}{p_k},\qquad
D_s := \max\{0,\ (p_k-P_{t_0+s})q_k\},
$$

$$
b_s := f^*\!\Bigl(\ell_s,\ \sigma_{\mathrm{TE}},\ g_{\mathrm{tax}}(D_s,\,h_k+s,\,\mathrm{cap}_{t_0}),\ \mathcal W_0+s\Bigr)\in\{0,1\}.
$$

The label is the **time-average firing frequency** over the window:

$$
\boxed{\ \tilde y_{\mathrm{BT}}(x)\ :=\ \frac1W\sum_{s=1}^{W} b_s\ =\ \frac{\#\{s\in[1,W]: \text{oracle fires}\}}{W}\ \in\ \Bigl\{0,\tfrac1W,\dots,1\Bigr\}.\ }
$$

This is exactly `ComputeBT`: `oracleDays / Window` with `Window = 30`. When fewer than $W$
forward days remain ($t_0+W\ge t_{\max}$) the functional is undefined and the code returns
`NaN`; those rows are dropped before modeling (every trainer's `SelectTarget`).

**Type summary.** $\tilde y_{\mathrm{BT}}:\mathcal X\to[0,1]\cup\{\mathrm{NaN}\}$; each
$b_s:\mathcal X\times\mathbb Z_{>0}\to\{0,1\}$; the window sum is a deterministic functional
of the realized price segment $(P_{t_0+1},\dots,P_{t_0+W})\in\mathbb R_{>0}^{W}$.

**Crucially, the oracle enters as a black box.** Swapping $f^*$'s internals (v0.2 AND →
v0.25 gates·$\mathbf 1[U>0]$) required *zero* change to this functional — the payoff of
`OracleBoundary` being stateless and referentially transparent.

## 2.4 Interpretation: from boundary indicator to urgency field

$\tilde y_{\mathrm{BT}}(x)$ estimates the **harvest urgency** — the fraction of the near
future during which this lot *would* be harvestable. A lot deep in a persistent drawdown fires
on most of the next 30 days ($\to1$); a lot grazing the threshold fires intermittently. It is
a sampled estimate of the interior posterior $\eta$, whose level sets

$$
L_c=\{x\in\mathcal X:\eta(x)=c\},\qquad c\in[0,1],
$$

are $(d-1)$-dimensional **contours of harvest urgency**. The hard oracle only exposes
$\partial\Omega$; the soft label exposes the graded field inside it.

Two *orthogonal* axes of softness now coexist and should not be conflated:

| Object | Axis | Question |
|---|---|---|
| $U(x)$ | **cross-sectional** | how valuable is this lot *right now*? |
| $\tilde y_{\mathrm{BT}}(x)$ | **temporal** | how often would it clear the test over 30 days? |

## 2.5 The full label family (schema v4)

| Label | Type | Definition |
|---|---|---|
| $Y_{\mathrm{Oracle}}$ | $\{0,1\}$ | $f^*(x)$ — the acting oracle |
| $Y_{\mathrm{Soft\,BT}}$ | $[0,1]\cup\{\mathrm{NaN}\}$ | §2.3, realized path |
| $Y_{\mathrm{Soft\,GBM}}$ | $[0,1]$ | §2.6, simulated ensemble |
| $Y_{\mathrm{TaxValue}}$ | $\mathbb R_{\ge0}$ | $=g_{\mathrm{tax}}(\mathrm{ledger}_t,h_k,\ell_k)$, the regression target |
| $Y_{\mathrm{Utility}}$ | $\mathbb R$ | $=U(x)$, raw objective before thresholding |

Two disciplines attach to this table:

1. **Leakage rule for $Y_{\mathrm{TaxValue}}$.** It equals the `TaxValue` *feature* by
   construction, so any regression on it must exclude that feature — enforced by
   `FeatureLists.NumericFeaturesTaxValueRegression` (16 features). The task is recovering
   $g_{\mathrm{tax}}$ from raw state, not copying a column.
2. **$Y_{\mathrm{Utility}}$ is per-lot and one-step.** Summing it across the lots harvested on
   one day charges the *shared* $\sigma_{\mathrm{TE}}$ once per lot, so it is not by itself a
   sequential reward; the v0.4 reward is pinned at portfolio level in
   `spec/SymbolTable.md` §I.

The binary target consumed by the classification trainers is, for `target = "soft_bt"`,
$y:=\mathbf 1[\tilde y_{\mathrm{BT}}(x)>0]$ ("fires at least once"), and for
`target = "oracle"`, $y:=Y_{\mathrm{Oracle}}$.

## 2.6 The GBM soft label

`Y_Soft_GBM` replaces the realized path with Monte-Carlo paths under geometric Brownian
motion. The price process solves

$$
dS_u=\mu S_u\,du+\sigma S_u\,dW_u,\qquad S_{t_0}=P_{t_0},\ dW_u\sim\mathcal N(0,du),
$$

discretized at $\Delta=1/252$ with the Itô-corrected log-Euler scheme
(`GbmSimulator.SimulatePaths`):

$$
S_{s} = S_{s-1}\exp\!\Bigl((\mu-\tfrac12\sigma^2)\Delta + \sigma\sqrt\Delta\,Z_s\Bigr),\quad Z_s\overset{\mathrm{iid}}\sim\mathcal N(0,1),
$$

with $\mu=0$ (risk-neutral default) and $\sigma$ the annualized trailing-21-day realized vol
($\sigma=\widehat{\mathrm{std}}(r)\sqrt{252}$, fallback $0.20$). The label is the
**first-passage frequency** over $N_{\mathrm{paths}}=200$ paths,

$$
\tilde y_{\mathrm{GBM}}(x)=\frac1{N_{\mathrm{paths}}}\sum_{p=1}^{N_{\mathrm{paths}}}\mathbf 1\!\bigl[\exists\,s\in[1,W]:b_s^{(p)}=1\bigr]\ \in[0,1],
$$

an unbiased Monte-Carlo estimator of $\mathbb P(\exists s\le W:\text{oracle fires})$ under the
GBM measure (`FractionFiring` breaks a path on first fire — once harvested, later moves on
that path are counterfactual). The $Z_s$ come from Box–Muller,
$Z=\sqrt{-2\ln U_1}\cos(2\pi U_2)$. The contrast with §2.3 is exactly *realized path* (one
trajectory) vs. *simulated ensemble* (expectation over the price law).

---

# §3. Train/Test Protocol and Preprocessing

## 3.1 Protocol (identical across all supervised trainers)

Given filtered rows $D=\{(x_i,y_i)\}_{i=1}^N$:

1. **Split** $D=D_{\mathrm{tr}}\sqcup D_{\mathrm{te}}$, test fraction $0.20$ — via the
   `DataSplit` facade (§3.2), which dispatches on `SplitPolicy`. Both models share the
   identical split, which is what makes the champion-vs-control comparison valid.
2. **Median imputation** fit on $D_{\mathrm{tr}}$ only (`MedianImputer.Fit`), applied to both
   folds — a training-fold-only invariant.
3. **Balanced class weights** on $D_{\mathrm{tr}}$ only:
   $$w_c=\frac{N_{\mathrm{tr}}}{K\,n_c},\quad c\in\{0,1\},\ K=2,$$
   the ML.NET analogue of sklearn's `class_weight='balanced'`.
4. **5-fold CV** over a finite hyperparameter grid, on $D_{\mathrm{tr}}$ only, scored by
   **PR-AUC**.
5. **Refit** on all of $D_{\mathrm{tr}}$ with the CV-best hyperparameters.
6. **Evaluate once** on $D_{\mathrm{te}}$ — only after the CV leaderboard is written (§7).

## 3.2 Two partition schemes (v0.26)

$$
\text{SplitPolicy}\in\{\textsf{StratifiedRandom},\ \textsf{TemporalPurged}\}.
$$

**Stratified random** (legacy default) shuffles within each class, so $D_{\mathrm{te}}$ is a
random 20% of *all* years and inherits the full-sample prevalence exactly.

**Temporal purged** splits chronologically on $\texttt{Timestep}$ at a row-mass boundary
$T^\star$, and *purges* an embargo band of width $E$:

$$
D_{\mathrm{te}}=\{i: t_i\ge T^\star\},\qquad
D_{\mathrm{tr}}=\{i: t_i\le T^\star-E-1\},\qquad E\ge W=30 .
$$

The condition $E\ge W$ is the point: a training row at $t\le T^\star-E-1$ has its entire label
window $(t,t+W]$ ending strictly before $T^\star$, so **no training label can see the test
period**. Without it, the 30-day forward windows of rows adjacent to the boundary overlap the
test set. `PurgedFolds` applies the same purge on *both* sides of each interior CV block.

**The measured consequence (v0.26).** ROC-AUC held or rose under the temporal scheme while
PR-AUC fell sharply. Since ROC-AUC is prevalence-insensitive and the deterministic oracle
target stayed at $\approx1.0$ across all schemes, **ranking leakage is ruled out**; the PR-AUC
move is the prevalence collapse of the aged-out tail (§5.2). Standing rule: report ROC-AUC,
PR-AUC, *and* test-period positive rate together.

## 3.3 The preprocessing map $\phi$

`PreprocessingPipeline.Build` realizes

$$
\phi(x,z)=\Bigl[\,\underbrace{\mathrm{Norm}(x)}_{\in\mathbb R^{17}}\ \big\Vert\ \underbrace{\mathrm{OneHot}(\mathrm{Clean}(z))}_{\in\{0,1\}^{m}}\,\Bigr]\in\mathbb R^{d},\quad d=17+m,
$$

where $\mathrm{Norm}(x)_j=(x_j-\mu_j)/\sigma_j$ with $(\mu_j,\sigma_j)$ estimated on the
training fold, and $m=|\mathcal Z_{\mathrm{tr}}|$ is the sector vocabulary learned on the
training fold. For tree models $\mathrm{Norm}$ is order-preserving per coordinate, hence a
no-op for split selection (kept for schema consistency). All objectives below are functions of
$\phi_i:=\phi(x_i,z_i)$.

---

# §4. Supervised Models (GBT + logistic)

"$\sum_i$" abbreviates $\sum_{i\in D_{\mathrm{tr}}}$; $w_{y_i}$ is the balanced weight of
example $i$'s class.

## 4.1 Logistic regression — `LbfgsLogisticRegression`

Hypothesis: a calibrated linear-logit posterior $\hat\eta:\mathbb R^d\to(0,1)$,

$$
\hat\eta(\phi)=\sigma(w^\top\phi+b)=\bigl(1+e^{-(w^\top\phi+b)}\bigr)^{-1},\qquad w\in\mathbb R^d,\ b\in\mathbb R.
$$

Weighted regularized cross-entropy:

$$
\min_{w,b}\ \sum_i w_{y_i}\Bigl[-y_i\log\hat\eta(\phi_i)-(1-y_i)\log\bigl(1-\hat\eta(\phi_i)\bigr)\Bigr]+\lambda\lVert w\rVert_2^2,
$$

with $\lambda=1/C$, $C\in\{0.01,0.1,1,10\}$ by CV. The gradient
$\nabla_w=\sum_i w_{y_i}(\hat\eta(\phi_i)-y_i)\phi_i+2\lambda w$ vanishes at the optimum;
L-BFGS solves it.

## 4.2 Gradient-boosted trees — `FastTree`

A boosted additive ensemble fit to the **functional gradient** of logistic loss. With
$F_0\equiv\mathrm{logit}(\bar y)$, iterate for $m=1,\dots,M$:

$$
F_m(\phi)=F_{m-1}(\phi)+\nu\,f_m(\phi),\qquad
f_m=\arg\min_{f\in\mathcal T_J}\sum_i w_{y_i}\bigl(r_{i}^{(m)}-f(\phi_i)\bigr)^2,
$$

where $\mathcal T_J$ is the class of $J$-leaf regression trees, $\nu$ the learning rate, and

$$
r_i^{(m)}=-\frac{\partial \ell_{\log}(y_i,F)}{\partial F}\Big|_{F=F_{m-1}(\phi_i)}=y_i-\sigma\!\bigl(F_{m-1}(\phi_i)\bigr).
$$

Final score $F_M(\phi)\in\mathbb R$ is Platt-calibrated,
$\hat\eta(\phi)=\sigma(aF_M(\phi)+b)$. Grid (8 configs): $M\in\{100,200\}$,
$\nu\in\{0.10,0.05\}$, $J\in\{20,31\}$.

## 4.3 Why exactly these two (and what was retired)

The pair is a *measurement instrument*, not a zoo: logistic regression is the linear control
whose gap to GBT measures how much non-linearity the target actually has. Measured
(`decisions/GYTD_Redesign_Plan.md` §6.1): the gap is ≈0.015 on the scalarized oracle target
(its level-set boundary is nearly linear-recoverable) and ≈0.19 on the temporal soft target
(genuinely non-linear, oracle-invariant). Retired in the pre-v0.3 downsizing, with their
findings in `archive/RetiredComponents.md`: the elastic-net logistic (same hypothesis class
as §4.1, dominated; every linear model chose its weakest penalty), the random forest (bagging
blurs the thin positive region boosting sharpens), and the least-squares-on-$\{0,1\}$
demonstrator (codomain $\mathbb R$, heteroskedastic Bernoulli errors — not BLUE).

## 4.4 Tax-value regression (v0.25) — the regression analogue

`TaxValueRegressionPipeline` regresses the continuous $Y_{\mathrm{TaxValue}}$ on the
16-feature set (excluding `TaxValue` itself, §2.5):

$$
\min_{\theta}\ \sum_i\bigl(Y^{\mathrm{TaxValue}}_i-\hat g_\theta(\phi_i)\bigr)^2,
$$

for $\hat g_\theta\in\{\text{SDCA linear},\ \text{FastTree}\}$. The target is
**zero-inflated** ($\approx98\%$ of rows have no harvestable loss), so alongside
RMSE/MAE/$R^2$ the artifact reports RMSE on the $\{Y^{\mathrm{TaxValue}}>0\}$ subset, where
the structure lives.

**Why it is a clean experiment.** $g_{\mathrm{tax}}$ (§1.3) is a $\min$/$\max$-kinked function
with a discrete rate jump at $h=365$ — exactly what a hyperplane cannot represent and
axis-aligned splits can. Measured: $R^2\approx0.10$ (linear) vs $\approx0.92$ (trees). This is
the regression mirror of the classification story, and the fitted $\hat g$ is a warm-start
candidate for the v0.4 value function.

---

# §5. Evaluation Functionals

## 5.1 Definitions

Given scored test rows $\{(y_i,\hat\eta_i)\}$ sorted by $\hat\eta$ descending, with
$n_+=\#\{y_i=1\}$, $n_-=\#\{y_i=0\}$, sweep the threshold and accumulate
$\mathrm{TP}(\tau),\mathrm{FP}(\tau)$:

$$
\mathrm{TPR}(\tau)=\frac{\mathrm{TP}}{n_+},\quad
\mathrm{FPR}(\tau)=\frac{\mathrm{FP}}{n_-},\quad
\mathrm{Prec}(\tau)=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\quad
\mathrm{Rec}(\tau)=\mathrm{TPR}(\tau).
$$

ROC-AUC is $\int_0^1\mathrm{TPR}\,d(\mathrm{FPR})$; PR-AUC is the average-precision step
integral $\sum_k(\mathrm{Rec}_k-\mathrm{Rec}_{k-1})\mathrm{Prec}_k$; $F_1$ is the harmonic
mean of precision and recall, reported at $\tau=0.5$ and at the maximizing $\tau^\star$.

## 5.2 The prevalence asymmetry — why both AUCs are reported

The two AUCs have **different no-skill baselines**, and confusing them is the single easiest
misreading of this project's numbers:

$$
\mathrm{ROC\text{-}AUC}_{\text{no-skill}}=\tfrac12\quad\text{(prevalence-invariant)},
\qquad
\mathrm{PR\text{-}AUC}_{\text{no-skill}}=p:=\frac{n_+}{n_++n_-}\quad\text{(prevalence-bounded)}.
$$

A no-skill classifier has $\mathrm{Prec}(\tau)\equiv p$ at every recall, so its PR curve is the
horizontal line $y=p$ and its area is $p$. Hence **the "0.5 = random" rule is ROC-only**; a
PR-AUC of $0.46$ at $p=0.0022$ is $\approx209\times$ no-skill, not "barely better than
chance." They coincide only on balanced data ($p=\tfrac12$).

A second-order consequence matters for model comparison: PR-AUC's prevalence sensitivity is
**largest for imperfect rankers**. A perfect ranker has $\mathrm{Prec}\equiv1$ up to the last
positive, so its PR-AUC $\approx1$ at *any* $p$. This is why, under the temporal scheme, the
deterministic oracle target barely moved while the stochastic soft target fell sharply — the
same prevalence crash, filtered through different ranking quality.

**PR-AUC remains the CV selection criterion** (the positive class is rare and precision/recall
geometry is the operating regime), but per the v0.26 standing rule, ROC-AUC and the
test-period prevalence are reported alongside it — never PR-AUC alone.

---

# §6. Unsupervised Structure — retired from 𝒳, moving to Σ̂

PCA on the standardized feature covariance $C\in\mathbb R^{17\times17}$ and K-means on
per-symbol technical aggregates were course-era diagnostics; both found no exploitable
low-dimensional or clustered structure in lot-state space (`archive/RetiredComponents.md`
§4–§5). Their successor is a **different object**: the eigendecomposition of the
point-in-time **return** covariance $\hat\Sigma_t\in\mathbb R^{N\times N}$ (v0.3-4), which
serves covariance cleaning (Marchenko–Pastur), the agent's factor summaries $U_k^\top\delta w$
(v0.4b), and core+reserve universe reduction (v0.45, issue #6).

---

# §7. Champion Selection (test-set discipline)

Both models are CV-scored by mean PR-AUC on $D_{\mathrm{tr}}$ and ranked on one leaderboard;
the champion is the argmax,

$$
\mathcal M^\star=\operatorname*{arg\,max}_{\mathcal M\in\{\mathrm{GBT},\,\mathrm{logistic}\}}\ \widehat{\mathrm{PRAUC}}_{\mathrm{CV}}(\mathcal M),
$$

computed by the pure function `MLnetPipeline.SelectChampion` (testable without any test set).
Only **after** the leaderboard is written are both models refit and evaluated on
$D_{\mathrm{te}}$. Evaluating the control on test selects nothing on test: the pair is
pre-registered and the selection rule reads CV only. Because the split is shared, the
leaderboard comparison and the test estimates are on the same partition.

---

# §8. The Chain of Approximations

$$
f^*_{\mathrm{true}}\ \xrightarrow{\ \text{myopic}\ }\ f^*_{\mathrm{oracle}}\ \xrightarrow{\ \text{forward window}\ }\ \tilde y_{\mathrm{BT}}\ \xrightarrow{\ \text{ERM over }\mathcal H\ }\ \hat\eta.
$$

The mechanistic oracle $f^*$ fixes the boundary $\partial\Omega$ (Source 1 of tax alpha —
irreducible by any classifier). The soft label samples the interior urgency field, and the
supervised models learn $\hat\eta$, whose level sets prioritize *which* in-region lots to
harvest *first* (Source 2 — closeable, the project's empirical contribution).

**Where each arrow is known to be lossy**, and what closes it:

| Arrow | Loss | Closed by |
|---|---|---|
| true $\to$ oracle | myopia: one-step $U$, no sequential value | v0.4 policy layer (reward $=\sum_t\gamma^tU$) |
| oracle $\to$ soft | frozen portfolio state over the window | endogenous state in the v0.4 environment |
| soft $\to$ $\hat\eta$ | ERM error, finite $\mathcal H$ | model capacity — measured, and small for trees |

The conditional-independence assumption of §1.5 is the arrow that *cannot* be closed
supervised: shared depletable resources couple lots within a day, which is the structural
argument for a policy object rather than a per-row classifier.

---

# Appendix A. Standing Definitions (every symbol pinned)

*This appendix is the maximally explicit layer: one object per display, type stated at the
point of use. It restates §0–§3 for a reader who wants no inference.*

The feature space is

$$
\mathcal X\subset\mathbb R^{17}.
$$

A feature vector is

$$
x\in\mathcal X,\qquad x=\langle x_1,\dots,x_{17}\rangle.
$$

A lot-indexed observation at lot $k$ and day $t$ is

$$
x_{k,t}\in\mathcal X.
$$

The binary label space is

$$
\mathcal Y=\{0,1\}.
$$

A dataset of $N$ examples is

$$
D=\{(x_i,y_i)\}_{i=1}^{N},\qquad x_i\in\mathcal X,\qquad y_i\in\{0,1\}.
$$

The ideal posterior function is

$$
\eta(x)=\mathbb P(Y=1\mid X=x),\qquad \eta:\mathcal X\to[0,1].
$$

The supervised learning goal is to learn an estimate

$$
\hat\eta:\mathcal X\to[0,1],
$$

interpreted as the model's estimated probability that lot $x$ should be harvested.

The oracle is a deterministic measurable map

$$
f^*:\mathcal X\to\{0,1\}.
$$

The scalarized objective is

$$
U:\mathcal X\to\mathbb R.
$$

The tax-value map is

$$
g_{\mathrm{tax}}:\ \mathcal L\times\mathbb Z_{\ge0}\times\mathbb R\ \to\ \mathbb R_{\ge0},
$$

where $\mathcal L$ is the space of ledger states
$\mathrm{ledger}_t=(G^{\mathrm{net}}_t,C^{\mathrm{fwd}}_t)\in\mathbb R\times\mathbb R_{\ge0}$.

The offset capacity is

$$
\mathrm{cap}:\ \mathcal L\to\mathbb R_{\ge0}.
$$

The preprocessing map is

$$
\phi:\ \mathcal X\times\mathcal Z\ \to\ \mathbb R^{d},\qquad d=17+m .
$$

The backtest soft label is

$$
\tilde y_{\mathrm{BT}}:\ \mathcal X\to[0,1]\cup\{\mathrm{NaN}\}.
$$

The GBM soft label is

$$
\tilde y_{\mathrm{GBM}}:\ \mathcal X\to[0,1].
$$

The indicator is $\mathbf 1[\,\cdot\,]:\{\text{prop}\}\to\{0,1\}$ and the logistic sigmoid is
$\sigma:\mathbb R\to(0,1)$, $\sigma(z)=(1+e^{-z})^{-1}$.

### A.1 Constants

The constant table lives in exactly one place — [`SymbolTable.md`](SymbolTable.md) §K — where
`docs-check` compares every value against its C# declaration. (A second copy here is how
constants drift.)

---|---|---|
| $\theta_1$ loss threshold | $0.02$ | `OracleConfig.LossThreshold` |
| $\theta_3$ wash window | $30$ days | `OracleConfig.WashSaleDays` (IRS §1091) |
| $\theta_{\max}$ TE ceiling | $0.15$ | `OracleConfig.TrackingErrorCeiling` |
| $\lambda$ TE price | $90{,}000$ | `OracleConfig.Lambda` |
| $c_{\mathrm{trade}}$ | $\$10$ | `OracleConfig.CTrade` |
| $\tau_{\mathrm{ST}}$ | $0.37$ | `TaxLedger.TauShortTerm` |
| $\tau_{\mathrm{LT}}$ | $0.20$ | `TaxLedger.TauLongTerm` |
| $\tau_{\mathrm{fut}}$ | $0.20$ | `TaxLedger.TauFuture` |
| $\delta$ carryforward discount | $0.5$ | `TaxLedger.CarryforwardDiscount` |
| ordinary offset cap | $\$3{,}000$/yr | `TaxLedger.AnnualOrdinaryOffsetCap` (§1211(b)) |
| $W$ label horizon | $30$ days | `SoftLabelBuilder.Window` |
| $E$ embargo | $\ge30$ days | `SplitPolicy.EmbargoDays` |

---

## Cross-reference

- `data_memo_theory.md` / `_part2.md` — probability space, boundary geometry, ERM theory, the v0.3–v0.4 program.
- `PortfolioMath.md` — lot accounting, the state triple, the TaxLedger.
- `SimulationMath.md` — price process, day loop, soft-label closures.
- `GYTD_Redesign_Plan.md` — why the oracle is scalarized; §6.1 the measured ablation.
- `ValidationHardening_v026.md` — purged temporal splits and the leakage-vs-prevalence diagnosis.
- `MLNetLayer.md` — implementation architecture + ML.NET ↔ sklearn parameter map.
- `MLNetLeakageAudit.md` — training-fold-only invariants.
- `src/Core/Portfolio/LotStateVector.cs` — canonical d=17 schema · `TaxLedger.cs` — $g_{\mathrm{tax}}$.
- `src/Core/Oracle/OracleBoundary.cs`, `OracleConfig.cs` — $f^*$, $U$, constants.
- `src/Core/Simulation/SoftLabelBuilder.cs`, `GbmSimulator.cs` — $\tilde y_{\mathrm{BT}}$, $\tilde y_{\mathrm{GBM}}$.
- `src/ML/CSharp/MLNet/Splits/` — `DataSplit`, `SplitPolicy`, `TemporalSplit`.
- `src/ML/CSharp/MLNet/Models/*.cs` — trainer implementations.

# ML Mathematical Derivations — Lot Vector Space and Model Family

### Gabriel Kung, Co-Authored by Claude Sonnet

> **Purpose.** This memo is written *in service of clarity*: it makes explicit, with
> full domain/codomain typing for every object, the mathematics of (1) the
> `LotStateVector` feature space, (2) the `soft_bt` label as it is actually
> constructed in the simulation layer, and (3) each supervised and unsupervised
> model implemented in the ML.NET pillar. It is the mathematical companion to the
> conceptual `data_memo_theory.md` and the implementation-facing
> `MLNetPipeline.md` / `MLNetSemanticReconciliation.md`.
>
> The level of rigor targeted is that of a stochastic-calculus derivation: every
> variable is introduced with its type as a map between sets, every random object
> with its measurable structure. For comparison, the convention used throughout
> is the same one used to write geometric Brownian motion as
> $$dS_t = \mu\,S_t\,dt + \sigma\,S_t\,dW_t,\qquad S_t:\Omega\to\mathbb{R}_{>0},\quad W_t:\Omega\times[0,T]\to\mathbb{R},\quad dW_t\sim\mathcal N(0,dt).$$
>
> **Cross-references.** `DataMemo/data_memo_theory.md` (probability space, oracle
> boundary geometry, Stokes/currents analogy), `DataMemo/PortfolioMath.md` (lot
> accounting), `DataMemo/SimulationMath.md` (price process), `src/Core/Portfolio/LotStateVector.cs`
> (canonical schema), `src/Core/Oracle/OracleBoundary.cs` (the map $f^*$),
> `src/Core/Simulation/SoftLabelBuilder.cs` + `GbmSimulator.cs` (soft labels),
> `src/ML/CSharp/MLNet/Models/*.cs` (trainers).

---

## 0. Notation and Standing Conventions

| Symbol | Type | Meaning |
|---|---|---|
| $\Omega_{\mathrm{prob}}$ | sample space | underlying probability space $(\Omega_{\mathrm{prob}},\mathcal F,\mathbb P)$ for price randomness |
| $\mathcal X$ | $\subset\mathbb R^d$ | feature space (Borel measurable) |
| $\mathcal Y$ | $\{0,1\}$ or $[0,1]$ | label space (hard / soft) |
| $v_{k,t}$ | `LotStateVector` | one lot $k$ at simulation day $t$ — one row of `lots.csv` |
| $x$ | element of $\mathcal X$ | numeric+categorical feature vector extracted from $v$ |
| $\phi$ | $\mathcal X\to\mathbb R^{d}$ | preprocessing map (normalize $\oplus$ one-hot) |
| $f^*$ | $\mathcal X\to\{0,1\}$ | the mechanistic oracle (`OracleBoundary.Label`) |
| $\eta$ | $\mathcal X\to[0,1]$ | posterior $\eta(x)=\mathbb P(Y=1\mid X=x)$ |
| $\tilde y_{\mathrm{BT}}$ | $\mathcal X\to[0,1]$ | backtest soft label (`Y_Soft_BT`) |
| $\tilde y_{\mathrm{GBM}}$ | $\mathcal X\to[0,1]$ | Monte-Carlo soft label (`Y_Soft_GBM`) |

Throughout, $\mathbf 1[\,\cdot\,]:\{\text{prop}\}\to\{0,1\}$ is the indicator and $\sigma(z)=(1+e^{-z})^{-1}$ is the logistic sigmoid $\sigma:\mathbb R\to(0,1)$.

---

## 1. The Lot Vector Space $\mathcal X$

### 1.1 The feature-extraction map

A single observation is produced by the simulation engine through the
feature-extraction map applied to one open lot in the portfolio at one day:

$$
g:\ \mathcal S_t\times P_t\ \longrightarrow\ \mathcal X^{\,|\mathcal K_t|},
\qquad
g(\mathcal S_t, P_t) = \bigl(x_{k,t}\bigr)_{k\in\mathcal K_t},
$$

where $\mathcal S_t$ is the portfolio state at day $t$, $P_t\in\mathbb R_{>0}^{500}$ the
price cross-section, and $\mathcal K_t$ the set of open lots. The $k$-th component
$x_{k,t}\in\mathcal X$ is exactly one `LotStateVector` (minus its labels and
metadata). Writing the projection onto a single lot,

$$
\phi_{\mathrm{lot}}:(\mathrm{Lot}_k,\mathcal S_t,P_t)\ \longmapsto\ x_{k,t}\in\mathbb R^{15}.
$$

### 1.2 Coordinates with explicit types

The 15 numeric coordinates (`FeatureLists.NumericFeatures`, in schema order) are
the following maps. For lot $k$ with shares $q_k\in\mathbb Z_{>0}$, cost basis
$p_k\in\mathbb R_{>0}$, purchase day $s_k\in\mathbb Z_{\ge0}$, current price
$P_t\in\mathbb R_{>0}$, portfolio value $V_t\in\mathbb R_{>0}$:

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
G^{\mathrm{YTD}}_t &\in\mathbb R
   &&\text{net realized gain YTD (shared state)}\\
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
\alpha_{\mathrm{tax}} &= \tau(h_k)\cdot|G_{\mathrm{lot}}|\cdot\mathbf 1[G^{\mathrm{YTD}}_t>0]\in\mathbb R_{\ge0}
   &&\text{tax alpha}\\
\mathrm{DaysToYE} &\in\mathbb Z_{\ge0}
   &&\text{days to year-end}
\end{aligned}
$$

where $H_t,L_t$ are the day-$t$ high/low and $\tau:\mathbb Z_{\ge0}\to\{\tau_{\mathrm{ST}},\tau_{\mathrm{LT}}\}$,
$\tau(h)=\tau_{\mathrm{ST}}\mathbf 1[h<365]+\tau_{\mathrm{LT}}\mathbf 1[h\ge365]$, with $\tau_{\mathrm{ST}}>\tau_{\mathrm{LT}}$.

### 1.3 Direct-sum decomposition

The coordinates partition by *origin of state* into four blocks:

$$
\mathcal X \;=\; \underbrace{\mathcal X_{\mathrm{lot}}}_{\mathbb R^6}\ \oplus\
\underbrace{\mathcal X_{\mathrm{port}}}_{\mathbb R^3}\ \oplus\
\underbrace{\mathcal X_{\mathrm{asset}}}_{\mathbb R^4}\ \oplus\
\underbrace{\mathcal X_{\mathrm{derived}}}_{\mathbb R^2},
\qquad \dim\mathcal X = 6+3+4+2 = 15.
$$

- $\mathcal X_{\mathrm{lot}}=(L,H,S,B,W,K)$ — intrinsic to the lot.
- $\mathcal X_{\mathrm{port}}=(G^{\mathrm{YTD}}_t,\sigma_{\mathrm{TE}},\mathcal W^{A_i}_t)$ — shared portfolio state $\mathcal S_t$.
- $\mathcal X_{\mathrm{asset}}=(R_t,\Sigma\mathrm{Range},\Delta\mathrm{MA}_{50},\Delta\mathrm{MA}_{200})$ — from the price series.
- $\mathcal X_{\mathrm{derived}}=(\alpha_{\mathrm{tax}},\mathrm{DaysToYE})$ — composites.

The categorical field $z=\texttt{Sector}\in\mathcal Z$ is appended separately (§3.2).

### 1.4 The observation pair

Each row of `lots.csv` is a point

$$
\bigl(x_{k,t},\,y_{k,t}\bigr)\in\mathcal X\times\mathcal Y,\qquad
\mathcal Y=\underbrace{\{0,1\}}_{Y_{\mathrm{Oracle}}}\times\underbrace{[0,1]}_{Y_{\mathrm{Soft\,BT}}}\times\underbrace{[0,1]}_{Y_{\mathrm{Soft\,GBM}}}.
$$

The **panel index** is $(i,t)$ with $i$ the lot/stock and $t$ the day; under the
oracle, $Y_{i,t}$ is $\sigma(X_{i,t})$-measurable, so observations are treated as
conditionally independent given features (the i.i.d.-conditional-on-$x$ assumption
of `data_memo_theory.md` §2.2).

---

## 2. The Targets: Oracle Boundary and the `soft_bt` Label

### 2.1 The oracle map $f^*$

`OracleBoundary.Label` realizes the deterministic measurable function

$$
f^*:\mathcal X\to\{0,1\},\qquad
f^*(x)=\mathbf 1[\ell\le-\theta_1]\cdot\mathbf 1[\sigma_{\mathrm{TE}}\le\theta_2]\cdot\mathbf 1[G^{\mathrm{YTD}}_t>0]\cdot\mathbf 1[\mathcal W^{A_i}_t\ge\theta_3],
$$

with the project constants
$\theta_1=0.02$, $\theta_2=0.05$, $\theta_3=30$ (`LossThreshold`,
`TrackingErrorCap`, `WashSaleDays`). $f^*=\mathbf 1_\Omega$ is the indicator of the
harvest region $\Omega=H_1\cap H_2\cap H_3\cap H_4$, an intersection of four
halfspaces — a convex polytope in the relevant coordinates. The hard label is

$$
Y_{\mathrm{Oracle}}=f^*(x)\in\{0,1\}.
$$

### 2.2 The `soft_bt` label as a path functional

The soft backtest label upgrades the *pointwise* gate $f^*$ to a *forward-looking
frequency*. Fix a snapshot lot at day $t_0$ with **frozen portfolio state**

$$
\bigl(G^{\mathrm{YTD}}_{t_0},\ \sigma_{\mathrm{TE}},\ \mathcal W_0:=\mathcal W^{A_i}_{t_0},\ p_k\bigr),
$$

(held constant — see `SoftLabelBuilder.ComputeBT`), and let
$\{P_{t_0+s}\}_{s=1}^{W}$ be the **actual realized** closing prices over the forward
window $W=30$ trading days. Only two things vary across the window: the price
$P_{t_0+s}$ (hence the lot's unrealized return) and the wash-sale clock, which
advances deterministically as $\mathcal W_0+s$.

Define the per-step forward unrealized return and the per-step oracle firing:

$$
\ell_s := \frac{P_{t_0+s}-p_k}{p_k},\qquad
b_s := f^*\!\Bigl(\ell_s,\ \sigma_{\mathrm{TE}},\ G^{\mathrm{YTD}}_{t_0},\ \mathcal W_0+s\Bigr)\in\{0,1\}.
$$

Because $\sigma_{\mathrm{TE}}$ and $G^{\mathrm{YTD}}_{t_0}$ are frozen and (in the
generating regime) satisfy the TE/gains gates, $b_s$ reduces to the conjunction of
the *loss* gate and the *wash* gate evaluated along the realized path:

$$
b_s = \mathbf 1[\ell_s\le-\theta_1]\cdot\mathbf 1[\sigma_{\mathrm{TE}}\le\theta_2]\cdot\mathbf 1[G^{\mathrm{YTD}}_{t_0}>0]\cdot\mathbf 1[\mathcal W_0+s\ge\theta_3].
$$

The label is the **time-average firing frequency** over the window:

$$
\boxed{\ \tilde y_{\mathrm{BT}}(x)\ :=\ \frac1W\sum_{s=1}^{W} b_s\ =\ \frac{\#\{s\in[1,W]: \text{oracle fires}\}}{W}\ \in\ \Bigl\{0,\tfrac1W,\tfrac2W,\dots,1\Bigr\}.\ }
$$

This is exactly `ComputeBT`: `oracleDays / Window` with `Window = 30`. When fewer
than $W$ forward days remain ($t_0+W\ge t_{\max}$), the functional is undefined and
the code returns `NaN`; those rows are dropped before modeling
($\{v:\text{not }\mathrm{NaN}(\tilde y_{\mathrm{BT}})\}$, see every trainer's `SelectTarget`).

**Type summary.**
$\tilde y_{\mathrm{BT}}:\mathcal X\to[0,1]\cup\{\mathrm{NaN}\}$; each $b_s:\mathcal X\times\mathbb Z_{>0}\to\{0,1\}$; the window sum is a deterministic functional of the realized price path segment $(P_{t_0+1},\dots,P_{t_0+W})\in\mathbb R_{>0}^{W}$.

### 2.3 Interpretation: from boundary indicator to urgency field

$\tilde y_{\mathrm{BT}}(x)$ estimates the **harvest urgency** — the fraction of the
near future during which this lot *would* be harvestable. A lot deep in a
persistent drawdown fires on most of the next 30 days ($\tilde y_{\mathrm{BT}}\to1$);
a lot grazing the $-2\%$ threshold fires intermittently ($\tilde y_{\mathrm{BT}}$ small).
In the language of `data_memo_theory.md` §10, $\tilde y_{\mathrm{BT}}$ is a sampled
estimate of the interior posterior $\eta$, whose level sets

$$
\partial\Omega_{L_c}=\{x\in\mathcal X:\eta(x)=c\},\qquad c\in[0,1],
$$

are $(d-1)$-dimensional **contours of harvest urgency**. The hard oracle only sees
$\partial\Omega_{1/2}$; the soft label exposes the whole graded field inside $\Omega$.

### 2.4 The binary target derived from `soft_bt`

The supervised trainers consume a *binary* target. For `target = "soft_bt"` the
positive class is "fires at least once in the window":

$$
y := \mathbf 1[\tilde y_{\mathrm{BT}}(x)>0]\in\{0,1\},
$$

(every trainer's `SelectTarget`: `r.Y_Soft_BT > 0f`). For `target = "oracle"`,
$y=\mathbf 1[Y_{\mathrm{Oracle}}=1]$. Thus the soft label is used here in its
"any-fire" thresholded form; the raw continuous $\tilde y_{\mathrm{BT}}$ remains
available for future regression/calibration work.

### 2.5 (Companion) the GBM soft label

For completeness, `Y_Soft_GBM` replaces the realized path with Monte-Carlo paths
under geometric Brownian motion. The price process solves

$$
dS_u=\mu S_u\,du+\sigma S_u\,dW_u,\qquad S_{t_0}=P_{t_0},\ W_u:\Omega_{\mathrm{prob}}\times[0,T]\to\mathbb R,\ dW_u\sim\mathcal N(0,du),
$$

discretized at $\Delta=1/252$ with the Itô-corrected log-Euler scheme
(`GbmSimulator.SimulatePaths`):

$$
S_{s} = S_{s-1}\exp\!\Bigl((\mu-\tfrac12\sigma^2)\Delta + \sigma\sqrt\Delta\,Z_s\Bigr),\quad Z_s\overset{\mathrm{iid}}\sim\mathcal N(0,1),
$$

with $\mu=0$ (risk-neutral default `AnnualDrift`) and $\sigma$ the annualized
trailing-21-day realized vol (`EstimateVol`: $\sigma=\widehat{\mathrm{std}}(r)\sqrt{252}$).
The label is the **first-passage frequency** over $N_{\mathrm{paths}}=200$ paths,

$$
\tilde y_{\mathrm{GBM}}(x)=\frac1{N_{\mathrm{paths}}}\sum_{p=1}^{N_{\mathrm{paths}}}\mathbf 1\!\bigl[\exists\,s\in[1,W]:b_s^{(p)}=1\bigr]\ \in[0,1],
$$

an unbiased Monte-Carlo estimator of $\mathbb P(\exists s\le W:\text{oracle fires})$
under the GBM measure (`FractionFiring`, which `break`s a path on first fire). The
$Z_s$ are generated by Box–Muller, $Z=\sqrt{-2\ln U_1}\cos(2\pi U_2)$ with
$U_1\sim\mathrm{Unif}(0,1],U_2\sim\mathrm{Unif}[0,1)$. The contrast with §2.2 is
exactly *realized path* (one deterministic trajectory) vs. *simulated ensemble*
(expectation over the price law).

---

## 3. Shared Train/Test Protocol and Preprocessing

### 3.1 Protocol (identical across all supervised trainers)

Given filtered rows $D=\{(x_i,y_i)\}_{i=1}^N$:

1. **Stratified split** $D=D_{\mathrm{tr}}\sqcup D_{\mathrm{te}}$, test fraction $0.20$, seed $42$ (`StratifiedSplit.Split`). Same seed ⇒ *all models share the identical split*, which is what makes the champion-selection comparison valid.
2. **Median imputation** fit on $D_{\mathrm{tr}}$ only (`MedianImputer.Fit`), applied to both folds — a training-fold-only invariant (leak-free).
3. **Balanced class weights** computed on $D_{\mathrm{tr}}$ only:
   $$w_c=\frac{N_{\mathrm{tr}}}{K\,n_c},\quad c\in\{0,1\},\ K=2,$$
   with $n_c=\#\{i\in D_{\mathrm{tr}}:y_i=c\}$ (`ClassWeights.AttachBalancedWeights`). This is the ML.NET analogue of sklearn's `class_weight='balanced'`.
4. **5-fold stratified CV** over a finite hyperparameter grid, on $D_{\mathrm{tr}}$ only (`StratifiedKFold` + `GridSearchCV`), scored by **PR-AUC** (average precision).
5. **Refit** on all of $D_{\mathrm{tr}}$ with the CV-best hyperparameters.
6. **Evaluate once** on $D_{\mathrm{te}}$ — but only for the champion model(s) (rubric: best 1–2 touch the test set).

### 3.2 The preprocessing map $\phi$

`PreprocessingPipeline.Build` realizes

$$
\phi(x,z)=\Bigl[\,\underbrace{\mathrm{Norm}(x)}_{\in\mathbb R^{15}}\ \big\Vert\ \underbrace{\mathrm{OneHot}(\mathrm{Clean}(z))}_{\in\{0,1\}^{m}}\,\Bigr]\in\mathbb R^{d},\quad d=15+m,
$$

where $\mathrm{Norm}$ is mean–variance standardization
$\mathrm{Norm}(x)_j=(x_j-\mu_j)/\sigma_j$ with $(\mu_j,\sigma_j)$ estimated on the
training fold, and $m=|\mathcal Z_{\mathrm{tr}}|$ is the sector vocabulary learned on
the training fold. For tree models $\mathrm{Norm}$ is order-preserving per
coordinate, hence a no-op for split selection (kept only for schema consistency).

All model objectives below are functions of $\phi_i:=\phi(x_i,z_i)$.

---

## 4. Supervised Models

For each, "$\sum_{i}$" abbreviates $\sum_{i\in D_{\mathrm{tr}}}$ and $w_{y_i}$ is the balanced weight of example $i$'s class.

### 4.1 Logistic regression — `LbfgsLogisticRegression` / `LogisticTrainer`

Hypothesis: a calibrated linear-logit posterior
$\hat\eta:\mathbb R^d\to(0,1)$,

$$
\hat\eta(\phi)=\sigma(w^\top\phi+b)=\bigl(1+e^{-(w^\top\phi+b)}\bigr)^{-1},\qquad w\in\mathbb R^d,\ b\in\mathbb R.
$$

Weighted regularized cross-entropy (negative log-likelihood) objective:

$$
\min_{w,b}\ \sum_i w_{y_i}\Bigl[-y_i\log\hat\eta(\phi_i)-(1-y_i)\log\bigl(1-\hat\eta(\phi_i)\bigr)\Bigr]+\lambda\lVert w\rVert_2^2,
$$

with $\lambda=1/C$ and $C\in\{0.01,0.1,1,10\}$ selected by CV. The gradient
$\nabla_w=\sum_i w_{y_i}(\hat\eta(\phi_i)-y_i)\phi_i+2\lambda w$ vanishes at the optimum;
L-BFGS solves it. Coefficients are recovered by walking
$\textsf{CalibratedModelParametersBase}\to\textsf{LinearBinaryModelParameters}$
(`ExtractCoefficients`/`FindLinearPredictor`).

### 4.2 Elastic-net logistic — `SdcaLogisticRegression` / `ElasticNetTrainer`

Same logistic hypothesis, but the penalty is the elastic net
$\lambda_1\lVert w\rVert_1+\lambda_2\lVert w\rVert_2^2$:

$$
\min_{w,b}\ \sum_i w_{y_i}\,\ell_{\log}\!\bigl(y_i,\ w^\top\phi_i+b\bigr)+\lambda_1\lVert w\rVert_1+\lambda_2\lVert w\rVert_2^2,
\qquad \ell_{\log}(y,z)=\log\!\bigl(1+e^{z}\bigr)-yz,
$$

solved by stochastic dual coordinate ascent (SDCA). Grid:
$\lambda_1\in\{0.001,0.01,0.1\}$, $\lambda_2\in\{0.001,0.01,0.1\}$ (9 configs). The
$\ell_1$ term induces **sparsity** (feature selection by zeroing weights), the
$\ell_2$ term **shrinkage**. Semantic map to sklearn
`LogisticRegression(penalty='elasticnet', l1_ratio=\rho, C=c)`:
$\lambda_1=\rho/c$, $\lambda_2=(1-\rho)/(2c)$ (see `MLNetSemanticReconciliation.md` §5).
Same coefficient-extraction walk as §4.1 (SDCA yields the same calibrated linear
base type).

### 4.3 Gradient-boosted trees — `FastTree` / `GradientBoostedTreesTrainer`

A boosted additive ensemble of regression trees fit to the **functional gradient**
of logistic loss. Let $F_0\equiv\mathrm{logit}(\bar y)$ and iterate for $m=1,\dots,M$:

$$
F_m(\phi)=F_{m-1}(\phi)+\nu\,f_m(\phi),\qquad
f_m=\arg\min_{f\in\mathcal T_J}\sum_i w_{y_i}\bigl(r_{i}^{(m)}-f(\phi_i)\bigr)^2,
$$

where $\mathcal T_J$ is the class of regression trees with $J$ leaves, $\nu$ the
learning rate (shrinkage), and the pseudo-residual is the negative gradient of the
logistic loss in score space,

$$
r_i^{(m)}=-\frac{\partial \ell_{\log}(y_i,F)}{\partial F}\Big|_{F=F_{m-1}(\phi_i)}=y_i-\sigma\!\bigl(F_{m-1}(\phi_i)\bigr).
$$

The final score $F_M(\phi)\in\mathbb R$ is mapped to a probability by Platt
calibration $\hat\eta(\phi)=\sigma(aF_M(\phi)+b)$. Grid (8 configs):
$M=\texttt{numberOfTrees}\in\{100,200\}$,
$\nu=\texttt{learningRate}\in\{0.10,0.05\}$,
$J=\texttt{numberOfLeaves}\in\{20,31\}$. sklearn map:
$M\!\equiv\!\texttt{n\_estimators}$, $\nu\!\equiv\!\texttt{learning\_rate}$,
$J\!\equiv\!\texttt{max\_leaf\_nodes}$.

### 4.4 Random forest — `FastForest` / `RandomForestTrainer`

A **bagged** ensemble — averaging, not boosting. Draw $T$ bootstrap resamples
$D^{(t)}$ of $D_{\mathrm{tr}}$; on each, grow a tree $f_t$ choosing each split from a
random feature subset of fraction $\kappa=\texttt{FeatureFraction}=0.7$. The score
is the ensemble average

$$
F(\phi)=\frac1T\sum_{t=1}^{T}f_t(\phi)\in[0,1],
$$

interpreted directly as a probability proxy (FastForest is **uncalibrated** — it
emits `Score` but no `Probability` column, which is why `BinaryMetrics.Compute`
falls back to `Score`). Variance reduction comes from decorrelation: with per-tree
variance $\varsigma^2$ and pairwise correlation $\rho$,
$\mathrm{Var}(F)=\rho\varsigma^2+\frac{1-\rho}{T}\varsigma^2$, and feature
subsampling lowers $\rho$. Grid (4 configs):
$T\in\{100,200\}$, $J\in\{20,31\}$. The random feature subsampling
$\kappa$ is ML.NET's analogue of sklearn's `max_features` (default $\approx\sqrt p/p$).

### 4.5 Linear regression on a binary target — `Sdca` regression / `LinearRegressionTrainer` (deliberate poor fit)

Hypothesis: an **unbounded** affine map (no sigmoid),

$$
\hat y:\mathbb R^d\to\mathbb R,\qquad \hat y(\phi)=w^\top\phi+b,
$$

trained against the float label $\tilde Y\in\{0,1\}$ (`FloatLabel`) under
$\ell_2$-regularized squared error:

$$
\min_{w,b}\ \sum_i w_{y_i}\bigl(\tilde Y_i-w^\top\phi_i-b\bigr)^2+\lambda\lVert w\rVert_2^2,\qquad \lambda\in\{10^{-4},10^{-3},10^{-2}\}.
$$

**Why this is structurally misspecified as a probabilistic classifier.** The target
of the pipeline is the calibrated posterior $\eta:\mathcal X\to[0,1]$. OLS fits a
hyperplane to $\{0,1\}$ outcomes, so its image is all of $\mathbb R$, not $[0,1]$:
predictions escape the unit interval and cannot be read as probabilities. The
diagnostic reported is the escape fraction

$$
\mathrm{FractionOutsideUnit}=\frac1{|D_{\mathrm{te}}|}\sum_{i\in D_{\mathrm{te}}}\mathbf 1\bigl[\hat y(\phi_i)<0\ \lor\ \hat y(\phi_i)>1\bigr]\in[0,1],
$$

(empirically $\approx25\%$). Further, Gauss–Markov optimality requires homoskedastic
errors, but a Bernoulli outcome has $\mathrm{Var}(Y\mid x)=\eta(x)(1-\eta(x))$ —
intrinsically heteroskedastic — so OLS standard errors are invalid and the fit is
not BLUE for this DGP. To still produce ROC/PR numbers, the raw score is used as a
probability *proxy* ($\hat\eta:=\hat y$, clipping implied by the metric sweep); this
is precisely the abuse the `FractionOutsideUnit` statistic quantifies. Geometrically
(per `data_memo_theory.md` §3–4): the oracle is a level-curve boundary $\partial\Omega$
and the calibrated models learn a smooth field on $\Omega^\circ$ bounded in $[0,1]$;
an unbounded hyperplane has neither the boundedness nor the level-set geometry, so it
is the sharpest principled negative control.

---

## 5. Evaluation Functionals (`BinaryMetrics`)

Given scored test rows $\{(y_i,\hat\eta_i)\}$ sorted by $\hat\eta$ descending,
with $n_+=\#\{y_i=1\}$, $n_-=\#\{y_i=0\}$, sweep the threshold and accumulate
$\mathrm{TP}(\tau),\mathrm{FP}(\tau)$. Define

$$
\mathrm{TPR}(\tau)=\frac{\mathrm{TP}(\tau)}{n_+},\quad
\mathrm{FPR}(\tau)=\frac{\mathrm{FP}(\tau)}{n_-},\quad
\mathrm{Prec}(\tau)=\frac{\mathrm{TP}(\tau)}{\mathrm{TP}(\tau)+\mathrm{FP}(\tau)},\quad
\mathrm{Rec}(\tau)=\mathrm{TPR}(\tau).
$$

ROC-AUC is the trapezoidal integral $\int_0^1\mathrm{TPR}\,d(\mathrm{FPR})$; PR-AUC
is the average-precision step integral $\sum_k(\mathrm{Rec}_k-\mathrm{Rec}_{k-1})\mathrm{Prec}_k$.
The $F_1$ at threshold $\tau$ is the harmonic mean
$F_1=2\,\mathrm{Prec}\cdot\mathrm{Rec}/(\mathrm{Prec}+\mathrm{Rec})$, reported at
$\tau=0.5$ and at the $F_1$-maximizing $\tau^\*$. **PR-AUC is the CV selection
criterion** because the positive class (harvest events) is rare — precision/recall
geometry is the relevant operating regime, and PR-AUC is sensitive to it where
ROC-AUC is optimistic under imbalance.

---

## 6. Unsupervised Models

### 6.1 PCA — `PcaPipeline`

On the standardized numeric training matrix $X\in\mathbb R^{n\times15}$ (columns
zero-mean, unit-variance), form the sample covariance and eigendecompose:

$$
C=\frac1{n-1}X^\top X\in\mathbb R^{15\times15},\qquad C u_j=\lambda_j u_j,\ \ \lambda_1\ge\dots\ge\lambda_{15}\ge0,\ u_j\in\mathbb R^{15}.
$$

The explained-variance ratio is $\rho_j=\lambda_j/\sum_k\lambda_k$, and the retained
dimension is the smallest $r$ with cumulative coverage past the threshold,

$$
r=\min\Bigl\{r':\sum_{j=1}^{r'}\rho_j\ge0.95\Bigr\},
$$

projecting $x\mapsto U_r^\top x\in\mathbb R^r$ with $U_r=[u_1\cdots u_r]$. The low-rank
structure is justified by the factor model $\Sigma=B\Sigma_F B^\top+D$ of
`data_memo_theory.md` §6 — a spectral gap after the leading factors.

### 6.2 K-means — `KMeansPipeline`

Aggregate, per symbol $s$, the four asset-level features across time into
$g_s\in\mathbb R^4$ (then standardize), and solve the within-cluster sum-of-squares
problem

$$
\min_{\{\mu_c\}_{c=1}^k,\ \{a_s\}}\ \sum_s\lVert g_s-\mu_{a_s}\rVert_2^2,\qquad a_s\in\{1,\dots,k\},\ \mu_c\in\mathbb R^4,
$$

by Lloyd's alternating algorithm (assign $a_s=\arg\min_c\lVert g_s-\mu_c\rVert$;
update $\mu_c=\mathrm{mean}\{g_s:a_s=c\}$). The cluster count is chosen by maximum
average silhouette over $k\in\{5,10,15,20,25\}$, where the silhouette of point $s$ is
$\mathrm{sil}(s)=\frac{b(s)-a(s)}{\max\{a(s),b(s)\}}\in[-1,1]$ with $a(s)$ the mean
intra-cluster distance and $b(s)$ the mean distance to the nearest other cluster.

---

## 7. Champion Selection (test-set discipline)

All models are CV-scored by mean PR-AUC on $D_{\mathrm{tr}}$ and ranked on a single
leaderboard. Let $\widehat{\mathrm{PRAUC}}_{\mathrm{CV}}(\mathcal M)$ be model
$\mathcal M$'s mean fold PR-AUC. The champion set is the top one or two classifiers,

$$
\mathcal M^\star=\operatorname*{top-2}_{\mathcal M}\ \widehat{\mathrm{PRAUC}}_{\mathrm{CV}}(\mathcal M),
$$

and **only** $\mathcal M^\star$ (plus the deliberate poor-fit demonstrator) is refit
and evaluated on $D_{\mathrm{te}}$. The test set is thus touched exactly once, by the
champion only — satisfying the rubric constraint and preventing test-set
adaptation. Because the seed-42 split is shared, the leaderboard comparison and the
champion's test estimate are on the same partition of the data.

---

## 8. The Chain of Approximations (summary)

$$
f^*_{\mathrm{true}}\ \xrightarrow{\ \text{myopic}\ }\ f^*_{\mathrm{oracle}}\ \xrightarrow{\ \text{forward window}\ }\ \tilde y_{\mathrm{BT}}\ \xrightarrow{\ \text{ERM over }\mathcal H\ }\ \hat\eta.
$$

The mechanistic oracle $f^*$ fixes the boundary $\partial\Omega$ (Source 1 of
tax-alpha — irreducible by any classifier). The soft label $\tilde y_{\mathrm{BT}}$
samples the interior urgency field, and the supervised models learn $\hat\eta$ — a
smooth $[0,1]$-valued estimate of $\eta$ whose level sets $L_c$ prioritize *which*
in-region lots to harvest *first* (Source 2 — closeable, and the project's empirical
contribution). The linear-regression comparator exists precisely to show that a
model without the $[0,1]$ codomain and level-set geometry cannot represent this
field at all.

---

## Cross-reference

- `DataMemo/data_memo_theory.md` — probability space, oracle/boundary geometry, currents/Stokes, ERM theory.
- `DataMemo/MLNetPipeline.md` — high-level pipeline and prior model equations.
- `DataMemo/MLNetSemanticReconciliation.md` — ML.NET ↔ sklearn parameter map (§§5–15).
- `DataMemo/MLNetLeakageAudit.md` — training-fold-only invariants.
- `src/Core/Portfolio/LotStateVector.cs` — canonical 15-feature schema.
- `src/Core/Oracle/OracleBoundary.cs` — the map $f^*$ and thresholds $\theta_1,\theta_2,\theta_3$.
- `src/Core/Simulation/SoftLabelBuilder.cs`, `GbmSimulator.cs` — $\tilde y_{\mathrm{BT}}$, $\tilde y_{\mathrm{GBM}}$.
- `src/ML/CSharp/MLNet/Models/*.cs` — trainer implementations.

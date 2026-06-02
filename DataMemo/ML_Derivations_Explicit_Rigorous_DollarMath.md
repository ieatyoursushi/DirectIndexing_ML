# ML Mathematical Derivations

## Purpose

This document makes every symbol unambiguous at the point where it is used.

The goal is not to be minimal. Every object is introduced with its type as a map between sets, so there is no ambiguity about what lives where.

For example, instead of only writing

$$
f(x)=\mathbb E[R\mid X=x],
$$

this document writes

$$
f(x)=\mathbb E[R\mid X=x],
\qquad
f:\mathcal X\to\mathbb R,
\qquad
X\in\mathcal X,
\qquad
R\in\mathbb R.
$$

That redundancy is intentional.

The overall pipeline is

```text
Portfolio State
      ↓
Feature Extraction  →  x ∈ X ⊂ ℝ¹⁵
      ↓
Oracle Boundary     →  f*(x) ∈ {0,1}
      ↓
Soft Labels         →  ỹ_BT(x) ∈ [0,1]
      ↓
ML Model            →  η̂(x) ∈ [0,1]
      ↓
Estimated Harvest Probability
```

Cross-references: `data_memo_theory.md` (probability space, oracle geometry, ERM theory), `src/Core/Portfolio/LotStateVector.cs` (schema), `src/Core/Oracle/OracleBoundary.cs` (the map $f^*$), `src/Core/Simulation/SoftLabelBuilder.cs` (soft labels).

---

# 0. Standing Definitions

The feature space is

$$
\mathcal X\subset\mathbb R^{15}.
$$

A feature vector is

$$
x\in\mathcal X,
\qquad
x=<x_1,\dots,x_{15}>.
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
D=\{(x_i,y_i)\}_{i=1}^{N},
\qquad
x_i\in\mathcal X,
\qquad
y_i\in\{0,1\}.
$$

The ideal posterior function is

$$
\eta(x)=\mathbb P(Y=1\mid X=x),
\qquad
\eta:\mathcal X\to[0,1],
\qquad
X\in\mathcal X,
\qquad
Y\in\{0,1\}.
$$

The supervised learning goal is to learn an estimate

$$
\hat\eta:\mathcal X\to[0,1],
\qquad
\hat\eta(x)\in[0,1],
$$

interpreted as the model's estimated probability that lot $x$ should be harvested.

---

# 1. Feature Space

## 1.0 Feature-extraction map

The feature extraction map is

$$
\phi_{\mathrm{lot}}:
(\mathrm{Lot}_k,\mathcal S_t,P_t)
\longrightarrow
x_{k,t},
\qquad
\phi_{\mathrm{lot}}:\mathrm{Lot}_k\times\mathcal S_t\times\mathbb R_{>0}\to\mathcal X.
$$

Here $\mathrm{Lot}_k$ is lot $k$, $\mathcal S_t$ is the portfolio state at day $t$, $P_t\in\mathbb R_{>0}$ is the current asset price, and $x_{k,t}\in\mathcal X\subset\mathbb R^{15}$.

The feature space decomposes as

$$
\mathcal X
=
\mathcal X_{\mathrm{lot}}
\oplus
\mathcal X_{\mathrm{port}}
\oplus
\mathcal X_{\mathrm{asset}}
\oplus
\mathcal X_{\mathrm{derived}},
\qquad
\dim(\mathcal X)=6+3+4+2=15.
$$

---

## 1.1 Lot-State Coordinates

$$
x_{\mathrm{lot}}
=
\text{<}L,H,S,B,W,K\text{>}
\in
\mathcal X_{\mathrm{lot}}
\subset\mathbb R^6.
$$

| Symbol | Formula | Type | Meaning |
|--------|---------|------|---------|
| $L$ | $(P_t-p_k)/p_k$ | $(-1,\infty)$ | Normalized unrealized return |
| $H$ | $t-s_k$ | $\mathbb Z_{\ge0}$ | Holding period in days |
| $S$ | $\mathbf 1[H\ge365]$ | $\{0,1\}$ | Long-term holding flag |
| $B$ | $p_k$ | $\mathbb R_{>0}$ | Cost basis per share |
| $W$ | $q_kP_t/V_t$ | $(0,1)$ | Portfolio weight of this lot |
| $K$ | lot count | $\mathbb Z_{>0}$ | Open lots for same ticker |

Here $p_k\in\mathbb R_{>0}$ is cost basis, $s_k\in\mathbb Z_{\ge0}$ is purchase day, $q_k\in\mathbb Z_{>0}$ is share count, and $V_t\in\mathbb R_{>0}$ is total portfolio value.

$L$ is negative for a harvestable lot (bought above current price).

---

## 1.2 Portfolio-State Coordinates

$$
x_{\mathrm{port}}
=
\text{<}G_t^{YTD},\sigma_{TE},\mathcal W_t\text{>}
\in
\mathcal X_{\mathrm{port}}
\subset\mathbb R^3.
$$

| Symbol | Type | Meaning |
|--------|------|---------|
| $G_t^{YTD}$ | $\mathbb R$ | Net realized gain/loss year-to-date |
| $\sigma_{TE}$ | $\mathbb R_{\ge0}$ | Annualized tracking error vs. benchmark |
| $\mathcal W_t$ | $\mathbb Z_{\ge0}\cup\{999\}$ | Days since last harvest of this ticker |

These are shared portfolio-level state variables, not lot-specific. $\mathcal W_t=999$ indicates the ticker has never been harvested in the simulation.

---

## 1.3 Asset-State Coordinates

$$
x_{\mathrm{asset}}
=
<R_t,\Sigma_{\mathrm{Range}},\Delta\mathrm{MA}_{50},\Delta\mathrm{MA}_{200}\text{>}
\in
\mathcal X_{\mathrm{asset}}
\subset\mathbb R^4.
$$

| Symbol | Formula | Type | Meaning |
|--------|---------|------|---------|
| $R_t$ | $(P_t-P_{t-1})/P_{t-1}$ | $\mathbb R$ | Daily return |
| $\Sigma_{\mathrm{Range}}$ | $(H_t^{\mathrm{price}}-L_t^{\mathrm{price}})/P_{t-1}$ | $\mathbb R_{\ge0}$ | Intraday range volatility proxy |
| $\Delta\mathrm{MA}_{50}$ | $(P_t-\mathrm{MA}_{50})/\mathrm{MA}_{50}$ | $\mathbb R$ | Deviation from 50-day moving average |
| $\Delta\mathrm{MA}_{200}$ | $(P_t-\mathrm{MA}_{200})/\mathrm{MA}_{200}$ | $\mathbb R$ | Deviation from 200-day moving average |

Here $H_t^{\mathrm{price}},L_t^{\mathrm{price}}\in\mathbb R_{>0}$ are the day-$t$ high and low prices.

---

## 1.4 Derived Coordinates

$$
x_{\mathrm{derived}}
=
\text{<}\alpha_{\mathrm{tax}},\mathrm{DaysToYE})\text{>}
\in
\mathcal X_{\mathrm{derived}}
\subset\mathbb R^2.
$$

The tax-alpha coordinate is

$$
\alpha_{\mathrm{tax}}
=
\tau(H)\cdot|G_{\mathrm{lot}}|\cdot\mathbf 1[G_t^{YTD}>0],
\qquad
\alpha_{\mathrm{tax}}\in\mathbb R_{\ge0}.
$$

Here $G_{\mathrm{lot}}=L\cdot B=(P_t-p_k)$ is the dollar unrealized gain/loss, and $\tau:\mathbb Z_{\ge0}\to\{\tau_{ST},\tau_{LT}\}$ is the applicable tax rate,

$$
\tau(H)
=
\begin{cases}
\tau_{ST} & H<365 \quad\text{(short-term, ordinary income rate)}\\
\tau_{LT} & H\ge365 \quad\text{(long-term, preferential rate)}
\end{cases}
\qquad
\tau_{ST}>\tau_{LT}.
$$

Short-term losses are worth more to harvest because they offset income taxed at a higher rate. The indicator $\mathbf 1[G_t^{YTD}>0]$ ensures tax alpha is zero when there are no gains to offset.

$$
\mathrm{DaysToYE}\in\mathbb Z_{\ge0}.
$$

Here $\mathrm{DaysToYE}$ is the number of calendar days remaining in the tax year.

---

# 2. Oracle Boundary

The oracle is a deterministic function

$$
f^*:\mathcal X\to\{0,1\},
\qquad
f^*(x)\in\{0,1\}.
$$

The four oracle gates are:

$$
g_1(x)=\mathbf 1[L\le -\theta_1],
\qquad
g_1:\mathcal X\to\{0,1\},
\qquad
\theta_1=0.02.
$$

$$
g_2(x)=\mathbf 1[\sigma_{TE}\le\theta_2],
\qquad
g_2:\mathcal X\to\{0,1\},
\qquad
\theta_2=0.05.
$$

$$
g_3(x)=\mathbf 1[G_t^{YTD}>0],
\qquad
g_3:\mathcal X\to\{0,1\}.
$$

$$
g_4(x)=\mathbf 1[\mathcal W_t\ge\theta_3],
\qquad
g_4:\mathcal X\to\{0,1\},
\qquad
\theta_3=30.
$$

The oracle is the product of all four gates:

$$
f^*(x)
=
g_1(x)\cdot g_2(x)\cdot g_3(x)\cdot g_4(x)
=
\mathbf 1[L\le-0.02]\cdot\mathbf 1[\sigma_{TE}\le0.05]\cdot\mathbf 1[G_t^{YTD}>0]\cdot\mathbf 1[\mathcal W_t\ge30].
$$

The hard oracle label is

$$
Y_{\mathrm{Oracle}}=f^*(X),
\qquad
Y_{\mathrm{Oracle}}\in\{0,1\},
\qquad
X\in\mathcal X.
$$

The harvest region is

$$
\Omega
=
\{x\in\mathcal X:f^*(x)=1\}
=
H_1\cap H_2\cap H_3\cap H_4,
$$

where each $H_i=\{x:g_i(x)=1\}$ is a closed halfspace in $\mathcal X$. Their intersection $\Omega$ is a convex polytope. The boundary $\partial\Omega$ is the set of threshold-touching hypersurfaces where any single gate is exactly binding.

The oracle answers the binary question: *Is this lot harvestable right now?* Its limitations — it is myopic, ignores future prices, cannot rank lots inside $\Omega$ — motivate the soft label construction below.

---

# 3. Backtest Soft Label

The backtest soft label is a function

$$
\tilde y_{BT}:\mathcal X\to[0,1].
$$

The oracle answers *harvest now?* The soft label asks: *how persistently harvestable is this lot over the next 30 trading days?*

**Frozen portfolio state.** Fix a snapshot at day $t_0$. The portfolio coordinates $G_{t_0}^{YTD}$, $\sigma_{TE}$, and cost basis $p_k$ are held constant across the forward window (as in the simulation layer). Only two things change step-by-step: the realized price $P_{t_0+s}$ (which updates $L_s$) and the wash-sale clock $\mathcal W_{t_0+s}=\mathcal W_{t_0}+s$ (which advances deterministically).

**Construction.** For a forward window $W=30$, and each step $s\in\{1,\dots,W\}$, define the per-step forward unrealized return and oracle firing indicator:

$$
L_s=\frac{P_{t_0+s}-p_k}{p_k},
\qquad
b_s(x)=f^*\!\bigl(L_s,\,\sigma_{TE},\,G_{t_0}^{YTD},\,\mathcal W_{t_0}+s\bigr)\in\{0,1\}.
$$

The soft backtest label is the time-average firing frequency:

$$
\tilde y_{BT}(x)
=
\frac{1}{W}
\sum_{s=1}^{W}b_s(x)
=
\frac{\#\{\text{days oracle fires in next 30}\}}{30}
\in\left\{0,\frac{1}{30},\frac{2}{30},\dots,1\right\}.
$$

**Interpretation.**

| Fires | $\tilde y_{BT}$ | Meaning |
|-------|-----------------|---------|
| 0 days | 0.00 | Never harvestable in the window |
| 6 days | 0.20 | Intermittently harvestable |
| 15 days | 0.50 | Harvestable half the window |
| 30 days | 1.00 | Persistently harvestable |

$\tilde y_{BT}$ is a harvest-urgency score, not a binary yes/no. A lot at $L=-40\%$ in a sustained drawdown scores near 1; a lot grazing $L=-2\%$ in a volatile market scores near 0.

When fewer than $W$ forward days remain in the data ($t_0+W\ge t_{\max}$), the label is undefined and the row is excluded from training (`NaN` rows dropped in `SelectTarget`).

**Binary target for classification.** The trainers use the "fires at least once" threshold:

$$
Y_{BT}
=
\mathbf 1[\tilde y_{BT}(X)>0],
\qquad
Y_{BT}\in\{0,1\},
\qquad
X\in\mathcal X.
$$

**Level-set interpretation** (from `data_memo_theory.md` §5.A). The learned posterior $\hat\eta$ defines a continuous urgency field over $\mathcal X$. Its level sets

$$
L_c
=
\{x\in\mathcal X:\hat\eta(x)=c\},
\qquad
c\in[0,1],
$$

are $(d-1)$-dimensional contours of harvest urgency in feature space. The oracle boundary $\partial\Omega$ corresponds to one such contour at $c\approx0.5$ under the hard-label regime; the soft label exposes the full graded field inside $\Omega$.

---

# 4. GBM Soft Label

The GBM soft label replaces the realized future price path with Monte Carlo simulated paths.

The simulated price process is geometric Brownian motion (GBM):

$$
dS_t=\mu S_t\,dt+\sigma S_t\,dW_t,
\qquad
S_t\in\mathbb R_{>0},
\qquad
W_t:\Omega_{\mathrm{prob}}\times[0,T]\to\mathbb R,
\qquad
dW_t\sim\mathcal N(0,dt).
$$

The discrete simulation update with time step $\Delta=1/252$ is

$$
S_s
=
S_{s-1}
\exp\!\left(
(\mu-\tfrac12\sigma^2)\Delta
+\sigma\sqrt{\Delta}\,Z_s
\right),
\qquad
Z_s\overset{\mathrm{iid}}\sim\mathcal N(0,1),
\qquad
Z_s\in\mathbb R.
$$

The $(\mu-\sigma^2/2)$ term is the Itô drift correction — it ensures the median (not the mean) of log-price tracks the drift $\mu$. Without it simulated paths have an upward bias equal to $\sigma^2T/2$. The implementation uses $\mu=0$ (risk-neutral default) and $\sigma$ estimated from trailing 21-day realized volatility, annualized as $\hat\sigma=\widehat{\mathrm{std}}(r)\cdot\sqrt{252}$.

Let $N_{\mathrm{paths}}=200$ be the number of simulated paths. For each path $p$, define the first-passage indicator:

$$
c_p(x)
=
\mathbf 1[\exists\,s\in\{1,\dots,W\}:b_s^{(p)}(x)=1],
\qquad
c_p:\mathcal X\to\{0,1\}.
$$

Here once a path fires, simulation of that path stops (first-passage semantics). The GBM soft label is

$$
\tilde y_{GBM}(x)
=
\frac{1}{N_{\mathrm{paths}}}
\sum_{p=1}^{N_{\mathrm{paths}}}c_p(x),
\qquad
\tilde y_{GBM}(x)\in[0,1].
$$

This is an unbiased Monte Carlo estimator of the probability the oracle fires within the window:

$$
\tilde y_{GBM}(x)
\approx
\mathbb P(\exists\,s\le W:\text{oracle fires}\mid X=x),
\qquad
\mathbb P(\cdots)\in[0,1].
$$

**Contrast with $\tilde y_{BT}$.** The backtest label uses one realized trajectory (deterministic, from actual historical prices). The GBM label uses 200 simulated trajectories (stochastic, from a fitted GBM). Both freeze the same portfolio state. The GBM label is more appropriate for out-of-sample generalization; the backtest label is more interpretable and directly grounded in realized data.

---

# 5. Shared Training Pipeline

Every supervised model uses the same pipeline on the same train/test split.

**Step 1 — Stratified train/test split.**

$$
D=D_{\mathrm{train}}\sqcup D_{\mathrm{test}},
\qquad
|D_{\mathrm{train}}|\approx0.8N,
\qquad
|D_{\mathrm{test}}|\approx0.2N.
$$

The split is stratified by class and uses `seed=42`. Because all models share the same seed, the leaderboard comparison is on an identical partition — a prerequisite for fair champion selection.

**Step 2 — Median imputation.** A map

$$
m:\mathcal X\to\mathcal X
$$

fit on $D_{\mathrm{train}}$ only, then applied to both folds. Fitting on the training fold only prevents label-leakage from test-set statistics.

**Step 3 — Balanced class weights.** Computed on $D_{\mathrm{train}}$ only:

$$
w_c=\frac{|D_{\mathrm{train}}|}{2\,n_c},
\qquad
c\in\{0,1\},
\qquad
n_c=\#\{i\in I_{\mathrm{train}}:y_i=c\}.
$$

Each training example carries weight $w_{y_i}\in\mathbb R_{>0}$. This is the ML.NET analogue of sklearn's `class_weight='balanced'`.

**Step 4 — 5-fold stratified cross-validation.** Hyperparameters are selected by mean PR-AUC over 5 folds, using $D_{\mathrm{train}}$ only.

**Step 5 — Refit on full $D_{\mathrm{train}}$.** Best hyperparameters from Step 4.

**Step 6 — Evaluate on $D_{\mathrm{test}}$ once.** Only the champion model(s) touch the test set (see §14).

**Preprocessing map.** The ML.NET pipeline applies mean-variance normalization and sector one-hot encoding before training:

$$
\phi(x,z)
=
\bigl[\,\mathrm{Norm}(x)\ \|\ \mathrm{OneHot}(\mathrm{Clean}(z))\,\bigr]
\in\mathbb R^{d},
\qquad
d=15+m,
$$

where $m=|\mathcal Z_{\mathrm{train}}|$ is the sector vocabulary learned on the training fold. Model weights $w$ below are over $\mathbb R^d$; the text writes $\mathbb R^{15}$ for clarity when sector encoding is not the focus.

---

# 6. Logistic Regression

Logistic regression learns a calibrated linear-logit posterior. 

The linear score is

$$
z(x)=w^\top x+b,
\qquad
z:\mathcal X\to\mathbb R,
\qquad
w\in\mathbb R^{15},
\qquad
b\in\mathbb R,
\qquad
z(x)\in\mathbb R.
$$

The sigmoid link is

$$
\sigma(u)=\frac{1}{1+e^{-u}},
\qquad
\sigma:\mathbb R\to(0,1),
\qquad
\sigma(u)\in(0,1).
$$

The predicted probability is

$$
\hat\eta(x)=\sigma(z(x))=\sigma(w^\top x+b),
\qquad
\hat\eta:\mathcal X\to(0,1).
$$

The binary cross-entropy loss for one example is (The Loss Function)

$$
\ell_{CE}(y,\hat\eta(x))
=
-y\log(\hat\eta(x))-(1-y)\log(1-\hat\eta(x)),
\qquad
\ell_{CE}:\{0,1\}\times(0,1)\to\mathbb R_{\ge0}.
$$

The weighted regularized objective is

$$
J(w,b)
=
\sum_{i\in I_{\mathrm{train}}}
w_{y_i}\,\ell_{CE}(y_i,\hat\eta(x_i))
+
\lambda\|w\|_2^2,
\qquad
J:\mathbb R^{15}\times\mathbb R\to\mathbb R_{\ge0},
\qquad
\lambda=1/C.
$$

The training problem is

$$
(w^*,b^*)
=
\arg\min_{w\in\mathbb R^{15},\,b\in\mathbb R}
J(w,b).
$$

Solved by L-BFGS. Hyperparameter grid: $C\in\{0.01,0.1,1.0,10.0\}$ (4 configs $\times$ 5 folds).

$\sigma$ is the link function that maps the linear score $z(x)\in\mathbb R$ into a probability $\hat\eta(x)\in(0,1)$. The objective is $J(w,b)$, not $\sigma$.

*Strengths:* fast to train, fully interpretable coefficients, well-calibrated probabilities, low variance. Baseline for all comparisons.

-footnote: needs expansion on loss objective / minimizing function synthesis in how the empirical function is generated where function J is eventually used to make the empirically learned function η^. along the diff in processes of what happens during testing (this) and vs testing (where the empirically learned function is used) in the base testing/training split and different types like k fold cv splits.


---

# 7. Elastic-Net Logistic Regression

The probability model is identical to logistic regression:

$$
z(x)=w^\top x+b,
\qquad
\hat\eta(x)=\sigma(z(x)),
\qquad
\hat\eta:\mathcal X\to(0,1).
$$

The difference is the regularization. The elastic-net objective adds both an $L_1$ and an $L_2$ penalty:

$$
J_{EN}(w,b)
=
\sum_{i\in I_{\mathrm{train}}}
w_{y_i}\,\ell_{CE}(y_i,\hat\eta(x_i))
+
\lambda_1\|w\|_1
+
\lambda_2\|w\|_2^2,
\qquad
J_{EN}:\mathbb R^{15}\times\mathbb R\to\mathbb R_{\ge0}.
$$

The two norms are

$$
\|w\|_1=\sum_{j=1}^{15}|w_j|,
\qquad
\|w\|_2^2=\sum_{j=1}^{15}w_j^2.
$$

The $L_1$ term encourages sparsity — it drives small coefficients exactly to zero, performing implicit feature selection. The $L_2$ term shrinks remaining coefficients, reducing variance.

Hyperparameter grid: $\lambda_1\in\{0.001,0.01,0.1\}$, $\lambda_2\in\{0.001,0.01,0.1\}$ (9 configs $\times$ 5 folds). Solved by SDCA.

The training problem is

$$
(w^*,b^*)
=
\arg\min_{w\in\mathbb R^{15},\,b\in\mathbb R}
J_{EN}(w,b).
$$

*Strengths:* automatic feature selection via $L_1$, reduced overfitting via $L_2$, calibrated probabilities from the logistic link.

---

# 8. Gradient Boosted Trees

Gradient boosted trees build an additive ensemble of regression trees, each correcting the errors of the previous ensemble.

The score function is

$$
F_M:\mathcal X\to\mathbb R,
\qquad
F_M(x)\in\mathbb R.
$$

It is built additively, starting from an initial constant score $F_0(x)=\mathrm{logit}(\bar y)$:

$$
F_M(x)
=
F_0(x)
+
\sum_{m=1}^{M}\nu\, f_m(x).
$$

Here $M\in\mathbb Z_{>0}$ is the number of trees, $\nu\in\mathbb R_{>0}$ is the learning rate (shrinkage), and $f_m:\mathcal X\to\mathbb R$ is the $m$-th regression tree.

Each tree $f_m$ is fit to the pseudo-residuals from the previous step — the negative gradient of logistic loss in score space:

$$
r_i^{(m)}
=
y_i-\sigma(F_{m-1}(x_i)),
\qquad
r_i^{(m)}\in\mathbb R,
\qquad
y_i\in\{0,1\},
\qquad
\sigma(F_{m-1}(x_i))\in(0,1).
$$

Each tree approximates the map $x_i\mapsto r_i^{(m)}$ by minimizing weighted squared error.

The model probability uses Platt calibration to map the raw score into $[0,1]$:

$$
\hat\eta(x)
=
\sigma(aF_M(x)+b),
\qquad
a\in\mathbb R,
\qquad
b\in\mathbb R,
\qquad
\hat\eta(x)\in(0,1).
$$

Hyperparameter grid: $M\in\{100,200\}$, $\nu\in\{0.10,0.05\}$, $J\in\{20,31\}$ leaves (8 configs $\times$ 5 folds).

*Strengths:* captures nonlinear interactions, often the strongest single model, sequentially reduces bias by targeting residuals. The small learning rate $\nu$ with many trees gives a regularized fit that generalizes well.

---

# 9. Random Forest

A random forest builds many independent trees in parallel, then averages them.

Each tree is a function

$$
f_t:\mathcal X\to[0,1],
\qquad
t\in\{1,\dots,T\},
\qquad
T\in\mathbb Z_{>0}.
$$

Trees are trained on bootstrap resamples of $D_{\mathrm{train}}$. At each split, only a random fraction $\kappa\in(0,1)$ of features are considered, which decorrelates the trees.

The forest score is

$$
F(x)
=
\frac{1}{T}
\sum_{t=1}^{T}f_t(x),
\qquad
F:\mathcal X\to[0,1],
\qquad
F(x)\in[0,1].
$$

**Variance decomposition.** If each tree has variance $\varsigma^2$ and pairwise correlation $\rho$, the ensemble variance is

$$
\mathrm{Var}(F(x))
=
\rho\varsigma^2+\frac{1-\rho}{T}\varsigma^2.
$$

As $T\to\infty$ the second term vanishes. The feature subsampling (fraction $\kappa=0.7$) lowers $\rho$, making the variance reduction effective.

FastForest is uncalibrated — it emits a raw `Score` but no `Probability` column. The score is used directly as a probability proxy for evaluation.

Hyperparameter grid: $T\in\{100,200\}$, $J\in\{20,31\}$ leaves (4 configs $\times$ 5 folds).

*Strengths:* low variance through averaging, handles nonlinear relationships, robust to irrelevant features, harder to overfit than a single deep tree. Contrast with boosting: boosting reduces bias sequentially; bagging (RF) reduces variance in parallel.

---

# 10. Linear Regression as Negative Control

Linear regression is deliberately included as a structurally misspecified model to serve as a negative control.

The model is

$$
\hat y(x)=w^\top x+b,
\qquad
\hat y:\mathcal X\to\mathbb R,
\qquad
w\in\mathbb R^{15},
\qquad
b\in\mathbb R,
\qquad
\hat y(x)\in\mathbb R.
$$

The squared-error loss for one example is

$$
\ell_{SE}(y,\hat y(x))
=
(y-\hat y(x))^2,
\qquad
y\in\{0,1\},
\qquad
\hat y(x)\in\mathbb R,
\qquad
\ell_{SE}:\{0,1\}\times\mathbb R\to\mathbb R_{\ge0}.
$$

The weighted regularized objective is

$$
J_{OLS}(w,b)
=
\sum_{i\in I_{\mathrm{train}}}
w_{y_i}
(y_i-\hat y(x_i))^2
+
\lambda\|w\|_2^2,
\qquad
J_{OLS}:\mathbb R^{15}\times\mathbb R\to\mathbb R_{\ge0}.
$$

Hyperparameter grid: $\lambda\in\{10^{-4},10^{-3},10^{-2}\}$ (3 configs $\times$ 5 folds).

**Why this is structurally wrong.** The pipeline targets a calibrated posterior $\hat\eta:\mathcal X\to[0,1]$. The oracle boundary $\partial\Omega$ is a level-curve of this posterior in feature space. Linear regression has no sigmoid link, so

$$
\hat y(x)\in\mathbb R,
\qquad\text{but a valid probability requires}\qquad
\hat\eta(x)\in[0,1].
$$

Predictions escape $[0,1]$ — the diagnostic $\mathrm{FractionOutside}$ counts this:

$$
\mathrm{FractionOutside}
=
\frac{1}{|D_{\mathrm{test}}|}
\sum_{i\in I_{\mathrm{test}}}
\mathbf 1\bigl[\hat y(x_i)<0\ \lor\ \hat y(x_i)>1\bigr]
\in[0,1].
$$

(Empirically $\approx 25\%$ of test predictions fall outside $[0,1]$.) Further, Bernoulli outcomes are intrinsically heteroskedastic ($\mathrm{Var}(Y\mid x)=\eta(x)(1-\eta(x))$), violating the Gauss–Markov homoskedasticity assumption. Linear regression is the sharpest principled contrast to the logistic/GBT/RF family.

---

# 11. Evaluation Metrics

For each test example $i$, the model produces

$$
\hat\eta_i=\hat\eta(x_i),
\qquad
\hat\eta_i\in[0,1],
\qquad
y_i\in\{0,1\}.
$$

A threshold rule is

$$
\hat y_i(\tau)
=
\mathbf 1[\hat\eta_i\ge\tau],
\qquad
\tau\in[0,1],
\qquad
\hat y_i(\tau)\in\{0,1\}.
$$

Sweeping $\tau$ over sorted $\hat\eta$ values traces out the ROC and PR curves.

**Precision and Recall.**

$$
P(\tau)
=
\frac{TP(\tau)}{TP(\tau)+FP(\tau)},
\qquad
R(\tau)
=
\frac{TP(\tau)}{TP(\tau)+FN(\tau)},
\qquad
P(\tau),R(\tau)\in[0,1].
$$

**$F_1$ score.**

$$
F_1(\tau)
=
\frac{2\,P(\tau)\,R(\tau)}{P(\tau)+R(\tau)},
\qquad
F_1(\tau)\in[0,1].
$$

Reported at $\tau=0.5$ and at the $\tau^*$ that maximizes $F_1$.

**ROC-AUC.** The trapezoidal integral under the TPR vs. FPR curve. Measures overall ranking quality — does the model rank positive lots above negative lots?

**PR-AUC (average precision).** The step-integral under the Precision vs. Recall curve:

$$
\mathrm{PRAUC}=\sum_k(R_k-R_{k-1})\cdot P_k.
$$

**PR-AUC is the primary model-selection metric.** Harvest events are rare (class imbalance). ROC-AUC is optimistic under imbalance because it gives equal weight to the large negative class. PR-AUC is sensitive to the model's behavior in the minority class, where tax-alpha is generated.

---

# 12. PCA

Let the standardized numeric data matrix be

$$
X\in\mathbb R^{n\times15},
\qquad
n=|D_{\mathrm{train}}|.
$$

Each row is one observation with zero mean and unit variance per column.

The sample covariance matrix is

$$
C=\frac{1}{n-1}X^\top X,
\qquad
C\in\mathbb R^{15\times15}.
$$

The eigendecomposition is

$$
C\,u_j=\lambda_j\,u_j,
\qquad
u_j\in\mathbb R^{15},
\qquad
\lambda_j\in\mathbb R_{\ge0},
\qquad
\lambda_1\ge\lambda_2\ge\dots\ge\lambda_{15}\ge0.
$$

The explained-variance ratio is

$$
\rho_j
=
\frac{\lambda_j}{\displaystyle\sum_{k=1}^{15}\lambda_k},
\qquad
\rho_j\in[0,1],
\qquad
\sum_{j=1}^{15}\rho_j=1.
$$

The retained dimension is the smallest $r$ that covers 95% of variance:

$$
r
=
\min\!\left\{r':\sum_{j=1}^{r'}\rho_j\ge0.95\right\}.
$$

The PCA projection is

$$
\pi_r(x)=U_r^\top x,
\qquad
U_r=[u_1,\dots,u_r]\in\mathbb R^{15\times r},
\qquad
\pi_r:\mathbb R^{15}\to\mathbb R^r.
$$

The low-rank structure is justified by the factor model for asset returns ($\Sigma=B\Sigma_F B^\top+D$, from `data_memo_theory.md` §6): a spectral gap after the leading market and sector factors means $r\ll15$.

---

# 13. K-Means

Each asset $s$ is represented by a four-dimensional aggregate of asset-level features:

$$
g_s\in\mathbb R^4,
\qquad
g_s=(R_t,\Sigma_{\mathrm{Range}},\Delta\mathrm{MA}_{50},\Delta\mathrm{MA}_{200})_s.
$$

K-means chooses cluster centers

$$
\mu_1,\dots,\mu_k\in\mathbb R^4,
$$

and an assignment map

$$
c:\{s\}\to\{1,\dots,k\},
\qquad
c(s)\in\{1,\dots,k\},
$$

minimizing within-cluster sum of squares:

$$
J_{KM}
=
\min_{\mu_1,\dots,\mu_k,\,c}
\sum_s\|g_s-\mu_{c(s)}\|_2^2,
\qquad
J_{KM}\in\mathbb R_{\ge0}.
$$

Lloyd's algorithm alternates between (1) assigning each point to its nearest center and (2) updating each center to the mean of its cluster, until convergence.

The cluster count $k\in\{5,10,15,20,25\}$ is chosen by maximum average silhouette score:

$$
\mathrm{sil}(s)
=
\frac{b(s)-a(s)}{\max\{a(s),b(s)\}},
\qquad
\mathrm{sil}(s)\in[-1,1],
$$

where $a(s)$ is the mean intra-cluster distance and $b(s)$ is the mean distance to the nearest other cluster. K-means here is used for exploratory discovery of market-regime clusters, not for prediction.

---

# 14. Big Picture

## Approximation chain

The full sequence of approximations is

$$
f^*_{\mathrm{true}}
\xrightarrow{\ \text{myopic approximation}\ }
f^*_{\mathrm{oracle}}
\xrightarrow{\ \text{forward window}\ }
\tilde y_{BT}
\xrightarrow{\ \text{ERM over }\mathcal H\ }
\hat\eta.
$$

1. $f^*_{\mathrm{true}}$ is the truly optimal harvest policy — a stopping-time problem over future prices, intractable without price-path knowledge.
2. $f^*_{\mathrm{oracle}}$ is the mechanistic threshold rule — admissible from current state, near-optimal under stylized assumptions, but myopic and unable to rank lots inside $\Omega$.
3. $\tilde y_{BT}$ adds forward-window information — it converts the binary boundary into a graded urgency field.
4. $\hat\eta$ is the learned model — a smooth calibrated posterior over $\mathcal X$.

## What each layer contributes

```text
Oracle:     defines the harvest boundary ∂Ω
Soft label: measures harvest urgency inside Ω
ML model:   learns η̂(x), a smooth field whose level sets
            rank candidate lots by urgency and prioritize execution
```

The oracle fixes the boundary (Source 1 of tax-alpha — the choice of $\theta_1,\theta_2$ — irreducible by any classifier). The ML model addresses the interior (Source 2 — which lots inside $\Omega$ to harvest first, and how urgently — closeable by learning from soft labels).

The ML model's job is not to replace the oracle. Its job is to learn the harvest urgency landscape inside $\Omega$ and estimate $\hat\eta(x)$ — enabling graded prioritization that a binary rule structurally cannot provide.

# Data Memo — PSTAT 231 (Formal Mapping)
### Gabriel Kung 
## *Machine Learning for Tax-Loss Harvesting Decisions in a Simulated Direct Indexing Portfolio*

---

## 1. The Probability Space and Data-Generating Process

Before specifying any model, the underlying probabilistic structure must be made explicit.

Fix a probability space $(\mathcal{X} \times \mathcal{Y},\ \mathcal{F},\ \mathcal{D})$ where:

- $\mathcal{X} \subset \mathbb{R}^d$ is the **feature space** (a measurable subset of $\mathbb{R}^d$, $d \approx 10$–$20$), equipped with the Borel $\sigma$-algebra $\mathcal{B}(\mathcal{X})$.
- $\mathcal{Y} = \{0, 1\}$ is the **label space**, equipped with the discrete $\sigma$-algebra $2^{\mathcal{Y}}$.
- $\mathcal{D}$ is an unknown **joint distribution** over $\mathcal{X} \times \mathcal{Y}$, the true data-generating mechanism.
- $\mathcal{F} = \mathcal{B}(\mathcal{X}) \otimes 2^{\mathcal{Y}}$ is the product $\sigma$-algebra.

Each observation $(x_i, y_i) \in \mathcal{X} \times \{0,1\}$ is a realization of a random vector $(X, Y)$ where $(X, Y) \sim \mathcal{D}$.

The **marginal** $\mathcal{D}_X$ is the distribution over feature vectors — the law of $X$. The **conditional** $\mathcal{D}_{Y|X}$ encodes label uncertainty given features. Together they factor as:

$$\mathcal{D}(x, y) = \mathcal{D}_{Y|X}(y \mid x) \cdot \mathcal{D}_X(x)$$

The key unknown object driving everything downstream is the **posterior function**:

$$\eta: \mathcal{X} \to [0,1], \qquad \eta(x) := P(Y = 1 \mid X = x) = \mathbb{E}[Y \mid X = x]$$

$\eta$ is a measurable function (a random variable $\mathcal{X} \to \mathbb{R}$ in the measure-theoretic sense — it is $\mathcal{B}(\mathcal{X})$-measurable). It is the single object any classifier is ultimately trying to estimate, approximate, or threshold.

The **Bayes-optimal classifier** is the minimizer of risk under 0-1 loss:

$$f^{\text{Bayes}}(x) := \mathbb{1}\!\left[\eta(x) \geq \tfrac{1}{2}\right]$$

This minimizes the **Bayes risk** $R^* := \mathbb{E}_{x}\!\left[\min\{\eta(x),\, 1-\eta(x)\}\right]$, which is a lower bound on the risk of any classifier. No algorithm can do better in expectation than $f^{\text{Bayes}}$.

The oracle rule (defined precisely in §3) is an explicit approximation to $f^{\text{Bayes}}$ under strong structural assumptions about the form of $\eta$.

---

## 2. Feature Space Geometry and the Panel Structure

### 2.1 Feature Vectors as Elements of a Structured Space

Each observation corresponds to a single **tax lot** at a single **time step**. The feature vector $x \in \mathcal{X}$ is:

$$x = \underbrace{(B,\ G,\ \ell,\ h,\ s,\ w,\ k,\ G_{\text{YTD}},\ \sigma_{\text{TE}})}_{\mathcal{X}_{\text{lot}} \subset \mathbb{R}^9}\ \times\ \underbrace{(P_t,\ \hat{\sigma},\ r_t,\ z)}_{\mathcal{X}_{\text{asset}} \subset \mathbb{R}^3 \times \mathcal{Z}}$$

where the types are:
- $B \in \mathbb{R}_{>0}$: cost basis (scalar, dollars per share)
- $G = P_t - B \in \mathbb{R}$: unrealized gain/loss (signed scalar)
- $\ell = G / B \in (-1, \infty)$: normalized unrealized return (dimensionless scalar)
- $h \in \mathbb{Z}_{\geq 0}$: holding period in days (non-negative integer, often treated as continuous)
- $s = \mathbb{1}[h \geq 365] \in \{0,1\}$: short/long-term flag (binary)
- $w \in (0,1)$: portfolio weight of the lot (scalar)
- $k \in \mathbb{Z}_{>0}$: lot count for this ticker (positive integer)
- $G_{\text{YTD}} \in \mathbb{R}$: net realized gain year-to-date (signed scalar)
- $\sigma_{\text{TE}} \in \mathbb{R}_{\geq 0}$: current portfolio tracking error (non-negative scalar)
- $P_t \in \mathbb{R}_{>0}$: current asset price
- $\hat{\sigma} \in \mathbb{R}_{>0}$: realized volatility estimate
- $r_t \in \mathbb{R}$: recent return
- $z \in \mathcal{Z}$: sector (categorical; embedded as a vector in $\mathbb{R}^k$ via one-hot or learned embedding)

The full feature space is thus $\mathcal{X} \subset \mathbb{R}^d$ with $d$ depending on the encoding of $z$.

### 2.2 Panel Structure and the i.i.d. Assumption

The raw dataset is a **panel**: observations are indexed by $(i, t)$ where $i \in \{1,\ldots, n\}$ indexes the stock/lot and $t \in \{1,\ldots, m\}$ indexes the time step. The total number of observations is $N = nm$.

The true joint distribution of the panel is a stochastic process:

$$\{(X_{i,t}, Y_{i,t})\}_{i \in [n],\, t \in [m]}$$

This process exhibits both **cross-sectional dependence** (assets are correlated via shared factor exposures — see §6) and **temporal dependence** (each lot's features evolve as a Markov chain or more complex time series).

However, the key modeling choice is to treat each observation $(x_{i,t}, y_{i,t})$ as **conditionally independent** given its feature vector:

$$Y_{i,t} \perp Y_{j,s} \mid X_{i,t}, \quad \forall (i,t) \neq (j,s)$$

This is justified by the structure of the oracle label (§3): $y_{i,t}$ is a deterministic function of $x_{i,t}$ alone, so conditioning on $x_{i,t}$ screens off all other lots and time steps. In measure-theoretic terms, $Y_{i,t}$ is $\sigma(X_{i,t})$-measurable under the oracle. The portfolio is time-dependent, but this dependence is **encapsulated in the features** $x_{i,t}$ themselves — $\sigma_{\text{TE}}$, $G_{\text{YTD}}$, and $h$ all carry forward-accumulated state. The temporal process is thus **summarized** by the Markovian feature snapshot; the ML model conditions on the sufficient statistic.

This separation of concerns — portfolio state machine (temporal) vs. harvest classifier (atemporal conditional on state) — is the architectural invariant that permits applying standard ERM-based classification without time-series machinery.

---

## 3. The Oracle: Labeling Function, Decision Region, and Boundary Geometry

### 3.1 Oracle as a Concept

Formally, an **oracle** is a deterministic measurable function $f^*: \mathcal{X} \to \{0,1\}$ that induces the training labels. The oracle here is the **mechanical direct indexing rule** — a threshold policy derived from first-order tax-alpha considerations:

$$f^*(x) = \mathbb{1}\!\left[\ell \leq -\theta_1\right] \cdot \mathbb{1}\!\left[\sigma_{\text{TE}} \leq \theta_2\right] \cdot \mathbb{1}\!\left[G_{\text{YTD}} > 0\right]$$

where:
- $\theta_1 > 0$: the loss threshold (e.g., $\theta_1 = 0.02$, harvest at 2% unrealized loss)
- $\theta_2 > 0$: the tracking error budget
- $G_{\text{YTD}} > 0$: ensures harvestable gains exist to offset

This is a conjunction of halfspace indicators — each condition defines a closed halfspace in $\mathcal{X}$, and $f^*$ is the indicator of their intersection:

$$\Omega := \{x \in \mathcal{X} : f^*(x) = 1\} = H_1 \cap H_2 \cap H_3$$

where $H_1 = \{x : \ell \leq -\theta_1\}$, $H_2 = \{x : \sigma_{\text{TE}} \leq \theta_2\}$, $H_3 = \{x : G_{\text{YTD}} > 0\}$. The **harvest region** $\Omega \subset \mathcal{X}$ is therefore a convex polytope (intersection of halfspaces) in the relevant feature coordinates.

### 3.2 Why the Oracle Cannot Be the True Optimum

The true optimal label for lot $(i,t)$ would require knowledge of the price path $\{P_{i,s}\}_{s > t}$: specifically, whether the tax benefit realized at $t$ dominates the cost of being out of the position (or holding a correlated replacement) over the 30-day wash-sale window, compounded forward.

Formally, the truly optimal harvest decision is a **stopping time** problem: find $\tau^* = \arg\min_\tau \mathbb{E}[\text{after-tax portfolio value at horizon} \mid \mathcal{F}_t]$, which is a solution to an optimal stopping problem over the filtration $\mathcal{F}_t = \sigma(\{P_{i,s}\}_{s \leq t})$ of the price process. This is:

1. Forward-looking (inadmissible as a training label without data leakage)
2. Path-dependent (requires solving a Bellman equation or PDE)
3. Generally intractable without strong distributional assumptions

The oracle substitutes a **myopic, threshold-based approximation** that is admissible (computable from current state) and near-optimal under stylized conditions. The gap between oracle labels and true labels is a form of **structural label noise** — not random, but systematically biased by the oracle's inability to anticipate future prices. The ML model cannot overcome this ceiling; it can only approximate $f^*$, not $f^*_{\text{true}}$.

---

## 4. The Boundary–Interior Analogy: Generalized Stokes and Currents

This is the conceptually richest part of the problem structure and deserves careful treatment. The analogy is not merely rhetorical — it has a precise formulation in the language of **geometric measure theory** and **currents**.

### 4.1 Setup: The Feature Space as a Manifold

Treat $\mathcal{X} \subset \mathbb{R}^d$ as an oriented smooth manifold (with boundary, if $\mathcal{X}$ has constraints). The harvest region $\Omega \subset \mathcal{X}$ is a compact submanifold-with-boundary (a measurable set with rectifiable boundary under mild regularity assumptions on the oracle thresholds $\theta_1, \theta_2$).

### 4.2 Currents and the Generalized Stokes Theorem

A **$k$-current** $T$ on $\mathcal{X}$ is a continuous linear functional on the space $\mathcal{D}^k(\mathcal{X})$ of compactly supported smooth $k$-forms on $\mathcal{X}$. This is the dual space to differential forms — a distributional generalization of submanifolds.

Any oriented compact $k$-dimensional submanifold $M \subset \mathcal{X}$ induces a $k$-current $[\![M]\!]$ via:

$$[\![M]\!](\omega) = \int_M \omega, \quad \omega \in \mathcal{D}^k(\mathcal{X})$$

The **boundary operator** on currents, $\partial: \mathcal{D}_k \to \mathcal{D}_{k-1}$, is defined by duality with the exterior derivative:

$$(\partial T)(\omega) := T(d\omega)$$

The **generalized Stokes theorem** is then the tautology: $\partial [\![M]\!] = [\![\partial M]\!]$, or explicitly:

$$\int_{\partial M} \omega = \int_M d\omega$$

Now apply this to our problem. The harvest region $\Omega \subset \mathcal{X}$ is a $d$-dimensional region, so $[\![\Omega]\!]$ is a $d$-current — integration over the full region. Its boundary is:

$$\partial [\![\Omega]\!] = [\![\partial \Omega]\!]$$

where $\partial \Omega = \{x : \ell = -\theta_1\} \cup \{x : \sigma_{\text{TE}} = \theta_2\} \cup \{x : G_{\text{YTD}} = 0\}$ (the boundary hypersurfaces corresponding to each binding oracle condition) is a $(d-1)$-current — integration over the decision boundary.

### 4.3 The Analogy, Made Precise

| Object | Differential Geometry | This Problem |
|---|---|---|
| $d$-current $[\![\Omega]\!]$ | Integration over interior region | ML model: estimates $\eta(x)$ over $\Omega^\circ$ |
| $(d-1)$-current $[\![\partial \Omega]\!]$ | Integration over boundary | Mechanical oracle: fires on $\partial \Omega$ |
| $\partial [\![\Omega]\!] = [\![\partial \Omega]\!]$ | Stokes theorem | Oracle threshold is the boundary of the ML model's domain |
| Closed current ($\partial T = 0$) | Topological invariant / cycle | Oracle-consistent decisions with no boundary leakage |

The **mechanical direct indexing rule** is precisely the boundary current $[\![\partial \Omega]\!]$: it activates exactly when the feature vector hits the boundary hypersurface $\partial \Omega$.

The **ML model** approximates the interior current $[\![\Omega]\!]$: it learns the posterior $\eta(x)$ over the full interior $\Omega^\circ$, capturing graded probability of harvest even for feature vectors that are strictly inside the decision region.

The critical implication from Stokes: **knowing the boundary current does not uniquely determine the interior current**, up to the kernel of $\partial$. Formally, if $T$ and $T'$ are two currents with $\partial T = \partial T' = [\![\partial \Omega]\!]$, then $T - T'$ is a **closed current** ($\partial(T - T') = 0$) — it lives in $\ker(\partial)$. In cohomological terms, $T - T'$ represents a homology class in $H_d(\mathcal{X}; \mathbb{R})$.

Applied: there exist many posterior functions $\eta$ that agree on $\partial \Omega$ (the oracle's firing surface) but differ throughout $\Omega^\circ$. The ML model resolves this ambiguity via its hypothesis class and training data. **The oracle constrains $\hat{f}$ only at the boundary**; the inductive bias of the learning algorithm determines behavior in the interior.

Conversely, the ML model's learned interior $\eta(x)$ cannot retroactively validate whether the oracle's boundary $\partial \Omega$ (the choice of $\theta_1, \theta_2$) is optimal — that would require evaluating performance outside $\Omega$ and reasoning about regions the oracle never labels as positive. The interior does not determine the boundary.

### 4.4 The Dirichlet Problem Interpretation

A complementary formulation: the oracle defines **Dirichlet boundary conditions** on the posterior:

$$\eta(x) = 1, \quad x \in \partial \Omega \quad (\text{oracle fires with certainty on the threshold})$$

The ML model is then solving the interior problem: given this boundary condition, find $\eta: \Omega^\circ \to [0,1]$ consistent with the training data. In classical PDE theory, this is the **Dirichlet problem**: find $u$ on $\Omega$ satisfying $\mathcal{L}u = 0$ in $\Omega^\circ$ and $u = g$ on $\partial \Omega$ for some differential operator $\mathcal{L}$ and boundary data $g$.

There is no natural PDE constraining $\eta$ in the interior — it is not harmonic or biharmonic by any physical argument. But the ML model's **regularization** acts as the surrogate differential operator: it selects among all interior extensions compatible with boundary data by imposing smoothness (e.g., Tikhonov/RKHS regularization penalizes $\|\nabla \eta\|^2$), which is exactly a Dirichlet energy penalty. The unique minimizer of $\int_\Omega \|\nabla \eta\|^2\, dx$ subject to Dirichlet conditions on $\partial \Omega$ is the harmonic extension — a smooth, unique solution to $\Delta \eta = 0$ in $\Omega^\circ$.

Regularized ML models are therefore implicitly solving a **variational problem** on $\Omega$ where the penalty encodes the smoothness assumption about $\eta$'s interior structure.

---

## 5. Statistical Learning Theory: ERM and Generalization

### 5.1 Risk, Empirical Risk, and the ERM Principle

Fix a loss function $\mathcal{L}: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\geq 0}$. For classification, the canonical choice is the **0-1 loss** $\mathcal{L}(y, \hat{y}) = \mathbb{1}[y \neq \hat{y}]$.

The **true risk** (population risk) of a classifier $f: \mathcal{X} \to \mathcal{Y}$ under $\mathcal{D}$ is:

$$R(f) := \mathbb{E}_{(X,Y) \sim \mathcal{D}}\!\left[\mathcal{L}(f(X), Y)\right] = P_{(X,Y) \sim \mathcal{D}}(f(X) \neq Y)$$

$R: \mathcal{H} \to \mathbb{R}_{\geq 0}$ is a functional on the hypothesis class $\mathcal{H}$ (a function space). The goal is:

$$f^{\text{opt}} = \arg\min_{f \in \mathcal{H}} R(f)$$

Since $\mathcal{D}$ is unknown, $R(f)$ is not directly computable. Given a training sample $S = \{(x_i, y_i)\}_{i=1}^N \overset{\text{i.i.d.}}{\sim} \mathcal{D}$, the **empirical risk** is:

$$\hat{R}_S(f) := \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

$\hat{R}_S: \mathcal{H} \to \mathbb{R}_{\geq 0}$ is a random functional — it depends on the sample $S$. **Empirical Risk Minimization (ERM)** produces:

$$\hat{f}_{\text{ERM}} = \arg\min_{f \in \mathcal{H}} \hat{R}_S(f)$$

The fundamental question of statistical learning theory is: under what conditions does $R(\hat{f}_{\text{ERM}}) \to R(f^{\text{opt}})$ as $N \to \infty$?

### 5.2 Generalization Bound via Rademacher Complexity

The **generalization gap** $R(\hat{f}) - \hat{R}_S(\hat{f})$ is controlled by the **Rademacher complexity** of $\mathcal{H}$:

$$\mathfrak{R}_N(\mathcal{H}) := \mathbb{E}_{S, \boldsymbol{\sigma}}\!\left[\sup_{f \in \mathcal{H}} \frac{1}{N} \sum_{i=1}^N \sigma_i f(x_i)\right]$$

where $\sigma_i \overset{\text{i.i.d.}}{\sim} \text{Uniform}(\{-1, +1\})$ are **Rademacher variables** — random signs. $\mathfrak{R}_N(\mathcal{H})$ measures how well the best $f \in \mathcal{H}$ can correlate with random noise: a larger hypothesis class that can fit more patterns also fits more noise, yielding higher $\mathfrak{R}_N$.

The **Rademacher generalization bound** states: with probability at least $1 - \delta$ over draws of $S$,

$$R(\hat{f}) \leq \hat{R}_S(\hat{f}) + 2\mathfrak{R}_N(\mathcal{H}) + \sqrt{\frac{\log(1/\delta)}{2N}}$$

For the oracle-labeled problem here, the training labels are **realizable**: since $f^* = \mathbb{1}_\Omega$ is a conjunction of halfspace indicators, if $\mathcal{H}$ contains $f^*$ then $\hat{R}_S(f^*) = 0$, and the generalization bound reduces to:

$$R(\hat{f}) \leq 2\mathfrak{R}_N(\mathcal{H}) + \sqrt{\frac{\log(1/\delta)}{2N}}$$

This makes the complexity of $\mathcal{H}$ critical: sufficiently expressive to contain $f^*$ (or approximate it well), but not so expressive that $\mathfrak{R}_N(\mathcal{H})$ is large.

### 5.3 The Oracle as a Structured Noise Process

Even in the realizable case (ML model can perfectly fit oracle labels), there remains a gap between $f^*_{\text{oracle}}$ and $f^*_{\text{true}}$ — the truly optimal policy. This gap constitutes **systematic label noise** with structure:

$$y_{i,t} = f^*(x_{i,t}) + \varepsilon_{\text{oracle}}(x_{i,t})$$

where $\varepsilon_{\text{oracle}}(x) := f^*_{\text{true}}(x) - f^*(x)$ is a deterministic but unknown **misspecification error**. This is an **agnostic** learning setting relative to the true labels, even though it is realizable relative to oracle labels.

The irreducible component of this error is the Bayes risk induced by the oracle misspecification — no ML algorithm can close this gap without access to future price data.

---

## 6. Dimensionality Reduction: Factor Structure and PCA

### 6.1 The Factor Model for Asset Returns

The S&P 500 return covariance matrix $\Sigma \in \mathbb{R}^{500 \times 500}$ is **approximately low-rank** due to shared macroeconomic factor exposures. The **Fama-French** (or more general APT) $k$-factor model specifies:

$$r_i = \alpha_i + \sum_{j=1}^k \beta_{ij} f_j + \varepsilon_i, \quad i = 1,\ldots, 500$$

In matrix form: $\mathbf{r} = \boldsymbol{\alpha} + B \mathbf{f} + \boldsymbol{\varepsilon}$ where:
- $\mathbf{r} \in \mathbb{R}^{500}$: vector of asset returns
- $B \in \mathbb{R}^{500 \times k}$: **factor loading matrix** (type: linear map $\mathbb{R}^k \to \mathbb{R}^{500}$)
- $\mathbf{f} \in \mathbb{R}^k$: factor returns (random vector, $k \ll 500$)
- $\boldsymbol{\varepsilon} \in \mathbb{R}^{500}$: idiosyncratic returns, assumed $\text{Cov}(\boldsymbol{\varepsilon}) = D$ (diagonal)
- $\text{Cov}(\mathbf{f}) = \Sigma_F \in \mathbb{R}^{k \times k}$: factor covariance matrix

Under this model:
$$\Sigma = \text{Cov}(\mathbf{r}) = B \Sigma_F B^\top + D$$

The **systematic component** $B \Sigma_F B^\top$ is a rank-$k$ positive semidefinite matrix (type: PSD bilinear form on $\mathbb{R}^{500}$). This implies $\Sigma$ is well-approximated by a rank-$k$ matrix plus a diagonal perturbation — exactly the structure PCA exploits.

### 6.2 PCA as Eigendecomposition in the Factor Basis

Let $\Sigma = U \Lambda U^\top$ be the eigendecomposition of $\Sigma$, where $U \in O(500)$ (orthogonal matrix, type: change-of-basis map on $\mathbb{R}^{500}$) and $\Lambda = \text{diag}(\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_{500} \geq 0)$.

The first $k$ eigenvectors (columns of $U_k \in \mathbb{R}^{500 \times k}$) span the **principal subspace** — the $k$-dimensional subspace of $\mathbb{R}^{500}$ capturing maximal variance. The **explained variance ratio** of the $k$-truncation is:

$$\frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^{500} \lambda_j} = \frac{\text{tr}(U_k \Lambda_k U_k^\top)}{\text{tr}(\Sigma)}$$

In practice for S&P 500 returns, the top 5–10 factors (market, size, value, momentum, quality, etc.) explain 60–80% of cross-sectional variance, with a sharp **spectral gap** between $\lambda_k$ and $\lambda_{k+1}$ — justifying truncation.

### 6.3 Subset Selection as a Combinatorial Optimization

The goal of the unsupervised component is not merely dimensionality reduction on features — it is **stock subset selection**: find the smallest $S \subset \{1,\ldots, 500\}$ with $|S| = n$ such that a portfolio with weights $w_S \in \Delta^{n-1}$ (the probability simplex) achieves:

$$\sigma_{\text{TE}}^2(S) = (w_S^{\text{ext}} - w^B)^\top \Sigma (w_S^{\text{ext}} - w^B) \leq \epsilon^2$$

where $w_S^{\text{ext}} \in \mathbb{R}^{500}$ is the extended weight vector (zero on assets not in $S$) and $w^B \in \mathbb{R}^{500}$ is the benchmark weight vector. $\sigma_{\text{TE}}^2$ is a **quadratic form** in the weight difference vector with kernel $\Sigma$.

This is a combinatorial optimization (NP-hard in general) typically relaxed to:
- **Greedy selection**: sequentially add the asset that maximally reduces $\sigma_{\text{TE}}^2$
- **$\ell_1$-penalized regression** (basis pursuit): regress benchmark weights on asset returns, select assets with nonzero coefficients
- **Spectral methods**: select assets that span the principal subspace $U_k$

---

## 7. Tracking Error as a Quadratic Form

Let $r_P(t) = w_S^\top r_S(t)$ and $r_B(t) = (w^B)^\top r(t)$ be portfolio and benchmark returns at time $t$. Define the **tracking difference** $d(t) := r_P(t) - r_B(t)$.

Then:

$$\sigma_{\text{TE}}^2 = \text{Var}(d(t)) = \mathbb{E}[d(t)^2] - (\mathbb{E}[d(t)])^2$$

In the special case $\mathbb{E}[d] \approx 0$ (which holds if $w_S^{\text{ext}} \approx w^B$ in expectation):

$$\sigma_{\text{TE}}^2 \approx \mathbb{E}[(w_S^{\text{ext}} - w^B)^\top r r^\top (w_S^{\text{ext}} - w^B)] = (w_S^{\text{ext}} - w^B)^\top \Sigma (w_S^{\text{ext}} - w^B)$$

where $\Sigma = \mathbb{E}[r r^\top] - \mathbb{E}[r]\mathbb{E}[r]^\top$ is the return covariance matrix (type: symmetric PSD bilinear form on $\mathbb{R}^{500}$).

Let $\delta w = w_S^{\text{ext}} - w^B \in \mathbb{R}^{500}$ (the **active weight vector**). Then:

$$\sigma_{\text{TE}}^2 = \delta w^\top \Sigma\, \delta w = \|\delta w\|_\Sigma^2$$

This is the squared norm in the **Mahalanobis metric** induced by $\Sigma$. Minimizing tracking error is equivalent to finding $\delta w$ of small norm in this covariance-weighted geometry, which penalizes deviations in directions of high variance (the principal components of $\Sigma$) more heavily.

The tracking error budget $\sigma_{\text{TE}} \leq \theta_2$ defines a **Mahalanobis ellipsoid** in weight space:

$$\mathcal{E}_{\theta_2} = \{\delta w \in \mathbb{R}^{500} : \delta w^\top \Sigma\, \delta w \leq \theta_2^2\}$$

Harvesting a lot shifts $w_S$ (and hence $\delta w$), potentially pushing outside $\mathcal{E}_{\theta_2}$ — this is the tracking error constraint the oracle's $\theta_2$ enforces.

---

## 8. Tax Alpha: Expected Value Formalization

Define the **tax alpha** generated by harvesting lot $(i,t)$ with unrealized loss $G_{i,t} = P_t - B_i < 0$:

$$\alpha_{\text{tax}}(x) = \tau(h) \cdot |G_{i,t}| \cdot \mathbb{1}[G_{\text{YTD}} > 0]$$

where $\tau(h)$ is the applicable marginal tax rate — a step function of holding period:

$$\tau(h) = \begin{cases} \tau_{\text{ST}} & h < 365 \text{ (short-term, ordinary income rate)} \\ \tau_{\text{LT}} & h \geq 365 \text{ (long-term, preferential rate)} \end{cases}$$

Note $\tau_{\text{ST}} > \tau_{\text{LT}}$ in all US tax brackets, so short-term losses are more valuable to harvest (in a world where you have short-term gains to offset).

The **net value** of a harvest decision accounts for the cost (tracking error induced) and benefit (tax deferral):

$$V(x) = \alpha_{\text{tax}}(x) - \lambda \cdot \sigma_{\text{TE}}(x)$$

where $\lambda > 0$ is a **Lagrange multiplier** on the tracking error constraint — the shadow price of the tracking error budget. The oracle threshold $\theta_1$ can be derived as the break-even condition $V(x) = 0$, solving for $\ell^*$ such that:

$$\tau(h) \cdot B \cdot |\ell^*| = \lambda \cdot \frac{\partial \sigma_{\text{TE}}}{\partial w}$$

The ML model's implicit goal (learning $\eta(x) \approx \mathbb{1}[V(x) > 0]$) is an approximation to this value-threshold rule, but without requiring the explicit derivation of $\lambda$ — the model learns the effective trade-off from labeled data.

---

## 9. The Full Learning Problem: ERM over an Oracle-Constrained Hypothesis Class

### 9.1 Restricted Hypothesis Class

Because the ML model is only trained on oracle-labeled data, it effectively operates on the **oracle-constrained hypothesis class**:

$$\mathcal{H}_\Omega := \{f \in \mathcal{H} : f(x) = 0\ \forall x \notin \Omega\}$$

The model cannot output $\hat{y} = 1$ on $\Omega^c$ (it has never seen positive labels there), creating an implicit constraint. More precisely, the empirical distribution $\hat{\mathcal{D}}_X$ places zero mass on $\Omega^c$ positive labels, so ERM has no gradient signal outside $\Omega$.

This is the rigorous statement of "the ML model cannot operate outside the oracle's boundary": the oracle partitions $\mathcal{X}$ into $\Omega$ (where $f^*$ can be $1$) and $\Omega^c$ (where $f^* = 0$ always), and the training distribution provides no information about whether $\theta_1$ is suboptimal.

### 9.2 Approximation–Estimation Decomposition

The excess risk of the learned classifier decomposes as:

$$\underbrace{R(\hat{f}) - R(f^{\text{Bayes}})}_{\text{excess risk}} = \underbrace{R(f^{\text{opt}}_\mathcal{H}) - R(f^{\text{Bayes}})}_{\text{approximation error (bias)}} + \underbrace{R(\hat{f}) - R(f^{\text{opt}}_\mathcal{H})}_{\text{estimation error (variance)}}$$

where $f^{\text{opt}}_\mathcal{H} = \arg\min_{f \in \mathcal{H}} R(f)$ is the best function in the hypothesis class.

- **Approximation error** is irreducible for fixed $\mathcal{H}$ — it measures how well the class can represent $f^{\text{Bayes}}$. Expanding $\mathcal{H}$ reduces this.
- **Estimation error** is controlled by $\mathfrak{R}_N(\mathcal{H})$ — it vanishes as $N \to \infty$ for sufficiently regular $\mathcal{H}$.
- The **oracle gap** $R(f^{\text{Bayes}}_{\text{oracle}}) - R(f^{\text{Bayes}}_{\text{true}})$ is an additional term absent from classical ERM analysis — it reflects the cost of using oracle labels instead of true optimal labels.

This three-way decomposition is the theoretical justification for the project's claim that "the ML model approximates the oracle rather than the true optimum": the irreducible oracle gap is a modeling constraint, not a failure of the learning algorithm.

---

## 10. Summary: The Theoretical Architecture

The project can be summarized as the following chain of approximations:

$$f^*_{\text{true}} \xrightarrow{\ \text{myopic approximation}\ } f^*_{\text{oracle}} \xrightarrow{\ \text{ERM over } \mathcal{H}\ } \hat{f}$$

with associated error decomposition:

$$R(\hat{f}) - R(f^*_{\text{true}}) = \underbrace{\left[R(f^*_{\text{oracle}}) - R(f^*_{\text{true}})\right]}_{\text{oracle gap (irreducible)}} + \underbrace{\left[R(f^{\text{opt}}_\mathcal{H}) - R(f^*_{\text{oracle}})\right]}_{\text{approximation error}} + \underbrace{\left[R(\hat{f}) - R(f^{\text{opt}}_\mathcal{H})\right]}_{\text{estimation error}}$$

The project's contribution is controlling the estimation error (via well-chosen $\mathcal{H}$ and $N$) while acknowledging the oracle gap is the binding constraint on practical tax alpha improvement. The geometric structure of the problem — oracle as boundary current, ML model as interior current, the Stokes relationship between them — precisely characterizes why no amount of model complexity can substitute for a better oracle, and why the oracle's threshold $\theta$ is a modeling assumption external to the learning problem itself.

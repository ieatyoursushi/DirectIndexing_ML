# ML Derivations — Mathematical Foundations

Per-algorithm derivations of the methods used in the ML pipeline.  Living document —
new sections added as more model types are added in v0.2+.

For high-level architecture and definitions of standard metrics (ROC-AUC, F1, etc.),
see [`MLPipeline.md`](MLPipeline.md).

---

## §1  Logistic Regression

### 1.1  Model

Logistic regression models the **log-odds** of the positive class as a linear function
of the features:

$$
\log\!\frac{P(y=1 \mid x)}{1 - P(y=1 \mid x)} = \beta_0 + \beta^\top x
$$

Equivalently, the probability is the **sigmoid** of the linear predictor:

$$
P(y=1 \mid x) = \sigma(\beta_0 + \beta^\top x), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

### 1.2  Likelihood

For a sample of $n$ i.i.d. observations $\{(x_i, y_i)\}_{i=1}^n$ with $y_i \in \{0, 1\}$,
the Bernoulli likelihood is:

$$
L(\beta) = \prod_{i=1}^{n} p_i^{y_i} (1 - p_i)^{1 - y_i}, \qquad p_i = \sigma(\beta_0 + \beta^\top x_i)
$$

The **negative log-likelihood** (NLL) is the loss we minimise:

$$
\mathcal{L}(\beta) = -\sum_{i=1}^{n} \bigl[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \bigr]
$$

### 1.3  Gradient

Using $\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$, the gradient with respect to
$\beta$ collapses to a clean form:

$$
\nabla_\beta\, \mathcal{L}(\beta) = \sum_{i=1}^{n} (p_i - y_i)\, x_i = X^\top (p - y)
$$

where $X \in \mathbb{R}^{n \times d}$ is the design matrix and $p = \sigma(X\beta)$.
The intercept follows the same form with $x_{i,0} = 1$.

### 1.4  Convexity & solver

$\mathcal{L}(\beta)$ is **convex** in $\beta$ (the Hessian $X^\top \text{diag}(p_i(1-p_i)) X$
is positive semi-definite).  Convexity guarantees a unique global minimum and means
any descent method converges.  scikit-learn's `lbfgs` solver uses a limited-memory
quasi-Newton method that approximates the Hessian implicitly — fast and well-suited
to the small-dimensional, large-sample regime we have ($d \approx 20$, $n \approx 10^5$).

### 1.5  L2 regularisation

The L2-penalised objective is:

$$
\mathcal{L}_{\text{L2}}(\beta) = -\sum_{i=1}^{n} \bigl[ y_i \log p_i + (1-y_i) \log(1-p_i) \bigr] + \frac{1}{2C} \|\beta\|_2^2
$$

where $C$ is scikit-learn's inverse-regularisation parameter (larger $C$ → weaker penalty).
The penalty shrinks $\|\beta\|_2$, reducing variance at the cost of a small bias.
This is **mandatory** when features are highly correlated (we saw |r| > 0.7 between
L, W, TaxAlpha in EDA) — without it, coefficients become unstable.

### 1.6  Class weighting

Under severe class imbalance, the unweighted NLL is dominated by easy negatives.
`class_weight='balanced'` reweights each example by the inverse class frequency:

$$
w_{i} = \frac{n}{2 \cdot n_{y_i}}, \qquad n_{y_i} = \#\{j : y_j = y_i\}
$$

The reweighted loss becomes:

$$
\mathcal{L}_{\text{bal}}(\beta) = -\sum_{i=1}^{n} w_i \bigl[ y_i \log p_i + (1-y_i) \log(1-p_i) \bigr] + \frac{1}{2C} \|\beta\|_2^2
$$

This is equivalent to upsampling the minority class until both classes contribute
equally to the loss, without actually duplicating data.

### 1.7  Predicted probabilities & threshold

For new $x$, the model outputs $\hat p(x) = \sigma(\hat\beta_0 + \hat\beta^\top x)$.
The default decision rule is $\hat y = \mathbf{1}[\hat p(x) \ge 0.5]$, but the optimal
threshold under class imbalance depends on the metric we care about (precision, recall,
F1).  After fitting we sweep thresholds across the PR curve and report the F1-optimal one.

---

## §2  Principal Component Analysis (PCA)

### 2.1  Setup

Let $X \in \mathbb{R}^{n \times d}$ be the centred (column-mean-zero), standardised
(column-std = 1) data matrix.  PCA finds an orthonormal basis $\{v_1, \ldots, v_d\}$
in feature space that diagonalises the sample covariance.

### 2.2  Objective

The first principal component is the unit vector $v_1$ maximising the projected variance:

$$
v_1 = \arg\max_{\|v\|=1}\; v^\top \hat\Sigma\, v, \qquad \hat\Sigma = \frac{1}{n-1} X^\top X
$$

Subsequent components $v_k$ maximise the same quantity subject to orthogonality with
$v_1, \ldots, v_{k-1}$.

### 2.3  Eigendecomposition

The solution is the eigendecomposition of $\hat\Sigma$:

$$
\hat\Sigma\, v_k = \lambda_k\, v_k, \qquad \lambda_1 \ge \lambda_2 \ge \ldots \ge \lambda_d \ge 0
$$

The $k$-th principal component is the eigenvector $v_k$; $\lambda_k$ is the variance
explained along that axis.

### 2.4  SVD equivalence

In practice scikit-learn computes PCA via the **singular value decomposition** of $X$:

$$
X = U\, S\, V^\top
$$

Then $V$ contains the principal components as columns (loadings) and
$\lambda_k = s_k^2 / (n - 1)$ for singular values $s_k$.  SVD is numerically more stable
than forming $\hat\Sigma$ explicitly when $d$ is moderate.

### 2.5  Variance explained

Component $k$ explains a fraction:

$$
\text{ratio}_k = \frac{\lambda_k}{\sum_{j=1}^{d} \lambda_j}
$$

of the total feature variance.  We keep the smallest $k^*$ such that
$\sum_{j=1}^{k^*} \text{ratio}_j \ge 0.95$ — the **95% cumulative variance** rule.

### 2.6  Why standardise first

Without standardisation, features with large numeric range (e.g. `G_YTD` in dollars,
~$10^5$) would dominate $\hat\Sigma$'s diagonal and the first PC would essentially
align with that single feature.  Standardisation (`StandardScaler`) puts all features
on unit variance so the loadings reflect genuine multivariate structure, not units.

---

## §3  K-means Clustering

### 3.1  Objective

Given $n$ points $\{x_i\}_{i=1}^n \subset \mathbb{R}^d$ and a target cluster count $k$,
K-means seeks a partition $\{C_1, \ldots, C_k\}$ of the points and centroids
$\{\mu_1, \ldots, \mu_k\}$ minimising the **within-cluster sum of squares** (WCSS):

$$
\min_{\{C_j, \mu_j\}}\; \sum_{j=1}^{k} \sum_{x \in C_j} \|x - \mu_j\|_2^2
$$

This is also called **inertia** in scikit-learn.

### 3.2  Lloyd's algorithm

K-means is solved approximately by alternating two steps:

1. **Assignment step**: each point joins the nearest centroid:
$$
C_j^{(t+1)} = \bigl\{x_i : j = \arg\min_{\ell}\, \|x_i - \mu_\ell^{(t)}\|_2\bigr\}
$$

2. **Update step**: each centroid moves to the mean of its assigned points:
$$
\mu_j^{(t+1)} = \frac{1}{|C_j^{(t+1)}|} \sum_{x \in C_j^{(t+1)}} x
$$

### 3.3  Convergence

The objective is **monotone non-increasing** under both steps (the assignment step
moves each point to a closer centroid, the update step minimises WCSS for fixed
assignments), and the configuration space is finite (number of possible partitions
is bounded).  Therefore Lloyd's algorithm converges in finite steps — to a **local**
minimum, not necessarily the global one.

### 3.4  K-means++ initialisation

Random initial centroids can converge to poor local minima.  K-means++ picks the first
centroid uniformly at random, then each subsequent centroid is sampled with probability
proportional to its squared distance to the closest already-chosen centroid.  This
biases towards a well-spread initialisation and substantially reduces variance across
runs.  `n_init=10` repeats the procedure 10 times and keeps the best result.

### 3.5  Selecting k

K-means requires $k$ as input.  Two complementary diagnostics:

#### Elbow method

Plot WCSS (inertia) against $k$.  WCSS decreases monotonically, but the rate of decrease
typically slows past the "true" $k$, producing an elbow.  Pick $k$ at the elbow.

#### Silhouette score

For each point $x_i$ in cluster $C_a$:

$$
s_i = \frac{b_i - a_i}{\max(a_i, b_i)}
$$

where
- $a_i$ = mean distance from $x_i$ to other points in its own cluster $C_a$
- $b_i = \min_{b \ne a}\; \text{mean distance from } x_i \text{ to points in } C_b$

$s_i \in [-1, 1]$; values near 1 mean the point is well-clustered, near 0 mean it's on
a boundary, negative means it's likely misassigned.  The overall silhouette is the mean
$s_i$ across all points.

We pick $k$ that **maximises silhouette** (subject to elbow visual sanity).

### 3.6  Application: lot substitution

We aggregate the 122K-row `lots.csv` to one row per **symbol** (~503 rows) by averaging
each ticker's asset-level features (R_t, SigmaRange, DeltaMA50, DeltaMA200) and keeping
its sector.  K-means on this stable per-symbol fingerprint groups stocks that move
similarly.

When the simulation v0.3 harvests a lot, instead of waiting 30 days (wash-sale), it
will look up the harvested ticker's cluster and buy a same-cluster substitute — keeping
the portfolio close to the benchmark on the dimensions that the cluster captures
(sector, daily-return scale, MA deviation profile).

---

## §4  Future Sections (v0.2+)

To be added as models are implemented:

- **Linear Discriminant Analysis (LDA)** — generative classification; class-conditional
  Gaussian with shared covariance.
- **Quadratic Discriminant Analysis (QDA)** — class-conditional Gaussian with separate covariances.
- **Elastic Net Logistic Regression** — L1 + L2 mixed penalty, sparse coefficients.
- **Decision Trees** — recursive partitioning, Gini / entropy criteria, pruning.
- **Random Forests** — bagging + feature subsampling, variance reduction.
- **Gradient Boosting** — additive forward-stagewise fitting of weak learners.
- **Support Vector Machines** — margin maximisation, dual formulation, kernel trick.

# Volatility Sub-Model v0.3 — σ as a state variable of the whole system

> **Status: DESIGN RECORD** for v0.3-6 … v0.3-10. Written *before* the code, per the v0.3 plan.
> Frozen once v0.3 exits, and superseded rather than edited after that. The live definitions are
> rows in [`../spec/SymbolTable.md`](../spec/SymbolTable.md): `cov_hat_pit`, `sigma_hat_*`,
> `qlike`, `z_barrier`, `fhs_*`.
>
> **Provenance of results.** GARCH, QLIKE, Ledoit–Wolf, the reflection principle and FHS are
> standard results, cited in §10. They are not derived in this repository. What *is* this
> repository's are:
> - the choice of objects;
> - the measurability contract;
> - the leakage analysis;
> - the evaluation protocol.

---

## 0. Why σ is structural, not "another feature"

Volatility enters the system in three places, and they have different jobs:

| Role | Where σ acts | What goes wrong without it | PR |
|---|---|---|---|
| **R0 — risk accounting** | $\sigma_{\mathrm{TE}}=\sqrt{252\,\delta w^\top\hat\Sigma_t\,\delta w}$ inside $U$ and the RL reward | $\hat\Sigma$ fit on the full history is a look-ahead (F1). Equal-name $\delta w$ ignores position size | v0.3-6 |
| **R1 — feature** | $\hat\sigma_{i,t}$, $\hat\sigma_{m,t}$, and the barrier coordinate $z$ in $x$ | the classifier must reconstruct "how likely is this lot to cross the loss trigger" from $L$ and the noisy one-day `SigmaRange` | v0.3-8 |
| **R2 — labels** | the σ path inside $\tilde y_{\mathrm{GBM}}$ | constant-σ Gaussian paths understate clustered drawdowns, so the soft label is biased in exactly the regimes that matter | v0.3-9 |
| **R3 — environment** | the synthetic world the RL agent trains in | GBM worlds have no volatility clustering, so the agent never meets a 2008 | v0.3-10 |

The same estimator object $\hat\sigma$ serves all four roles. The **effect** of each role is
measured as a separate ablation arm, because three simultaneous changes cannot be attributed.
The build is unconditional, since R0 is a correctness fix. Only the *materiality* of R1–R3 is
an empirical question.

**The Markov argument (why this matters for v0.4).** Under GARCH or EWMA, the next-period
variance $\sigma^2_{t+1}$ is a deterministic function of $\mathcal F_t$. So the pair
$(P_t,\sigma^2_{t+1})$ is a Markov state for the price process.

Under a latent stochastic-volatility model (e.g. Heston), $\sigma_t$ is *not*
$\mathcal F_t$-measurable, and the harvest problem becomes a POMDP. The agent must then act on a
filter $E[\sigma_t\mid\mathcal F_t]$, and EWMA is the cheapest such filter.

Either way, an RL state without a volatility summary is not Markov in the quantities that drive
the value of waiting. The value of waiting is the optimal-stopping continuation value, and it is
increasing in σ.

---

## 1. Objects and types

| Symbol | Type | Measurability | Meaning |
|---|---|---|---|
| $r_{i,t}=P_{i,t}/P_{i,t-1}-1$ | $\mathbb R$ (NaN if unpriced) | $\mathcal F_t$ | daily simple return of name $i$ |
| $r_{m,t}=\frac1{N_t}\sum_i r_{i,t}$ | $\mathbb R$ | $\mathcal F_t$ | equal-weight universe return (the benchmark's return) |
| $\hat\sigma^2_{i,t}$ | $\mathbb R_{>0}$, daily units | $\mathcal F_t$ | one-step-ahead forecast of $\mathrm{Var}(r_{i,t+1}\mid\mathcal F_t)$, built from $r_{i,\le t}$ |
| $\hat\sigma_{m,t}$ | $\mathbb R_{>0}$, annualized | $\mathcal F_t$ | the same, for $r_m$ |
| $\hat\Sigma_t$ | $\mathbb R^{N\times N}$, symmetric PSD, daily units | $\mathcal F_{t}$ (window ends at $t$) | point-in-time return covariance |
| $w^P_t,\ w^B_t$ | $\Delta^{N-1}$ (simplex) | $\mathcal F_t$ | portfolio dollar weights; benchmark weights (equal over priced names) |
| $\delta w_t=w^P_t-w^B_t$ | $\mathbb R^N$, $\mathbf 1^\top\delta w=0$ | $\mathcal F_t$ | active weights |
| $\sigma_{\mathrm{TE},t}=\sqrt{252\,\delta w_t^\top\hat\Sigma_t\delta w_t}$ | $\mathbb R_{\ge0}$, annualized | $\mathcal F_t$ | ex-ante tracking error |

**Measurability contract (the rule all code obeys):**
- a feature at row $(k,t)$ may use $r_{\cdot,s}$ only for $s\le t$, since the close $P_t$ is
  observed when the row is written;
- a *label* may use $s>t$, which is what makes it a label;
- a *fitted parameter* (GARCH $\theta$, LW intensity) used at $t$ may use only data in a window
  ending at or before $t$.

That third rule is the one that is easy to violate (§5).

---

## 2. Covariance $\hat\Sigma_t$ (v0.3-6, fixes F1)

**Current (legacy) estimator.** `TrackingErrorProxy.ComputeCovariance` builds a pairwise
available-case sample covariance over the **entire** loaded history. On day $t$ it therefore
contains $r_{t+1},\dots,r_T$: the future crash is in today's TE. It is kept as an arm
(`--cov=fullsample`) so the look-ahead's effect can be *measured*.

**Point-in-time sample.** $S_t=\frac1{L-1}\sum_{s=t-L+1}^{t}(r_s-\bar r)(r_s-\bar r)^\top$, with
$L=252$.

*Rank problem.* $\mathrm{rank}(S_t)\le L-1=251<N\approx409$, so $S_t$ is singular. The quadratic
form $\delta w^\top S_t\delta w$ is still well defined and $\ge0$. But its estimation noise is
large, because the smallest sample eigenvalues are biased toward zero and the largest away from
it. Marchenko–Pastur gives the scale of this bias: with $\varrho=N/L\approx1.6$, the null bulk
spans $\bar\sigma^2(1\pm\sqrt\varrho)^2$.

**Ledoit–Wolf (constant-correlation target).** The target and the shrunk estimator are

$$
F_{ij}=\bar\rho\sqrt{s_{ii}s_{jj}}\ (i\ne j),\quad F_{ii}=s_{ii},\qquad
\hat\Sigma_t=\rho^*F_t+(1-\rho^*)S_t,
$$

where $\bar\rho$ is the average sample correlation and $\rho^*\in[0,1]$ is the closed-form
optimal intensity $\rho^*=\max\{0,\min\{\hat\kappa/T,1\}\}$, with $\hat\kappa=(\hat\pi-\hat\psi)/\hat\gamma$:

- $\hat\pi=\sum_{ij}\hat\pi_{ij}$, where $\hat\pi_{ij}=\frac1T\sum_t\big[(x_{it}-\bar x_i)(x_{jt}-\bar x_j)-s_{ij}\big]^2$
  (the asymptotic variance of $\sqrt T s_{ij}$);
- $\hat\psi=\sum_i\hat\pi_{ii}+\sum_{i\ne j}\frac{\bar\rho}2\Big(\sqrt{\tfrac{s_{jj}}{s_{ii}}}\hat\vartheta_{ii,ij}+\sqrt{\tfrac{s_{ii}}{s_{jj}}}\hat\vartheta_{jj,ij}\Big)$,
  where $\hat\vartheta_{ii,ij}=\frac1T\sum_t\big[(x_{it}-\bar x_i)^2-s_{ii}\big]\big[(x_{it}-\bar x_i)(x_{jt}-\bar x_j)-s_{ij}\big]$;
- $\hat\gamma=\|F-S\|_F^2$ (the misspecification of the target).

$\hat\Sigma_t$ is PSD, because it is a convex combination of the PSD $S$ and the PSD $F$ (when
$\bar\rho\ge-1/(N-1)$). It is full rank whenever $\rho^*>0$.

**Engineering choices (recorded as such):**
- *Refit cadence:* every 21 trading days (monthly), held fixed in between. The cost is
  $O(N^2L)$ per refit.
- *Missing data:* a name with fewer than $L/2$ returns in the window keeps its diagonal and the
  target correlation $\bar\rho$. Within the window, missing returns are set to the name's window
  mean, which biases those $s_{ij}$ toward 0, recorded.
- *Warm-up:* before $L$ returns exist, use the expanding window from the first price (minimum 60
  days), then switch.

**Dollar active weights (Q2).** The legacy $\delta w$ is $1/n_{\text{open names}}-1/N$, which
treats every *name* held as equal regardless of how many dollars it holds. With contributions
and the trim, position sizes diverge, so
$w^P_{i,t}=\sum_{k\in A_i}q_kP_{i,t}/V_t$. The benchmark stays equal-weight over the priced
universe, and cap weights are an extension (SSGA weights in `constituents.json`).

**What v0.3-6 measures:**
- the σ_TE level and its shift;
- the *flip rate* of $f^*$ and of $\mathbf 1[U>0]$ vs `fullsample`, on the baseline run's rows (spectator);
- label deltas;
- temporal PR-AUC.

There is no gate: this is a correctness fix.

---

## 3. Per-name σ̂ (v0.3-7)

All three estimators expose one interface, `IVolEstimator`. Each produces $\hat\sigma^2_{i,t}$
(the forecast for $t+1$ from $r_{\le t}$) and a horizon-$h$ forecast of the variance sum
$V_{t,h}=\sum_{s=1}^hE_t[\sigma^2_{t+s}]$.

| Estimator | Recursion | Fitted parameters | Term structure $E_t[\sigma^2_{t+s}]$ |
|---|---|---|---|
| `Trailing21` (legacy) | sample variance of $r_{t-20..t}$ | none | flat |
| `Ewma(0.94)` (RiskMetrics) | $\hat\sigma^2_{t}=\lambda\hat\sigma^2_{t-1}+(1-\lambda)r_t^2$ | none ($\lambda$ fixed) | flat: IGARCH, $\alpha+\beta=1$ |
| `Garch11` | $\hat\sigma^2_{t}=\omega+\alpha r_t^2+\beta\hat\sigma^2_{t-1}$ | $(\omega,\alpha,\beta)$ by Gaussian QMLE | $\bar\sigma^2+(\alpha+\beta)^{s-1}(\hat\sigma^2_t-\bar\sigma^2)$, mean-reverting to $\bar\sigma^2=\omega/(1-\alpha-\beta)$ |

**Why EWMA first.** It has no fitted parameters, so it has **no leakage surface**, which is the
first hazard in §5. It is also the limiting case of GARCH, so a GARCH gain over EWMA is
attributable to mean reversion in the term structure, which matters at the 30-day label
horizon.

**GARCH discipline:**
- variance targeting ($\omega=\bar\sigma^2_{\text{window}}(1-\alpha-\beta)$) reduces the QMLE to
  two parameters;
- constraints: $\alpha,\beta\ge0$, $\alpha+\beta<1$;
- `Fit(window)` **rejects** any window that reaches past the forecast origin;
- refit on a cadence (quarterly) over an expanding or rolling window ending at the refit date;
- per-name fits with a pooled fallback when a name has fewer than 500 observations.

**Market σ̂.** $\hat\sigma_{m,t}$ is EWMA on $r_{m,t}$.

---

## 4. Evaluation: QLIKE (v0.3-7)

The loss is

$$
\mathrm{QLIKE}(\hat\sigma^2,\ \tilde\sigma^2)=\frac{\tilde\sigma^2}{\hat\sigma^2}-\ln\frac{\tilde\sigma^2}{\hat\sigma^2}-1\ \ \ge0,
$$

where $\tilde\sigma^2$ is a **noisy but conditionally unbiased** proxy of the realized variance
(here $r^2_{t+1}$, or $\frac1h\sum_{s\le h}r^2_{t+s}$ for horizon $h$).

QLIKE is in Patton's (2011) class of losses for which ranking forecasts by expected loss
against an unbiased proxy gives the **same ranking** as against the true variance. MSE on
$\sigma$ (not $\sigma^2$) is not in that class.

QLIKE is also the Gaussian negative log-likelihood up to constants. So "lower QLIKE" means
"better density forecast", which is what the barrier coordinate and FHS consume.

**Protocol (`vol-eval` → `data/artifacts-vol/qlike.json`):**
- estimators: constant (expanding mean), `Trailing21`, `Ewma`, `Garch11`, and on real data the
  Parkinson range estimator $\mathrm{SigmaRange}^2/(4\ln2)$;
- horizons $h\in\{1,5,21\}$;
- the loss is averaged over (name, day), overall and by regime (terciles of $\hat\sigma_{m}$);
- **pre-registered expectation:** EWMA and GARCH beat `Trailing21` overall, and the gap is
  largest in the high-vol tercile.

On GBM worlds σ is constant by construction, so the constant estimator should win, and that is
the control. On FHS worlds (§8) and real data, the clustering estimators should win.

---

## 5. Leakage hazards

1. **Fitted-on-all-years.** A GARCH fitted once on 2006–2026 encodes the 2008 and 2020
   variance levels into every 2007 forecast. The mitigation is the walk-forward `Fit(window)`
   only, with a test that a window reaching $t$ throws. This is the same signature discipline
   as `MedianImputer.Fit(trainingFold)`.
2. **Stacking.** σ̂ is an *estimated* input. If its parameters were tuned on the same rows the
   classifier is evaluated on, the classifier's test error would understate the deployed error.
   EWMA has nothing tuned. GARCH is fit on the past only, and never on label outcomes.
3. **Market vol as a date fingerprint.** $\hat\sigma_{m,t}$ is identical for every lot on day
   $t$, so it is a near-injective function of the date. Under a *random* split, train and test
   share days. A model can then learn day-level label rates through $\hat\sigma_m$, which is
   leakage through the date and not a vol effect.
   - Mitigation: every σ̂ ablation runs under `--split=temporal` (purged, embargo
     $\ge T_{\mathrm{fwd}}$; standing rule 5).
   - Metrics are also reported **stratified by $\hat\sigma_m$ tercile**, so a gain concentrated
     in one regime is visible as such.
4. **Look-ahead covariance** (F1, §2): fixed by construction in v0.3-6.
5. **Label/feature asymmetry.** FHS *labels* (§7) may use the future, but the σ̂ inside the FHS
   simulation must be the $\mathcal F_t$ forecast. The simulation *starts* from what is known at
   $t$; otherwise the label would encode realized future vol twice.

---

## 6. Role R1 — the barrier coordinate (v0.3-8)

The loss gate fires when $\ell\le-\theta_1$, i.e. when $P\le(1-\theta_1)p_k$. The log-distance to
the trigger is

$$
d_{k,t}=\Big(\ln\frac{P_t}{(1-\theta_1)\,p_k}\Big)^+\ \ \in\mathbb R_{\ge0},
$$

which is $0$ once the lot is already past the trigger.

For a driftless log-price $X_s=\sigma W_s$, the reflection principle gives

$$
\Pr\Big(\min_{0\le s\le h}X_s\le-d\Big)=2\,\Phi\!\Big(-\frac{d}{\sigma\sqrt h}\Big)=2\Phi(-z),\qquad
z=\frac{d}{\sigma\sqrt h}.
$$

With drift $\nu$ in the log price, the closed form is

$$
\Pr=\Phi\!\Big(\tfrac{-d-\nu h}{\sigma\sqrt h}\Big)+e^{-2\nu d/\sigma^2}\,\Phi\!\Big(\tfrac{-d+\nu h}{\sigma\sqrt h}\Big).
$$

We use the driftless form: over 30 days $\nu h$ is small next to $\sigma\sqrt h$, and the drift
is not reliably estimable per name.

**Columns (schema v6):**
- `SigmaHat` $=\sqrt{252}\hat\sigma_{i,t}$ (EWMA);
- `SigmaMkt` $=\sqrt{252}\hat\sigma_{m,t}$;
- `ZBarrier` $=z$, with $h=T_{\mathrm{fwd}}/252$ and $\sigma=$ `SigmaHat`, using the horizon
  variance $V_{t,h}$ when the estimator has a term structure;
- `PBarrier` $=2\Phi(-z)$.

**What it is and is not.** $2\Phi(-z)$ is the probability that the *loss gate alone* is crossed
within the label horizon, under a local GBM with the forecast σ. It ignores the wash, TE and
$U$ gates. It is therefore a **coordinate**, not a label, and in particular not $\tilde y$.

Its value is representational:
- $\tilde y_{\mathrm{GBM}}$ is, to first order, a monotone function of $z$ when the other gates
  are open;
- $z$ is a *nonlinear* function of $(L,\hat\sigma)$, which trees can approximate but a
  hyperplane cannot.

**Ablations (all under `--split=temporal`):**
- (a) GBT and logistic, each with vs without the four columns;
- (b) the **linear-tier representation test**: the share of the GBT–logistic soft-target gap
  that closes once $z$ is a coordinate (if most of it closes, the "nonlinearity" the trees were
  buying was mostly this one function);
- (c) PR-AUC and ROC per $\hat\sigma_m$ tercile, and calibration of $\hat\eta$ against realized
  $\tilde y_{\mathrm{BT}}$ inside vol buckets.

---

## 7. Role R2 — FHS soft labels (v0.3-9)

$\tilde y_{\mathrm{GBM}}$ simulates 200 constant-σ Gaussian paths. Filtered historical
simulation (Barone-Adesi et al. 1999) keeps the empirical shape of returns while letting σ
evolve:

1. Standardize the name's own history up to $t$: $\varepsilon_s=r_s/\hat\sigma_{s-1}$, $s\le t$.
   This is the empirical, fat-tailed, unit-variance innovation pool.
2. For each path $j$: $\sigma^2_{t+1}=\hat\sigma^2_t$ (the $\mathcal F_t$ forecast). For
   $s=1..T_{\mathrm{fwd}}$, draw $\varepsilon^*$ from the pool, set
   $r^*_{t+s}=\sigma_{t+s}\varepsilon^*$, and update $\sigma^2_{t+s+1}$ by the *same* recursion
   (EWMA or GARCH).
3. Evaluate the frozen-state oracle step $\varphi$ along the path, exactly as now.

The GBM path is kept as an arm: `--soft-gbm=gbm|fhs`.

**Measured:**
- the label shift;
- the bias $E[\tilde y_{\mathrm{model}}-\tilde y_{\mathrm{BT}}\mid\text{vol bucket}]$.

$\tilde y_{\mathrm{BT}}$ is a single realized path: noisy but unbiased for the true conditional
firing frequency. The model-based labels are low-variance, and FHS's bias in the high-vol bucket
should be smaller than GBM's. That bias reduction is the result to report.

---

## 8. Role R3 — the FHS world (v0.3-10)

`PriceLoader.FromFhs(real, days, seed)` is a synthetic universe for RL training:

- **Date-block bootstrap:** draw a whole historical date $\tau$ and use its *cross-sectional*
  standardized residual vector $(\varepsilon_{1,\tau},\dots,\varepsilon_{N,\tau})$. This
  preserves the contemporaneous correlation structure without estimating $\hat\Sigma$.
- **Rescale** along simulated per-name σ paths (the EWMA/GARCH recursion driven by the drawn
  residuals). This gives volatility clustering, while a residual bootstrap alone would give
  i.i.d. returns.
- Deterministic in the seed, and a sibling of `FromGbm`, selected by `simulate-mc --world=gbm|fhs`.

**Tests:**
- the ACF of $|r|$ at lags 1–5 is positive (vs ≈0 for GBM);
- excess kurtosis is $>0$;
- the mean pairwise correlation is within tolerance of the source's;
- same seed ⇒ same world.

**Train/eval separation for v0.4:** train on FHS worlds, evaluate walk-forward on real history.
They must never be the same history.

---

## 9. Defaults

| Choice | Default | Why |
|---|---|---|
| Σ̂ window $L$ | 252 d_trd | one year; matches annualization |
| Σ̂ refit | 21 d_trd | cost vs staleness |
| EWMA $\lambda$ | 0.94 | RiskMetrics daily; no fit |
| GARCH refit | 63 d_trd, expanding window, ≥500 obs | stability |
| Barrier $h$ | $T_{\mathrm{fwd}}/252$ | the label horizon |
| FHS paths | 200 | same as GBM labels |

---

## 10. References (standard background)

- O. Ledoit, M. Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix", *J. Portfolio
  Management* 30(4). Constant-correlation target and closed-form intensity.
- T. Bollerslev (1986), "Generalized Autoregressive Conditional Heteroskedasticity", *J.
  Econometrics* 31.
- J.P. Morgan/Reuters (1996), *RiskMetrics — Technical Document*, 4th ed. EWMA, λ = 0.94.
- A. Patton (2011), "Volatility forecast comparison using imperfect volatility proxies", *J.
  Econometrics* 160. The robustness of QLIKE/MSE-on-variance rankings to proxy noise.
- G. Barone-Adesi, K. Giannopoulos, L. Vosper (1999), "VaR without correlations for portfolios of
  derivative securities", *J. Futures Markets* 19. Filtered historical simulation.
- V. Marchenko, L. Pastur (1967): the eigenvalue law of sample covariance matrices.
- The reflection principle for Brownian motion: e.g. S. Shreve, *Stochastic Calculus for
  Finance II*, §3.7.

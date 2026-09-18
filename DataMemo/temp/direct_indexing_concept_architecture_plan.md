# Direct Indexing / Tax-Alpha Research Architecture

**Status:** Architecture context and implementation roadmap\
**Current major completed component:** supervised machine-learning layer
(oracle / oracle-propensity modeling)\
**Primary implementation language:** C# / .NET ecosystem, with
portfolio-state snapshots exported to ML-compatible data\
**Intended use of this document:** durable context for future agentic
models, implementation work, research design, and documentation. Treat
this as the current architectural plan; do not assume that reinforcement
learning, universe compression, or replacement optimization have already
been implemented.

------------------------------------------------------------------------

## 1. Executive Summary

This project is a modular direct-indexing research system designed to
maximize **after-tax portfolio value** while maintaining benchmark
exposure, respecting tax constraints, and controlling implementation
costs.

The project began as a graduate introductory ML project centered on a
hand-designed harvesting oracle and a supervised model that learns the
oracle's boundary / near-future firing propensity. That supervised layer
is now a major completed component. The larger project is no longer
only:

> "Can an ML model classify tax-loss harvesting opportunities?"

It is now:

> "Given an investable universe, tax constraints, tracking-error limits,
> replacement options, and dynamic market conditions, can a modular
> decision system improve after-tax outcomes relative to strong
> deterministic direct-indexing baselines?"

The architecture is intentionally **hybrid**:

-   empirical estimation and unsupervised learning discover market
    structure;
-   deterministic constraints encode legal, accounting, and portfolio
    feasibility;
-   supervised ML estimates uncertain per-lot opportunity quantities;
-   constrained optimization chooses feasible portfolios and
    replacements;
-   optional RL controls genuinely sequential, portfolio-level
    tradeoffs.

The core principle is:

$$
\boxed{
\text{Supervised models estimate the world; deterministic optimization enforces feasibility; RL decides long-horizon tradeoffs.}
}
$$

RL is therefore **not presumed to replace** the existing GBT. The
supervised GBT remains a valuable opportunity model, candidate ranker,
baseline, interpretability device, and possible input to an eventual
higher-level policy.

------------------------------------------------------------------------

## 2. Research Objective and Scope

### 2.1 End-to-end objective

The mature system should optimize a version of after-tax economic
utility:

$$
\max_{\pi}
\mathbb E_{\pi}
\left[
W_T^{\text{after-tax}}
-
\lambda_{\mathrm{TE}}\,\mathcal P_{\mathrm{TE}}
-
\lambda_{\mathrm{turn}}\,\mathrm{Turnover}
-
\lambda_{\mathrm{cost}}\,\mathrm{TradingCost}
-
\lambda_{\mathrm{risk}}\,\mathcal P_{\mathrm{risk}}
\right],
$$

where:

-   $W_T^{\text{after-tax}}$ is terminal or discounted after-tax wealth;
-   $\mathcal P_{\mathrm{TE}}$ is a tracking-error / exposure-deviation
    penalty;
-   turnover and trading costs penalize implementation burden;
-   additional penalties can represent concentration, infeasibility, or
    compliance risk.

The expected outcome is **not** unlimited "tax alpha." Tax-loss
harvesting is bounded by real economic constraints:

-   losses must actually occur;
-   wash-sale and related compliance constraints limit re-entry;
-   benchmark tracking limits the extent of portfolio deviation;
-   transaction costs and turnover can offset gross tax benefit;
-   the portfolio cannot continuously create tax losses without economic
    downside or exposure changes.

The project therefore seeks **persistent incremental improvement**, not
implausible multiples of a baseline direct-indexing tax benefit.

### 2.2 What counts as success

The main end-to-end comparison should not be PR-AUC alone. The key
economic metrics are:

$$
\begin{aligned}
&\text{after-tax terminal wealth},\\
&\text{discounted tax benefit / tax alpha},\\
&\text{tracking error and factor-exposure mismatch},\\
&\text{turnover and trading costs},\\
&\text{realized losses and gain offsets},\\
&\text{embedded unrealized gains},\\
&\text{replacement feasibility and compliance violations},\\
&\text{stability across market regimes}.
\end{aligned}
$$

The system should be judged against strong baselines such as:

1.  no direct indexing / benchmark fund;
2.  basic deterministic loss-threshold harvesting;
3.  deterministic oracle;
4.  GBT-based candidate ranking with fixed execution thresholds;
5.  GBT-based ranking plus constrained subset optimizer;
6.  optional RL-enabled high-level policy.

------------------------------------------------------------------------

## 3. Current Major Milestone: Supervised ML Layer

### 3.1 What is already built conceptually

The supervised ML layer uses per-lot snapshots of a joint lot,
portfolio, market, tax, and calendar state. The intended role is to
learn the geometry of the oracle's decision region and, more
importantly, to produce a **continuous opportunity / near-future firing
signal**.

The reported GBT performance is approximately **0.8--0.9 PR-AUC** under
the current evaluation configuration. This is encouraging for an
imbalanced opportunity-ranking problem, but it must not be interpreted
as "the model captures 80--90% of tax alpha."

PR-AUC measures ranking/classification performance with respect to a
label. It does **not** directly measure:

$$
\text{after-tax wealth}
\quad\text{or}\quad
\text{realized tax alpha}.
$$

### 3.2 State representation

Each ML observation corresponds to one lot $k$ at one timestep $t$:

$$
x_t^k = g(\mathcal S_t, P_t, k),
$$

where $\mathcal S_t$ is the portfolio / tax / market state and $P_t$ is
the portfolio.

The record is conceptually a frozen "photograph" of:

-   lot-level state;
-   portfolio and tax-ledger state;
-   asset / market features;
-   derived tax and utility quantities;
-   labels generated by the oracle or future-path construction.

Representative groups of features include:

$$
\begin{aligned}
\text{Lot state: }&
(\ell, h, s, B, W, K, q),\\
\text{Tax / portfolio state: }&
(G_{\mathrm{YTD}}, C_{\mathrm{loss}}, O, \sigma_{\mathrm{TE}}, \mathcal W),\\
\text{Market state: }&
(r_t, \sigma_{\mathrm{range}}, \Delta MA_{50}, \Delta MA_{200}),\\
\text{Derived state: }&
(\mathrm{TaxValue}, \mathrm{DaysToYE}).
\end{aligned}
$$

Where, informally:

-   $\ell$: normalized unrealized return;
-   $h$: holding period;
-   $s$: short-term / long-term tax flag;
-   $B$: basis;
-   $W$: lot weight;
-   $K$: number of open lots in ticker;
-   $G_{\mathrm{YTD}}$: realized net gains/losses year-to-date;
-   $C_{\mathrm{loss}}$: loss carryforward;
-   $O$: ordinary-income offset capacity;
-   $\sigma_{\mathrm{TE}}$: current tracking error;
-   $\mathcal W$: wash-sale / recent-harvest clock;
-   $\mathrm{TaxValue}$: immediate, capacity-aware tax-value estimate;
-   $\mathrm{DaysToYE}$: days remaining in tax year.

### 3.3 Labels

The schema supports several related labels. They have distinct purposes
and should not be conflated.

#### Hard oracle label

$$
Y_{\mathrm{Oracle}}(x_t^k)\in\{0,1\}.
$$

This records whether the acting oracle would harvest / approve lot $k$
at the current state.

It supports:

-   oracle imitation;
-   binary classification;
-   interpretable boundary analysis;
-   supervised baseline construction.

#### Historical backtest propensity label

$$
Y_{\mathrm{Soft,BT}}(x_t^k)
=
\frac{1}{H}
\sum_{j=1}^{H}
\mathbf 1
\left[
\text{oracle fires at } t+j
\right],
\qquad H \approx 30 \text{ trading days}.
$$

This is the key soft / propensity-style target.

It is generated using actual future historical prices, while the
relevant portfolio state is held frozen for the forward construction. It
represents a **near-future oracle firing fraction**:

> Given a snapshot of the state now, how frequently does the oracle
> enter its positive region over the next $H$ days on the realized
> historical path?

It should be interpreted as:

$$
\widehat p_{\mathrm{BT}}(x_t^k)
\approx
\mathbb E[
Y_{\mathrm{Soft,BT}}(x_t^k)
\mid x_t^k
].
$$

It is **not identical** to the action "harvest today." A high propensity
can indicate a persistent or likely opportunity, but the best current
action still depends on portfolio-wide constraints and the future
consequences of acting now.

#### GBM / simulated-path propensity label

$$
Y_{\mathrm{Soft,GBM}}(x_t^k)\in[0,1].
$$

This estimates the fraction of simulated forward price paths for which
the oracle fires within the same horizon. It is useful for controlled
simulations and comparisons between realized-path and model-path target
construction.

#### Tax-value target

$$
Y_{\mathrm{TaxValue}}\in\mathbb R_{\ge 0}.
$$

This is a continuous tax-value target. In the current design, it can
numerically equal the `TaxValue` feature. Therefore, any regression
using this target must exclude the corresponding feature to avoid direct
leakage.

#### Utility target

$$
Y_{\mathrm{Utility}}
=
\mathrm{TaxValue}
-
\lambda \sigma_{\mathrm{TE}}^2
-
c_{\mathrm{trade}}.
$$

This is the scalarized one-step objective before thresholding. It is
useful as a diagnostic, regression target, and bridge toward a
per-decision RL reward, but it is not itself a full sequential
objective.

#### Spectator gated label

A counterfactual / diagnostic label records what a legacy gated oracle
would have said on the same row even when a different acting oracle
produced the trajectory.

The important distinction is:

$$
\boxed{
\text{spectator label} \neq \text{acting policy}.
}
$$

The trajectory was generated by the acting oracle. The spectator label
is useful for comparing decision-boundary geometry but does not undo the
fact that actions alter later portfolio states.

### 3.4 Why the supervised layer matters even with future RL

The GBT has at least five durable roles:

1.  **Strong baseline:** it tests whether learned oracle propensity
    beats fixed thresholding or simple hand rules.
2.  **Candidate ranking:** it reduces a large holding universe to the
    few lots worth considering.
3.  **State estimate / learned sensor:** its scores can become inputs to
    a higher-level controller.
4.  **Interpretability:** it provides feature importance and opportunity
    geometry that can be audited.
5.  **Ablation anchor:** any RL policy must demonstrate value beyond
    this high-quality supervised baseline.

The correct question is not:

$$
\text{“Will RL mog the GBT?”}
$$

It is:

$$
\text{“Does sequential control create economic value beyond strong per-lot opportunity estimation?”}
$$

------------------------------------------------------------------------

## 4. Fundamental Conceptual Distinctions

### 4.1 Prediction versus decision versus control

The project includes three different task types.

#### Prediction

Estimate an uncertain future-relevant quantity:

$$
x_t^k
\rightarrow
\widehat p_{\mathrm{BT}}(x_t^k),
\quad
\widehat{\mathrm{TaxValue}}(x_t^k),
\quad
\widehat{\sigma}_{t:t+H}.
$$

Examples: GBT oracle propensity, future tax-value estimates, volatility
forecasts.

#### One-step constrained decision

Choose the best feasible action now:

$$
\max_{A_t}
\sum_{k\in A_t}
\widehat U_t^k
$$

subject to tracking-error, wash-sale, turnover, liquidity, and
concentration constraints.

This is usually best handled by deterministic constrained optimization,
not necessarily RL.

#### Sequential control

Choose actions that change the state distribution and therefore affect
later opportunities:

$$
a_t
\rightarrow
s_{t+1}
\rightarrow
a_{t+1}
\rightarrow \cdots
$$

Examples:

-   harvesting now changes basis;
-   a substitute changes future tracking error;
-   wash clocks block or enable later trades;
-   using TE budget today may reduce future flexibility;
-   tax-ledger state changes future tax value;
-   holding a substitute changes future harvestability.

This is where RL can be justified.

### 4.2 Information limit versus economic limit

There are two distinct ceilings.

#### Information limit

The state representation can only support decisions based on information
available at time $t$. Better features or estimators can help if they
change the optimal action.

#### Economic limit

No model can create unlimited harvestable losses or violate the cost of
benchmark fidelity. Opportunity is constrained by market dispersion,
taxes, trading costs, and tracking requirements.

Multiple improvements may help, but their gains can overlap. Do not
assume that benefits from a volatility model, GBT, replacement
optimization, and RL simply add together.

------------------------------------------------------------------------

## 5. Full Target Architecture

The intended mature system is:

$$
\boxed{
\text{Universe construction}
\rightarrow
\text{portfolio / state simulation}
\rightarrow
\text{supervised opportunity estimation}
\rightarrow
\text{candidate screening}
\rightarrow
\text{high-level policy}
\rightarrow
\text{replacement optimization}
\rightarrow
\text{portfolio feasibility / execution}
\rightarrow
\text{state transition and evaluation}.
}
$$

A practical modular decomposition follows.

  -----------------------------------------------------------------------
  Layer                   Main role               Recommended initial
                                                  type
  ----------------------- ----------------------- -----------------------
  0\. Data and accounting Prices, benchmark       Mechanistic system
                          holdings, lots, tax     
                          ledger, constraints     

  1\. Universe / core     Choose manageable       Empirical +
  basket                  active holdings and     deterministic
                          reserve universe        optimization

  2\. Risk representation Covariance,             Empirical /
                          PCA/factors, clusters,  unsupervised
                          exposures               

  3\. Per-lot opportunity Predict oracle          Supervised ML (GBT)
  estimation              propensity and/or       
                          future value            

  4\. Candidate screen    Reduce active decisions Deterministic
                          to feasible high-value  filtering + GBT ranking
                          lots                    

  5\. High-level control  Decide timing,          Constrained optimizer
                          selectivity, resource   first; optional RL
                          allocation              later

  6\. Replacement         Pick legal, low-damage  Mechanistic filter +
  selection               substitute conditional  deterministic optimizer
                          on harvest              

  7\. Execution and       Update holdings, lots,  Mechanistic simulation
  transition              basis, TE, wash clocks, / accounting
                          ledger                  

  8\. Evaluation          Walk-forward economic   Backtesting methodology
                          comparison and          
                          ablations               
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 6. Universe Compression and Core-Plus-Reserve Construction

### 6.1 Proposed hybrid universe

Let:

$$
\mathcal U_{\mathrm{full}}
=
\{\text{all benchmark constituents}\}.
$$

Choose an actively held core set:

$$
\mathcal C\subseteq\mathcal U_{\mathrm{full}},
\qquad
|\mathcal C|\approx 20\%-30\%\cdot|\mathcal U_{\mathrm{full}}|.
$$

For an approximately 500-name index, this suggests an initial core on
the order of 100--150 names, subject to empirical validation.

The remainder is a reserve pool:

$$
\mathcal R
=
\mathcal U_{\mathrm{full}}\setminus\mathcal C.
$$

The actual holdings are dynamic:

$$
\mathcal H_t
\subseteq
\mathcal C\cup\mathcal R.
$$

A core holding may be temporarily replaced by a reserve asset after a
harvest. Thus the system is not "only a 100-stock portfolio forever"; it
is a **managed core plus substitute reservoir**.

### 6.2 Correct target of the core basket

The core basket should target the **benchmark**, not "track the full
500-name direct-indexing implementation."

The objective is approximately:

$$
r_t^{\mathrm{portfolio}}
\approx
r_t^{\mathrm{benchmark}},
$$

while retaining enough cross-sectional dispersion and substitution
flexibility to make tax-loss harvesting viable.

A portfolio that tracks beta well but destroys most idiosyncratic
dispersion may be a poor DI design because it resembles a compressed
index fund with limited harvestable variation.

### 6.3 What PCA and clustering do

PCA, covariance estimation, factor modeling, and clustering are not
necessarily trading policies.

They estimate empirical market structure:

$$
R_t
\approx
F_t B^\top + \epsilon_t,
$$

or group names according to return behavior, factor exposures, sector
structure, beta, and residual characteristics.

Useful roles:

-   estimate covariance / risk structure;
-   create compact factor-state features for RL or optimization;
-   group economically similar names;
-   select representative core names;
-   construct replacement candidate pools;
-   estimate tracking-error consequences.

PCA is particularly suitable for factor/risk compression. Clustering is
often more natural for choosing representative holdings and candidate
substitutes.

### 6.4 Basket selection as deterministic optimization

Initial basket selection does not require another trained predictive
model. It can use empirical estimates inside a mechanistic optimizer:

$$
\min_{\mathcal C,w}
\quad
\underbrace{\mathrm{TE}(\mathcal C,w)}_{\text{benchmark fidelity}}
+
\lambda_{\mathrm{complexity}}|\mathcal C|
-
\eta\cdot\underbrace{\mathrm{DispersionPotential}(\mathcal C)}_{\text{TLH opportunity}}
$$

subject to:

$$
\begin{aligned}
&\sum_{i\in\mathcal C} w_i=1,\\
&w_i\ge 0,\\
&|\mathcal C|\le N,\\
&\text{sector / factor / market-cap constraints},\\
&\text{liquidity and concentration limits}.
\end{aligned}
$$

This is an empirically calibrated, deterministic portfolio-construction
problem:

$$
\boxed{
\text{data}
\rightarrow
\text{covariance / PCA / clusters}
\rightarrow
\text{constrained index-tracking optimizer}
\rightarrow
\text{core basket}.
}
$$

It should be run offline and updated on a slow schedule such as
benchmark reconstitution, quarterly review, or annual review---not
re-learned daily by RL.

------------------------------------------------------------------------

## 7. Replacement Selection

### 7.1 Purpose

When a currently held lot $i$ is harvested, the system must choose
whether and how to replace its exposure. The replacement problem is
conditional:

$$
\text{given harvest of } i,\quad \text{select } j.
$$

It is not the same as the initial core-basket problem and should
initially be a mechanistic / constrained optimization module.

### 7.2 Eligibility first

Define a candidate set:

$$
\mathcal R_i^{\mathrm{eligible}}
=
\left\{
j\in\mathcal U_{\mathrm{full}}:
\begin{array}{l}
j\text{ passes wash-sale / compliance logic},\\
j\text{ is liquid and tradable},\\
j\text{ is not restricted},\\
j\text{ does not cause unacceptable concentration},\\
j\text{ is available in the reserve pool or permitted substitute set}
\end{array}
\right\}.
$$

Eligibility must be handled by explicit rules and the tax / compliance
engine. Correlation alone is not a compliance rule.

### 7.3 Replacement scoring

For each feasible candidate $j$, estimate the portfolio consequence of
replacing $i$ with $j$. A baseline score may be:

$$
\mathrm{SubScore}(i,j;s_t)
=
-
\alpha\widehat{\Delta\mathrm{TE}}_{i\rightarrow j}
-
\beta\,\mathrm{FactorDistance}(i,j)
-
\gamma\,\mathrm{SectorMismatch}(i,j)
-
\delta\,\mathrm{LiquidityCost}(j)
-
\zeta\,\mathrm{ConcentrationPenalty}(j)
+
\xi\,\widehat{\mathrm{FutureOpportunity}}(j).
$$

Then select:

$$
j^*
=
\arg\max_{j\in\mathcal R_i^{\mathrm{eligible}}}
\mathrm{SubScore}(i,j;s_t).
$$

The first implementation can use deterministic empirical estimates:

-   rolling correlations;
-   covariance-based $\Delta\mathrm{TE}$;
-   PCA/factor loading distances;
-   sector and industry constraints;
-   beta and market-cap similarity;
-   liquidity and concentration penalties.

Optional supervised models can later improve individual estimates such
as:

$$
\widehat{\Delta\mathrm{TE}}(i,j\mid s_t),
\qquad
\widehat{\mathrm{FutureOpportunity}}(j\mid s_t),
\qquad
\widehat{\mathrm{ReplacementRegret}}(i,j\mid s_t).
$$

They are enhancements, not requirements.

### 7.4 Fallback logic

If there is no acceptable substitute, the system should follow
explicitly defined fallback behavior, such as an allowed exposure proxy,
temporary cash, or re-entry after the applicable wash-sale window. This
should be treated as an explicit policy choice with its own tracking and
tax consequences.

------------------------------------------------------------------------

## 8. The GBT's Operational Role in the Mature System

### 8.1 GBT scores currently held lots, not abstract reserve assets

Only assets currently held generate lots that can be harvested today.
Therefore, the GBT's daily operational domain is:

$$
\mathcal H_t
=
\{\text{currently held lots}\}.
$$

The reserve pool is not directly harvestable when not held. It becomes
relevant after a harvest decision, when the replacement module is
invoked.

A candidate set can be constructed as:

$$
\mathcal A_t^{\mathrm{candidate}}
=
\left\{
k\in\mathcal H_t:
\hat p_{\mathrm{BT}}(x_t^k)>\tau_{\mathrm{screen}},
\;
\ell_k<0,
\;
\text{eligible under feasibility rules}
\right\}.
$$

This means:

$$
|\mathcal A_t^{\mathrm{candidate}}|
\ll
|\mathcal H_t|
\ll
|\mathcal U_{\mathrm{full}}|.
$$

### 8.2 Label consistency after changing the portfolio construction

The soft label is portfolio-dependent:

$$
Y_{\mathrm{Soft,BT}}
=
f(
\text{lot state},
\text{tax state},
\text{TE state},
\text{wash state},
\text{future prices},
\text{acting oracle}
).
$$

Changing from a full 500-name environment to a sparse-core-plus-reserve
environment can change:

-   lot weights;
-   tracking error;
-   feasible replacement choices;
-   number and distribution of open lots;
-   tax-ledger trajectory;
-   available loss dispersion;
-   actual action feasibility.

Therefore, the correct long-run process is:

$$
\boxed{
\text{hybrid portfolio simulator}
\rightarrow
\text{generate hybrid-environment lot trajectories}
\rightarrow
\text{generate labels}
\rightarrow
\text{train / validate GBT}.
}
$$

Do not casually train only on full-universe labels and deploy in a
compressed hybrid portfolio without testing transfer. A full-index model
may still generalize, but that is an empirical hypothesis that needs
evaluation.

------------------------------------------------------------------------

## 9. Where RL Fits

### 9.1 RL is optional, not mandatory

RL should only be added if there is evidence that sequential effects are
economically material after strong deterministic and supervised
baselines are established.

A good reason to add RL is:

$$
a_t
\text{ materially changes }
s_{t+1}
\text{ in ways that alter later feasible and valuable actions}.
$$

Examples:

-   spending tracking-error budget today reduces later flexibility;
-   harvesting now changes future basis and harvestable losses;
-   substitute choices affect future exposure and loss opportunities;
-   tax-ledger usage changes the value of later losses;
-   wash-clock consequences make timing nontrivial.

If these effects are minor after a GBT-plus-constrained-optimizer
baseline, RL may not add enough value to justify its complexity.

### 9.2 RL as synthesizer, not replacement

The intended role is:

$$
\boxed{
\text{supervised models estimate local opportunity; RL synthesizes portfolio-wide, long-horizon tradeoffs.}
}
$$

The RL policy can receive:

$$
s_t^{\mathrm{RL}}
=
\Big(
\text{tax ledger},
\text{portfolio TE},
\text{market / PCA factors},
\text{candidate GBT scores},
\text{candidate tax values},
\text{sector exposures},
\text{replacement-feasibility summaries},
\text{turnover budget},
\text{wash-state summaries}
\Big).
$$

It should not initially choose independent binary actions over all S&P
constituents:

$$
a_t\in\{0,1\}^{500}
$$

is an unnecessarily large and unstable action space.

A more tractable high-level action is:

$$
a_t
=
(
\tau_t,
B_t^{\mathrm{TE}},
B_t^{\mathrm{turnover}},
m_t,
\text{harvest / defer regime}
),
$$

where the policy controls:

-   a dynamic GBT-score threshold $\tau_t$;
-   amount of TE budget to spend;
-   maximum turnover or number of trades;
-   number of top candidates considered;
-   aggressive versus conservative harvesting regime;
-   possible allocation across sectors / clusters.

The deterministic subset and replacement optimizers then execute the
best feasible lower-level decision.

### 9.3 Supervised predictions as RL inputs

This is a valid hybrid / model-augmented RL design:

$$
\text{supervised specialists}
\rightarrow
\text{forecast features}
\rightarrow
\text{RL controller}
\rightarrow
\text{portfolio action}.
$$

Examples of supervised inputs:

$$
\hat p_{\mathrm{BT}}(x_t^k),
\qquad
\widehat{\mathrm{TaxValue}}(x_t^k),
\qquad
\widehat{\sigma}_{t:t+H},
\qquad
\widehat{\Delta\mathrm{TE}}(i,j).
$$

However, only include predictions that add incremental decision
information. A model output that is simply a noisy duplicate of
`TaxValue` may not improve the policy.

### 9.4 Strict anti-leakage rule

If supervised predictions are fed into RL or a subsequent optimizer,
they must be produced in a temporally valid way:

$$
\text{train predictors on historical past}
\rightarrow
\text{predict future block}
\rightarrow
\text{feed predictions into policy}.
$$

Do not train a predictor on the full backtest period, provide its
in-sample scores to RL, and claim out-of-sample sequential performance.

------------------------------------------------------------------------

## 10. Deterministic, Empirical, Supervised, and RL Components

The system should not force every subproblem into ML.

### 10.1 Empirical / unsupervised estimates

These learn or estimate market structure from historical data:

-   covariance matrices;
-   PCA factors and loadings;
-   correlation clusters;
-   factor exposures;
-   rolling beta;
-   liquidity and volatility statistics.

They are data-driven, but not necessarily supervised predictors.

### 10.2 Mechanistic / deterministic functions

These encode known structure, accounting identities, constraints, or
explicit objectives:

-   tax ledger;
-   lot accounting and basis updates;
-   wash-sale clock;
-   tax-rate / offset-capacity calculations;
-   feasibility filtering;
-   tracking-error calculation;
-   basket optimizer;
-   substitute optimizer;
-   turnover and concentration constraints.

These can use empirical estimated inputs while remaining deterministic.

### 10.3 Supervised ML

Use supervised ML for uncertain quantities that cannot be adequately
computed from a direct formula:

-   oracle firing propensity;
-   future tax opportunity;
-   future volatility / opportunity state;
-   predicted TE consequence if a simple covariance model is
    insufficient;
-   replacement regret or future harvestability.

### 10.4 RL

Use RL only for decisions where action timing changes the later state in
a meaningful way:

-   how aggressively to harvest today;
-   whether to defer in anticipation of better future opportunity;
-   how to allocate TE / turnover budgets over time;
-   how to balance immediate tax value against future optionality.

------------------------------------------------------------------------

## 11. Recommended Development Roadmap

### Phase A --- Preserve and validate the supervised milestone

1.  Freeze and document the current `LotStateVector` and label
    semantics.
2.  Record exact training/test split methods, feature exclusions, and
    target definitions.
3.  Validate GBT under strict time-series methodology.
4.  Keep the GBT as a first-class baseline and deployable candidate
    scorer.
5.  Interpret the reported PR-AUC only in relation to its explicit
    binary / ranking setup.

**Critical validation note:** labels with a 30-day forward horizon
overlap heavily across adjacent dates. Random row splits can leak future
context. Use chronological splits and a purge/embargo interval at least
as long as the forward-label horizon around train/test boundaries.

### Phase B --- Add deterministic universe and replacement layers

1.  Implement covariance / factor / clustering estimation.
2.  Define and solve an offline cardinality-constrained
    benchmark-tracking basket problem.
3.  Construct the core-plus-reserve universe.
4.  Implement replacement eligibility logic.
5.  Implement deterministic substitute scoring and whole-portfolio
    feasibility checks.
6.  Re-run the simulator in this hybrid environment.
7.  Regenerate labels and retrain / test the GBT in the environment that
    will actually be used.

### Phase C --- Add a constrained non-RL execution baseline

Before RL, use:

$$
\text{GBT candidate scores}
+
\text{deterministic subset optimizer}
+
\text{replacement optimizer}.
$$

This answers whether portfolio-wide feasibility and cross-lot
interactions alone capture most of the achievable benefit.

### Phase D --- Decide whether RL is warranted

Only proceed if the constrained baseline leaves economically meaningful
sequential timing value unresolved.

RL should initially be high-level and low-dimensional:

$$
\text{choose thresholds, budgets, and aggressiveness}
$$

rather than:

$$
\text{choose every lot trade directly}.
$$

### Phase E --- End-to-end comparative research

Run walk-forward comparisons among:

1.  benchmark / no DI;
2.  basic deterministic harvesting;
3.  oracle;
4.  GBT with fixed rules;
5.  GBT + deterministic constrained optimizer;
6.  GBT + policy/RL + replacement optimizer.

Report economic performance, implementation cost, TE, and
robustness---not only classification metrics.

------------------------------------------------------------------------

## 12. Key Evaluation and Research Questions

### 12.1 Supervised-learning questions

-   How well can observable state predict hard oracle action?
-   How well can it predict forward firing propensity?
-   Which feature groups contribute unique signal?
-   Does the model generalize across regimes, sectors, portfolio sizes,
    and core/reserve constructions?
-   Does higher propensity translate into useful ranked execution
    candidates?

### 12.2 Portfolio-construction questions

-   How small can the core basket be before TE or tax opportunity
    degrades unacceptably?
-   Does a 20--30% core plus reserve pool outperform a full-universe
    implementation after complexity and costs?
-   Does compression reduce TLH opportunity by eliminating too much
    idiosyncratic dispersion?
-   Which empirical universe-construction method produces the best net
    outcome?

### 12.3 Replacement questions

-   Does factor/correlation/sector-based replacement maintain benchmark
    fidelity sufficiently?
-   Does replacement selection preserve future harvestability?
-   What is the tradeoff between immediate TE damage and future tax
    opportunity?
-   How frequently is no good substitute available under the constraint
    system?

### 12.4 RL / sequential questions

-   Does a high-level sequential policy outperform GBT + deterministic
    constrained execution?
-   Does it learn to defer only when later opportunity actually
    compensates for missed current tax value?
-   Does it improve tax alpha per unit of TE or turnover?
-   Does it remain stable across market regimes?
-   Does it add value beyond what the deterministic oracle already
    encodes?

------------------------------------------------------------------------

## 13. Guardrails and Non-Negotiable Design Rules

1.  **Do not equate PR-AUC with tax alpha.**\
    Supervised ranking performance is a diagnostic; economic evaluation
    is separate.

2.  **Do not assume RL is automatically better.**\
    A strong oracle and GBT may be difficult to improve upon. RL must
    earn its complexity through incremental economic value.

3.  **Do not let every component become an ML model.**\
    Known constraints and transparent portfolio objectives belong in
    deterministic functions or optimization.

4.  **Do not train in one portfolio environment and silently deploy in
    another.**\
    Labels depend partly on portfolio state. Regenerate / validate after
    material changes in universe construction.

5.  **Do not expose future information through labels or model
    features.**\
    Use walk-forward training, purged splits, and out-of-fold
    predictions where predictions feed later components.

6.  **Do not make the first RL action space per-stock binary across the
    full universe.**\
    Use GBT screening, aggregation, budgets, and constrained lower-level
    execution.

7.  **Keep tax/compliance assumptions explicit and configurable.**\
    Tax and wash-sale treatment belong in an auditable domain module,
    not in untracked model heuristics.

8.  **Evaluate net outcomes, not gross harvested losses alone.**\
    Include tax timing, realized gains, trading costs, TE, and future
    optionality.

------------------------------------------------------------------------

## 14. Current Working Blueprint

$$
\boxed{
\begin{aligned}
&\textbf{Input data + tax/accounting engine}\\
&\downarrow\\
&\textbf{Empirical risk structure}
\quad
(\hat\Sigma,\ \text{PCA factors},\ \text{clusters},\ \text{exposures})\\
&\downarrow\\
&\textbf{Offline core-basket optimizer}
\quad
(\mathcal C,\ w_{\mathcal C})\\
&\downarrow\\
&\textbf{Dynamic portfolio state / lot simulator}\\
&\downarrow\\
&\textbf{Supervised GBT opportunity model}
\quad
\widehat Y_{\mathrm{Soft,BT}},\
\widehat U,\
\widehat{\text{future value}}\\
&\downarrow\\
&\textbf{Candidate screen}
\quad
(\text{held lots only; feasibility filters})\\
&\downarrow\\
&\textbf{High-level policy}
\quad
(\text{deterministic first; RL only if justified})\\
&\downarrow\\
&\textbf{Replacement eligibility + substitute optimizer}\\
&\downarrow\\
&\textbf{Portfolio feasibility, execution, and state transition}\\
&\downarrow\\
&\textbf{Walk-forward after-tax evaluation and ablations}.
\end{aligned}
}
$$

------------------------------------------------------------------------

## 15. Final Architectural Position

The supervised GBT layer is **not a disposable precursor to RL**. It is
a major finished component and should persist as:

$$
\boxed{
\text{a learned per-lot opportunity estimator, candidate generator, interpretable baseline, and policy input.}
}
$$

PCA/clustering and basket construction are best viewed as:

$$
\boxed{
\text{empirical market-structure estimation + deterministic portfolio optimization.}
}
$$

Replacement selection is best viewed as:

$$
\boxed{
\text{mechanistic eligibility constraints + empirically informed deterministic substitution optimization.}
}
$$

RL is a later, conditional extension whose correct role is:

$$
\boxed{
\text{to synthesize forecasts and portfolio state into long-horizon timing / resource-allocation decisions.}
}
$$

The most defensible development order is:

$$
\boxed{
\text{supervised GBT}
\rightarrow
\text{deterministic core/reserve construction}
\rightarrow
\text{constrained execution and replacement}
\rightarrow
\text{evaluate sequential gap}
\rightarrow
\text{optional high-level RL}.
}
$$

This order preserves interpretability, avoids unnecessary action-space
explosion, makes every contribution testable through ablations, and
keeps the project grounded in the true economic objective: better
after-tax outcomes under realistic direct-indexing constraints. (essentailly general question prompts around the goals of the project being an ML-based DI system)


Here is the converted text with inline math (`$...$`) and display math (`$$...$$`) formatting:

---

## Appendix: Design Questions, Hypotheses, and Evolving Project Intent (essentially what I prompted into GPT 5.5 Pro)

This section records the motivating questions and design intuitions that produced the current architecture. It is included so future agents and contributors understand not only the selected design, but also the unresolved tradeoffs and the reasoning that led to it.

### Original economic and product motivation

* The project began with a practical direct-indexing question: for a high-net-worth taxable portfolio, how much of the long-run capital-gains burden can realistically be reduced or deferred through direct indexing and tax-loss harvesting?

* A representative thought experiment was a portfolio growing from approximately:

  $$
  \$10\text{M} \rightarrow \$75\text{M}
  $$

  over roughly 20 years, with direct-indexing tax benefits compared against ordinary benchmark exposure.

* The motivating question was whether a better harvesting policy could create meaningfully more tax alpha than generic direct-indexing rules, even if the incremental improvement appeared modest relative to total portfolio value.

* The working intuition is that a reduction such as:

  $$
  20\%\text{ effective tax rate} \rightarrow 17\%\text{ effective tax rate}
  $$

  may sound like only three percentage points, but is economically equivalent to reducing the tax bill itself by:

  $$
  \frac{3}{20} = 15\%.
  $$

* The project should not assume that harvested losses are a permanent dollar-for-dollar reduction in final liquidation taxes. Tax-loss harvesting often creates value through timing, deferral, offset capacity, and the compounding value of capital that remains invested.

* The project is motivated partly by a possible future solo-RIA / high-net-worth-client application. The intended client-facing value proposition is not "the model produces a huge magic tax alpha," but rather:

  > Evidence-based, personalized direct indexing that seeks to improve after-tax outcomes while preserving benchmark exposure and controlling implementation costs.

### Initial supervised-learning hypothesis

* The original major ML hypothesis was that a supervised classifier or regressor could improve harvesting decisions beyond simple generic threshold rules.

* Rather than labeling a lot only as:

  $$
  \text{loss} < -x\% \Rightarrow \text{harvest},
  $$

  the project constructs a richer lot-level state containing tax, portfolio, market, calendar, and wash-sale information.

* The central supervised target evolved into an oracle-propensity label:

  $$
  Y_{\mathrm{Soft,BT}} = \frac{\text{number of future days oracle fires}}{\text{future-label horizon}},
  $$

  with a typical future horizon of approximately 30 trading days.

* The intended interpretation is not simply "will the lot lose money?" It is closer to:

  > Given the current frozen lot and portfolio state, how frequently does the oracle regard this lot as harvest-worthy over the near future historical path?

* The user specifically questioned whether this type of propensity target makes the supervised model closer to policy learning than ordinary binary classification. The current answer is: it is still supervised prediction, but it produces a richer opportunity/timing score that can support later policy decisions.

* The GBT is currently the major finished component. Its reported PR-AUC is approximately:

  $$
  0.8\text{–}0.9.
  $$

* This is understood as promising oracle-ranking performance, not as a direct estimate that the system captures 80–90% of total tax alpha.

### Questions about the oracle's ceiling

* A key concern is whether the current deterministic oracle already contains most of the economically relevant logic, making RL redundant.

* The oracle currently incorporates or is intended to incorporate elements such as:

  * immediate tax value;
  * realized gains and losses year-to-date;
  * loss carryforwards;
  * ordinary-income offset capacity;
  * holding period and tax treatment;
  * tracking-error penalties;
  * trading costs;
  * wash-sale timing;
  * portfolio constraints.

* The scalarized oracle is conceptually similar to:

  $$
  U(x) = \mathrm{TaxValue}(x) - \lambda\sigma_{\mathrm{TE}}^2 - c_{\mathrm{trade}}.
  $$

* The user's core uncertainty is whether a policy that already uses this type of utility can leave enough unmodeled sequential value for RL to improve upon.

* The current architectural position is that RL should not be treated as automatically superior. Its value must be demonstrated relative to the deterministic oracle and the supervised GBT baseline.

### Volatility and narrow predictive submodels

* The project considered whether a better volatility-estimation model might be a meaningful standalone source of improvement.

* The concern is that a volatility model can improve forecast RMSE without changing many actual harvest decisions.

* The relevant causal chain is:

  $$
  \text{better volatility estimate} \rightarrow \text{different harvest decision} \rightarrow \text{different portfolio trajectory} \rightarrow \text{better after-tax outcome}.
  $$

* A volatility model is therefore useful only to the extent that it changes decisions in economically important marginal cases.

* The user's broader insight was that multiple specialized models may each contribute a different "axis of information," including:

  * volatility / uncertainty;
  * future oracle propensity;
  * tax value;
  * tracking-error effects;
  * replacement quality;
  * future harvestability.

* These models should not be assumed to create additive gains automatically. Their information may overlap, and their incremental value should be tested through ablations.

### GBT versus RL

* The user asked whether an eventual RL policy would necessarily outperform or replace the supervised GBT.

* The intended answer encoded in the architecture is no.

* The GBT is trained to estimate oracle-related opportunity:

  $$
  x_t^k \rightarrow \widehat{Y}_{\mathrm{Soft,BT}}(x_t^k),
  $$

  while RL would seek to optimize a cumulative portfolio objective:

  $$
  \max_{\pi} \mathbb E_{\pi} \left[ \sum_t \gamma^t r_t \right].
  $$

* The GBT asks a local predictive question:

  > Which currently held lots appear likely to become or remain oracle-worthy soon?

* RL asks a sequential control question:

  > Given the entire portfolio state and future consequences of acting now, how aggressively should the portfolio harvest, defer, substitute, or preserve flexibility?

* The user's desired eventual architecture is therefore not:

  $$
  \text{GBT versus RL},
  $$

  but:

  $$
  \text{GBT and specialized predictors} \rightarrow \text{RL / high-level controller} \rightarrow \text{portfolio action}.
  $$

### RL as a knowledge synthesizer

* The user identified the most promising role for RL as a synthesizer of narrower supervised models.

* Under this view, specialized models estimate uncertain local quantities:

  $$
  \widehat p_{\mathrm{oracle}}, \quad \widehat{\mathrm{TaxValue}}, \quad \widehat\sigma, \quad \widehat{\Delta\mathrm{TE}}, \quad \widehat{\mathrm{FutureOpportunity}}.
  $$

* A higher-level policy can then combine those forecasts with the portfolio state to decide:

  * whether to harvest now or defer;
  * how much tracking-error budget to use;
  * how much turnover to allow;
  * how many GBT-ranked candidates to act on;
  * whether current opportunity is worth sacrificing future optionality;
  * how aggressively to pursue harvesting under the current tax and market regime.

* The intended framing is:

  $$
  \boxed{
  \text{specialist supervised models estimate the world;} \quad \text{RL decides what to do over time.}
  }
  $$

### Dimensionality reduction and reduced direct-indexing universes

* The user considered whether a direct-indexing implementation could actively hold only approximately:

  $$
  20\%\text{–}30\%
  $$

  of the benchmark's constituent count while preserving approximately:

  $$
  95\%\text{–}99\%
  $$

  of benchmark-tracking quality.

* The initial idea was that PCA, clustering, or similar methods could select a compact active basket, while the remaining index constituents remain available as potential replacement candidates after harvesting.

* The desired structure is:

  $$
  \mathcal U_{\mathrm{full}} = \mathcal C \cup \mathcal R,
  $$

  where:

  * $\mathcal C$ is the active core basket;
  * $\mathcal R$ is the reserve / substitution pool.

* The core basket should be optimized to track the benchmark, not merely imitate a hypothetical full direct-indexing portfolio.

* The non-core universe is not discarded. It remains valuable as optionality for replacement selection.

* The user's concern is that reducing the active universe could lower RL action-space complexity while still preserving enough benchmark fidelity and tax-loss opportunity.

### Relationship between the GBT and a compressed portfolio

* The user asked whether the GBT can remain trained on the entire benchmark universe even if RL or the active portfolio focuses only on a 20–30% core basket.

* The current intended answer is nuanced:

  * the GBT can score all **currently held lots** that arise in the hybrid portfolio;
  * reserve assets not currently held are not harvest candidates, but may be evaluated by the replacement module after a harvest decision;
  * the final GBT should ideally be trained or at least revalidated on trajectories generated by the actual core-plus-reserve environment.

* This matters because:

  $$
  Y_{\mathrm{Soft,BT}}
  $$

  depends partly on portfolio state, including tracking error, lot weights, replacement feasibility, tax ledger path, and wash-sale behavior.

* Therefore, a full-index-trained model may transfer to a compressed environment, but that transfer must be treated as an empirical hypothesis rather than assumed.

### Initial basket selection

* The user asked what type of component should choose the initial 20–30% basket.

* The intended design is not a daily RL policy.

* The initial basket is an offline portfolio-construction problem using empirical structure plus explicit constraints:

  $$
  \text{historical returns} \rightarrow \text{covariance / PCA / clustering / factor estimates} \rightarrow \text{deterministic constrained optimizer} \rightarrow \text{core basket}.
  $$

* Relevant objectives include:

  * low tracking error;
  * representative factor and sector exposure;
  * liquidity;
  * concentration limits;
  * low implementation complexity;
  * preservation of enough cross-sectional dispersion for tax-loss harvesting.

* The user specifically questioned whether this must be another trained ML model. The intended answer is no: historical data can provide empirical estimates while the final selection remains a deterministic optimization problem.

### Replacement selection after harvest

* The user asked what chooses the replacement when a held stock is harvested.

* The desired replacement process is separate from initial basket construction.

* It should begin with mechanistic eligibility filtering:

  * tax / wash-sale logic;
  * liquidity;
  * availability;
  * concentration constraints;
  * sector and exposure restrictions;
  * substitute-pool availability.

* Among eligible candidates, the replacement is selected through a deterministic score or constrained optimization based on empirical estimates such as:

  $$
  \widehat{\Delta\mathrm{TE}}_{i\rightarrow j}, \quad \text{factor distance}, \quad \text{correlation}, \quad \text{beta similarity}, \quad \text{sector similarity}, \quad \text{liquidity cost}, \quad \widehat{\text{future harvestability}}.
  $$

* The conceptual selection rule is:

  $$
  j^* = \arg\max_{j\in\mathcal R_i^{\mathrm{eligible}}} \mathrm{SubScore}(i,j;s_t).
  $$

* The fallback when no suitable substitute exists may involve an explicitly configured temporary exposure proxy, cash, or later repurchase after the relevant wash-sale window.

* The user does not currently intend to make replacement selection another autonomous RL agent. That would create unnecessary interacting-policy complexity and obscure attribution of performance.

### Deterministic versus trained components

* A recurring user question was whether basket selection and replacement selection need supervised or unsupervised trained models.

* The intended final answer is:

  $$
  \boxed{
  \text{They can be fully deterministic functions after empirical quantities are estimated.}
  }
  $$

* The broader design principle is:

  $$
  \text{data} \rightarrow \text{empirical estimates} \rightarrow \text{mechanistic constraints / objective} \rightarrow \text{deterministic optimizer} \rightarrow \text{action}.
  $$

* Examples of empirical estimates:

  * covariance;
  * PCA loadings;
  * factor exposure;
  * correlation;
  * beta;
  * volatility;
  * liquidity statistics;
  * historical tracking-error consequences.

* Examples of deterministic components:

  * tax ledger;
  * basis tracking;
  * wash-sale clock;
  * eligibility filtering;
  * tracking-error calculation;
  * basket optimization;
  * substitute optimization;
  * turnover and concentration enforcement.

* Supervised models should be reserved for uncertain forward-looking quantities that cannot be cleanly computed from known formulas.

### Current intended hierarchy of importance

* The current supervised GBT layer is the major completed research/engineering milestone.

* In a mature end-to-end tax-alpha system, the highest-level quantitative objective would likely be portfolio-level after-tax utility, not PR-AUC.

* The intended hierarchy is:

  $$
  \text{portfolio construction} \rightarrow \text{state and opportunity estimation} \rightarrow \text{constrained portfolio decision} \rightarrow \text{after-tax evaluation}.
  $$

* In a future version where RL is justified, the RL policy is the main sequential controller, but it does not erase the importance of the supervised GBT or deterministic portfolio-construction layers.

* The preferred implementation sequence remains:

  $$
  \boxed{
  \text{supervised GBT} \rightarrow \text{core-plus-reserve construction} \rightarrow \text{deterministic constrained execution} \rightarrow \text{measure remaining sequential gap} \rightarrow \text{optional high-level RL}.
  }
  $$

### Open questions to preserve for future work

* How much tax alpha remains after a strong GBT-plus-constrained-optimizer baseline?

* Does a compact 20–30% core materially reduce tax-loss opportunity relative to a much broader direct-indexing universe?

* What active-basket size gives the best tradeoff among tracking error, complexity, turnover, and harvestable dispersion?

* Does the full-universe GBT transfer well to the hybrid core-plus-reserve environment, or must it be retrained from hybrid trajectories?

* Which replacement features best predict realized tracking and future harvestability?

* Is a volatility model useful after accounting for the GBT's current features, or does it mainly duplicate existing information?

* Does RL discover economically useful timing behavior beyond a deterministic utility-based oracle and constrained subset optimizer?

* Are the gains robust after strict walk-forward validation, purged temporal splits, transaction costs, realistic tax treatment, and market-regime variation?

* Does the system's complexity create enough incremental net benefit to justify the added operational and model-risk burden?

This appendix should be read as a record of design intent and active research questions, not as evidence that all listed components are implemented or empirically validated.
# Policy Layer v0.4 — the harvest problem as sequential control

> **Status: DESIGN RECORD** for v0.4a (the constrained-execution baseline) and v0.4b (RL). It
> was written at the end of v0.3, once the state it needs exists. Frozen once v0.4b starts, and
> superseded rather than edited after that.
>
> The live, typed rows are in [`../spec/SymbolTable.md`](../spec/SymbolTable.md) §I
> (`mdp_state`, `mdp_action`, `mdp_transition`, `reward`, `return`, `fed_predictions`). This
> record holds the **reasoning** behind those rows, the seams to build, the tests to write, and
> the deltas against the archived architecture thread
> ([`../archive/architecture_thread/`](../archive/architecture_thread/direct_indexing_concept_architecture_plan_contextualized.md)).

---

## 0. What is being decided

v0.3 built the two halves of the state:
- the **Schedule D ledger** answers "what is an opportunity worth";
- the **volatility sub-model** answers "how uncertain and how dynamic is it".

v0.4 asks whether a policy that *sequences* harvests beats the myopic per-lot rule, and by how
much per unit of tracking error. Myopic here means the oracle, or its learned surrogate.

There are two stages, and each has a go/no-go:

| Stage | Policy | Learns? | Gate |
|---|---|---|---|
| v0.4a | walk-forward GBT scores → deterministic budgeted subset optimizer (rung 5) | no | rung 5 vs rung 4: how much of the gap to the oracle ceiling a one-step optimizer captures |
| v0.4b | low-dimensional state-dependent policy over the same executor (rung 6) | yes | rung 6 vs rung 5 at a fixed TE budget, walk-forward, regime-stratified |

RL's *earnable surplus* is defined as the rung-6 − rung-5 gap. It is the value of **timing**
(waiting, sequencing across days, spending the year's capacity) that a one-step optimizer cannot
see. If v0.4a closes most of the gap, v0.4b shrinks.

---

## 1. Objects (typed)

| Object | Type | Notes |
|---|---|---|
| state $s_t$ | flat record in $\mathbb R^{d_s}$, $\mathcal F_t$-measurable | §2 |
| action $a_t=(\vartheta_t,m_t,b_t,g_t)$ | $[0,1]\times\mathbb Z_{\ge0}\times\mathbb R_{\ge0}\times\mathbb R_{\ge0}$ | §3; $g_t$ (trim budget) is new vs the SymbolTable row |
| executor $E(s_t,a_t)\mapsto(A_t,\ \Gamma_t)$ | deterministic map to a harvest set $A_t\subseteq\mathcal K_t$ and a trim set $\Gamma_t$ | §3 |
| transition $\mathcal T(\cdot\mid s,a)$ | Markov kernel; randomness only through $P_{t+1}$ | `SimulationEngine.ProcessDay` |
| reward $r_t$ | $\mathcal S\times\mathcal A\to\mathbb R$, dollars/day | §4 |
| policy $\pi_\psi(a\mid s)$ | parameterized map, $\psi\in\mathbb R^{p}$ with $p$ small | §5 |
| value $V^\pi(s)=\mathbb E_\pi[\sum_t\gamma^tr_t\mid s_0=s]$ | $\mathcal S\to\mathbb R$ | — |

---

## 2. State

$$
s_t=\Big(\underbrace{G^{\mathrm{ST}},G^{\mathrm{LT}},C^{\mathrm{ST}},C^{\mathrm{LT}},O_t,\mathrm{cap}_t}_{\text{ledger (v0.3-3)}},\
\underbrace{\hat\sigma_{\mathrm{TE},t},\ \Phi_k^\top\delta w_t}_{\text{risk (v0.3-6)}},\
\underbrace{\hat\sigma_{m,t},\ q_{\{.1,.5,.9\}}(\hat\sigma_{i,t}),\ q_{\{.5,.9\}}(P^{\mathrm{touch}})}_{\text{volatility (v0.3-7/8)}},\
\underbrace{V_t,\ \mathrm{DaysToYE}_t}_{\text{scale, clock}},\
\underbrace{\omega_t}_{\text{wash}},\
\underbrace{\xi_t}_{\text{scores}}\Big)
$$

- **Ledger.** All of it: the four pools plus the derived $O_t$ and $\mathrm{cap}_t$. The v0.3-4
  finding is that a pure-TLH book is *carryforward-saturated*: $\mathrm{cap}_t=0$ on >99% of
  days, so the marginal harvest is worth $\tau_f\delta$. How far the book is from saturation is
  therefore the single most decision-relevant ledger fact. It is also what a client with outside
  gains (v0.5 personas) changes.
- **Risk.** $\hat\sigma_{\mathrm{TE}}$ comes from the point-in-time Ledoit–Wolf $\hat\Sigma_t$
  with dollar active weights. $\Phi_k^\top\delta w$ is the active exposure to the top-$k$
  eigenvectors of $\hat\Sigma_t$, which is PCA's successor role (risk *directions*, not lot
  compression).
- **Volatility.** The cross-sectional quantiles of $\hat\sigma_i$ over *held* lots, $\hat\sigma_m$,
  and the quantiles of $P^{\mathrm{touch}}=2\Phi(-z)$ over loss lots. Under EWMA/GARCH,
  $\sigma^2_{t+1}$ is $\mathcal F_t$-measurable, so these make the state Markov in the
  quantities that drive the value of waiting (VolatilityModel_v03 §0).
- **Wash.** $\omega_t$ is the fraction of loss-deep lots blocked by §1091, plus the median days
  to unblock.
- **Scores.** $\xi_t$ is the set of quantiles of $\hat\eta$ over gate-passing held lots. The
  **fed-predictions rule** applies: $\hat\eta$ must come from a model trained walk-forward on
  rows $\le t-E$, with $E\ge T_{\mathrm{fwd}}$.

Every component already exists as a column, an engine quantity, or an estimator output. v0.4
adds no new estimators, only aggregations.

---

## 3. Action and executor

The action is **low-dimensional**, never per-lot binaries:
- $\vartheta_t$: a score threshold;
- $m_t$: the maximum number of harvests today;
- $b_t$: a TE budget (the projected $\hat\sigma^2_{\mathrm{TE}}$ after today's trades must stay
  $\le b_t$);
- $g_t$: a **gain-realization budget** in dollars for the trim (v0.3-4). $g_t=0$ means no trim.

The executor $E$ works as follows:
1. Take the gate-passing held lots (loss depth, §1091, and the TE ceiling are hard gates — they
   are law and risk policy, not choices).
2. Score them ($\hat\eta$ or $U$), keep those with score ≥ $\vartheta_t$, and sort.
3. Add them greedily while $|A_t|<m_t$ and the projected TE stays ≤ $b_t$, updating the ledger
   sequentially, so each harvest is valued against the ledger left by the previous one.
4. Then trim overweight gain lots, highest basis first, until $g_t$ is spent.

**Fixed points of the class:**
- the oracle is $(\text{score}=U,\ \vartheta=0,\ m=\infty,\ b=\theta_{\max}^2,\ g=0)$;
- NeverHarvest is $\vartheta=\infty$;
- the v0.3-4 trim is a constant $g$.

So rungs 1–3 lie **inside** the policy class, and "RL beats the oracle" is a well-posed statement.

---

## 4. Reward

**Running TE cost (unchanged from SymbolTable §I).**
$-\lambda^{\mathrm{run}}_t\hat\sigma^2_{\mathrm{TE}}(s_{t+1})$, with
$\lambda^{\mathrm{run}}_t=\kappa_rV_t/(2\cdot252)$, plus $-c\,(|A_t|+|\Gamma_t|)$ for trading.
It is a running cost, so its sum is the time-integral of active risk. The increment form
telescopes away the interior of the path (finding F3).

**Tax term: generalized for gains (new in this record).** The pinned
$B_t=\sum_{j\in A_t}g_{\mathrm{tax}}(\cdot)$ values *harvests*. Once the trim realizes gains, a
gain sale must **cost** tax, and the per-harvest sum has no term for that. Define the tax-position
potential

$$
W_{\mathrm{tax}}(\mathrm{ledger})=-\,\mathrm{Paid}-T(\mathrm{ledger})+\tau_f\,\delta\,\textstyle\sum C'(\mathrm{ledger}),
$$

where:
- $\mathrm{Paid}$ is the tax of closed years;
- $T$ and $C'$ are the year-close of the open year (`LedgerState.Close`, `schedule_d`).

Set

$$
B_t=W_{\mathrm{tax}}(\mathrm{ledger}_t^{+})-W_{\mathrm{tax}}(\mathrm{ledger}_t^{-})
\qquad(\text{after vs before the day's trades, and across the year-end roll on Jan 1}).
$$

**The roll is not continuous (found while implementing `RunMetrics`; an earlier draft claimed it
was).** At the roll $\mathrm{Paid}\mathrel{+}=T$ and the pools reset. The new year's
"as-if-closed" netting then applies carryforward to the next \$3k ordinary line *immediately*. So
with no current-year gains, $W$ jumps by

$$
(\tau_{\mathrm{ord}}-\tau_f\delta)\cdot\min(O_{\max},C)\qquad(=0.27\times\$3\text{k}=\$810\text{ when }C\ge\$3\text{k}).
$$

This is real value: each year \$3k of banked carryforward, carried at $\tau_f\delta=0.10$,
becomes a full-rate ordinary deduction worth $\tau_{\mathrm{ord}}=0.37$. The roll term belongs in
$B_t$. It is policy-dependent through $C$ and attributable to past harvests, so leaving it out
would understate every harvest in a saturated book. `RunMetrics` books it separately
(`RollTrueUp`), and the identity $W_T=\sum\Delta W_{\text{trades}}+\sum\Delta W_{\text{roll}}$ is
tested.

On a harvest-only day, $B_t=\sum_jg_{\mathrm{tax}}$ exactly: $g_{\mathrm{tax}}$ *is* this
difference, taken one harvest at a time (MLDerivations §1.3). On a trim day the gain's tax enters
with the right sign and at the right rate. That rate is whatever the gain is *not* sheltered from:
$\tau_f\delta$ while carryforward beyond the \$3k line absorbs it (the gain spends banked value),
the ordinary rate where it displaces the \$3k line, and $\tau_{\mathrm{LT}}$ or $\tau_{\mathrm{ST}}$
once nothing shelters it.

**Why telescoping is acceptable here and not for TE.** $\sum_tB_t=W_{\mathrm{tax},T}-W_{\mathrm{tax},0}$
depends only on the endpoints. For taxes that is the *objective*: the terminal tax position.
For TE, the objective is the path integral. A potential-difference reward for an
endpoint-objective is exact. For a path-objective it is the F3 bug.

With $\gamma<1$, the discounted sum of potential differences is the endpoint objective plus a
$(1-\gamma)$-weighted interior term. This is the shaping identity of Ng–Harada–Russell (1999) in
reverse. Two consequences:
- $\gamma$ is chosen close to 1 for the tax term, and the episode ends at a year-end;
- the terminal $W$ includes the banked carryforward at $\tau_f\delta$, so ending an episode
  never "loses" harvested value.

**Calibration.** $\kappa_r$ is the client's relative active-risk aversion. It is calibrated so
that the oracle's operating TE is optimal for the myopic problem: the first-order condition of
$U$ at the median marginal harvest. It is then held fixed across rungs, since rungs compared at
different $\kappa_r$ are not comparable.

---

## 5. Algorithms, in order of complexity

| Step | Method | Why this order |
|---|---|---|
| 0 (v0.4a) | budgeted greedy/ILP subset optimizer over walk-forward scores | establishes rung 5; no learning |
| 1 | **CEM** (cross-entropy method) over $\psi$ for a *state-dependent* threshold $\vartheta_t=\mathrm{sigmoid}(\psi^\top\phi(s_t))$, with $m,b,g$ either constant or similarly parameterized | $p\approx10$–$30$ parameters; derivative-free; parallel over FHS seeds; no value function. Directly tests "does waiting value depend on σ̂ and the ledger?" |
| 2 | **fitted-Q** with the GBT substrate: regress $r_t+\gamma\max_{a'}\hat Q(s_{t+1},a')$ on $(s_t,a_t)$ over a discretized action grid | reuses ML.NET GBT, so no new runtime; off-policy, so it can use logged CEM episodes |
| 3 | PPO | only if steps 1–2 leave a material residual against an upper bound (e.g. hindsight-optimal harvest timing on the same paths) |

**The optimal-stopping view.** For one lot, "harvest now vs wait" is optimal stopping. The value
of waiting is the continuation value: a deeper loss later, or a better ledger moment. It is
increasing in $\hat\sigma$ and decreasing in the time left before the year-end and in wash
blocking.

$\hat\eta$ is $P(\text{oracle fires within }T_{\mathrm{fwd}})$. It is **not** the continuation
value, so a policy that thresholds $\hat\eta$ is still myopic.

The barrier coordinate $z$ and the σ̂ summaries in $s_t$ are exactly the inputs a
continuation-value approximation needs. Step 1's state-dependent $\vartheta_t$ is the cheapest
policy class that can express "wait more when σ̂ is high and DaysToYE is large".

---

## 6. Environment and evaluation protocol

- **Train worlds:** `PriceLoader.FromFhs` (clustered σ, fat tails, correlation preserved), over
  many seeds. From the real cache on your machine, or from `GarchFactorPanel` when standalone.
- **Eval world:** real history, walk-forward, on years **never** used as an FHS source for the
  policy being evaluated. Rolling-origin: source ≤ year $y$, evaluate year $y+1$.
- **Metrics:** `RunMetrics` (v0.3-11): after-tax terminal wealth, $B$ used-now vs banked, ex-ante
  vs realized TE, turnover, and costs. Report **tax alpha per unit of realized TE**, per regime
  tercile of $\hat\sigma_m$, with seed dispersion.
- **Gate (v0.4b):** rung 6 > rung 5 at the same realized-TE budget, in the median and the low
  tercile of seeds, in ≥2 of 3 regimes.

---

## 7. Runtime decision (made at the start of v0.4b, against these criteria)

| Criterion | C#-native | Python (gymnasium + SB3) |
|---|---|---|
| Steps 0–2 | native: engine, executor, GBT, CEM all in-process | needs a step bridge (IPC per day) or a port of the engine |
| Step 3 (PPO) | TorchSharp, a thinner ecosystem | first-class |
| Throughput | one process, `Parallel.For` over seeds | bridge latency × days × episodes |
| Reproducibility | one seed path, byte-identity harness reused | two RNG stacks |

**Default:** C#-native for steps 0–2. Python is only adopted if step 3 is triggered, through a
serialized observation/action record. `RunMetrics` and the flat records are defined
serializable from v0.3-11 so either is possible.

---

## 8. Seams and tests to write

| Seam | Where | Test |
|---|---|---|
| `IHarvestPolicy.Decide(s_t) → a_t` | `src/Core/Policy/` (v0.3-11 seam, v0.4 implementations) | the Oracle policy through the executor is byte-identical to the engine's built-in oracle |
| `Executor.Apply(s_t, a_t)` | same | NeverHarvest gives zero harvests; the TE budget is never exceeded ex ante; sequential ledger valuation equals Σ g_tax |
| `TaxPosition.W(ledger)` | `TaxLedger` | $W_T=\sum\Delta W_{\text{trades}}+\sum\Delta W_{\text{roll}}$; the roll jump equals $(\tau_{\mathrm{ord}}-\tau_f\delta)\min(O_{\max},C)$ with no gains; $\Delta W$ = Σ g_tax on a harvest-only day; a gain sale under saturating carryforward (remaining net loss ≥ \$3k) has $\Delta W=-\tau_f\delta\cdot\text{gain}$ |
| `StateRecord.From(engine, t)` | `src/Core/Policy/` | every field is $\mathcal F_t$ (perturb $r_{>t}$ → identical record) |
| fed $\hat\eta$ | walk-forward scorer | model trained on rows $\le t-E$ (asserted on the training-window end) |
| CEM driver | `src/Core/Policy/Cem.cs` | recovers a known optimal threshold on a toy MDP |

---

## 9. Deltas vs the architecture thread

| Topic | Architecture thread (archived) | This design | Why |
|---|---|---|---|
| Reward | accumulated `Y_Utility`, or Σ taxValue − λΔσ_TE | running TE cost plus the **potential-difference tax term** $\Delta W_{\mathrm{tax}}$ | the increment TE form telescopes (F3); summed per-lot $U$ charges the shared TE once per lot; Σ g_tax cannot price a gain sale once the trim exists |
| Action | $(\tau_t,\text{TE budget},\text{turnover budget},m,\text{regime})$ | $(\vartheta,m,b,g)$ with a state-dependent $\vartheta$ | "regime" is *state* (σ̂_m), not an action; turnover is controlled through $m$, $g$ and $c$; the trim budget $g$ is the first non-harvest action |
| State | ledger + TE + factor/score/wash summaries (+ σ̂ "if the gate passed") | the typed record of §2: ledger **by character** with $\mathrm{cap}_t$, PIT $\hat\Sigma$ exposures, **σ̂ unconditionally** | σ̂ is structural (VolatilityModel_v03 §0); saturation is the key ledger fact (v0.3-4 finding) |
| Environment | `MonteCarloEngine` / GBM, then real history | FHS worlds for training, real walk-forward for evaluation, never the same history | GBM has no clustering; `MonteCarloEngine` was retired (one engine, several price sources) |
| Algorithm | PPO/SAC | v0.4a optimizer → CEM → fitted-Q → PPO only if a residual remains | low-dim action, few parameters, expensive episodes: derivative-free search first |
| Optimal stopping | implicit | explicit (§5): $\hat\eta$ is not the continuation value | this is what makes RL's surplus *timing* value, which v0.4a cannot capture |
| Gate | rung 6 vs 5 | rung 6 vs 5 at equal realized TE, walk-forward, regime- and seed-stratified | comparisons at different TE are not comparisons |
| Runtime | PPO in Python implied | criteria table (§7), C#-native default | steps 0–2 need no Python |
| Benchmark | equal weight | equal weight now, cap-weight recorded as an extension | TE level depends on it; SSGA weights exist in `constituents.json` |

---

## 10. Open questions (recorded, not blocking)

- **$\delta$ is the dominant economic parameter of a loss-only client.** Should it be
  state-dependent (a hazard of absorption that rises with expected future gains)? This is the v0.5
  outside-gains personas.
- **Partial §1091 disallowance** vs the current blocking. Disallowed losses add to the
  replacement's basis rather than vanishing. Blocking is conservative, and the policy never needs
  to model basis adjustment.
- The **cap-weighted benchmark**.
- The **upper bound for the RL gate**: hindsight-optimal harvest timing on the same paths, solved
  as a DP per lot. It is cheap and makes "material residual" quantitative.

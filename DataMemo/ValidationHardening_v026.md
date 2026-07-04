# Validation Hardening (v0.26) — Purged Chronological Splits and What They Revealed

> **Status:** shipped (branch `feature/v026-temporal-validation`). This memo records the
> method, the measured random-vs-temporal deltas, and — the load-bearing part — the
> **diagnosis** that separates leakage-removal from regime shift. The gate result is a *pass
> with a mandate*: ranking skill survives honest temporal validation; the headline PR-AUC drop
> is the cost-basis-aging prevalence crash, which is precisely the v0.3 P0.
>
> Adopted from the GPT-thread architecture review, §4 item 1 and §5 (v0.26) of
> `DataMemo/temp/direct_indexing_concept_architecture_plan_contextualized.md`.

---

## 1. Why the old splits were suspect

Every model in v0.1–v0.25 was selected and evaluated under **stratified *random*** splits
(`StratifiedSplit` / `StratifiedKFold`). But the soft labels are **forward-looking**:
`Y_Soft_BT(x_t)` is a deterministic function of prices over the window `(t, t+30]`. Two rows
whose timesteps are within 30 trading days therefore share overlapping future context. A
random split scatters such neighbours across train and test, so a model can be scored on a
future it partially memorised in training — classic look-ahead leakage in a panel with
overlapping labels (López de Prado, *Advances in Financial ML*, ch. 7).

The report already named a chronological evaluation as "the natural next robustness check."
v0.26 builds it.

## 2. What was built

Three files under `src/ML/CSharp/MLNet/Splits/`, plus a CLI surface, plus tests:

- **`TemporalSplit.cs`** — pure functions on the `Timestep` axis:
  - `TrainTest`: chronological split at a row-mass boundary; train rows within an **embargo
    gap** `E` of the boundary are *purged*. With `E = 30`, any surviving training row at
    `t ≤ T* − 31` has its label window `⊆ (t, T*−1]`, strictly before the test period → zero
    overlap.
  - `PurgedFolds`: contiguous k-fold with **both-side** embargo around each validation block
    (the purged k-fold of the finance-ML literature — uses more data per fold than
    forward-chaining while giving the same no-overlap guarantee).
- **`SplitPolicy.cs`** — a process-wide mode set once from the CLI, read by every trainer.
  Default is `StratifiedRandom`, so existing commands reproduce v0.25 numbers bit-for-bit.
  `--split=temporal`, `--embargo=N` (default 30 = the label horizon), `--testfrac=F`
  (`0.5` ⇒ the decade walk-forward). Temporal-mode artifacts write to
  `data/artifacts-mlnet-temporal/` so random-split baselines are never clobbered.
- **`DataSplit.cs`** — a facade (`TrainTest` / `Folds`) that dispatches on `SplitPolicy`. The
  14 trainer/pipeline call sites were renamed mechanically; **the partition semantics now
  change in exactly one place**, which is the altitude the review asked for.
- `TemporalSplitTests` (5): boundary/embargo honesty, purge accounting, both-side fold
  embargo + partition coverage, determinism (no RNG), facade dispatch.

No trainer, metric, or preprocessing code changed — only *where the row boundary is drawn*.

## 3. The measurement (canonical scalarized arm, 20y)

**CV PR-AUC, stratified-random → temporal-purged (80/20):**

| model | oracle rand | oracle temp | soft rand | soft temp |
|---|---|---|---|---|
| GBT | 0.9956 | 0.9474 | 0.8158 | 0.4267 |
| RF | 0.9883 | 0.9252 | 0.6332 | 0.3911 |
| logistic | 0.9804 | 0.8621 | 0.6233 | 0.3965 |
| elastic net | 0.8107 | **0.8694** | 0.5874 | 0.3663 |
| linreg | 0.8227 | **0.8942** | 0.5590 | 0.2775 |

**Test-set GBT, the decisive pair — ROC-AUC (prevalence-*insensitive*) vs PR-AUC
(prevalence-*bounded*):**

| target · split | ROC-AUC | PR-AUC |
|---|---|---|
| soft · random | 0.9935 | 0.8126 |
| soft · **temporal** | **0.9970** | 0.4585 |
| oracle · random | 0.9986 | 0.9951 |
| oracle · **temporal** | **1.0000** | 0.9507 |

**Positive-rate by period (the confound made explicit):**

| region | rows | oracle+ | soft+ |
|---|---|---|---|
| train (2006 – ~2022) | 1,466,887 | 0.246% | 3.931% |
| test (~2022 – 2026, last 3.6y) | 369,969 | **0.012%** | **0.221%** |
| decade test (~2016 – 2026) | 924,909 | 0.018% | 0.333% |

## 4. Diagnosis — leakage vs. regime, disentangled

The PR-AUC collapse looks alarming (soft GBT 0.81 → 0.46) until you read it against ROC-AUC:

1. **ROC-AUC did not drop — it rose.** ROC-AUC measures *ranking*: P(model scores a random
   positive above a random negative). It is insensitive to class prevalence. If random splits
   had been materially leaking future context into the ranking, removing the leak would
   *lower* ROC-AUC. It held (0.9935 → 0.9970 soft; 0.9986 → 1.0000 oracle). **The champions'
   ranking skill transfers across time; there was no material ranking leakage.**

2. **PR-AUC is prevalence-bounded, and the test-period prevalence crashed 18×.** The no-skill
   AUPRC *equals* the positive rate, and the whole PR curve scales with it. The temporal test
   period is the cost-basis-*aged* tail: soft+ falls from 3.93% (train era) to 0.22% (test
   era). Identical ranking scored on an 18×-rarer positive class yields a mechanically lower
   AUPRC. In **lift over no-skill**, the temporal model is *better*, not worse: 0.81/0.032 ≈
   25× (random) vs 0.46/0.0022 ≈ **210×** (temporal).

3. **Why the tail is so sparse: cost-basis aging (the report's #1 finding).** Lots opened once
   at warmup hold their basis for two decades; by 2022 a position is so deep in the money that
   even a bear market cannot push it 2% underwater. The recent years are a near-dead harvest
   regime *by construction of the current simulator*, not by market calm.

4. **The CV-table wrinkle confirms it.** Raw-linear models (elastic net, linreg) *improve* on
   the oracle target under temporal splits (0.81 → 0.87, 0.82 → 0.89) while the trees dip
   slightly — consistent with an aged-out tail whose boundary is *simpler* (few, deep-loss
   positives), which a hyperplane handles better and leaves the trees less to exploit.

**Conclusion.** The v0.1–v0.25 random-split PR-AUC was **not leakage-inflated in its ranking
content** (ROC held), but it **was a regime-averaged number** dominated by the harvest-rich
early years — a figure a chronologically-deployed system will not see once its book ages. The
honest performance statement going forward is the pair: *ranking (ROC-AUC) is robust across
time; absolute precision-recall (PR-AUC) is regime-dependent and low in aged-out books.*

## 5. Gate decision → v0.3 mandate

**Pass.** The champions survive honest temporal validation on the metric that measures skill.
The one thing the harder split exposed — the aged-out-tail prevalence crash — is not a
methodology failure; it is the strongest possible motivation for the **v0.3 cost-basis-aging
fix** (contributions / rebalancing / the sell-winner trim), which restores harvestable supply
→ recovers test-period prevalence → recovers PR-AUC without touching ranking.

**Standing methodology change from v0.26 on:** report **ROC-AUC, PR-AUC, and test-period
positive-rate together** — never PR-AUC alone — and prefer `--split=temporal` for any claim
about deployed-forward performance. Random splits remain the default only for reproducing the
historical v0.25 ablation numbers.

## 6. Reproduce

```text
# canonical arm, honest 80/20 temporal split (embargo = 30d label horizon):
dotnet run mlnet-oracle --split=temporal      # → data/artifacts-mlnet-temporal/
dotnet run mlnet-soft   --split=temporal
# decade walk-forward (train ~2006–2016, test ~2016–2026):
dotnet run mlnet-oracle --split=temporal --testfrac=0.5
dotnet run mlnet-soft   --split=temporal --testfrac=0.5
```

*Cross-references: `GYTD_Redesign_Plan.md` §6.1 (the random-split ablation this hardens);
the report's cost-basis-aging section (the prevalence-crash mechanism);
`DataMemo/temp/direct_indexing_concept_architecture_plan_contextualized.md` §4–§5 (why v0.26
gates v0.3). Method: López de Prado, purged & embargoed cross-validation.*

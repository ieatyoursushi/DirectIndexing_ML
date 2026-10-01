# The ML.NET Layer — architecture, pipeline shape, and the sklearn reconciliation

> **Status: LIVE SPEC** — must match the code; checked by `dotnet run --project src -- docs-check`.
> Index of every symbol ↔ code member ↔ test: [`SymbolTable.md`](SymbolTable.md).

> **Purpose.** Everything about *how* the ML layer is built in C#/ML.NET: why it exists in this
> language at all (§1), the typed pipeline shape from `LotStateVector` outward (§2), and the
> complete map of every place ML.NET's parameterization differs from sklearn's and how the
> codebase bridges it (§3).
>
> **This document replaces three memos** — `MLNetPipeline.md`, `MLNetVsPython.md`, and
> `MLNetSemanticReconciliation.md` — which were written when the project carried *two parallel
> ML pipelines on two branches* (`feature/MLnet-layer` vs `feature/ML-layer`) that were meant
> to be checked out interchangeably and compared run-for-run. **That world ended at PR #11:**
> the ML.NET layer won, was merged to `master`, and is now the only ML pipeline. Python is
> confined to the report/render layer behind the `PythonRunner` subprocess seam. The
> cross-branch comparison premise has been removed throughout; the *parameter reconciliation*
> it produced is preserved in §3 because it remains the authoritative reference for anyone
> reading these models against sklearn intuitions.
>
> The leakage invariants live separately and deliberately in
> **`MLNetLeakageAudit.md`** — that memo is cited from the README's methodology section as the
> load-bearing statement of the "invariants are types, not conventions" discipline, and it
> stays standalone.
>
> Current as of **schema v3 (d = 17)**, scalarized oracle (v0.25), purged temporal splits
> available (v0.26).

---

# §1. Why ML.NET and not sklearn

## 1.1 What sklearn was designed around

sklearn's estimator API was designed for **rapid experimental iteration in a notebook**. The
contract is intentionally minimal: every model exposes `.fit(X, y)` and `.predict(X)`; every
transform exposes `.fit_transform`. `X` is a NumPy array, which has no enforced schema. Column
names exist only as long as you keep a DataFrame around; the moment you call `.values`, they
vanish and the model sees positional indices.

This is a *correct* design for its use case — research and prototyping where the developer
iterates fifty times an hour and doesn't want a type system in the way. It is *not* a good fit
when:

- the data has a stable schema you already control (`LotStateVector` here),
- you want stage-by-stage output schemas to be inspectable,
- you want the difference between "the thing that learns" and "the thing that transforms"
  visible in the type system,
- the partition / weighting / impute order matters and you'd rather have the compiler enforce
  it than rely on the library doing it inside `.fit()`.

## 1.2 Schema-first, not workaround

A Python pipeline must read `data/lots.csv` via `pd.read_csv` because Python has no access to
the C# type that produced it. ML.NET reads directly from the producing record:

```csharp
var data = mlContext.Data.LoadFromEnumerable(snapshots);
// snapshots: IEnumerable<LotStateVector>
// schema is the record's field declaration; no inference, no LoadColumn ordering
```

The `IDataView` that emerges carries every column's *name* and *type* through every stage.
`Concatenate("Features", …)` references columns by name, not position. If `LotStateVector`
gains a field and the chain still references an old name, the compiler catches it before
runtime. This is exactly what made the v0.25 schema migration (d=15→17, `G_YTD` → the ledger
triple, `TaxAlpha` → `TaxValue`) a *mechanical, compiler-checked* change rather than a hunt
for positional mismatches.

## 1.3 `IEstimator<T>` vs `ITransformer` — the right type-level split

sklearn collapses two distinct mathematical objects into one Python class:

- the function from data to a model: $D \to M$ (an unfitted estimator)
- the function from data to data: $X \to X'$ (a fitted transformer)

ML.NET separates them:

- `IEstimator<TTransformer>` — "I haven't seen data yet; call `.Fit(data)` and I'll produce a `TTransformer`."
- `ITransformer` — "I've already seen the training data; call `.Transform(data)` and I'll produce output."

This is the type-level distinction between **training** and **inference**. A
`TransformerChain<ITransformer>` is an *applied* pipeline; an `EstimatorChain<TTransformer>` is
the unfitted recipe. The fitted thing carries the parameters; the unfitted thing is just a
description of how to compute them.

## 1.4 Partitioning as a visible function, not an opaque parameter

```python
# sklearn — stratification is a knob
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
```

```csharp
// ML.NET — the partition is a function you can read
public static (List<LotStateVector> Train, List<LotStateVector> Test) Split(
    IReadOnlyList<LotStateVector> data,
    Func<LotStateVector, int> labelSelector,
    double testFraction = 0.20,
    int seed = 42)
{ /* bucket by class; shuffle; slice */ }
```

For someone whose mental model is *"the partition of $\mathcal D$ into train and test is a
function $f:\mathcal D\times(\mathcal X\to\{0,1\})\times[0,1]\to\mathcal D^2$"* — the second
form **is** that function, written down. The first is the same function hidden behind someone
else's parameter.

This paid off concretely in v0.26. Adding purged chronological splits meant writing a second
partition function (`TemporalSplit`) and a policy that selects between them (`SplitPolicy`),
routed through one facade (`DataSplit`). Because partitioning was already a visible function
rather than a library knob, the change touched *one seam* and left all fourteen trainer call
sites signature-identical.

## 1.5 Where Python is genuinely better — and kept

The architecture preserves Python for presentation:

- **EDA + report rendering** — `scripts/eda.py`, `scripts/build_report_notebook.py`,
  `scripts/report.py`, `scripts/render.py`, `scripts/codebook.py`. Matplotlib's REPL feedback
  cycle for iterating on plot styling genuinely beats C#'s compile-edit-run loop.
- The Python side gets `pandas + matplotlib + nbformat/nbclient + jinja2` only — **no sklearn,
  no ML logic**.

The boundary is **JSON/CSV in, PNG/HTML/notebook out**. C# owns all ML semantics; Python owns
presentation. `PythonRunner.cs` is the only interop seam.

## 1.6 On "but Python is the standard"

True — for **ML research and reproducing published work**. Less relevant here: this is tabular
binary classification on a typed domain model owned in C#, at ~1.85M rows × 17 features, well
within the regime ML.NET was designed for (typed tabular pipelines intended for production
deployment).

**When this inverts.** If the project grows a neural training stage, nonlinear dimensionality
reduction (UMAP/ICA), or the **v0.4 reinforcement-learning policy layer**, those stages move
*back* to Python via `PythonRunner` — the seam is already in place. ML.NET handles tabular;
Python handles what only Python can. (Written in v0.1; the v0.4b RL layer now on the roadmap
is precisely the predicted case.)

---

# §2. The pipeline shape

## 2.1 Mental model

```text
LotStateVector  (C# record, already typed — d=17 + labels + metadata)
       │
       ▼   LoadFromEnumerable<LotStateVector>
IDataView  (named, typed columns — no schema loss, no CSV round-trip)
       │
       ▼   DataSplit  →  SplitPolicy ∈ {StratifiedRandom, TemporalPurged}
       │                  (v0.26: purged chronological splits, embargo ≥ 30d)
       ▼   MedianImputer.Fit(trainFold) → Apply       (training-fold-only)
       │   ClassWeights.AttachBalancedWeights(trainFold)
       ▼   EstimatorChain
            ├─ CustomMapping          (Sector "" → "Unknown")
            ├─ OneHotEncoding         (SectorClean → SectorOneHot)
            ├─ NormalizeMeanVariance  (17 numeric features, by name)
            └─ Concatenate("Features", numerics ‖ SectorOneHot)
       │
       ▼   Trainer  (GBT = champion · logistic = linear control)
       │
       ▼   ITransformer   (fitted, inspectable at every stage)
       │
       ▼   BinaryMetrics.Compute  (manual step-function PR-AUC + ROC + F1 sweep)
       │
       ▼   JSON metrics + ROC/PR curve points  → data/artifacts-mlnet{dataset}{split}/
       │
       ▼   PythonRunner subprocess  → EDA + model plots, codebook (schema-drift assert)
```

Every arrow is typed; nothing is positional. `LotStateVector.cs` is the single source of
truth — no `[LoadColumn]` projection, no DataFrame intermediary, no point at which a column
name degrades to an index.

## 2.2 Design choices, stage by stage

| Stage | The sklearn shape | This codebase | Why the difference is principled |
|---|---|---|---|
| **Schema** | implicit in `read_csv` + a `NUMERIC_FEATURES` list | explicit in `LotStateVector` + `FeatureLists.cs` | C# has the type system Python lacks; re-encoding "schema-by-convention" in a language with real records is an anti-pattern |
| **Loading** | CSV → DataFrame → `.values` (positional) | record → `LoadFromEnumerable` → `IDataView` (named) | no CSV inside the ML layer; the structure that produced `lots.csv` *is* the structure the trainer sees |
| **Split** | `train_test_split(stratify=…)` | `DataSplit` → `StratifiedSplit` \| `TemporalSplit` | the partition is the thing being studied; visible function > hidden parameter (and it's what made v0.26 cheap) |
| **K-fold** | `StratifiedKFold(n_splits=5)` | `StratifiedKFold.Folds` \| `TemporalSplit.PurgedFolds` | same reason; purged folds have no sklearn equivalent at all |
| **Grid search** | `GridSearchCV` black box | `GridSearchCV.Search(grid, factory, scorer)` | a hyperparameter config is *a function from a parameter dict to an estimator*; the factory makes that explicit |
| **Preprocessing** | `ColumnTransformer([...])` | typed `EstimatorChain`, named I/O per stage | `ColumnTransformer` exists because sklearn pipelines drop column names; ML.NET's don't |
| **Class balancing** | `class_weight='balanced'` (correct, by accident — inside `.fit()`) | `ClassWeights.AttachBalancedWeights(trainFold, …)` — the only signature that exists | makes the training-fold-only requirement *structurally enforced* |
| **Median impute** | `fillna(median())` on the full frame (subtle leak) | `MedianImputer.Fit(trainFold) → dict; Apply(rows, dict)` | the two-call shape makes "fit on train, apply elsewhere" the natural path |
| **Plots** | matplotlib inline | C# emits JSON → Python reads JSON → matplotlib | visualization iteration is genuinely better in Python's REPL; the boundary stays |

## 2.3 Output artifacts

`data/artifacts-mlnet{dataset}{split}/` — written by C#. The directory name *records how the
numbers were produced*, so arms never overwrite each other. It is derived, not hand-named:
`{dataset}` is the `--lots` file's arm tag and `{split}` is `SplitPolicy.ArtifactTag`.

| `--lots` | split | Directory |
|---|---|---|
| `data/lots.csv` (default) | random | `artifacts-mlnet/` — canonical |
| `data/lots.csv` | `--split=temporal` | `artifacts-mlnet-temporal/` |
| `data/lots_contrib.csv` | `--split=temporal` | `artifacts-mlnet_contrib-temporal/` |
| `data/lots-mc.csv` | random | `artifacts-mlnet-mc/` |

(`simulate` names its datasets the same way: `lots{_contrib}{_ctrade<x>}.csv`. The pre-v0.3
`artifacts-mlnet-gated/` arm is retired with the gated oracle.)

Contents: `{model}_{target}_metrics.json` (CV scores, test ROC/PR-AUC, F1 at both thresholds,
ROC + PR curve points, chosen hyperparameters), `{model}_{target}_model.zip`,
`logistic_{target}_coefficients.csv`, `{target}_cv_leaderboard.json` (names the champion and
each model's role), and `tax_value_regression_metrics.json`. The retired unsupervised set
(`pca_*`, `kmeans_*`, `cluster_assignments.*`) is no longer written.

`src/Export/eda-mlnet/`, `src/Export/models-mlnet/`, `src/Export/codebook/` — written by Python
(PNGs, index.html, codebook). `src/Export/report/` is the **frozen** course submission.

---

# §3. sklearn ↔ ML.NET reconciliation

Every place ML.NET's defaults or parameterization differ from sklearn's, and how this codebase
bridges the gap. This section is the authoritative reference for reading these models against
sklearn intuitions.

## 3.1 Summary table

| Item | sklearn knob | ML.NET equivalent | File |
|---|---|---|---|
| Regularization | `C` (inverse) | `L2Regularization = 1/C` | `LogisticTrainer.cs` |
| Class weights | `class_weight='balanced'` | `WeightedRow` + `ExampleWeightColumnName` | `ClassWeights.cs` |
| Median impute | `fillna(median())` | manual, on the training fold | `MedianImputer.cs` |
| Sector NaN | `handle_unknown="ignore"` | `CustomMapping` → `"Unknown"` | `PreprocessingPipeline.cs` |
| F1-optimal threshold | manual sweep | manual sweep | `BinaryMetrics.cs` |
| PR-AUC | step-function AP | step-function AP (manual) | `BinaryMetrics.cs` |
| Seed | `random_state=42` | `MLContext(seed:42)` + `Random(42)` | everywhere |
| K-fold partition | sklearn internal | round-robin per class | `StratifiedKFold.cs` |
| **Purged temporal CV** | *(no equivalent)* | `TemporalSplit.PurgedFolds` | `TemporalSplit.cs` |
| Solver iter cap | `max_iter=100` | `MaximumNumberOfIterations=200` | `LogisticTrainer.cs` |
| GBT trees / leaves | `n_estimators` / `max_leaf_nodes` | `NumberOfTrees` / `NumberOfLeaves` | `GradientBoostedTreesTrainer.cs` |
| Uncalibrated probability | sklearn always calibrates | `Score` used as proxy (no current trainer needs it) | `BinaryMetrics.cs` |
| Champion selection | `best_estimator_` | `SelectChampion` = argmax CV; `RunAllSupervised` leaderboard | `MLnetPipeline.cs` |

## 3.2 `C` ↔ `L2Regularization` — opposite directions

**sklearn:** `C` is the *inverse* regularization strength — higher `C` = weaker penalty.
**ML.NET:** `L2Regularization` is the direct coefficient — higher = stronger penalty.

**Reconciliation.** `l2 = 1f / C`. The grid $C\in\{0.01,0.1,1,10\}$ becomes
$L2\in\{100,10,1,0.1\}$. Both `bestC` and `l2Used` are recorded in the metrics JSON so a
reader can verify the mapping was applied. Source: `LogisticTrainer.Run` → `L2Used = 1.0 / bestC`.

## 3.3 Class balancing

**sklearn** computes $w_k = N/(n_{\text{classes}}\cdot n_k)$ *inside* `.fit()`, so the
calculation lands on the training fold only — correct, but by accident of where it runs.
**ML.NET** has no `class_weight`; weighting is via an example-weight column.

**Reconciliation.** Compute the identical formula in
`ClassWeights.AttachBalancedWeights(trainFold, …)`, attach as a `Weight` column on a typed
`WeightedRow`, pass `ExampleWeightColumnName = "Weight"`. The math is bit-identical; the
difference is that the training-fold restriction is *visible in the signature* rather than
buried in library internals (`MLNetLeakageAudit.md` §1).

## 3.4 Missing values — Median vs Mean

`ReplaceMissingValues.ReplacementMode` exposes `Mean`, `Min`, `Max`, `Default` — **not
Median**. So medians are computed manually on the training fold in `MedianImputer.Fit` and
applied *before* `LoadFromEnumerable`, bypassing the built-in transform entirely. Using `Mean`
would diverge meaningfully on heavy-tailed features — `L = (P_t − p_k)/p_k` has a fat right
tail in bull windows.

## 3.5 Sector NaN handling

Pandas reads empty CSV cells as `NaN` and `OneHotEncoder(handle_unknown="ignore")` emits an
all-zero row. ML.NET reads empty string *as* empty string and would create an empty-string
category. A `CustomMapping` inserted before `OneHotEncoding` rewrites empty/whitespace
`Sector` to literal `"Unknown"`, making the unknown bucket a deliberate, named training-time
category.

## 3.6 PR-AUC — the definition actually matters

**sklearn:** `average_precision_score` — the step-function area
$\sum_n (R_n - R_{n-1})P_n$.
**ML.NET:** `AreaUnderPrecisionRecallCurve` uses a **trapezoid** approximation. These differ
by $O(1/n)$, which is significant on highly imbalanced data.

**Reconciliation.** `BinaryMetrics.Compute` implements the step-function average precision
identically to sklearn (`prAuc += (recl - prevRecall) * prec;`), bypassing ML.NET's built-in.
Verified on the `Y_Oracle` sanity baseline where AP ≈ 1.0 if both compute it correctly.

## 3.7 Calibration and the `Score` fallback

FastTree (GBT) applies Platt calibration internally and emits a `Probability` column, as does
L-BFGS logistic regression. `BinaryMetrics.Compute` checks for `Probability` and falls back to
`Score` — a path only the retired FastForest and least-squares trainers exercised, kept for any
future uncalibrated scorer. The fallback:

- **preserves** valid ROC and PR curves (ranking is calibration-invariant),
- **preserves** valid AUC metrics (AUC is a ranking statistic),
- **breaks** probability estimates at a given threshold.

So confusion matrices and F1 at $\tau=0.5$ are meaningful for a $[0,1]$-bounded `Score` but not
for an unbounded one.

## 3.8 Solver

Both are L-BFGS on a convex objective, so both find the same global optimum — but convergence
criteria differ (sklearn `tol=1e-4, max_iter=100`; ML.NET `OptimizationTolerance=1e-7`,
unbounded iterations, capped here at 200). This is the most likely source of small numeric
differences in fitted coefficients.

## 3.9 Retired reconciliations

The elastic-net objective map ($\lambda_1=\rho/C$, $\lambda_2=(1-\rho)/(2C)$), the RF
`FeatureFraction` vs `max_features='sqrt'` note, the linear-regression-demonstrator plumbing,
and the PCA-loadings-via-MathNet-SVD recovery went with their trainers in the pre-v0.3
downsizing — findings in `archive/RetiredComponents.md`, text at tag
`archive/v0.3-pre-downsize`. `MLReadyRow.FloatLabel` survives: the tax-value regression uses it
for its genuinely continuous target.

---

## Cross-reference

- `MLNetLeakageAudit.md` — the training-fold-only invariants, made structural. **Read this one separately; it is the load-bearing discipline statement.**
- `MLDerivations.md` — the mathematics of the feature space, oracle, labels, and every model objective.
- `ValidationHardening_v026.md` — why splits became temporal, and the leakage-vs-prevalence diagnosis.
- `GYTD_Redesign_Plan.md` — the v0.25 oracle redesign the schema migration served.
- `src/ML/CSharp/MLNet/` — `Schema/FeatureLists.cs`, `Splits/`, `Preprocessing/`, `Models/`, `Metrics/`, `Tuning/`.

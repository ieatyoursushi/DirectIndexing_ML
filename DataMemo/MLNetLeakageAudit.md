# MLNet Leakage Audit — training-fold-only invariants

This memo enumerates the points in the pipeline where the test set could leak into training — and shows how the ML.NET pipeline's structure makes the correct order **structurally enforced** rather than just "documented and hope".

For each item: the Python pipeline's behavior, the ML.NET pipeline's behavior, and the structural reason the ML.NET version cannot regress without an active code change.

## 1. Class weights

**The math.** For `class_weight='balanced'`, the weight for class $k$ is $w_k = N / (n_{\text{classes}} \cdot n_k)$ where $n_k$ is the count of class $k$ examples and $N$ is the total. If $n_k$ is counted on the *full* dataset before splitting, the test set's class distribution leaks into the training weights.

**Python (`logistic.py`).** Uses `LogisticRegression(class_weight='balanced')`. sklearn computes $n_k$ inside `.fit()`, so it sees only the training fold's labels. Correct — but *by accident*: nothing about the surrounding code structure enforces this; if a developer ever computed weights manually with `compute_class_weight('balanced', y_full)`, the leak would be silent.

**ML.NET (`Preprocessing/ClassWeights.cs`).** The only signature available is:

```csharp
public static List<WeightedRow> AttachBalancedWeights(
    IReadOnlyList<LotStateVector> trainingFold,
    Func<LotStateVector, int> labelSelector)
```

The parameter is named `trainingFold`. There is no overload that takes a "full dataset" plus a "train indices" pair. Inside `GridSearchCV.Search`, the call site is:

```csharp
foreach (fold) {
    var (train, val) = folds[f];
    var trainW = ClassWeights.AttachBalancedWeights(train, labelSelector);  // ← train only
    ...
}
```

The leak is structurally unrepresentable: to introduce it, you'd have to add a new overload or fundamentally change the function's shape.

## 2. Median imputation

**The math.** Median is a sample statistic. Computing it on the full dataset and then splitting means $\hat{m}_{\text{train}}$ is biased toward the test set's distribution.

**Python (`split.py`).** Calls `X[c].fillna(X[c].median())` *before* the train/test split. The median is computed on the union of train + test. Minor numerical leak at 122K rows, but the *shape* is wrong — and on a smaller dataset it would matter.

**ML.NET (`Preprocessing/MedianImputer.cs`).** Two-call shape:

```csharp
public static Dictionary<string, float> Fit(IReadOnlyList<LotStateVector> trainingFold);
public static List<MLReadyRow> Apply(IReadOnlyList<LotStateVector> rows,
                                      Dictionary<string, float> medians, ...);
```

`Fit` only takes training rows; `Apply` takes already-computed medians plus any list. The natural usage is:

```csharp
var medians = MedianImputer.Fit(train);     // train only
var trainReady = MedianImputer.Apply(train, medians, ...);
var testReady  = MedianImputer.Apply(test,  medians, ...);  // uses TRAIN medians
```

`Fit` cannot accept test data because it's the *function that computes medians*; calling it on test would be a name-meaning mismatch, not just a convention violation.

## 3. PCA — retired, and the rule it generalizes to

The feature-space PCA (`Models/PcaPipeline.cs`) was retired in the pre-v0.3 downsizing
(`archive/RetiredComponents.md` §4). Its invariant was the `EstimatorChain.Fit(trainView)`
shape plus a MathNet SVD over the *same* training-fold matrix, so test variance could not tilt
the principal axes.

The rule carries over to its successor, the eigendecomposition of the return covariance
$\hat\Sigma_t$ (v0.3-4). There it takes the time-series form: **a covariance used on day $t$
may be estimated only from returns observed before $t$** ($\mathcal F_{t-1}$-measurable). The
current `TrackingErrorProxy` violates exactly this. It estimates $\hat\Sigma$ once from the
full price history, so $\sigma_{\mathrm{TE}}$ (a feature) peeks forward (ROADMAP finding F1).
The split-level leakage control cannot see a leak built into a feature, which is why it is
recorded here.

## 4. NormalizeMeanVariance (per-feature standardisation)

**The math.** Same as PCA — mean and variance are sample statistics.

**Python (`preprocessing.py`).** The `ColumnTransformer` containing `StandardScaler` is inside the sklearn `Pipeline`, so sklearn fits it on the training fold inside `.fit()`. Correct by accident, like class weights.

**ML.NET (`Preprocessing/PreprocessingPipeline.cs`).** `NormalizeMeanVariance` is inside the `EstimatorChain`. The chain is fit on the training fold; the fitted transformer is applied to validation/test. Same `.Fit / .Transform` invariant as PCA.

## 5. Sector vocabulary (`OneHotEncoding`)

**The math.** The one-hot encoding's vocabulary should be set from training data only. If the test set has a sector unseen in training, it should map to the "unknown" bucket — not silently create a new column.

**Python.** `OneHotEncoder(handle_unknown="ignore")` handles this correctly when fit on training. Same accidental-correctness pattern.

**ML.NET (`Preprocessing/PreprocessingPipeline.cs`).** The `OneHotEncoding` transform is inside the `EstimatorChain` and fitted on the training fold. The fitted transformer's vocabulary is fixed; test-set sectors outside that vocabulary map to all-zeros (the default unknown handling). Plus we add a `CustomMapping` *before* one-hot that explicitly coerces empty `Sector` to `"Unknown"`, so the "unknown" bucket is itself a deliberate training-time category.

## Summary

| Place | Python correctness | ML.NET correctness | Mechanism |
|---|---|---|---|
| Class weights | ✓ accident (sklearn does it inside `.fit()`) | ✓ structural (only `Fit(trainFold)` signature exists) |
| Median impute | ✗ leak (full dataset before split) | ✓ structural (`Fit` only takes training fold) |
| PCA *(retired)* | ✗ leak when used as features (`PCA().fit(full)`) | ✓ structural while it existed; successor rule: point-in-time $\hat\Sigma_t$ (§3) |
| Standardisation | ✓ accident (inside sklearn pipeline) | ✓ structural (inside EstimatorChain) |
| One-hot vocabulary | ✓ accident (inside sklearn pipeline) | ✓ structural (inside EstimatorChain) |

The pattern: *accidents become invariants when you have a type-shaped pipeline*. That's the whole architectural argument condensed.

## Regression tests — what exists, and the recorded gap

What is tested (`src/Tests/MLNetTests.cs`, run by `dotnet run --project src -- test`):
- `PreprocessingTests.Test_MedianImputerReplacesNaNs`: NaNs are replaced with the
  **training-fold** median.
- `PreprocessingTests.Test_ClassWeightsBalanced`: balanced weights are computed from the list
  passed in.
- `TemporalSplitTests`: the purge/embargo arithmetic (v0.26).

These invariants are enforced **structurally**: the only `Fit` / `AttachBalancedWeights`
signatures take a single list, which every call site obtains from `DataSplit`'s training
fold. Tests pin the *behaviour* of those functions.

**Recorded gap.** Earlier revisions of this memo described a
`src/Tests/MLNet/LeakageRegressionTests.cs` that fits on train, refits on train + test, and
asserts that the production path matches the train-only parameters. That test never existed
in the repository's history. It is now a backlog item (ROADMAP §open) rather than a claim.

# MLNet Semantic Reconciliation

Every place ML.NET's defaults / parameterization differ from sklearn's — and how this codebase bridges them so the two branches' outputs are comparable.

Cross-branch comparison: `data/artifacts/logistic_*_metrics.json` (Python) vs `data/artifacts-mlnet/logistic_*_metrics.json` (ML.NET). Test ROC-AUC and PR-AUC should agree to within ~5%. Larger gaps indicate one of the items below isn't reconciled.

## 1. `C` ↔ `L2Regularization` — opposite directions

**sklearn.** `LogisticRegression(C=1.0)` — `C` is the *inverse* regularization strength. Higher `C` = weaker regularization = less penalty on large weights.

**ML.NET.** `LbfgsLogisticRegressionBinaryTrainer.Options.L2Regularization = 1.0f` — direct regularization coefficient. Higher value = stronger regularization.

**Reconciliation.** Map sklearn's `C` to ML.NET's `L2` via `l2 = 1f / C`. The grid `{0.01, 0.1, 1.0, 10.0}` for `C` becomes the grid `{100, 10, 1, 0.1}` for `L2`. Both `C_searched` and `l2_used` are recorded in `logistic_{target}_metrics.json` so a reader can verify the mapping was applied:

```json
{
  "bestC":   0.1,
  "l2Used":  10.0,
  ...
}
```

Source: `LogisticTrainer.cs:Run` → `L2Used = 1.0 / bestC`.

## 2. Class balancing — `class_weight='balanced'` has no direct equivalent

**sklearn.** `LogisticRegression(class_weight='balanced')` — computes per-class weights as $w_k = N / (n_{\text{classes}} \cdot n_k)$ *inside* `.fit()` so the calculation is on the training fold only.

**ML.NET.** `LbfgsLogisticRegressionBinaryTrainer` has no `class_weight` parameter; weighting is via an example-weight column.

**Reconciliation.** Compute the same formula manually in `ClassWeights.AttachBalancedWeights(trainFold, ...)`, attach as a `Weight` column on a typed `WeightedRow` (which inherits `LotStateVector`), and pass `ExampleWeightColumnName = "Weight"` to the trainer. The math is bit-identical; the difference is that the training-fold restriction is now visible in code rather than buried in sklearn internals (see `MLNetLeakageAudit.md` §1).

## 3. Missing-value imputation — `Median` vs `Mean`

**sklearn.** `df[c].fillna(df[c].median())`.

**ML.NET.** `ReplaceMissingValues.Options.ReplacementMode` exposes `Mean`, `Min`, `Max`, `Default` — **not Median**.

**Reconciliation.** Compute medians manually on the training fold in `MedianImputer.Fit` and replace NaNs in `MedianImputer.Apply` *before* `LoadFromEnumerable`. This bypasses `ReplaceMissingValues` entirely, keeping median semantics intact. Using `Mean` would diverge meaningfully on heavy-tailed features (e.g. `L = (P_t − p_k)/p_k` has a fat right tail in bull-market windows).

## 4. Sector NaN handling — empty string vs `"Unknown"`

**Python.** Pandas reads empty CSV cells as `NaN`; `OneHotEncoder` treats them via `handle_unknown="ignore"` (no column generated). Effectively, a NaN Sector contributes a zero one-hot row — same numeric effect as an "Unknown" bucket.

**ML.NET.** Empty string is read as empty string, not NaN. `OneHotEncoding` would create an empty-string category.

**Reconciliation.** A `CustomMapping` transform inserted *before* `OneHotEncoding` rewrites empty or whitespace `Sector` to literal `"Unknown"`. This makes the unknown bucket a deliberate, named training-time category rather than relying on downstream NaN handling.

## 5. F1-optimal threshold

**Python (`logistic.py`).** Sweeps the PR curve, picks `argmax F1`, reports confusion matrices at both `0.5` and the best-F1 threshold.

**ML.NET (`Metrics/BinaryMetrics.cs`).** Same sweep, inside `BinaryMetrics.Compute`. Stored in metrics JSON as `bestThreshold`, `f1AtBest`, `confusionAtBest`. The 0.5 default is recorded separately as `f1At05`, `confusionAt05`.

## 6. PR-AUC computation

**Python.** `sklearn.metrics.average_precision_score` — the "average precision" definition: $\sum_n (R_n - R_{n-1}) P_n$ (the step-function area, not the trapezoid).

**ML.NET.** `BinaryClassification.Evaluate` returns `AreaUnderPrecisionRecallCurve` which uses a trapezoid approximation. **These differ by O(1/n) — significant on highly imbalanced data.**

**Reconciliation.** `BinaryMetrics.Compute` implements the step-function average precision identically to sklearn:

```csharp
prAuc += (recl - prevRecall) * prec;
```

Same recurrence, same accumulator. Bypasses ML.NET's built-in PR-AUC. Verified empirically on the Y_Oracle sanity baseline (where AP ≈ 1.0 if both compute it correctly).

## 7. Random seed

Both branches use `seed: 42` throughout. In ML.NET:

```csharp
var ml = new MLContext(seed: 42);
StratifiedSplit.Split(..., seed: 42);
StratifiedKFold.Folds(..., seed: 42);
```

`MLContext(seed: 42)` seeds the trainer-internal RNG; the manual `Random(42)` seeds the partition shuffles. Both are needed for full reproducibility.

## 8. PCA loadings + explained variance

**Python.** `PCA().fit(X).explained_variance_ratio_`, `.components_` — directly available on the fitted estimator.

**ML.NET.** `ProjectToPrincipalComponents` returns the projected data but does *not* expose components or eigenvalues.

**Reconciliation.** `PcaPipeline.Run` builds the EstimatorChain (standardise → ProjectToPrincipalComponents) and fits it. It then separately runs `MathNet.Numerics.LinearAlgebra.Matrix.Svd` on the same standardised matrix the chain saw, extracts:

- explained variance ratios from $\sigma_i^2 / (n - 1)$ normalised,
- loadings from $V$ (the right singular vectors).

The training-fold-only invariant holds because the matrix passed to SVD is built from the same `trainReady` list `chain.Fit` saw — there is no separate fit on the full dataset (see `MLNetLeakageAudit.md` §3).

## 9. Stratified K-fold partition strategy

**sklearn.** `StratifiedKFold` sorts by class hash then partitions into contiguous slices. The exact partition is opaque.

**ML.NET (`StratifiedKFold.cs`).** Shuffles each class with a seeded RNG, then assigns row $i$ within that class to fold $(i \bmod k)$. Equivalent class-proportion outcome, easier to read.

Test (`StratifiedKFoldTests`) asserts every fold has ≥1 positive and that folds partition the input — independent of the algorithmic choice.

## 10. Solver — `lbfgs` in both

**sklearn.** `LogisticRegression(solver='lbfgs')`.

**ML.NET.** `LbfgsLogisticRegressionBinaryTrainer`.

Same algorithm family, but the convergence criteria differ:

- sklearn default: `tol=1e-4`, `max_iter=100`.
- ML.NET default: `OptimizationTolerance=1e-7`, `MaximumNumberOfIterations=int.MaxValue`.

`LogisticTrainer.cs` sets `MaximumNumberOfIterations = 200` to keep training bounded. This is the most likely source of small numeric differences between the two branches' fitted coefficients — both solvers find the same global optimum (L2-regularized logistic loss is convex), but they stop at slightly different points.

## Summary table

| Item | sklearn knob | ML.NET equivalent | File |
|---|---|---|---|
| Regularization | `C` (inverse) | `L2Regularization = 1/C` | `LogisticTrainer.cs` |
| Class weights | `class_weight='balanced'` | `WeightedRow` + `ExampleWeightColumnName` | `ClassWeights.cs` |
| Median impute | `fillna(median())` | manual on training fold | `MedianImputer.cs` |
| Sector NaN | `handle_unknown="ignore"` | CustomMapping → "Unknown" | `PreprocessingPipeline.cs` |
| F1-opt threshold | manual sweep | manual sweep | `BinaryMetrics.cs` |
| PR-AUC | step-function AP | step-function AP (manual) | `BinaryMetrics.cs` |
| Seed | `random_state=42` | `MLContext(seed:42)` + `Random(42)` | everywhere |
| PCA loadings | `.components_` | MathNet SVD | `PcaPipeline.cs` |
| K-fold partition | sklearn internal | round-robin per class | `StratifiedKFold.cs` |
| Solver iter cap | `max_iter=100` | `MaximumNumberOfIterations=200` | `LogisticTrainer.cs` |
| GBT leaves | `max_leaf_nodes` | `NumberOfLeaves` | `GradientBoostedTreesTrainer.cs` |
| GBT trees | `n_estimators` | `NumberOfTrees` | `GradientBoostedTreesTrainer.cs` |
| RF feature frac | `max_features='sqrt'` | `FeatureFraction=0.7` | `RandomForestTrainer.cs` |
| Elastic Net L1 | `l1_ratio * (1/C)` | `L1Regularization` (direct) | `ElasticNetTrainer.cs` |
| Elastic Net L2 | `(1-l1_ratio) * (1/C)` | `L2Regularization` (direct) | `ElasticNetTrainer.cs` |
| Uncalibrated prob | sklearn always calibrates | Score used as proxy | `BinaryMetrics.cs` |
| Champion selection | `best_estimator_` | `RunAllSupervised` CV leaderboard | `MLnetPipeline.cs` |

## 11. GBT — `FastTreeBinaryTrainer` vs sklearn `GradientBoostingClassifier`

**sklearn.** `GradientBoostingClassifier(n_estimators=..., learning_rate=..., max_leaf_nodes=...)`.

**ML.NET.** `FastTreeBinaryTrainer.Options { NumberOfTrees, LearningRate, NumberOfLeaves }`.

**Reconciliation.** `NumberOfTrees ≡ n_estimators` (number of boosting rounds), `LearningRate ≡ learning_rate` (shrinkage multiplier per tree), `NumberOfLeaves ≡ max_leaf_nodes`. Note that sklearn's `max_depth` controls the tree structure via recursion depth while ML.NET's `NumberOfLeaves` controls it via maximum leaf count — they're equivalent controls on the same axis (higher = more complex tree). Set `NumberOfLeaves ∈ {20, 31}` to span sklearn's common defaults.

**Preprocessing note.** `NormalizeMeanVariance` applied for schema consistency but is a no-op for tree models: axis-aligned split decisions are invariant to monotone feature scaling. Both branches normalize; the normalization has no effect on GBT/RF but doesn't hurt.

## 12. Random Forest — `FastForestBinaryTrainer` vs sklearn `RandomForestClassifier`

**sklearn.** `RandomForestClassifier(n_estimators=..., max_leaf_nodes=..., max_features='sqrt')`.

**ML.NET.** `FastForestBinaryTrainer.Options { NumberOfTrees, NumberOfLeaves, FeatureFraction }`.

**Reconciliation.** `NumberOfTrees ≡ n_estimators`, `NumberOfLeaves ≡ max_leaf_nodes`. `FeatureFraction = 0.7` approximates sklearn's `max_features='sqrt'` (for 15 features, sqrt(15)/15 ≈ 0.26; FastForest's default of 0.7 is more aggressive feature sharing — matches LightGBM-style training more than sklearn's conservative default). Reported in `rf_{target}_metrics.json` for transparency.

**Calibration.** sklearn's `RandomForestClassifier.predict_proba` is calibrated via Platt scaling. FastForest is *uncalibrated* by default — "Probability" column is absent in scored output. `BinaryMetrics.Compute` falls back to "Score" as the probability proxy (§ below), which produces valid ROC/PR curves at the cost of miscalibrated probability magnitudes. This is acceptable for ranking and AUC computation; it would matter for downstream Brier-score or calibration analysis.

## 13. Elastic Net — `SdcaLogisticRegressionBinaryTrainer` vs sklearn `LogisticRegression(penalty='elasticnet')`

**sklearn.** `LogisticRegression(penalty='elasticnet', solver='saga', C=c, l1_ratio=ρ)`.

The sklearn objective:
$$\min_w \frac{1}{n}\sum_i \ell(y_i, w \cdot x_i) + \frac{1}{C} \Big[\rho \|w\|_1 + \frac{1-\rho}{2}\|w\|_2^2\Big]$$

**ML.NET.** `SdcaLogisticRegressionBinaryTrainer.Options { L1Regularization, L2Regularization }`.

The ML.NET objective:
$$\min_w \frac{1}{n}\sum_i \ell(y_i, w \cdot x_i) + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2$$

**Reconciliation.** Direct mapping: $\lambda_1 = \rho/C$ and $\lambda_2 = (1-\rho)/(2C)$. The grid searches over $(\lambda_1, \lambda_2)$ directly rather than $(\rho, C)$ — this avoids the interaction between two hyperparameters and is equivalent. Choosing $(\lambda_1, \lambda_2) \in \{0.001, 0.01, 0.1\}^2$ covers a 100× range in both penalties and includes the corners $(\lambda_1=0, \lambda_2>0)$ (pure ridge) and $(\lambda_1>0, \lambda_2=0)$ (pure lasso) approximately.

## 14. Linear Regression — poor-fit demonstration

**sklearn equivalent.** `LinearRegression()` or `Ridge()` fit on binary $\{0,1\}$ targets.

**ML.NET.** `SdcaRegressionTrainer.Options { LabelColumnName = "FloatLabel", ExampleWeightColumnName = "Weight", L2Regularization }`.

**Label type.** Binary trainers use `bool Label`; regression trainers require `float`. `MLReadyRow` carries both: `Label: bool` for classifiers, `FloatLabel: float` (set by `MedianImputer.Apply` as `label ? 1f : 0f`) for regression.

**Probability proxy.** Regression "Score" is used as the probability for ROC/PR computation (`BinaryMetrics.Compute` Score fallback). The key failure diagnostic is `fractionOutsideUnit` — the fraction of test scores outside $[0,1]$. This is impossible for logistic/GBT/RF (they produce probabilities by construction) and systematically non-zero for linear regression on imbalanced data (the hyperplane does not respect the unit interval).

**Champion selection.** `LinearRegressionTrainer` participates in the CV leaderboard but is excluded from champion selection (only classifiers compete). Its test metrics are always emitted for comparison, clearly labelled `"modelType": "regression_demonstration"`.

## 15. Calibration + `BinaryMetrics.Compute` fallback

FastTree (GBT) applies Platt calibration internally, producing a "Probability" column. FastForest (RF) does not — the scored output has only "Score". `BinaryMetrics.Compute(MLContext, IDataView)` checks for the "Probability" column; if absent, it sets `Probability = Score` before sweeping the curve. This fallback:

- Produces valid ROC and PR curves (ordinal ranking is preserved regardless of calibration).
- Produces valid AUC metrics (AUC is a ranking statistic, not a calibration statistic).
- Produces **incorrect** probability estimates at a given threshold (the Score scale is not $[0,1]$ bounded for regression; for RF it is $[0,1]$ bounded but miscalibrated).

Confusion matrices and F1 at threshold 0.5 are thus meaningful for RF (Score is in $[0,1]$) but not meaningful for linear regression (Score can exceed those bounds). The `fractionOutsideUnit` field in the linreg JSON makes this explicit.

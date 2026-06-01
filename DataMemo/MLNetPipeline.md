# MLNetPipeline — design from `LotStateVector` outward

This memo describes the **ML.NET v0.1 layer** that lives on branch `feature/MLnet-layer`. It is an *alternate* to the Python `lots_pipeline/` on `feature/ML-layer`, not a replacement. The two branches are designed to be checked out interchangeably and compared run-for-run.

The Python pipeline is documented in `MLPipeline.md` on the other branch; this memo's job is to explain **what changes, what stays, and *why* the change is principled rather than aesthetic.**

## The mental model

```
LotStateVector  (C# record, already typed)
       │
       ▼   LoadFromEnumerable<LotStateVector>
IDataView  (named, typed columns — no schema loss, no CSV round-trip)
       │
       ▼   EstimatorChain
            ├─ CustomMapping  (Sector NaN → "Unknown")
            ├─ OneHotEncoding (SectorClean → SectorOneHot)
            ├─ NormalizeMeanVariance (15 numeric features by name)
            └─ Concatenate("Features", numerics + SectorOneHot)
       │
       ▼   Trainer
            LbfgsLogisticRegression(L2 = 1/C, weight = "Weight")
       │
       ▼   ITransformer   (fitted, inspectable at every stage)
       │
       ▼   BinaryClassification.Evaluate + manual PR-AUC + silhouette
       │
       ▼   JSON metrics + ROC/PR/scree/elbow curve points
       │
       ▼   PythonRunner subprocess
            scripts/eda.py     → EDA PNGs from lots.csv
            scripts/render.py  → model PNGs + index.html from JSON
```

Every arrow is typed. Nothing is positional. The schema declared in [`LotStateVector.cs`](../src/Core/Portfolio/LotStateVector.cs) is the single source of truth — there is no separate `[LoadColumn]` projection, no pandas DataFrame intermediary, no point at which a column name is reduced to a positional index.

## Mathematical derivations in `LotStateVector` space

Let each row be
\[
v_i=(x_i,\; z_i,\; y_i^{\text{oracle}},\; \tilde y_i^{\text{BT}})
\]
where:
- \(x_i\in\mathbb{R}^{15}\) are the numeric fields in `FeatureLists.NumericFeatures` (all sourced from `LotStateVector`),
- \(z_i\in\{1,\dots,m\}\) is sector, one-hot encoded to \(e(z_i)\in\{0,1\}^m\),
- the final model input is \(\phi_i=[\operatorname{Norm}(x_i), e(z_i)]\in\mathbb{R}^{15+m}\).

Targets are induced directly from `LotStateVector` labels:
\[
y_i=
\begin{cases}
Y\_\text{Oracle}(i)\in\{0,1\}, & \texttt{target="oracle"}\\
\mathbf{1}\!\left[\tilde y_i^{\text{BT}}>0\right], & \texttt{target="soft\_bt"}
\end{cases}
\]
with NaN \(\tilde y_i^{\text{BT}}\) rows removed before split/CV.

### Supervised models (implemented)

- **Logistic (`LbfgsLogisticRegression`)**  
  \[
  \hat p_i=\sigma(w^\top\phi_i+b),\quad
  \min_{w,b}\;\sum_i \alpha_i\Big[-y_i\log \hat p_i-(1-y_i)\log(1-\hat p_i)\Big]+\lambda\|w\|_2^2
  \]
  where \(\alpha_i\) are class-balance weights and \(\lambda=1/C,\; C\in\{0.01,0.1,1,10\}\).

- **Gradient-boosted trees (`FastTree`)**  
  Additive model \(F_M(\phi)=\sum_{m=1}^M \nu\,f_m(\phi)\), where each \(f_m\) is a regression tree fit to current pseudo-residuals of logistic loss.

- **Random forest (`FastForest`)**  
  Ensemble \(F(\phi)=\frac1T\sum_{t=1}^T f_t(\phi)\) over bagged trees with random feature sub-sampling; class probabilities come from averaged tree votes/scores.

- **Elastic net (`SdcaLogisticRegression`)**  
  Logistic loss with mixed penalty:
  \[
  \min_{w,b}\;\sum_i \ell_{\log}(y_i,w^\top\phi_i+b)+\lambda_1\|w\|_1+\lambda_2\|w\|_2^2
  \]

- **Linear regression demo (`Sdca`)**  
  \[
  \hat y_i=w^\top\phi_i+b,\quad
  \min_{w,b}\;\sum_i(y_i-\hat y_i)^2+\lambda\|w\|_2^2
  \]
  included as an intentionally misspecified baseline for binary targets.

### Unsupervised models (implemented)

- **PCA on numeric `LotStateVector` subspace**  
  With standardized matrix \(X\in\mathbb{R}^{n\times 15}\), covariance
  \[
  C=\frac{1}{n-1}X^\top X.
  \]
  Eigenpairs \((\lambda_j, u_j)\) of \(C\) give principal axes \(u_j\) and explained-variance ratios \(\lambda_j/\sum_k\lambda_k\). Keep smallest \(r\) with cumulative variance \(\ge 0.95\).

- **K-means on per-symbol aggregates**  
  For each symbol \(s\), aggregate four asset-level coordinates from `LotStateVector`: \((R_t,\Sigma_{\text{Range}},\Delta MA50,\Delta MA200)\), then standardize and solve
  \[
  \min_{\{\mu_c\},\{a_s\}}\sum_s\left\|g_s-\mu_{a_s}\right\|_2^2,\quad a_s\in\{1,\dots,k\}.
  \]
  Choose \(k\in\{5,10,15,20,25\}\) by maximal silhouette score.

## What changes vs Python

| Stage | Python (`lots_pipeline/`) | ML.NET (`src/ML/CSharp/MLNet/`) | Why the change is principled |
|---|---|---|---|
| **Schema** | implicit in `pd.read_csv` + `NUMERIC_FEATURES` list | explicit in `LotStateVector` record + `FeatureLists.cs` | C# has the type system Python lacks. Re-encoding sklearn's "schema-by-convention" workaround in a language with real records is anti-pattern. |
| **Loading** | CSV → DataFrame → `.values` (positional ndarray) | `LotStateVector` → `LoadFromEnumerable` → `IDataView` (named columns) | No CSV inside the ML layer. The data structure that produced `lots.csv` IS the data structure the trainer sees. |
| **Stratified split** | `train_test_split(..., stratify=y)` (sklearn knob) | `StratifiedSplit.Split(list, labelSelector, ...)` (visible 50-line function) | The partition logic is the thing being studied. Visible function > hidden parameter. |
| **K-fold** | `StratifiedKFold(n_splits=5)` (sklearn knob) | `StratifiedKFold.Folds(...)` (visible 60-line function, round-robin per class) | Same reason. |
| **Grid search** | `GridSearchCV(...)` (sklearn black box) | `GridSearchCV.Search(grid, factory, scorer, ...)` (visible nested loop) | A hyperparameter configuration is *just a function from a parameter dict to an estimator* — the factory pattern makes that explicit. |
| **Preprocessing** | `ColumnTransformer([("num", StandardScaler(), ...), ("cat", OneHotEncoder(), ...)])` | typed `EstimatorChain` with named-column input/output at every stage | `ColumnTransformer` exists because sklearn pipelines drop column names. ML.NET pipelines don't. |
| **Class balancing** | `class_weight='balanced'` (sklearn does it inside `.fit()` — correctly, by accident) | `ClassWeights.AttachBalancedWeights(trainFold, ...)` — only signature available | Makes the training-fold-only requirement *structurally enforced* rather than relying on the library to get the order right. See `MLNetLeakageAudit.md`. |
| **Median impute** | `df[c].fillna(df[c].median())` on full dataset (subtle leak) | `MedianImputer.Fit(trainFold) → dict; Apply(rows, dict)` | Same reason. The two-call shape makes "fit on train, apply elsewhere" the natural path. |
| **PCA** | `PCA().fit(df[NUMERIC].dropna().values)` on full dataset (subtle leak) | `EstimatorChain.Fit(trainView)` (training fold only) + MathNet SVD on same matrix for loadings | EstimatorChain shape forces the fit-on-train invariant. ML.NET's PCA transform hides loadings, so MathNet recovers them from the same training matrix the chain saw. |
| **K-means k selection** | `silhouette_score` on per-symbol aggregate, pick max | `SilhouetteScore.Compute` on per-symbol aggregate, pick max | No change — semantics are identical; just hand-written rather than imported. |
| **Plots** | matplotlib inline | C# emits JSON → `scripts/render.py` reads JSON → matplotlib | Visualization iteration is genuinely better in Python's REPL. The boundary stays. |
| **Report** | n/a (Python produced inline) | `scripts/render.py` → `index.html` via Jinja2 | Pure templating, zero ML logic in Python. |

## What stays in Python

- **EDA plotting** — `scripts/eda.py` reads `data/lots.csv` and emits class balance, correlation heatmap, feature distributions, missing-data summary.
- **Model plotting + HTML stitching** — `scripts/render.py` reads `data/artifacts-mlnet/*.json` and emits ROC/PR, scree, elbow PNGs + `index.html`.

Together ≈ 200 LOC. Zero sklearn. Zero ML logic. Pure pandas + matplotlib + jinja2.

## Output artifacts

`data/artifacts-mlnet/` (C# writes):
- `pca_components.json`, `pca_loadings.csv`, `pca_scree.json`, `pca_model.zip`
- `kmeans_centers.json`, `cluster_assignments.json`, `cluster_assignments.csv`, `kmeans_elbow.json`
- `logistic_{oracle,soft_bt}_metrics.json` (CV scores, test ROC/PR-AUC, F1 at both thresholds, ROC + PR curve points, best `C`, `l2_used`)
- `logistic_{oracle,soft_bt}_model.zip`, `logistic_{oracle,soft_bt}_coefficients.csv`

`src/Export/eda-mlnet/`, `src/Export/models-mlnet/` (Python writes):
- EDA PNGs + summary.json, model PNGs, `index.html`

## Related memos

- **`MLNetVsPython.md`** — the architectural argument (schema-first vs estimator-API).
- **`MLNetLeakageAudit.md`** — the training-fold-only invariants made structural.
- **`MLNetSemanticReconciliation.md`** — every place ML.NET defaults differ from sklearn (`C` vs `L2`, class-weight, median impute, etc.).

For the lot/feature codebook itself, see [`Codebook.md`](Codebook.md) on `feature/ML-layer` — the schema is identical across branches.

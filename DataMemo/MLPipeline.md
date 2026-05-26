# ML Pipeline — Architecture & Standard Definitions

This memo documents the ML modeling layer (Layer 3) that sits on top of the simulation
layer's output (`data/lots.csv`).  It covers the file structure, the C#/Python boundary,
the supervised/unsupervised targets, and the standard ML definitions used throughout
the codebase.

For mathematical derivations of each algorithm (logistic regression, PCA, K-means),
see [`MLDerivations.md`](MLDerivations.md).

---

## §1  Architecture Overview

```
[ C# Simulation ]                                       [ C# Simulation v0.3 — future ]
     │ produces                                              ▲ reads cluster_id
     ▼                                                       │ at harvest time
data/lots.csv ──► [ src/ML/Python/ ] ──► data/artifacts/* ──┘
                       │  EDA       ─►  src/Export/eda/*.png
                       │  PCA       ─►  pca_loadings.csv, pca_components.json
                       │  K-means   ─►  cluster_assignments.json, kmeans_centers.json
                       │  LR(softBT)─►  logistic_soft_bt_{model.joblib, metrics.json, coefficients.csv}
                       └  LR(oracle)─►  logistic_oracle_{...}      (sanity baseline)
                                       + ROC/PR curves → src/Export/models/*.png
```

### File layout (after the v0.1 restructure)

```
FinalProject/
├── DataMemo/                         # all documentation (math, ML, codebook)
│   ├── PortfolioMath.md
│   ├── SimulationMath.md
│   ├── MLPipeline.md                 # this doc
│   ├── MLDerivations.md              # per-algorithm math derivations
│   └── Codebook.md                   # variable descriptions for the lots.csv schema
│
├── src/                              # C# project root (also contains the Python runtime)
│   ├── Core/                         # Oracle, Portfolio, Simulation
│   ├── DataCollection/               # MarketDataDownloader.cs, Models.cs
│   ├── Export/                       # ALL export artifacts (code + image outputs)
│   │   ├── SimulationExporter.cs     # C# writer for lots.csv
│   │   ├── eda/                      # PNGs from EDA + PCA/K-means tuning curves
│   │   └── models/                   # ROC/PR curves from logistic regression
│   ├── ML/                           # ML sub-application
│   │   ├── CSharp/                   # C# side
│   │   │   └── PythonRunner.cs       # Process.Start wrapper, streams stdout/stderr
│   │   └── Python/                   # Python side (uv-managed package)
│   │       ├── pyproject.toml        # deps: pandas, numpy, sklearn, matplotlib, joblib
│   │       ├── uv.lock
│   │       ├── lots_pipeline/        # importable package
│   │       │   ├── io.py             # load lots.csv, save artifacts
│   │       │   ├── targets.py        # soft_bt / oracle target derivation
│   │       │   ├── split.py          # stratified train/test split
│   │       │   ├── preprocessing.py  # ColumnTransformer (scale + one-hot)
│   │       │   ├── eda.py            # class balance, corr heatmap, dist plots
│   │       │   ├── pca.py            # PCA fit + scree plot + loadings export
│   │       │   ├── kmeans.py         # per-symbol K-means with elbow tuning
│   │       │   └── logistic.py       # LR + GridSearchCV(StratifiedKFold)
│   │       ├── scripts/              # CLI entry points (called by PythonRunner.cs)
│   │       │   ├── run_eda.py
│   │       │   ├── train_unsupervised.py
│   │       │   └── train_supervised.py
│   │       └── tests/test_smoke.py
│   ├── Tests/                        # C# unit tests
│   ├── DirectIndexing.csproj         # excludes ML/Python/** from compilation
│   └── Program.cs                    # ml-eda, ml-unsupervised, ml-supervised, ml-baseline, ml-all
│
├── data/                             # ALL contents are temp, re-generable
│   ├── raw/                          # downloaded JSON price files
│   ├── constituents.json
│   ├── lots.csv                      # C# simulation output → Python input
│   ├── lots-mc.csv                   # MC simulation output
│   └── artifacts/                    # Python output (joblib, JSON, CSV)
│
├── DirectIndexing.sln
└── README.md
```

The Python runtime (`src/ML/Python/`) lives inside the `src/` tree so the entire
project is self-contained under one root, but `DirectIndexing.csproj` explicitly
excludes that subtree from C# compilation (`<Compile Remove="ML/Python/**" />`),
so the .NET build doesn't touch any Python files.

### C# / Python boundary

The boundary is **file-based**.  C# writes `lots.csv`; Python reads it, produces all
artifacts (JSON, CSV, joblib, PNG), and writes them under `data/artifacts/` (model
artifacts, regenerable) and `src/Export/{eda,models}/` (plot exports).
C# orchestrates via [`PythonRunner.cs`](../src/ML/CSharp/PythonRunner.cs), which shells
out to `uv run python -m <module>` (with CWD set to `src/ML/Python/`) and streams
stdout/stderr to the host console.

No in-memory data passing.  This keeps the integration trivial to debug and means each
side can be exercised in isolation (run Python standalone, or only C# without ML).

### Orchestration via Program.cs

```bash
dotnet run --project src -- ml-eda             # EDA plots → Results/eda/
dotnet run --project src -- ml-unsupervised    # PCA + K-means → data/artifacts/
dotnet run --project src -- ml-supervised      # LR on Y_Soft_BT (PRIMARY)
dotnet run --project src -- ml-baseline        # LR on Y_Oracle (sanity check)
dotnet run --project src -- ml-all             # chain all of the above
```

---

## §2  Target Definitions

Three label columns are produced by the simulation; only **two** are used as ML targets.

| Column | Type | Source | Used as ML target? |
|--------|------|--------|---------------------|
| `Y_Oracle`   | binary {0,1}   | `OracleBoundary.Label` — deterministic Boolean of features | Yes — **sanity baseline only** |
| `Y_Soft_BT`  | float [0,1]    | Fraction of next 30 actual trading days the oracle would fire | Yes — **primary target** (thresholded > 0) |
| `Y_Soft_GBM` | float [0,1]    | Fraction of 200 GBM-simulated paths the oracle would fire | No (v0.1) — reserved for future use |

### Why `Y_Soft_BT` is the primary target

`Y_Soft_BT` is the genuinely uncertain quantity given current features.  It depends on
**future realised prices** which are not fully encoded in the current snapshot, so a
model approximating it has real predictive value.  Thresholding at `> 0` produces a
binary "will the oracle fire at all in the next 30 trading days?" classification target
with positive rate ≈ 8% (vs Y_Oracle's ≈ 0.8%).

### Why `Y_Oracle` is only a sanity baseline

`Y_Oracle` is the **mechanistic** label:

$$
Y^{\text{Oracle}}_{k,t} = \mathbf{1}[\ell_k \le -0.02] \cdot \mathbf{1}[\sigma_{\text{TE},t} \le 0.05] \cdot \mathbf{1}[G^{\text{YTD}}_t > 0] \cdot \mathbf{1}[\mathcal{W}_{k,t} \ge 30]
$$

— a deterministic Boolean function of the features that are in the row itself.  Any
sufficiently expressive model can recover it exactly.  Approximating it provides no
**learning** value, but the test ROC AUC on Y_Oracle is a useful pipeline diagnostic:
if a non-linear model (decision tree, random forest) doesn't achieve ROC AUC ≈ 1.0 on
the held-out set, something is broken.

For the **linear** logistic regression baseline (v0.1), Y_Oracle is harder to fit
than it looks: the oracle is a multiplicative AND of four indicator functions, which
is non-linear in the original feature space.  LR can only approximate it (we observe
ROC AUC ≈ 0.98).  Tree-based models in v0.2 should approach 1.0.

---

## §3  Standard ML Definitions

Quick reference for the metrics and procedures used in this layer.  Mathematical
derivations live in `MLDerivations.md`.

### Train/test split

A held-out test set is reserved before any training or hyperparameter tuning.
Used: **80/20 split**, `random_state=42`.

### Stratified sampling

Sampling that preserves the class proportions of the outcome variable across the
split.  Critical when the positive class is rare:

- `Y_Oracle` (≈ 0.8% positive): a naive split could leave the test fold with single-digit positives.
- `Y_Soft_BT-binary` (≈ 8% positive): still benefits — fold-to-fold variance shrinks.

Implementation: `train_test_split(stratify=y)` and `StratifiedKFold` for CV.

### k-fold cross-validation

The training set is partitioned into k equal folds.  For each of k iterations, k-1
folds are used for fitting and 1 fold is held out for scoring; the k held-out scores
are averaged.  Used: **k = 5** with stratification on the outcome.

### Confusion matrix

|                   | Predicted 0 | Predicted 1 |
|-------------------|-------------|-------------|
| **Actual 0**      | TN          | FP          |
| **Actual 1**      | FN          | TP          |

- **Precision** = TP / (TP + FP) — of predicted positives, how many are real
- **Recall** (Sensitivity, TPR) = TP / (TP + FN) — of real positives, how many were caught
- **F1**       = 2 · (P · R) / (P + R) — harmonic mean of precision and recall
- **Specificity** (TNR) = TN / (TN + FP)

### ROC curve & ROC-AUC

The ROC curve plots TPR vs FPR as the classification threshold sweeps from 1 → 0.
**ROC-AUC** is the area under that curve; 0.5 = random, 1.0 = perfect.  Equivalent to
the probability that a random positive scores higher than a random negative.

### Precision-Recall curve & PR-AUC (Average Precision)

The PR curve plots precision vs recall as threshold sweeps.  **PR-AUC** is more
informative than ROC-AUC for imbalanced data: ROC-AUC stays high even when precision
is poor (because TN dominates), while PR-AUC penalises false positives correctly.

Baseline PR-AUC = positive rate.  For `Y_Oracle` the random baseline is ≈ 0.008.

### Class weighting

When positive class is rare, the unweighted log-loss is dominated by easy negatives.
Setting `class_weight='balanced'` re-scales each class's contribution to the loss
inversely proportional to its frequency, so the model spends comparable effort on
both classes.  See `MLDerivations.md` §1 for the formal derivation.

### Threshold tuning

The default 0.5 decision threshold rarely maximises F1 under class imbalance.
After fitting, we sweep thresholds across the PR curve and report the F1-optimal
threshold alongside metrics at default 0.5.

### Hyperparameter, grid, refit

A **hyperparameter** is a configuration value not learned from data (e.g. `C` for LR,
`n_clusters` for K-means).  A **grid** is a discrete set of candidate values.
**Refit-on-best**: after CV identifies the best config, refit on the full training
set with that config before evaluating on the test set.

### Multicollinearity

When two or more predictors are highly correlated, individual coefficient estimates
become unstable (high variance) even if predictive power as a whole is preserved.
This motivates PCA (orthogonal projection) and regularisation (L2 shrinkage).

---

## §4  Hyperparameter Grids Used

### LogisticRegression (v0.1)

| Parameter         | Grid                          | Fixed     |
|-------------------|-------------------------------|-----------|
| `C` (inverse λ)   | {0.01, 0.1, 1.0, 10.0}        | grid      |
| `penalty`         | L2 (default in sklearn 1.8+)  | fixed     |
| `class_weight`    | `'balanced'`                  | fixed     |
| `solver`          | `'lbfgs'`                     | fixed     |
| `max_iter`        | 2000                          | fixed     |

CV: `StratifiedKFold(n_splits=5)`.  Scoring: `average_precision` (PR-AUC).

### K-means (v0.1)

| Parameter      | Grid                  | Selection criterion |
|----------------|-----------------------|----------------------|
| `n_clusters`   | {5, 10, 15, 20, 25}   | max silhouette       |
| `n_init`       | 10                    | fixed                |
| `random_state` | 42                    | fixed                |

### PCA (v0.1)

| Parameter             | Value                                    |
|-----------------------|------------------------------------------|
| `n_components`        | full = 15, then truncate to k for 95% cumulative variance |
| `svd_solver`          | `'full'`                                 |
| Input standardisation | `StandardScaler` (mandatory)             |

---

## §5  Codebook & Symbol Glossary

Variable names match the C# `LotStateVector` record exactly.  See
[`data/codebook.md`](../data/codebook.md) for the full per-column description.

| Symbol | Code name      | Domain        | Origin         |
|--------|----------------|---------------|----------------|
| ℓ      | `L`            | float         | unrealized return of lot |
| H      | `H`            | int           | holding period (days)    |
| S      | `S`            | {0, 1}        | long-term flag (≥365 days) |
| p_k    | `B`            | float         | cost basis per share     |
| W      | `W`            | float [0, 1]  | lot's weight in portfolio |
| K      | `K`            | int           | number of open lots of same symbol |
| G_YTD  | `G_YTD`        | float         | net realised gain this year |
| σ_TE   | `Sigma_TE`     | float ≥ 0     | annualised tracking error |
| 𝒲     | `WashClock`    | int           | days since last harvest of this ticker |
| r_t    | `R_t`          | float         | daily return of underlying |
| σ_rng  | `SigmaRange`   | float ≥ 0     | (high − low) / close      |
| Δ_50   | `DeltaMA50`    | float         | deviation from 50-day MA  |
| Δ_200  | `DeltaMA200`   | float         | deviation from 200-day MA |
| α_tax  | `TaxAlpha`     | float ≥ 0     | tax savings if harvested  |
| DaysYE | `DaysToYE`     | int ≥ 0       | trading days to year-end  |

---

## §6  Roadmap

| Version | Scope |
|---------|-------|
| **v0.1** (this iteration) | EDA + PCA + K-means + LR on Y_Soft_BT + LR on Y_Oracle (sanity). Python pipeline + C# orchestration only. |
| v0.2    | Add LDA / QDA, elastic net, decision tree, random forest, gradient boosted trees.  Hyperparameter tuning expanded. Model comparison framework. |
| v0.3    | Wire `cluster_assignments.json` into the C# simulation: at harvest time, pick substitute lot from same K-means cluster to reduce TE during wash-sale window. |
| v0.4    | RMT cleaning of Σ inside TrackingErrorProxy (Marchenko-Pastur eigenvalue filtering). |
| v1.0    | Production inference path: C# loads `logistic_*.joblib` (or ONNX export) at simulation time to drive harvest decisions instead of the deterministic oracle. |

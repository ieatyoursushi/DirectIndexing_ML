# `lots_pipeline` — Python ML runtime

Python side of the ML modeling layer. Consumes `../../../data/lots.csv` (produced by
the C# simulation) and writes:

- **Trained model artifacts** (regenerable) → `../../../data/artifacts/`
- **EDA plots** → `../../Export/eda/`
- **Model evaluation plots** → `../../Export/models/`

See `../../../DataMemo/MLPipeline.md` for architecture and standard ML definitions,
`../../../DataMemo/MLDerivations.md` for mathematical derivations.

## Setup (one-time)

```bash
cd src/ML/Python
uv sync                              # installs deps from pyproject.toml + uv.lock
```

## Standalone usage (from `src/ML/Python/`)

```bash
uv run python -m scripts.run_eda \
    --in ../../../data/lots.csv --out ../../Export/eda/
uv run python -m scripts.train_unsupervised \
    --in ../../../data/lots.csv --out ../../../data/artifacts/ --results ../../Export/eda/
uv run python -m scripts.train_supervised \
    --in ../../../data/lots.csv --out ../../../data/artifacts/ \
    --results ../../Export/models/ --target soft_bt
uv run python -m scripts.train_supervised \
    --in ../../../data/lots.csv --out ../../../data/artifacts/ \
    --results ../../Export/models/ --target oracle
uv run pytest tests/
```

## Orchestrated via C# (from repo root)

```bash
dotnet run --project src -- ml-eda          # EDA  → src/Export/eda/
dotnet run --project src -- ml-unsupervised # PCA + K-means → data/artifacts/ + src/Export/eda/
dotnet run --project src -- ml-supervised   # LR on Y_Soft_BT (primary)
dotnet run --project src -- ml-baseline     # LR on Y_Oracle (sanity)
dotnet run --project src -- ml-all          # chain all of the above
```

## Package layout

```
src/ML/Python/
├── pyproject.toml                # uv project file (deps + dev-deps)
├── uv.lock
├── lots_pipeline/                # importable package
│   ├── io.py                     # load lots.csv + JSON/CSV/model writers
│   ├── targets.py                # soft_bt / oracle target derivation
│   ├── split.py                  # stratified split + feature matrix
│   ├── preprocessing.py          # ColumnTransformer (scale + one-hot)
│   ├── eda.py                    # class balance / corr / dist / missing
│   ├── pca.py                    # PCA + scree + loadings
│   ├── kmeans.py                 # per-symbol K-means + elbow tuning
│   └── logistic.py               # LR + GridSearchCV(StratifiedKFold)
├── scripts/                      # CLI entry points (called by PythonRunner.cs)
│   ├── run_eda.py
│   ├── train_unsupervised.py
│   └── train_supervised.py
└── tests/
    └── test_smoke.py             # end-to-end on synthetic 2K-row frame
```

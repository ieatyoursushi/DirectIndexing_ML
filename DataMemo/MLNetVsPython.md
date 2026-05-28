# MLNet vs Python — the architectural argument

Why this project has *two* ML layers — and why the C# / ML.NET version is not a port of the Python one but a different design.

## The thing sklearn was designed around

sklearn's estimator API was designed for **rapid experimental iteration in a notebook**. The contract is intentionally minimal: every model exposes `.fit(X, y)` and `.predict(X)`; every transform exposes `.fit_transform`. `X` is a NumPy array, which has no enforced schema. Column names exist only as long as you keep a DataFrame around; the moment you call `.values`, they vanish and the model sees positional indices.

This is a *correct* design for its use case — research and prototyping where the developer iterates 50 times an hour and doesn't want a type system in the way. It is *not* a good fit when:

- the data has a stable schema you already control (`LotStateVector` here),
- you want stage-by-stage output schemas to be inspectable,
- you want the difference between "the thing that learns" and "the thing that transforms" visible in the type system,
- the partition / weighting / impute order matters and you'd rather have the compiler enforce it than rely on the library doing it inside `.fit()`.

## What ML.NET offers that's actually different

### Schema-first, not workaround

The Python pipeline reads `data/lots.csv` via `pd.read_csv` because Python has no access to the C# type that produced it. ML.NET can read directly from the producing record:

```csharp
var data = mlContext.Data.LoadFromEnumerable(snapshots);
// snapshots: IEnumerable<LotStateVector>
// schema is the record's field declaration; no inference, no LoadColumn ordering
```

The `IDataView` that emerges carries every column's *name* and *type* through every stage. The Concatenate("Features", ...) call references columns by name, not by position. If the underlying record changes (`LotStateVector` gains a field) and the chain still references the old name, the compiler catches it before runtime.

### `IEstimator<T>` vs `ITransformer` — the right type-level split

sklearn collapses two distinct mathematical objects into one Python class:

- the function from data to a model: `D → M` (an unfitted estimator)
- the function from data to data: `X → X′` (a fitted transformer)

ML.NET separates them:

- `IEstimator<TTransformer>` — "I haven't seen any data yet; call `.Fit(data)` and I'll produce a `TTransformer`."
- `ITransformer` — "I've already seen the training data; call `.Transform(data)` and I'll produce transformed output."

This is the type-level distinction between "training" and "inference". Once you've seen it expressed in a type system you stop conflating them mentally. A `TransformerChain<ITransformer>` is an *applied* pipeline; an `EstimatorChain<TTransformer>` is the unfitted recipe. The fitted thing carries the parameters; the unfitted thing is just a description of how to compute them.

### Stratification as a visible function, not an opaque parameter

```python
# sklearn — stratification is a knob
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
```

```csharp
// ML.NET — stratification is a 50-line function you can read
public static (List<LotStateVector> Train, List<LotStateVector> Test) Split(
    IReadOnlyList<LotStateVector> data,
    Func<LotStateVector, int> labelSelector,
    double testFraction = 0.20,
    int seed = 42)
{
    var byClass = new Dictionary<int, List<LotStateVector>>();
    foreach (var r in data) { ... }   // bucket by class
    foreach (var (_, bucket) in byClass) {
        Shuffle(bucket, rng);
        int nTest = (int)Math.Round(bucket.Count * testFraction);
        for (int i = 0; i < bucket.Count; i++)
            (i < nTest ? test : train).Add(bucket[i]);
    }
    ...
}
```

For someone whose mental model is *"the partition of $\mathcal{D}$ into train and test is a function $f: \mathcal{D} \times (\mathcal{X} \to \{0,1\}) \times [0,1] \to \mathcal{D}^2$"* — the second form *is* that function, written down. The first form is the same function hidden behind a parameter on someone else's API.

This isn't an aesthetic preference. It changes what you can *read* about the pipeline.

## Where Python is *not* replaced

The architecture preserves Python for what Python is genuinely better at:

- **Exploratory visualization** — `scripts/eda.py` and `scripts/render.py`. Matplotlib's REPL feedback cycle for iterating on plot styling is genuinely faster than C#'s compile-edit-run loop. The Python side gets `pandas + matplotlib + jinja2` only — no sklearn, no scipy, no ML logic.
- **HTML report rendering** — Jinja2 templating. Could be done in C# (Razor, etc.) but Jinja2 + matplotlib is the natural shape.

The boundary is **JSON in, PNG/HTML out**. C# does all ML semantics; Python does presentation. The `PythonRunner.cs` subprocess wrapper is the only interop seam.

## On "Python is the standard"

True — for **ML research and reproducing published work**. Not relevant here:

- This is tabular binary classification on a typed domain model owned in C#.
- The target deliverable is a course submission, not a paper.
- v0.1 doesn't need pretrained models, doesn't read other people's code, doesn't reproduce papers.
- The dataset is ~122K rows × 15 features — well within the regime ML.NET was designed for ([the ML.NET design intent](https://learn.microsoft.com/en-us/dotnet/machine-learning/) is explicitly "production deployment of typed tabular ML pipelines").

The "Python is standard" argument applies in regimes this project doesn't touch.

## When this might invert

If the project grows into:
- a neural network training stage (`feature/neural-layer`),
- UMAP / ICA / nonlinear dim reduction,
- reinforcement learning policy layer,

then those stages move *back* to Python via `PythonRunner.cs` — exactly the architecture that's already in place. ML.NET handles tabular; Python handles things only Python can do. The interop seam is designed to support both.

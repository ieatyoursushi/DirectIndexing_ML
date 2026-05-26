# Code Review: Mental Model Sync — Simulation Layer

> DNI (Do Not Implement). This is a read-only conceptual sync confirming or correcting your mental
> models for each file in the simulation layer. No code changes planned.

---

## I. PriceLoader — "dictionaries like 2D arrays with diff datatypes"

**Mostly right, sharpen the indexing model.**

The internal structure is `Dictionary<string, float[]>` — symbol is the key, the value is a
dense float array indexed by `dayIndex` (an integer position in the shared trading calendar).

```
_close["AAPL"][200]  →  AAPL closing price on trading day 200
_close["MSFT"][201]  →  MSFT closing price on trading day 201
```

Think of it as a 2D matrix with named rows (symbols) and integer-indexed columns (days), stored
as a hash map of arrays rather than a true 2D array. The calendar list `_calendar[t]` maps each
integer index back to a `DateOnly`.

**`CreateForTesting`** — your read is correct. It's a static factory that bypasses all JSON/API
deserialization by directly populating the same internal `_close`, `_return`, etc. dictionaries
with synthetic data. The same lookup API (`GetClose`, `DailyReturn`, `HasData`) works identically
in test and production code — that's the design point. Close prices are reconstructed from returns
using a $100 base: `close[t] = close[t-1] * (1 + ret[t])`.

**NaN sentinel:** Any (symbol, day) pair with no real quote is stored as `float.NaN`, not a zero
or a missing key. `HasData(sym, t)` checks for NaN explicitly. This avoids false zero-price reads.

---

## II. SimulationEngine — call-chain and ProcessDay structure

**Correct framing, missing some ProcessDay internals.**

Full call chain:

```
Run(initialPortfolioValue)
 └─ InitializePortfolio(day0=200, totalValue)   ← allocates lots, seeds G_YTD
 └─ for t = 200 … DayCount-1:
     ProcessDay(t)
       ├─ GetClosesDecimal(t)                   ← O(n_symbols) map lookup
       ├─ PortfolioValue = sum(shares × close)  ← for feature W (weight)
       ├─ _te.Update(openSymbols, t)            ← σ_TE computed BEFORE lot loop
       ├─ for each lot in OpenLots (copy):
       │   ├─ ExtractSnapshot(lot, t, …)        ← builds LotStateVector, calls OracleBoundary
       │   └─ if Y_Oracle == 1 → Harvest(lot)  ← closes lot, schedules reopen
       ├─ process _reopenQueue[t]               ← lots whose 30-day window just cleared
       ├─ _state.AdvanceDay()                   ← increments all wash clocks by 1
       └─ year-end check → ResetForNewYear() + SeedGYTD(_seedAmount)
```

Two things worth noting:

1. **`ToList()` defensive copy**: the lot loop iterates over `_state.OpenLots.ToList()` because
   `Harvest` mutates `OpenLots` mid-loop. Without the copy, the iterator would throw.

2. **Reopen queue semantics**: harvested lots are *absent* from `OpenLots` for exactly 30 days
   (IRS wash-sale window). They live in `_reopenQueue[t + 30]`. On day `t+30`, they're
   re-instantiated as new `Lot` objects at whatever the market price is that day.

---

## III. SoftLabelBuilder — "discrete extrema" → INCORRECT

**Your intuition is off here. Soft labels are forward-looking probability estimates, not extrema.**

The two soft labels answer different questions about the *same* oracle function `f*(x)`:

| Label | Question |
|---|---|
| `Y_Soft_GBM` | If I simulate 200 possible future price paths from today's price using GBM, what **fraction of those paths** would cause the oracle to fire at least once in the next 30 days? |
| `Y_Soft_BT` | Looking at the **actual** next 30 trading days of real price data, what fraction of those days would the oracle have fired? |

Both produce a float in [0, 1] — a **soft probability**, not a binary label. `Y_Oracle` (the hard
label) is binary (0 or 1), but the soft labels are continuous, which gives the eventual ML model
a richer training signal.

**No ML is used inside SoftLabelBuilder.** It's pure simulation/statistics:

```
ComputeGBM:
  1. Estimate trailing 21-day realised vol → annualise → σ
  2. Call GbmSimulator.FractionFiring(200 paths, 30 steps)
  3. Each path: S_{t+1} = S_t × exp((μ - σ²/2)/252 + σ/√252 × Z)  [Itô drift]
  4. At each step, evaluate oracle predicate (freeze G_YTD, σ_TE, washClock from snapshot)
  5. Return: (paths that fired at least once) / 200

ComputeBT:
  1. Load actual prices for t+1 … t+30
  2. For each day, call OracleBoundary.Label with frozen portfolio state
  3. Return: (days oracle fires) / 30
```

**First-passage semantics for GBM**: once a path triggers the oracle on step `s`, that path is
counted once and stopped. Steps `s+1 … 30` on that path are counterfactual (the lot would have
been harvested).

**Frozen portfolio state**: both methods freeze `G_YTD`, `σ_TE`, `costBasis`, and `washClock` at
the snapshot's values. The WashClock advances by `+s` inside the window to simulate the 30-day
progress, but the portfolio-level quantities don't change. This is a deliberate approximation —
modelling full portfolio interaction for 200 paths × 503 lots × 30 steps is prohibitively expensive.

The GameStop analogy doesn't map well here. A better one: "Given a stock is sitting at 5% below
its cost basis right now, what's the probability it hits the harvest threshold (≥2% loss) at some
point in the next 30 trading days, given current market volatility?"

---

## IV. TrackingErrorProxy — "calculated when harvested" → INCORRECT

**σ_TE is updated every single trading day, not at harvest time. It's a feature, not an event.**

The correct mental model:

```
Each day t in ProcessDay:
  sigmaTE = _te.Update(openSymbols, t)   ← called BEFORE the lot loop
  
  for each lot:
    snap = ExtractSnapshot(lot, t, close, portValue, sigmaTE)
    //                                              ^^^^^^^^
    //                  σ_TE is now a column in LotStateVector (Sigma_TE field)
    //                  Oracle gate 2: Label fires only if sigmaTE <= 0.05
```

`TrackingErrorProxy.Update` does:

```
1. portRet  = mean daily return of open-lot symbols
2. benchRet = mean daily return of ALL 503 symbols (full S&P 500)
3. diff[t]  = portRet - benchRet           → rolling buffer (30 entries)
4. sigmaTE  = std(diff_window) × √252      → annualised tracking error
5. return sigmaTE (clipped to 0 if <5 observations)
```

**Why this gate exists:** harvesting removes a stock from the portfolio. If the portfolio has
already drifted far from the S&P 500 benchmark (σ_TE high), harvesting more stocks makes it worse.
The gate prevents cascading divergence.

**The bug we fixed:** The old code computed `portRet = (V_t - V_{t-1}) / V_{t-1}` using total
portfolio dollar value. When you harvest a lot, `V_t` drops by the lot's dollar value on the
*following* day (the lot is gone, so it contributes 0 to the next day's value). For a $1M
portfolio with 503 lots, removing one lot ≈ −0.2% "return," but with heavy harvest periods it
compounded to −33% spurious "returns" and σ_TE > 116% (measured in one real run). The oracle's
σ_TE ≤ 5% gate was permanently closed for the rest of the simulation after that.

After the fix (equal-weight return of open symbols), structural lot changes are invisible to the
TE calculation — only genuine benchmark divergence contributes.

---

## V. SimulationExporter — "PriceLoader data being mutated" → INCORRECT

**SimulationExporter has zero dependency on PriceLoader. The data flow is strictly one-way.**

The correct data flow:

```
PriceLoader ──reads─→ SimulationEngine ──produces─→ List<LotStateVector>
                                                              │
              PriceLoader ──reads─→ SoftLabelBuilder ──mutates (via with-expr)
                                                              │
                                              SimulationExporter ──writes─→ lots.csv
```

**SoftLabelBuilder does NOT mutate PriceLoader.** It reads from PriceLoader (price/return arrays)
and replaces placeholder soft labels in `List<LotStateVector>` using C# record `with` expressions:

```csharp
snapshots[i] = snap with
{
    Y_Soft_GBM = ComputeGBM(snap),
    Y_Soft_BT  = ComputeBT(snap)
};
```

`with` on a `record` allocates a new object; the original `snap` is discarded. The list slot
`snapshots[i]` is overwritten with the new object. `Parallel.For` is safe here because each
index `i` is written by exactly one worker.

**SimulationExporter** just iterates the list and serializes each `LotStateVector` to CSV:
- `float.NaN` → empty string (Excel/pandas friendly)
- Strings containing commas → quoted
- Column order is hardcoded to match the LotStateVector field order

No PriceLoader, no oracle, no state. Pure serialization.

---

## Summary: what was wrong

| Your model | Reality |
|---|---|
| PriceLoader dicts ≈ "different datatypes" | `Dict<string, float[]>` = hash-keyed matrix; key=symbol, index=dayIndex |
| SoftLabelBuilder finds "discrete extrema" | Computes soft probability via 200 GBM path simulations (no ML) |
| σ_TE calculated at harvest time | Updated every day *before* the lot loop; it's a per-row feature |
| SoftLabelBuilder mutates PriceLoader | SoftLabelBuilder reads PriceLoader, mutates LotStateVector list |
| SimulationExporter depends on PriceLoader | Zero dependency; only receives `List<LotStateVector>` |

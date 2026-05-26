# Codebook — `data/lots.csv`

Per-column description for the dataset produced by the C# simulation engine
(`SimulationEngine` → `LotStateVector` → `SimulationExporter`).

Each row represents one **(lot, day) snapshot**: a single open tax lot observed on
a single trading day during the backtesting simulation.  A "lot" is an indivisible
share-block purchased at a specific cost basis; the same ticker may have multiple
open lots across the simulation lifecycle (e.g. one initial lot + one reopened lot
after a wash-sale window).

## Data source

- **Underlying prices**: S&P 500 daily OHLC data from
  [Financial Modeling Prep](https://financialmodelingprep.com/) API (`/v3/sp500-constituent`
  for the constituent list, `/stable/historical-price-eod/full` for prices).
- **Date range**: rolling 2-year window ending today (~504 trading days). Re-running
  `dotnet run --project src -- download` incrementally extends the window.
- **Simulation period**: starts at trading day 200 (warmup for MA-200 feature),
  ends at the last available day (~day 500).  Produces ~122K rows.

Formal citation:
> Financial Modeling Prep API.  S&P 500 Constituents and Historical EOD Prices.
> Accessed 2026.  https://site.financialmodelingprep.com/

## Schema

### Lot-level features

| Column | Type   | Description |
|--------|--------|-------------|
| `L`    | float  | **Unrealized return** of the lot: $(P_t - p_k) / p_k$, where $p_k$ is the lot's cost basis per share and $P_t$ is today's close. Negative means a loss (harvest candidate). |
| `H`    | int    | **Holding period** in days. Triggers long-term tax treatment at 365. |
| `S`    | {0, 1} | **Long-term flag**: 1 if `H >= 365`, else 0. |
| `B`    | float  | **Cost basis** per share ($p_k$ from the math docs). Constant across the lot's lifetime. |
| `W`    | float  | **Lot weight** in portfolio: $q_k \cdot P_t / V_t$ where $q_k$ is shares, $V_t$ is total portfolio value. |
| `K`    | int    | **Lot count for symbol** — how many open lots of the same ticker currently exist. |

### Portfolio-level features (shared across all lots same day)

| Column      | Type   | Description |
|-------------|--------|-------------|
| `G_YTD`     | float  | **Year-to-date realised gain** ($G^{\text{YTD}}_t$). Net of all harvests this calendar year. Seeded to 5% of initial portfolio value on day 0 / Jan 1 to represent prior gains the client has elsewhere available for offset. |
| `Sigma_TE`  | float  | **Annualised tracking error** ($\sigma_{\text{TE},t}$) computed via the quadratic form $\sqrt{\delta w^\top \Sigma\, \delta w \cdot 252}$. See [`SimulationMath.md`](SimulationMath.md) §5. |
| `WashClock` | int    | **Days since last harvest of this ticker** ($\mathcal{W}_{k,t}$). 999 if never harvested. The oracle requires `WashClock >= 30` (IRS wash-sale rule). |

### Asset-level features (specific to the lot's ticker)

| Column        | Type   | Description |
|---------------|--------|-------------|
| `R_t`         | float  | **Daily return** of the underlying: $(P_t - P_{t-1}) / P_{t-1}$. |
| `SigmaRange`  | float  | **Range volatility proxy**: $(H_t - L_t) / P_{t-1}$ using today's high/low. |
| `DeltaMA50`   | float  | **Deviation from 50-day moving average**: $(P_t - \overline{P}_{50}) / \overline{P}_{50}$. |
| `DeltaMA200`  | float  | **Deviation from 200-day moving average**: $(P_t - \overline{P}_{200}) / \overline{P}_{200}$. |

### Derived features

| Column      | Type   | Description |
|-------------|--------|-------------|
| `TaxAlpha`  | float  | Estimated tax savings if the lot is harvested today: $0.20 \cdot |\text{loss}|$ if long-term, else $0.37 \cdot |\text{loss}|$. Zero if `G_YTD <= 0`. |
| `DaysToYE`  | int    | Trading days remaining until December 31 of the current year. |

### Labels

| Column        | Type   | Description |
|---------------|--------|-------------|
| `Y_Oracle`    | {0, 1} | **Hard label**: $\mathbf{1}[\ell \le -2\%] \cdot \mathbf{1}[\sigma_{\text{TE}} \le 5\%] \cdot \mathbf{1}[G^{\text{YTD}} > 0] \cdot \mathbf{1}[\mathcal{W} \ge 30]$ — fires when the four-gate oracle would harvest this lot today. |
| `Y_Soft_GBM`  | float [0, 1] | **Soft label, stochastic**: fraction of 200 GBM-simulated paths over the next 30 days where the oracle would fire at least once. Reserved for future use; not a v0.1 ML target. |
| `Y_Soft_BT`   | float [0, 1] | **Soft label, deterministic**: fraction of the next 30 actual trading days where the oracle would fire (computed from real forward prices). NaN for the last ~30 rows per lot (insufficient forward window). **Primary v0.1 ML target** after thresholding `> 0`. |

### Metadata

| Column     | Type   | Description |
|------------|--------|-------------|
| `Symbol`   | string | Ticker symbol (e.g. "AAPL"). |
| `Sector`   | string | S&P 500 sector (e.g. "Information Technology"). **Currently empty in this dataset** — `constituents.json` was bootstrapped without sector metadata; re-running `--download` with a fresh constituent fetch repopulates it. Treated as `"Unknown"` by the preprocessor. |
| `Timestep` | int    | Trading-day index in the simulation calendar (200 ≤ t ≤ ~500). |

## Missing data

| Column        | NaN count (latest run) | Reason |
|---------------|-------|--------|
| `Sector`      | 100% | `constituents.json` bootstrapped without real sector metadata. Filled with `"Unknown"` in preprocessing. Will populate when re-downloaded with FMP key on a paid plan. |
| `Y_Soft_BT`   | ≈ 8.7%  | Last ~30 rows per lot — insufficient forward window. Rows dropped before training. |
| `DeltaMA200`  | ≈ 0.2%  | Early days where MA-200 isn't yet computable for a ticker (despite the 200-day warmup, some tickers have missing history). Filled with column median. |
| `DeltaMA50`   | ≈ 0.04% | Same as above, less impactful. Filled with column median. |

## File format

CSV with one header row.  Empty cells represent NaN (matches pandas default).
Strings containing commas are quoted.  Encoding: UTF-8.

Approximate dimensions:
- **Rows**: ~122K (one per open-lot per trading day, ~503 lots × ~245 trading days each, minus harvest-induced absences)
- **Columns**: 21
- **File size**: ~15 MB

## Related documents

- [`SimulationMath.md`](SimulationMath.md) — how each column is computed by the simulation
- [`MLPipeline.md`](MLPipeline.md) — ML architecture and definitions
- [`MLDerivations.md`](MLDerivations.md) — mathematical derivations
- [`PortfolioMath.md`](PortfolioMath.md) — portfolio domain model

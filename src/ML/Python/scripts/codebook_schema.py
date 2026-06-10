"""Single source of truth for the lots.csv column schema.

Mirrors the C# `LotStateVector` record (src/Core/Portfolio/LotStateVector.cs)
and `FeatureLists` (src/ML/CSharp/MLNet/Schema/FeatureLists.cs). If a column
is added there, add it here — `scripts.codebook` asserts the CSV header
matches this list exactly, so drift fails loudly instead of silently.

Each entry: name, dtype, units, role, description, encoding, missing, source.
Mathematical definitions follow DataMemo/SimulationMath.md and
DataMemo/PortfolioMath.md.
"""
from __future__ import annotations

COLUMNS: list[dict] = [
    {
        "name": "L",
        "dtype": "float",
        "units": "unitless (fractional return)",
        "role": "feature (lot-level)",
        "description": (
            "Unrealized return of the lot: L = (P_t − p_k) / p_k, where P_t is "
            "today's close and p_k is the lot's cost basis per share. Negative "
            "values are paper losses; L ≤ −0.02 is the oracle's loss gate."
        ),
        "encoding": "Continuous in (−1, ∞). Negative = loss position.",
        "missing": "None.",
        "source": "Lot.UnrealizedReturn(P_t)",
    },
    {
        "name": "H",
        "dtype": "int",
        "units": "trading days",
        "role": "feature (lot-level)",
        "description": (
            "Holding period of the lot: H = t − s_k, the number of simulation "
            "days since the lot was purchased."
        ),
        "encoding": "Non-negative integer.",
        "missing": "None.",
        "source": "Lot.HoldingPeriod(t)",
    },
    {
        "name": "S",
        "dtype": "int (binary)",
        "units": "—",
        "role": "feature (lot-level)",
        "description": (
            "Long-term holding flag: S = 1[H ≥ 365]. Determines whether the "
            "long-term or (higher) short-term capital-gains tax rate applies "
            "in the tax-alpha formula."
        ),
        "encoding": "0 = short-term (< 365 days), 1 = long-term (≥ 365 days).",
        "missing": "None.",
        "source": "Lot.IsLongTerm(t)",
    },
    {
        "name": "B",
        "dtype": "float",
        "units": "US dollars per share",
        "role": "feature (lot-level)",
        "description": "Cost basis per share p_k — the price the lot was purchased at.",
        "encoding": "Positive continuous.",
        "missing": "None.",
        "source": "Lot.CostBasis",
    },
    {
        "name": "W",
        "dtype": "float",
        "units": "unitless (portfolio fraction)",
        "role": "feature (lot-level)",
        "description": (
            "Lot weight in the portfolio: W = q_k · P_t / V_t, the lot's share "
            "of total portfolio market value on day t."
        ),
        "encoding": "Continuous in (0, 1).",
        "missing": "None.",
        "source": "derived in SimulationEngine",
    },
    {
        "name": "K",
        "dtype": "int",
        "units": "count",
        "role": "feature (lot-level)",
        "description": (
            "Number of open lots in the same ticker as this lot (including "
            "itself). Stays 1 in v0.1 unless harvested lots are re-opened."
        ),
        "encoding": "Positive integer.",
        "missing": "None.",
        "source": "counted from PortfolioState.OpenLots",
    },
    {
        "name": "G_YTD",
        "dtype": "float",
        "units": "US dollars",
        "role": "feature (portfolio-level)",
        "description": (
            "Net realized gain/loss for the calendar year to date, shared by "
            "every lot at the same timestep. Seeded at +$500,000 (5% of the "
            "$10M portfolio) at simulation start and re-seeded after each "
            "year-end reset to simulate externally realized gains. Harvesting "
            "a loss pushes G_YTD down; the oracle only fires while G_YTD > 0 "
            "(there must be gains to offset)."
        ),
        "encoding": "Signed continuous. Positive = net realized gains.",
        "missing": "None.",
        "source": "PortfolioState.G_YTD",
    },
    {
        "name": "Sigma_TE",
        "dtype": "float",
        "units": "annualized volatility (fraction)",
        "role": "feature (portfolio-level)",
        "description": (
            "Forward-looking tracking-error estimate vs the equal-weight "
            "benchmark: σ_TE = sqrt(δwᵀ Σ̂ δw · 252), where Σ̂ is the daily "
            "return covariance matrix and δw the active-weight deviation from "
            "holding every constituent. Shared by every lot at the same "
            "timestep. The oracle requires σ_TE ≤ 0.05 (5% budget)."
        ),
        "encoding": "Positive continuous; 0.05 = 5% annualized TE.",
        "missing": "None.",
        "source": "TrackingErrorProxy.Update",
    },
    {
        "name": "WashClock",
        "dtype": "int",
        "units": "calendar days",
        "role": "feature (portfolio-level)",
        "description": (
            "Days since the last harvest of this lot's ticker. The IRS "
            "wash-sale rule blocks re-claiming a loss within 30 days, so the "
            "oracle requires WashClock ≥ 30. Clocks persist across year-end."
        ),
        "encoding": "Non-negative integer. Sentinel 999 = ticker never harvested.",
        "missing": "None (sentinel encodes 'never').",
        "source": "PortfolioState.GetWashClock",
    },
    {
        "name": "R_t",
        "dtype": "float",
        "units": "unitless (daily return)",
        "role": "feature (asset-level)",
        "description": "One-day simple return of the ticker: R_t = (P_t − P_{t−1}) / P_{t−1}.",
        "encoding": "Signed continuous.",
        "missing": "None.",
        "source": "PriceLoader close series",
    },
    {
        "name": "SigmaRange",
        "dtype": "float",
        "units": "unitless (fraction of price)",
        "role": "feature (asset-level)",
        "description": (
            "Range-based intraday volatility proxy: (High_t − Low_t) / P_{t−1}."
        ),
        "encoding": "Positive continuous.",
        "missing": "None.",
        "source": "PriceLoader OHLC",
    },
    {
        "name": "DeltaMA50",
        "dtype": "float",
        "units": "unitless (fractional deviation)",
        "role": "feature (asset-level)",
        "description": (
            "Price deviation from the 50-day moving average: "
            "(P_t − MA_50) / MA_50. Momentum / mean-reversion signal."
        ),
        "encoding": "Signed continuous.",
        "missing": (
            "NaN (empty cell) when fewer than 50 prior closes exist for the "
            "ticker; median-imputed inside each training fold."
        ),
        "source": "computed in SimulationEngine",
    },
    {
        "name": "DeltaMA200",
        "dtype": "float",
        "units": "unitless (fractional deviation)",
        "role": "feature (asset-level)",
        "description": (
            "Price deviation from the 200-day moving average: "
            "(P_t − MA_200) / MA_200. The 200-day warmup window exists so this "
            "is defined from the first active simulation day."
        ),
        "encoding": "Signed continuous.",
        "missing": (
            "NaN (empty cell) when fewer than 200 prior closes exist (sparse "
            "price history); median-imputed inside each training fold."
        ),
        "source": "computed in SimulationEngine",
    },
    {
        "name": "TaxAlpha",
        "dtype": "float",
        "units": "US dollars",
        "role": "feature (derived)",
        "description": (
            "Estimated tax savings from harvesting this lot today: "
            "TaxAlpha = τ(H) · |unrealized loss in dollars| · 1[G_YTD > 0], "
            "where τ(H) is the short- or long-term marginal tax rate selected "
            "by the holding period."
        ),
        "encoding": "Non-negative continuous; 0 when the lot has no loss or no gains exist to offset.",
        "missing": "None.",
        "source": "derived in SimulationEngine",
    },
    {
        "name": "DaysToYE",
        "dtype": "int",
        "units": "calendar days",
        "role": "feature (derived)",
        "description": (
            "Calendar days remaining until December 31 of the simulated tax "
            "year. Year-end is when G_YTD resets, so harvest urgency varies "
            "with this clock."
        ),
        "encoding": "Integer in [0, 365].",
        "missing": "None.",
        "source": "calendar arithmetic in SimulationEngine",
    },
    {
        "name": "Y_Oracle",
        "dtype": "int (binary)",
        "units": "—",
        "role": "label (hard)",
        "description": (
            "Deterministic oracle harvest decision — the conjunction of four "
            "gates: 1[L ≤ −0.02] · 1[Sigma_TE ≤ 0.05] · 1[G_YTD > 0] · "
            "1[WashClock ≥ 30]. This is the decision boundary the supervised "
            "models try to learn. Never used as a model input."
        ),
        "encoding": "0 = do not harvest, 1 = harvest. Positive rate ≈ 0.85%.",
        "missing": "None.",
        "source": "OracleBoundary.Label",
    },
    {
        "name": "Y_Soft_GBM",
        "dtype": "float",
        "units": "probability",
        "role": "label (soft, stochastic)",
        "description": (
            "Probability the oracle fires within the next 30 trading days, "
            "estimated as the fraction of 200 geometric-Brownian-motion "
            "forward price paths (per-stock σ calibrated from trailing 21-day "
            "realized volatility) on which the oracle predicate is hit, with "
            "portfolio state frozen at the snapshot. First-passage semantics: "
            "each path counts at most once."
        ),
        "encoding": "Continuous in [0, 1] in increments of 1/200.",
        "missing": "None.",
        "source": "SoftLabelBuilder + GbmSimulator.FractionFiring",
    },
    {
        "name": "Y_Soft_BT",
        "dtype": "float",
        "units": "fraction of days",
        "role": "label (soft, deterministic)",
        "description": (
            "Fraction of the next 30 actual trading days on which the oracle "
            "would fire, computed from the real forward price series with "
            "portfolio state frozen at the snapshot. The primary supervised "
            "training target (binarized as Y_Soft_BT > 0 for classification)."
        ),
        "encoding": "Continuous in [0, 1] in increments of 1/30.",
        "missing": (
            "NaN (empty cell) when fewer than 30 forward days remain in the "
            "data window — structurally missing for the final 30 timesteps "
            "(670–699). These rows are excluded from soft-label training."
        ),
        "source": "SoftLabelBuilder (real forward window)",
    },
    {
        "name": "Symbol",
        "dtype": "string",
        "units": "—",
        "role": "metadata (dropped before modeling)",
        "description": "Ticker symbol of the lot's asset, e.g. 'AAPL'. S&P 500 constituent.",
        "encoding": "Uppercase ticker string.",
        "missing": "None.",
        "source": "SPY holdings (constituents.json)",
    },
    {
        "name": "Sector",
        "dtype": "string (categorical)",
        "units": "—",
        "role": "categorical feature",
        "description": (
            "GICS-style sector of the ticker from the SPY holdings file. In "
            "the v0.1 data this column is degenerate: ≈99.5% of rows carry "
            "the placeholder '-' and the rest are empty, so after cleaning it "
            "is effectively a single 'Unknown' category."
        ),
        "encoding": (
            "'-' or empty → 'Unknown' before one-hot encoding (vocabulary fit "
            "on the training fold only)."
        ),
        "missing": "'-' placeholder ≈99.5% of rows; empty ≈0.5%.",
        "source": "SPY holdings (constituents.json)",
    },
    {
        "name": "Timestep",
        "dtype": "int",
        "units": "trading-day index",
        "role": "metadata (dropped before modeling)",
        "description": (
            "Simulation day index t. Days 0–199 are the moving-average warmup "
            "(no rows emitted); active rows span t = 200–699, roughly two "
            "calendar years of real price history."
        ),
        "encoding": "Integer in [200, 699].",
        "missing": "None.",
        "source": "SimulationEngine day loop",
    },
]

#: Header order expected in data/lots.csv (must match SimulationExporter).
EXPECTED_HEADER: list[str] = [c["name"] for c in COLUMNS]

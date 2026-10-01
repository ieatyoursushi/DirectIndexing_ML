"""Single source of truth for the lots.csv column schema.

Mirrors the C# `LotStateVector` record (src/Core/Portfolio/LotStateVector.cs)
and `FeatureLists` (src/ML/CSharp/MLNet/Schema/FeatureLists.cs). If a column
is added there, add it here — `scripts.codebook` asserts the CSV header
matches this list exactly, so drift fails loudly instead of silently.

Each entry: name, dtype, units, role, description, encoding, missing, source.
Mathematical definitions follow DataMemo/spec/ (SymbolTable.md is the index).
Schema version: v5 (27 columns, d = 19) — v0.3-3 split the ledger by §1222 character.
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
        "name": "NetST",
        "dtype": "float",
        "units": "US dollars",
        "role": "feature (portfolio-level, TaxLedger)",
        "description": (
            "G^ST — signed net SHORT-term (held <= 1 calendar year, §1222) realized "
            "gain/loss for the calendar year to date, shared by every lot at the "
            "same timestep. Harvested ST losses push it down, ST gains up. Resets "
            "to 0 at year-end (Schedule D netting; leftover loss becomes CarryST)."
        ),
        "encoding": "Signed continuous. Positive = net ST gains.",
        "missing": "None.",
        "source": "TaxLedger.NetShortTerm (via PortfolioState.Ledger)",
    },
    {
        "name": "NetLT",
        "dtype": "float",
        "units": "US dollars",
        "role": "feature (portfolio-level, TaxLedger)",
        "description": (
            "G^LT — signed net LONG-term (held > 1 calendar year) realized "
            "gain/loss for the calendar year to date. Resets to 0 at year-end."
        ),
        "encoding": "Signed continuous. Positive = net LT gains.",
        "missing": "None.",
        "source": "TaxLedger.NetLongTerm",
    },
    {
        "name": "CarryST",
        "dtype": "float",
        "units": "US dollars",
        "role": "feature (portfolio-level, TaxLedger)",
        "description": (
            "C^ST — short-term capital-loss carryforward from prior years "
            "(26 USC §1212(b) preserves character). Enters this year's netting "
            "as a ST loss, so it is CONSUMED by this year's gains before a new "
            "harvest can use them (ROADMAP F8)."
        ),
        "encoding": "Non-negative continuous; changes only at the year-end roll.",
        "missing": "None.",
        "source": "TaxLedger.CarryShortTerm (year-end roll)",
    },
    {
        "name": "CarryLT",
        "dtype": "float",
        "units": "US dollars",
        "role": "feature (portfolio-level, TaxLedger)",
        "description": (
            "C^LT — long-term capital-loss carryforward from prior years. "
            "The $3,000 ordinary deduction is taken from short-term loss first, "
            "so loss-only books bank mostly into the character they harvest."
        ),
        "encoding": "Non-negative continuous; changes only at the year-end roll.",
        "missing": "None.",
        "source": "TaxLedger.CarryLongTerm (year-end roll)",
    },
    {
        "name": "OrdinaryOffsetBudget",
        "dtype": "float",
        "units": "US dollars",
        "role": "feature (portfolio-level, TaxLedger)",
        "description": (
            "The §1211(b) ordinary-income allowance NOT yet claimed if the "
            "year closed today: $3,000 minus the deduction of the Schedule D "
            "netting of (NetST, NetLT, CarryST, CarryLT). Carryforward claims it "
            "before a new harvest can. Together with the gains left after netting "
            "it forms offsetCapacity, the dollars of a new loss usable this year."
        ),
        "encoding": "Continuous in [0, 3000]. Derived from the four ledger columns.",
        "missing": "None.",
        "source": "TaxLedger.OrdinaryOffsetBudget (derived)",
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
            "Calendar days to this lot's nearest §1091 event: min(days since the "
            "ticker's last LOSS sale, days since a DIFFERENT open lot of the ticker "
            "was acquired), capped at 999. The wash-sale window is ±30 calendar days "
            "inclusive, so the oracle requires WashClock > 30 (v0.3-1: both sides, "
            "calendar-dated). Clocks are date differences and persist across year-end."
        ),
        "encoding": "Non-negative integer, capped at 999 (= no §1091 event on record).",
        "missing": "None (sentinel encodes 'never').",
        "source": "PortfolioState.WashClock(lot)",
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
        "name": "TaxValue",
        "dtype": "float",
        "units": "US dollars",
        "role": "feature (derived, lot-level × TaxLedger)",
        "description": (
            "Dollar value of harvesting this lot today, as a counterfactual "
            "difference of the Schedule D year-end netting S: "
            "TaxValue = [T(ledger) − T(ledger ⊕ loss)] + τ_future·δ·[ΔC], "
            "i.e. this year's tax saved (at the rate of whatever the loss "
            "displaces: a ST gain 0.37, a LT gain 0.20, the $3k ordinary line "
            "0.37 — nothing if carryforward already absorbs those) plus the "
            "newly banked carryforward ΔC at τ_future = 0.20, discounted δ = 0.5. "
            "The lot's own §1222 character (S) only decides which pool the loss "
            "enters. Supersedes the v0.2 TaxAlpha and the v0.25 blended pool."
        ),
        "encoding": "Non-negative continuous; 0 when the lot is not at a loss.",
        "missing": "None.",
        "source": "TaxLedger.ComputeTaxValue(lossDollars, isLongTerm)",
    },
    {
        "name": "DaysToYE",
        "dtype": "int",
        "units": "calendar days",
        "role": "feature (derived)",
        "description": (
            "Calendar days remaining until December 31 of the simulated tax "
            "year. Year-end is when the ledger's annual accumulators reset "
            "(and net losses roll into CarryST / CarryLT), so harvest urgency "
            "varies with this clock."
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
            "Deterministic scalarized-oracle harvest decision: "
            "1[L ≤ −0.02] · 1[WashClock > 30] · 1[Sigma_TE ≤ 0.15] · 1[U > 0], "
            "U = TaxValue − λ·Sigma_TE² − c_trade (λ = 90,000, c_trade = $10). "
            "The decision boundary is the level set {U = 0}. This is the "
            "cross-sectional target the supervised models recover (the leakage "
            "control: it is deterministic in current features). Never used as "
            "a model input."
        ),
        "encoding": "0 = do not harvest, 1 = harvest. Positive rate ≈ 1.6%.",
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
        "name": "Y_TaxValue",
        "dtype": "float",
        "units": "US dollars",
        "role": "label (continuous regression target)",
        "description": (
            "Cross-sectional regression target: taxValue_k of this lot at this "
            "timestep — the capacity-aware harvest value from the TaxLedger. "
            "Numerically identical to the TaxValue feature by construction in "
            "v0.25, so regressions on this target MUST exclude TaxValue from "
            "the feature set (the task is recovering g(ledger, H, L) from raw "
            "features). First member of the issue #17 richer-label family."
        ),
        "encoding": "Non-negative continuous dollars.",
        "missing": "None.",
        "source": "TaxLedger.ComputeTaxValue at snapshot time",
    },
    {
        "name": "Y_Utility",
        "dtype": "float",
        "units": "US dollars",
        "role": "label (diagnostic / RL reward)",
        "description": (
            "Raw scalarized objective before thresholding: "
            "U(x) = TaxValue − λ·Sigma_TE² − c_trade (λ = 90,000; c_trade = $10 "
            "flat round-trip harvest friction, override via --ctrade=). The "
            "scalarized oracle fires iff U > 0 (plus the hard gates), so the "
            "decision boundary is the level set {U = 0}. "
            "Computed under the run's OracleConfig. Never a feature — 𝟙[U > 0] is the oracle's own boundary; "
            "exported as the issue-#17 continuous target and the v0.4 RL "
            "per-decision reward."
        ),
        "encoding": "Signed continuous dollars.",
        "missing": "None.",
        "source": "OracleBoundary.Utility(TaxValue, Sigma_TE, config)",
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

# The d = 19 numeric feature block, in schema order — derived, never restated.
# Must equal C# FeatureLists.NumericFeatures (asserted by tests/test_codebook_schema.py).
NUMERIC_FEATURES: list[str] = [
    c["name"] for c in COLUMNS
    if c["role"].startswith("feature") and not c["dtype"].startswith("string")
]


def repo_root(start=None):
    """Walk up from `start` (default cwd) to the directory holding DirectIndexing.sln —
    mirrors PythonRunner.LocateRepoRoot on the C# side."""
    from pathlib import Path
    d = (Path(start) if start else Path.cwd()).resolve()
    for candidate in (d, *d.parents):
        if (candidate / "DirectIndexing.sln").exists():
            return candidate
    raise FileNotFoundError(f"no DirectIndexing.sln above {d}")

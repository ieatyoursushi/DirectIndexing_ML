using DirectIndexing.Core.Oracle;
using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.Core.Simulation;

/// <summary>
/// Second-pass labeller — fills Y_Soft_GBM and Y_Soft_BT on each LotStateVector
/// after the main backtesting day loop has set Y_Oracle.
///
/// Both strategies freeze the portfolio state at the snapshot's timestep:
///   - The TaxLedger scalars (RealizedGainsYTD and the derived offsetCapacity)
///     and Sigma_TE are held constant (from the snapshot fields).
///   - The wash-sale clock advances by the number of days into the window.
///   - The holding period advances with it (τ(h) can flip short→long mid-window).
///   - The cost basis p_k and share count q_k are held constant, so the loss is
///     re-dollarized at each forward price for the scalarized taxValue term.
///
/// The oracle itself stays a black box 𝒳 → {0,1}: swapping its internals
/// (4-gate AND → gates·𝟙[U&gt;0]) changes nothing about the Cesàro-averaging
/// machinery here (GYTD_Redesign_Plan.md v2 §5.1).
///
/// Y_Soft_GBM  — stochastic: 200 GBM paths, σ calibrated from trailing 21-day vol.
/// Y_Soft_BT   — deterministic: fraction of next 30 actual days oracle would fire.
/// </summary>
public sealed class SoftLabelBuilder
{
    private readonly PriceLoader  _prices;
    private readonly OracleConfig _oracle;

    private const int Window    = 30;   // forward simulation window (trading days)
    private const int VolWindow = 21;   // trailing days for σ estimate

    // Shared GbmSimulator instance — same Paths/Horizon as the old inline constants.
    // Static so all parallel Parallel.For workers share one simulator configuration
    // while each supplies its own thread-local Random instance.
    private static readonly GbmSimulator _gbm = new(paths: 200, horizon: Window);

    public SoftLabelBuilder(PriceLoader prices, OracleConfig? oracleConfig = null)
    {
        _prices = prices;
        _oracle = oracleConfig ?? OracleConfig.Scalarized;
    }

    // ── Public entry point ───────────────────────────────────────────────────

    /// <summary>
    /// Fills Y_Soft_GBM and Y_Soft_BT on every snapshot in place (via record with-expression).
    /// Runs in parallel over snapshots for performance.
    /// </summary>
    public void Label(List<LotStateVector> snapshots)
    {
        Console.WriteLine($"[SoftLabelBuilder] Labelling {snapshots.Count} snapshots " +
                          $"(oracle={_oracle.Mode}) …");

        Parallel.For(0, snapshots.Count, i =>
        {
            var snap = snapshots[i];
            snapshots[i] = snap with
            {
                Y_Soft_GBM = ComputeGBM(snap),
                Y_Soft_BT  = ComputeBT(snap)
            };
        });

        Console.WriteLine("[SoftLabelBuilder] Done.");
    }

    // ── Frozen-state oracle step (shared by both strategies) ─────────────────

    /// <summary>
    /// Evaluates the oracle at forward step s under frozen portfolio state:
    /// price is the step's (simulated or historical) close; everything ledger-
    /// and TE-shaped comes from the snapshot; wash clock and holding period
    /// advance with s.
    /// </summary>
    private int StepLabel(
        float price, int s,
        float costBasis, float shares, int holdingDays0, int initClock,
        float gYtdF, float sigmaTE, decimal frozenCapacity)
    {
        float ell = costBasis > 0f ? (price - costBasis) / costBasis : 0f;

        decimal lossDollars = ell < 0f
            ? (decimal)(costBasis - price) * (decimal)shares
            : 0m;
        decimal taxValue = TaxLedger.ComputeTaxValue(
            lossDollars, holdingDays0 + s, frozenCapacity);

        return OracleBoundary.Label(
            unrealizedReturn: (decimal)ell,
            sigmaTE:          sigmaTE,
            netRealizedYtd:   (decimal)gYtdF,
            washClock:        initClock + s,
            taxValue:         taxValue,
            config:           _oracle);
    }

    // ── GBM soft label ────────────────────────────────────────────────────────

    private float ComputeGBM(LotStateVector snap)
    {
        if (!_prices.HasData(snap.Symbol, snap.Timestep)) return float.NaN;

        float currentClose = _prices.GetClose(snap.Symbol, snap.Timestep);
        if (float.IsNaN(currentClose) || currentClose <= 0f) return float.NaN;

        float annualSigma = EstimateVol(snap.Symbol, snap.Timestep);
        if (float.IsNaN(annualSigma) || annualSigma <= 0f)
            annualSigma = 0.20f;   // fallback: 20% annual vol

        // Frozen state from snapshot — captured by the closure below
        float   gYtdF     = snap.RealizedGainsYTD;
        float   sigmaTE   = snap.Sigma_TE;
        int     initClock = snap.WashClock;
        float   costBasis = snap.B;
        float   shares    = snap.Shares;
        int     h0        = snap.H;
        decimal frozenCap = (decimal)Math.Max(snap.RealizedGainsYTD, 0f)
                          + (decimal)snap.OrdinaryOffsetBudget;

        // Delegate path simulation and first-passage counting to GbmSimulator.
        // Per-snapshot Random, deterministically seeded from (Symbol, Timestep) with a
        // stable polynomial hash (string.GetHashCode is randomized per process) so
        // Y_Soft_GBM is reproducible across runs and safe under Parallel.For.
        int seed = snap.Timestep;
        foreach (char c in snap.Symbol) seed = unchecked(seed * 31 + c);
        var rng = new Random(seed);

        return _gbm.FractionFiring(
            startPrice:  currentClose,
            annualSigma: annualSigma,
            firesOnStep: (price, s) =>
                StepLabel(price, s, costBasis, shares, h0, initClock,
                          gYtdF, sigmaTE, frozenCap) == 1,
            rng: rng);
    }

    // ── Backtesting soft label ────────────────────────────────────────────────

    private float ComputeBT(LotStateVector snap)
    {
        int t0   = snap.Timestep;
        int tMax = _prices.DayCount;

        // Not enough forward data — return NaN (will be excluded from training)
        if (t0 + Window >= tMax) return float.NaN;

        float   gYtdF     = snap.RealizedGainsYTD;
        float   sigmaTE   = snap.Sigma_TE;
        int     initClock = snap.WashClock;
        float   costBasis = snap.B;
        float   shares    = snap.Shares;
        int     h0        = snap.H;
        decimal frozenCap = (decimal)Math.Max(snap.RealizedGainsYTD, 0f)
                          + (decimal)snap.OrdinaryOffsetBudget;

        int oracleDays = 0;

        for (int s = 1; s <= Window; s++)
        {
            int t = t0 + s;
            if (!_prices.HasData(snap.Symbol, t)) continue;

            float price = _prices.GetClose(snap.Symbol, t);

            if (StepLabel(price, s, costBasis, shares, h0, initClock,
                          gYtdF, sigmaTE, frozenCap) == 1)
                oracleDays++;
        }

        return (float)oracleDays / Window;
    }

    // ── Trailing volatility estimate ─────────────────────────────────────────

    private float EstimateVol(string symbol, int t)
    {
        var returns = _prices.GetReturnArray(symbol);
        int start   = Math.Max(0, t - VolWindow);
        int count   = 0;
        double sum  = 0, sumSq = 0;

        for (int i = start; i < t; i++)
        {
            float r = returns[i];
            if (float.IsNaN(r)) continue;
            sum   += r;
            sumSq += (double)r * r;
            count++;
        }

        if (count < 5) return float.NaN;

        double mean     = sum / count;
        double variance = sumSq / count - mean * mean;
        double dailyStd = Math.Sqrt(Math.Max(variance, 0));

        return (float)(dailyStd * Math.Sqrt(252));   // annualise
    }
}

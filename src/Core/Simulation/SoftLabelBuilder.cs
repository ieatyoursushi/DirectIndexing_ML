using DirectIndexing.Core.Oracle;
using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.Core.Simulation;

/// <summary>
/// Second-pass labeller — fills Y_Soft_GBM and Y_Soft_BT on each LotStateVector
/// after the main backtesting day loop has set Y_Oracle.
///
/// Both strategies freeze the portfolio state at the snapshot's timestep:
///   - G_YTD and Sigma_TE are held constant (from the snapshot fields).
///   - The wash-sale clock advances by the number of days into the window.
///   - The cost basis p_k is held constant (from snapshot.B).
///
/// Y_Soft_GBM  — stochastic: 200 GBM paths, σ calibrated from trailing 21-day vol.
/// Y_Soft_BT   — deterministic: fraction of next 30 actual days oracle would fire.
/// </summary>
public sealed class SoftLabelBuilder
{
    private readonly PriceLoader _prices;

    private const int Window    = 30;   // forward simulation window (trading days)
    private const int VolWindow = 21;   // trailing days for σ estimate

    // Shared GbmSimulator instance — same Paths/Horizon as the old inline constants.
    // Static so all parallel Parallel.For workers share one simulator configuration
    // while each supplies its own thread-local Random instance.
    private static readonly GbmSimulator _gbm = new(paths: 200, horizon: Window);

    public SoftLabelBuilder(PriceLoader prices) => _prices = prices;

    // ── Public entry point ───────────────────────────────────────────────────

    /// <summary>
    /// Fills Y_Soft_GBM and Y_Soft_BT on every snapshot in place (via record with-expression).
    /// Runs in parallel over snapshots for performance.
    /// </summary>
    public void Label(List<LotStateVector> snapshots)
    {
        Console.WriteLine($"[SoftLabelBuilder] Labelling {snapshots.Count} snapshots …");

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
        float gYtdF    = snap.G_YTD;
        float sigmaTE  = snap.Sigma_TE;
        int   initClock= snap.WashClock;
        float costBasis= snap.B;

        // Delegate path simulation and first-passage counting to GbmSimulator.
        // Thread-local Random so Parallel.For workers don't share state.
        var rng = new Random();

        return _gbm.FractionFiring(
            startPrice:  currentClose,
            annualSigma: annualSigma,
            firesOnStep: (price, s) =>
            {
                float ell  = costBasis > 0f ? (price - costBasis) / costBasis : 0f;
                int   wash = initClock + s;
                return OracleBoundary.Label((decimal)ell, sigmaTE, (decimal)gYtdF, wash) == 1;
            },
            rng: rng);
    }

    // ── Backtesting soft label ────────────────────────────────────────────────

    private float ComputeBT(LotStateVector snap)
    {
        int t0   = snap.Timestep;
        int tMax = _prices.DayCount;

        // Not enough forward data — return NaN (will be excluded from training)
        if (t0 + Window >= tMax) return float.NaN;

        float gYtdF   = snap.G_YTD;
        float sigmaTE = snap.Sigma_TE;
        int   initClock = snap.WashClock;
        float costBasis = snap.B;

        int oracleDays = 0;

        for (int s = 1; s <= Window; s++)
        {
            int t = t0 + s;
            if (!_prices.HasData(snap.Symbol, t)) continue;

            float price = _prices.GetClose(snap.Symbol, t);
            float ell   = costBasis > 0f ? (price - costBasis) / costBasis : 0f;
            int   wash  = initClock + s;

            int label = OracleBoundary.Label(
                unrealizedReturn: (decimal)ell,
                sigmaTE:          sigmaTE,
                gYtd:             (decimal)gYtdF,
                washClock:        wash);

            if (label == 1) oracleDays++;
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

using DirectIndexing.Core.Portfolio;
using DirectIndexing.Core.Simulation.Covariance;

namespace DirectIndexing.Core.Simulation;

/// <summary>How the active-weight vector δw is formed (DataMemo/decisions/VolatilityModel_v03.md §2).</summary>
public enum TeWeighting
{
    /// <summary>Legacy: δw_i = 1/n_open − 1/N for every held NAME (position size ignored).</summary>
    Names,
    /// <summary>v0.3-6: δw_i = (dollars in i)/V − 1/N_t over the names priced today.</summary>
    Dollars,
}

/// <summary>
/// Annualised ex-ante tracking error σ_TE = √(252 · δwᵀ Σ̂_t δw).
///
///   Σ̂_t — from an <see cref="ICovarianceEstimator"/>: the legacy full-history sample
///          (a look-ahead, ROADMAP F1) or the point-in-time / Ledoit–Wolf estimators (v0.3-6).
///   δw  — active weights, by <see cref="TeWeighting"/>: legacy equal-per-name, or dollar
///          weights vs an equal-weight benchmark over the names priced today.
///
/// v0.1 used std(r_port − r_bench, window=30) × √252 (rolling scalar approach). v0.2 moved to
/// the quadratic form, which exposes cross-stock correlation and survives structural lot
/// removal (TE Test 3).
/// Cost per day: O(N²) for v = Σ̂δw; Σ̂ refits are the estimator's business.
/// </summary>
public sealed class TrackingErrorProxy
{
    private readonly ICovarianceEstimator  _covEst;
    private readonly TeWeighting           _weighting;
    private readonly List<string>          _symbols;  // sorted — defines row/col order
    private readonly Dictionary<string,int> _symIdx;
    private readonly int                   _N;

    /// <summary>Legacy arm: full-sample Σ̂, equal-per-name δw (byte-identical to v0.2–v0.3-5).</summary>
    public TrackingErrorProxy(PriceLoader prices)
        : this(prices, new FullSampleCovariance(prices), TeWeighting.Names) { }

    public TrackingErrorProxy(PriceLoader prices, ICovarianceEstimator covariance, TeWeighting weighting)
    {
        _covEst    = covariance;
        _weighting = weighting;
        _symbols   = covariance.Symbols.ToList();
        _N         = _symbols.Count;
        _symIdx    = _symbols.Select((s, i) => (s, i)).ToDictionary(x => x.s, x => x.i);
        Console.WriteLine($"[TrackingErrorProxy] Σ̂ = {covariance.Name} ({_N}×{_N}), δw = {weighting}.");
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>The engine's call: σ_TE on day t for the open book at today's closes.</summary>
    // [math:delta_w] — DataMemo/spec/SymbolTable.md
    public float Update(int t, IReadOnlyList<Lot> openLots, IReadOnlyDictionary<string, decimal> closes)
    {
        if (_weighting == TeWeighting.Names)
            return Update(openLots.Select(l => l.Symbol), t);

        var dollars = new double[_N];
        double V = 0;
        foreach (var lot in openLots)
            if (_symIdx.TryGetValue(lot.Symbol, out int i) && closes.TryGetValue(lot.Symbol, out decimal px))
            {
                double v = (double)(lot.Shares * px);
                dollars[i] += v;
                V += v;
            }
        if (V <= 0) return 0f;

        int nPriced = 0;
        for (int i = 0; i < _N; i++) if (closes.ContainsKey(_symbols[i])) nPriced++;
        if (nPriced == 0) return 0f;

        var dw = new float[_N];
        for (int i = 0; i < _N; i++)
            dw[i] = (float)(dollars[i] / V - (closes.ContainsKey(_symbols[i]) ? 1.0 / nPriced : 0.0));
        return Quadratic(_covEst.At(t), dw);
    }

    /// <summary>
    /// Legacy equal-per-name form: δw_i = 1/n_open − 1/N if name i is held, else −1/N.
    /// <paramref name="t"/> selects Σ̂_t (irrelevant for the time-invariant full-sample arm).
    /// </summary>
    // [math:sigma_te] — DataMemo/spec/SymbolTable.md
    public float Update(IEnumerable<string> openSymbols, int t = 0)
    {
        var openSet = new HashSet<string>();
        foreach (var s in openSymbols)
            if (_symIdx.ContainsKey(s)) openSet.Add(s);

        int nOpen = openSet.Count;
        if (nOpen == 0) return 0f;

        float wPort  = 1f / nOpen;
        float wBench = 1f / _N;

        var dw = new float[_N];
        for (int i = 0; i < _N; i++)
            dw[i] = openSet.Contains(_symbols[i]) ? wPort - wBench : -wBench;
        return Quadratic(_covEst.At(t), dw);
    }

    /// <summary>√(252 · δwᵀ Σ δw), floored at 0.</summary>
    private float Quadratic(float[,] cov, float[] dw)
    {
        double variance = 0;
        for (int i = 0; i < _N; i++)
        {
            double vi = 0;
            for (int j = 0; j < _N; j++)
                vi += cov[i, j] * dw[j];
            variance += dw[i] * vi;
        }
        return MathF.Sqrt(MathF.Max((float)variance, 0f) * 252f);
    }
}

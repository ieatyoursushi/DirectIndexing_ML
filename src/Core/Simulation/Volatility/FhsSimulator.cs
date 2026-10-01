namespace DirectIndexing.Core.Simulation.Volatility;

/// <summary>
/// Filtered historical simulation (Barone-Adesi, Giannopoulos &amp; Vosper 1999) — the σ̂ label role
/// R2 of DataMemo/decisions/VolatilityModel_v03.md §7. A sibling of <see cref="GbmSimulator"/>:
/// same horizon, same path count, same first-passage counting, but each path
///   • draws its innovations from the name's OWN standardized residual pool
///     ε_s = r_s / σ̂_{s−1} for s ≤ t (fat tails kept, no parametric error law), centered on the
///     pool mean (drift does not confound the comparison with the zero-drift GBM label) and
///     rescaled to unit variance — a noisy σ̂ inflates E[ε²] above 1 (Jensen), which would double
///     count the level σ̂ already carries; the pool keeps only the SHAPE;
///   • starts from the 𝓕_t forecast σ̂²_t and evolves σ along the path by the SAME EWMA
///     recursion σ²_{s+1} = λσ²_s + (1−λ)r*²_s (volatility clustering inside the window).
/// Log-price step: ln P_{s} − ln P_{s−1} = σ_s ε* − σ_s²/2 (the GBM simulator's Itô convention).
/// </summary>
// [math:y_soft_fhs] — DataMemo/spec/SymbolTable.md
public sealed class FhsSimulator
{
    public int Paths   { get; }
    public int Horizon { get; }

    private readonly Dictionary<string, (double[] Var1, int[] ValidIdx, double[] Eps, double[] PrefixSum, double[] PrefixSq)> _pool = new();

    public FhsSimulator(PriceLoader prices, int paths = 200, int horizon = 30)
    {
        Paths = paths; Horizon = horizon;
        foreach (var sym in prices.Symbols)
        {
            var r = prices.GetReturnArray(sym);
            var v = new EwmaVol().Path(r).Var1;
            var idx = new List<int>(); var eps = new List<double>();
            for (int s = 1; s < r.Length; s++)
                if (!float.IsNaN(r[s]) && !double.IsNaN(v[s - 1]) && v[s - 1] > 0)
                {
                    idx.Add(s); eps.Add(r[s] / Math.Sqrt(v[s - 1]));
                }
            var prefix = new double[eps.Count + 1];
            var prefSq = new double[eps.Count + 1];
            for (int k = 0; k < eps.Count; k++) { prefix[k + 1] = prefix[k] + eps[k]; prefSq[k + 1] = prefSq[k] + eps[k] * eps[k]; }
            _pool[sym] = (v, idx.ToArray(), eps.ToArray(), prefix, prefSq);
        }
    }

    /// <summary>Residuals available at t (those with s ≤ t).</summary>
    public int PoolSize(string symbol, int t)
    {
        var idx = _pool[symbol].ValidIdx;
        int k = Array.BinarySearch(idx, t);
        return k >= 0 ? k + 1 : ~k;
    }

    /// <summary>
    /// Fraction of FHS paths on which <paramref name="firesOnStep"/> fires at least once within the
    /// horizon. NaN when the name has no 𝓕_t variance forecast or fewer than 20 residuals by t
    /// (the caller falls back to the GBM label).
    /// </summary>
    public float FractionFiring(string symbol, int t, float startPrice, Func<float, int, bool> firesOnStep, Random rng)
    {
        var (var1, _, eps, prefix, prefSq) = _pool[symbol];
        int n = PoolSize(symbol, t);
        double s2 = var1[t];
        if (n < 20 || double.IsNaN(s2) || s2 <= 0 || startPrice <= 0) return float.NaN;
        double mean = prefix[n] / n;
        double sd   = Math.Sqrt(Math.Max(prefSq[n] / n - mean * mean, 1e-12));

        int fired = 0;
        for (int p = 0; p < Paths; p++)
        {
            double logP = Math.Log(startPrice), h = s2;
            for (int s = 1; s <= Horizon; s++)
            {
                double e = (eps[rng.Next(n)] - mean) / sd;
                double sig = Math.Sqrt(h), rStar = sig * e;
                logP += rStar - 0.5 * h;
                h = EwmaVol.Lambda * h + (1 - EwmaVol.Lambda) * rStar * rStar;
                if (firesOnStep((float)Math.Exp(logP), s)) { fired++; break; }
            }
        }
        return (float)fired / Paths;
    }
}

namespace DirectIndexing.Core.Simulation;

/// <summary>
/// Computes annualised tracking error σ_TE = √(δw⊤ Σ δw × 252) via the full
/// quadratic form, where:
///   Σ   = N×N daily return covariance matrix, pre-computed once from the full
///          price history at construction time (pairwise available-case, Bessel correction).
///   δw  = active weight deviation vector:
///            δw_i = 1/n_open − 1/N   (lot i is open)
///            δw_i =         − 1/N   (lot i is not open / in wash-sale window)
///
/// Both portfolio and benchmark use equal weights, consistent with the simulation's
/// equal-dollar lot initialisation.
///
/// v0.1 used std(r_port − r_bench, window=30) × √252 (rolling scalar approach).
/// v0.2 (this class) uses the quadratic form for forward-looking estimates that expose
/// cross-stock correlations — the natural extension point for future RMT-based
/// Marchenko-Pastur eigenvalue cleaning.
///
/// Computational cost:
///   Construction : O(N² × T) ≈ 127M ops — once at load time
///   Per-day call : O(N²)     ≈ 253K ops — v = Σδw matrix-vector product
/// </summary>
public sealed class TrackingErrorProxy
{
    private readonly float[,]               _cov;      // [N, N] daily return covariance
    private readonly List<string>           _symbols;  // sorted — defines row/col order
    private readonly Dictionary<string,int> _symIdx;
    private readonly int                    _N;

    public TrackingErrorProxy(PriceLoader prices)
    {
        _symbols = prices.Symbols.OrderBy(s => s).ToList();
        _N       = _symbols.Count;
        _symIdx  = _symbols.Select((s, i) => (s, i)).ToDictionary(x => x.s, x => x.i);
        _cov     = ComputeCovariance(prices, _symbols, _N);
        Console.WriteLine($"[TrackingErrorProxy] Σ computed: {_N}×{_N}.");
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>
    /// Returns annualised tracking error for the current open-lot universe.
    /// Call once per trading day before the lot loop.
    /// </summary>
    public float Update(IEnumerable<string> openSymbols)
    {
        // Build open set (restrict to symbols known in our covariance universe)
        var openSet = new HashSet<string>();
        foreach (var s in openSymbols)
            if (_symIdx.ContainsKey(s)) openSet.Add(s);

        int nOpen = openSet.Count;
        if (nOpen == 0) return 0f;

        float wPort  = 1f / nOpen;
        float wBench = 1f / _N;

        // Pre-compute δw vector (N HashSet lookups instead of N² inside the loop)
        var dw = new float[_N];
        for (int i = 0; i < _N; i++)
            dw[i] = openSet.Contains(_symbols[i]) ? wPort - wBench : -wBench;

        // variance = δw⊤ Σ δw  (combined matrix-vector multiply + dot product)
        double variance = 0;
        for (int i = 0; i < _N; i++)
        {
            double vi = 0;
            for (int j = 0; j < _N; j++)
                vi += _cov[i, j] * dw[j];
            variance += dw[i] * vi;
        }

        return MathF.Sqrt(MathF.Max((float)variance, 0f) * 252f);
    }

    // ── Shared covariance estimator ──────────────────────────────────────────

    /// <summary>
    /// Pairwise available-case sample covariance with Bessel's correction.
    ///
    /// Σ_ij is estimated from all days t where both r_t^(i) and r_t^(j) are non-NaN.
    /// The resulting matrix is symmetric; diagonal entries are per-stock daily variances.
    ///
    /// Exposed as <c>internal static</c> so <see cref="MonteCarloEngine"/> can reuse
    /// it for its calibrated constructor without duplicating the logic.
    /// </summary>
    internal static float[,] ComputeCovariance(PriceLoader prices, List<string> symbols, int N)
    {
        var retArrays = symbols.Select(s => prices.GetReturnArray(s)).ToArray();
        int T         = retArrays[0].Length;

        // Step 1: per-symbol means (ignoring NaN)
        var means  = new double[N];
        var counts = new int[N];
        for (int i = 0; i < N; i++)
        {
            for (int t = 0; t < T; t++)
            {
                float r = retArrays[i][t];
                if (!float.IsNaN(r)) { means[i] += r; counts[i]++; }
            }
            if (counts[i] > 0) means[i] /= counts[i];
        }

        // Step 2: pairwise sample covariance (upper triangle → fill both halves)
        var cov = new float[N, N];
        for (int i = 0; i < N; i++)
        {
            for (int j = i; j < N; j++)
            {
                double sumCov = 0;
                int    cnt    = 0;
                for (int t = 0; t < T; t++)
                {
                    float ri = retArrays[i][t];
                    float rj = retArrays[j][t];
                    if (!float.IsNaN(ri) && !float.IsNaN(rj))
                    {
                        sumCov += (ri - means[i]) * (rj - means[j]);
                        cnt++;
                    }
                }
                float c = cnt > 1 ? (float)(sumCov / (cnt - 1)) : 0f;
                cov[i, j] = c;
                cov[j, i] = c;
            }
        }
        return cov;
    }
}
